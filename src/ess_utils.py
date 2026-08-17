'''
    ess_utils.py

    Shared, single-source-of-truth module for computing SHAP-based
    feature importance rankings, the Explanation Stability Score (ESS),
    Euclidean distance and PSI comparison metrics, and evaluation metrics.

    03_train_baseline.py, 04_ess_engine.py and 05_experiment_main.py all
    import from this module instead of each carrying their own copy of
    this logic. This was refactored from three near-duplicate copies of
    the same functions (one of which had a shape bug the others did not)
    into one tested implementation.

    GLOBAL LABEL SPACE
    -------------------
    All 8 classes are fixed in advance from the known CIC-DDoS2019 schema
    (BENIGN=0, LDAP=1, MSSQL=2, NetBIOS=3, Portmap=4, Syn=5, UDP=6,
    UDPLag=7), exactly as encoded in 02_preprocessing.py. No script in
    this pipeline should ever build a fresh per-batch remap of labels.
    A model is trained via the native xgboost.train() API with
    num_class fixed at NUM_CLASSES=8 regardless of how many classes are
    actually present in a given training batch. This means a batch with
    only 2 of the 8 classes present trains and evaluates in exactly the
    same label space as a batch with all 8 present, so predictions and
    ground truth are always comparable across batches and across models,
    which was the source of the label-mismatch bug in the earlier version
    of this pipeline.
'''

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

NUM_CLASSES = 8
ALL_CLASS_LABELS = list(range(NUM_CLASSES))

# Total number of temporal batches (Batch 1 = baseline, rest = deployment).
# Raised from 6 to 10 so the independence correlations in
# 06_independence_experiment.py have more than n=5 points to work with.
# Change this in ONE place and every script that imports DEPLOYMENT_BATCHES
# or N_BATCHES from here picks it up automatically.
N_BATCHES = 10
DEPLOYMENT_BATCHES = list(range(2, N_BATCHES + 1))

XGB_PARAMS = {
    "objective"    : "multi:softprob",
    "num_class"    : NUM_CLASSES,
    "max_depth"    : 6,
    "eta"          : 0.1,      # equivalent to sklearn's learning_rate
    "eval_metric"  : "mlogloss",
    "verbosity"    : 0,
}
N_ESTIMATORS = 100  # equivalent to sklearn's n_estimators -> num_boost_round
DEFAULT_SEED = 42


def train_xgb(X, y, num_boost_round=N_ESTIMATORS, seed=DEFAULT_SEED):
    '''
    Trains an XGBoost model using the native Booster API with a fixed
    global num_class, regardless of how many distinct classes are
    actually present in y. This is the fix for the "contiguous 0..k-1
    class" restriction in sklearn's XGBClassifier.fit(), and it means no
    per-batch label remapping is ever needed anywhere in this pipeline.

    seed is exposed as a parameter (rather than hardcoded) so the
    multi-seed robustness experiment can retrain with a different seed
    each run without touching this function.
    '''
    params = dict(XGB_PARAMS)
    params["seed"] = seed
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def stratified_or_plain_split(X, y, test_size=0.2, random_state=DEFAULT_SEED):
    '''
    Attempts a stratified split; falls back to a plain random split if
    any class has too few members to stratify (this can genuinely
    happen with rare classes like UDPLag in some batches). Shared here
    so 05_experiment_main.py and the new sweep/multi-seed scripts don't
    each carry their own copy.
    '''
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )


def majority_class_f1(y_train, y_test):
    '''
    Trivial baseline 1: predict the single most frequent class from
    y_train for every sample in y_test. Gives a floor for F1 -- any
    real model should comfortably beat this, and reporting it alongside
    F1=0.99 headline numbers shows they mean something.
    '''
    majority_class = y_train.mode().iloc[0]
    y_pred = np.full(len(y_test), majority_class)
    return round(float(f1_score(
        y_test, y_pred, average="weighted",
        labels=ALL_CLASS_LABELS, zero_division=0
    )), 4)


def random_baseline_f1(y_test, seed=DEFAULT_SEED):
    '''
    Trivial baseline 2: predict a uniformly random class (from the full
    8-class label space) for every sample in y_test. A second, weaker
    floor than the majority-class baseline.
    '''
    rng = np.random.RandomState(seed)
    y_pred = rng.randint(0, NUM_CLASSES, size=len(y_test))
    return round(float(f1_score(
        y_test, y_pred, average="weighted",
        labels=ALL_CLASS_LABELS, zero_division=0
    )), 4)


def predict_proba(booster, X):
    '''
    Returns an (n_samples, NUM_CLASSES) probability matrix, always with
    all 8 columns present regardless of what the model was trained on.
    '''
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)  # shape (n_samples, num_class) for multi:softprob


def predict_labels(booster, X):
    proba = predict_proba(booster, X)
    return np.argmax(proba, axis=1)


def evaluate_model(booster, X_test, y_test):
    '''
    Evaluates a booster on held-out data. Both F1 and AUC-ROC are
    computed with labels explicitly pinned to the full 0-7 global
    label set, so a batch that only contains 2 or 3 classes still
    scores correctly rather than crashing or silently realigning
    columns to the wrong classes.
    '''
    y_pred = predict_labels(booster, X_test)
    y_proba = predict_proba(booster, X_test)

    f1 = f1_score(
        y_test, y_pred, average="weighted",
        labels=ALL_CLASS_LABELS, zero_division=0
    )

    try:
        auc = roc_auc_score(
            y_test, y_proba,
            multi_class="ovr", average="weighted",
            labels=ALL_CLASS_LABELS
        )
    except ValueError:
        # can happen if y_test has fewer than 2 distinct classes present
        auc = float("nan")

    return round(float(f1), 4), round(float(auc), 4)


def compute_shap_rankings(explainer, X_batch, feature_names):
    '''
    Computes SHAP values for a batch and returns:
      - ranking_df: feature name, mean |SHAP| score, rank (1 = most important)
      - shap_array: the raw per-sample SHAP array, kept 2D (rows, features)
                    after averaging over any class dimension, for PSI/ED use
      - mean_abs_shap: 1D array of length n_features

    Handles both cases returned by shap.TreeExplainer.shap_values():
      - 3D array (rows, features, classes) when the model outputs
        probabilities for more than 2 classes actually present
      - 2D array (rows, features) when only 2 classes are effectively
        being distinguished (binary-shaped SHAP output)
    This dynamic handling is the fix for the crash that occurred when a
    batch contains very few classes (e.g. a 2-class batch).
    '''
    shap_values = explainer.shap_values(X_batch)
    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        # (rows, features, classes) -> mean over rows, then over classes
        per_row_abs = np.mean(np.abs(shap_array), axis=0)   # (features, classes)
        mean_abs_shap = np.mean(per_row_abs, axis=1)          # (features,)
        # keep a 2D (rows, features) version for PSI/Euclidean by
        # averaging the class dimension per row first
        shap_array_2d = np.mean(np.abs(shap_array), axis=2)   # (rows, features)
    elif shap_array.ndim == 2:
        # (rows, features) already
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)   # (features,)
        shap_array_2d = np.abs(shap_array)                     # (rows, features)
    else:
        raise ValueError(
            f"Unexpected SHAP array shape {shap_array.shape}; "
            f"expected 2D or 3D output."
        )

    ranking_df = pd.DataFrame({
        "feature": feature_names,
        "shap_score": mean_abs_shap,
    }).sort_values("shap_score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    return ranking_df, shap_array_2d, mean_abs_shap


def compute_ess(baseline_rank_vector, batch_rank_vector):
    '''ESS = Spearman rank correlation between two feature ranking vectors.'''
    correlation, pvalue = spearmanr(baseline_rank_vector, batch_rank_vector)
    return float(correlation), float(pvalue)


def compute_euclidean(baseline_scores, batch_scores):
    '''Euclidean distance between two mean |SHAP| score vectors.'''
    return float(np.sqrt(np.sum((np.asarray(baseline_scores) - np.asarray(batch_scores)) ** 2)))


def compute_psi(baseline_array, batch_array, n_bins=10, epsilon=1e-10):
    '''
    PSI between the baseline SHAP value distribution and a deployment
    batch's SHAP value distribution (per Chapter 3, Section 3.6.2 --
    this is computed on SHAP values, not raw input features, by design).
    '''
    baseline_flat = np.clip(np.asarray(baseline_array).flatten(), epsilon, None)
    batch_flat = np.clip(np.asarray(batch_array).flatten(), epsilon, None)

    min_val = min(baseline_flat.min(), batch_flat.min())
    max_val = max(baseline_flat.max(), batch_flat.max())
    if min_val == max_val:
        return 0.0
    bins = np.linspace(min_val, max_val, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline_flat, bins=bins)
    batch_counts, _ = np.histogram(batch_flat, bins=bins)

    baseline_pct = np.clip(baseline_counts / (baseline_counts.sum() + epsilon), epsilon, None)
    batch_pct = np.clip(batch_counts / (batch_counts.sum() + epsilon), epsilon, None)

    psi = np.sum((batch_pct - baseline_pct) * np.log(batch_pct / baseline_pct))
    return float(psi)
