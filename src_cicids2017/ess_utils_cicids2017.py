'''
    ess_utils_cicids2017.py

    CIC-IDS2017 equivalent of src/ess_utils.py.

    WHY THIS FILE EXISTS SEPARATELY, RATHER THAN JUST IMPORTING
    ess_utils.py DIRECTLY EVERYWHERE
    -------------------------------------------------------------
    NUM_CLASSES, ALL_CLASS_LABELS, the label map, and XGB_PARAMS'
    num_class are all hardcoded to the DDoS2019 8-class schema in the
    real ess_utils.py. Importing train_xgb / evaluate_model /
    majority_class_f1 / random_baseline_f1 unchanged would silently
    train and score every model as if it had 8 possible classes, when
    CIC-IDS2017 actually has 15 (confirmed by
    check_cicids2017_structure.py). This file mirrors those same
    functions, same logic, with the correct CIC-IDS2017 constants.

    WHAT IS IMPORTED DIRECTLY FROM THE REAL ess_utils.py, UNCHANGED
    -------------------------------------------------------------
    compute_ess, compute_euclidean, compute_psi, compute_shap_rankings.
    These operate only on ranking vectors, score vectors and SHAP
    arrays, nothing in them depends on how many classes exist or what
    they're called. Importing them directly, not copying them, is what
    guarantees both experiments are computing these four metrics in
    the literal same way, not just a similar way.
'''

import os
import sys
import warnings
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import UndefinedMetricWarning

# This warning fires whenever a batch's test split has zero examples
# of some class (routine here, since several deployment batches only
# contain 2-3 of the 15 total classes), and is already handled
# correctly by the try/except in evaluate_model() below, which falls
# back to NaN when AUC genuinely can't be computed. Silencing it here,
# once, at the source, keeps every script that imports from this file
# readable instead of burying the handful of lines that actually
# matter under dozens of repeated, already-understood warnings.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Make src/ importable so the four dataset-agnostic functions can be
# pulled from the real ess_utils.py directly, rather than copied.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SRC_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "src"))
if MAIN_SRC_DIR not in sys.path:
    sys.path.append(MAIN_SRC_DIR)

from ess_utils import compute_ess, compute_euclidean, compute_psi, compute_shap_rankings  # noqa: F401


# ============================================================
# CIC-IDS2017-SPECIFIC CONSTANTS
# 15 classes confirmed from check_cicids2017_structure.py's real output
# ============================================================

NUM_CLASSES = 15
ALL_CLASS_LABELS = list(range(NUM_CLASSES))

# Same convention as the original pipeline's label_map_multi:
# BENIGN = 0, everything else in alphabetical order.
LABEL_MAP_MULTI = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "Infiltration": 9,
    "PortScan": 10,
    "SSH-Patator": 11,
    "Web Attack - Brute Force": 12,
    "Web Attack - Sql Injection": 13,
    "Web Attack - XSS": 14,
}

# Classes too rare for reliable per-class analysis (real row counts
# from the raw files: Heartbleed=11, Web Attack Sql Injection=21,
# Infiltration=36). Still included in the overall multiclass model and
# weighted F1, which handles rare classes fine in aggregate, but
# excluded from any per-class breakdown, same honesty standard as the
# LDAP/UDPLag exceptions already acknowledged in Chapter 5.
RARE_CLASSES = ["Heartbleed", "Web Attack - Sql Injection", "Infiltration"]


def map_label_to_int(raw_label):
    '''
    Maps a raw Label string to its integer class code. Matches the
    three "Web Attack" classes by distinctive substring rather than
    requiring an exact string match, since these three specific labels
    contain an en dash that can decode slightly differently depending
    on how a file is read. Reading with encoding="cp1252" (already
    done in 01_load_and_batch.py) fixes this in the normal case; this
    substring match is a safety net so a leftover decoding quirk can't
    silently produce an unmapped label that pandas would turn into NaN
    and quietly drop.

    Raises ValueError on anything unrecognised, deliberately, rather
    than silently mapping an unexpected label to NaN. A loud failure
    here is far easier to catch than a class that quietly vanishes
    partway through the pipeline.
    '''
    label = str(raw_label).strip()

    if label in LABEL_MAP_MULTI:
        return LABEL_MAP_MULTI[label]

    if "Web Attack" in label:
        if "Brute Force" in label:
            return LABEL_MAP_MULTI["Web Attack - Brute Force"]
        if "Sql Injection" in label or "SQL Injection" in label:
            return LABEL_MAP_MULTI["Web Attack - Sql Injection"]
        if "XSS" in label:
            return LABEL_MAP_MULTI["Web Attack - XSS"]

    raise ValueError(
        f"Unrecognised label: {raw_label!r}. This does not match any entry "
        f"in LABEL_MAP_MULTI and does not contain a recognised 'Web Attack' "
        f"substring. Check for a new or unexpected class name before "
        f"proceeding, do not silently skip it."
    )


# Batch 1 = baseline (Monday + Tuesday combined).
# Batches 2-7 = the six deployment batches agreed on:
# Wednesday, Thursday-WebAttacks, Thursday-Infiltration,
# Friday-Bot, Friday-PortScan, Friday-DDoS.
N_BATCHES = 7
DEPLOYMENT_BATCHES = list(range(2, N_BATCHES + 1))

XGB_PARAMS = {
    "objective"    : "multi:softprob",
    "num_class"    : NUM_CLASSES,
    "max_depth"    : 6,
    "eta"          : 0.1,       # matches ess_utils.py exactly, deliberately not retuned
    "eval_metric"  : "mlogloss",
    "verbosity"    : 0,
}
N_ESTIMATORS = 100
DEFAULT_SEED = 42


def train_xgb(X, y, num_boost_round=N_ESTIMATORS, seed=DEFAULT_SEED):
    '''CIC-IDS2017 equivalent of ess_utils.train_xgb. Identical logic;
    only difference is NUM_CLASSES=15 baked into XGB_PARAMS above
    instead of 8.'''
    params = dict(XGB_PARAMS)
    params["seed"] = seed
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def stratified_or_plain_split(X, y, test_size=0.2, random_state=DEFAULT_SEED):
    '''Identical logic to ess_utils.stratified_or_plain_split. This one
    has no dataset-specific constants in it at all, so it would be
    equally safe to import directly -- it's kept here instead so this
    file mirrors the real one's structure exactly and can be read
    side by side with it.'''
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )


def majority_class_f1(y_train, y_test):
    '''Trivial baseline 1, CIC-IDS2017 version: predict the single most
    frequent class from y_train for every sample in y_test.'''
    majority_class = y_train.mode().iloc[0]
    y_pred = np.full(len(y_test), majority_class)
    return round(float(f1_score(
        y_test, y_pred, average="weighted",
        labels=ALL_CLASS_LABELS, zero_division=0
    )), 4)


def random_baseline_f1(y_test, seed=DEFAULT_SEED):
    '''Trivial baseline 2, CIC-IDS2017 version: predict a uniformly
    random class from the full 15-class label space.'''
    rng = np.random.RandomState(seed)
    y_pred = rng.randint(0, NUM_CLASSES, size=len(y_test))
    return round(float(f1_score(
        y_test, y_pred, average="weighted",
        labels=ALL_CLASS_LABELS, zero_division=0
    )), 4)


def predict_proba(booster, X):
    '''Returns an (n_samples, NUM_CLASSES) probability matrix, always
    with all 15 columns present regardless of what the model was
    trained on.'''
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)


def predict_labels(booster, X):
    proba = predict_proba(booster, X)
    return np.argmax(proba, axis=1)


def evaluate_model(booster, X_test, y_test):
    '''Evaluates a booster on held-out data, CIC-IDS2017 version. Both
    F1 and AUC-ROC are computed with labels pinned to the full 0-14
    global label set, so a batch that only contains a few classes still
    scores correctly instead of crashing or silently misaligning
    columns.'''
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