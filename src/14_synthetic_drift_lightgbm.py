'''
    14_synthetic_drift_lightgbm.py

    Purpose: Repeat the controlled synthetic drift experiment with
    LightGBM instead of XGBoost, to check whether the behaviour of ESS,
    F1 and PSI under directed drift still holds when the tree-learning
    algorithm changes.

    What stays IDENTICAL to 12_synthetic_drift_injection.py:
    - the same seven attack classes, raw November files and 41 selected features
    - the same sampling (30,000 rows, seed 42), cleaning and 70/30 split
    - the same rule for choosing features to perturb (top 5 continuous
      features by baseline SHAP rank, at least 20 distinct values)
    - the same directed drift: attack features moved alpha of the way
      from the attack mean toward the benign mean, alpha 0.0 to 1.0
    - the same ESS, Euclidean and PSI functions from ess_utils.py
    - the model is never retrained across drift steps

    What CHANGES:
    - the classifier is LightGBM (lgb.train, binary objective)
    - SHAP output is normalised to one 2D (rows, features) array for the
      attack class before ranking, because different SHAP versions
      return binary LightGBM output in different shapes

    Model settings: matched to the XGBoost settings on the main capacity
    controls (100 boosting rounds, learning rate 0.1, max depth 6, same
    seed). num_leaves is 63 so the largest possible tree is close to a
    depth-6 XGBoost tree (up to 64 leaves). Two leaf-safety settings are
    also set to XGBoost's default values: min_sum_hessian_in_leaf = 1
    (XGBoost min_child_weight = 1) and lambda_l2 = 1 (XGBoost
    reg_lambda = 1). Without these, LightGBM's much weaker defaults let
    leaves form from almost no information, which produced extreme raw
    scores (in the hundreds of thousands) for classes with very few
    benign rows. Everything else is left at LightGBM defaults. Nothing
    is tuned, so the comparison does not favour either model.

    Output: outputs/results/synthetic_drift_injection_lightgbm.csv, with
    the same columns as the XGBoost output plus a "model" column.
'''

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ess_utils import compute_shap_rankings, compute_ess, compute_euclidean, compute_psi

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION (identical to the XGBoost version except LGB_PARAMS)
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "outputs/results"
RAW_DIR = "data/raw/03-11"
OUTPUT_FILE = f"{RESULTS_DIR}/synthetic_drift_injection_lightgbm.csv"
SAMPLE_SIZE = 30_000
SEED = 42

ATTACK_CLASSES = ["LDAP", "MSSQL", "NetBIOS", "Portmap", "Syn", "UDP", "UDPLag"]

PERTURB_TOP_N = 5
MIN_UNIQUE_VALUES = 20
DRIFT_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# LightGBM settings matched to the XGBoost ones where an equivalent exists
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,     # same as XGBoost eta
    "max_depth": 6,           # same as XGBoost max_depth
    "num_leaves": 63,         # close to the 64-leaf maximum of a depth-6 tree
    "min_sum_hessian_in_leaf": 1.0,  # same as XGBoost default min_child_weight = 1
    "lambda_l2": 1.0,                # same as XGBoost default reg_lambda = 1
    "seed": SEED,
    "deterministic": True,    # makes repeated runs give the same model
    "verbosity": -1,
}
NUM_BOOST_ROUND = 100         # same as XGBoost num_boost_round

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    feature_names = json.load(f)


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING (unchanged from the XGBoost version)
# ──────────────────────────────────────────────────────────────────────────

def load_and_clean(class_name, sample_size, seed=SEED):
    '''
    Loads one raw CIC-DDoS2019 CSV in chunks, labels rows as 0 = BENIGN
    and 1 = attack, samples rows, keeps the 41 selected features and
    fills missing or infinite values with the column median.

    Memory saving: only the 41 selected features and the label column
    are read from the file (usecols). The other columns, including
    large text columns such as Flow ID, IP addresses and Timestamp,
    are never loaded. This does NOT change which rows are sampled,
    because df.sample() picks row positions using only the number of
    rows and the seed, and the number of rows is unchanged.
    '''
    path = f"{RAW_DIR}/{class_name}.csv"
    wanted = set(feature_names)

    def keep_column(col):
        # raw column names can have leading spaces, e.g. " Label"
        name = col.strip()
        return name in wanted or "label" in name.lower()

    chunks = []
    for encoding in ["utf-8", "latin-1"]:
        try:
            reader = pd.read_csv(
                path, encoding=encoding, low_memory=False,
                chunksize=100_000, usecols=keep_column,
            )
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)
            break
        except UnicodeDecodeError:
            chunks = []
            continue
    df = pd.concat(chunks, ignore_index=True)

    label_col = next(c for c in df.columns if "label" in c.lower())
    df["label_binary"] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    X = df_sample[feature_names].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    y = df_sample["label_binary"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


# ──────────────────────────────────────────────────────────────────────────
# MODEL-SPECIFIC HELPERS (the only new logic in this file)
# ──────────────────────────────────────────────────────────────────────────

def train_lgb(X_train, y_train):
    '''
    Trains the binary LightGBM model. Uses plain numpy arrays so that
    LightGBM does not rename feature columns that contain spaces.
    '''
    dtrain = lgb.Dataset(X_train.values, label=y_train.values)
    return lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)


def predict_labels(model, X):
    '''Returns 0/1 predictions using a 0.5 threshold on the attack probability.'''
    return (model.predict(X.values) > 0.5).astype(int)


def normalise_binary_shap(shap_values):
    '''
    Returns SHAP values as ONE 2D (rows, features) array for the attack class.

    Depending on the SHAP version, a binary LightGBM model returns either:
      - a 2D array (rows, features): used as it is
      - a list of two 2D arrays [benign, attack]: the attack array is kept
      - a 3D array (rows, features, 2): the attack slice is kept

    Without this step, the list case would become a 3D array with the
    class axis FIRST, and compute_shap_rankings would average over the
    wrong axis without any error.
    '''
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 2:
        return arr[:, :, 1]
    raise ValueError(f"Unexpected SHAP output shape {arr.shape} for a binary model.")


class BinaryShapExplainer:
    '''
    Small wrapper around shap.TreeExplainer so that compute_shap_rankings
    (from ess_utils.py) always receives a 2D (rows, features) array.
    Every call also checks that the SHAP shape matches the data shape,
    so a shape problem stops the script instead of producing wrong
    rankings silently.
    '''

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X):
        values = normalise_binary_shap(self.explainer.shap_values(np.asarray(X)))
        expected = (X.shape[0], X.shape[1])
        if values.shape != expected:
            raise ValueError(f"SHAP shape {values.shape} does not match data shape {expected}.")
        return values


# ──────────────────────────────────────────────────────────────────────────
# EXPERIMENT FOR ONE CLASS (same steps as the XGBoost version)
# ──────────────────────────────────────────────────────────────────────────

def run_for_class(class_name):
    '''
    Trains one fixed LightGBM model for this class, picks the features to
    perturb from its baseline SHAP ranking, then applies each drift step
    and records F1, ESS, Euclidean distance and PSI.
    '''
    print("\n" + "#" * 60)
    print(f"# CLASS: {class_name} (LightGBM)")
    print("#" * 60)

    X_full, y_full = load_and_clean(class_name, SAMPLE_SIZE)
    print(f"Sample shape: {X_full.shape} | Labels: {y_full.value_counts().to_dict()}")

    if y_full.nunique() < 2:
        print(f"SKIPPING {class_name}: only one class in the sample.")
        return []

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=SEED, stratify=y_full
    )

    # step 1: train the fixed baseline model
    model = train_lgb(X_train, y_train)
    baseline_f1 = f1_score(y_test, predict_labels(model, X_test), average="weighted")
    print(f"Held-out F1 on unperturbed data (sanity check): {baseline_f1:.4f}")

    # step 2: baseline SHAP ranking and features to perturb
    explainer = BinaryShapExplainer(model)
    baseline_ranking_df, baseline_shap_array, baseline_scores = compute_shap_rankings(
        explainer, X_train, feature_names
    )
    print(f"Baseline SHAP array shape (rows, features): {baseline_shap_array.shape}")

    nunique = X_train.nunique()
    continuous_candidates = [
        f for f in baseline_ranking_df["feature"] if nunique.get(f, 0) >= MIN_UNIQUE_VALUES
    ]
    perturb_features = continuous_candidates[:PERTURB_TOP_N]
    print(f"Perturbing: {perturb_features}")

    attack_means = {f: X_train.loc[y_train == 1, f].mean() for f in perturb_features}
    benign_means = {f: X_train.loc[y_train == 0, f].mean() for f in perturb_features}

    baseline_rank_vector = baseline_ranking_df.set_index("feature").reindex(feature_names)["rank"].values

    # step 3: directed drift, attack traffic shifting toward benign
    class_results = []
    attack_pool = X_test.loc[y_test == 1]

    for i, alpha in enumerate(DRIFT_STEPS):
        X_step = attack_pool.sample(
            n=min(SAMPLE_SIZE, len(attack_pool)), random_state=SEED + i, replace=True
        ).copy().reset_index(drop=True)
        y_step = pd.Series(np.ones(len(X_step), dtype=int))

        for f in perturb_features:
            X_step[f] = X_step[f] + alpha * (benign_means[f] - attack_means[f])

        step_f1 = f1_score(y_step, predict_labels(model, X_step), average="weighted", zero_division=0)

        step_ranking_df, step_shap_array, step_scores = compute_shap_rankings(
            explainer, X_step, feature_names
        )
        step_rank_vector = step_ranking_df.set_index("feature").reindex(feature_names)["rank"].values

        ess, pvalue = compute_ess(baseline_rank_vector, step_rank_vector)
        euclidean = compute_euclidean(baseline_scores, step_scores)
        psi = compute_psi(baseline_shap_array, step_shap_array)

        print(f"  Step {i+1} (alpha={alpha:.1f}): F1={step_f1:.4f} | ESS={ess:.4f} | "
              f"Euclidean={euclidean:.4f} | PSI={psi:.4f}")

        class_results.append({
            "model": "LightGBM", "class": class_name, "step": i + 1, "alpha": alpha,
            "f1": round(step_f1, 4), "ess": round(ess, 4), "ess_pvalue": round(pvalue, 4),
            "euclidean": round(euclidean, 4), "psi": round(psi, 4),
            "baseline_holdout_f1": round(baseline_f1, 4),
            "perturbed_features": "; ".join(perturb_features),
        })

    return class_results


# ──────────────────────────────────────────────────────────────────────────
# RUN ALL SEVEN CLASSES, SAVING AFTER EACH ONE
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    for class_name in ATTACK_CLASSES:
        try:
            class_results = run_for_class(class_name)
        except Exception as e:
            print(f"\n!!! {class_name} FAILED with: {e}")
            print("!!! Skipping this class and continuing.")
            class_results = []

        all_results.extend(class_results)
        pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)
        print(f"(Progress saved: {len(all_results)} rows so far)")

    print("\n" + "=" * 60)
    print("Full LightGBM trajectory, all classes")
    print("=" * 60)
    print(pd.DataFrame(all_results).to_string(index=False))
    print(f"\nSaved: {OUTPUT_FILE}")

    print("\nReminders:")
    print("- Do not change alpha, PERTURB_TOP_N, the classes or LGB_PARAMS after seeing results.")
    print("- LightGBM may pick different features to perturb than XGBoost. That is expected,")
    print("  because each model has its own SHAP ranking. Report it, do not force them to match.")
    print("- Report each class honestly, including where LightGBM disagrees with XGBoost.")