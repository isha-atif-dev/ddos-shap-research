'''
    11_ldap_drift_check.py

    Purpose: A genuine "same class, two real points in time" drift
    check, separate from the main experiment, addressing the
    novel-class-vs-drift concern raised in supervisor feedback on
    Chapter 4.

    The main experiment (scripts 01-09) tests what happens when the
    model meets attack classes it has NEVER seen before. This script
    instead trains on LDAP traffic from one date (3 November 2018,
    already in the main pipeline) and checks the SAME, unretrained
    model against LDAP traffic from a different date (1 December 2018,
    the newly downloaded file), a class the model already knows in
    both cases. This isolates whether explanation drift can be seen on
    a class the model was trained on, separate from the novel-class
    effect studied elsewhere in this dissertation.

    What this file does:
    - Loads LDAP.csv from 3 November 2018 (data/raw/03-11/), keeping
      only the 41 features already selected by the main pipeline
    - Trains a small binary XGBoost classifier (LDAP vs BENIGN) on an
      80% split of this November data, and computes its baseline SHAP
      ranking with TreeExplainer, exactly as 03_train_baseline.py does
      for the main experiment
    - Loads DrDoS_LDAP.csv from 1 December 2018 (data/raw/01-12/),
      keeping the same 41 features, and evaluates the SAME November
      model against it WITHOUT any retraining
    - Reports: the November model's F1 score on December data (does
      accuracy hold up on a class the model already knows, unlike the
      novel-class scenario where it collapsed), the ESS between the
      November baseline ranking and the December ranking, and the
      Euclidean distance and PSI between the two SHAP outputs for
      comparison
    - Saves results to outputs/results/ldap_drift_check.csv
'''

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from ess_utils import compute_shap_rankings, compute_ess, compute_euclidean, compute_psi

RESULTS_DIR = "outputs/results"
NOV_FILE = "data/raw/03-11/LDAP.csv"
DEC_FILE = "data/raw/01-12/DrDoS_LDAP.csv"
SAMPLE_SIZE = 50_000
SEED = 42

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load the selected feature list from the main pipeline
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading selected feature list from the main pipeline ...")
print("=" * 60)

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    feature_names = json.load(f)
print(f"Using the same {len(feature_names)} selected features as the main pipeline.")


def load_and_clean(path, sample_size, seed=SEED):
    '''
    Loads one raw CIC-DDoS2019 CSV, strips column whitespace, samples
    rows, keeps only the selected features plus label, imputes missing
    or infinite values with the column median, and maps the label
    column to a binary 0/1 scheme (0 = BENIGN, 1 = anything else),
    matching the label_binary convention already used in
    02_preprocessing.py. This is deliberately not an exact-match
    dictionary, since raw CIC-DDoS2019 files can contain a small
    number of stray rows labelled with a neighbouring attack's name
    (e.g. a few NetBIOS rows inside LDAP.csv) due to overlapping
    capture windows on the collection day.
    '''
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    label_col = None
    for col in df.columns:
        if "label" in col.lower():
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found in {path}")

    df["label_binary"] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=seed)

    X = df_sample[feature_names].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    y = df_sample["label_binary"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    return X, y


# ──────────────────────────────────────────────────────────────────────────
# STEP 2: load and clean the November (baseline) LDAP data
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 2: Loading November baseline data: {NOV_FILE} ...")
print("=" * 60)

X_nov, y_nov = load_and_clean(NOV_FILE, SAMPLE_SIZE)
print(f"November sample shape: {X_nov.shape}")
print(f"Label distribution:\n{y_nov.value_counts()}")

X_nov_train, X_nov_test, y_nov_train, y_nov_test = train_test_split(
    X_nov, y_nov, test_size=0.2, random_state=SEED, stratify=y_nov
)

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: train the November baseline model
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training the November baseline model (binary: LDAP vs BENIGN) ...")
print("=" * 60)

params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "logloss",
    "seed": SEED,
    "verbosity": 0,
}
dtrain = xgb.DMatrix(X_nov_train, label=y_nov_train)
model = xgb.train(params, dtrain, num_boost_round=100)
print("Model trained.")

dtest = xgb.DMatrix(X_nov_test)
nov_pred = (model.predict(dtest) > 0.5).astype(int)
nov_f1 = f1_score(y_nov_test, nov_pred, average="weighted")
print(f"November held-out F1 (sanity check): {nov_f1:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: compute the November baseline SHAP ranking
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Computing November baseline SHAP ranking with TreeExplainer ...")
print("=" * 60)

explainer = shap.TreeExplainer(model)
nov_ranking_df, nov_shap_array, nov_scores = compute_shap_rankings(
    explainer, X_nov_train, feature_names
)
print("Top 5 features (November baseline):")
print(nov_ranking_df.head(5).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: load and clean the December comparison data
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 5: Loading December comparison data: {DEC_FILE} ...")
print("=" * 60)

X_dec, y_dec = load_and_clean(DEC_FILE, SAMPLE_SIZE)
print(f"December sample shape: {X_dec.shape}")
print(f"Label distribution:\n{y_dec.value_counts()}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: evaluate the UNRETRAINED November model on December data
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Evaluating the November model on December data (no retraining) ...")
print("=" * 60)

ddec = xgb.DMatrix(X_dec)
dec_proba = model.predict(ddec)
dec_pred = (dec_proba > 0.5).astype(int)

dec_f1 = f1_score(y_dec, dec_pred, average="weighted")
try:
    dec_auc = roc_auc_score(y_dec, dec_proba)
except ValueError:
    dec_auc = float("nan")

print(f"December forward-generalisation F1 : {dec_f1:.4f}")
print(f"December forward-generalisation AUC: {dec_auc:.4f}")
print("(If this stays high, unlike the near-total collapse in the main")
print(" novel-class experiment, that confirms accuracy holds up fine on")
print(" a class the model already knows.)")

# ──────────────────────────────────────────────────────────────────────────
# STEP 7: compute the December SHAP ranking and compare
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Computing December SHAP ranking and comparing to November ...")
print("=" * 60)

dec_ranking_df, dec_shap_array, dec_scores = compute_shap_rankings(
    explainer, X_dec, feature_names
)
print("Top 5 features (December, same model):")
print(dec_ranking_df.head(5).to_string(index=False))

nov_aligned = nov_ranking_df.set_index("feature").reindex(feature_names)
dec_aligned = dec_ranking_df.set_index("feature").reindex(feature_names)

ess, pvalue = compute_ess(nov_aligned["rank"].values, dec_aligned["rank"].values)
euclidean = compute_euclidean(nov_scores, dec_scores)
psi = compute_psi(nov_shap_array, dec_shap_array)

print(f"\nESS (November baseline vs December, same model): {ess:.4f} (p={pvalue:.4f})")
print(f"Euclidean distance (SHAP magnitude)             : {euclidean:.4f}")
print(f"PSI (SHAP distribution)                         : {psi:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 8: save results
# ──────────────────────────────────────────────────────────────────────────
result = {
    "november_holdout_f1": round(nov_f1, 4),
    "december_f1": round(dec_f1, 4),
    "december_auc": round(dec_auc, 4),
    "ess": round(ess, 4),
    "ess_pvalue": round(pvalue, 4),
    "euclidean": round(euclidean, 4),
    "psi": round(psi, 4),
}
pd.DataFrame([result]).to_csv(f"{RESULTS_DIR}/ldap_drift_check.csv", index=False)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for k, v in result.items():
    print(f"  {k}: {v}")
print(f"\nSaved: {RESULTS_DIR}/ldap_drift_check.csv")
print("\nInterpretation guide:")
print("- High December F1 + ESS clearly below 1.0 would be the cleanest possible")
print("  finding: accuracy holds up fine on a known class, but the model's")
print("  reasoning still shifted between the two dates. That is genuine drift,")
print("  distinct from the novel-class effect in the main experiment.")
print("- High December F1 + ESS very close to 1.0 would mean no meaningful")
print("  drift was detected for this class over this particular time gap,")
print("  which is also a legitimate, reportable result.")
print("- Report whichever outcome actually happens. Do not rerun with a")
print("  different seed or sample size looking for a specific answer.")