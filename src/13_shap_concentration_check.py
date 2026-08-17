'''
    13_shap_concentration_check.py

    Purpose: Test whether the mixed pattern seen in
    12_synthetic_drift_injection.py (ESS most sensitive for some
    classes, F1 for others, PSI for others) is predictable from how
    concentrated each class's baseline SHAP importance is across its
    41 features, rather than being unexplained variation.

    This is deliberately a FAST, standalone script: it only trains the
    baseline model and computes the baseline SHAP ranking for each
    class, exactly as 12_synthetic_drift_injection.py does before its
    drift loop, but it skips the eleven-step drift loop entirely, so
    it should run in a fraction of the time.

    What this file does, for each of the seven attack classes:
    - Loads and cleans that class's November CSV (same chunked-reading
      approach as script 12, so large files like MSSQL do not run out
      of memory)
    - Trains the same binary model and computes the baseline SHAP
      ranking across all 41 features
    - Computes three concentration measures from that ranking:
        top1_share : share of total |SHAP| importance held by the
                     single most important feature
        top3_share : share held by the top 3 features combined
        top5_share : share held by the top 5 features combined
      (all as a fraction of the sum of |SHAP| importance across all
      41 features, so 1.0 would mean total concentration in that many
      features and a low value means importance is spread evenly)
    - Saves one row per class to
      outputs/results/shap_concentration_check.csv
'''

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ess_utils import compute_shap_rankings

RESULTS_DIR = "outputs/results"
RAW_DIR = "data/raw/03-11"
SAMPLE_SIZE = 30_000
SEED = 42

ATTACK_CLASSES = ["LDAP", "MSSQL", "NetBIOS", "Portmap", "Syn", "UDP", "UDPLag"]

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    feature_names = json.load(f)


def load_and_clean(class_name, sample_size, seed=SEED):
    path = f"{RAW_DIR}/{class_name}.csv"
    chunks = []
    for encoding in ["utf-8", "latin-1"]:
        try:
            reader = pd.read_csv(path, encoding=encoding, low_memory=False, chunksize=100_000)
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


results = []

for class_name in ATTACK_CLASSES:
    print("=" * 60)
    print(f"CLASS: {class_name}")
    print("=" * 60)

    try:
        X_full, y_full = load_and_clean(class_name, SAMPLE_SIZE)
        if y_full.nunique() < 2:
            print(f"SKIPPING {class_name}: only one class in the sample.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.3, random_state=SEED, stratify=y_full
        )

        params = {
            "objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
            "eval_metric": "logloss", "seed": SEED, "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=100)
        f1 = f1_score(y_test, (model.predict(xgb.DMatrix(X_test)) > 0.5).astype(int), average="weighted")

        explainer = shap.TreeExplainer(model)
        ranking_df, _, scores = compute_shap_rankings(explainer, X_train, feature_names)

        total_importance = ranking_df["shap_score"].sum()
        top1_share = ranking_df["shap_score"].iloc[0] / total_importance
        top3_share = ranking_df["shap_score"].iloc[:3].sum() / total_importance
        top5_share = ranking_df["shap_score"].iloc[:5].sum() / total_importance

        print(f"Held-out F1: {f1:.4f}")
        print(f"Top 5 features:\n{ranking_df.head(5).to_string(index=False)}")
        print(f"top1_share={top1_share:.4f} | top3_share={top3_share:.4f} | top5_share={top5_share:.4f}")

        results.append({
            "class": class_name,
            "holdout_f1": round(f1, 4),
            "top1_share": round(top1_share, 4),
            "top3_share": round(top3_share, 4),
            "top5_share": round(top5_share, 4),
            "top_feature": ranking_df["feature"].iloc[0],
        })

    except Exception as e:
        print(f"!!! {class_name} FAILED with: {e}")

    # Save progress after every class
    pd.DataFrame(results).to_csv(f"{RESULTS_DIR}/shap_concentration_check.csv", index=False)

print("\n" + "=" * 60)
print("Full results")
print("=" * 60)
print(pd.DataFrame(results).to_string(index=False))
print(f"\nSaved: {RESULTS_DIR}/shap_concentration_check.csv")