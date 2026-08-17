'''
    04_ess_engine.py  (CORRECTED)

    Purpose: Compute the Explanation Stability Score (ESS) for each
    deployment batch by comparing its SHAP feature importance rankings
    against the baseline rankings from Batch 1, using the UNCHANGED
    baseline model throughout (i.e. this reproduces the "No Retraining"
    trajectory on its own, as a standalone sanity check of the ESS
    metric itself, independent of the full four-condition experiment
    in 05_experiment_main.py).

    FIX APPLIED: now imports compute_ess, compute_psi, compute_euclidean
    and compute_shap_rankings from ess_utils.py instead of carrying its
    own copy of this logic. The previous copy in this file assumed SHAP
    always returns a 3D (rows, features, classes) array, which crashes
    on any batch with very few classes present (ess_utils handles both
    2D and 3D shapes).

    NOTE: because this script and 05_experiment_main.py both compute ESS
    against the unchanged baseline model, the numbers in this script's
    output (ess_scores.csv) should be identical to the "No Retraining"
    rows of 05's experiment_results.csv. If you don't need this as a
    separate deliverable, 05's output already covers it -- but it's kept
    here since Chapter 3 documents it as a standalone module.
'''

import pandas as pd
import numpy as np
import os
import json
import pickle

from ess_utils import (
    compute_ess, compute_psi, compute_euclidean, compute_shap_rankings,
    DEPLOYMENT_BATCHES,
)

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "outputs/results"
MODELS_DIR = "outputs/models"
ESS_THRESHOLD = 0.97

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load baseline rankings, model and explainer
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading baseline rankings, model and explainer ...")
print("=" * 60)

baseline_rankings = pd.read_csv(f"{RESULTS_DIR}/baseline_feature_rankings.csv")
print(f"Baseline rankings loaded: {len(baseline_rankings)} features")

with open(f"{RESULTS_DIR}/feature_names.json", "r") as f:
    feature_names = json.load(f)

with open(f"{MODELS_DIR}/baseline_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{MODELS_DIR}/baseline_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

print("Baseline model and explainer loaded.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: prepare baseline vectors
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preparing baseline ranking vector ...")
print("=" * 60)

baseline_aligned = baseline_rankings.set_index("feature").reindex(feature_names)
baseline_rank_vector = baseline_aligned["rank"].values
baseline_score_vector = baseline_aligned["shap_score"].values

X_batch1 = pd.read_csv(f"{RESULTS_DIR}/batch_1_X.csv")[feature_names]
_, baseline_shap_array, baseline_scores = compute_shap_rankings(explainer, X_batch1, feature_names)
print("Baseline vectors ready.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: compute ESS, Euclidean and PSI for each deployment batch
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing ESS for each deployment batch ...")
print("=" * 60)

results = []

for batch_num in DEPLOYMENT_BATCHES:
    print(f"\n{'-' * 40}")
    print(f"Processing Batch {batch_num} ...")
    print(f"{'-' * 40}")

    X_batch = pd.read_csv(f"{RESULTS_DIR}/batch_{batch_num}_X.csv")[feature_names]

    batch_ranking_df, batch_shap_array, batch_scores = compute_shap_rankings(
        explainer, X_batch, feature_names
    )
    batch_aligned = batch_ranking_df.set_index("feature").reindex(feature_names)
    batch_rank_vector = batch_aligned["rank"].values

    ess, pvalue = compute_ess(baseline_rank_vector, batch_rank_vector)
    euclidean = compute_euclidean(baseline_scores, batch_scores)
    psi = compute_psi(baseline_shap_array, batch_shap_array)
    drift_detected = ess < ESS_THRESHOLD

    print(f"ESS (Spearman)     : {ess:.4f}")
    print(f"P-value            : {pvalue:.4f}")
    print(f"Euclidean Distance : {euclidean:.4f}")
    print(f"PSI                : {psi:.4f}")
    print(f"Drift Detected     : {drift_detected}")
    print(f"Top 5 features in Batch {batch_num}:")
    print(batch_ranking_df.head(5).to_string(index=False))

    results.append({
        "batch": batch_num,
        "ess": round(ess, 4),
        "pvalue": round(pvalue, 4),
        "euclidean": round(euclidean, 4),
        "psi": round(psi, 4),
        "drift_detected": drift_detected,
        "top_feature": batch_ranking_df.iloc[0]["feature"],
    })

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: save all ESS results
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Saving ESS results ...")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/ess_scores.csv", index=False)

print("\nESS Results Summary:")
print(results_df.to_string(index=False))
print(f"\nSaved: {RESULTS_DIR}/ess_scores.csv")
