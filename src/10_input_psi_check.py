'''
    10_input_psi_check.py

    Purpose: Supplementary check requested by supervisor feedback on
    Chapter 4. Computes PSI on the RAW INPUT FEATURES, the standard,
    industry definition of PSI following Khademi, Hopka and Upadhyay
    (2023) as already cited in Chapter 2, rather than on SHAP values.

    Section 4.4 reports a strong correlation between ESS and PSI
    (r = -0.840). Because that PSI is computed on SHAP values, the same
    numbers ESS is built from, part of that correlation may simply
    reflect the two metrics sharing an underlying source rather than
    genuine agreement between rank-based and distribution-based
    monitoring. This script computes a second, independent PSI directly
    on the 41 selected input features and checks whether the same
    correlation with ESS still shows up.

    This does NOT replace the SHAP-based PSI used throughout Chapter 4.
    It is an additional, supplementary check run alongside it.

    What this file does:
    - Loads Batch 1 (baseline) and Batches 2-10 (deployment) feature
      matrices, using the same 41 selected features as the main pipeline
    - For each deployment batch, computes PSI between the baseline and
      that batch separately for each of the 41 input features, reusing
      the exact PSI formula already defined and tested in ess_utils.py
    - Reports the mean PSI across all 41 features per batch (the usual
      way multiple features get rolled into one dashboard-level number
      in industry PSI monitoring), plus the single feature with the
      highest individual PSI in that batch
    - Computes the Spearman correlation between this input-feature PSI
      and ESS, and between this input-feature PSI and the SHAP-based
      PSI already reported in Section 4.4, to see whether the strong
      ESS-vs-PSI correlation reported there survives when PSI is
      computed on inputs rather than on SHAP values
    - Saves results to outputs/results/input_psi_check.csv
'''

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from ess_utils import compute_psi, DEPLOYMENT_BATCHES

RESULTS_DIR = "outputs/results"

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    feature_names = json.load(f)

print("=" * 60)
print("STEP 1: Loading Batch 1 (baseline) input features ...")
print("=" * 60)

X_baseline = pd.read_csv(f"{RESULTS_DIR}/batch_1_X.csv")[feature_names]

print("\n" + "=" * 60)
print("STEP 2: Computing per-feature PSI for each deployment batch ...")
print("=" * 60)

rows = []
for batch_num in DEPLOYMENT_BATCHES:
    X_batch = pd.read_csv(f"{RESULTS_DIR}/batch_{batch_num}_X.csv")[feature_names]

    per_feature_psi = {}
    for feat in feature_names:
        psi_val = compute_psi(X_baseline[feat].values, X_batch[feat].values)
        per_feature_psi[feat] = psi_val

    mean_psi = float(np.mean(list(per_feature_psi.values())))
    top_feature = max(per_feature_psi, key=per_feature_psi.get)
    top_psi = per_feature_psi[top_feature]

    print(f"Batch {batch_num}: mean input PSI = {mean_psi:.4f} "
          f"| highest single feature = {top_feature} ({top_psi:.4f})")

    rows.append({
        "batch": batch_num,
        "mean_input_psi": round(mean_psi, 4),
        "max_input_psi": round(top_psi, 4),
        "max_input_psi_feature": top_feature,
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(f"{RESULTS_DIR}/input_psi_check.csv", index=False)

print("\n" + "=" * 60)
print("STEP 3: Checking correlation against ESS and against SHAP-based PSI ...")
print("=" * 60)

ess_scores = pd.read_csv(f"{RESULTS_DIR}/ess_scores.csv")
merged = results_df.merge(ess_scores, on="batch")

r_ess, p_ess = spearmanr(merged["mean_input_psi"], merged["ess"])
r_psi, p_psi = spearmanr(merged["mean_input_psi"], merged["psi"])

print(f"Spearman(input PSI, ESS): r = {r_ess:.3f}, p = {p_ess:.3f}")
print(f"Spearman(input PSI, SHAP-based PSI): r = {r_psi:.3f}, p = {p_psi:.3f}")
print(f"\nSaved: {RESULTS_DIR}/input_psi_check.csv")
print("\nInterpretation guide:")
print("- If input PSI still correlates strongly with ESS, that is stronger")
print("  evidence that the two signals genuinely agree, not just an artefact")
print("  of sharing SHAP values.")
print("- If input PSI does NOT correlate the same way with ESS, that supports")
print("  the concern raised in Chapter 4 Section 4.4: the SHAP-based PSI")
print("  correlation was partly a product of shared computation, not a")
print("  substantive agreement between the two monitoring approaches.")
print("- Either outcome is a legitimate, reportable result -- do not rerun")
print("  with different bins or thresholds looking for a 'better' answer.")