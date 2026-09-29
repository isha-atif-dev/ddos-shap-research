'''
    04_imputation_leakage_check.py

    Purpose: Quantify whether the combined-dataset median imputation in
    02_preprocessing.py (Step 4) materially differs from what a
    baseline-only (Batch 1 only) median imputation would have produced.

    This does NOT change any saved outputs. It is a read-only diagnostic
    that reproduces Steps 1-4 of 02_preprocessing.py, then compares two
    medians per affected column:
        - median_full   : computed on the full combined dataset
                           (what was actually used)
        - median_batch1 : computed using only the rows that become
                           Batch 1 after the Step 7 split

    Since fillna() replaces every missing value in a column with a
    single median, columns where the two medians match would have
    received IDENTICAL imputed values either way. Only columns where
    the two medians differ actually represent leaked information.

    It also cross-checks whether any affected column survived feature
    selection (is in selected_features.json). If an affected column
    was dropped during feature selection, any leakage in it cannot
    have touched the reported experimental results at all.
'''

import pandas as pd
import numpy as np
import json

from ess_utils import N_BATCHES

RESULTS_DIR = "outputs/results"
DROP_COLS = ["Unnamed: 0", "Flow ID", "Source IP", "Destination IP"]

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: reload and reproduce Steps 1-3 of 02_preprocessing.py exactly
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Reloading combined dataset and reproducing preprocessing "
      "Steps 1-3 ...")
print("=" * 60)

combined = pd.read_csv(f"{RESULTS_DIR}/combined_dataset.csv", low_memory=False)

cols_to_drop = [c for c in DROP_COLS if c in combined.columns]
combined.drop(columns=cols_to_drop, inplace=True)

if "Timestamp" in combined.columns:
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
    combined.sort_values("Timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.drop(columns=["Timestamp"], inplace=True)

print(f"Shape after reproducing Steps 1-3: {combined.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: reproduce Step 4's inf-to-nan replacement, BEFORE imputation
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Replacing infinities with NaN (matches Step 4) ...")
print("=" * 60)

combined.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = combined.select_dtypes(include=[np.number]).columns

affected_cols = [c for c in numeric_cols if combined[c].isna().any()]
total_affected_cells = int(combined[affected_cols].isna().sum().sum())
total_cells = combined.shape[0] * len(numeric_cols)

print(f"Numeric columns checked : {len(numeric_cols)}")
print(f"Columns with missing/inf values : {len(affected_cols)}")
print(f"Total affected cells : {total_affected_cells} "
      f"({100 * total_affected_cells / total_cells:.4f}% of numeric cells)")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: determine Batch 1 row range using the same logic as Step 7
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Determining Batch 1 row range ...")
print("=" * 60)

total_rows = len(combined)
batch_size = total_rows // N_BATCHES
batch1_end = batch_size  # matches: i=0, end = batch_size since 0 < N_BATCHES - 1

print(f"Total rows: {total_rows} | N_BATCHES: {N_BATCHES} | "
      f"Batch 1 = rows 0 to {batch1_end}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: compare full-dataset median vs batch-1-only median, per column
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Comparing medians for each affected column ...")
print("=" * 60)

try:
    with open(f"{RESULTS_DIR}/selected_features.json") as f:
        selected_features = set(json.load(f))
except FileNotFoundError:
    print("selected_features.json not found -- run 02_preprocessing.py first.")
    selected_features = set()

results = []
for col in affected_cols:
    median_full = combined[col].median()
    median_batch1 = combined[col].iloc[:batch1_end].median()
    n_affected = int(combined[col].isna().sum())
    changed = not np.isclose(median_full, median_batch1, equal_nan=True)
    in_final_features = col in selected_features

    results.append({
        "column": col,
        "n_affected_cells": n_affected,
        "median_full": median_full,
        "median_batch1": median_batch1,
        "abs_difference": abs(median_full - median_batch1) if changed else 0.0,
        "would_change_imputed_value": changed,
        "survived_feature_selection": in_final_features,
    })

results_df = pd.DataFrame(results).sort_values(
    "n_affected_cells", ascending=False
)

print(results_df.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: summarise the practical impact
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Summary ...")
print("=" * 60)

changed_df = results_df[results_df["would_change_imputed_value"]]
cells_that_would_change = int(changed_df["n_affected_cells"].sum())
changed_and_kept = changed_df[changed_df["survived_feature_selection"]]

print(f"Columns where the two medians genuinely differ: {len(changed_df)} "
      f"of {len(affected_cols)} affected columns")
print(f"Total cells that would receive a different imputed value under "
      f"baseline-only imputation: {cells_that_would_change} "
      f"({100 * cells_that_would_change / total_cells:.5f}% of all numeric cells)")
print(f"Of those, columns that survived feature selection and are "
      f"actually used in the reported experiments: {len(changed_and_kept)}")

results_df.to_csv(f"{RESULTS_DIR}/imputation_leakage_check.csv", index=False)
print(f"\nSaved full comparison to: {RESULTS_DIR}/imputation_leakage_check.csv")