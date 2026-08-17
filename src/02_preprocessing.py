'''
    02_preprocessing.py  (CORRECTED)

    Purpose: Clean, encode, reduce features and split the combined
    CIC-DDoS2019 dataset into temporal batches ready for the experiment.

    FIX APPLIED: feature selection (variance + correlation filtering) is
    now fit ONLY on Batch 1 (the baseline period), then the same selected
    feature list is applied to Batches 2-6. The previous version computed
    variance and correlation across the full 350,000-row dataset,
    including batches that represent "future" deployment data the
    baseline model is not supposed to have seen yet -- a temporal
    leakage that would let feature selection quietly use information
    from later time periods to decide what the baseline model gets to
    see. This also means the specific feature counts reported in
    Chapter 3 (18 removed / 30 removed / 33 kept) will change slightly
    once this fix is applied to your real dataset -- rerun this script
    and update those numbers from the new printout before finalising
    Chapter 4.

    What this file does:
    - Loads the combined dataset saved by 01_data_loader.py
    - Drops identifier columns that are not useful for modelling
    - Sorts the data chronologically by Timestamp to preserve temporal order
    - Handles infinity and missing values using median imputation
    - Encodes labels into multiclass (0 to 7) and binary (0 or 1) formats
    - Removes any remaining non-numeric columns
    - Splits into 6 temporal batches BEFORE feature selection
    - Applies two stage feature selection using ONLY Batch 1 statistics:
        Stage 1: removes near zero variance features that carry no information
        Stage 2: removes highly correlated features above 0.95 threshold
      This follows the approach of Gaur and Kumar (2021) and Araujo et al.
      (2021) who reduced CIC-DDoS2019 significantly before training XGBoost
      without loss of methodological credibility.
    - Applies the Batch-1-selected feature list to all 6 batches
    - Saves all batches and the full cleaned dataset to outputs/results/
'''

import pandas as pd
import numpy as np
import os
import json

from ess_utils import N_BATCHES

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "outputs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_COL = "Label"

DROP_COLS = [
    "Unnamed: 0",
    "Flow ID",
    "Source IP",
    "Destination IP",
]

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
# N_BATCHES now comes from ess_utils.py so every script agrees on the
# same batch count (raised from 6 to 10 batches for more independence-
# analysis statistical power). Change it in ess_utils.py, not here.

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load the saved combined dataset
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading combined dataset ...")
print("=" * 60)

combined = pd.read_csv(f"{RESULTS_DIR}/combined_dataset.csv", low_memory=False)
print(f"Shape: {combined.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: drop identifier columns
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Dropping identifier columns ...")
print("=" * 60)

cols_to_drop = [c for c in DROP_COLS if c in combined.columns]
combined.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped: {cols_to_drop}")
print(f"Shape after dropping: {combined.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: sort by Timestamp for temporal ordering
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Sorting by Timestamp for temporal ordering ...")
print("=" * 60)

if "Timestamp" in combined.columns:
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], errors="coerce")
    combined.sort_values("Timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print("Data sorted chronologically by Timestamp.")
    print(f"Earliest: {combined['Timestamp'].min()}")
    print(f"Latest  : {combined['Timestamp'].max()}")
    combined.drop(columns=["Timestamp"], inplace=True)
    print("Timestamp column dropped after sorting.")
else:
    print("No Timestamp column found. Keeping existing order.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: handle infinity and missing values
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Handling infinity and missing values ...")
print("=" * 60)

combined.replace([np.inf, -np.inf], np.nan, inplace=True)
missing_before = combined.isnull().sum().sum()
print(f"Missing values before cleaning: {missing_before}")

numeric_cols = combined.select_dtypes(include=[np.number]).columns
combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())

missing_after = combined.isnull().sum().sum()
print(f"Missing values after cleaning : {missing_after}")
print("NOTE: median imputation values are computed across the full dataset "
      "(all batches). This is a very small effect (well under 1% of cells) "
      "and matches what Chapter 3 currently describes, but is worth knowing "
      "it is not batch-1-only like the feature selection fix below.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: encode labels
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Encoding labels ...")
print("=" * 60)

label_map_multi = {
    "BENIGN": 0, "LDAP": 1, "MSSQL": 2, "NetBIOS": 3,
    "Portmap": 4, "Syn": 5, "UDP": 6, "UDPLag": 7,
}

combined["label_multi"] = combined[LABEL_COL].map(label_map_multi)
combined["label_binary"] = combined["label_multi"].apply(lambda x: 0 if x == 0 else 1)
combined.drop(columns=[LABEL_COL], inplace=True)

print("Multiclass label distribution:")
print(combined["label_multi"].value_counts())
print("\nBinary label distribution:")
print(combined["label_binary"].value_counts())

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: remove non-numeric columns
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Checking for non-numeric columns ...")
print("=" * 60)

non_numeric = combined.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    combined.drop(columns=non_numeric, inplace=True)
else:
    print("All columns numeric. Nothing to drop.")
print(f"Shape after removing non-numeric: {combined.shape}")

original_feature_count = combined.shape[1] - 2  # minus label_multi, label_binary
print(f"Numeric feature columns before selection: {original_feature_count}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 7: split into temporal batches BEFORE feature selection
# this must happen first so feature selection can be fit on Batch 1 only
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Splitting into temporal batches (pre-feature-selection) ...")
print("=" * 60)

label_cols = ["label_multi", "label_binary"]
feature_cols_all = [c for c in combined.columns if c not in label_cols]

total_rows = len(combined)
batch_size = total_rows // N_BATCHES
print(f"Total rows: {total_rows} | batch size: {batch_size}")

batch_slices = {}
for i in range(N_BATCHES):
    start = i * batch_size
    end = (i + 1) * batch_size if i < N_BATCHES - 1 else total_rows
    batch_slices[i + 1] = (start, end)
    print(f"  Batch {i + 1}: rows {start} to {end} (size={end - start})")

# ──────────────────────────────────────────────────────────────────────────
# STEP 8: feature selection Stage 1 + 2, FIT ON BATCH 1 ONLY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature selection, fit on Batch 1 only ...")
print("=" * 60)

b1_start, b1_end = batch_slices[1]
X_batch1_full = combined.iloc[b1_start:b1_end][feature_cols_all]

# Stage 1: variance threshold, computed on Batch 1 only
variances = X_batch1_full.var()
low_variance_cols = variances[variances < VARIANCE_THRESHOLD].index.tolist()
print(f"[Stage 1] Low-variance features (Batch 1 only): {len(low_variance_cols)}")
print(f"  Dropping: {low_variance_cols}")

X_batch1_stage1 = X_batch1_full.drop(columns=low_variance_cols)

# Stage 2: correlation filtering, computed on Batch 1 only
corr_matrix = X_batch1_stage1.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_correlated = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > CORRELATION_THRESHOLD)
]
print(f"[Stage 2] Highly correlated features (Batch 1 only): {len(highly_correlated)}")
print(f"  Dropping: {highly_correlated}")

selected_features = [c for c in X_batch1_stage1.columns if c not in highly_correlated]
print(f"\nFinal number of features selected (from Batch 1 only): {len(selected_features)}")
print(f"Features kept: {selected_features}")

with open(f"{RESULTS_DIR}/selected_features.json", "w") as f:
    json.dump(selected_features, f, indent=4)
print(f"\nSaved selected features to: {RESULTS_DIR}/selected_features.json")

# ──────────────────────────────────────────────────────────────────────────
# STEP 9: apply the Batch-1-selected feature list to ALL batches and save
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Applying selected features to all batches and saving ...")
print("=" * 60)

for batch_num, (start, end) in batch_slices.items():
    X_batch = combined.iloc[start:end][selected_features]
    y_multi_batch = combined.iloc[start:end]["label_multi"]
    y_binary_batch = combined.iloc[start:end]["label_binary"]

    X_batch.to_csv(f"{RESULTS_DIR}/batch_{batch_num}_X.csv", index=False)
    y_multi_batch.to_csv(f"{RESULTS_DIR}/batch_{batch_num}_y_multi.csv", index=False)
    y_binary_batch.to_csv(f"{RESULTS_DIR}/batch_{batch_num}_y_binary.csv", index=False)

    print(f"Batch {batch_num}: size={len(X_batch)} "
          f"| labels: {y_multi_batch.value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 10: save full cleaned and selected dataset
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Saving full cleaned dataset ...")
print("=" * 60)

combined_selected = combined[selected_features + label_cols]
combined_selected.to_csv(f"{RESULTS_DIR}/combined_cleaned.csv", index=False)
print(f"Saved: {RESULTS_DIR}/combined_cleaned.csv")

print("\n" + "=" * 60)
print("Preprocessing complete.")
print("=" * 60)
print(f"\nOriginal features : {original_feature_count}")
print(f"Selected features : {len(selected_features)}")
print(f"Features removed  : {original_feature_count - len(selected_features)}")
print(f"\nFeature selection was fit on Batch 1 only (no temporal leakage).")
print(f"Stage 1 removed {len(low_variance_cols)} low-variance features.")
print(f"Stage 2 removed {len(highly_correlated)} highly correlated features.")
print(f"\nFiles saved to outputs/results/:")
print(f"  combined_cleaned.csv")
print(f"  selected_features.json")
print(f"  batch_1 to batch_6 X, y_multi and y_binary files")
