'''
    02_preprocessing.py  (CIC-IDS2017)

    Purpose: Clean, encode and feature-select the 7 real batches
    produced by 01_data_loader.py, ready for the experiment.

    HOW THIS DIFFERS FROM THE ORIGINAL DDoS2019 VERSION
    -----------------------------------------------------
    The original pipeline combined 7 sampled files into one table and
    split it into N equal-sized batches BY ROW POSITION after sorting
    by Timestamp. Here, the 7 batches already exist as genuinely
    separate, real chronological periods (Monday+Tuesday baseline,
    then five real weekday deployment periods), produced directly by
    01_data_loader.py. There is no row-position splitting to do here,
    each source file already IS a batch. This script combines them
    only briefly, to clean and encode all 7 consistently in one pass,
    then splits them back apart using their original batch identity.

    Feature selection is still fit on the baseline batch ONLY and then
    applied uniformly to the other six, exactly the same leakage-
    avoidance principle as the original pipeline.

    Missing-value imputation uses the median computed across all 7
    batches combined, matching the original pipeline's documented
    choice (not batch-1-only), so this stays methodologically
    identical to what Chapter 3 already describes and justifies.

    What this file does:
    - Loads the 7 batch CSVs from 01_data_loader.py
    - Tags each row with its batch number (1 = baseline, 2-7 = deployment)
    - Drops identifier-style columns, if present
    - Handles infinity and missing values via median imputation across
      the combined data
    - Encodes labels to integers using map_label_to_int() from
      ess_utils_cicids2017.py
    - Removes any remaining non-numeric columns
    - Fits two-stage feature selection (variance, then correlation) on
      batch 1 only, applies the same selected feature list to all 7
    - Saves batch_{1-7}_X.csv and batch_{1-7}_y_multi.csv, plus
      selected_features.json, to outputs_cicids2017/results/
'''

import os
import json
import numpy as np
import pandas as pd

from ess_utils_cicids2017 import map_label_to_int, N_BATCHES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed", "cicids2017"))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_COL = "Label"

# Maps each real output batch number to the file 01_data_loader.py
# produced for it. Batch 1 is the baseline; batches 2-7 are deployment,
# in chronological order.
BATCH_FILES = {
    1: "batch_00_baseline.csv",
    2: "batch_01_wednesday.csv",
    3: "batch_02_thursday_webattacks.csv",
    4: "batch_03_thursday_infiltration.csv",
    5: "batch_04_friday_bot.csv",
    6: "batch_05_friday_portscan.csv",
    7: "batch_06_friday_ddos.csv",
}

# Identifier-style columns dropped defensively (only if present), plus
# Destination Port, dropped for a documented data quality reason
# specific to this dataset: FTP-Patator and SSH-Patator target fixed
# ports (21 and 22) by protocol definition, so a model given raw
# Destination Port can reach near-perfect accuracy by looking up the
# port rather than learning behavioural traffic patterns. This showed
# up directly in the first baseline run: Destination Port ranked #1 in
# SHAP importance at roughly 9x the next feature, and F1 came out at a
# suspiciously perfect 1.0000. Excluding this column for this reason
# is established practice in published CIC-IDS2017 research, not a
# choice specific to this pipeline.
DROP_COLS = [
    "Unnamed: 0",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Destination Port",
]

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95


def main():
    # ── STEP 1: load all 7 batches, tagging each with its batch number ──
    print("=" * 60)
    print("STEP 1: Loading 7 batches ...")
    print("=" * 60)

    pieces = []
    for batch_num, fname in BATCH_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, low_memory=False)
        df["_batch_number"] = batch_num
        pieces.append(df)
        print(f"  Batch {batch_num} ({fname}): {df.shape[0]} rows")

    combined = pd.concat(pieces, ignore_index=True)
    print(f"\nCombined shape (all 7 batches, tagged): {combined.shape}")

    # ── STEP 2: drop identifier columns, if present ──
    print("\n" + "=" * 60)
    print("STEP 2: Dropping identifier columns (if present) ...")
    print("=" * 60)

    cols_to_drop = [c for c in DROP_COLS if c in combined.columns]
    combined.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped: {cols_to_drop}")
    print(f"Shape after dropping: {combined.shape}")

    # ── STEP 3: handle infinity and missing values ──
    print("\n" + "=" * 60)
    print("STEP 3: Handling infinity and missing values ...")
    print("=" * 60)

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_before = combined.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "_batch_number"]
    combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())

    missing_after = combined.isnull().sum().sum()
    print(f"Missing values after cleaning : {missing_after}")
    print("(Median computed across all 7 batches combined, matching the "
          "original pipeline's documented choice for this step.)")

    # ── STEP 4: encode labels ──
    print("\n" + "=" * 60)
    print("STEP 4: Encoding labels ...")
    print("=" * 60)

    combined["label_multi"] = combined[LABEL_COL].apply(map_label_to_int)
    combined.drop(columns=[LABEL_COL], inplace=True)

    print("Multiclass label distribution (whole dataset):")
    print(combined["label_multi"].value_counts().sort_index())

    # ── STEP 5: remove any remaining non-numeric columns ──
    print("\n" + "=" * 60)
    print("STEP 5: Checking for non-numeric columns ...")
    print("=" * 60)

    non_numeric = combined.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric = [c for c in non_numeric if c != "_batch_number"]
    if non_numeric:
        print(f"Dropping non-numeric columns: {non_numeric}")
        combined.drop(columns=non_numeric, inplace=True)
    else:
        print("All remaining columns numeric. Nothing to drop.")

    original_feature_count = combined.shape[1] - 2  # minus label_multi, _batch_number
    print(f"Numeric feature columns before selection: {original_feature_count}")

    # ── STEP 6: feature selection, fit on batch 1 (baseline) only ──
    print("\n" + "=" * 60)
    print("STEP 6: Feature selection, fit on Batch 1 (baseline) only ...")
    print("=" * 60)

    feature_cols_all = [c for c in combined.columns if c not in ("label_multi", "_batch_number")]
    batch1_mask = combined["_batch_number"] == 1
    X_batch1_full = combined.loc[batch1_mask, feature_cols_all]

    variances = X_batch1_full.var()
    low_variance_cols = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    print(f"[Stage 1] Low-variance features (Batch 1 only): {len(low_variance_cols)}")
    print(f"  Dropping: {low_variance_cols}")

    X_batch1_stage1 = X_batch1_full.drop(columns=low_variance_cols)

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

    with open(os.path.join(RESULTS_DIR, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=4)
    print(f"Saved: {RESULTS_DIR}/selected_features.json")

    # ── STEP 7: apply selected features to all 7 batches and save ──
    print("\n" + "=" * 60)
    print("STEP 7: Applying selected features to all batches and saving ...")
    print("=" * 60)

    for batch_num in BATCH_FILES.keys():
        mask = combined["_batch_number"] == batch_num
        X_batch = combined.loc[mask, selected_features]
        y_batch = combined.loc[mask, "label_multi"]

        X_batch.to_csv(os.path.join(RESULTS_DIR, f"batch_{batch_num}_X.csv"), index=False)
        y_batch.to_csv(os.path.join(RESULTS_DIR, f"batch_{batch_num}_y_multi.csv"), index=False)

        print(f"Batch {batch_num}: size={len(X_batch)} | "
              f"labels: {y_batch.value_counts().sort_index().to_dict()}")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print("=" * 60)
    print(f"\nOriginal features : {original_feature_count}")
    print(f"Selected features : {len(selected_features)}")
    print(f"Features removed  : {original_feature_count - len(selected_features)}")
    print(f"\nFiles saved to {RESULTS_DIR}:")
    print("  selected_features.json")
    print("  batch_1_X.csv / batch_1_y_multi.csv  (baseline)")
    print("  batch_2_X.csv / batch_2_y_multi.csv  through  batch_7_*")


if __name__ == "__main__":
    main()