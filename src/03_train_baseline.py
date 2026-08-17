'''
    03_train_baseline.py  (CORRECTED)

    Purpose: Train the XGBoost classifier on Batch 1 (the baseline period)
    and compute the baseline SHAP feature importance rankings.

    FIX APPLIED: training now uses the native xgboost.train() Booster API
    with num_class fixed at 8 (the full known CIC-DDoS2019 class schema),
    instead of sklearn's XGBClassifier.fit() with a per-batch label
    remap. The previous remap-based approach silently created a
    different label space every time it ran, which is what caused the
    evaluation mismatch bug in 05_experiment_main.py. There is now
    exactly one label space (the global label_multi encoding from
    02_preprocessing.py, 0-7) used everywhere in this pipeline, and no
    remapping step anywhere.

    What this file does:
    - Loads Batch 1 features and labels saved by 02_preprocessing.py
    - Trains an XGBoost classifier on the baseline period data
    - Evaluates the model on a held out test split from Batch 1
    - Computes SHAP values using TreeExplainer on the training data
    - Extracts and saves the baseline feature importance rankings
    - Saves the trained model for use in the experiment scripts
    - Saves evaluation metrics (F1, AUC-ROC) for the baseline period
'''

import pandas as pd
import numpy as np
import os
import json
import pickle
import shap
from sklearn.model_selection import train_test_split

from ess_utils import train_xgb, evaluate_model, compute_shap_rankings

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "outputs/results"
MODELS_DIR = "outputs/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load batch 1 (baseline training period)
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Batch 1 (baseline period) ...")
print("=" * 60)

X_batch1 = pd.read_csv(f"{RESULTS_DIR}/batch_1_X.csv")
y_batch1 = pd.read_csv(f"{RESULTS_DIR}/batch_1_y_multi.csv").squeeze()

print(f"Batch 1 features shape : {X_batch1.shape}")
print(f"Batch 1 labels shape   : {y_batch1.shape}")
print(f"Label distribution:\n{y_batch1.value_counts()}")
print(f"Classes present in Batch 1 (global 0-7 scheme): {sorted(y_batch1.unique())}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: split batch 1 into train and test
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Splitting Batch 1 into train and test ...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_batch1, y_batch1,
    test_size=0.2,
    random_state=42,
    stratify=y_batch1
)

print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: train XGBoost classifier (native API, fixed global num_class=8)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training XGBoost classifier (num_class=8, global labels) ...")
print("=" * 60)

model = train_xgb(X_train, y_train)
print("XGBoost model trained successfully.")
print("No label remapping was applied -- labels stay in the global 0-7 scheme.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: evaluate the model on the test set
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Evaluating model on test set ...")
print("=" * 60)

f1, auc = evaluate_model(model, X_test, y_test)
print(f"Baseline F1 score  : {f1:.4f}")
print(f"Baseline AUC-ROC   : {auc:.4f}")

baseline_metrics = {"batch": 1, "f1": f1, "auc_roc": auc}
with open(f"{RESULTS_DIR}/baseline_metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=4)
print(f"Saved: {RESULTS_DIR}/baseline_metrics.json")

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: compute SHAP values using TreeExplainer
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Computing SHAP values using TreeExplainer ...")
print("=" * 60)

print("Initialising TreeExplainer ...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values on training data ...")
print("This may take a few minutes ...")

feature_importance, shap_array_2d, mean_abs_shap = compute_shap_rankings(
    explainer, X_train, list(X_train.columns)
)

print(f"SHAP array (rows, features) shape: {shap_array_2d.shape}")
print("SHAP values computed successfully.")
print("Top 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: save the baseline rankings and model
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Saving baseline rankings and model ...")
print("=" * 60)

feature_importance.to_csv(f"{RESULTS_DIR}/baseline_feature_rankings.csv", index=False)
print(f"Saved: {RESULTS_DIR}/baseline_feature_rankings.csv")

with open(f"{MODELS_DIR}/baseline_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"Saved: {MODELS_DIR}/baseline_model.pkl")

with open(f"{MODELS_DIR}/baseline_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)
print(f"Saved: {MODELS_DIR}/baseline_explainer.pkl")

feature_names = list(X_train.columns)
with open(f"{RESULTS_DIR}/feature_names.json", "w") as f:
    json.dump(feature_names, f)
print(f"Saved: {RESULTS_DIR}/feature_names.json")

print("\n" + "=" * 60)
print("Baseline training complete.")
print("=" * 60)
print("\nFiles saved:")
print(f"  {RESULTS_DIR}/baseline_feature_rankings.csv")
print(f"  {RESULTS_DIR}/baseline_metrics.json")
print(f"  {RESULTS_DIR}/feature_names.json")
print(f"  {MODELS_DIR}/baseline_model.pkl")
print(f"  {MODELS_DIR}/baseline_explainer.pkl")