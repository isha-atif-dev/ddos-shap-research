'''
    03_train_baseline.py  (CIC-IDS2017)

    Purpose: Train the XGBoost classifier on Batch 1 (Monday + Tuesday,
    the baseline period) and compute the baseline SHAP feature
    importance rankings, mirroring the original 03_train_baseline.py.

    SHAP SAMPLE SIZE: 30,000 ROWS, NOT THE FULL TRAINING SPLIT
    -----------------------------------------------------
    CIC-IDS2017's baseline batch is 975,827 rows, roughly 28x larger
    than DDoS2019's 35,000-row baseline. XGBoost trains on the full
    split with no changes needed. SHAP is computed from a 30,000-row
    sample of that split instead, for three reasons, not one:

    1. ESS depends on the RELATIVE ORDER of features, not on pinning
       down each one's exact mean |SHAP| value to several decimal
       places. Sampling error only threatens a ranking when two
       features are nearly tied; this dataset's baseline results show
       the same pattern DDoS2019 did, one or two features dominating
       by a wide margin, well outside the regime a modest sample could
       disturb.
    2. This is not a new, untested choice invented for this
       experiment. It is the exact same sample size already used in
       12_synthetic_drift_injection.py and 13_shap_concentration_check.py,
       both already part of the defended dissertation. Reusing it here
       is consistency with established precedent.
    3. By the standards of the SHAP literature itself, 30,000 rows for
       an aggregate importance ranking is generous, published
       TreeExplainer studies routinely use samples in the hundreds to
       low thousands for this exact purpose.

    What this file does:
    - Loads Batch 1 features and labels saved by 02_preprocessing.py
    - Splits into train and test (80/20, stratified where possible)
    - Trains an XGBoost classifier on the full training split
    - Evaluates the model on the held-out test split
    - Computes SHAP values using TreeExplainer on a 30,000-row sample
      of the training split
    - Saves the baseline feature importance rankings, the trained
      model, the explainer, and evaluation metrics
'''

import os
import json
import pickle
import shap

from ess_utils_cicids2017 import (
    train_xgb, evaluate_model, compute_shap_rankings,
    stratified_or_plain_split, DEFAULT_SEED,
)

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "results"))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "models"))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SHAP_SAMPLE_SIZE = 30_000


def main():
    # ── STEP 1: load Batch 1 (baseline period) ──
    print("=" * 60)
    print("STEP 1: Loading Batch 1 (baseline period: Monday + Tuesday) ...")
    print("=" * 60)

    X_batch1 = pd.read_csv(os.path.join(RESULTS_DIR, "batch_1_X.csv"))
    y_batch1 = pd.read_csv(os.path.join(RESULTS_DIR, "batch_1_y_multi.csv")).squeeze()

    print(f"Batch 1 features shape : {X_batch1.shape}")
    print(f"Batch 1 labels shape   : {y_batch1.shape}")
    print(f"Label distribution:\n{y_batch1.value_counts().sort_index()}")
    print(f"Classes present in Batch 1: {sorted(y_batch1.unique())}")

    # ── STEP 2: split into train and test ──
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Batch 1 into train and test ...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = stratified_or_plain_split(
        X_batch1, y_batch1, test_size=0.2, random_state=DEFAULT_SEED
    )

    print(f"Train size : {X_train.shape}")
    print(f"Test size  : {X_test.shape}")

    # ── STEP 3: train XGBoost on the FULL training split ──
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBoost classifier (num_class=15, global labels) ...")
    print("=" * 60)

    model = train_xgb(X_train, y_train)
    print("XGBoost model trained successfully on the full training split.")

    # ── STEP 4: evaluate on the held-out test set ──
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating model on test set ...")
    print("=" * 60)

    f1, auc = evaluate_model(model, X_test, y_test)
    print(f"Baseline F1 score  : {f1:.4f}")
    print(f"Baseline AUC-ROC   : {auc:.4f}")

    baseline_metrics = {"batch": 1, "f1": f1, "auc_roc": auc}
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"Saved: {RESULTS_DIR}/baseline_metrics.json")

    # ── STEP 4B: verify a perfect/near-perfect F1 directly ──
    # A weighted F1 of exactly 1.0000 is unusual enough to confirm
    # directly rather than accept on faith. This prints the actual
    # per-class breakdown so we can see whether errors are genuinely
    # zero (plausible for FTP-Patator/SSH-Patator, well documented as
    # among the most separable classes in this dataset) or whether
    # something else is going on that the aggregate F1 is hiding.
    if f1 >= 0.999:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n" + "=" * 60)
        print("STEP 4B: F1 is at/near 1.0000, verifying with a confusion matrix ...")
        print("=" * 60)
        from ess_utils_cicids2017 import predict_labels
        y_pred = predict_labels(model, X_test)
        present_labels = sorted(y_test.unique())
        print("Confusion matrix (rows=true, cols=predicted), present classes only:")
        print(confusion_matrix(y_test, y_pred, labels=present_labels))
        print("\nPer-class precision/recall/F1:")
        print(classification_report(y_test, y_pred, labels=present_labels, zero_division=0))

    # ── STEP 5: compute SHAP values on a sample of the training split ──
    print("\n" + "=" * 60)
    print(f"STEP 5: Computing SHAP values (TreeExplainer, {SHAP_SAMPLE_SIZE}-row sample) ...")
    print("=" * 60)

    print("Initialising TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    shap_sample_size = min(SHAP_SAMPLE_SIZE, len(X_train))
    X_train_shap_sample = X_train.sample(n=shap_sample_size, random_state=DEFAULT_SEED)
    print(f"Computing SHAP values on a {shap_sample_size}-row sample of the "
          f"{len(X_train)}-row training split ...")
    print("This sample size matches the precedent already set in "
          "12_synthetic_drift_injection.py and 13_shap_concentration_check.py, "
          "well above what published TreeExplainer studies typically use "
          "for a stable aggregate importance ranking.")

    feature_importance, shap_array_2d, mean_abs_shap = compute_shap_rankings(
        explainer, X_train_shap_sample, list(X_train.columns)
    )

    print(f"SHAP array (rows, features) shape: {shap_array_2d.shape}")
    print("Top 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    # ── STEP 6: save rankings, model, explainer ──
    print("\n" + "=" * 60)
    print("STEP 6: Saving baseline rankings and model ...")
    print("=" * 60)

    feature_importance.to_csv(os.path.join(RESULTS_DIR, "baseline_feature_rankings.csv"), index=False)
    print(f"Saved: {RESULTS_DIR}/baseline_feature_rankings.csv")

    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {MODELS_DIR}/baseline_model.pkl")

    with open(os.path.join(MODELS_DIR, "baseline_explainer.pkl"), "wb") as f:
        pickle.dump(explainer, f)
    print(f"Saved: {MODELS_DIR}/baseline_explainer.pkl")

    feature_names = list(X_train.columns)
    with open(os.path.join(RESULTS_DIR, "feature_names.json"), "w") as f:
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


if __name__ == "__main__":
    main()