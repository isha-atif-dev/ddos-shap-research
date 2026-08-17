'''
    05_experiment_main.py  (UPDATED: forward-generalization + trivial
    baselines + reusable function + configurable batch count)

    NEW IN THIS VERSION
    --------------------
    1. Forward-generalization split: F1/AUC are now reported in TWO
       separate columns instead of one that got silently overwritten:
         - f1_on_arrival / auc_on_arrival: the CURRENT model (as it
           stood coming INTO this batch, i.e. possibly retrained at the
           end of a PREVIOUS batch, but never on this batch's own data)
           evaluated on this batch's held-out test split. This is the
           genuine forward-generalization number -- "how does the
           model that was deployed do on data it has truly never seen".
         - f1_after_retrain / auc_after_retrain: only populated if a
           retrain was triggered THIS batch. This is the retrained
           model evaluated on a held-out split of the SAME batch it
           was partly trained on -- useful to confirm the retrain
           worked, but it is NOT a forward-generalization number and
           should not be quoted as one. Previously these two numbers
           were both written into a single "f1" column, so the second
           one silently overwrote the first -- this was a legitimate
           criticism of the earlier version and is now fixed.
    2. Trivial baselines: majority_class_f1 and random_f1 are computed
       per batch (independent of retraining condition) so headline
       F1 numbers have a floor to be compared against.
    3. DEPLOYMENT_BATCHES now comes from ess_utils (10 batches instead
       of 6), and the whole experiment is wrapped in run_experiment()
       so 07_threshold_sweep.py and a future multi-seed script can call
       it directly with different parameters instead of duplicating it.
'''

import pandas as pd
import numpy as np
import os
import json
import pickle
import shap

from ess_utils import (
    train_xgb, evaluate_model, compute_shap_rankings,
    compute_ess, compute_psi, stratified_or_plain_split,
    majority_class_f1, random_baseline_f1,
    DEPLOYMENT_BATCHES, DEFAULT_SEED,
)

RESULTS_DIR = "outputs/results"
MODELS_DIR = "outputs/models"

CONDITIONS = ["No Retraining", "ESS Triggered", "Accuracy Triggered", "PSI Triggered"]


def load_batch(batch_num, feature_names):
    X = pd.read_csv(f"{RESULTS_DIR}/batch_{batch_num}_X.csv")[feature_names]
    y = pd.read_csv(f"{RESULTS_DIR}/batch_{batch_num}_y_multi.csv").squeeze()
    return X, y


def run_experiment(
    seed=DEFAULT_SEED,
    ess_threshold=0.97,
    f1_threshold=0.75,
    psi_threshold=0.1,
    deployment_batches=None,
    verbose=True,
):
    '''
    Runs all four retraining conditions across the deployment batches
    and returns a results DataFrame. Parametrized by seed and the three
    trigger thresholds so this same function powers:
      - the default single run (this script's __main__ block)
      - 07_threshold_sweep.py (varies ess_threshold, fixed seed)
      - a future multi-seed script (varies seed, fixed thresholds)
    '''
    deployment_batches = deployment_batches or DEPLOYMENT_BATCHES

    def log(msg):
        if verbose:
            print(msg)

    # ── load baseline artifacts ──────────────────────────────────────
    with open(f"{RESULTS_DIR}/feature_names.json", "r") as f:
        feature_names = json.load(f)
    baseline_rankings = pd.read_csv(f"{RESULTS_DIR}/baseline_feature_rankings.csv")
    with open(f"{MODELS_DIR}/baseline_model.pkl", "rb") as f:
        baseline_model = pickle.load(f)
    with open(f"{MODELS_DIR}/baseline_explainer.pkl", "rb") as f:
        baseline_explainer = pickle.load(f)

    baseline_aligned = baseline_rankings.set_index("feature").reindex(feature_names)
    baseline_rank_vector = baseline_aligned["rank"].values

    X_batch1, y_batch1 = load_batch(1, feature_names)
    X_b1_train, X_b1_test, y_b1_train, y_b1_test = stratified_or_plain_split(
        X_batch1, y_batch1, random_state=seed
    )
    _, baseline_shap_array, _ = compute_shap_rankings(baseline_explainer, X_b1_train, feature_names)

    # ── pre-compute per-batch splits, cumulative pools, trivial baselines ──
    batch_splits = {}
    cumulative_X = [X_b1_train]
    cumulative_y = [y_b1_train]
    cumulative_pool_at_batch = {}
    trivial_baselines = {}

    for batch_num in deployment_batches:
        X_batch, y_batch = load_batch(batch_num, feature_names)
        X_train, X_test, y_train, y_test = stratified_or_plain_split(
            X_batch, y_batch, random_state=seed
        )
        batch_splits[batch_num] = {
            "X_batch": X_batch, "y_batch": y_batch,
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
        }
        cumulative_X.append(X_train)
        cumulative_y.append(y_train)
        cumulative_pool_at_batch[batch_num] = (
            pd.concat(cumulative_X, ignore_index=True),
            pd.concat(cumulative_y, ignore_index=True),
        )
        trivial_baselines[batch_num] = {
            "majority_class_f1": majority_class_f1(y_train, y_test),
            "random_f1": random_baseline_f1(y_test, seed=seed),
        }

    # ── run one condition across all deployment batches ─────────────
    def run_condition(condition_name):
        log(f"\n{'=' * 60}\nRunning condition: {condition_name}\n{'=' * 60}")

        current_model = baseline_model
        current_explainer = baseline_explainer
        ref_rank_vector = baseline_rank_vector
        ref_shap_array = baseline_shap_array
        results = []

        for batch_num in deployment_batches:
            sp = batch_splits[batch_num]
            X_batch, X_test, y_test = sp["X_batch"], sp["X_test"], sp["y_test"]

            batch_ranking_df, batch_shap_array, _ = compute_shap_rankings(
                current_explainer, X_batch, feature_names
            )
            batch_aligned = batch_ranking_df.set_index("feature").reindex(feature_names)
            batch_rank_vector = batch_aligned["rank"].values

            ess, pvalue = compute_ess(ref_rank_vector, batch_rank_vector)
            psi = compute_psi(ref_shap_array, batch_shap_array)

            # forward-generalization: current model (inherited from a
            # PREVIOUS batch's retrain, or still baseline) on this
            # batch's held-out test split -- genuinely unseen data
            f1_on_arrival, auc_on_arrival = evaluate_model(current_model, X_test, y_test)

            retrained = False
            trigger_reason = None
            f1_after_retrain, auc_after_retrain = None, None

            if condition_name == "ESS Triggered" and ess < ess_threshold:
                trigger_reason = f"ESS {ess:.4f} < {ess_threshold}"
            elif condition_name == "Accuracy Triggered" and f1_on_arrival < f1_threshold:
                trigger_reason = f"F1 {f1_on_arrival:.4f} < {f1_threshold}"
            elif condition_name == "PSI Triggered" and psi > psi_threshold:
                trigger_reason = f"PSI {psi:.4f} > {psi_threshold}"

            if trigger_reason is not None:
                log(f"  Batch {batch_num}: trigger fired ({trigger_reason}), retraining ...")
                X_pool, y_pool = cumulative_pool_at_batch[batch_num]
                current_model = train_xgb(X_pool, y_pool, seed=seed)
                current_explainer = shap.TreeExplainer(current_model)
                # same-batch validation ONLY -- not a forward-generalization number
                f1_after_retrain, auc_after_retrain = evaluate_model(current_model, X_test, y_test)
                retrained = True

                new_ref_df, ref_shap_array, _ = compute_shap_rankings(
                    current_explainer, X_pool, feature_names
                )
                new_ref_aligned = new_ref_df.set_index("feature").reindex(feature_names)
                ref_rank_vector = new_ref_aligned["rank"].values

            log(f"  Batch {batch_num}: ESS={ess:.4f} | F1_on_arrival={f1_on_arrival:.4f} | "
                f"F1_after_retrain={f1_after_retrain} | PSI={psi:.4f} | Retrained={retrained}")

            results.append({
                "condition": condition_name,
                "batch": batch_num,
                "seed": seed,
                "ess": round(ess, 4),
                "f1_on_arrival": f1_on_arrival,
                "auc_on_arrival": auc_on_arrival,
                "f1_after_retrain": f1_after_retrain,
                "auc_after_retrain": auc_after_retrain,
                "psi": round(psi, 4),
                "pvalue": round(pvalue, 4),
                "retrained": retrained,
                "majority_class_f1": trivial_baselines[batch_num]["majority_class_f1"],
                "random_f1": trivial_baselines[batch_num]["random_f1"],
            })

        return results

    all_results = []
    for condition in CONDITIONS:
        all_results += run_condition(condition)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("=" * 60)
    print(f"Running experiment: seed={DEFAULT_SEED}, "
          f"batches={DEPLOYMENT_BATCHES}")
    print("=" * 60)

    results_df = run_experiment(seed=DEFAULT_SEED, verbose=True)
    results_df.to_csv(f"{RESULTS_DIR}/experiment_results.csv", index=False)

    print("\nFull Results Summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR}/experiment_results.csv")
    print("\nNOTE: 'f1' is now split into f1_on_arrival (forward-generalization,")
    print("the number to headline) and f1_after_retrain (same-batch validation")
    print("only, populated when retrained=True). majority_class_f1 and random_f1")
    print("are trivial baselines for the same batch/test-split.")
