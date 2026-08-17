'''
    05_experiment_main.py  (CIC-IDS2017)

    Purpose: Run all four retraining conditions, no retraining, ESS
    triggered, accuracy triggered, PSI triggered, across the six
    CIC-IDS2017 deployment batches, mirroring the original
    05_experiment_main.py exactly in logic and structure.

    THRESHOLDS ARE DELIBERATELY NOT RETUNED FOR THIS DATASET
    -----------------------------------------------------------
    ess_threshold=0.97, f1_threshold=0.75 and psi_threshold=0.1 are
    the same values used throughout the original dissertation. This
    is the stronger test of generalisability: if the original
    thresholds still produce sensible retraining behaviour on a
    dataset the method was never tuned against, that is real evidence
    the approach generalises. Retuning them here would answer a
    different, weaker question, whether SOME threshold works on this
    dataset, not whether the original one does.

    FORWARD-GENERALISATION VS SAME-BATCH VALIDATION
    -----------------------------------------------------------
    Identical distinction to the original: f1_on_arrival is the
    CURRENT model, possibly retrained at the end of a previous batch
    but never on this batch's own data, evaluated on this batch's
    held-out test split. This is the genuine forward-generalisation
    number. f1_after_retrain is only populated when a retrain fires
    this batch, and is evaluated on a held-out split of the SAME batch
    the model was partly trained on, useful to confirm the retrain
    worked, but not a forward-generalisation number.

    What this file does:
    - Loads the baseline artifacts saved by 03_train_baseline.py
    - For each of the four conditions, walks through the six
      deployment batches, computing ESS, PSI, and forward-
      generalisation F1 at each step, retraining on the cumulative
      pool when that condition's trigger fires
    - Records trivial baselines (majority-class F1, random F1) per
      batch, independent of condition
    - Saves the full results table to
      outputs_cicids2017/results/experiment_results.csv
'''

import os
import json
import pickle
import shap
import pandas as pd

from ess_utils_cicids2017 import (
    train_xgb, evaluate_model, compute_shap_rankings,
    compute_ess, compute_psi, stratified_or_plain_split,
    majority_class_f1, random_baseline_f1,
    DEPLOYMENT_BATCHES, DEFAULT_SEED,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "results"))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "models"))

CONDITIONS = ["No Retraining", "ESS Triggered", "Accuracy Triggered", "PSI Triggered"]

# CIC-IDS2017 batches (170k-693k rows) and cumulative training pools
# (over a million rows by the later batches) are far larger than
# DDoS2019's ~35k-row batches, which the original script's full-batch
# SHAP computation was sized around. Sampling down to 30,000 rows here
# uses the same justification already established for the baseline
# script: ESS depends on relative ranking, not exact magnitude, a
# 30,000-row sample is stable for that purpose, and this matches the
# precedent already set in 12_synthetic_drift_injection.py and
# 13_shap_concentration_check.py. Without this, the retraining pool
# alone would make later batches computationally impractical.
SHAP_SAMPLE_SIZE = 30_000


def sample_for_shap(X, sample_size=SHAP_SAMPLE_SIZE, seed=DEFAULT_SEED):
    '''Returns X unchanged if it's already at or under sample_size,
    otherwise a random sample of that size. Keeps every SHAP call in
    this script bounded regardless of how large a batch or cumulative
    pool has grown.'''
    if len(X) <= sample_size:
        return X
    return X.sample(n=sample_size, random_state=seed)


def load_batch(batch_num, feature_names):
    X = pd.read_csv(os.path.join(RESULTS_DIR, f"batch_{batch_num}_X.csv"))[feature_names]
    y = pd.read_csv(os.path.join(RESULTS_DIR, f"batch_{batch_num}_y_multi.csv")).squeeze()
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
    and returns a results DataFrame. Same signature and same default
    thresholds as the original run_experiment(), so this same function
    can later power a threshold sweep or multi-seed script for
    CIC-IDS2017 exactly the way 07 and 08 already do for DDoS2019,
    without duplicating this logic.
    '''
    deployment_batches = deployment_batches or DEPLOYMENT_BATCHES

    def log(msg):
        if verbose:
            print(msg)

    # ── load baseline artifacts ──────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    baseline_rankings = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_feature_rankings.csv"))
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "rb") as f:
        baseline_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "baseline_explainer.pkl"), "rb") as f:
        baseline_explainer = pickle.load(f)

    baseline_aligned = baseline_rankings.set_index("feature").reindex(feature_names)
    baseline_rank_vector = baseline_aligned["rank"].values

    X_batch1, y_batch1 = load_batch(1, feature_names)
    X_b1_train, X_b1_test, y_b1_train, y_b1_test = stratified_or_plain_split(
        X_batch1, y_batch1, random_state=seed
    )
    _, baseline_shap_array, _ = compute_shap_rankings(
        baseline_explainer, sample_for_shap(X_b1_train, seed=seed), feature_names
    )

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
                current_explainer, sample_for_shap(X_batch, seed=seed), feature_names
            )
            batch_aligned = batch_ranking_df.set_index("feature").reindex(feature_names)
            batch_rank_vector = batch_aligned["rank"].values

            ess, pvalue = compute_ess(ref_rank_vector, batch_rank_vector)
            psi = compute_psi(ref_shap_array, batch_shap_array)

            # forward-generalisation: current model (inherited from a
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
                # same-batch validation ONLY -- not a forward-generalisation number
                f1_after_retrain, auc_after_retrain = evaluate_model(current_model, X_test, y_test)
                retrained = True

                new_ref_df, ref_shap_array, _ = compute_shap_rankings(
                    current_explainer, sample_for_shap(X_pool, seed=seed), feature_names
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
        condition_results = run_condition(condition)
        all_results += condition_results

        # Save progress after every condition, not just at the end, so an
        # interruption (accidental or not) does not lose everything already
        # completed. Same pattern already used in
        # 12_synthetic_drift_injection.py and 13_shap_concentration_check.py.
        if verbose:
            partial_path = os.path.join(RESULTS_DIR, "experiment_results_partial.csv")
            pd.DataFrame(all_results).to_csv(partial_path, index=False)
            log(f"\n(Progress saved: {len(all_results)} rows across "
                f"{len(set(r['condition'] for r in all_results))} of {len(CONDITIONS)} "
                f"conditions so far -> {partial_path})")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("=" * 60)
    print(f"Running CIC-IDS2017 experiment: seed={DEFAULT_SEED}, "
          f"batches={DEPLOYMENT_BATCHES}")
    print("Thresholds: ess=0.97, f1=0.75, psi=0.1 (same as the original dissertation, not retuned)")
    print("=" * 60)

    results_df = run_experiment(seed=DEFAULT_SEED, verbose=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"), index=False)

    print("\nFull Results Summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR}/experiment_results.csv")
    print("\nNOTE: f1_on_arrival is the forward-generalisation number, the one to")
    print("headline. f1_after_retrain is same-batch validation only, populated")
    print("when retrained=True. majority_class_f1 and random_f1 are trivial")
    print("baselines for the same batch/test-split.")