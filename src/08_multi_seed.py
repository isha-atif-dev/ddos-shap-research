'''
    08_multi_seed.py

    Purpose: Repeat the full 4-condition experiment across several
    random seeds (varying both the train/test splits and XGBoost's own
    randomness) and report mean +/- std for the metrics that matter,
    instead of relying on a single run. One run agreeing with your
    hypothesis could be luck; five runs agreeing is much harder to
    argue with.

    Reuses run_experiment() from 05_experiment_main.py -- same
    experiment logic, just called once per seed.

    NOTE ON RUNTIME: each seed re-runs all four conditions across every
    deployment batch, including a SHAP TreeExplainer pass per batch per
    condition. On the full CIC-DDoS2019 batches (tens of thousands of
    rows each) this can take a genuinely long time per seed on a laptop
    CPU. Start with SEEDS having 3 values; only extend to 5 once you've
    timed how long one seed takes on your machine.
'''

import pandas as pd
import numpy as np
from importlib import import_module

main_module = import_module("05_experiment_main")
run_experiment = main_module.run_experiment

RESULTS_DIR = "outputs/results"

SEEDS = [42, 123, 7]  # extend to 5 seeds once you've confirmed the runtime is acceptable

print("=" * 60)
print(f"STEP 1: Running full experiment across seeds = {SEEDS} ...")
print("=" * 60)

all_runs = []
for seed in SEEDS:
    print(f"\n--- seed = {seed} ---")
    df = run_experiment(seed=seed, verbose=False)
    all_runs.append(df)
    for condition in main_module.CONDITIONS:
        cond_df = df[df["condition"] == condition].sort_values("batch")
        n_retrains = int(cond_df["retrained"].sum())
        final_row = cond_df.iloc[-1]
        final_f1 = final_row["f1_after_retrain"] if final_row["retrained"] else final_row["f1_on_arrival"]
        print(f"  {condition:<20} retrains={n_retrains} final_f1={final_f1:.4f}")

combined = pd.concat(all_runs, ignore_index=True)
combined.to_csv(f"{RESULTS_DIR}/multi_seed_raw_results.csv", index=False)

print("\n" + "=" * 60)
print("STEP 2: Aggregating mean +/- std across seeds ...")
print("=" * 60)

summary_rows = []
for condition in main_module.CONDITIONS:
    final_f1s = []
    retrain_counts = []
    for df in all_runs:
        cond_df = df[df["condition"] == condition].sort_values("batch")
        n_retrains = int(cond_df["retrained"].sum())
        final_row = cond_df.iloc[-1]
        final_f1 = final_row["f1_after_retrain"] if final_row["retrained"] else final_row["f1_on_arrival"]
        final_f1s.append(final_f1)
        retrain_counts.append(n_retrains)

    summary_rows.append({
        "condition": condition,
        "final_f1_mean": round(float(np.mean(final_f1s)), 4),
        "final_f1_std": round(float(np.std(final_f1s)), 4),
        "retrains_mean": round(float(np.mean(retrain_counts)), 2),
        "retrains_std": round(float(np.std(retrain_counts)), 2),
        "n_seeds": len(SEEDS),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_df.to_csv(f"{RESULTS_DIR}/multi_seed_summary.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/multi_seed_raw_results.csv")
print(f"Saved: {RESULTS_DIR}/multi_seed_summary.csv")

print("\n" + "=" * 60)
print("Interpretation guide:")
print("=" * 60)
print("- A small final_f1_std for ESS Triggered (e.g. well under 0.05)")
print("  across seeds means the result is not a one-off lucky split.")
print("- Compare ESS Triggered's retrains_mean against Accuracy Triggered's")
print("  -- consistently fewer retrains across ALL seeds is a much stronger")
print("  claim than 'fewer retrains in the one run we happened to show'.")
print("- If results vary a lot seed to seed, report that honestly as a")
print("  limitation rather than picking the seed that looks best.")
