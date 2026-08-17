'''
    07_threshold_sweep.py  (CIC-IDS2017)

    Purpose: Check whether ESS Triggered's failure to fire at
    tau=0.97 (see experiment_results.csv) is specific to that one
    threshold value, or whether ESS simply does not move enough on
    this dataset for ANY reasonable threshold to catch the batch 2/6/7
    collapses. Reruns ESS Triggered at several tau values and reports
    retrain count and final F1 for each, exactly mirroring the
    original 07_threshold_sweep.py's purpose for DDoS2019.

    This reuses run_experiment() from 05_experiment_main.py directly,
    no separate copy of the experiment loop, so this stays consistent
    with any future fix to the core experiment logic automatically.

    What this file does:
    - Runs the full 4-condition experiment once per ESS threshold in
      SWEEP_THRESHOLDS (Accuracy Triggered, PSI Triggered and No
      Retraining are identical across runs, since their trigger
      doesn't depend on tau; only ESS Triggered actually changes)
    - Extracts, for each tau: number of retrains, final-batch F1
      (forward-generalisation), and mean ESS across the run
    - Saves a summary table and prints it
'''

import os
import pandas as pd
from importlib import import_module

# "05_experiment_main" is not a valid bare identifier (starts with a
# digit), so it is imported via importlib rather than a normal import
# statement, same reason as the original 07_threshold_sweep.py.
main_module = import_module("05_experiment_main")
run_experiment = main_module.run_experiment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "results"))

SWEEP_THRESHOLDS = [0.90, 0.95, 0.97, 0.99]

print("=" * 60)
print(f"STEP 1: Running ESS Triggered at tau = {SWEEP_THRESHOLDS} ...")
print("=" * 60)

sweep_rows = []

for tau in SWEEP_THRESHOLDS:
    print(f"\n--- tau = {tau} ---")
    df = run_experiment(ess_threshold=tau, verbose=False)
    ess_df = df[df["condition"] == "ESS Triggered"].sort_values("batch")

    n_retrains = int(ess_df["retrained"].sum())
    final_row = ess_df.iloc[-1]
    final_f1 = final_row["f1_after_retrain"] if final_row["retrained"] else final_row["f1_on_arrival"]
    mean_ess = round(float(ess_df["ess"].mean()), 4)

    print(f"  Retrains: {n_retrains} | Final batch F1: {final_f1:.4f} | Mean ESS: {mean_ess:.4f}")

    sweep_rows.append({
        "ess_threshold": tau,
        "n_retrains": n_retrains,
        "final_batch_f1": round(float(final_f1), 4),
        "mean_ess": mean_ess,
        "n_batches": len(ess_df),
    })

sweep_df = pd.DataFrame(sweep_rows)

print("\n" + "=" * 60)
print("STEP 2: Threshold sensitivity summary")
print("=" * 60)
print(sweep_df.to_string(index=False))

# also fetch Accuracy Triggered's retrain count / final F1 once, as a
# reference point -- it does not depend on tau, so one run is enough
print("\nFor reference, Accuracy Triggered (tau-independent) from the tau=0.97 run:")
acc_df = run_experiment(ess_threshold=0.97, verbose=False)
acc_only = acc_df[acc_df["condition"] == "Accuracy Triggered"].sort_values("batch")
acc_retrains = int(acc_only["retrained"].sum())
acc_final = acc_only.iloc[-1]
acc_final_f1 = acc_final["f1_after_retrain"] if acc_final["retrained"] else acc_final["f1_on_arrival"]
print(f"  Retrains: {acc_retrains} | Final batch F1: {acc_final_f1:.4f}")

sweep_df.to_csv(os.path.join(RESULTS_DIR, "threshold_sweep.csv"), index=False)
print(f"\nSaved: {RESULTS_DIR}/threshold_sweep.csv")

print("\n" + "=" * 60)
print("Interpretation guide")
print("=" * 60)
print("- If n_retrains stays 0 even at tau=0.99 (the loosest possible reading,")
print("  since a HIGHER tau makes ESS Triggered fire MORE easily), that confirms")
print("  ESS genuinely does not move enough on this dataset's batch 2/6/7")
print("  collapses to cross any reasonable threshold, a structural finding,")
print("  not a tau=0.97-specific calibration issue.")
print("- If retrains appear at tau=0.99 but not at 0.97 or below, that indicates")
print("  ESS IS moving, just less than on DDoS2019, and a dataset-specific")
print("  threshold would be needed to catch it, itself a legitimate, reportable")
print("  limitation on how directly the original threshold transfers.")
print("- Report whichever pattern actually appears. Do not retune tau after")
print("  seeing this to make the headline comparison look better.")