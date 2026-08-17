'''
    07_threshold_sweep.py

    Purpose: Address the "is 0.97 cherry-picked?" question directly by
    rerunning the ESS Triggered condition at several tau values and
    reporting retrain count and final F1 for each. If ESS-triggered
    retraining performs well across a range of thresholds (not just the
    one chosen for the main experiment), that is evidence the result
    isn't an artefact of threshold tuning.

    This reuses run_experiment() from 05_experiment_main.py directly --
    no separate copy of the experiment loop -- so any future fix to the
    core experiment logic automatically applies here too.

    What this file does:
    - Runs the full 4-condition experiment once per ESS threshold in
      SWEEP_THRESHOLDS (Accuracy Triggered, PSI Triggered and No
      Retraining are identical across runs since their trigger doesn't
      depend on tau; only ESS Triggered actually changes)
    - Extracts, for each tau: number of retrains, final-batch F1
      (forward-generalization), and mean ESS across the run
    - Saves a summary table and prints it
'''

import pandas as pd
from importlib import import_module

# "05_experiment_main" isn't a valid bare identifier (starts with a
# digit), so it's imported via importlib rather than a normal import
# statement. This works because Python puts the running script's own
# folder (src/) at the front of sys.path automatically, same as how
# ess_utils imports already work in the other scripts.
main_module = import_module("05_experiment_main")
run_experiment = main_module.run_experiment

RESULTS_DIR = "outputs/results"

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

sweep_df.to_csv(f"{RESULTS_DIR}/threshold_sweep.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/threshold_sweep.csv")

print("\n" + "=" * 60)
print("Interpretation guide (fill in once real numbers are in):")
print("=" * 60)
print("- If final_batch_f1 stays high (close to Accuracy Triggered's) across")
print("  MOST of the tau range, that supports 'ESS-triggered retraining works")
print("  across a reasonable range of thresholds, not just 0.97'.")
print("- If retrain count rises sharply as tau increases (tighter threshold =")
print("  more retrains), report that trade-off explicitly rather than picking")
print("  the tau that looks best after the fact.")
print("- Do not silently switch your headline tau to whichever value in this")
print("  sweep looks strongest -- report the sweep as a robustness check on")
print("  the tau you already justified in Chapter 3 (0.97), not as a search")
print("  for a better one.")
