'''
    06_independence_experiment.py

    Purpose: Formal controlled analysis demonstrating that explanation
    drift, accuracy degradation and distribution shift are independent
    signals in the deployed DDoS detection model. Directly addresses
    RQ1 and RQ3 of the dissertation.

    What this file does:
    - Loads the experiment results from 05_experiment_main.py
    - Formally analyses the independence between ESS, F1 and PSI
      across all deployment batches under the no retraining condition
    - Computes correlation between each pair of signals to prove
      they move independently
    - Identifies the key finding that PSI remained stable while
      accuracy collapsed, and ESS remained stable throughout
    - Saves formal independence analysis results
    - Directly serves Objective 3, RQ1 and RQ3 of the dissertation
'''

import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr, wilcoxon

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

RESULTS_DIR   = "outputs/results"
ESS_THRESHOLD = 0.97
F1_THRESHOLD  = 0.75
PSI_THRESHOLD = 0.1

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load experiment results from script 05
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading experiment results ...")
print("=" * 60)

results_df = pd.read_csv(f"{RESULTS_DIR}/experiment_results.csv")

# extract only the no retraining condition
no_retrain = results_df[results_df["condition"] == "No Retraining"].copy()
no_retrain = no_retrain.reset_index(drop=True)

print("No Retraining condition results:")
print(no_retrain[["batch", "ess", "f1_on_arrival", "auc_on_arrival", "psi"]].to_string(index=False))
print("\nNOTE: f1_on_arrival is the forward-generalization number (model as it\n"
      "stood entering this batch, evaluated on this batch's held-out test\n"
      "split). Under No Retraining the model never changes, so this is simply\n"
      "the fixed baseline model evaluated fresh on each new batch.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: analyse independence between ESS, F1 and PSI
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Analysing independence between signals ...")
print("=" * 60)

ess_values = no_retrain["ess"].values
f1_values  = no_retrain["f1_on_arrival"].values
psi_values = no_retrain["psi"].values

# compute correlations between each pair of signals
ess_f1_spearman,  ess_f1_p  = spearmanr(ess_values, f1_values)
ess_psi_spearman, ess_psi_p = spearmanr(ess_values, psi_values)
f1_psi_spearman,  f1_psi_p  = spearmanr(f1_values,  psi_values)

ess_f1_pearson,  _  = pearsonr(ess_values, f1_values)
ess_psi_pearson, _  = pearsonr(ess_values, psi_values)
f1_psi_pearson,  _  = pearsonr(f1_values,  psi_values)

print(f"\n{'Signal Pair':<30} {'Spearman':>10} {'P-Value':>10} {'Pearson':>10}")
print("-" * 62)
print(f"{'ESS vs F1':<30} {ess_f1_spearman:>10.4f} "
      f"{ess_f1_p:>10.4f} {ess_f1_pearson:>10.4f}")
print(f"{'ESS vs PSI':<30} {ess_psi_spearman:>10.4f} "
      f"{ess_psi_p:>10.4f} {ess_psi_pearson:>10.4f}")
print(f"{'F1 vs PSI':<30} {f1_psi_spearman:>10.4f} "
      f"{f1_psi_p:>10.4f} {f1_psi_pearson:>10.4f}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: compute rate of change for each signal
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing rate of change per signal ...")
print("=" * 60)

def safe_pct_change(change, start_value):
    '''
    Percentage change is undefined when the starting value is exactly
    zero (division by zero). Returns None in that case instead of NaN
    or inf, so downstream printing can say "undefined" rather than
    silently printing "nan%" -- this can genuinely happen (e.g. PSI or
    F1 legitimately starting at 0.0 in Batch 2), not just in synthetic
    test data.
    '''
    if start_value == 0:
        return None
    return (change / start_value) * 100


def format_pct(pct):
    return f"{pct:.2f}%" if pct is not None else "undefined (started at 0)"


ess_change = ess_values[-1] - ess_values[0]
f1_change  = f1_values[-1]  - f1_values[0]
psi_change = psi_values[-1] - psi_values[0]

ess_pct = safe_pct_change(ess_change, ess_values[0])
f1_pct  = safe_pct_change(f1_change, f1_values[0])
psi_pct = safe_pct_change(psi_change, psi_values[0])

first_batch = int(no_retrain["batch"].iloc[0])
last_batch = int(no_retrain["batch"].iloc[-1])

print(f"\n{'Signal':<10} {f'Batch {first_batch}':>10} {f'Batch {last_batch}':>10} "
      f"{'Change':>10} {'Pct Change':>12}")
print("-" * 55)
print(f"{'ESS':<10} {ess_values[0]:>10.4f} {ess_values[-1]:>10.4f} "
      f"{ess_change:>10.4f} {format_pct(ess_pct):>18}")
print(f"{'F1':<10} {f1_values[0]:>10.4f} {f1_values[-1]:>10.4f} "
      f"{f1_change:>10.4f} {format_pct(f1_pct):>18}")
print(f"{'PSI':<10} {psi_values[0]:>10.4f} {psi_values[-1]:>10.4f} "
      f"{psi_change:>10.4f} {format_pct(psi_pct):>18}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: check which signals triggered across batches
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Checking which signals triggered across batches ...")
print("=" * 60)

print(f"\n{'Batch':<8} {'ESS':>8} {'ESS Flag':>10} "
      f"{'F1':>8} {'F1 Flag':>10} {'PSI':>8} {'PSI Flag':>10}")
print("-" * 65)

trigger_summary = []
for _, row in no_retrain.iterrows():
    ess_flag = row["ess"] < ESS_THRESHOLD
    f1_flag  = row["f1_on_arrival"] < F1_THRESHOLD
    psi_flag = row["psi"] > PSI_THRESHOLD

    print(f"{int(row['batch']):<8} {row['ess']:>8.4f} {str(ess_flag):>10} "
          f"{row['f1_on_arrival']:>8.4f} {str(f1_flag):>10} "
          f"{row['psi']:>8.4f} {str(psi_flag):>10}")

    trigger_summary.append({
        "batch"          : int(row["batch"]),
        "ess"            : row["ess"],
        "f1"             : row["f1_on_arrival"],
        "psi"            : row["psi"],
        "ess_triggered"  : ess_flag,
        "f1_triggered"   : f1_flag,
        "psi_triggered"  : psi_flag,
    })

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: formal independence conclusion
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Formal independence conclusion ...")
print("=" * 60)

ess_triggered_count = sum(1 for r in trigger_summary if r["ess_triggered"])
f1_triggered_count  = sum(1 for r in trigger_summary if r["f1_triggered"])
psi_triggered_count = sum(1 for r in trigger_summary if r["psi_triggered"])

print(f"\nOut of {len(trigger_summary)} deployment batches:")
print(f"  ESS triggered : {ess_triggered_count} times")
print(f"  F1  triggered : {f1_triggered_count} times")
print(f"  PSI triggered : {psi_triggered_count} times")

print(f"\nF1 total decline : {format_pct(f1_pct)} from Batch {first_batch} to Batch {last_batch}")
print(f"PSI total change : {format_pct(psi_pct)} from Batch {first_batch} to Batch {last_batch}")
print(f"ESS total change : {format_pct(ess_pct)} from Batch {first_batch} to Batch {last_batch}")

print("\nKey independence findings:")
print(f"  1. F1 changed by {format_pct(f1_pct)} (Batch {first_batch} to Batch {last_batch}) while PSI crossed "
      f"its threshold of {PSI_THRESHOLD} on {psi_triggered_count} of "
      f"{len(trigger_summary)} batches.")
if f1_triggered_count > 0 and psi_triggered_count < f1_triggered_count:
    print("     Accuracy degraded on more batches than the distribution "
          "signal flagged, consistent with accuracy and distribution "
          "carrying non-redundant information.")
elif f1_triggered_count == 0 and psi_triggered_count == 0:
    print("     Neither signal crossed its threshold in this run; no "
          "evidence of decoupling can be drawn from trigger counts alone "
          "(see the correlation coefficients above instead).")
else:
    print("     Accuracy and distribution flags moved together in this run; "
          "this particular batch sequence does not on its own demonstrate "
          "decoupling between the two.")

print(f"  2. ESS crossed its threshold of {ESS_THRESHOLD} on {ess_triggered_count} "
      f"of {len(trigger_summary)} batches, while F1 changed by {format_pct(f1_pct)} "
      f"over the same period.")
if ess_triggered_count != f1_triggered_count:
    print("     ESS and F1 crossed their respective thresholds a different "
          "number of times, consistent with explanation stability and "
          "predictive accuracy being at least partially decoupled signals.")
else:
    print("     ESS and F1 crossed their thresholds the same number of "
          "times in this run; this alone does not establish decoupling "
          "and should be read alongside the correlation coefficients above.")

print(f"  3. PSI crossed its threshold on {psi_triggered_count} of "
      f"{len(trigger_summary)} batches. "
      + ("PSI stayed silent despite the F1 changes observed, which would "
         "mean PSI alone is not a reliable stand-in for accuracy monitoring "
         "in this run." if psi_triggered_count == 0 and f1_triggered_count > 0
         else "PSI did register some of the same batches as F1 in this run, "
              "so this run alone does not support a strong claim that PSI "
              "fails to detect degradation here."))
print(f"  4. Spearman correlations: ESS vs F1 = {ess_f1_spearman:.3f} "
      f"(p={ess_f1_p:.3f}), ESS vs PSI = {ess_psi_spearman:.3f} "
      f"(p={ess_psi_p:.3f}), F1 vs PSI = {f1_psi_spearman:.3f} "
      f"(p={f1_psi_p:.3f}). With only {len(trigger_summary)} deployment "
      f"batches, these correlations should be read as indicative rather "
      f"than statistically conclusive.")

print("\nCONCLUSION:")
significance_alpha = 0.05
all_pvalues = {"ESS vs F1": ess_f1_p, "ESS vs PSI": ess_psi_p, "F1 vs PSI": f1_psi_p}
nan_pairs = [name for name, p in all_pvalues.items() if pd.isna(p)]

if nan_pairs:
    # a correlation is NaN when one of the two signals has zero variance
    # across every batch in this run (e.g. ESS never moved at all) --
    # Spearman/Pearson are undefined in that case, not "not significant"
    print(f"Correlation is undefined (NaN) for: {', '.join(nan_pairs)}. This "
          f"happens when one of the two signals had zero variance across "
          f"every batch in this run (it never changed), not because the "
          f"signals are unrelated. Check whether that signal is behaving "
          f"as expected before drawing any independence conclusion -- a "
          f"constant signal cannot demonstrate independence OR dependence.")

valid_pvalues = {name: p for name, p in all_pvalues.items() if not pd.isna(p)}
if valid_pvalues:
    sig_pairs = [name for name, p in valid_pvalues.items() if p < significance_alpha]
    if not sig_pairs:
        print(f"Of the pairs with a defined correlation "
              f"({', '.join(valid_pvalues.keys())}), none reached conventional "
              f"statistical significance (p >= 0.05), consistent with those "
              f"signals behaving independently over this deployment sequence. "
              f"Report as supportive evidence for RQ1/RQ3 given the small "
              f"batch count, not as definitive proof of independence.")
    else:
        print(f"The following pair(s) showed a statistically significant "
              f"correlation in this run: {', '.join(sig_pairs)}. Report this "
              f"honestly rather than as full support for independence -- the "
              f"correct framing is partial evidence for independence, with "
              f"the specific significant pair(s) discussed rather than omitted.")
elif not nan_pairs:
    print("No valid correlations were computed in this run.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: Wilcoxon signed-rank test, ESS Triggered vs Accuracy Triggered
# Paired by batch (same batches, same seed, only the retraining trigger
# differs), testing whether the two conditions' forward-generalization
# F1 distributions differ significantly across the deployment sequence.
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Wilcoxon signed-rank test (ESS Triggered vs Accuracy Triggered) ...")
print("=" * 60)

ess_cond = results_df[results_df["condition"] == "ESS Triggered"].sort_values("batch")
acc_cond = results_df[results_df["condition"] == "Accuracy Triggered"].sort_values("batch")

wilcoxon_result = None
if len(ess_cond) == len(acc_cond) and len(ess_cond) >= 6:
    ess_f1 = ess_cond["f1_on_arrival"].values
    acc_f1 = acc_cond["f1_on_arrival"].values
    diffs = ess_f1 - acc_f1
    if np.allclose(diffs, 0):
        print("ESS Triggered and Accuracy Triggered produced identical "
              "f1_on_arrival values on every batch -- Wilcoxon is undefined "
              "when all paired differences are zero. No test to report.")
    else:
        stat, p_value = wilcoxon(ess_f1, acc_f1)
        wilcoxon_result = {"statistic": float(stat), "p_value": float(p_value)}
        print(f"Wilcoxon statistic = {stat:.4f}, p = {p_value:.4f}")
        if p_value < 0.05:
            print(f"Statistically significant difference (p < 0.05) between "
                  f"ESS Triggered and Accuracy Triggered forward-generalization "
                  f"F1 across {len(ess_cond)} batches.")
        else:
            print(f"No statistically significant difference (p >= 0.05) between "
                  f"ESS Triggered and Accuracy Triggered forward-generalization "
                  f"F1 across {len(ess_cond)} batches -- report this honestly as "
                  f"'comparable performance' rather than 'ESS wins', and let the "
                  f"retrain-count difference carry the efficiency argument instead.")
else:
    print(f"Skipping Wilcoxon test: need matched batch counts and at least 6 "
          f"paired observations for the test to be meaningful "
          f"(currently {len(ess_cond)} batches). This is expected with fewer "
          f"than ~7 deployment batches -- rerun with more batches "
          f"(ess_utils.N_BATCHES) for this test to be usable.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 7: save formal independence results
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving independence analysis results ...")
print("=" * 60)

trigger_df = pd.DataFrame(trigger_summary)
trigger_df.to_csv(
    f"{RESULTS_DIR}/independence_experiment_results.csv", index=False
)

correlation_df = pd.DataFrame({
    "signal_pair"      : ["ESS vs F1", "ESS vs PSI", "F1 vs PSI"],
    "spearman"         : [ess_f1_spearman, ess_psi_spearman, f1_psi_spearman],
    "spearman_pvalue"  : [ess_f1_p,        ess_psi_p,        f1_psi_p],
    "pearson"          : [ess_f1_pearson,  ess_psi_pearson,  f1_psi_pearson],
})
correlation_df.to_csv(
    f"{RESULTS_DIR}/independence_correlations.csv", index=False
)

if wilcoxon_result is not None:
    import json
    with open(f"{RESULTS_DIR}/wilcoxon_ess_vs_accuracy.json", "w") as f:
        json.dump(wilcoxon_result, f, indent=4)
    print(f"Saved: {RESULTS_DIR}/wilcoxon_ess_vs_accuracy.json")

print(f"Saved: {RESULTS_DIR}/independence_experiment_results.csv")
print(f"Saved: {RESULTS_DIR}/independence_correlations.csv")

print("\n" + "=" * 60)
print("Independence experiment complete.")
print("=" * 60)
print("\nFiles saved:")
print(f"  {RESULTS_DIR}/independence_experiment_results.csv")
print(f"  {RESULTS_DIR}/independence_correlations.csv")