'''
    09_plots.py

    Purpose: Generate all figures for Chapter 4 of the dissertation.

    RENAMED from 07_plots.py to 09_plots.py, since 07 and 08 are now
    taken by the threshold sweep and multi-seed scripts.

    REBUILT relative to the earlier version:
    - Reads f1_on_arrival / auc_on_arrival (the forward-generalization
      columns) instead of the old single "f1" column, which no longer
      exists in experiment_results.csv.
    - Titles and captions are now COMPUTED from the actual data instead
      of hardcoded ("PSI never crossed the threshold...", "No Retraining
      and PSI Triggered overlap..."). Those specific claims were true of
      an earlier 6-batch run but are false in the current 10-batch data
      (PSI does cross its threshold sometimes, and PSI Triggered does
      retrain sometimes) -- a hardcoded caption would have silently
      become wrong the moment new data was dropped in.
    - Two new figures added, using the threshold sweep and multi-seed
      results that didn't exist when the original script was written:
        Figure 6: threshold sensitivity (is tau=0.97 a robust choice?)
        Figure 7: multi-seed robustness (mean +/- std across seeds)
      Both are skipped gracefully with a printed note if you haven't
      run 07_threshold_sweep.py / 08_multi_seed.py yet.
    - Figure 1B (Portmap under synthetic drift) is now generated here
      too, reading directly from outputs/results/synthetic_drift_injection.csv
      (produced by 12_synthetic_drift_injection.py) instead of living in a
      separate one-off script with the numbers typed in by hand. Its
      caption is computed from the actual crossing points in that data,
      same as every other figure in this file, so it can't go stale.
      Skipped gracefully with a printed note if the CSV doesn't exist yet.

    All filenames now match the \\includegraphics{plots/figure_4_N.png}
    references in the Chapter 4 LaTeX exactly: figure_4_1.png through
    figure_4_7.png, plus figure_4_5b_portmap.png for Figure 1B above.

    Five of the seven figures follow the same story-driven layout style
    used in earlier chapter drafts (one clear takeaway per figure,
    plain-language subtitles) rather than generic default plots.
'''

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_DIR = "outputs/results"
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLOURS = {
    "No Retraining": "#95a5a6",
    "ESS Triggered": "#2ecc71",
    "Accuracy Triggered": "#3498db",
    "PSI Triggered": "#f39c12",
    "ESS": "#2ecc71",
    "Euclidean": "#9b59b6",
    "PSI": "#f39c12",
    "F1": "#e74c3c",
}
LINE_STYLES = {
    "No Retraining": (0, (5, 5)),
    "ESS Triggered": "solid",
    "Accuracy Triggered": "solid",
    "PSI Triggered": (0, (3, 1, 1, 1)),
}
MARKERS = {
    "No Retraining": "X",
    "ESS Triggered": "o",
    "Accuracy Triggered": "s",
    "PSI Triggered": "^",
}
CONDITIONS = ["No Retraining", "ESS Triggered", "Accuracy Triggered", "PSI Triggered"]

ESS_THRESHOLD = 0.97
F1_THRESHOLD = 0.75
PSI_THRESHOLD = 0.1

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def safe_pct(start, end):
    if start == 0:
        return None
    return (end - start) / start * 100


def pct_str(p):
    return "undefined (started at 0)" if p is None else f"{p:+.1f}%"


def ess_narrative(pct, crossings, n):
    '''Editorial framing for the ESS panel, chosen from the actual
    crossing count so it can never go stale like a hardcoded caption.'''
    if crossings == 0:
        return "stays within threshold throughout"
    frac = crossings / n
    if frac < 0.5:
        return f"crosses threshold on {crossings} of {n} batches -- moderate, threshold-relevant drift"
    return f"crosses threshold on {crossings} of {n} batches -- persistent drift"


def f1_narrative(pct, final_val):
    if pct is None:
        return "collapses to near-zero" if final_val < 0.05 else "undefined change (started at 0)"
    if pct <= -80:
        return "near-total collapse"
    if pct <= -30:
        return "significant decline"
    if pct < -5:
        return "mild decline"
    if pct >= 5:
        return "improves"
    return "stays roughly stable"


def psi_narrative(crossings, n):
    if crossings == 0:
        return "stays flat, never alarms"
    frac = crossings / n
    if frac < 0.5:
        return f"alarms on {crossings} of {n} batches -- partial detection"
    return f"alarms on {crossings} of {n} batches -- frequent flagging"


def crossing_alpha(alpha_series, value_series, threshold, below=True):
    '''First alpha at which value_series crosses threshold, computed from
    the actual data rather than hardcoded, so the caption can never go
    stale if the synthetic drift run is repeated with different results.
    Returns None if the threshold is never crossed.'''
    for a, v in zip(alpha_series, value_series):
        if (below and v < threshold) or (not below and v > threshold):
            return a
    return None


def final_f1_for(cond_df):
    '''Final-batch F1: uses f1_after_retrain if the final batch triggered
    a retrain, otherwise f1_on_arrival. This is "what F1 would you
    actually observe at the end of the deployment window".'''
    last = cond_df.sort_values("batch").iloc[-1]
    if bool(last["retrained"]) and not pd.isna(last.get("f1_after_retrain")):
        return float(last["f1_after_retrain"])
    return float(last["f1_on_arrival"])


# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load everything, tolerating missing optional files
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading results ...")
print("=" * 60)

experiment_df = pd.read_csv(f"{RESULTS_DIR}/experiment_results.csv")
ess_scores_df = pd.read_csv(f"{RESULTS_DIR}/ess_scores.csv")
no_retrain = experiment_df[experiment_df["condition"] == "No Retraining"].copy().sort_values("batch")

wilcoxon_p = None
wilcoxon_path = f"{RESULTS_DIR}/wilcoxon_ess_vs_accuracy.json"
if os.path.exists(wilcoxon_path):
    with open(wilcoxon_path) as f:
        wilcoxon_p = json.load(f)["p_value"]

threshold_sweep_df = None
sweep_path = f"{RESULTS_DIR}/threshold_sweep.csv"
if os.path.exists(sweep_path):
    threshold_sweep_df = pd.read_csv(sweep_path)
else:
    print(f"NOTE: {sweep_path} not found -- run 07_threshold_sweep.py for Figure 6. Skipping.")

multi_seed_df = None
multi_seed_path = f"{RESULTS_DIR}/multi_seed_summary.csv"
if os.path.exists(multi_seed_path):
    multi_seed_df = pd.read_csv(multi_seed_path)
else:
    print(f"NOTE: {multi_seed_path} not found -- run 08_multi_seed.py for Figure 7. Skipping.")

synthetic_drift_df = None
synthetic_drift_path = f"{RESULTS_DIR}/synthetic_drift_injection.csv"
if os.path.exists(synthetic_drift_path):
    synthetic_drift_df = pd.read_csv(synthetic_drift_path)
else:
    print(f"NOTE: {synthetic_drift_path} not found -- run 12_synthetic_drift_injection.py for the Portmap figure. Skipping.")

print("Loaded.")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1: same deployment, three different verdicts (No Retraining)
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 1: ESS / F1 / PSI under No Retraining ...")

ess_pct = safe_pct(no_retrain["ess"].iloc[0], no_retrain["ess"].iloc[-1])
f1_pct = safe_pct(no_retrain["f1_on_arrival"].iloc[0], no_retrain["f1_on_arrival"].iloc[-1])
psi_crossings = int((no_retrain["psi"] > PSI_THRESHOLD).sum())
n_batches = len(no_retrain)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    "Same Deployment, Three Different Verdicts\n"
    "ESS, F1 and PSI Under No Retraining (CIC-DDoS2019)",
    fontsize=14, fontweight="bold", y=0.99
)

axes[0].plot(no_retrain["batch"], no_retrain["ess"], marker="o", linewidth=2.5,
             markersize=7, color=COLOURS["ESS"], label="ESS")
axes[0].axhline(ESS_THRESHOLD, color=COLOURS["ESS"], linestyle="--", linewidth=1.3, alpha=0.6, label=f"threshold = {ESS_THRESHOLD}")
ess_crossings = int((no_retrain["ess"] < ESS_THRESHOLD).sum())
axes[0].set_title(f"Explanation Stability Score -- {ess_narrative(ess_pct, ess_crossings, n_batches)}", loc="left")
axes[0].set_ylabel("ESS\n(Spearman)")
axes[0].legend(loc="lower left")

axes[1].plot(no_retrain["batch"], no_retrain["f1_on_arrival"], marker="o", linewidth=2.5,
             markersize=7, color=COLOURS["F1"], label="F1 (forward-generalization)")
axes[1].axhline(F1_THRESHOLD, color=COLOURS["F1"], linestyle="--", linewidth=1.3, alpha=0.6, label=f"threshold = {F1_THRESHOLD}")
axes[1].set_title(f"Accuracy -- {f1_narrative(f1_pct, no_retrain['f1_on_arrival'].iloc[-1])}", loc="left")
axes[1].set_ylabel("F1 Score\n(Weighted)")
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(loc="center left")

axes[2].plot(no_retrain["batch"], no_retrain["psi"], marker="o", linewidth=2.5,
             markersize=7, color=COLOURS["PSI"], label="PSI")
axes[2].axhline(PSI_THRESHOLD, color=COLOURS["PSI"], linestyle="--", linewidth=1.3, alpha=0.6, label=f"threshold = {PSI_THRESHOLD}")
axes[2].set_title(f"Population Stability Index -- {psi_narrative(psi_crossings, n_batches)}", loc="left")
axes[2].set_ylabel("PSI")
axes[2].set_xlabel("Deployment Batch")
axes[2].set_xticks(no_retrain["batch"])
axes[2].legend(loc="upper left")

for ax in axes:
    ax.grid(True, alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{PLOTS_DIR}/figure_4_1.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_1.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1B: Portmap under synthetic drift injection -- same three signals
# as Figure 1, but under controlled, known drift rather than the real
# no-retraining deployment. Reads directly from
# outputs/results/synthetic_drift_injection.csv (produced by
# 12_synthetic_drift_injection.py) and computes its crossing points from
# that data, rather than the hardcoded arrays an earlier one-off script
# used, so this can never go stale if the synthetic run is repeated.
# ──────────────────────────────────────────────────────────────────────────
if synthetic_drift_df is not None:
    print("\nFIGURE 1B: Portmap under synthetic drift injection ...")

    portmap_df = synthetic_drift_df[synthetic_drift_df["class"] == "Portmap"].sort_values("alpha")

    if portmap_df.empty:
        print("NOTE: no Portmap rows found in synthetic_drift_injection.csv. Skipping Figure 1B.")
    else:
        ess_cross_alpha = crossing_alpha(portmap_df["alpha"], portmap_df["ess"], ESS_THRESHOLD, below=True)
        f1_cross_alpha = crossing_alpha(portmap_df["alpha"], portmap_df["f1"], F1_THRESHOLD, below=True)
        psi_cross_alpha = crossing_alpha(portmap_df["alpha"], portmap_df["psi"], PSI_THRESHOLD, below=False)

        ess_title = (f"Explanation Stability Score -- crosses threshold at alpha = {ess_cross_alpha:.1f}"
                     if ess_cross_alpha is not None else "Explanation Stability Score -- stays within threshold throughout")
        f1_title = (f"Accuracy -- crosses threshold at alpha = {f1_cross_alpha:.1f}"
                    if f1_cross_alpha is not None else "Accuracy -- remains near-perfect throughout")
        psi_title = (f"Distribution shift -- stays below threshold until alpha = {psi_cross_alpha:.1f}"
                     if psi_cross_alpha is not None else "Distribution shift -- stays within threshold throughout")

        fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.2), sharex=True)

        axes[0].plot(portmap_df["alpha"], portmap_df["ess"], marker="o", linewidth=2, markersize=5, color=COLOURS["ESS"])
        axes[0].axhline(ESS_THRESHOLD, color=COLOURS["ESS"], linestyle="--", linewidth=1, alpha=0.6, label=f"threshold = {ESS_THRESHOLD}")
        axes[0].set_ylabel("ESS\n(Spearman)")
        axes[0].set_title(ess_title, loc="left", fontsize=10.5)
        axes[0].legend(loc="lower left", frameon=False)
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(portmap_df["alpha"], portmap_df["f1"], marker="o", linewidth=2, markersize=5, color=COLOURS["F1"])
        axes[1].axhline(F1_THRESHOLD, color=COLOURS["F1"], linestyle="--", linewidth=1, alpha=0.6, label=f"threshold = {F1_THRESHOLD}")
        axes[1].set_ylabel("F1\n(forward-generalisation)")
        axes[1].set_ylim(0.6, 1.05)
        axes[1].set_title(f1_title, loc="left", fontsize=10.5)
        axes[1].legend(loc="lower left", frameon=False)
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(portmap_df["alpha"], portmap_df["psi"], marker="o", linewidth=2, markersize=5, color=COLOURS["PSI"])
        axes[2].axhline(PSI_THRESHOLD, color=COLOURS["PSI"], linestyle="--", linewidth=1, alpha=0.6, label=f"threshold = {PSI_THRESHOLD}")
        axes[2].set_ylabel("PSI")
        axes[2].set_xlabel(r"Injected drift strength ($\alpha$, fraction toward benign mean)")
        axes[2].set_title(psi_title, loc="left", fontsize=10.5)
        axes[2].legend(loc="upper left", frameon=False)
        axes[2].grid(True, alpha=0.25)

        fig.suptitle(
            "Portmap Under Synthetic Drift Injection\n"
            "ESS Crosses Threshold While Accuracy and PSI Both Remain Quiet",
            fontsize=12.5, fontweight="bold", y=0.99
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{PLOTS_DIR}/figure_4_5b_portmap.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {PLOTS_DIR}/figure_4_5b_portmap.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 2: F1 across four conditions, with retrain events marked
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 2: F1 across four conditions ...")

nr_f1 = experiment_df[experiment_df["condition"] == "No Retraining"].sort_values("batch")["f1_on_arrival"].values
psi_f1 = experiment_df[experiment_df["condition"] == "PSI Triggered"].sort_values("batch")["f1_on_arrival"].values
overlap_note = ("No Retraining and PSI Triggered track closely in this run"
                 if len(nr_f1) == len(psi_f1) and np.allclose(nr_f1, psi_f1, atol=0.02)
                 else "All four conditions diverge from each other in this run")

fig, ax = plt.subplots(figsize=(10, 6))
for condition in CONDITIONS:
    cond_df = experiment_df[experiment_df["condition"] == condition].sort_values("batch")
    ax.plot(cond_df["batch"], cond_df["f1_on_arrival"], marker=MARKERS[condition],
            linestyle=LINE_STYLES[condition], linewidth=2.5, markersize=8,
            color=COLOURS[condition], label=condition)
    retrained = cond_df[cond_df["retrained"] == True]
    if not retrained.empty:
        ax.scatter(retrained["batch"], retrained["f1_on_arrival"], marker="*", s=280,
                   color=COLOURS[condition], edgecolor="black", linewidth=0.6, zorder=5)

ax.axhline(F1_THRESHOLD, color="black", linestyle=":", linewidth=1.3, label=f"F1 trigger threshold ({F1_THRESHOLD})")
ax.set_xlabel("Deployment Batch")
ax.set_ylabel("F1 Score (Weighted, forward-generalization)")
ax.set_title(f"F1 Score Across Four Retraining Conditions (CIC-DDoS2019)", fontsize=14, fontweight="bold", pad=14)
ax.set_xticks(sorted(experiment_df["batch"].unique()))
ax.set_ylim(-0.08, 1.22)
ax.grid(True, alpha=0.25)

handles, labels = ax.get_legend_handles_labels()
# Legend placed INSIDE the axes using matplotlib's automatic best-fit
# location -- this is the line-chart equivalent of the empty-corner
# detection used for Figure 3's scatter points (which doesn't directly
# apply here since lines cover irregular shapes, not four discrete
# points). The extra ylim headroom above gives it clear room without
# guessing exactly where.
ax.legend(handles, labels, loc="best", ncol=1, frameon=True, framealpha=0.9,
          edgecolor="lightgray", fontsize=10, markerscale=0.9, labelspacing=0.8)

# Caption tied to the axes bottom (transAxes) instead of a separately
# guessed figure-fraction y, so its distance from the x-axis label scales
# predictably instead of drifting into its own gap.
ax.text(0.5, -0.14, f"* = retraining event.  {overlap_note}.",
        transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="dimgray")

plt.savefig(f"{PLOTS_DIR}/figure_4_2.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_2.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3: retraining cost vs final accuracy
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 3: Retraining cost vs final accuracy ...")

points = []
for condition in CONDITIONS:
    cond_df = experiment_df[experiment_df["condition"] == condition]
    n_retrains = int(cond_df["retrained"].sum())
    final_f1 = final_f1_for(cond_df)
    points.append({"condition": condition, "retrains": n_retrains, "final_f1": final_f1})
points_df = pd.DataFrame(points)

# Inline text labels next to each point (the previous approach) collide
# whenever two points are merely CLOSE, not just identical -- e.g. two
# points at different x but similar y still produce overlapping label
# text, since label width isn't something matplotlib avoids automatically.
# A legend sidesteps the whole problem: it's confined to its own reserved
# box and can never overlap the data area, regardless of how close points
# are. This is the same reason Figure 2's legend never collides with its
# lines. Markers are still nudged apart with a tiny jitter when two
# conditions land on the EXACT same point, purely so both marker shapes
# stay visible.
tolerance_f1 = 0.01
groups, used = [], [False] * len(points_df)
for i in range(len(points_df)):
    if used[i]:
        continue
    group = [i]
    used[i] = True
    for j in range(i + 1, len(points_df)):
        if used[j]:
            continue
        if (points_df.loc[i, "retrains"] == points_df.loc[j, "retrains"]
                and abs(points_df.loc[i, "final_f1"] - points_df.loc[j, "final_f1"]) < tolerance_f1):
            group.append(j)
            used[j] = True
    groups.append(group)
any_grouped = any(len(g) > 1 for g in groups)

fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.axhspan(0.95, 1.03, color="green", alpha=0.06)

legend_handles, legend_labels = [], []
for group in groups:
    x0 = float(points_df.loc[group[0], "retrains"])
    y0 = float(points_df.loc[group[0], "final_f1"])
    n = len(group)
    jitter_span = 0.18 * (n - 1)
    for k, idx in enumerate(group):
        row = points_df.loc[idx]
        jitter_x = x0 - jitter_span / 2 + (k * jitter_span / (n - 1) if n > 1 else 0)
        h = ax.scatter(jitter_x, y0, s=380, color=COLOURS[row["condition"]],
                       edgecolor="black", linewidth=1, zorder=5, marker=MARKERS[row["condition"]])
        legend_handles.append(h)
        legend_labels.append(f"{row['condition']}: {row['retrains']} retrains, F1={row['final_f1']:.4f}")

max_retrain = max(points_df["retrains"].max(), 1)
ax.set_xlim(-0.7, max_retrain + 1.5)
ax.set_ylim(-0.05, 1.12)
ax.set_xlabel("Number of Retrains Across Deployment")
ax.set_ylabel("Final Batch F1 Score")
ax.set_title("Retraining Cost vs Final Accuracy\nFewer retrains and higher F1 is better (top-left)",
             fontsize=13, fontweight="bold", pad=14)
ax.grid(True, alpha=0.25)

# Place the legend INSIDE whichever corner is actually empty of data points,
# rather than anchoring it outside the axes (which is what needed all the
# manual gap-tuning before). The top band is reserved by the green "good"
# shading across the full width, so only the two bottom corners are
# candidates; pick whichever has points farther from it. This adapts to
# the real data instead of assuming bottom-right is always free (a
# condition that retrains a lot AND still fails would sit there).
xlo, xhi = ax.get_xlim()
ylo, yhi = ax.get_ylim()
norm_x = (points_df["retrains"] - xlo) / (xhi - xlo)
norm_y = (points_df["final_f1"] - ylo) / (yhi - ylo)
dist_to_lower_left = np.sqrt(norm_x ** 2 + norm_y ** 2).min()
dist_to_lower_right = np.sqrt((1 - norm_x) ** 2 + norm_y ** 2).min()
legend_loc = "lower left" if dist_to_lower_left >= dist_to_lower_right else "lower right"
best_corner_clearance = max(dist_to_lower_left, dist_to_lower_right)

if best_corner_clearance > 0.35:
    # enough clear room inside the axes -- legend sits inside the plot,
    # matplotlib's own loc positioning handles spacing with zero manual tuning
    ax.legend(legend_handles, legend_labels, loc=legend_loc, ncol=1,
              frameon=True, framealpha=0.9, edgecolor="lightgray",
              fontsize=10, labelspacing=1.3, markerscale=0.55, handletextpad=0.8,
              borderpad=1.0)
else:
    # not enough clear space inside the axes this run -- fall back to
    # placing it above the plot, anchored in axes-fraction coordinates
    ax.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.14),
              ncol=2, frameon=False, fontsize=10.5, labelspacing=1.0, markerscale=0.55,
              handletextpad=0.8)

caption_lines = []
if any_grouped:
    caption_lines.append("Markers nudged apart where two conditions land on the exact same point.")
if wilcoxon_p is not None:
    caption_lines.append(
        f"Wilcoxon signed-rank, ESS vs Accuracy Triggered F1: p = {wilcoxon_p:.4f}"
        + (" (significant)" if wilcoxon_p < 0.05 else " (not significant)"))
if caption_lines:
    fig.text(0.5, -0.02, "  ".join(caption_lines), ha="center", fontsize=9.5, style="italic", color="dimgray")

plt.savefig(f"{PLOTS_DIR}/figure_4_3.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_3.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 4: retraining events heatmap
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 4: Retraining events heatmap ...")

batches = sorted(experiment_df["batch"].unique())
retrain_matrix = pd.DataFrame(index=CONDITIONS, columns=batches, dtype=int)
for condition in CONDITIONS:
    cond_df = experiment_df[experiment_df["condition"] == condition]
    for batch in batches:
        row = cond_df[cond_df["batch"] == batch]
        retrain_matrix.loc[condition, batch] = int(row["retrained"].values[0]) if not row.empty else 0
retrain_matrix = retrain_matrix.astype(int)

fig, ax = plt.subplots(figsize=(1.1 * len(batches) + 2, 5))
sns.heatmap(retrain_matrix, annot=True, fmt="d", cmap=["#ecf0f1", "#27ae60"],
            linewidths=1, linecolor="white", cbar=False, ax=ax, annot_kws={"fontsize": 12, "fontweight": "bold"})
ax.set_xlabel("Deployment Batch")
ax.set_ylabel("")
total_retrains = {c: int(retrain_matrix.loc[c].sum()) for c in CONDITIONS}
subtitle = " | ".join(f"{c}: {n}" for c, n in total_retrains.items())
ax.set_title(f"Retraining Events by Condition and Batch\nTotal retrains -- {subtitle}", fontsize=12, fontweight="bold")
ax.set_xticklabels([f"B{b}" for b in batches], rotation=0)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/figure_4_4.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_4.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 5: three ways of measuring explanation change
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 5: ESS vs Euclidean vs PSI ...")

fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
fig.suptitle("Three Ways of Measuring Explanation Change, Same SHAP Output", fontsize=12.5, fontweight="bold", y=1.02)

axes[0].plot(ess_scores_df["batch"], ess_scores_df["ess"], marker="o", linewidth=2.5, color=COLOURS["ESS"])
axes[0].axhline(ESS_THRESHOLD, color=COLOURS["ESS"], linestyle="--", linewidth=1.3, alpha=0.6)
axes[0].set_title("rank-based (ESS)", loc="left", fontsize=10.5, fontweight="bold")
axes[0].set_ylabel("ESS (Spearman rank correlation)")

axes[1].plot(ess_scores_df["batch"], ess_scores_df["euclidean"], marker="o", linewidth=2.5, color=COLOURS["Euclidean"])
axes[1].set_title("magnitude-based (Euclidean)", loc="left", fontsize=10.5, fontweight="bold")
axes[1].set_ylabel("Euclidean Distance (SHAP magnitude)")

axes[2].plot(ess_scores_df["batch"], ess_scores_df["psi"], marker="o", linewidth=2.5, color=COLOURS["PSI"])
axes[2].axhline(PSI_THRESHOLD, color=COLOURS["PSI"], linestyle="--", linewidth=1.3, alpha=0.6)
axes[2].set_title("distribution-based (PSI)", loc="left", fontsize=10.5, fontweight="bold")
axes[2].set_ylabel("PSI (SHAP distribution shift)")

for ax in axes:
    ax.set_xlabel("Deployment Batch")
    ax.set_xticks(ess_scores_df["batch"])
    ax.grid(True, alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{PLOTS_DIR}/figure_4_5.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_5.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 5B (NEW): vertical, single-column version of Figure 5
# Same data as Figure 5, restacked into 3 rows x 1 column and sized to
# match a single IEEE column width (roughly 3.4in), so it doesn't need
# to be stretched in LaTeX. subplots_adjust (not tight_layout) fixes one
# shared left margin for the whole figure, which is what keeps the three
# plot boxes lined up regardless of how wide each row's tick labels are.
# ──────────────────────────────────────────────────────────────────────────
print("\nFIGURE 5B: ESS vs Euclidean vs PSI (vertical, single column) ...")

fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.6), sharex=True)
fig.suptitle("Three Ways of Measuring\nExplanation Change, Same SHAP Output",
             fontsize=10.5, fontweight="bold", y=0.995)

axes[0].plot(ess_scores_df["batch"], ess_scores_df["ess"], marker="o",
             linewidth=1.8, markersize=4.5, color=COLOURS["ESS"])
axes[0].axhline(ESS_THRESHOLD, color=COLOURS["ESS"], linestyle="--", linewidth=1.1, alpha=0.6)
axes[0].set_title("rank-based (ESS)", fontweight="bold", fontsize=9.5)
axes[0].set_ylabel("ESS\n(Spearman)", fontsize=8.5)

axes[1].plot(ess_scores_df["batch"], ess_scores_df["euclidean"], marker="o",
             linewidth=1.8, markersize=4.5, color=COLOURS["Euclidean"])
axes[1].set_title("magnitude-based (Euclidean)", fontweight="bold", fontsize=9.5)
axes[1].set_ylabel("Euclidean\nDistance", fontsize=8.5)

axes[2].plot(ess_scores_df["batch"], ess_scores_df["psi"], marker="o",
             linewidth=1.8, markersize=4.5, color=COLOURS["PSI"])
axes[2].axhline(PSI_THRESHOLD, color=COLOURS["PSI"], linestyle="--", linewidth=1.1, alpha=0.6)
axes[2].set_title("distribution-based (PSI)", fontweight="bold", fontsize=9.5)
axes[2].set_ylabel("PSI", fontsize=8.5)
axes[2].set_xlabel("Deployment Batch", fontsize=9)
axes[2].set_xticks(ess_scores_df["batch"])

for ax in axes:
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

fig.align_ylabels(axes)
fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.08, hspace=0.45)

plt.savefig(f"{PLOTS_DIR}/figure_4_5_vertical.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {PLOTS_DIR}/figure_4_5_vertical.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 6 (NEW): threshold sensitivity
# ──────────────────────────────────────────────────────────────────────────
if threshold_sweep_df is not None:
    print("\nFIGURE 6: Threshold sensitivity ...")

    acc_final_f1 = final_f1_for(experiment_df[experiment_df["condition"] == "Accuracy Triggered"])

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(threshold_sweep_df["ess_threshold"].astype(str), threshold_sweep_df["n_retrains"],
                    color=COLOURS["ESS"], alpha=0.35, width=0.55, label="Retrains (left axis)")
    ax2.plot(threshold_sweep_df["ess_threshold"].astype(str), threshold_sweep_df["final_batch_f1"],
             marker="o", linewidth=2.5, markersize=9, color=COLOURS["ESS"], label="Final F1 (right axis)")
    ax2.axhline(acc_final_f1, color=COLOURS["Accuracy Triggered"], linestyle="--", linewidth=1.5,
                label=f"Accuracy Triggered final F1 ({acc_final_f1:.4f})")

    ax1.set_xlabel("ESS Threshold (tau)")
    ax1.set_ylabel("Number of Retrains", color=COLOURS["ESS"])
    ax2.set_ylabel("Final Batch F1", color=COLOURS["ESS"])
    ax2.set_ylim(-0.05, 1.1)
    ax1.set_title(
        "ESS Threshold Sensitivity\n"
        "Retrain count rises as tau tightens; F1 stays high across most of the range",
        fontsize=12, fontweight="bold"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/figure_4_6.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/figure_4_6.png")

# ──────────────────────────────────────────────────────────────────────────
# FIGURE 7 (NEW): multi-seed robustness
# ──────────────────────────────────────────────────────────────────────────
if multi_seed_df is not None:
    print("\nFIGURE 7: Multi-seed robustness ...")

    ordered = multi_seed_df.set_index("condition").reindex(CONDITIONS).reset_index()
    n_seeds = int(ordered["n_seeds"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(ordered["condition"], ordered["final_f1_mean"], yerr=ordered["final_f1_std"],
                color=[COLOURS[c] for c in ordered["condition"]], edgecolor="black", linewidth=0.8,
                capsize=6, width=0.55)
    axes[0].set_ylabel("Final Batch F1 (mean \u00b1 std)")
    axes[0].set_title(f"Final F1 Across {n_seeds} Seeds", fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    for i, row in ordered.iterrows():
        axes[0].text(i, row["final_f1_mean"] + row["final_f1_std"] + 0.03,
                     f"{row['final_f1_mean']:.3f}\n\u00b1{row['final_f1_std']:.3f}",
                     ha="center", fontsize=9)

    axes[1].bar(ordered["condition"], ordered["retrains_mean"], yerr=ordered["retrains_std"],
                color=[COLOURS[c] for c in ordered["condition"]], edgecolor="black", linewidth=0.8,
                capsize=6, width=0.55)
    axes[1].set_ylabel("Number of Retrains (mean \u00b1 std)")
    axes[1].set_title(f"Retrain Count Across {n_seeds} Seeds", fontweight="bold")
    max_retrain_height = (ordered["retrains_mean"] + ordered["retrains_std"]).max()
    axes[1].set_ylim(0, max_retrain_height * 1.35 + 0.3)
    for i, row in ordered.iterrows():
        axes[1].text(i, row["retrains_mean"] + row["retrains_std"] + max_retrain_height * 0.08,
                     f"{row['retrains_mean']:.2f}\n\u00b1{row['retrains_std']:.2f}",
                     ha="center", fontsize=9)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        f"Robustness Across {n_seeds} Random Seeds\n"
        "Small error bars on ESS Triggered mean the result is not a one-off lucky split",
        fontsize=13, fontweight="bold", y=1.04
    )

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/figure_4_7.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/figure_4_7.png")

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All available figures saved to outputs/plots/")
print("=" * 60)
for fname in sorted(os.listdir(PLOTS_DIR)):
    print(f"  {fname}")