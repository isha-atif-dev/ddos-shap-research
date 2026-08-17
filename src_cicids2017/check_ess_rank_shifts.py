'''
    check_ess_rank_shifts.py

    Purpose: The threshold sweep showed ESS moves in a tighter band on
    CIC-IDS2017 (roughly 0.97-0.99) than it did on DDoS2019, needing
    tau=0.99 rather than 0.97 to catch the same real collapses. This
    script explains WHY, directly, rather than inferring it from a
    single concentration percentage: it recomputes the SHAP ranking
    for each deployment batch against the same unretrained baseline
    model used in the No Retraining / ESS Triggered trajectory, and
    reports exactly which features moved rank, by how many positions,
    at each batch. It also reports baseline concentration (top1/top3/
    top5 share) for direct comparison against DDoS2019's known
    baseline concentration (Fwd Packet Length Min held 56.3% of total
    importance there, roughly 7x the runner-up).

    This reuses the baseline model, explainer and rankings already
    saved by 03_train_baseline.py. No new model training happens here.

    What this file does:
    - Loads the baseline feature rankings and baseline model/explainer
    - Reports baseline concentration (top1/top3/top5 share of total
      |SHAP| importance), for direct comparison against DDoS2019
    - For each of the 6 deployment batches, computes the SHAP ranking
      using the SAME unretrained baseline model (matching exactly what
      drove the No Retraining / ESS Triggered trajectory already
      reported), then reports the features with the largest rank
      shift versus baseline
    - Saves a summary to outputs_cicids2017/results/ess_rank_shifts.csv
'''

import os
import json
import pickle
import pandas as pd

from ess_utils_cicids2017 import compute_shap_rankings, DEPLOYMENT_BATCHES, DEFAULT_SEED

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "results"))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "outputs_cicids2017", "models"))

SHAP_SAMPLE_SIZE = 30_000
TOP_N_SHIFTS_TO_SHOW = 10


def load_batch(batch_num, feature_names):
    X = pd.read_csv(os.path.join(RESULTS_DIR, f"batch_{batch_num}_X.csv"))[feature_names]
    return X


def main():
    print("=" * 60)
    print("STEP 1: Loading baseline rankings, model and explainer ...")
    print("=" * 60)

    baseline_rankings = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_feature_rankings.csv"))
    with open(os.path.join(RESULTS_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, "baseline_model.pkl"), "rb") as f:
        baseline_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "baseline_explainer.pkl"), "rb") as f:
        baseline_explainer = pickle.load(f)

    baseline_aligned = baseline_rankings.set_index("feature").reindex(feature_names)
    baseline_rank_by_feature = baseline_aligned["rank"].to_dict()

    # ── baseline concentration, for direct comparison against DDoS2019 ──
    print("\n" + "=" * 60)
    print("STEP 2: Baseline SHAP concentration ...")
    print("=" * 60)

    total_importance = baseline_rankings["shap_score"].sum()
    sorted_rankings = baseline_rankings.sort_values("shap_score", ascending=False)
    top1_share = sorted_rankings["shap_score"].iloc[0] / total_importance
    top3_share = sorted_rankings["shap_score"].iloc[:3].sum() / total_importance
    top5_share = sorted_rankings["shap_score"].iloc[:5].sum() / total_importance

    print(f"top1_share = {top1_share:.4f} | top3_share = {top3_share:.4f} | top5_share = {top5_share:.4f}")
    print(f"Top feature: {sorted_rankings.iloc[0]['feature']} "
          f"({sorted_rankings.iloc[0]['shap_score']:.4f})")
    print(f"Second: {sorted_rankings.iloc[1]['feature']} "
          f"({sorted_rankings.iloc[1]['shap_score']:.4f})")
    print("For comparison, DDoS2019's baseline: top1_share was 0.563 with the "
          "runner-up at 0.082 (top feature ~7x the second). A markedly lower "
          "top1_share here would mean CIC-IDS2017's baseline is LESS "
          "concentrated, not more, which would predict ESS should be MORE "
          "sensitive here, not less, the opposite of what the threshold sweep "
          "found. Worth checking directly rather than assuming.")

    # ── per-batch rank shift comparison ──
    print("\n" + "=" * 60)
    print("STEP 3: Comparing each deployment batch's ranking against baseline ...")
    print("=" * 60)

    all_shift_rows = []

    for batch_num in DEPLOYMENT_BATCHES:
        print(f"\n{'-' * 40}")
        print(f"Batch {batch_num}")
        print(f"{'-' * 40}")

        X_batch = load_batch(batch_num, feature_names)
        sample_size = min(SHAP_SAMPLE_SIZE, len(X_batch))
        X_sample = X_batch.sample(n=sample_size, random_state=DEFAULT_SEED)

        batch_ranking_df, _, _ = compute_shap_rankings(baseline_explainer, X_sample, feature_names)
        batch_rank_by_feature = batch_ranking_df.set_index("feature")["rank"].to_dict()

        shifts = []
        for feature in feature_names:
            base_rank = baseline_rank_by_feature.get(feature)
            batch_rank = batch_rank_by_feature.get(feature)
            if base_rank is not None and batch_rank is not None:
                shifts.append({
                    "batch": batch_num,
                    "feature": feature,
                    "baseline_rank": int(base_rank),
                    "batch_rank": int(batch_rank),
                    "rank_shift": int(batch_rank - base_rank),
                    "abs_shift": abs(int(batch_rank - base_rank)),
                })

        shifts_df = pd.DataFrame(shifts).sort_values("abs_shift", ascending=False)
        all_shift_rows.extend(shifts)

        print(f"Top {TOP_N_SHIFTS_TO_SHOW} features by rank shift (baseline_rank -> batch_rank):")
        for _, row in shifts_df.head(TOP_N_SHIFTS_TO_SHOW).iterrows():
            direction = "UP" if row["rank_shift"] < 0 else "DOWN"
            print(f"  {row['feature']}: {row['baseline_rank']} -> {row['batch_rank']} "
                  f"({direction} {row['abs_shift']} positions)")

        mean_abs_shift = shifts_df["abs_shift"].mean()
        top5_still_top5 = len(set(shifts_df[shifts_df['baseline_rank'] <= 5]['feature']) &
                               set(shifts_df[shifts_df['batch_rank'] <= 5]['feature']))
        print(f"  Mean |rank shift| across all {len(feature_names)} features: {mean_abs_shift:.1f} positions")
        print(f"  Of baseline's top 5 features, {top5_still_top5}/5 remain in this batch's top 5")

    all_shifts_df = pd.DataFrame(all_shift_rows)
    all_shifts_df.to_csv(os.path.join(RESULTS_DIR, "ess_rank_shifts.csv"), index=False)
    print(f"\nSaved full detail: {RESULTS_DIR}/ess_rank_shifts.csv")

    print("\n" + "=" * 60)
    print("Interpretation guide")
    print("=" * 60)
    print("- If the SAME small set of top features stays in the top 5 across every")
    print("  batch, even while lower-ranked features shuffle substantially, that is")
    print("  a direct, concrete explanation for why ESS (dominated by the largest")
    print("  rank movements, but still an overall correlation across all features)")
    print("  can stay in a tight, high band: the part of the ranking practitioners")
    print("  would actually look at first is barely moving, even as F1 collapses.")
    print("- If top features ARE reordering substantially but ESS still reads high,")
    print("  that would suggest the correlation is being diluted by the sheer")
    print("  number of features (44) rather than top-rank stability specifically,")
    print("  a different, also legitimate explanation worth reporting honestly.")
    print("- Report whichever pattern the numbers actually show.")


if __name__ == "__main__":
    main()