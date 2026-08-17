'''
    12_synthetic_drift_injection.py  (REBUILT: directed drift + all classes)

    Purpose: A controlled, synthetic drift experiment, addressing the
    same novel-class-vs-drift concern as 11_ldap_drift_check.py, but
    from the opposite direction. That script checked for drift that
    may or may not have been present in real data across two real
    dates, and found none for LDAP. This script does not wait for real
    drift to exist; it manufactures a known, controlled amount of it,
    following the method already used in Pawlicki, Kozik and Choras
    (cited in Chapter 2), who simulate gradual drift on real NetFlow
    data and check whether SHAP-based signals track it.

    HOW THIS VERSION DIFFERS FROM THE FIRST TWO ATTEMPTS:
    Attempt 1 multiplied top features by a growing factor -- this did
    almost nothing, because scaling values that are already far past
    a tree's decision threshold doesn't move them across it.
    Attempt 2 added a growing number of standard deviations in a fixed
    direction -- this moved ESS a small, real amount, but only in
    step-shaped jumps whenever a threshold happened to be crossed, and
    never approached the main pipeline's tau = 0.97 boundary even at
    4 standard deviations.
    This version instead shifts the top continuous features of ATTACK
    traffic toward the mean values seen in BENIGN traffic for that same
    class's training data, by a growing percentage from 0% to 100%.
    This is not an arbitrary direction: it directly simulates an
    attacker adapting their traffic to look more like normal traffic in
    order to evade detection, the adversarial drift scenario already
    discussed in Chapters 1 and 2, and it is mechanically guaranteed to
    eventually move samples across the model's decision boundary, since
    at 100% the perturbed features are, on average, indistinguishable
    from benign traffic.

    This version also loops over all seven attack classes present in
    the 3 November data automatically (Portmap does not require the
    December file at all, since this script never uses real
    testing-day data, so it is included here even though it was not
    used in 11_ldap_drift_check.py).

    The model is NEVER retrained across the synthetic steps for any
    class, matching the no-retraining condition used throughout the
    main experiment.

    What this file does, for EACH of the seven attack classes:
    - Loads and cleans that class's November CSV, trains a binary
      model (that class vs BENIGN), computes its baseline SHAP ranking
    - Filters the top-ranked baseline features down to continuous ones
      only (excluding near-binary/low-cardinality features), then
      selects PERTURB_TOP_N of those
    - Computes, from the training data, the mean value of each
      selected feature separately for the attack rows and the benign
      rows
    - For each alpha in DRIFT_STEPS (0.0 to 1.0), draws a fresh sample
      of held-out attack rows, moves each selected feature alpha of
      the way from its own attack-class mean toward the benign-class
      mean, and evaluates the SAME unretrained model: forward-
      generalisation F1, ESS against the original baseline ranking,
      Euclidean distance and PSI
    - Saves the full step-by-step trajectory for all seven classes,
      combined into one file, to
      outputs/results/synthetic_drift_injection.csv
'''

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ess_utils import compute_shap_rankings, compute_ess, compute_euclidean, compute_psi

RESULTS_DIR = "outputs/results"
RAW_DIR = "data/raw/03-11"
SAMPLE_SIZE = 30_000
SEED = 42

ATTACK_CLASSES = ["LDAP", "MSSQL", "NetBIOS", "Portmap", "Syn", "UDP", "UDPLag"]

PERTURB_TOP_N = 5            # how many continuous top-ranked features to perturb (raised from 3)
MIN_UNIQUE_VALUES = 20       # features with fewer distinct values than this are excluded as near-binary/categorical
DRIFT_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # fraction of the way from attack mean to benign mean

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    feature_names = json.load(f)


def load_and_clean(class_name, sample_size, seed=SEED):
    '''
    Loads one raw CIC-DDoS2019 CSV in chunks, matching the approach
    already proven to work on files this large in 01_data_loader.py.
    A single, non-chunked pd.read_csv call ran out of memory on the
    larger files (e.g. MSSQL.csv, ~1.8GB), even though it worked fine
    on smaller ones like LDAP.csv (~900MB).
    '''
    path = f"{RAW_DIR}/{class_name}.csv"
    chunks = []
    for encoding in ["utf-8", "latin-1"]:
        try:
            reader = pd.read_csv(path, encoding=encoding, low_memory=False, chunksize=100_000)
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)
            break
        except UnicodeDecodeError:
            chunks = []
            continue
    df = pd.concat(chunks, ignore_index=True)

    label_col = next(c for c in df.columns if "label" in c.lower())
    df["label_binary"] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    df_sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    X = df_sample[feature_names].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    y = df_sample["label_binary"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


def run_for_class(class_name):
    print("\n" + "#" * 60)
    print(f"# CLASS: {class_name}")
    print("#" * 60)

    print("-" * 60)
    print(f"Loading and cleaning {class_name}.csv ...")
    print("-" * 60)
    X_full, y_full = load_and_clean(class_name, SAMPLE_SIZE)
    print(f"Sample shape: {X_full.shape} | Label distribution:\n{y_full.value_counts().to_dict()}")

    if y_full.nunique() < 2:
        print(f"SKIPPING {class_name}: only one class present in the sample, cannot train a binary model.")
        return []

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=SEED, stratify=y_full
    )

    print("-" * 60)
    print("Training the fixed baseline model ...")
    print("-" * 60)
    params = {
        "objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
        "eval_metric": "logloss", "seed": SEED, "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    baseline_f1 = f1_score(y_test, (model.predict(xgb.DMatrix(X_test)) > 0.5).astype(int), average="weighted")
    print(f"Held-out F1 on unperturbed data (sanity check): {baseline_f1:.4f}")

    print("-" * 60)
    print("Computing baseline SHAP ranking and selecting features to perturb ...")
    print("-" * 60)
    explainer = shap.TreeExplainer(model)
    baseline_ranking_df, baseline_shap_array, baseline_scores = compute_shap_rankings(
        explainer, X_train, feature_names
    )

    nunique = X_train.nunique()
    continuous_candidates = [f for f in baseline_ranking_df["feature"] if nunique.get(f, 0) >= MIN_UNIQUE_VALUES]
    perturb_features = continuous_candidates[:PERTURB_TOP_N]
    print(f"Perturbing: {perturb_features}")

    attack_means = {f: X_train.loc[y_train == 1, f].mean() for f in perturb_features}
    benign_means = {f: X_train.loc[y_train == 0, f].mean() for f in perturb_features}
    for f in perturb_features:
        print(f"  {f}: attack mean = {attack_means[f]:.4f} | benign mean = {benign_means[f]:.4f}")

    baseline_aligned = baseline_ranking_df.set_index("feature").reindex(feature_names)
    baseline_rank_vector = baseline_aligned["rank"].values

    print("-" * 60)
    print("Running directed drift steps (attack traffic shifting toward benign) ...")
    print("-" * 60)

    class_results = []
    # only perturb rows that are actually attack rows, matching the adversarial-evasion framing
    attack_pool = X_test.loc[y_test == 1]

    for i, alpha in enumerate(DRIFT_STEPS):
        X_step = attack_pool.sample(n=min(SAMPLE_SIZE, len(attack_pool)), random_state=SEED + i, replace=True).copy()
        X_step = X_step.reset_index(drop=True)
        y_step = pd.Series(np.ones(len(X_step), dtype=int))  # all attack rows, forward-generalisation checks whether the model still catches them

        for f in perturb_features:
            X_step[f] = X_step[f] + alpha * (benign_means[f] - attack_means[f])

        dstep = xgb.DMatrix(X_step)
        step_pred = (model.predict(dstep) > 0.5).astype(int)
        step_f1 = f1_score(y_step, step_pred, average="weighted", zero_division=0)

        step_ranking_df, step_shap_array, step_scores = compute_shap_rankings(explainer, X_step, feature_names)
        step_aligned = step_ranking_df.set_index("feature").reindex(feature_names)
        step_rank_vector = step_aligned["rank"].values

        ess, pvalue = compute_ess(baseline_rank_vector, step_rank_vector)
        euclidean = compute_euclidean(baseline_scores, step_scores)
        psi = compute_psi(baseline_shap_array, step_shap_array)

        print(f"  Step {i+1} (alpha={alpha:.1f}): F1={step_f1:.4f} | ESS={ess:.4f} | Euclidean={euclidean:.4f} | PSI={psi:.4f}")

        class_results.append({
            "class": class_name, "step": i + 1, "alpha": alpha,
            "f1": round(step_f1, 4), "ess": round(ess, 4), "ess_pvalue": round(pvalue, 4),
            "euclidean": round(euclidean, 4), "psi": round(psi, 4),
        })

    return class_results


# ──────────────────────────────────────────────────────────────────────────
# Run across all seven classes, saving progress after each one
# ──────────────────────────────────────────────────────────────────────────
all_results = []
for class_name in ATTACK_CLASSES:
    try:
        class_results = run_for_class(class_name)
    except Exception as e:
        print(f"\n!!! {class_name} FAILED with: {e}")
        print(f"!!! Skipping {class_name} and continuing with the remaining classes.")
        class_results = []

    all_results.extend(class_results)

    # Save after every class, not just at the end, so a crash on a later
    # class does not lose the classes that already finished successfully.
    pd.DataFrame(all_results).to_csv(f"{RESULTS_DIR}/synthetic_drift_injection.csv", index=False)
    print(f"\n(Progress saved: {len(all_results)} rows across {len(set(r['class'] for r in all_results))} classes so far)")

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 60)
print("Full trajectory, all classes")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nSaved: {RESULTS_DIR}/synthetic_drift_injection.csv")

print("\n" + "=" * 60)
print("Interpretation guide")
print("=" * 60)
print("- alpha=0.0 is an unperturbed sanity check per class: ESS should start")
print("  very close to 1.0 there.")
print("- Watch for the alpha at which ESS first crosses 0.97, the actual")
print("  threshold used throughout the main experiment, and note what F1 is")
print("  doing at that same alpha. F1 still high at that point is the cleanest")
print("  possible finding: explanation drift visible before accuracy loss, on")
print("  a known class, via a realistic evasion-style shift.")
print("- Classes will very likely differ from each other. Report the actual")
print("  pattern per class rather than averaging it away or picking the")
print("  single best-looking class to feature.")
print("- Do not change alpha, PERTURB_TOP_N, or which classes are included")
print("  after seeing results, to chase a cleaner-looking curve.")