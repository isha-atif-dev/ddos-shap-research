'''
    15_model_signal_comparison.py

    Purpose: Compare how ESS, PSI and F1 respond to the controlled
    synthetic drift under XGBoost and LightGBM, using the same fixed
    monitoring thresholds as the main experiment for both models.

    Rule (decided before looking at the comparison):
    - a signal "fires" at the first alpha > 0 where it crosses its threshold
        ESS < 0.97, PSI > 0.1, F1 < 0.75
    - alpha = 0 is excluded because it is the unperturbed sanity check
    - "earliest" = the signal(s) that fire at the smallest alpha
    - "ESS unique" = ESS fires at an alpha where neither PSI nor F1 has fired yet

    Inputs:
    - outputs/results/synthetic_drift_injection.csv            (XGBoost)
    - outputs/results/synthetic_drift_injection_lightgbm.csv   (LightGBM)

    Output:
    - outputs/results/model_signal_comparison.csv, one row per model and class
'''

import numpy as np
import pandas as pd

RESULTS_DIR = "outputs/results"
THRESHOLDS = {
    "ESS": ("ess", lambda s: s < 0.97),
    "PSI": ("psi", lambda s: s > 0.1),
    "F1": ("f1", lambda s: s < 0.75),
}


def first_firing_alpha(group, column, crosses):
    '''Returns the first alpha > 0 where this signal crosses its threshold, or NaN.'''
    fired = group.loc[crosses(group[column]) & (group["alpha"] > 0), "alpha"]
    return fired.min() if len(fired) else np.nan


def summarise(group):
    '''Builds one comparison row for one model and one class.'''
    firsts = {name: first_firing_alpha(group, col, rule) for name, (col, rule) in THRESHOLDS.items()}
    fired = {k: v for k, v in firsts.items() if not np.isnan(v)}
    earliest = min(fired.values()) if fired else np.nan
    others = [firsts["PSI"], firsts["F1"]]
    ess_unique = (not np.isnan(firsts["ESS"])) and all(
        np.isnan(o) or o > firsts["ESS"] for o in others
    )
    return pd.Series({
        "ess_first_alpha": firsts["ESS"],
        "psi_first_alpha": firsts["PSI"],
        "f1_first_alpha": firsts["F1"],
        "earliest_signal": ", ".join(k for k, v in fired.items() if v == earliest) if fired else "none",
        "ess_unique": ess_unique,
        "ess_min": group["ess"].min(),
        "psi_max": group["psi"].max(),
        "f1_min": group["f1"].min(),
    })


if __name__ == "__main__":
    xgb = pd.read_csv(f"{RESULTS_DIR}/synthetic_drift_injection.csv")
    xgb["model"] = "XGBoost"
    lgb = pd.read_csv(f"{RESULTS_DIR}/synthetic_drift_injection_lightgbm.csv")
    both = pd.concat([xgb, lgb[xgb.columns]], ignore_index=True)

    table = both.groupby(["model", "class"]).apply(summarise, include_groups=False).reset_index()
    table.to_csv(f"{RESULTS_DIR}/model_signal_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR}/model_signal_comparison.csv")