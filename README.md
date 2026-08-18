# Explanation Stability Score (ESS): Monitoring Explanation Drift in Deployed DDoS Detection Models

This repository contains the full experimental pipeline for the MRes dissertation *"Detecting Explanation Drift in DDoS Models: A SHAP Stability Score Framework"* (Isha Atif University of Greater Manchester).

The dissertation proposes the **Explanation Stability Score (ESS)**, a Spearman rank correlation between a baseline and deployment-batch SHAP feature importance ranking, and tests whether this can serve as an independent, complementary signal for monitoring deployed XGBoost-based DDoS classifiers, alongside accuracy and Population Stability Index (PSI) monitoring.

## Repository Structure

```
src/
├── ess_utils.py                      # Shared utilities: ESS/PSI/Euclidean computation, model training, evaluation
├── 01_data_loader.py                 # Loads and batches raw CIC-DDoS2019 CSVs
├── 02_preprocessing.py               # Cleaning, label encoding, two-stage feature selection
├── 03_train_baseline.py              # Trains the baseline XGBoost model, computes baseline SHAP ranking
├── 04_ess_engine.py                  # Standalone no-retraining ESS check (see note below)
├── 05_experiment_main.py             # Core experiment: four retraining conditions across all deployment batches
├── 06_independence_experiment.py     # Formal independence analysis (RQ1, RQ3): correlates ESS, F1, PSI
├── 07_threshold_sweep.py             # Threshold sensitivity check across tau = {0.90, 0.95, 0.97, 0.99}
├── 08_multi_seed.py                  # Repeats the full experiment across multiple random seeds
├── 09_plots.py                       # Generates all dissertation figures from saved results
├── 10_input_psi_check.py             # PSI recomputed on raw input features rather than SHAP values
├── 11_ldap_drift_check.py            # Real-date comparison (Nov vs Dec) on a known class
├── 12_synthetic_drift_injection.py   # Controlled synthetic drift check, all seven attack classes
├── 13_shap_concentration_check.py    # Baseline SHAP concentration per class (explains RQ5's mixed result)
├── check_chapter3_numbers.py         # Diagnostic: verifies reported dataset/split figures against raw data
└── check_testing_day_compatibility.py # Diagnostic: verifies class overlap across collection days

outputs/
├── results/     # CSV/JSON outputs from every script above
├── models/      # Saved baseline model and SHAP explainer (pickled)
└── plots/       # All generated figures (figure_4_1.png through figure_4_7.png, etc.)
```

## Setup

```bash
pip install xgboost shap scikit-learn pandas numpy scipy matplotlib
```

Requires the CIC-DDoS2019 dataset (Canadian Institute for Cybersecurity), placed under `data/raw/` before running `01_data_loader.py`. The dataset is publicly available and was used under its original terms of release; see the dissertation's Ethical Considerations section (Chapter 3) for details on data governance.

## Running the Pipeline

Scripts are numbered in the order they must be run, each depends on outputs from the previous ones:

```bash
python 01_data_loader.py
python 02_preprocessing.py
python 03_train_baseline.py
python 05_experiment_main.py
python 06_independence_experiment.py
python 07_threshold_sweep.py
python 08_multi_seed.py
python 09_plots.py
python 10_input_psi_check.py
python 11_ldap_drift_check.py
python 12_synthetic_drift_injection.py
python 13_shap_concentration_check.py
```

**Note on `04_ess_engine.py`**: this is a standalone script computing ESS under no retraining only. Its output is intentionally identical to the "No Retraining" rows already produced by `05_experiment_main.py`, it exists as a documented, standalone module in Chapter 3's methodology, not because it produces new results. It can be run independently but is not a dependency for any other script.

**Note on runtime**: `08_multi_seed.py` re-runs the full four-condition experiment once per random seed, including a SHAP TreeExplainer pass per batch per condition. This can take a considerable amount of time on a standard laptop CPU; start with a small number of seeds and time one full run before extending further.

## Core Methodology

The Explanation Stability Score is defined as:

```
ESS(t) = Spearman(R_baseline, R_t)
```

where `R_baseline` is the SHAP feature importance ranking computed at the end of the baseline training period, and `R_t` is the same ranking recomputed on deployment batch `t`. Full mathematical definition and justification given in Chapter 3, Section 3.6.4 of the dissertation.

Four retraining conditions are compared across nine deployment batches: No Retraining, ESS-Triggered, Accuracy-Triggered, and PSI-Triggered, with retraining conducted on the cumulative pool of all batches seen up to that point once a condition's trigger threshold is crossed.

## Key Outputs Referenced in the Dissertation

| File | Used in |
|---|---|
| `experiment_results.csv` | Chapter 4, main four-condition comparison (Table 6) |
| `independence_correlations.csv` | Chapter 4, Section 4.4, signal independence analysis |
| `threshold_sweep.csv` | Chapter 4, Section 4.8, threshold sensitivity |
| `multi_seed_summary.csv` / `multi_seed_raw_results.csv` | Chapter 4, Section 4.9; Appendix D |
| `synthetic_drift_injection.csv` | Chapter 4, Section 4.5; Chapter 5, Section 5.8; Appendix C |
| `input_psi_check.csv` | Chapter 5, Section 5.5 |
| `ldap_drift_check.csv` | Chapter 4, real-date comparison |
| `shap_concentration_check.csv` | Chapter 5, Section 5.8, concentration hypothesis |
| `selected_features.json` / `feature_names.json` | Appendix A |

## Citation

If referencing this work, please cite the dissertation:

```
Atif, I. (2026). Detecting Explanation Drift in DDoS Models: A SHAP Stability
Score Framework. MRes Dissertation, University of Greater Manchester.
```

