# Explanation Stability Score (ESS): Monitoring Explanation Drift in Deployed DDoS Detection Models

This repository contains the full experimental pipeline for the journal manuscript *"Monitoring Explanation Drift in Deployed DDoS Detection Models through SHAP Feature-Ranking Stability"* and the MRes dissertation it extends, *"Detecting Explanation Drift in DDoS Models: A SHAP Stability Score Framework"* (University of Greater Manchester, 2026).

The work proposes the **Explanation Stability Score (ESS)**, the Spearman rank correlation between a reference and a current SHAP feature-importance ranking, as a label-free signal for monitoring deployed tree-based DDoS detectors and triggering retraining. ESS is evaluated alongside accuracy-based and PSI-based monitoring on CIC-DDoS2019, using XGBoost and LightGBM.

## Repository Structure

```
src/
├── ess_utils.py                        # Shared utilities: ESS, PSI and Euclidean computation, training, evaluation
├── 01_data_loader.py                   # Loads and samples the raw CIC-DDoS2019 CSVs
├── 02_preprocessing.py                 # Cleaning, label encoding, batch splitting, feature selection (Batch 1 only)
├── 03_train_baseline.py                # Trains the baseline XGBoost model and computes the baseline SHAP ranking
├── 03_imputation_leakage_check.py      # Diagnostic: effect of full-sample median imputation
├── 04_ess_engine.py                    # Standalone no-retraining ESS check (see note below)
├── 05_experiment_main.py               # Temporal experiment: four retraining conditions across nine deployment batches
├── 06_independence_experiment.py       # Signal relationship analysis: correlations between ESS, F1 and PSI
├── 07_threshold_sweep.py               # Threshold sensitivity, tau = {0.90, 0.95, 0.97, 0.99}
├── 08_multi_seed.py                    # Repeats the temporal experiment across seeds 42, 123 and 7
├── 09_plots.py                         # Generates figures from saved results
├── 10_input_psi_check.py               # PSI recomputed on raw input features rather than SHAP values
├── 11_ldap_drift_check.py              # Real-date comparison (3 Nov vs 1 Dec 2018) on a known class
├── 12_synthetic_drift_injection.py     # Controlled evasion-style drift injection, seven attack classes (XGBoost)
├── 13_shap_concentration_check.py      # Baseline SHAP concentration per class (exploratory)
├── 14_synthetic_drift_lightgbm.py      # Same drift injection with LightGBM under matched settings
├── 15_model_signal_comparison.py       # First alpha at which ESS, PSI and F1 cross their thresholds, both models
├── check_lgb_raw_scores.py             # Diagnostic: checks LightGBM raw scores stay in a normal range
├── check_chapter3_numbers.py           # Diagnostic: verifies reported dataset and batch figures
└── check_testing_day_compatibility.py  # Diagnostic: checks feature overlap with the second capture day

outputs/
├── results/     # CSV and JSON outputs from every script above
├── models/      # Saved baseline model and SHAP explainer
└── plots/       # Generated figures
```

## Setup

```bash
pip install xgboost lightgbm shap scikit-learn pandas numpy scipy matplotlib seaborn
```

The experiments were run with Python 3.14 on Windows 11 (Intel Core i5-8250U, 16 GB RAM).

The CIC-DDoS2019 dataset (Canadian Institute for Cybersecurity) is not included because of its size. Place the 3 November 2018 CSV files under `data/raw/03-11/`. The real-date check (`11_ldap_drift_check.py`) also needs the 1 December 2018 LDAP file under `data/raw/01-12/`.

## Running the Pipeline

Run the scripts from the repository root, in this order. Each depends on outputs from earlier ones.

```bash
python src/01_data_loader.py
python src/02_preprocessing.py
python src/03_train_baseline.py
python src/05_experiment_main.py
python src/06_independence_experiment.py
python src/07_threshold_sweep.py
python src/08_multi_seed.py
python src/09_plots.py
python src/10_input_psi_check.py
python src/11_ldap_drift_check.py
python src/12_synthetic_drift_injection.py
python src/13_shap_concentration_check.py
python src/14_synthetic_drift_lightgbm.py
python src/15_model_signal_comparison.py
```

**Note on `04_ess_engine.py`:** computes ESS under no retraining only. Its output matches the "No Retraining" rows of `05_experiment_main.py`. It is kept as a standalone module and is not a dependency for any other script.

**Note on runtime:** `08_multi_seed.py` reruns the full four-condition experiment once per seed, including a SHAP pass per batch per condition, and can take a long time on a laptop CPU.

**Note on memory:** the raw capture files are large. `14_synthetic_drift_lightgbm.py` reads only the 41 selected features and the label column, which reduces memory use without changing which rows are sampled.

## Core Methodology

```
ESS(t) = Spearman(R_ref, R_t)
```

`R_ref` is the reference SHAP feature-importance ranking and `R_t` is the ranking for deployment batch `t`. Under no retraining, `R_ref` is the Batch 1 baseline ranking. Under the retraining conditions, `R_ref` is recomputed from the retrained model on the accumulated training pool after each retraining event. Drift is flagged when ESS falls below 0.97.

**Temporal experiment:** four conditions (no retraining, ESS-triggered, accuracy-triggered with F1 < 0.75, PSI-triggered with PSI > 0.1) across nine chronological deployment batches, with cumulative retraining.

**Controlled drift injection:** for each attack class, the five highest-ranked continuous features of held-out attack traffic are moved a fraction alpha (0.0 to 1.0) of the way towards the benign mean, with the detector never retrained. This is run with XGBoost (script 12) and with LightGBM (script 14). LightGBM uses 100 rounds, maximum depth 6, learning rate 0.1, 63 leaves, and minimum leaf Hessian and L2 regularisation of 1, matching the corresponding XGBoost settings. Neither model is tuned.

## Key Outputs

| File | Content |
|---|---|
| `experiment_results.csv` | Temporal experiment, four retraining conditions |
| `ess_scores.csv` | ESS, Euclidean distance and PSI under no retraining |
| `independence_correlations.csv` | Correlations between ESS, F1 and PSI |
| `threshold_sweep.csv` | Threshold sensitivity |
| `multi_seed_summary.csv`, `multi_seed_raw_results.csv` | Robustness across seeds |
| `input_psi_check.csv` | PSI on raw input features |
| `ldap_drift_check.csv` | Real-date comparison |
| `synthetic_drift_injection.csv` | Drift injection, XGBoost |
| `synthetic_drift_injection_lightgbm.csv` | Drift injection, LightGBM |
| `model_signal_comparison.csv` | Threshold crossings for both models |
| `shap_concentration_check.csv` | Baseline SHAP concentration (exploratory) |
| `selected_features.json`, `feature_names.json` | The 41 retained features |

## Citation

If referencing this work, please cite the dissertation:

```
Atif, I. (2026). Detecting Explanation Drift in DDoS Models: A SHAP Stability
Score Framework. MRes Dissertation, University of Greater Manchester.
```

A journal version is currently in preparation.