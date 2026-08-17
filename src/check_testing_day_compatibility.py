'''
    check_testing_day_compatibility.py

    Purpose: A quick, read-only check of a CIC-DDoS2019 testing-day CSV
    file, run BEFORE building any new experiment on it. This does not
    train anything or compute SHAP, it just tells you whether the file
    is even usable alongside your existing pipeline.

    Usage: update TESTING_DAY_FILE below to point at whichever testing
    day CSV you downloaded (e.g. the LDAP or NetBIOS file), then run
    this script.

    What it checks:
    - Which label(s)/attack type(s) are present in this file, and
      how many rows of each
    - Whether the file has a Timestamp column, and what date range
      it actually covers
    - How many of your 41 selected features (from
      outputs/results/selected_features.json) already exist as
      column names in this file, and which ones (if any) are missing
'''

import pandas as pd
import json

# EDIT THIS to point at whichever testing-day CSV you downloaded
TESTING_DAY_FILE = "data/raw/01-12/DrDoS_LDAP.csv"

RESULTS_DIR = "outputs/results"

print("=" * 60)
print("STEP 1: Loading selected feature list from the main pipeline ...")
print("=" * 60)

with open(f"{RESULTS_DIR}/selected_features.json") as f:
    selected_features = json.load(f)
print(f"Main pipeline uses {len(selected_features)} selected features.")

print("\n" + "=" * 60)
print(f"STEP 2: Loading testing day file: {TESTING_DAY_FILE} ...")
print("=" * 60)

df = pd.read_csv(TESTING_DAY_FILE, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Shape: {df.shape}")

print("\n" + "=" * 60)
print("STEP 3: Checking label / attack type column ...")
print("=" * 60)

label_col = None
for col in df.columns:
    if "label" in col.lower():
        label_col = col
        break

if label_col:
    print(f"Found label column: '{label_col}'")
    print(df[label_col].value_counts())
else:
    print("No column with 'label' in its name was found. Check column names manually:")
    print(list(df.columns))

print("\n" + "=" * 60)
print("STEP 4: Checking date range (if a Timestamp column exists) ...")
print("=" * 60)

if "Timestamp" in df.columns:
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    print(f"Earliest: {ts.min()}")
    print(f"Latest  : {ts.max()}")
else:
    print("No 'Timestamp' column found. Check column names manually if timing matters.")

print("\n" + "=" * 60)
print("STEP 5: Checking overlap with the 41 selected features ...")
print("=" * 60)

present = [f for f in selected_features if f in df.columns]
missing = [f for f in selected_features if f not in df.columns]

print(f"Selected features present in this file : {len(present)} / {len(selected_features)}")
if missing:
    print(f"Selected features NOT found in this file: {missing}")
else:
    print("All selected features are present. This file is compatible with the existing pipeline.")