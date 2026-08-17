'''
    check_chapter3_numbers.py

    Purpose: Print the two real numbers needed to fill in the bracketed
    placeholders in Chapter 3 (Section 3.8.2 and Section 3.10.2), using
    files already saved by the main pipeline. This does not rerun any
    part of the pipeline; it only reads files that should already exist
    in outputs/results/ from your original run.

    What this file does:
    - Reads outputs/results/combined_dataset.csv and prints the Label
      column's value counts, for the Section 3.8.2 placeholder
    - Reads outputs/results/batch_1_y_multi.csv through
      batch_10_y_multi.csv and prints the value counts of each, for the
      Section 3.10.2 placeholder (translated back from the numeric
      label_multi encoding to the class names using the same mapping
      defined in 02_preprocessing.py)
'''

import pandas as pd

RESULTS_DIR = "outputs/results"

LABEL_MAP_MULTI = {
    0: "BENIGN", 1: "LDAP", 2: "MSSQL", 3: "NetBIOS",
    4: "Portmap", 5: "Syn", 6: "UDP", 7: "UDPLag",
}

print("=" * 60)
print("FOR SECTION 3.8.2 (Sampling Strategy)")
print("=" * 60)
try:
    combined = pd.read_csv(f"{RESULTS_DIR}/combined_dataset.csv", low_memory=False)
    label_col = next(c for c in combined.columns if "label" in c.lower())
    print(combined[label_col].value_counts())
except FileNotFoundError:
    print("combined_dataset.csv not found in outputs/results/. "
          "If you no longer have this file, you would need to rerun "
          "01_data_loader.py once to regenerate it.")

print("\n" + "=" * 60)
print("FOR SECTION 3.10.2 (Temporal Batch Splitting)")
print("=" * 60)
for batch_num in range(1, 11):
    path = f"{RESULTS_DIR}/batch_{batch_num}_y_multi.csv"
    try:
        y = pd.read_csv(path).squeeze()
        counts = y.value_counts().rename(index=LABEL_MAP_MULTI)
        print(f"Batch {batch_num} (size={len(y)}): {counts.to_dict()}")
    except FileNotFoundError:
        print(f"Batch {batch_num}: {path} not found. "
              f"If missing, you would need to rerun 02_preprocessing.py once.")