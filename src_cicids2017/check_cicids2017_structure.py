"""
check_cicids2017_structure.py

Run this once, right after extracting CIC-IDS2017, before touching
the actual preprocessing pipeline. It answers three questions we need
before adapting anything from ess_utils.py or the DDoS2019 scripts:

1. What are the actual file names and sizes for each day?
2. What are the real column names? (CIC-IDS2017 is notorious for
   leading/trailing whitespace in column headers, e.g. " Label"
   instead of "Label", which will silently break any code that
   assumes exact column name matches.)
3. What does the class distribution look like per day, so we can
   confirm or adjust the batch plan, and get the real class list
   needed to write the CIC-IDS2017 equivalent of NUM_CLASSES and
   ALL_CLASS_LABELS from ess_utils.py.

DATA_DIR is built from this script's own location on disk, not from
wherever the terminal happens to be sitting when you run it. This
means you can run it as `python check_cicids2017_structure.py` from
inside src_cicids2017, or as a full path from the repo root like you
just did, either way it finds the same folder correctly.
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw", "cicids2017")
DATA_DIR = os.path.normpath(DATA_DIR)

print(f"Looking for CSV files in: {DATA_DIR}\n")

csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

if not csv_files:
    print(f"No CSV files found in '{DATA_DIR}'.")
    print("Check that the folder exists and contains .csv files directly")
    print("(not inside a further subfolder).")
else:
    print(f"Found {len(csv_files)} CSV file(s):\n")

    all_columns = {}
    all_classes_seen = set()

    for fname in csv_files:
        path = os.path.join(DATA_DIR, fname)
        size_mb = os.path.getsize(path) / (1024 * 1024)

        # Read only the header first, fast, to check column names
        header_df = pd.read_csv(path, nrows=0)
        columns = list(header_df.columns)
        all_columns[fname] = columns

        # Find whatever the label column actually is (robust to
        # leading/trailing whitespace, which this dataset is known for)
        label_col = None
        for c in columns:
            if c.strip().lower() == "label":
                label_col = c
                break

        print("=" * 80)
        print(f"File: {fname}  ({size_mb:.1f} MB)")
        print(f"Columns ({len(columns)} total): {columns[:5]} ... {columns[-3:]}")

        if label_col is None:
            print("  WARNING: no column matching 'Label' found. Check manually.")
        else:
            print(f"  Label column found as: {repr(label_col)}")
            # Read just the label column for the class distribution,
            # avoids loading the whole file into memory
            label_series = pd.read_csv(path, usecols=[label_col])[label_col]
            counts = label_series.value_counts()
            print(f"  Class distribution ({len(label_series)} rows total):")
            for cls, n in counts.items():
                print(f"    {cls!r}: {n}")
                all_classes_seen.add(str(cls).strip())
        print()

    # Flag if column names differ across files, worth knowing before
    # trying to concatenate or process them uniformly
    unique_column_sets = set(tuple(cols) for cols in all_columns.values())
    print("=" * 80)
    if len(unique_column_sets) == 1:
        print("All files share identical column names and order. Good.")
    else:
        print(f"WARNING: files have {len(unique_column_sets)} different column layouts.")
        print("This means per-file column alignment will be needed before combining them.")

    print("=" * 80)
    print(f"All distinct classes seen across every file ({len(all_classes_seen)} total):")
    for cls in sorted(all_classes_seen):
        print(f"  {cls!r}")
    print("\nThis list is exactly what NUM_CLASSES and ALL_CLASS_LABELS need to be")
    print("set to in the CIC-IDS2017 equivalent of ess_utils.py.")