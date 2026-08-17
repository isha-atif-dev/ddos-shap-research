'''
    01_data_loader.py
    
    Purpose: Load, sample, combine and inspect the raw CIC-DDoS2019 data.
    
    What this file does:
    - Loads all 7 CSV files safely using chunked reading to avoid memory crashes
    - Samples 50,000 rows from each file with a fixed seed of 42
    - Combines everything into one table of 350,000 rows and 88 columns
    - Finds the label column and shows class distribution
    - Saves the combined dataset to outputs/results/combined_dataset.csv
'''

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

# folder where the 7 raw CSV files are stored
RAW_DIR = "data/raw/03-11"

# folder where results will be saved
OUT_DIR = "outputs/results"
os.makedirs(OUT_DIR, exist_ok=True)

# the 7 CSV files to load
CSV_FILES = [
    "LDAP.csv",
    "MSSQL.csv",
    "NetBIOS.csv",
    "Portmap.csv",
    "Syn.csv",
    "UDP.csv",
    "UDPlag.csv",
]

# how many rows to keep from each file
SAMPLE_PER_FILE = 50_000

# how many rows to read at a time to avoid memory crashes
CHUNK_SIZE = 100_000

# ──────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION: chunked CSV reader and sampler
# ──────────────────────────────────────────────────────────────────────────

def sample_large_csv(fpath, n_samples, chunk_size, random_state=42):
    """
    Reads a large CSV file in chunks to avoid memory errors.
    Tries utf-8 encoding first, then latin-1 as a fallback.
    Returns a random sample of n_samples rows from the full file.
    Returns None if the file cannot be read.
    """
    chunks = []

    for encoding in ["utf-8", "latin-1"]:
        try:
            # reader is a TextFileReader object, not a dataframe yet
            # it delivers the file in batches of chunk_size rows at a time
            reader = pd.read_csv(
                fpath,
                encoding=encoding,
                low_memory=False,
                chunksize=chunk_size
            )

            # each chunk is a real dataframe of chunk_size rows
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()
                chunks.append(chunk)

            # encoding worked, no need to try the next one
            break

        except UnicodeDecodeError:
            # wrong encoding, wipe chunks and try the next one
            chunks = []
            continue

        except Exception as e:
            # something else went wrong, print and stop trying
            print(f"  Error reading chunk: {e}")
            break

    # if nothing was collected, the file could not be read
    if not chunks:
        print("  Could not read file.")
        return None

    # stick all chunks together into one complete table
    combined_file = pd.concat(chunks, ignore_index=True)

    # sample n_samples rows (or fewer if file is smaller)
    sample_num = min(n_samples, len(combined_file))
    return combined_file.sample(n=sample_num, random_state=random_state)

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: load and sample each CSV file
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and sampling CSV files ...")
print("=" * 60)

frames = []

for fname in CSV_FILES:
    fpath = os.path.join(RAW_DIR, fname)
    print(f"\nLoading {fname} ...")

    df_sample = sample_large_csv(fpath, SAMPLE_PER_FILE, CHUNK_SIZE)

    if df_sample is not None:
        print(f"  Sampled  : {len(df_sample)} rows")
        print(f"  Columns  : {list(df_sample.columns[:5])} ...")
        frames.append(df_sample)
    else:
        print(f"  Skipping {fname} due to read error.")

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: combine all sampled files into one dataset
# ──────────────────────────────────────────────────────────────────────────

print("STEP 2: Combining all files ...")
combined = pd.concat(frames, ignore_index=True)
print(f"Combined shape: {combined.shape}")

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: inspect column names
# ──────────────────────────────────────────────────────────────────────────

print("STEP 3: Inspecting column names ...")
print(list(combined.columns))

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: find and inspect the label column
# ──────────────────────────────────────────────────────────────────────────

print("STEP 4: Finding label column and checking class distribution ...")

for col in combined.columns:
    if "label" in col.lower():
        print(f"\nFound label column: '{col}'")
        print(combined[col].value_counts())

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: save combined dataset
# ──────────────────────────────────────────────────────────────────────────

print("STEP 5: Saving combined dataset ...")

combined.to_csv(f"{OUT_DIR}/combined_dataset.csv", index=False)
print(f"Saved to {OUT_DIR}/combined_dataset.csv")

