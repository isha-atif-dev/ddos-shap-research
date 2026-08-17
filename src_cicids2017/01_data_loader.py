"""
01_load_and_batch.py

Step 1 of the CIC-IDS2017 validation pipeline: load the raw CSVs and
organise them into the baseline period and six deployment batches
agreed on after inspecting the real data.

WHY MONDAY + TUESDAY AS THE BASELINE
-------------------------------------
Monday alone is 100% BENIGN traffic, a model trained only on Monday
would never see an attack and couldn't learn anything to distinguish.
Tuesday adds FTP-Patator and SSH-Patator, giving the baseline model
real class diversity, genuinely parallel to the DDoS2019 study's
two-class baseline (Portmap + BENIGN).

WHY THE ENCODING FIX
---------------------
The original CIC-IDS2017 files were saved in Windows-1252 encoding,
not UTF-8. Read the normal way, the "Web Attack" class names come out
corrupted (a broken character in place of the en-dash). Reading with
encoding="cp1252" fixes this at the source, rather than trying to
patch broken strings after the fact.

WHY COLUMNS ARE STRIPPED
--------------------------
Every column in this dataset has a leading space (' Label', not
'Label'; ' Destination Port', not 'Destination Port'), confirmed
identical across all 8 files by the earlier inspection script. This
strips that whitespace once, here, so nothing downstream has to deal
with it.

OUTPUT
------
One CSV per batch, saved to data/processed/cicids2017/, named
batch_00_baseline.csv through batch_06_friday_ddos.csv, ready for the
next script (02_preprocessing.py) to clean and encode.
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "raw", "cicids2017"))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "processed", "cicids2017"))

os.makedirs(OUT_DIR, exist_ok=True)

# Maps each output batch to the raw file(s) that make it up, in the
# chronological order agreed on. Baseline combines two files; every
# deployment batch is a single file.
BATCH_PLAN = {
    "batch_00_baseline": [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ],
    "batch_01_wednesday": [
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "batch_02_thursday_webattacks": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ],
    "batch_03_thursday_infiltration": [
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "batch_04_friday_bot": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ],
    "batch_05_friday_portscan": [
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ],
    "batch_06_friday_ddos": [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
}


def load_raw_csv(filename):
    """Loads one raw CIC-IDS2017 file with the correct encoding and
    stripped column names."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    print(f"Reading raw files from: {RAW_DIR}")
    print(f"Writing batch files to: {OUT_DIR}\n")

    summary_rows = []

    for batch_name, source_files in BATCH_PLAN.items():
        pieces = [load_raw_csv(f) for f in source_files]
        batch_df = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]

        out_path = os.path.join(OUT_DIR, f"{batch_name}.csv")
        batch_df.to_csv(out_path, index=False)

        class_counts = batch_df["Label"].value_counts().to_dict()
        summary_rows.append({
            "batch": batch_name,
            "source_files": ", ".join(source_files),
            "total_rows": len(batch_df),
            "classes": class_counts,
        })

        print("=" * 80)
        print(f"{batch_name}  ->  {out_path}")
        print(f"  Source file(s): {', '.join(source_files)}")
        print(f"  Total rows: {len(batch_df)}")
        print(f"  Class distribution: {class_counts}")

    print("\n" + "=" * 80)
    print(f"Done. {len(BATCH_PLAN)} batch files written to {OUT_DIR}")
    print("\nNext step: 02_preprocessing.py will clean and encode these batches")
    print("(identifier column removal, missing value imputation, label encoding),")
    print("the same steps used in the original DDoS2019 pipeline.")


if __name__ == "__main__":
    main()