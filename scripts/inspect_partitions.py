from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")


# ============================================================
# FIND CHB-MIT METADATA FILES
# ============================================================

metadata_files = sorted(DATA_DIR.glob("chb*_metadata.csv"))

if not metadata_files:
    print("\nERROR: No CHB metadata files found.")
    print(f"Expected files like: {DATA_DIR / 'chb01_metadata.csv'}")
    raise SystemExit(1)


print("=" * 90)
print("CHB-MIT PARTITION INSPECTION")
print("=" * 90)

print(f"\nMetadata files found: {len(metadata_files)}")

for file in metadata_files:
    print(f"  {file.name}")


# ============================================================
# LOAD ALL METADATA
# ============================================================

all_metadata = []

for file in metadata_files:

    df = pd.read_csv(file)

    if df.empty:
        print(f"\nWARNING: {file.name} is empty.")
        continue

    all_metadata.append(df)


if not all_metadata:
    print("\nERROR: Metadata files were found but contain no data.")
    raise SystemExit(1)


metadata = pd.concat(
    all_metadata,
    ignore_index=True
)


# ============================================================
# BASIC COLUMN CHECK
# ============================================================

required_columns = [
    "dataset_id",
    "subject_id",
    "edf_id",
    "window_index",
    "start_sec",
    "end_sec",
    "label",
    "sampling_rate",
    "n_channels",
    "n_samples",
]

missing_columns = [
    col for col in required_columns
    if col not in metadata.columns
]

if missing_columns:

    print("\nERROR: Missing metadata columns:")
    for col in missing_columns:
        print(f"  - {col}")

    print("\nAvailable columns:")
    print(list(metadata.columns))

    raise SystemExit(1)


# ============================================================
# SUBJECT SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("SUBJECT-WISE SUMMARY")
print("=" * 90)

summary_rows = []

for subject in sorted(metadata["subject_id"].unique()):

    subject_df = metadata[
        metadata["subject_id"] == subject
    ]

    total_windows = len(subject_df)

    seizure_windows = int(
        subject_df["label"].sum()
    )

    non_seizure_windows = (
        total_windows - seizure_windows
    )

    seizure_ratio = (
        seizure_windows / total_windows * 100
        if total_windows > 0
        else 0
    )

    n_edfs = subject_df["edf_id"].nunique()

    summary_rows.append({
        "subject": subject,
        "windows": total_windows,
        "seizure": seizure_windows,
        "non_seizure": non_seizure_windows,
        "seizure_%": seizure_ratio,
        "EDFs": n_edfs,
    })


summary = pd.DataFrame(summary_rows)


print(
    f"\n{'Subject':<10}"
    f"{'Windows':>12}"
    f"{'Seizure':>12}"
    f"{'Non-Seizure':>14}"
    f"{'Seizure %':>12}"
    f"{'EDFs':>8}"
)

print("-" * 90)

for _, row in summary.iterrows():

    print(
        f"{row['subject']:<10}"
        f"{row['windows']:>12,}"
        f"{row['seizure']:>12,}"
        f"{row['non_seizure']:>14,}"
        f"{row['seizure_%']:>11.2f}%"
        f"{row['EDFs']:>8}"
    )


# ============================================================
# TOTALS
# ============================================================

total_windows = len(metadata)

total_seizure = int(
    metadata["label"].sum()
)

total_non_seizure = (
    total_windows - total_seizure
)

total_subjects = (
    metadata["subject_id"].nunique()
)

total_edfs = (
    metadata["edf_id"].nunique()
)

seizure_ratio = (
    total_seizure / total_windows * 100
)


print("\n")
print("=" * 90)
print("TOTAL DATASET")
print("=" * 90)

print(f"\nSubjects          : {total_subjects}")
print(f"EDF recordings    : {total_edfs}")
print(f"Total windows     : {total_windows:,}")
print(f"Seizure windows   : {total_seizure:,}")
print(f"Non-seizure       : {total_non_seizure:,}")
print(f"Seizure ratio     : {seizure_ratio:.2f}%")


# ============================================================
# CHECK DUPLICATE WINDOWS
# ============================================================

print("\n")
print("=" * 90)
print("DUPLICATE CHECK")
print("=" * 90)

duplicate_columns = [
    "subject_id",
    "edf_id",
    "window_index",
]

duplicates = metadata.duplicated(
    subset=duplicate_columns
).sum()

print(
    f"\nDuplicate window records : {duplicates}"
)

if duplicates == 0:
    print("Status                   : PASS")
else:
    print("Status                   : WARNING")


# ============================================================
# CHECK SUBJECT / EDF CONSISTENCY
# ============================================================

print("\n")
print("=" * 90)
print("SUBJECT / EDF CONSISTENCY CHECK")
print("=" * 90)

edf_subject_counts = (
    metadata
    .groupby("edf_id")["subject_id"]
    .nunique()
)

bad_edfs = edf_subject_counts[
    edf_subject_counts > 1
]

print(
    f"\nEDFs belonging to multiple subjects : "
    f"{len(bad_edfs)}"
)

if len(bad_edfs) == 0:
    print("Status                             : PASS")
else:
    print("Status                             : FAIL")

    print("\nProblematic EDFs:")
    print(bad_edfs)


# ============================================================
# CHECK LABEL VALUES
# ============================================================

print("\n")
print("=" * 90)
print("LABEL CHECK")
print("=" * 90)

unique_labels = sorted(
    metadata["label"].dropna().unique()
)

print(
    f"\nUnique labels : {unique_labels}"
)

if set(unique_labels).issubset({0, 1}):
    print("Status        : PASS")
else:
    print("Status        : FAIL")


# ============================================================
# CHECK SAMPLING RATE / SHAPE
# ============================================================

print("\n")
print("=" * 90)
print("DATA SHAPE CHECK")
print("=" * 90)

print(
    "\nSampling rates:"
)

print(
    metadata["sampling_rate"]
    .value_counts()
    .sort_index()
)

print(
    "\nNumber of channels:"
)

print(
    metadata["n_channels"]
    .value_counts()
    .sort_index()
)

print(
    "\nNumber of samples per window:"
)

print(
    metadata["n_samples"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CHECK FOR NaN / INF IN METADATA
# ============================================================

print("\n")
print("=" * 90)
print("METADATA QUALITY CHECK")
print("=" * 90)

numeric_columns = [
    "window_index",
    "start_sec",
    "end_sec",
    "label",
    "sampling_rate",
    "n_channels",
    "n_samples",
]

nan_count = int(
    metadata[numeric_columns]
    .isna()
    .sum()
    .sum()
)

inf_count = 0

for col in numeric_columns:

    values = pd.to_numeric(
        metadata[col],
        errors="coerce"
    )

    inf_count += int(
        np.isinf(values).sum()
    )


print(f"\nNaN values  : {nan_count}")
print(f"Inf values  : {inf_count}")

if nan_count == 0 and inf_count == 0:
    print("Status      : PASS")
else:
    print("Status      : WARNING")


# ============================================================
# IMPORTANT WARNING
# ============================================================

print("\n")
print("=" * 90)
print("PARTITIONING RULE")
print("=" * 90)

print("""
DO NOT randomly split individual EEG windows.

The final train/validation/test split must be SUBJECT-WISE.

That means:

    A subject can belong to only ONE partition.

Example:

    Train       -> some complete subjects
    Validation  -> different complete subjects
    Test        -> completely unseen subjects

This prevents patient-level data leakage.
""")


# ============================================================
# FINISHED
# ============================================================

print("=" * 90)
print("INSPECTION COMPLETE")
print("=" * 90)