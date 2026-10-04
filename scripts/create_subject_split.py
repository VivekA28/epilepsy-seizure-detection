from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/partitions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIXED SUBJECT-WISE SPLIT
# ============================================================

TRAIN_SUBJECTS = [
    "chb01",
    "chb03",
    "chb04",
    "chb06",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb14",
    "chb15",
]

VAL_SUBJECTS = [
    "chb07",
    "chb08",
]

TEST_SUBJECTS = [
    "chb02",
    "chb05",
    "chb13",
]


SPLITS = {
    "train": TRAIN_SUBJECTS,
    "validation": VAL_SUBJECTS,
    "test": TEST_SUBJECTS,
}


# ============================================================
# LOAD METADATA
# ============================================================

metadata_files = sorted(
    DATA_DIR.glob("chb*_metadata.csv")
)

if not metadata_files:
    raise FileNotFoundError(
        f"No metadata files found in {DATA_DIR}"
    )


metadata = pd.concat(
    [pd.read_csv(f) for f in metadata_files],
    ignore_index=True
)


# ============================================================
# CHECK SUBJECTS
# ============================================================

available_subjects = set(
    metadata["subject_id"].unique()
)

split_subjects = set(
    TRAIN_SUBJECTS +
    VAL_SUBJECTS +
    TEST_SUBJECTS
)


missing_subjects = (
    split_subjects - available_subjects
)

if missing_subjects:
    raise ValueError(
        f"Subjects missing from metadata: {missing_subjects}"
    )


# Check every available subject is assigned
unassigned_subjects = (
    available_subjects - split_subjects
)

if unassigned_subjects:
    raise ValueError(
        f"Subjects not assigned to any split: "
        f"{unassigned_subjects}"
    )


# Check no subject appears in multiple splits
if (
    set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS)
    or
    set(TRAIN_SUBJECTS) & set(TEST_SUBJECTS)
    or
    set(VAL_SUBJECTS) & set(TEST_SUBJECTS)
):
    raise ValueError(
        "Subject leakage detected between splits."
    )


# ============================================================
# CREATE PARTITIONS
# ============================================================

print("=" * 90)
print("CREATING SUBJECT-WISE PARTITIONS")
print("=" * 90)


partition_summary = []


for split_name, subjects in SPLITS.items():

    split_df = metadata[
        metadata["subject_id"].isin(subjects)
    ].copy()

    output_file = (
        OUTPUT_DIR /
        f"{split_name}_metadata.csv"
    )

    split_df.to_csv(
        output_file,
        index=False
    )

    total_windows = len(split_df)

    seizure_windows = int(
        split_df["label"].sum()
    )

    non_seizure_windows = (
        total_windows - seizure_windows
    )

    seizure_ratio = (
        seizure_windows /
        total_windows *
        100
    )

    n_edfs = split_df["edf_id"].nunique()

    print("\n" + "-" * 90)

    print(f"Split              : {split_name}")
    print(f"Subjects           : {', '.join(subjects)}")
    print(f"Subjects count     : {len(subjects)}")
    print(f"EDF recordings     : {n_edfs}")
    print(f"Total windows      : {total_windows:,}")
    print(f"Seizure windows    : {seizure_windows:,}")
    print(f"Non-seizure        : {non_seizure_windows:,}")
    print(f"Seizure ratio      : {seizure_ratio:.2f}%")
    print(f"Saved to           : {output_file}")

    partition_summary.append({
        "split": split_name,
        "subjects": len(subjects),
        "edfs": n_edfs,
        "windows": total_windows,
        "seizure_windows": seizure_windows,
        "non_seizure_windows": non_seizure_windows,
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    partition_summary
)

summary_file = (
    OUTPUT_DIR /
    "partition_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL LEAKAGE CHECK
# ============================================================

print("\n")
print("=" * 90)
print("LEAKAGE CHECK")
print("=" * 90)

train_set = set(TRAIN_SUBJECTS)
val_set = set(VAL_SUBJECTS)
test_set = set(TEST_SUBJECTS)

print(
    f"\nTrain ∩ Validation : "
    f"{train_set & val_set}"
)

print(
    f"Train ∩ Test       : "
    f"{train_set & test_set}"
)

print(
    f"Validation ∩ Test  : "
    f"{val_set & test_set}"
)


if (
    not (train_set & val_set)
    and
    not (train_set & test_set)
    and
    not (val_set & test_set)
):
    print("\nSTATUS: PASS")
    print("No subject appears in more than one partition.")


# ============================================================
# FINAL TOTAL CHECK
# ============================================================

partition_total = sum(
    x["windows"]
    for x in partition_summary
)

original_total = len(metadata)

print("\n")
print("=" * 90)
print("TOTAL WINDOW CHECK")
print("=" * 90)

print(
    f"\nOriginal windows   : {original_total:,}"
)

print(
    f"Partition windows  : {partition_total:,}"
)

if partition_total == original_total:
    print("STATUS             : PASS")
else:
    print("STATUS             : FAIL")


print("\n")
print("=" * 90)
print("PARTITIONING COMPLETE")
print("=" * 90)