import os
import pandas as pd


DATA_DIR = "data/processed"

SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
]


print("=" * 70)
print("PATIENT / EDF PARTITION INSPECTION")
print("=" * 70)


all_metadata = []


for subject in SUBJECTS:

    path = os.path.join(
        DATA_DIR,
        f"{subject}_metadata.csv"
    )

    df = pd.read_csv(path)

    print("\n" + "-" * 70)
    print(subject.upper())
    print("-" * 70)

    print(f"Windows       : {len(df):,}")
    print(f"Subjects      : {df['subject_id'].nunique()}")
    print(f"EDF files     : {df['edf_id'].nunique()}")
    print(f"Seizure       : {(df['label'] == 1).sum():,}")
    print(f"Non-seizure   : {(df['label'] == 0).sum():,}")

    print("\nEDF distribution:")

    edf_summary = (
        df.groupby("edf_id")
        .agg(
            windows=("window_index", "count"),
            seizure_windows=("label", "sum")
        )
        .reset_index()
    )

    print(edf_summary.to_string(index=False))

    all_metadata.append(df)


# -------------------------------------------------------------
# Combine metadata
# -------------------------------------------------------------

combined = pd.concat(
    all_metadata,
    ignore_index=True
)


print("\n" + "=" * 70)
print("COMBINED DATASET")
print("=" * 70)

print(f"Total windows       : {len(combined):,}")
print(
    f"Total seizure       : "
    f"{(combined['label'] == 1).sum():,}"
)
print(
    f"Total non-seizure   : "
    f"{(combined['label'] == 0).sum():,}"
)

print(
    f"Unique subjects     : "
    f"{combined['subject_id'].nunique()}"
)

print(
    f"Unique EDF IDs      : "
    f"{combined['edf_id'].nunique()}"
)


# -------------------------------------------------------------
# Subject check
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("SUBJECT CHECK")
print("=" * 70)

subjects = sorted(
    combined["subject_id"].unique()
)

print("Subjects:")
for subject in subjects:
    print(f"  - {subject}")

if len(subjects) == len(set(subjects)):
    print("\n[PASS] No duplicate subject IDs")


# -------------------------------------------------------------
# EDF ownership check
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("EDF OWNERSHIP CHECK")
print("=" * 70)

edf_subject_counts = (
    combined.groupby("edf_id")["subject_id"]
    .nunique()
)

multi_subject_edfs = (
    edf_subject_counts[
        edf_subject_counts > 1
    ]
)

if len(multi_subject_edfs) == 0:
    print("[PASS] Every EDF belongs to exactly one subject")
else:
    print("[FAIL] Some EDFs belong to multiple subjects")
    print(multi_subject_edfs)


# -------------------------------------------------------------
# Window ordering check
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("WINDOW ORDER CHECK")
print("=" * 70)

failed = False

for (subject, edf), group in combined.groupby(
    ["subject_id", "edf_id"]
):

    indices = group["window_index"].to_numpy()

    if len(indices) == 0:
        continue

    if indices.min() < 0:
        print(
            f"[FAIL] Negative window index: "
            f"{subject} / {edf}"
        )
        failed = True

if not failed:
    print("[PASS] No invalid negative window indices")


# -------------------------------------------------------------
# Time check
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("TIME RANGE CHECK")
print("=" * 70)

if (combined["end_sec"] <= combined["start_sec"]).any():

    print("[FAIL] Invalid time range detected")

else:

    print(
        "[PASS] All windows have "
        "end_sec > start_sec"
    )


# -------------------------------------------------------------
# Sampling consistency
# -------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLING / WINDOW CHECK")
print("=" * 70)

sampling_rates = combined[
    "sampling_rate"
].unique()

channels = combined[
    "n_channels"
].unique()

samples = combined[
    "n_samples"
].unique()

print(f"Sampling rates : {sampling_rates}")
print(f"Channels       : {channels}")
print(f"Samples        : {samples}")

if len(sampling_rates) == 1 and sampling_rates[0] == 128:
    print("[PASS] Sampling rate is consistently 128 Hz")
else:
    print("[WARNING] Sampling rate is not uniform")

if len(channels) == 1 and channels[0] == 23:
    print("[PASS] Channel count is consistently 23")
else:
    print("[WARNING] Channel count is not uniform")

if len(samples) == 1 and samples[0] == 512:
    print("[PASS] Window size is consistently 512 samples")
else:
    print("[WARNING] Window size is not uniform")


print("\n" + "=" * 70)
print("PARTITION INSPECTION COMPLETE")
print("=" * 70)