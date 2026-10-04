import numpy as np
import pandas as pd
from pathlib import Path

subjects = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb12",
    "chb13",
    "chb14",
    "chb15",
]

processed_dir = Path("data/processed")

total_windows = 0
total_seizures = 0

print("=" * 80)
print("PROCESSED DATASET VALIDATION")
print("=" * 80)

for subject in subjects:

    print(f"\n{'-' * 80}")
    print(f"Checking {subject.upper()}")
    print(f"{'-' * 80}")

    windows_file = processed_dir / f"{subject}_windows.npy"
    labels_file = processed_dir / f"{subject}_labels.npy"
    metadata_file = processed_dir / f"{subject}_metadata.csv"
    channels_file = processed_dir / f"{subject}_channel_names.txt"

    # --------------------------------------------------
    # Check files exist
    # --------------------------------------------------

    required_files = [
        windows_file,
        labels_file,
        metadata_file,
        channels_file,
    ]

    missing = [str(f) for f in required_files if not f.exists()]

    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"   {f}")
        continue

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    windows = np.load(windows_file, mmap_mode="r")
    labels = np.load(labels_file)
    metadata = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    n_windows = len(windows)
    n_labels = len(labels)
    n_metadata = len(metadata)

    print(f"Windows shape      : {windows.shape}")
    print(f"Labels shape       : {labels.shape}")
    print(f"Metadata shape     : {metadata.shape}")

    # --------------------------------------------------
    # Shape validation
    # --------------------------------------------------

    if len(windows.shape) == 3 and windows.shape[1:] == (23, 512):
        print("✅ Window shape     : (N, 23, 512)")
    else:
        print("❌ Window shape     :", windows.shape)

    # --------------------------------------------------
    # Length validation
    # --------------------------------------------------

    if n_windows == n_labels == n_metadata:
        print("✅ Lengths match")
    else:
        print("❌ Length mismatch")

    # --------------------------------------------------
    # Label validation
    # --------------------------------------------------

    unique_labels = np.unique(labels)

    print(f"Unique labels      : {unique_labels}")

    if set(unique_labels).issubset({0, 1}):
        print("✅ Labels are only 0/1")
    else:
        print("❌ Unexpected label values")

    seizure_count = int(np.sum(labels == 1))
    nonseizure_count = int(np.sum(labels == 0))

    print(f"Seizure windows    : {seizure_count}")
    print(f"Non-seizure windows: {nonseizure_count}")

    # --------------------------------------------------
    # NaN / Inf validation
    # --------------------------------------------------

    has_nan = np.isnan(windows).any()
    has_inf = np.isinf(windows).any()

    if not has_nan:
        print("✅ NaN check        : PASS")
    else:
        print("❌ NaN check        : FAIL")

    if not has_inf:
        print("✅ Inf check        : PASS")
    else:
        print("❌ Inf check        : FAIL")

    # --------------------------------------------------
    # Metadata validation
    # --------------------------------------------------

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

    if not missing_columns:
        print("✅ Metadata columns: PASS")
    else:
        print("❌ Missing metadata columns:", missing_columns)

    # --------------------------------------------------
    # Sampling rate
    # --------------------------------------------------

    if "sampling_rate" in metadata.columns:
        sampling_rates = metadata["sampling_rate"].unique()
        print(f"Sampling rates    : {sampling_rates}")

        if len(sampling_rates) == 1 and sampling_rates[0] == 128:
            print("✅ Sampling rate    : 128 Hz")
        else:
            print("❌ Unexpected sampling rate")

    # --------------------------------------------------
    # Channel count
    # --------------------------------------------------

    if "n_channels" in metadata.columns:
        channel_counts = metadata["n_channels"].unique()
        print(f"Channel counts     : {channel_counts}")

        if len(channel_counts) == 1 and channel_counts[0] == 23:
            print("✅ Channel count    : 23")
        else:
            print("❌ Unexpected channel count")

    # --------------------------------------------------
    # Samples per window
    # --------------------------------------------------

    if "n_samples" in metadata.columns:
        sample_counts = metadata["n_samples"].unique()
        print(f"Samples/window     : {sample_counts}")

        if len(sample_counts) == 1 and sample_counts[0] == 512:
            print("✅ Samples/window   : 512")
        else:
            print("❌ Unexpected sample count")

    # --------------------------------------------------
    # Subject ID
    # --------------------------------------------------

    if "subject_id" in metadata.columns:
        subject_ids = metadata["subject_id"].unique()
        print(f"Subject IDs        : {subject_ids}")

        if len(subject_ids) == 1 and subject_ids[0] == subject:
            print("✅ Subject ID       : PASS")
        else:
            print("❌ Subject ID       : CHECK")

    # --------------------------------------------------
    # Metadata label consistency
    # --------------------------------------------------

    if "label" in metadata.columns:

        metadata_labels = metadata["label"].to_numpy()

        if np.array_equal(labels, metadata_labels):
            print("✅ Labels vs metadata: MATCH")
        else:
            print("❌ Labels vs metadata: MISMATCH")

    # --------------------------------------------------
    # Channel names
    # --------------------------------------------------

    with open(channels_file, "r", encoding="utf-8") as f:
        channels = [line.strip() for line in f if line.strip()]

    print(f"Channel names      : {len(channels)}")

    if len(channels) == 23:
        print("✅ Channel names    : 23")
    else:
        print("❌ Channel names    :", len(channels))

    # --------------------------------------------------
    # Totals
    # --------------------------------------------------

    total_windows += n_windows
    total_seizures += seizure_count


# ======================================================
# FINAL SUMMARY
# ======================================================

print("\n")
print("=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

print(f"Total windows       : {total_windows}")
print(f"Total seizure windows: {total_seizures}")

if total_windows > 0:
    percentage = (total_seizures / total_windows) * 100
    print(f"Overall seizure %   : {percentage:.2f}%")

print("=" * 80)
print("Validation completed.")
print("=" * 80)