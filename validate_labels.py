import os
import numpy as np
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
print("CHB-MIT LABEL ALIGNMENT VALIDATION")
print("=" * 70)

total_windows = 0
total_seizure = 0
total_non_seizure = 0


for subject in SUBJECTS:

    print("\n" + "-" * 70)
    print(f"VALIDATING {subject.upper()}")
    print("-" * 70)

    eeg_path = os.path.join(
        DATA_DIR,
        f"{subject}_windows.npy"
    )

    fft_path = os.path.join(
        DATA_DIR,
        f"{subject}_fft_features.npy"
    )

    labels_path = os.path.join(
        DATA_DIR,
        f"{subject}_labels.npy"
    )

    metadata_path = os.path.join(
        DATA_DIR,
        f"{subject}_metadata.csv"
    )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    eeg = np.load(eeg_path, mmap_mode="r")
    fft = np.load(fft_path, mmap_mode="r")
    labels = np.load(labels_path)

    metadata = pd.read_csv(metadata_path)

    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------

    print(f"EEG windows     : {eeg.shape}")
    print(f"FFT features    : {fft.shape}")
    print(f"Labels          : {labels.shape}")
    print(f"Metadata        : {metadata.shape}")

    assert len(eeg) == len(fft), (
        "EEG and FFT row counts do not match"
    )

    assert len(eeg) == len(labels), (
        "EEG and labels row counts do not match"
    )

    assert len(eeg) == len(metadata), (
        "EEG and metadata row counts do not match"
    )

    print("[PASS] All row counts match")

    # ---------------------------------------------------------
    # Label comparison
    # ---------------------------------------------------------

    metadata_labels = metadata["label"].to_numpy()

    assert np.array_equal(
        labels,
        metadata_labels
    ), "Labels.npy and metadata labels do not match"

    print("[PASS] labels.npy matches metadata labels")

    # ---------------------------------------------------------
    # Check labels are valid
    # ---------------------------------------------------------

    unique_labels = np.unique(labels)

    print(f"Unique labels   : {unique_labels}")

    assert np.all(
        np.isin(unique_labels, [0, 1])
    ), "Unexpected label values found"

    print("[PASS] Labels contain only 0 and 1")

    # ---------------------------------------------------------
    # Count labels
    # ---------------------------------------------------------

    seizure_count = int(np.sum(labels == 1))
    non_seizure_count = int(np.sum(labels == 0))

    print(f"Non-seizure     : {non_seizure_count:,}")
    print(f"Seizure         : {seizure_count:,}")

    # ---------------------------------------------------------
    # Check FFT rows are finite
    # ---------------------------------------------------------

    nan_count = np.isnan(fft).sum()
    inf_count = np.isinf(fft).sum()

    assert nan_count == 0
    assert inf_count == 0

    print("[PASS] FFT contains no NaN / Inf")

    # ---------------------------------------------------------
    # Check window index
    # ---------------------------------------------------------

    assert "window_index" in metadata.columns

    window_indices = metadata["window_index"].to_numpy()

    # We do not require global sequential numbering because
    # preprocessing may reset the index for each EDF file.

    assert np.all(window_indices >= 0)

    print("[PASS] Window indices are valid")

    # ---------------------------------------------------------
    # Subject consistency
    # ---------------------------------------------------------

    assert metadata["subject_id"].nunique() == 1
    assert metadata["subject_id"].iloc[0] == subject

    print("[PASS] Subject ID is consistent")

    total_windows += len(labels)
    total_seizure += seizure_count
    total_non_seizure += non_seizure_count

    print(f"[PASS] {subject.upper()} label validation successful")


print("\n" + "=" * 70)
print("LABEL ALIGNMENT VALIDATION COMPLETE")
print("=" * 70)

print(f"\nTotal windows      : {total_windows:,}")
print(f"Total non-seizure  : {total_non_seizure:,}")
print(f"Total seizure      : {total_seizure:,}")

print("\nEEG ↔ FFT alignment     : PASS")
print("EEG ↔ Labels alignment  : PASS")
print("Labels ↔ Metadata       : PASS")
print("Label values            : PASS")
print("Window indices          : PASS")
print("Subject IDs             : PASS")

print("\nALL LABEL ALIGNMENT CHECKS PASSED")