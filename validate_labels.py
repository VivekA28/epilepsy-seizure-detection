"""
Validation of Window, FFT Feature, and Label Alignment for CHB-MIT EEG.

Ensures strict 1-to-1 index synchronization between preprocessed EEG windows,
extracted FFT feature matrices, binary labels, and metadata tables.

In a multimodal hybrid pipeline (e.g. 2D-CNN temporal feature extractor fused
with 1D-FFT log band powers and LSTM sequence models), every index `i` across all
representations must correspond to the exact same 4.0-second time interval.
Any row misalignment would corrupt training supervision and evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Directory containing preprocessed data artifacts
DATA_DIR = Path("data/processed")
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def validate_subject_labels(subject: str) -> dict:
    """
    Verify alignment across modalities and label representations for a single subject.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'chb01').

    Returns
    -------
    dict
        Dictionary containing counts for total windows, seizure windows,
        and non-seizure baseline windows.

    Raises
    ------
    AssertionError
        If row counts diverge between representations or label values mismatch.
    """
    # Load arrays in memory-mapped read-only mode to prevent RAM bloat
    eeg = np.load(DATA_DIR / f"{subject}_windows.npy", mmap_mode="r")
    fft = np.load(DATA_DIR / f"{subject}_fft_features.npy", mmap_mode="r")
    labels = np.load(DATA_DIR / f"{subject}_labels.npy")
    metadata = pd.read_csv(DATA_DIR / f"{subject}_metadata.csv")

    n = len(eeg)

    # Step 1: Verify row-dimension equality across all four artifacts
    assert len(fft) == n, f"{subject}: FFT rows ({len(fft)}) != EEG rows ({n})"
    assert len(labels) == n, f"{subject}: Labels count ({len(labels)}) != EEG rows ({n})"
    assert len(metadata) == n, f"{subject}: Metadata count ({len(metadata)}) != EEG rows ({n})"

    # Step 2: Verify element-wise identity between standalone labels.npy and metadata['label']
    meta_labels = metadata["label"].to_numpy()
    assert np.array_equal(labels, meta_labels), f"{subject}: labels.npy != metadata labels"

    # Step 3: Verify binary domain constraint (only {0, 1} classes permitted)
    unique_vals = set(np.unique(labels))
    assert unique_vals.issubset({0, 1}), f"{subject}: Unexpected label values: {unique_vals}"

    seizures = int(labels.sum())
    non_seizures = n - seizures

    print(
        f"  [PASS] {subject.upper()}: {n:,} windows | {seizures} seizure ({100*seizures/n:.2f}%) "
        f"| Row & label synchronization verified."
    )
    return {"windows": n, "seizures": seizures, "non_seizures": non_seizures}


def main():
    """Execute cross-modal synchronization checks across all target subjects."""
    print("=" * 60)
    print("CHB-MIT MULTI-MODALITY ALIGNMENT VALIDATION")
    print("=" * 60)

    stats = [validate_subject_labels(s) for s in SUBJECTS]
    total_w = sum(x["windows"] for x in stats)
    total_s = sum(x["seizures"] for x in stats)

    print("=" * 60)
    print(f"ALL ALIGNMENT CHECKS PASSED: {total_w:,} windows, {total_s} seizures ({100*total_s/total_w:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()