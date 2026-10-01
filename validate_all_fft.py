"""
Integration Validation for CHB-MIT FFT Datasets Across Subjects chb01–chb05.

Validates that all precomputed FFT feature matrices:
  1. Exist on disk alongside their respective metadata and channel definitions.
  2. Have consistent 2D shape (N, 115) corresponding to N windows and 115 features.
  3. Strictly match the row counts of preprocessed metadata tables.
  4. Contain exactly 23 electrode channel names and 115 feature identifiers.
  5. Contain zero NaN or infinite values across all windows and subjects.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Processed dataset location and invariant configurations
DATA_DIR = Path("data/processed")
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]
EXPECTED_FEATURES = 115  # 23 channels × 5 clinical bands
EXPECTED_CHANNELS = 23   # Bipolar 10-20 montage channels


def validate_subject_fft(subject: str) -> int:
    """
    Validate FFT dataset integrity and alignment for a single subject.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'chb01').

    Returns
    -------
    int
        Total number of validated EEG windows for this subject.

    Raises
    ------
    FileNotFoundError
        If any required data artifact is missing from disk.
    AssertionError
        If dimensional, row-count, or numerical integrity checks fail.
    """
    fft_path = DATA_DIR / f"{subject}_fft_features.npy"
    meta_path = DATA_DIR / f"{subject}_metadata.csv"
    ch_path = DATA_DIR / f"{subject}_channel_names.txt"
    names_path = DATA_DIR / f"{subject}_fft_feature_names.txt"

    # Step 1: Ensure all four required pipeline artifacts are present
    for p in [fft_path, meta_path, ch_path, names_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    # Step 2: Load feature matrix in read-only memory-mapped mode to prevent RAM spikes
    fft = np.load(fft_path, mmap_mode="r")
    metadata = pd.read_csv(meta_path)

    with open(ch_path, "r") as f:
        channels = [line.strip() for line in f if line.strip()]
    with open(names_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # Step 3: Validate matrix dimensionality and feature width
    assert fft.ndim == 2, f"{subject}: Expected 2D FFT matrix, got ndim={fft.ndim}"
    assert fft.shape[1] == EXPECTED_FEATURES, (
        f"{subject}: Expected {EXPECTED_FEATURES} features, got {fft.shape[1]}"
    )

    # Step 4: Validate strict row count correspondence with metadata
    assert len(fft) == len(metadata), (
        f"{subject}: FFT rows ({len(fft)}) != metadata rows ({len(metadata)})"
    )

    # Step 5: Validate channel montage count and feature label counts
    assert len(channels) == EXPECTED_CHANNELS, (
        f"{subject}: Expected {EXPECTED_CHANNELS} channels, got {len(channels)}"
    )
    assert len(feature_names) == EXPECTED_FEATURES, (
        f"{subject}: Expected {EXPECTED_FEATURES} feature names, got {len(feature_names)}"
    )

    # Step 6: Scan entire matrix for non-finite entries (NaN, +inf, -inf)
    nans = int(np.isnan(fft).sum())
    infs = int(np.isinf(fft).sum())
    assert nans == 0 and infs == 0, f"{subject}: Non-finite values detected (NaN: {nans}, Inf: {infs})"

    print(
        f"  [PASS] {subject.upper()}: {len(fft):,} windows × {EXPECTED_FEATURES} features "
        f"| Range: [{fft.min():.2f}, {fft.max():.2f}] | Mean: {fft.mean():.2f}"
    )
    return len(fft)


def main():
    """Run cross-subject validation across all configured CHB-MIT subjects."""
    print("=" * 60)
    print("CHB-MIT MULTI-SUBJECT FFT DATASET VALIDATION")
    print("=" * 60)

    # Validate each subject sequentially and accumulate total window count
    total_windows = sum(validate_subject_fft(s) for s in SUBJECTS)

    print("=" * 60)
    print(f"ALL {len(SUBJECTS)} SUBJECTS VALIDATED SUCCESSFULLY ({total_windows:,} total windows)")
    print("=" * 60)


if __name__ == "__main__":
    main()