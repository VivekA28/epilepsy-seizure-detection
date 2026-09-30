"""
Integration validation for CHB-MIT FFT datasets across subjects chb01–chb05.

Verifies that all precomputed FFT matrices exist, have shape (N, 115),
contain no NaNs/Infs, and match metadata row counts.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]
EXPECTED_FEATURES = 115
EXPECTED_CHANNELS = 23


def validate_subject_fft(subject: str) -> int:
    """Validate FFT dataset integrity for one subject; returns window count."""
    fft_path = DATA_DIR / f"{subject}_fft_features.npy"
    meta_path = DATA_DIR / f"{subject}_metadata.csv"
    ch_path = DATA_DIR / f"{subject}_channel_names.txt"
    names_path = DATA_DIR / f"{subject}_fft_feature_names.txt"

    for p in [fft_path, meta_path, ch_path, names_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    fft = np.load(fft_path, mmap_mode="r")
    metadata = pd.read_csv(meta_path)

    with open(ch_path, "r") as f:
        channels = [line.strip() for line in f if line.strip()]
    with open(names_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    assert fft.ndim == 2, f"{subject}: Expected 2D FFT matrix, got ndim={fft.ndim}"
    assert fft.shape[1] == EXPECTED_FEATURES, (
        f"{subject}: Expected {EXPECTED_FEATURES} features, got {fft.shape[1]}"
    )
    assert len(fft) == len(metadata), (
        f"{subject}: FFT rows ({len(fft)}) != metadata rows ({len(metadata)})"
    )
    assert len(channels) == EXPECTED_CHANNELS, (
        f"{subject}: Expected {EXPECTED_CHANNELS} channels, got {len(channels)}"
    )
    assert len(feature_names) == EXPECTED_FEATURES, (
        f"{subject}: Expected {EXPECTED_FEATURES} feature names, got {len(feature_names)}"
    )

    nans = int(np.isnan(fft).sum())
    infs = int(np.isinf(fft).sum())
    assert nans == 0 and infs == 0, f"{subject}: Non-finite values detected (NaN: {nans}, Inf: {infs})"

    print(
        f"  [PASS] {subject.upper()}: {len(fft):,} windows × {EXPECTED_FEATURES} features "
        f"| Range: [{fft.min():.2f}, {fft.max():.2f}] | Mean: {fft.mean():.2f}"
    )
    return len(fft)


def main():
    print("=" * 60)
    print("CHB-MIT MULTI-SUBJECT FFT DATASET VALIDATION")
    print("=" * 60)

    total_windows = sum(validate_subject_fft(s) for s in SUBJECTS)

    print("=" * 60)
    print(f"ALL {len(SUBJECTS)} SUBJECTS VALIDATED SUCCESSFULLY ({total_windows:,} total windows)")
    print("=" * 60)


if __name__ == "__main__":
    main()