"""
Batch generation of FFT log band-power features for CHB-MIT EEG subjects.

Extracts 115 frequency-domain features per EEG window (23 channels × 5 bands)
and saves them as memory-mapped arrays to minimize RAM utilization.
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from fft_features import extract_fft_features, FS

DATA_DIR = Path("data/processed")
DEFAULT_SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]
EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 512
EXPECTED_FEATURES = 115


def load_channel_names(path: Path) -> List[str]:
    """Load ordered channel names from text file."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_features_for_subject(subject: str, force: bool = False) -> None:
    """Extract and persist FFT features for all windows of a given subject."""
    windows_path = DATA_DIR / f"{subject}_windows.npy"
    metadata_path = DATA_DIR / f"{subject}_metadata.csv"
    channel_path = DATA_DIR / f"{subject}_channel_names.txt"
    features_path = DATA_DIR / f"{subject}_fft_features.npy"
    names_path = DATA_DIR / f"{subject}_fft_feature_names.txt"

    for p in [windows_path, metadata_path, channel_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    windows = np.load(windows_path, mmap_mode="r")
    if windows.ndim != 3 or windows.shape[1] != EXPECTED_CHANNELS or windows.shape[2] != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected (N, {EXPECTED_CHANNELS}, {EXPECTED_SAMPLES}), got {windows.shape}"
        )

    n_windows = len(windows)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != n_windows:
        raise ValueError(
            f"Metadata row count ({len(metadata)}) != windows count ({n_windows})"
        )

    channel_names = load_channel_names(channel_path)
    if len(channel_names) != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} channels, found {len(channel_names)}"
        )

    if features_path.exists() and not force:
        cached = np.load(features_path, mmap_mode="r")
        if cached.shape == (n_windows, EXPECTED_FEATURES):
            print(f"{subject}: FFT features already cached ({cached.shape}) -> {features_path}")
            return
        features_path.unlink()

    print(f"\nProcessing {subject.upper()}: {n_windows:,} windows...")
    output = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, EXPECTED_FEATURES),
    )

    feature_names = None
    for i in range(n_windows):
        features, names = extract_fft_features(
            windows[i],
            fs=FS,
            channel_names=channel_names,
        )
        output[i] = features

        if feature_names is None:
            feature_names = names

        if (i + 1) % 10000 == 0 or (i + 1) == n_windows:
            pct = 100.0 * (i + 1) / n_windows
            print(f"  [{subject}] {i + 1:,} / {n_windows:,} ({pct:.1f}%)")

    output.flush()

    with open(names_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print(f"Saved {subject.upper()} FFT features: {n_windows:,} × {EXPECTED_FEATURES} -> {features_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FFT features for CHB-MIT EEG windows.")
    parser.add_argument("subjects", nargs="*", default=DEFAULT_SUBJECTS, help="Subjects to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if cached")
    return parser.parse_args()


def main():
    args = parse_args()
    for subject in args.subjects:
        generate_features_for_subject(subject, force=args.force)


if __name__ == "__main__":
    main()