"""
Batch Generation of FFT Log Band-Power Features for CHB-MIT EEG Subjects.

This script processes preprocessed multi-channel EEG window arrays (N, 23, 512)
and computes 115 log band-power features per window (23 channels × 5 clinical bands).

Design Principles:
    - Memory-Efficient Streaming: Uses disk-backed memory-mapped NumPy arrays
      (np.lib.format.open_memmap) to write output features directly to disk row-by-row,
      preventing out-of-memory (OOM) errors on large recording sessions.
    - Caching & Idempotency: Automatically skips generation if complete, valid
      features are already cached on disk (override with --force flag).
    - Alignment Preservation: Verifies strict 1-to-1 alignment between EEG windows,
      subject metadata tables, and electrode channel orders.
    - Feature Label Export: Persists canonical feature names (`<channel>__<band>`)
      to ensure transparent mapping in multimodal models (CNN + FFT + LSTM) and SHAP.
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from fft_features import extract_fft_features, FS

# Base paths and dataset invariants
DATA_DIR = Path("data/processed")
DEFAULT_SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]
EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 512
EXPECTED_FEATURES = 115  # 23 channels * 5 clinical bands (delta, theta, alpha, beta, gamma)


def load_channel_names(path: Path) -> List[str]:
    """
    Load ordered electrode channel names from a newline-delimited text file.

    Parameters
    ----------
    path : Path
        Path to channel names text file (e.g. data/processed/chb01_channel_names.txt).

    Returns
    -------
    list of str
        Ordered list of channel label strings.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_features_for_subject(subject: str, force: bool = False) -> None:
    """
    Extract and persist FFT features for all windows of a given subject.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'chb01').
    force : bool, default=False
        If True, recompute and overwrite existing cached features on disk.

    Raises
    ------
    FileNotFoundError
        If raw EEG window arrays, metadata, or channel names do not exist.
    ValueError
        If array shapes, channel counts, or row counts fail alignment checks.
    """
    windows_path = DATA_DIR / f"{subject}_windows.npy"
    metadata_path = DATA_DIR / f"{subject}_metadata.csv"
    channel_path = DATA_DIR / f"{subject}_channel_names.txt"
    features_path = DATA_DIR / f"{subject}_fft_features.npy"
    names_path = DATA_DIR / f"{subject}_fft_feature_names.txt"

    # Step 1: Verify presence of all required input files
    for p in [windows_path, metadata_path, channel_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    # Step 2: Validate EEG window tensor dimensions using read-only memmap
    windows = np.load(windows_path, mmap_mode="r")
    if windows.ndim != 3 or windows.shape[1] != EXPECTED_CHANNELS or windows.shape[2] != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected (N, {EXPECTED_CHANNELS}, {EXPECTED_SAMPLES}), got {windows.shape}"
        )

    n_windows = len(windows)

    # Step 3: Validate metadata row count alignment
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != n_windows:
        raise ValueError(
            f"Metadata row count ({len(metadata)}) != windows count ({n_windows})"
        )

    # Step 4: Validate electrode montage configuration
    channel_names = load_channel_names(channel_path)
    if len(channel_names) != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} channels, found {len(channel_names)}"
        )

    # Step 5: Cache inspection: skip if already computed with correct dimensions
    if features_path.exists() and not force:
        cached = np.load(features_path, mmap_mode="r")
        if cached.shape == (n_windows, EXPECTED_FEATURES):
            print(f"{subject}: FFT features already cached ({cached.shape}) -> {features_path}")
            return
        # Stale or incomplete cache; remove before recomputing
        features_path.unlink()

    print(f"\nProcessing {subject.upper()}: {n_windows:,} windows...")

    # Step 6: Initialize disk-backed memory-mapped array for streaming writes
    # Avoids allocating massive RAM buffers by writing directly to disk
    output = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, EXPECTED_FEATURES),
    )

    feature_names = None

    # Step 7: Iterate over windows, compute features, and stream to disk
    for i in range(n_windows):
        features, names = extract_fft_features(
            windows[i],
            fs=FS,
            channel_names=channel_names,
        )
        output[i] = features

        if feature_names is None:
            feature_names = names

        # Periodic progress logging
        if (i + 1) % 10000 == 0 or (i + 1) == n_windows:
            pct = 100.0 * (i + 1) / n_windows
            print(f"  [{subject}] {i + 1:,} / {n_windows:,} ({pct:.1f}%)")

    # Flush OS write buffers to guarantee data is fully persisted on disk
    output.flush()

    # Step 8: Persist canonical feature column labels
    with open(names_path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print(f"Saved {subject.upper()} FFT features: {n_windows:,} × {EXPECTED_FEATURES} -> {features_path}")


def parse_args():
    """Parse command-line arguments for subject selection and execution flags."""
    parser = argparse.ArgumentParser(description="Generate FFT features for CHB-MIT EEG windows.")
    parser.add_argument("subjects", nargs="*", default=DEFAULT_SUBJECTS, help="Subjects to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if cached")
    return parser.parse_args()


def main():
    """Main execution entry point: iterates through specified subjects."""
    args = parse_args()
    for subject in args.subjects:
        generate_features_for_subject(subject, force=args.force)


if __name__ == "__main__":
    main()