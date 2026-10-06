#!/usr/bin/env python3
"""
Dedicated script to correct the 23-channel permutation for CHB12, CHB13, CHB14, and CHB15.

Canonical CHB01-CHB11 order:
  0..7   : Left Temporal + Left Parasagittal (FP1-F7 .. P3-O1)
  8..11  : Right Parasagittal (FP2-F4 .. P4-O2)
  12..15 : Right Temporal (FP2-F8 .. P8-O2)
  16..17 : Central Midline (FZ-CZ, CZ-PZ)
  18..22 : Transverse / Basal (P7-T7 .. T8-P8)

Current CHB12-CHB15 processed order:
  0..7   : Left Temporal + Left Parasagittal (FP1-F7 .. P3-O1)
  8..9   : Central Midline (FZ-CZ, CZ-PZ)
  10..17 : Right Parasagittal + Right Temporal (FP2-F4 .. P8-O2)
  18..22 : Transverse / Basal (P7-T7 .. T8-P8)

The required permutation from CURRENT -> CANONICAL is:
  permutation = [
      0, 1, 2, 3,
      4, 5, 6, 7,
      10, 11, 12, 13,
      14, 15, 16, 17,
      8, 9,
      18, 19, 20, 21, 22
  ]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BACKUP_DIR = PROJECT_ROOT / "backups" / "processed_before_channel_fix_20261006_015500" / "data" / "processed"

AFFECTED_SUBJECTS = ["chb12", "chb13", "chb14", "chb15"]

PERMUTATION = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    10, 11, 12, 13,
    14, 15, 16, 17,
    8, 9,
    18, 19, 20, 21, 22
]

CANONICAL_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
]

EXPECTED_OLD_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
]


def normalize_channel_name(ch: str) -> str:
    """Normalize channel name for semantic comparison."""
    ch = ch.strip()
    if ch in {"T8-P8-0", "T8-P8-1"}:
        return "T8-P8"
    return ch


def inspect_subject(subject: str) -> dict:
    """Gather diagnostic info for a subject without modifying anything."""
    win_path = DATA_DIR / f"{subject}_windows.npy"
    lbl_path = DATA_DIR / f"{subject}_labels.npy"
    meta_path = DATA_DIR / f"{subject}_metadata.csv"
    chan_path = DATA_DIR / f"{subject}_channel_names.txt"

    for p in (win_path, lbl_path, meta_path, chan_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    windows = np.load(win_path, mmap_mode="r")
    labels = np.load(lbl_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)

    with open(chan_path, "r", encoding="utf-8") as f:
        current_channels = [line.strip() for line in f if line.strip()]

    if len(current_channels) != 23:
        raise ValueError(f"{subject}: expected 23 channels in {chan_path}, found {len(current_channels)}")

    if windows.ndim != 3 or windows.shape[1:] != (23, 512):
        raise ValueError(f"{subject}: unexpected window shape {windows.shape}")

    n_windows = windows.shape[0]
    if len(labels) != n_windows or len(meta) != n_windows:
        raise ValueError(f"{subject}: row count mismatch: windows={n_windows}, labels={len(labels)}, meta={len(meta)}")

    seizure_count = int(np.sum(labels == 1))

    # Compute permuted channel names
    permuted_channels = [current_channels[i] for i in PERMUTATION]

    return {
        "subject": subject,
        "n_windows": n_windows,
        "original_shape": windows.shape,
        "expected_shape": windows.shape,
        "seizure_count": seizure_count,
        "metadata_rows": len(meta),
        "current_channels": current_channels,
        "permuted_channels": permuted_channels,
        "is_currently_old_order": (current_channels == EXPECTED_OLD_CHANNELS),
        "is_already_canonical": (current_channels == CANONICAL_CHANNELS),
        "win_path": win_path,
        "lbl_path": lbl_path,
        "meta_path": meta_path,
        "chan_path": chan_path,
    }


def dry_run_report():
    print("=" * 90)
    print("DRY-RUN / AUDIT REPORT — CHANNEL ORDER CORRECTION")
    print("=" * 90)
    print(f"Permutation mapping (Current -> Canonical):")
    print(f"  {PERMUTATION}\n")

    all_valid = True
    reports = []

    for subject in AFFECTED_SUBJECTS:
        info = inspect_subject(subject)
        reports.append(info)

        print("-" * 90)
        print(f"Subject                 : {subject.upper()}")
        print(f"Original shape          : {info['original_shape']}")
        print(f"Expected shape          : {info['expected_shape']}")
        print(f"Number of windows       : {info['n_windows']:,}")
        print(f"Seizure windows         : {info['seizure_count']:,}")
        print(f"Metadata rows           : {info['metadata_rows']:,}")
        print(f"Current matches old fmt : {info['is_currently_old_order']}")
        print(f"Already canonical       : {info['is_already_canonical']}")

        print(f"Original channels (first 10):  {info['current_channels'][:10]}")
        print(f"Corrected channels (first 10): {info['permuted_channels'][:10]}")

        # Verify that permuting current_channels yields CANONICAL_CHANNELS
        if info["permuted_channels"] != CANONICAL_CHANNELS:
            print("❌ ERROR: Permutation of current channels does NOT match canonical channels!")
            all_valid = False
        else:
            print("✅ Permuted channel names match CANONICAL_CHANNELS exactly.")

    print("\n" + "=" * 90)
    if all_valid:
        print("DRY-RUN STATUS: PASS — All pre-conditions verified.")
    else:
        print("DRY-RUN STATUS: FAIL — Pre-conditions violated.")
    print("=" * 90)
    return reports


def apply_correction(chunk_size: int = 10000):
    print("=" * 90)
    print("APPLYING LOSSLESS CHANNEL REORDERING TO AFFECTED SUBJECTS")
    print("=" * 90)

    # 1. Verify safe backup exists before anything else
    if not BACKUP_DIR.exists():
        raise RuntimeError(f"ABORTING: Safe backup directory not found at {BACKUP_DIR}")

    for subject in AFFECTED_SUBJECTS:
        backup_win = BACKUP_DIR / f"{subject}_windows.npy"
        if not backup_win.exists():
            raise RuntimeError(f"ABORTING: Missing backup for {subject} at {backup_win}")

    print(f"✅ Verified safe backup exists for all affected subjects in:\n  {BACKUP_DIR}\n")

    for subject in AFFECTED_SUBJECTS:
        print("-" * 90)
        print(f"Processing {subject.upper()}...")
        info = inspect_subject(subject)

        if info["is_already_canonical"]:
            print(f"  {subject} is ALREADY in canonical order. Skipping.")
            continue

        if not info["is_currently_old_order"]:
            raise RuntimeError(f"{subject}: current channels do not match expected old order! Aborting.")

        win_path = info["win_path"]
        chan_path = info["chan_path"]
        n_windows = info["n_windows"]

        # Temporary output file
        tmp_win_path = DATA_DIR / f"{subject}_windows_corrected_tmp.npy"
        if tmp_win_path.exists():
            tmp_win_path.unlink()

        orig_mmap = np.load(win_path, mmap_mode="r")
        tmp_mmap = np.lib.format.open_memmap(
            tmp_win_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_windows, 23, 512),
        )

        print(f"  Streaming {n_windows:,} windows in chunks of {chunk_size:,}...")
        for start_idx in range(0, n_windows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_windows)
            chunk = np.array(orig_mmap[start_idx:end_idx], copy=True)
            # Apply permutation along channel axis (axis 1)
            reordered_chunk = chunk[:, PERMUTATION, :]
            tmp_mmap[start_idx:end_idx] = reordered_chunk

        tmp_mmap.flush()

        print(f"  Verifying lossless integrity for {subject} before atomic commit...")
        # Check finite
        if not np.isfinite(tmp_mmap[::50]).all():
            raise RuntimeError(f"CORRUPTION DETECTED: {subject} temporary array contains NaN/Inf!")

        # Verify exact numeric equivalence across ALL 23 channels on spot-checked slices and full bounds
        test_indices = np.linspace(0, n_windows - 1, min(n_windows, 500), dtype=int)
        for target_c in range(23):
            source_c = PERMUTATION[target_c]
            c_corrected = tmp_mmap[test_indices, target_c, :]
            c_original = orig_mmap[test_indices, source_c, :]
            if not np.array_equal(c_corrected, c_original):
                raise RuntimeError(
                    f"NUMERICAL MISMATCH for {subject}: channel {target_c} does not match source channel {source_c}!"
                )

        print(f"  ✅ Numerical permutation verified for all 23 channels.")

        # Atomic commit
        del orig_mmap
        del tmp_mmap

        # Atomic rename
        os.replace(tmp_win_path, win_path)
        print(f"  Committed corrected array to {win_path.name}")

        # Update channel_names.txt atomically
        tmp_chan_path = DATA_DIR / f"{subject}_channel_names_tmp.txt"
        with open(tmp_chan_path, "w", encoding="utf-8") as f:
            for ch in CANONICAL_CHANNELS:
                f.write(f"{ch}\n")
        os.replace(tmp_chan_path, chan_path)
        print(f"  Updated {chan_path.name} to CANONICAL order.")

    print("\n" + "=" * 90)
    print("ALL AFFECTED SUBJECTS SUCCESSFULLY AND LOSSLESSLY REORDERED")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Fix channel ordering for CHB12-CHB15.")
    parser.add_argument("--apply", action="store_true", help="Apply the correction (default is dry-run)")
    args = parser.parse_args()

    reports = dry_run_report()

    if args.apply:
        apply_correction()
    else:
        print("\nTo apply the correction, run:")
        print("  python scripts/fix_channel_order.py --apply\n")


if __name__ == "__main__":
    main()
