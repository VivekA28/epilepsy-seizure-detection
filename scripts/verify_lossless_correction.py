#!/usr/bin/env python3
"""
Forensic lossless-verification script.
Compares corrected CHB12-CHB15 arrays in data/processed against the backup
in backups/processed_before_channel_fix_20261006_015500/data/processed.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURR_DIR = PROJECT_ROOT / "data" / "processed"
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


def verify_subject(subject: str):
    print("=" * 80)
    print(f"VERIFYING LOSSLESS CORRECTION FOR: {subject.upper()}")
    print("=" * 80)

    # 1. Paths
    c_win_p = CURR_DIR / f"{subject}_windows.npy"
    b_win_p = BACKUP_DIR / f"{subject}_windows.npy"

    c_lbl_p = CURR_DIR / f"{subject}_labels.npy"
    b_lbl_p = BACKUP_DIR / f"{subject}_labels.npy"

    c_meta_p = CURR_DIR / f"{subject}_metadata.csv"
    b_meta_p = BACKUP_DIR / f"{subject}_metadata.csv"

    # 2. Check files exist
    for p in (c_win_p, b_win_p, c_lbl_p, b_lbl_p, c_meta_p, b_meta_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # 3. Load data
    c_win = np.load(c_win_p, mmap_mode="r")
    b_win = np.load(b_win_p, mmap_mode="r")

    c_lbl = np.load(c_lbl_p)
    b_lbl = np.load(b_lbl_p)

    c_meta = pd.read_csv(c_meta_p)
    b_meta = pd.read_csv(b_meta_p)

    # A. Shape check
    assert c_win.shape == b_win.shape, f"Shape mismatch: {c_win.shape} vs {b_win.shape}"
    print(f"1. Shape unchanged                : PASS ({c_win.shape})")

    # B. Number of windows
    n_windows = c_win.shape[0]
    assert n_windows == b_win.shape[0] == len(c_lbl) == len(b_lbl) == len(c_meta) == len(b_meta)
    print(f"2. Number of windows              : PASS ({n_windows:,})")

    # C. Labels byte-for-byte / equality
    assert np.array_equal(c_lbl, b_lbl), "Labels array mismatch!"
    print(f"3. Labels byte-for-byte unchanged : PASS")

    # D. Metadata row count & contents
    assert len(c_meta) == len(b_meta), "Metadata row count mismatch!"
    print(f"4. Metadata row count unchanged   : PASS ({len(c_meta):,})")

    # E. Subject IDs, EDF IDs, Window indices
    assert (c_meta["subject_id"] == b_meta["subject_id"]).all(), "Subject ID mismatch!"
    print(f"5. Subject IDs unchanged          : PASS")

    assert (c_meta["edf_id"] == b_meta["edf_id"]).all(), "EDF ID mismatch!"
    print(f"6. EDF IDs unchanged              : PASS ({c_meta['edf_id'].nunique()} unique EDFs)")

    assert (c_meta["window_index"] == b_meta["window_index"]).all(), "Window index mismatch!"
    print(f"7. Window indices unchanged       : PASS")

    # F. Seizure window count
    c_seiz = int(np.sum(c_lbl == 1))
    b_seiz = int(np.sum(b_lbl == 1))
    assert c_seiz == b_seiz, f"Seizure count mismatch: {c_seiz} vs {b_seiz}"
    print(f"8. Seizure-window count unchanged : PASS ({c_seiz:,} seizures)")

    # G. No NaN / Inf
    # Chunked check across entire array
    chunk_size = 10000
    for s_idx in range(0, n_windows, chunk_size):
        e_idx = min(s_idx + chunk_size, n_windows)
        chunk = c_win[s_idx:e_idx]
        if not np.isfinite(chunk).all():
            raise ValueError(f"NaN/Inf detected in chunk {s_idx}:{e_idx}")
    print(f"9. NaN/Inf check                  : PASS (all finite)")

    # H. Complete numerical channel permutation check
    print("10. Full numerical channel permutation verification:")
    for target_c in range(23):
        source_c = PERMUTATION[target_c]
        # Verify in chunks across all windows
        for s_idx in range(0, n_windows, chunk_size):
            e_idx = min(s_idx + chunk_size, n_windows)
            c_slice = c_win[s_idx:e_idx, target_c, :]
            b_slice = b_win[s_idx:e_idx, source_c, :]
            if not np.array_equal(c_slice, b_slice):
                raise ValueError(
                    f"NUMERICAL MISMATCH: target channel {target_c} does not match backup channel {source_c} "
                    f"at windows {s_idx}:{e_idx}"
                )
        print(f"    Target channel {target_c:2d} == Backup channel {source_c:2d} [EXACT NUMERIC MATCH]")

    print(f"\n{subject.upper()} LOSSLESS VERIFICATION: 100% PERFECT MATCH ✅\n")
    return True


def main():
    print("\n" + "#" * 80)
    print("COMPREHENSIVE LOSSLESS VERIFICATION: CHB12 - CHB15")
    print("#" * 80 + "\n")

    all_passed = True
    for sub in AFFECTED_SUBJECTS:
        try:
            verify_subject(sub)
        except Exception as e:
            print(f"❌ FAILED for {sub}: {e}")
            all_passed = False

    print("#" * 80)
    if all_passed:
        print("FINAL VERDICT: ALL 4 SUBJECTS LOSSLESSLY VERIFIED (100% EXACT EQUIVALENCE) ✅")
    else:
        print("FINAL VERDICT: VERIFICATION FAILED ❌")
    print("#" * 80)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
