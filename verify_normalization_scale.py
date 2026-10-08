#!/usr/bin/env python3
"""
verify_normalization_scale.py
------------------------------
Standalone, read-only script to verify the normalization scale of CHB-MIT EEG
processed window arrays (data/processed/{subject}_windows.npy) for subjects
chb01 through chb15.

Classification criteria:
- "NORMALIZED": global std between 0.5 and 2.0
- "RAW/UNNORMALIZED": global std below 0.01
- "UNKNOWN/OTHER": neither (flagged for manual review)

Uses mmap_mode="r" and random sampling of up to 2000 windows (random.seed(42))
for fast, memory-safe, reproducible verification.
"""

from pathlib import Path
import random
import sys
import numpy as np

# Subjects to inspect
SUBJECTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_SIZE = 2000
RANDOM_SEED = 42


def verify_subject_scale(subject: str, data_dir: Path = DATA_DIR):
    win_path = data_dir / f"{subject}_windows.npy"
    if not win_path.exists():
        return {
            "subject": subject,
            "exists": False,
            "error": f"File not found: {win_path}",
        }

    # 1. Load data/processed/{subject}_windows.npy with mmap_mode="r"
    windows_mmap = np.load(win_path, mmap_mode="r")
    n_windows = windows_mmap.shape[0]

    # 2. Sample windows reproducibly for speed and low memory footprint
    # Seed both python random and numpy for complete reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if n_windows <= SAMPLE_SIZE:
        sample = np.array(windows_mmap)
        n_sampled = n_windows
    else:
        sample_indices = np.sort(
            np.random.choice(n_windows, size=SAMPLE_SIZE, replace=False)
        )
        sample = np.array(windows_mmap[sample_indices])
        n_sampled = SAMPLE_SIZE

    # Compute global std, min, max
    global_std = float(sample.std())
    global_min = float(sample.min())
    global_max = float(sample.max())

    # Compute per-channel std (axis 0: sampled windows, axis 2: time points)
    # Windows shape is (N, channels, time_samples) -> e.g. (2000, 23, 512)
    per_channel_std = sample.std(axis=(0, 2)).tolist()

    # 3. Classify the subject
    if 0.5 <= global_std <= 2.0:
        classification = "NORMALIZED"
    elif global_std < 0.01:
        classification = "RAW/UNNORMALIZED"
    else:
        classification = "UNKNOWN/OTHER"

    return {
        "subject": subject,
        "exists": True,
        "n_windows": n_windows,
        "n_sampled": n_sampled,
        "global_std": global_std,
        "global_min": global_min,
        "global_max": global_max,
        "per_channel_std": per_channel_std,
        "classification": classification,
    }


def format_num(val: float) -> str:
    """Format numbers appropriately based on magnitude."""
    if abs(val) < 0.01 and abs(val) > 0:
        return f"{val:+.4e}"
    return f"{val:+9.4f}"


def format_std(val: float) -> str:
    """Format standard deviation appropriately."""
    if val < 0.01:
        return f"{val:.4e}"
    return f"{val:.4f}    "


def main():
    print("=" * 86)
    print("CHB-MIT EEG DATA NORMALIZATION SCALE VERIFICATION (chb01 - chb15)")
    print("=" * 86)
    print(f"Data directory: {DATA_DIR}")
    print(f"Sample size   : {SAMPLE_SIZE} windows (random.seed={RANDOM_SEED})")
    print(f"Mode          : READ-ONLY (mmap_mode='r')")
    print("-" * 86)

    results = []
    needing_reprocessing = []
    already_normalized = []
    unknown_flagged = []

    for sub in SUBJECTS:
        res = verify_subject_scale(sub)
        results.append(res)
        if not res["exists"]:
            unknown_flagged.append(sub)
        elif res["classification"] == "NORMALIZED":
            already_normalized.append(sub)
        elif res["classification"] == "RAW/UNNORMALIZED":
            needing_reprocessing.append(sub)
        else:
            unknown_flagged.append(sub)

    # Detailed per-channel statistics
    print("\nPER-CHANNEL STD SUMMARY (23 CHANNELS PER SUBJECT):")
    print("-" * 86)
    for res in results:
        if not res["exists"]:
            print(f"{res['subject']}: [FILE NOT FOUND]")
            continue
        c_stds = res["per_channel_std"]
        c_min = min(c_stds)
        c_max = max(c_stds)
        c_mean = sum(c_stds) / len(c_stds)
        if res["classification"] == "RAW/UNNORMALIZED":
            c_summary = f"min={c_min:.2e}, mean={c_mean:.2e}, max={c_max:.2e}"
        else:
            c_summary = f"min={c_min:.4f}, mean={c_mean:.4f}, max={c_max:.4f}"
        print(f"  {res['subject']} (N={res['n_windows']:6d}, sample={res['n_sampled']:4d}): "
              f"23-ch std range [{c_summary}]")

    print("\n" + "=" * 86)
    print("SUMMARY TABLE: NORMALIZATION SCALE VERIFICATION")
    print("=" * 86)
    header = f"{'subject':<9} | {'global_std':<12} | {'global_min':<13} | {'global_max':<13} | {'classification':<18}"
    print(header)
    print("-" * len(header))

    for res in results:
        if not res["exists"]:
            print(f"{res['subject']:<9} | {'MISSING':<12} | {'MISSING':<13} | {'MISSING':<13} | NOT FOUND")
            continue
        g_std_str = format_std(res["global_std"])
        g_min_str = format_num(res["global_min"])
        g_max_str = format_num(res["global_max"])
        cls_str = res["classification"]
        if cls_str == "UNKNOWN/OTHER":
            cls_str = f"UNKNOWN/OTHER ({res['global_std']})"
        print(f"{res['subject']:<9} | {g_std_str:<12} | {g_min_str:<13} | {g_max_str:<13} | {cls_str:<18}")

    print("=" * 86)
    print()
    print(f"Subjects needing reprocessing: {needing_reprocessing}")
    print(f"Subjects already correctly normalized: {already_normalized}")

    if unknown_flagged:
        print(f"FLAGGED FOR MANUAL REVIEW (UNKNOWN/OTHER): {unknown_flagged}")
    print()


if __name__ == "__main__":
    main()
