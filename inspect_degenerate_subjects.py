#!/usr/bin/env python3
"""
Standalone read-only inspection script to diagnose raw window signals and
cached CNN features for chb01, chb13, chb02, chb05.
"""

from pathlib import Path
import numpy as np

DATA_DIR = Path("data/processed")
CNN_DIR = DATA_DIR / "cnn_features"
SUBJECTS = ["chb01", "chb13", "chb02", "chb05"]
CHUNK_SIZE = 5000


def inspect_subject_windows(subject: str):
    win_path = DATA_DIR / f"{subject}_windows.npy"
    if not win_path.exists():
        raise FileNotFoundError(f"Missing windows file: {win_path}")

    windows = np.load(win_path, mmap_mode="r")
    n_windows, n_channels, n_samples = windows.shape
    total_elements = n_windows * n_channels * n_samples
    channel_elements = n_windows * n_samples

    # Accumulators for whole array
    global_min = float("inf")
    global_max = float("-inf")
    total_sum = 0.0
    total_sq_sum = 0.0
    nan_count = 0
    inf_count = 0
    abs_gt_100_count = 0

    # Accumulators per channel
    ch_min = np.full(n_channels, float("inf"), dtype=np.float64)
    ch_max = np.full(n_channels, float("-inf"), dtype=np.float64)
    ch_sum = np.zeros(n_channels, dtype=np.float64)
    ch_sq_sum = np.zeros(n_channels, dtype=np.float64)
    ch_max_abs = np.zeros(n_channels, dtype=np.float64)

    for start in range(0, n_windows, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_windows)
        chunk = np.asarray(windows[start:end], dtype=np.float64)

        # Whole array checks
        chunk_nans = int(np.isnan(chunk).sum())
        chunk_infs = int(np.isinf(chunk).sum())
        nan_count += chunk_nans
        inf_count += chunk_infs

        abs_gt_100_count += int((np.abs(chunk) > 100.0).sum())

        valid_mask = np.isfinite(chunk)
        if valid_mask.any():
            valid_vals = chunk[valid_mask]
            global_min = min(global_min, float(valid_vals.min()))
            global_max = max(global_max, float(valid_vals.max()))
            total_sum += float(valid_vals.sum())
            total_sq_sum += float((valid_vals ** 2).sum())

        # Per-channel stats across this chunk (axis 0 = window, axis 2 = sample)
        for c in range(n_channels):
            c_data = chunk[:, c, :]
            c_valid = np.isfinite(c_data)
            if c_valid.any():
                c_vals = c_data[c_valid]
                ch_min[c] = min(ch_min[c], float(c_vals.min()))
                ch_max[c] = max(ch_max[c], float(c_vals.max()))
                ch_sum[c] += float(c_vals.sum())
                ch_sq_sum[c] += float((c_vals ** 2).sum())
                ch_max_abs[c] = max(ch_max_abs[c], float(np.abs(c_vals).max()))

    overall_mean = total_sum / total_elements
    overall_var = max(0.0, (total_sq_sum / total_elements) - (overall_mean ** 2))
    overall_std = np.sqrt(overall_var)

    ch_mean = ch_sum / channel_elements
    ch_var = np.maximum(0.0, (ch_sq_sum / channel_elements) - (ch_mean ** 2))
    ch_std = np.sqrt(ch_var)

    flat_channels = []
    extreme_channels = []
    for c in range(n_channels):
        if ch_std[c] < 1e-4:
            flat_channels.append(c)
        if ch_max_abs[c] > 1000.0:
            extreme_channels.append(c)

    return {
        "shape": windows.shape,
        "global_min": global_min,
        "global_max": global_max,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "abs_gt_100_count": abs_gt_100_count,
        "ch_min": ch_min,
        "ch_max": ch_max,
        "ch_std": ch_std,
        "flat_channels": flat_channels,
        "extreme_channels": extreme_channels,
    }


def inspect_subject_cnn_features(subject: str):
    feat_path = CNN_DIR / f"{subject}_cnn_features.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing CNN features: {feat_path}")

    feat = np.load(feat_path, mmap_mode="r")
    feat_arr = np.asarray(feat, dtype=np.float64)

    nan_count = int(np.isnan(feat_arr).sum())
    inf_count = int(np.isinf(feat_arr).sum())
    f_min = float(np.nanmin(feat_arr))
    f_max = float(np.nanmax(feat_arr))
    f_mean = float(np.nanmean(feat_arr))
    f_std = float(np.nanstd(feat_arr))

    dim_var = np.var(feat_arr, axis=0)
    near_const_dims = int((dim_var < 1e-6).sum())

    return {
        "shape": feat.shape,
        "f_min": f_min,
        "f_max": f_max,
        "f_mean": f_mean,
        "f_std": f_std,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "dim_var": dim_var,
        "near_const_dims": near_const_dims,
    }


def main():
    print("=" * 85)
    print("INSPECTION OF RAW WINDOWS AND CACHED CNN FEATURES")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print("=" * 85)

    summary_records = {}

    for subject in SUBJECTS:
        print("\n" + "#" * 85)
        print(f"SUBJECT: {subject}")
        print("#" * 85)

        # 1. Raw Windows
        print(f"\n--- 1. Raw EEG Windows (data/processed/{subject}_windows.npy) ---")
        w_stats = inspect_subject_windows(subject)
        print(f"Shape                     : {w_stats['shape']}")
        print(f"Global Min                : {w_stats['global_min']:.6f}")
        print(f"Global Max                : {w_stats['global_max']:.6f}")
        print(f"Global Mean               : {w_stats['overall_mean']:.6f}")
        print(f"Global Std                : {w_stats['overall_std']:.6f}")
        print(f"Count of NaN              : {w_stats['nan_count']}")
        print(f"Count of Inf              : {w_stats['inf_count']}")
        print(f"Count of abs() > 100      : {w_stats['abs_gt_100_count']:,}")

        print("\nPer-Channel Statistics (23 channels):")
        print(f"{'Chan':<5} | {'Min':>12} | {'Max':>12} | {'Std':>12} | {'Flags':<25}")
        print("-" * 75)
        for c in range(w_stats["shape"][1]):
            flags = []
            if w_stats["ch_std"][c] < 1e-4:
                flags.append("FLAT (std < 1e-4)")
            if max(abs(w_stats["ch_min"][c]), abs(w_stats["ch_max"][c])) > 1000.0:
                flags.append("EXTREME (abs > 1000)")
            flag_str = ", ".join(flags) if flags else "OK"
            print(
                f"{c:<5d} | {w_stats['ch_min'][c]:>12.4f} | {w_stats['ch_max'][c]:>12.4f} | "
                f"{w_stats['ch_std'][c]:>12.4f} | {flag_str:<25}"
            )

        print(f"\nFlat channels count (std < 1e-4)   : {len(w_stats['flat_channels'])} {w_stats['flat_channels']}")
        print(f"Extreme channels count (abs > 1000): {len(w_stats['extreme_channels'])} {w_stats['extreme_channels']}")

        # 2. CNN Features
        print(f"\n--- 2. Cached CNN Features (data/processed/cnn_features/{subject}_cnn_features.npy) ---")
        c_stats = inspect_subject_cnn_features(subject)
        print(f"Shape                     : {c_stats['shape']}")
        print(f"Global Min                : {c_stats['f_min']:.6f}")
        print(f"Global Max                : {c_stats['f_max']:.6f}")
        print(f"Global Mean               : {c_stats['f_mean']:.6f}")
        print(f"Global Std                : {c_stats['f_std']:.6f}")
        print(f"Count of NaN              : {c_stats['nan_count']}")
        print(f"Count of Inf              : {c_stats['inf_count']}")
        print(f"Near-Constant Dims (var < 1e-6): {c_stats['near_const_dims']}/128")

        summary_records[subject] = {
            "abs_gt_100": w_stats["abs_gt_100_count"],
            "flat_channels": len(w_stats["flat_channels"]),
            "extreme_channels": len(w_stats["extreme_channels"]),
            "near_const_dims": c_stats["near_const_dims"],
            "w_min": w_stats["global_min"],
            "w_max": w_stats["global_max"],
            "w_std": w_stats["overall_std"],
            "c_std": c_stats["f_std"],
        }

    # 3. Summary Table
    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE SUMMARY COMPARISON ACROSS ALL 4 SUBJECTS")
    print("=" * 90)
    header = f"{'Metric':<35} | {'chb01 (Ref)':^12} | {'chb13 (Works)':^13} | {'chb02 (Broken)':^14} | {'chb05 (Broken)':^14}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Raw abs() > 100 count", "abs_gt_100", "{:,}"),
        ("Flat channels (std < 1e-4)", "flat_channels", "{}"),
        ("Extreme channels (abs > 1000)", "extreme_channels", "{}"),
        ("Near-constant CNN dims (var < 1e-6)", "near_const_dims", "{}/128"),
        ("Raw global min", "w_min", "{:.4f}"),
        ("Raw global max", "w_max", "{:.4f}"),
        ("Raw global std", "w_std", "{:.4f}"),
        ("CNN global std", "c_std", "{:.4f}"),
    ]

    for label, key, fmt in metrics:
        v01 = fmt.format(summary_records["chb01"][key])
        v13 = fmt.format(summary_records["chb13"][key])
        v02 = fmt.format(summary_records["chb02"][key])
        v05 = fmt.format(summary_records["chb05"][key])
        print(f"{label:<35} | {v01:^12} | {v13:^13} | {v02:^14} | {v05:^14}")

    print("=" * 90)


if __name__ == "__main__":
    main()
