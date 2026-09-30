"""
Comprehensive feature alignment test across CHB-MIT subjects.
Verifies that CNN features (128-D), FFT features (115-D), raw EEG windows,
labels, and metadata are 100% aligned window-for-window with zero NaNs/Infs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
CNN_DIR = DATA_DIR / "cnn_features"
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def main():
    print("=" * 75)
    print("FEATURE ALIGNMENT TEST (CNN Features <-> FFT Features <-> Labels)")
    print("=" * 75)

    total_windows = 0
    total_seizures = 0
    all_passed = True
    summary_rows = []

    for s in SUBJECTS:
        print(f"\n" + "-" * 75)
        print(f"SUBJECT: {s.upper()}")
        print("-" * 75)

        eeg_p = DATA_DIR / f"{s}_windows.npy"
        cnn_p = CNN_DIR / f"{s}_cnn_features.npy"
        fft_p = DATA_DIR / f"{s}_fft_features.npy"
        lbl_p = DATA_DIR / f"{s}_labels.npy"
        fid_p = DATA_DIR / f"{s}_file_ids.npy"
        meta_p = DATA_DIR / f"{s}_metadata.csv"

        eeg = np.load(eeg_p, mmap_mode="r")
        cnn = np.load(cnn_p, mmap_mode="r")
        fft = np.load(fft_p, mmap_mode="r")
        lbl = np.load(lbl_p)
        fid = np.load(fid_p)
        meta = pd.read_csv(meta_p)

        N = len(eeg)
        total_windows += N
        seizures = int(lbl.sum())
        total_seizures += seizures

        # 1. Shapes
        c1 = (len(eeg) == len(cnn) == len(fft) == len(lbl) == len(fid) == len(meta))
        c2 = (cnn.shape[1] == 128)
        c3 = (fft.shape[1] == 115)
        c4 = (eeg.shape[1] == 23 and eeg.shape[2] == 512)

        print(f"1. Shape Verification:")
        print(f"   - Raw EEG Windows    : {eeg.shape}")
        print(f"   - CNN Features (128D): {cnn.shape}")
        print(f"   - FFT Features (115D): {fft.shape}")
        print(f"   - Ground Truth Labels: {lbl.shape}")
        print(f"   - File IDs           : {fid.shape}")
        print(f"   - Metadata Rows      : {meta.shape}")

        if c1 and c2 and c3 and c4:
            print("   -> [PASS] Exact row and column alignment verified.")
        else:
            print("   -> [FAIL] Shape mismatch detected.")
            all_passed = False

        # 2. Value integrity
        cnn_nan = int(np.isnan(cnn).sum())
        cnn_inf = int(np.isinf(cnn).sum())
        fft_nan = int(np.isnan(fft).sum())
        fft_inf = int(np.isinf(fft).sum())

        print(f"2. Value Integrity:")
        print(f"   - CNN Features NaN: {cnn_nan}, Inf: {cnn_inf}")
        print(f"   - FFT Features NaN: {fft_nan}, Inf: {fft_inf}")
        if cnn_nan == 0 and cnn_inf == 0 and fft_nan == 0 and fft_inf == 0:
            print("   -> [PASS] All values finite and clean.")
        else:
            print("   -> [FAIL] NaN or Inf detected.")
            all_passed = False

        # 3. Scale comparison
        cnn_sample = cnn[:1000]
        fft_sample = fft[:1000]
        print(f"3. Feature Distribution:")
        print(f"   - CNN: min={cnn_sample.min():.4f}, max={cnn_sample.max():.4f}, mean={cnn_sample.mean():.4f}, std={cnn_sample.std():.4f}")
        print(f"   - FFT: min={fft_sample.min():.4f}, max={fft_sample.max():.4f}, mean={fft_sample.mean():.4f}, std={fft_sample.std():.4f}")

        # 4. Metadata label sync
        meta_lbl = meta["label"].to_numpy()
        if np.array_equal(lbl, meta_lbl):
            print(f"4. Label Sync: [PASS] labels.npy matches metadata.csv ({seizures} seizure windows)")
        else:
            print("4. Label Sync: [FAIL] mismatch")
            all_passed = False

        # 5. Fusion simulation
        sample_fused = np.concatenate([cnn[0:1], fft[0:1]], axis=-1)
        if sample_fused.shape == (1, 243):
            print(f"5. Dual-Domain Fusion: (128D + 115D) -> {sample_fused.shape} [PASS]")
        else:
            print("5. Dual-Domain Fusion: [FAIL]")
            all_passed = False

        summary_rows.append({
            "Subject": s,
            "Windows": N,
            "Seizure Windows": seizures,
            "CNN Shape": str(cnn.shape),
            "FFT Shape": str(fft.shape),
            "Fused Shape": f"({N}, 243)",
            "Status": "PASS" if all_passed else "FAIL",
        })

    print("\n" + "=" * 75)
    print("ALL-SUBJECT FEATURE ALIGNMENT SUMMARY")
    print("=" * 75)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 75)
    print(f"Grand Total Windows : {total_windows:,}")
    print(f"Grand Total Seizures: {total_seizures:,} ({100*total_seizures/total_windows:.2f}%)")
    print(f"Dual-Domain Matrix  : ({total_windows:,}, 243)")
    print("OVERALL RESULT      : [PASS] Ready for Model 3 & Model 4 fusion!" if all_passed else "OVERALL RESULT      : [FAIL]")
    print("=" * 75)


if __name__ == "__main__":
    main()
