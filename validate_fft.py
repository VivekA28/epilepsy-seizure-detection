"""
Unit validation for FFT feature extraction on real CHB-MIT EEG data.

Validates mathematical properties, frequency resolution, band bin boundaries,
and numerical stability on single windows.
"""

from pathlib import Path
import numpy as np
from fft_features import extract_fft_features, FS, N_SAMPLES, BANDS

DATA_DIR = Path("data/processed")
WINDOWS_PATH = DATA_DIR / "chb01_windows.npy"
CHANNELS_PATH = DATA_DIR / "chb01_channel_names.txt"


def check(condition: bool, success_msg: str, fail_msg: str) -> None:
    if condition:
        print(f"  [PASS] {success_msg}")
    else:
        print(f"  [FAIL] {fail_msg}")
        raise AssertionError(fail_msg)


def main():
    print("=" * 60)
    print("FFT FEATURE EXTRACTION VALIDATION")
    print("=" * 60)

    # 1. Load sample window and channel names
    windows = np.load(WINDOWS_PATH, mmap_mode="r")
    with open(CHANNELS_PATH, "r") as f:
        channel_names = [line.strip() for line in f if line.strip()]

    window = windows[0]
    check(window.ndim == 2, "EEG window is 2D (channels, samples).", "Window is not 2D.")
    check(window.shape[0] == 23, "Channel count == 23.", f"Got {window.shape[0]} channels.")
    check(window.shape[1] == 512, "Sample count == 512.", f"Got {window.shape[1]} samples.")

    # 2. Spectral resolution checks
    freq_res = FS / N_SAMPLES
    check(freq_res == 0.25, f"Frequency resolution is 0.25 Hz ({FS} / {N_SAMPLES}).", "Resolution mismatch.")

    rfft_freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / FS)
    check(len(rfft_freqs) == 257, "rFFT produces 257 frequency bins.", f"Got {len(rfft_freqs)} bins.")
    check(rfft_freqs[0] == 0.0, "FFT begins at 0.0 Hz.", "FFT does not start at 0 Hz.")
    check(rfft_freqs[-1] == 64.0, "Nyquist frequency == 64.0 Hz.", "Nyquist mismatch.")

    # 3. Useful band checks (0.5 - 40.0 Hz)
    useful_mask = (rfft_freqs >= 0.5) & (rfft_freqs <= 40.0)
    useful_freqs = rfft_freqs[useful_mask]
    check(len(useful_freqs) == 159, "Useful range (0.5–40 Hz) contains 159 bins.", f"Got {len(useful_freqs)} bins.")

    for band_name, (low, high) in BANDS.items():
        band_bins = np.sum((useful_freqs >= low) & (useful_freqs < high))
        check(band_bins > 0, f"Band '{band_name}' ({low}-{high} Hz) has {band_bins} bins.", f"Band '{band_name}' has 0 bins.")

    # 4. Feature extraction execution
    features, names = extract_fft_features(window, fs=FS, channel_names=channel_names)
    check(features.shape == (115,), "Feature vector shape is exactly (115,).", f"Got {features.shape}.")
    check(len(names) == 115, "Feature names count == 115.", f"Got {len(names)} names.")

    # 5. Integrity and numerical stability
    check(np.isnan(features).sum() == 0, "No NaN values detected.", "NaN detected.")
    check(np.isinf(features).sum() == 0, "No Inf values detected.", "Inf detected.")

    print("\nFeature Summary:")
    print(f"  Range: [{features.min():.4f}, {features.max():.4f}] | Mean: {features.mean():.4f} | Std: {features.std():.4f}")
    print("=" * 60)
    print("ALL FFT UNIT VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()