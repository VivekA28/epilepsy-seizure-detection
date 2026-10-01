"""
Unit Validation for FFT Feature Extraction on Real CHB-MIT EEG Data.

Validates mathematical properties, frequency resolution, band bin boundaries,
and numerical stability on single windows from patient recordings.

Core Mathematical Assertions:
    1. Input tensor shape: (n_channels=23, n_samples=512) for 4s epoch at 128 Hz.
    2. Discrete frequency resolution: df = fs / N = 128 / 512 = 0.25 Hz.
    3. One-sided rFFT bin count: N / 2 + 1 = 257 bins spanning [0.0, 64.0] Hz.
    4. Physiological filter range: [0.5, 40.0] Hz spanning exactly 159 bins.
    5. Band coverage: Each of the 5 clinical bands (delta, theta, alpha, beta, gamma)
       must contain non-zero bin counts.
    6. Feature dimension: Deterministic output of shape (115,) matching 23 channels × 5 bands.
    7. Numerical health: Absence of NaN or Inf values in log-power outputs.
"""

from pathlib import Path
import numpy as np
from fft_features import extract_fft_features, FS, N_SAMPLES, BANDS

# Data paths for patient CHB01 reference validation
DATA_DIR = Path("data/processed")
WINDOWS_PATH = DATA_DIR / "chb01_windows.npy"
CHANNELS_PATH = DATA_DIR / "chb01_channel_names.txt"


def check(condition: bool, success_msg: str, fail_msg: str) -> None:
    """
    Assertion helper that logs PASS/FAIL status and raises AssertionError upon failure.

    Parameters
    ----------
    condition : bool
        Evaluated boolean predicate.
    success_msg : str
        Message logged if condition is True.
    fail_msg : str
        Error message raised if condition is False.
    """
    if condition:
        print(f"  [PASS] {success_msg}")
    else:
        print(f"  [FAIL] {fail_msg}")
        raise AssertionError(fail_msg)


def main():
    """Execute complete suite of FFT mathematical and dimensional unit tests."""
    print("=" * 60)
    print("FFT FEATURE EXTRACTION VALIDATION")
    print("=" * 60)

    # 1. Load sample window and electrode channel names
    windows = np.load(WINDOWS_PATH, mmap_mode="r")
    with open(CHANNELS_PATH, "r") as f:
        channel_names = [line.strip() for line in f if line.strip()]

    window = windows[0]
    check(window.ndim == 2, "EEG window is 2D (channels, samples).", "Window is not 2D.")
    check(window.shape[0] == 23, "Channel count == 23.", f"Got {window.shape[0]} channels.")
    check(window.shape[1] == 512, "Sample count == 512.", f"Got {window.shape[1]} samples.")

    # 2. Spectral resolution and Nyquist checks
    # For N=512 samples at fs=128 Hz, bin spacing df must be 0.25 Hz
    freq_res = FS / N_SAMPLES
    check(freq_res == 0.25, f"Frequency resolution is 0.25 Hz ({FS} / {N_SAMPLES}).", "Resolution mismatch.")

    # Real FFT produces non-redundant frequencies from 0 to fs/2 (Nyquist)
    rfft_freqs = np.fft.rfftfreq(N_SAMPLES, d=1.0 / FS)
    check(len(rfft_freqs) == 257, "rFFT produces 257 frequency bins (N/2 + 1).", f"Got {len(rfft_freqs)} bins.")
    check(rfft_freqs[0] == 0.0, "FFT begins at 0.0 Hz (DC component).", "FFT does not start at 0 Hz.")
    check(rfft_freqs[-1] == 64.0, "Nyquist frequency == 64.0 Hz (fs / 2).", "Nyquist mismatch.")

    # 3. Useful band checks (0.5 - 40.0 Hz)
    # Range [0.5, 40.0] with df=0.25 Hz yields exactly (40.0 - 0.5) / 0.25 + 1 = 159 bins
    useful_mask = (rfft_freqs >= 0.5) & (rfft_freqs <= 40.0)
    useful_freqs = rfft_freqs[useful_mask]
    check(len(useful_freqs) == 159, "Useful range (0.5–40 Hz) contains 159 bins.", f"Got {len(useful_freqs)} bins.")

    # Verify that every defined clinical band has non-empty bin coverage
    for band_name, (low, high) in BANDS.items():
        band_bins = np.sum((useful_freqs >= low) & (useful_freqs < high))
        check(band_bins > 0, f"Band '{band_name}' ({low}-{high} Hz) has {band_bins} bins.", f"Band '{band_name}' has 0 bins.")

    # 4. Feature extraction execution on real window
    features, names = extract_fft_features(window, fs=FS, channel_names=channel_names)
    check(features.shape == (115,), "Feature vector shape is exactly (115,) [23 channels × 5 bands].", f"Got {features.shape}.")
    check(len(names) == 115, "Feature names count == 115.", f"Got {len(names)} names.")

    # 5. Integrity and numerical stability checks (absence of log(0) -inf / nan artifacts)
    check(np.isnan(features).sum() == 0, "No NaN values detected.", "NaN detected.")
    check(np.isinf(features).sum() == 0, "No Inf values detected.", "Inf detected.")

    print("\nFeature Summary:")
    print(f"  Range: [{features.min():.4f}, {features.max():.4f}] | Mean: {features.mean():.4f} | Std: {features.std():.4f}")
    print("=" * 60)
    print("ALL FFT UNIT VALIDATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()