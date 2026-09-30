"""FFT log band-power feature extraction for multi-channel EEG windows."""

from typing import List, Optional, Tuple
import numpy as np

FS: float = 128.0
N_SAMPLES: int = 512
EPSILON: float = 1e-8

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def extract_fft_features(
    window: np.ndarray,
    fs: float = FS,
    channel_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract log band-power features across 5 frequency bands for each EEG channel.

    Parameters
    ----------
    window : np.ndarray
        Array of shape (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    channel_names : list of str, optional
        Electrode channel labels.

    Returns
    -------
    features : np.ndarray
        1D float32 vector of shape (n_channels * 5,).
    feature_names : list of str
        Deterministic `<channel>__<band>` feature names.
    """
    window = np.asarray(window)
    if window.ndim != 2:
        raise ValueError(f"Expected 2D input (channels, samples), got shape {window.shape}")

    n_channels, n_samples = window.shape
    if n_samples != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples per window, got {n_samples}")
    if not np.isfinite(window).all():
        raise ValueError("Input window contains NaN or Inf values.")

    if channel_names is None:
        channel_names = [f"channel_{i + 1}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(f"Channel names count ({len(channel_names)}) != channels ({n_channels})")

    # Hann windowing to suppress spectral leakage before real FFT
    hann = np.hanning(n_samples)
    power = np.abs(np.fft.rfft(window * hann, axis=1)) ** 2

    # Frequency axis resolution: df = fs / N = 0.25 Hz
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    useful_mask = (frequencies >= 0.5) & (frequencies <= 40.0)
    useful_freqs = frequencies[useful_mask]
    useful_power = power[:, useful_mask]

    features = []
    feature_names = []

    for ch_idx in range(n_channels):
        ch_power = useful_power[ch_idx]
        ch_name = channel_names[ch_idx]

        for band_name, (low_f, high_f) in BANDS.items():
            band_mask = (useful_freqs >= low_f) & (useful_freqs < high_f)
            if not np.any(band_mask):
                raise ValueError(f"No FFT bins found for band {band_name} ({low_f}-{high_f} Hz)")

            # Sum power within band and apply stabilized log
            band_power = np.sum(ch_power[band_mask])
            features.append(np.log(band_power + EPSILON))
            feature_names.append(f"{ch_name}__{band_name}")

    features = np.asarray(features, dtype=np.float32)
    expected_dim = n_channels * len(BANDS)

    if features.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim} features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Extracted FFT features contain NaN or Inf values.")

    return features, feature_names


if __name__ == "__main__":
    windows = np.load("data/processed/chb01_windows.npy", mmap_mode="r")
    with open("data/processed/chb01_channel_names.txt") as f:
        ch_names = [line.strip() for line in f if line.strip()]

    feats, names = extract_fft_features(windows[0], channel_names=ch_names)
    print(f"Verified: {len(feats)} features extracted (shape: {feats.shape}, no NaNs/Infs).")