"""
FFT Log Band-Power Feature Extraction for Multi-Channel EEG Windows.

This module computes frequency-domain log band-power features from fixed-length
EEG time windows (typically 4.0 seconds at 128 Hz = 512 samples across 23 channels).

Clinical EEG Frequency Bands:
    - Delta (0.5 – 4.0 Hz) : Slow-wave activity; prominent during deep sleep and post-ictal slowing.
    - Theta (4.0 – 8.0 Hz) : Drowsiness, focal pathology, and rhythmic ictal onset patterns.
    - Alpha (8.0 – 13.0 Hz) : Resting posterior dominant rhythm in wakefulness with eyes closed.
    - Beta  (13.0 – 30.0 Hz): Fast activity; associated with active cognition and certain seizure morphologies.
    - Gamma (30.0 – 40.0 Hz): High-frequency synchronization; bounded at 40 Hz by preprocessing bandpass filter.

Signal Processing Pipeline:
    1. Apply Hann window taper to reduce edge discontinuities and spectral leakage.
    2. Compute one-sided Real FFT (rFFT) along the sample dimension (axis 1).
    3. Calculate power spectral density: P(f) = |X(f)|^2.
    4. Restrict to physiological range [0.5, 40.0] Hz (df = 128 / 512 = 0.25 Hz).
    5. Sum power within each frequency band for each channel.
    6. Apply stabilized natural log: log(band_power + epsilon).
    7. Output 1D feature vector of shape (n_channels * 5,) with deterministic ordering.
"""

from typing import List, Optional, Tuple
import numpy as np

# Standard acquisition sampling rate after preprocessing downsampling (Hz)
FS: float = 128.0

# Fixed window length: 4.0 seconds at 128 Hz = 512 samples
N_SAMPLES: int = 512

# Small positive constant to prevent log(0) singularity and numerical instability
EPSILON: float = 1e-8

# Standard clinical EEG frequency band boundaries in Hz [low_freq, high_freq)
# Key order defines the deterministic feature layout: channel-then-band.
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
        Multi-channel EEG array of shape (n_channels, n_samples).
        Typically (23, 512) representing 4 seconds of 23-channel EEG at 128 Hz.
    fs : float, default=128.0
        Sampling frequency in Hz. Determines FFT bin spacing df = fs / n_samples.
    channel_names : list of str, optional
        Electrode channel identifiers (e.g., ['FP1-F7', 'F7-T7', ...]).
        If None, generic identifiers ['channel_1', 'channel_2', ...] are assigned.

    Returns
    -------
    features : np.ndarray
        1D float32 array of shape (n_channels * 5,) containing log band-power values.
    feature_names : list of str
        Deterministic labels formatted as `<channel>__<band>` matching features order.

    Raises
    ------
    ValueError
        If input is not 2D, sample length != 512, contains non-finite values,
        or channel names count does not match the channel dimension.
    """
    window = np.asarray(window)
    if window.ndim != 2:
        raise ValueError(f"Expected 2D input (channels, samples), got shape {window.shape}")

    n_channels, n_samples = window.shape
    if n_samples != N_SAMPLES:
        raise ValueError(f"Expected {N_SAMPLES} samples per window, got {n_samples}")
    if not np.isfinite(window).all():
        raise ValueError("Input window contains NaN or Inf values.")

    # Assign fallback electrode labels if custom channel names are not provided
    if channel_names is None:
        channel_names = [f"channel_{i + 1}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(f"Channel names count ({len(channel_names)}) != channels ({n_channels})")

    # Step 1: Hann windowing
    # Non-periodic window boundaries cause sharp step discontinuities in DFT,
    # spreading energy into distant frequency bins (spectral leakage).
    # Hann window smoothly tapers ends to zero, concentrating main-lobe energy.
    hann = np.hanning(n_samples)

    # Step 2 & 3: Real FFT & Power Spectrum
    # Since EEG signals are real-valued, rfft computes the non-redundant positive
    # half of the spectrum: N/2 + 1 = 257 bins spanning [0, Nyquist=fs/2=64 Hz].
    # Power spectral density is obtained by squaring magnitude |X(f)|^2.
    power = np.abs(np.fft.rfft(window * hann, axis=1)) ** 2

    # Step 4: Construct frequency axis and restrict to physiological range [0.5, 40.0] Hz
    # Bin resolution: df = fs / N = 128.0 / 512 = 0.25 Hz per bin
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    useful_mask = (frequencies >= 0.5) & (frequencies <= 40.0)
    useful_freqs = frequencies[useful_mask]
    useful_power = power[:, useful_mask]

    features = []
    feature_names = []

    # Step 5 & 6: Aggregate power per clinical band and apply log compression
    for ch_idx in range(n_channels):
        ch_power = useful_power[ch_idx]
        ch_name = channel_names[ch_idx]

        for band_name, (low_f, high_f) in BANDS.items():
            # Select bins within half-open interval [low_freq, high_freq)
            band_mask = (useful_freqs >= low_f) & (useful_freqs < high_f)
            if not np.any(band_mask):
                raise ValueError(f"No FFT bins found for band {band_name} ({low_f}-{high_f} Hz)")

            # Sum total power across bins falling within the clinical band
            band_power = np.sum(ch_power[band_mask])

            # Apply natural logarithm with small epsilon to compress high dynamic range
            # and stabilize distribution for downstream gradient-based estimators
            features.append(np.log(band_power + EPSILON))
            feature_names.append(f"{ch_name}__{band_name}")

    features = np.asarray(features, dtype=np.float32)
    expected_dim = n_channels * len(BANDS)

    # Step 7: Integrity validation
    if features.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim} features, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Extracted FFT features contain NaN or Inf values.")

    return features, feature_names


if __name__ == "__main__":
    # Smoke test on the first preprocessed CHB01 EEG window
    windows = np.load("data/processed/chb01_windows.npy", mmap_mode="r")
    with open("data/processed/chb01_channel_names.txt") as f:
        ch_names = [line.strip() for line in f if line.strip()]

    feats, names = extract_fft_features(windows[0], channel_names=ch_names)
    print(f"Verified: {len(feats)} features extracted (shape: {feats.shape}, no NaNs/Infs).")