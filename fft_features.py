"""
FFT feature extraction for CHB-MIT EEG.

Input:
    One EEG window with shape:
        (n_channels, n_samples)

Current expected input:
    (23, 512)

Sampling rate:
    128 Hz

FFT pipeline:
    1. Hann window
    2. Real FFT (rFFT)
    3. Power spectrum
    4. Keep 0.5-40 Hz
    5. Calculate five frequency-band powers
    6. Apply log(power + 1e-8)
    7. Return deterministic channel-then-band features

Output:
    23 channels × 5 bands = 115 features

Feature naming:
    <channel_name>__<band_name>

Example:
    FP1-F7__delta
    FP1-F7__theta
    FP1-F7__alpha
    FP1-F7__beta
    FP1-F7__gamma
"""


import numpy as np


# =====================================================================
# Configuration
# =====================================================================

FS = 128.0

N_SAMPLES = 512

EPSILON = 1e-8


# Frequency bands
#
# The order here is important because it determines
# the deterministic feature ordering.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# =====================================================================
# FFT feature extraction
# =====================================================================

def extract_fft_features(
    window,
    fs=FS,
    channel_names=None,
):
    """
    Extract FFT band-power features from one EEG window.

    Parameters
    ----------
    window : np.ndarray
        EEG data with shape:

            (n_channels, n_samples)

        Current CHB-MIT data:

            (23, 512)

    fs : float
        Sampling frequency.

        Current value:
            128 Hz

    channel_names : list[str], optional
        Actual EEG channel names in the same order as
        the channels in the input window.

        Example:

            [
                "FP1-F7",
                "F7-T7",
                "T7-P7",
                ...
            ]

        If None, generic names such as channel_1,
        channel_2, etc. are used.

    Returns
    -------
    features : np.ndarray
        One-dimensional feature vector.

        For 23 channels:

            shape = (115,)

    feature_names : list[str]
        Names corresponding to every feature.
    """

    # =================================================================
    # 1. Convert input to NumPy array
    # =================================================================

    window = np.asarray(
        window
    )


    # =================================================================
    # 2. Validate input dimensions
    # =================================================================

    if window.ndim != 2:

        raise ValueError(
            f"Expected 2D input "
            f"(channels, samples), "
            f"got shape {window.shape}"
        )


    n_channels, n_samples = (
        window.shape
    )


    # =================================================================
    # 3. Validate number of samples
    # =================================================================

    if n_samples != N_SAMPLES:

        raise ValueError(
            f"Expected {N_SAMPLES} samples "
            f"per window, "
            f"got {n_samples}"
        )


    # =================================================================
    # 4. Validate input values
    # =================================================================

    if not np.isfinite(window).all():

        raise ValueError(
            "Input window contains "
            "NaN or Inf values."
        )


    # =================================================================
    # 5. Validate / create channel names
    # =================================================================

    if channel_names is None:

        channel_names = [
            f"channel_{i + 1}"
            for i in range(n_channels)
        ]

    else:

        # Convert to list in case another sequence
        # was provided.
        channel_names = list(
            channel_names
        )


    # Number of names must match number of channels
    if len(channel_names) != n_channels:

        raise ValueError(
            f"Number of channel names "
            f"({len(channel_names)}) does not "
            f"match number of channels "
            f"({n_channels})."
        )


    # =================================================================
    # 6. Create Hann window
    # =================================================================

    hann = np.hanning(
        n_samples
    )


    # =================================================================
    # 7. Apply Hann window
    #
    # window:
    #     (channels, samples)
    #
    # hann:
    #     (samples,)
    #
    # NumPy broadcasting applies the Hann window
    # independently to every channel.
    # =================================================================

    windowed_signal = (
        window * hann
    )


    # =================================================================
    # 8. Real FFT
    # =================================================================

    fft_result = np.fft.rfft(
        windowed_signal,
        axis=1
    )


    # =================================================================
    # 9. Convert FFT to power spectrum
    # =================================================================

    power = (
        np.abs(fft_result) ** 2
    )


    # =================================================================
    # 10. Create frequency axis
    #
    # For:
    #
    #     N  = 512
    #     fs = 128 Hz
    #
    # frequency resolution:
    #
    #     fs / N
    #     = 128 / 512
    #     = 0.25 Hz
    #
    # rFFT frequencies:
    #
    #     0.00
    #     0.25
    #     0.50
    #     ...
    #     64.00 Hz
    # =================================================================

    frequencies = np.fft.rfftfreq(
        n_samples,
        d=1.0 / fs
    )


    # =================================================================
    # 11. Keep useful frequency range
    #
    # Required range:
    #
    #     0.5 Hz <= f <= 40 Hz
    # =================================================================

    useful_frequency_mask = (
        (frequencies >= 0.5)
        &
        (frequencies <= 40.0)
    )


    useful_frequencies = (
        frequencies[
            useful_frequency_mask
        ]
    )


    useful_power = (
        power[
            :,
            useful_frequency_mask
        ]
    )


    # =================================================================
    # 12. Extract band powers
    # =================================================================

    features = []

    feature_names = []


    # =================================================================
    # IMPORTANT FEATURE ORDER
    #
    # Channel 1:
    #     delta
    #     theta
    #     alpha
    #     beta
    #     gamma
    #
    # Channel 2:
    #     delta
    #     theta
    #     alpha
    #     beta
    #     gamma
    #
    # ...
    #
    # Channel 23:
    #     delta
    #     theta
    #     alpha
    #     beta
    #     gamma
    #
    # Total:
    #
    #     23 × 5 = 115 features
    # =================================================================

    for channel_idx in range(
        n_channels
    ):

        channel_power = (
            useful_power[
                channel_idx
            ]
        )


        for band_name, (
            low_freq,
            high_freq
        ) in BANDS.items():

            # ---------------------------------------------------------
            # Find frequency bins belonging to this band
            #
            # Lower boundary included.
            # Upper boundary excluded.
            #
            # Example:
            # Delta = 0.5 <= f < 4.0
            # ---------------------------------------------------------

            band_mask = (
                (useful_frequencies >= low_freq)
                &
                (useful_frequencies < high_freq)
            )


            # ---------------------------------------------------------
            # Make sure the band contains frequency bins
            # ---------------------------------------------------------

            if not np.any(
                band_mask
            ):

                raise ValueError(
                    f"No FFT frequency bins "
                    f"found for band "
                    f"{band_name}: "
                    f"{low_freq}-{high_freq} Hz"
                )


            # ---------------------------------------------------------
            # Calculate band power
            # ---------------------------------------------------------

            band_power = np.sum(
                channel_power[
                    band_mask
                ]
            )


            # ---------------------------------------------------------
            # Log transform
            #
            # Required:
            #
            # log(band_power + 1e-8)
            # ---------------------------------------------------------

            log_band_power = np.log(
                band_power + EPSILON
            )


            # ---------------------------------------------------------
            # Store feature value
            # ---------------------------------------------------------

            features.append(
                log_band_power
            )


            # ---------------------------------------------------------
            # Store actual channel name + band
            # ---------------------------------------------------------

            feature_names.append(
                f"{channel_names[channel_idx]}"
                f"__{band_name}"
            )


    # =================================================================
    # 13. Convert features to NumPy array
    # =================================================================

    features = np.asarray(
        features,
        dtype=np.float32
    )


    # =================================================================
    # 14. Validate output shape
    # =================================================================

    expected_features = (
        n_channels * len(BANDS)
    )


    if features.shape != (
        expected_features,
    ):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}. "
            f"Expected "
            f"({expected_features},)"
        )


    # =================================================================
    # 15. Validate feature names
    # =================================================================

    if len(feature_names) != (
        expected_features
    ):

        raise ValueError(
            f"Expected "
            f"{expected_features} feature names, "
            f"got {len(feature_names)}"
        )


    # =================================================================
    # 16. Validate final feature values
    # =================================================================

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            "FFT features contain "
            "NaN or Inf values."
        )


    # =================================================================
    # 17. Return
    # =================================================================

    return (
        features,
        feature_names
    )


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "FFT FEATURE EXTRACTION TEST"
    )

    print(
        "=" * 60
    )


    # =================================================================
    # Load CHB01 processed windows
    # =================================================================

    data_path = (
        "data/processed/"
        "chb01_windows.npy"
    )


    print(
        f"\nLoading: {data_path}"
    )


    windows = np.load(
        data_path,
        mmap_mode="r"
    )


    print(
        f"Dataset shape: "
        f"{windows.shape}"
    )


    # =================================================================
    # Select first window
    # =================================================================

    window = windows[0]


    print(
        f"Single window shape: "
        f"{window.shape}"
    )


    # =================================================================
    # Load actual CHB01 channel names
    # =================================================================

    channel_names_path = (
        "data/processed/"
        "chb01_channel_names.txt"
    )


    with open(
        channel_names_path,
        "r"
    ) as f:

        channel_names = [
            line.strip()
            for line in f
            if line.strip()
        ]


    print(
        f"Channel names loaded: "
        f"{len(channel_names)}"
    )


    # =================================================================
    # Extract FFT features
    # =================================================================

    (
        features,
        feature_names
    ) = extract_fft_features(
        window,
        fs=FS,
        channel_names=channel_names
    )


    # =================================================================
    # Print basic information
    # =================================================================

    print(
        f"\nSampling rate: "
        f"{FS} Hz"
    )


    print(
        f"Samples: "
        f"{window.shape[1]}"
    )


    print(
        f"Frequency resolution: "
        f"{FS / window.shape[1]:.2f} Hz"
    )


    print(
        f"Number of channels: "
        f"{window.shape[0]}"
    )


    print(
        f"Number of frequency bands: "
        f"{len(BANDS)}"
    )


    print(
        f"Number of FFT features: "
        f"{len(features)}"
    )


    # =================================================================
    # Print first 15 features
    # =================================================================

    print(
        "\nFirst 15 features:"
    )


    for name, value in zip(
        feature_names[:15],
        features[:15]
    ):

        print(
            f"{name:35s} "
            f"{value:.6f}"
        )


    # =================================================================
    # Print last 5 features
    # =================================================================

    print(
        "\nLast 5 features:"
    )


    for name, value in zip(
        feature_names[-5:],
        features[-5:]
    ):

        print(
            f"{name:35s} "
            f"{value:.6f}"
        )


    # =================================================================
    # Final validation checks
    # =================================================================

    print(
        "\nNaN count:",
        np.isnan(features).sum()
    )


    print(
        "Inf count:",
        np.isinf(features).sum()
    )


    print(
        "Feature vector shape:",
        features.shape
    )


    print(
        "\nFFT test completed."
    )