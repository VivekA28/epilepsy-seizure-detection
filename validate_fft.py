"""
Validation script for FFT feature extraction.

This script validates the FFT implementation using one real
CHB-MIT EEG window.

Checks:
    1. Input shape
    2. Sampling rate
    3. Number of channels
    4. Number of samples
    5. Frequency resolution
    6. FFT frequency bins
    7. Useful frequency range
    8. Frequency-band boundaries
    9. Number of output features
    10. Feature-name ordering
    11. NaN / Inf values
    12. Feature statistics
    13. Feature/value alignment

No neural-network training is performed.
"""


import numpy as np

from fft_features import (
    extract_fft_features,
    FS,
    N_SAMPLES,
    BANDS,
)


# =====================================================================
# Configuration
# =====================================================================

WINDOWS_PATH = (
    "data/processed/chb01_windows.npy"
)


CHANNEL_NAMES_PATH = (
    "data/processed/chb01_channel_names.txt"
)


EXPECTED_CHANNELS = 23

EXPECTED_SAMPLES = 512

EXPECTED_FS = 128.0

EXPECTED_FEATURES = 115

EXPECTED_FREQUENCY_RESOLUTION = 0.25


# =====================================================================
# Helper functions
# =====================================================================

def check(condition, success_message, failure_message):
    """
    Print PASS/FAIL and stop execution if a check fails.
    """

    if condition:

        print(
            f"[PASS] {success_message}"
        )

    else:

        print(
            f"[FAIL] {failure_message}"
        )

        raise AssertionError(
            failure_message
        )


# =====================================================================
# Start validation
# =====================================================================

print("=" * 70)

print("FFT VALIDATION")

print("=" * 70)


# =====================================================================
# 1. Load processed CHB01 data
# =====================================================================

print(
    "\n[1] Loading CHB01 EEG data..."
)


windows = np.load(
    WINDOWS_PATH,
    mmap_mode="r"
)


print(
    f"Dataset shape: {windows.shape}"
)


# =====================================================================
# 2. Validate dataset shape
# =====================================================================

print(
    "\n[2] Checking EEG window shape..."
)


window = windows[0]


check(
    window.ndim == 2,
    "EEG window is 2-dimensional.",
    f"Expected 2D EEG window, got {window.ndim}D."
)


n_channels, n_samples = (
    window.shape
)


check(
    n_channels == EXPECTED_CHANNELS,
    f"Number of channels = {EXPECTED_CHANNELS}.",
    (
        f"Expected {EXPECTED_CHANNELS} channels, "
        f"got {n_channels}."
    )
)


check(
    n_samples == EXPECTED_SAMPLES,
    f"Number of samples = {EXPECTED_SAMPLES}.",
    (
        f"Expected {EXPECTED_SAMPLES} samples, "
        f"got {n_samples}."
    )
)


# =====================================================================
# 3. Validate sampling rate
# =====================================================================

print(
    "\n[3] Checking sampling rate..."
)


check(
    FS == EXPECTED_FS,
    f"Sampling rate = {EXPECTED_FS} Hz.",
    (
        f"Expected sampling rate "
        f"{EXPECTED_FS} Hz, got {FS} Hz."
    )
)


# =====================================================================
# 4. Calculate frequency resolution
# =====================================================================

print(
    "\n[4] Checking frequency resolution..."
)


frequency_resolution = (
    FS / N_SAMPLES
)


print(
    f"Calculated frequency resolution: "
    f"{frequency_resolution:.2f} Hz"
)


check(
    np.isclose(
        frequency_resolution,
        EXPECTED_FREQUENCY_RESOLUTION
    ),
    (
        "Frequency resolution = "
        f"{EXPECTED_FREQUENCY_RESOLUTION:.2f} Hz."
    ),
    (
        f"Expected frequency resolution "
        f"{EXPECTED_FREQUENCY_RESOLUTION:.2f} Hz, "
        f"got {frequency_resolution:.2f} Hz."
    )
)


# =====================================================================
# 5. Check FFT frequency bins
# =====================================================================

print(
    "\n[5] Checking FFT frequency bins..."
)


frequencies = np.fft.rfftfreq(
    N_SAMPLES,
    d=1.0 / FS
)


expected_number_of_bins = (
    N_SAMPLES // 2 + 1
)


check(
    len(frequencies)
    == expected_number_of_bins,
    (
        f"rFFT contains "
        f"{expected_number_of_bins} frequency bins."
    ),
    (
        f"Expected "
        f"{expected_number_of_bins} frequency bins, "
        f"got {len(frequencies)}."
    )
)


print(
    f"First frequency: "
    f"{frequencies[0]:.2f} Hz"
)


print(
    f"Second frequency: "
    f"{frequencies[1]:.2f} Hz"
)


print(
    f"Last frequency: "
    f"{frequencies[-1]:.2f} Hz"
)


check(
    np.isclose(
        frequencies[0],
        0.0
    ),
    "FFT starts at 0 Hz.",
    (
        f"FFT starts at "
        f"{frequencies[0]} Hz."
    )
)


check(
    np.isclose(
        frequencies[1],
        EXPECTED_FREQUENCY_RESOLUTION
    ),
    (
        "FFT bin spacing is "
        f"{EXPECTED_FREQUENCY_RESOLUTION:.2f} Hz."
    ),
    (
        "FFT bin spacing does not match "
        "the expected resolution."
    )
)


check(
    np.isclose(
        frequencies[-1],
        FS / 2
    ),
    (
        f"Nyquist frequency = "
        f"{FS / 2:.2f} Hz."
    ),
    (
        f"Unexpected Nyquist frequency: "
        f"{frequencies[-1]:.2f} Hz."
    )
)


# =====================================================================
# 6. Check useful frequency range
# =====================================================================

print(
    "\n[6] Checking useful frequency range..."
)


useful_mask = (
    (frequencies >= 0.5)
    &
    (frequencies <= 40.0)
)


useful_frequencies = (
    frequencies[useful_mask]
)


print(
    f"Useful frequency bins: "
    f"{len(useful_frequencies)}"
)


print(
    f"First useful frequency: "
    f"{useful_frequencies[0]:.2f} Hz"
)


print(
    f"Last useful frequency: "
    f"{useful_frequencies[-1]:.2f} Hz"
)


check(
    np.isclose(
        useful_frequencies[0],
        0.5
    ),
    "Useful range starts at 0.5 Hz.",
    (
        f"Useful range starts at "
        f"{useful_frequencies[0]:.2f} Hz."
    )
)


check(
    np.isclose(
        useful_frequencies[-1],
        40.0
    ),
    "Useful range ends at 40 Hz.",
    (
        f"Useful range ends at "
        f"{useful_frequencies[-1]:.2f} Hz."
    )
)


# =====================================================================
# 7. Check frequency bands
# =====================================================================

print(
    "\n[7] Checking frequency bands..."
)


for band_name, (
    low_freq,
    high_freq
) in BANDS.items():

    band_mask = (
        (frequencies >= low_freq)
        &
        (frequencies < high_freq)
    )


    band_frequencies = (
        frequencies[band_mask]
    )


    check(
        len(band_frequencies) > 0,
        (
            f"{band_name}: "
            f"{low_freq}-{high_freq} Hz "
            f"contains "
            f"{len(band_frequencies)} bins."
        ),
        (
            f"{band_name}: no frequency bins "
            f"found."
        )
    )


    print(
        f"  {band_name:6s}: "
        f"{low_freq:4.1f} - "
        f"{high_freq:4.1f} Hz | "
        f"{len(band_frequencies):2d} bins | "
        f"{band_frequencies[0]:.2f} - "
        f"{band_frequencies[-1]:.2f} Hz"
    )


# =====================================================================
# 8. Load actual channel names
# =====================================================================

print(
    "\n[8] Checking channel names..."
)


with open(
    CHANNEL_NAMES_PATH,
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


check(
    len(channel_names)
    == EXPECTED_CHANNELS,
    (
        f"Exactly {EXPECTED_CHANNELS} "
        "channel names found."
    ),
    (
        f"Expected {EXPECTED_CHANNELS} "
        f"channel names, got "
        f"{len(channel_names)}."
    )
)


# =====================================================================
# 9. Extract FFT features
# =====================================================================

print(
    "\n[9] Extracting FFT features..."
)


features, feature_names = (
    extract_fft_features(
        window,
        fs=FS,
        channel_names=channel_names
    )
)


print(
    f"Feature vector shape: "
    f"{features.shape}"
)


# =====================================================================
# 10. Validate feature count
# =====================================================================

print(
    "\n[10] Checking feature count..."
)


check(
    len(features)
    == EXPECTED_FEATURES,
    (
        f"Exactly {EXPECTED_FEATURES} "
        "FFT features produced."
    ),
    (
        f"Expected {EXPECTED_FEATURES} "
        f"features, got {len(features)}."
    )
)


check(
    len(feature_names)
    == EXPECTED_FEATURES,
    (
        f"Exactly {EXPECTED_FEATURES} "
        "feature names produced."
    ),
    (
        f"Expected {EXPECTED_FEATURES} "
        f"feature names, got "
        f"{len(feature_names)}."
    )
)


# =====================================================================
# 11. Validate feature ordering
# =====================================================================

print(
    "\n[11] Checking feature ordering..."
)


expected_feature_names = []


for channel_name in channel_names:

    for band_name in BANDS.keys():

        expected_feature_names.append(
            f"{channel_name}__{band_name}"
        )


check(
    feature_names
    == expected_feature_names,
    (
        "Feature names follow deterministic "
        "channel-then-band ordering."
    ),
    (
        "Feature ordering does not match "
        "the expected channel-then-band order."
    )
)


# =====================================================================
# 12. Check feature/value alignment
# =====================================================================

print(
    "\n[12] Checking feature/value alignment..."
)


check(
    len(feature_names)
    == len(features),
    (
        "Every feature value has exactly "
        "one feature name."
    ),
    (
        "Feature-name count does not match "
        "feature-value count."
    )
)


# =====================================================================
# 13. Check NaN values
# =====================================================================

print(
    "\n[13] Checking NaN values..."
)


nan_count = int(
    np.isnan(features).sum()
)


print(
    f"NaN count: {nan_count}"
)


check(
    nan_count == 0,
    "No NaN values found.",
    f"Found {nan_count} NaN values."
)


# =====================================================================
# 14. Check positive infinity
# =====================================================================

print(
    "\n[14] Checking +Inf values..."
)


positive_inf_count = int(
    np.isposinf(features).sum()
)


print(
    f"+Inf count: "
    f"{positive_inf_count}"
)


check(
    positive_inf_count == 0,
    "No +Inf values found.",
    (
        f"Found {positive_inf_count} "
        "+Inf values."
    )
)


# =====================================================================
# 15. Check negative infinity
# =====================================================================

print(
    "\n[15] Checking -Inf values..."
)


negative_inf_count = int(
    np.isneginf(features).sum()
)


print(
    f"-Inf count: "
    f"{negative_inf_count}"
)


check(
    negative_inf_count == 0,
    "No -Inf values found.",
    (
        f"Found {negative_inf_count} "
        "-Inf values."
    )
)


# =====================================================================
# 16. Feature statistics
# =====================================================================

print(
    "\n[16] Feature statistics..."
)


print(
    f"Minimum: "
    f"{features.min():.6f}"
)


print(
    f"Maximum: "
    f"{features.max():.6f}"
)


print(
    f"Mean: "
    f"{features.mean():.6f}"
)


print(
    f"Standard deviation: "
    f"{features.std():.6f}"
)


# =====================================================================
# 17. Print feature samples
# =====================================================================

print(
    "\n[17] Feature/value alignment sample..."
)


print(
    "\nFirst 10 features:"
)


for name, value in zip(
    feature_names[:10],
    features[:10]
):

    print(
        f"  {name:35s} "
        f"{value:.6f}"
    )


print(
    "\nLast 5 features:"
)


for name, value in zip(
    feature_names[-5:],
    features[-5:]
):

    print(
        f"  {name:35s} "
        f"{value:.6f}"
    )


# =====================================================================
# 18. Final summary
# =====================================================================

print(
    "\n"
    + "=" * 70
)


print(
    "FFT VALIDATION SUCCESSFUL"
)


print(
    "=" * 70
)


print(
    "\nAll validation checks passed."
)


print(
    "\nValidated:"
)


print(
    "  ✓ EEG shape: 23 × 512"
)


print(
    "  ✓ Sampling rate: 128 Hz"
)


print(
    "  ✓ Frequency resolution: 0.25 Hz"
)


print(
    "  ✓ Useful range: 0.5–40 Hz"
)


print(
    "  ✓ Delta band"
)


print(
    "  ✓ Theta band"
)


print(
    "  ✓ Alpha band"
)


print(
    "  ✓ Beta band"
)


print(
    "  ✓ Gamma band"
)


print(
    "  ✓ 115 FFT features"
)


print(
    "  ✓ Deterministic feature ordering"
)


print(
    "  ✓ No NaN values"
)


print(
    "  ✓ No +Inf values"
)


print(
    "  ✓ No -Inf values"
)


print(
    "  ✓ Feature/value alignment"
)


print(
    "\nReady for FFT processing of CHB01–CHB05."
)