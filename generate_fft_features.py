"""
Generate FFT features for CHB-MIT EEG subjects.

This script applies the validated FFT feature extractor to every
EEG window for the selected subject.

Current subjects:
    chb01
    chb02
    chb03
    chb04
    chb05

Input:
    data/processed/{subject}_windows.npy
    data/processed/{subject}_metadata.csv
    data/processed/{subject}_channel_names.txt

Output:
    data/processed/{subject}_fft_features.npy
    data/processed/{subject}_fft_feature_names.txt

The output feature matrix has:

    n_windows × 115

For example:

    CHB01:
        72,951 × 115

The script processes one window at a time and writes directly
to a memory-mapped NumPy array, avoiding a large RAM allocation.
"""


import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from fft_features import (
    extract_fft_features,
    FS,
)


# =====================================================================
# Configuration
# =====================================================================

SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
]


DATA_DIR = Path(
    "data/processed"
)


EXPECTED_FEATURES = 115


# =====================================================================
# Helper: load channel names
# =====================================================================

def load_channel_names(
    path
):
    """
    Load channel names from a text file.

    One channel name is expected per line.
    """

    with open(
        path,
        "r"
    ) as f:

        channel_names = [
            line.strip()
            for line in f
            if line.strip()
        ]


    return channel_names


# =====================================================================
# Process one subject
# =====================================================================

def generate_features_for_subject(
    subject
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"FFT FEATURE GENERATION - {subject.upper()}"
    )

    print(
        "=" * 70
    )


    # =================================================================
    # Define input paths
    # =================================================================

    windows_path = (
        DATA_DIR
        / f"{subject}_windows.npy"
    )


    metadata_path = (
        DATA_DIR
        / f"{subject}_metadata.csv"
    )


    channel_names_path = (
        DATA_DIR
        / f"{subject}_channel_names.txt"
    )


    # =================================================================
    # Define output paths
    # =================================================================

    features_path = (
        DATA_DIR
        / f"{subject}_fft_features.npy"
    )


    feature_names_path = (
        DATA_DIR
        / f"{subject}_fft_feature_names.txt"
    )


    # =================================================================
    # Check input files
    # =================================================================

    required_files = [
        windows_path,
        metadata_path,
        channel_names_path,
    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )


    # =================================================================
    # Load EEG windows using memory mapping
    # =================================================================

    print(
        f"\nLoading EEG windows:"
        f"\n{windows_path}"
    )


    windows = np.load(
        windows_path,
        mmap_mode="r"
    )


    print(
        f"Input shape: "
        f"{windows.shape}"
    )


    # =================================================================
    # Validate input shape
    # =================================================================

    if windows.ndim != 3:

        raise ValueError(
            f"Expected 3D EEG array "
            f"(windows, channels, samples), "
            f"got {windows.shape}"
        )


    n_windows = windows.shape[0]

    n_channels = windows.shape[1]

    n_samples = windows.shape[2]


    if n_channels != 23:

        raise ValueError(
            f"Expected 23 channels, "
            f"got {n_channels}"
        )


    if n_samples != 512:

        raise ValueError(
            f"Expected 512 samples, "
            f"got {n_samples}"
        )


    # =================================================================
    # Load metadata
    # =================================================================

    print(
        "\nLoading metadata..."
    )


    metadata = pd.read_csv(
        metadata_path
    )


    print(
        f"Metadata shape: "
        f"{metadata.shape}"
    )


    # Metadata must contain one row per EEG window
    if len(metadata) != n_windows:

        raise ValueError(
            "Metadata/window count mismatch:\n"
            f"EEG windows: {n_windows}\n"
            f"Metadata rows: {len(metadata)}"
        )


    # =================================================================
    # Load channel names
    # =================================================================

    print(
        "\nLoading channel names..."
    )


    channel_names = load_channel_names(
        channel_names_path
    )


    print(
        f"Number of channel names: "
        f"{len(channel_names)}"
    )


    if len(channel_names) != n_channels:

        raise ValueError(
            f"Expected {n_channels} "
            f"channel names, "
            f"got {len(channel_names)}"
        )


    # =================================================================
    # Create output memory-mapped array
    # =================================================================

    print(
        "\nCreating FFT feature output..."
    )


    print(
        f"Output shape: "
        f"({n_windows}, "
        f"{EXPECTED_FEATURES})"
    )


    fft_features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            n_windows,
            EXPECTED_FEATURES
        )
    )


    # =================================================================
    # Process first window
    #
    # We use the first window to verify the extractor before
    # processing the complete subject.
    # =================================================================

    print(
        "\nTesting first EEG window..."
    )


    first_window = windows[0]


    first_features, feature_names = (
        extract_fft_features(
            first_window,
            fs=FS,
            channel_names=channel_names
        )
    )


    if len(first_features) != (
        EXPECTED_FEATURES
    ):

        raise ValueError(
            "FFT extractor returned "
            f"{len(first_features)} features. "
            f"Expected {EXPECTED_FEATURES}."
        )


    if not np.isfinite(
        first_features
    ).all():

        raise ValueError(
            "First FFT feature vector "
            "contains NaN or Inf."
        )


    # Save first result
    fft_features[0] = (
        first_features
    )


    print(
        "[PASS] First window produced "
        f"{len(first_features)} valid features."
    )


    # =================================================================
    # Process remaining windows
    # =================================================================

    print(
        "\nProcessing all EEG windows..."
    )


    # Process in simple sequential order.
    # This makes the output row index match
    # the original EEG window index.
    for i in range(
        1,
        n_windows
    ):

        window = windows[i]


        features, current_names = (
            extract_fft_features(
                window,
                fs=FS,
                channel_names=channel_names
            )
        )


        # -------------------------------------------------------------
        # Validate feature count
        # -------------------------------------------------------------

        if len(features) != (
            EXPECTED_FEATURES
        ):

            raise ValueError(
                f"Window {i}: "
                f"expected {EXPECTED_FEATURES} "
                f"features, got {len(features)}"
            )


        # -------------------------------------------------------------
        # Validate numerical values
        # -------------------------------------------------------------

        if not np.isfinite(
            features
        ).all():

            raise ValueError(
                f"Window {i}: "
                "FFT features contain NaN or Inf."
            )


        # -------------------------------------------------------------
        # Validate feature-name ordering
        #
        # It should never change between windows.
        # -------------------------------------------------------------

        if current_names != feature_names:

            raise ValueError(
                f"Window {i}: "
                "feature-name ordering changed."
            )


        # -------------------------------------------------------------
        # Save features
        # -------------------------------------------------------------

        fft_features[i] = (
            features
        )


        # -------------------------------------------------------------
        # Progress reporting
        # -------------------------------------------------------------

        if (
            (i + 1) % 5000 == 0
            or
            i == n_windows - 1
        ):

            percentage = (
                (i + 1)
                / n_windows
                * 100
            )


            print(
                f"  Processed "
                f"{i + 1:,} / "
                f"{n_windows:,} "
                f"({percentage:.1f}%)"
            )


    # =================================================================
    # Flush feature array to disk
    # =================================================================

    fft_features.flush()


    # =================================================================
    # Save feature names
    # =================================================================

    with open(
        feature_names_path,
        "w"
    ) as f:

        for name in feature_names:

            f.write(
                name + "\n"
            )


    # =================================================================
    # Final validation
    # =================================================================

    print(
        "\nRunning final output checks..."
    )


    # Check output shape
    output = np.load(
        features_path,
        mmap_mode="r"
    )


    if output.shape != (
        n_windows,
        EXPECTED_FEATURES
    ):

        raise ValueError(
            f"Unexpected output shape: "
            f"{output.shape}. "
            f"Expected "
            f"({n_windows}, "
            f"{EXPECTED_FEATURES})."
        )


    # Check numerical validity
    #
    # This checks the entire generated feature matrix.
    # It does not load the entire array into RAM.
    # NumPy operates over the memory-mapped array.
    # =================================================================

    nan_count = np.isnan(
        output
    ).sum()


    inf_count = np.isinf(
        output
    ).sum()


    print(
        f"NaN count: {nan_count}"
    )


    print(
        f"Inf count: {inf_count}"
    )


    if nan_count != 0:

        raise ValueError(
            f"Generated FFT features contain "
            f"{nan_count} NaN values."
        )


    if inf_count != 0:

        raise ValueError(
            f"Generated FFT features contain "
            f"{inf_count} Inf values."
        )


    # =================================================================
    # Final statistics
    # =================================================================

    print(
        f"Minimum: "
        f"{output.min():.6f}"
    )


    print(
        f"Maximum: "
        f"{output.max():.6f}"
    )


    print(
        f"Mean: "
        f"{output.mean():.6f}"
    )


    print(
        f"Standard deviation: "
        f"{output.std():.6f}"
    )


    # =================================================================
    # Close references
    # =================================================================

    del output
    del fft_features
    del windows
    del metadata

    gc.collect()


    # =================================================================
    # Final subject summary
    # =================================================================

    print(
        "\n"
        + "-" * 70
    )


    print(
        f"FFT GENERATION COMPLETE - "
        f"{subject.upper()}"
    )


    print(
        "-" * 70
    )


    print(
        f"Input windows: "
        f"{n_windows:,}"
    )


    print(
        f"Output features: "
        f"{EXPECTED_FEATURES}"
    )


    print(
        f"Feature matrix: "
        f"{n_windows:,} × "
        f"{EXPECTED_FEATURES}"
    )


    print(
        f"\nSaved FFT features to:"
        f"\n{features_path}"
    )


    print(
        f"\nSaved feature names to:"
        f"\n{feature_names_path}"
    )


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "CHB-MIT FFT FEATURE GENERATION"
    )

    print(
        "=" * 70
    )


    # -------------------------------------------------------------
    # Optional subject argument
    #
    # Example:
    #
    # python generate_fft_features.py chb01
    #
    # If no argument is provided, all CHB01-CHB05
    # subjects are processed.
    # -------------------------------------------------------------

    if len(sys.argv) > 1:

        subject = (
            sys.argv[1].lower()
        )


        if subject not in SUBJECTS:

            raise ValueError(
                f"Invalid subject: {subject}\n"
                f"Expected one of: "
                f"{', '.join(SUBJECTS)}"
            )


        generate_features_for_subject(
            subject
        )


    else:

        # ---------------------------------------------------------
        # Process all five subjects
        # ---------------------------------------------------------

        for subject in SUBJECTS:

            generate_features_for_subject(
                subject
            )


    print(
        "\n"
        + "=" * 70
    )


    print(
        "ALL REQUESTED FFT PROCESSING COMPLETE"
    )


    print(
        "=" * 70
    )