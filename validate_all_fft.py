import os
import numpy as np
import pandas as pd


DATA_DIR = "data/processed"

SUBJECTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
]

EXPECTED_FEATURES = 115
EXPECTED_CHANNELS = 23
EXPECTED_SAMPLES = 512


print("=" * 70)
print("CHB-MIT FFT DATASET VALIDATION")
print("=" * 70)

total_windows = 0

for subject in SUBJECTS:

    print("\n" + "-" * 70)
    print(f"VALIDATING {subject.upper()}")
    print("-" * 70)

    fft_path = os.path.join(
        DATA_DIR,
        f"{subject}_fft_features.npy"
    )

    metadata_path = os.path.join(
        DATA_DIR,
        f"{subject}_metadata.csv"
    )

    channel_path = os.path.join(
        DATA_DIR,
        f"{subject}_channel_names.txt"
    )

    feature_names_path = os.path.join(
        DATA_DIR,
        f"{subject}_fft_feature_names.txt"
    )

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    required_files = [
        fft_path,
        metadata_path,
        channel_path,
        feature_names_path,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    fft = np.load(fft_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)

    with open(channel_path, "r") as f:
        channels = [line.strip() for line in f if line.strip()]

    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------

    print(f"FFT shape       : {fft.shape}")
    print(f"Metadata shape  : {metadata.shape}")
    print(f"Channels        : {len(channels)}")
    print(f"Feature names   : {len(feature_names)}")

    assert fft.ndim == 2, "FFT data must be 2D"

    assert fft.shape[1] == EXPECTED_FEATURES, (
        f"Expected {EXPECTED_FEATURES} FFT features, "
        f"got {fft.shape[1]}"
    )

    assert len(channels) == EXPECTED_CHANNELS, (
        f"Expected {EXPECTED_CHANNELS} channels, "
        f"got {len(channels)}"
    )

    assert len(feature_names) == EXPECTED_FEATURES, (
        f"Expected {EXPECTED_FEATURES} feature names, "
        f"got {len(feature_names)}"
    )

    # ---------------------------------------------------------
    # Row alignment
    # ---------------------------------------------------------

    assert len(fft) == len(metadata), (
        "FFT rows and metadata rows do not match"
    )

    print(f"[PASS] FFT rows = metadata rows")

    # ---------------------------------------------------------
    # NaN / Inf
    # ---------------------------------------------------------

    nan_count = np.isnan(fft).sum()
    inf_count = np.isinf(fft).sum()

    print(f"NaN count       : {nan_count}")
    print(f"Inf count       : {inf_count}")

    assert nan_count == 0, "NaN values found"
    assert inf_count == 0, "Inf values found"

    print("[PASS] No NaN / Inf")

    # ---------------------------------------------------------
    # Metadata checks
    # ---------------------------------------------------------

    required_metadata = [
        "dataset_id",
        "subject_id",
        "edf_id",
        "window_index",
        "start_sec",
        "end_sec",
        "label",
        "sampling_rate",
        "n_channels",
        "n_samples",
    ]

    for column in required_metadata:
        assert column in metadata.columns, (
            f"Missing metadata column: {column}"
        )

    print("[PASS] Required metadata columns present")

    # ---------------------------------------------------------
    # Metadata values
    # ---------------------------------------------------------

    assert metadata["subject_id"].nunique() == 1
    assert metadata["subject_id"].iloc[0] == subject

    assert metadata["sampling_rate"].eq(128).all()
    assert metadata["n_channels"].eq(23).all()
    assert metadata["n_samples"].eq(512).all()

    print("[PASS] Sampling rate = 128 Hz")
    print("[PASS] Channels = 23")
    print("[PASS] Samples/window = 512")

    # ---------------------------------------------------------
    # Feature ordering
    # ---------------------------------------------------------

    expected_names = []

    bands = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    ]

    for channel in channels:
        for band in bands:
            expected_names.append(
                f"{channel}__{band}"
            )

    assert feature_names == expected_names, (
        "Feature ordering does not match "
        "channel-then-band ordering"
    )

    print("[PASS] Feature ordering is deterministic")

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print(f"Minimum         : {fft.min():.6f}")
    print(f"Maximum         : {fft.max():.6f}")
    print(f"Mean            : {fft.mean():.6f}")
    print(f"Std deviation   : {fft.std():.6f}")

    total_windows += len(fft)

    print(f"[PASS] {subject.upper()} validation successful")


print("\n" + "=" * 70)
print("ALL FFT DATASETS VALIDATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSubjects validated : {len(SUBJECTS)}")
print(f"Total windows      : {total_windows:,}")
print(f"Features/window    : {EXPECTED_FEATURES}")
print(f"Channels           : {EXPECTED_CHANNELS}")
print(f"Samples/window     : {EXPECTED_SAMPLES}")

print("\nNo NaN values")
print("No Inf values")
print("115 FFT features per window")
print("Metadata row alignment verified")
print("Deterministic feature ordering verified")

print("\nFFT DATASET VALIDATION COMPLETE")