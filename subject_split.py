"""
Reproducible subject-level train/validation/test splitting for EEG experiments.

This module defines the FINAL patient-independent split interface.

Responsibilities
----------------
1. Validate subject metadata and core processed arrays.
2. Enforce disjoint, duplicate-free subject partitions.
3. Report partition statistics.
4. Save/load the exact split as a JSON manifest.
5. Provide subject-index mappings that model-specific datasets can consume.

Important
---------
This module does NOT construct CNN/LSTM datasets and does NOT contain
model-specific training logic. Those responsibilities stay in the
corresponding training scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "results" / "subject_split.json"

REQUIRED_METADATA_COLUMNS = {
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
}


def _as_list(subjects: Iterable[str], partition_name: str) -> list[str]:
    """Normalize a subject iterable and reject empty/invalid entries."""
    result = list(subjects)

    if not result:
        raise ValueError(f"{partition_name} subject list is empty.")

    if any(not isinstance(subject, str) or not subject.strip() for subject in result):
        raise ValueError(
            f"{partition_name} subject list contains an invalid subject: {result}"
        )

    return result


def _check_duplicates(subjects: Sequence[str], partition_name: str) -> None:
    """Reject duplicate subjects instead of masking them with set()."""
    if len(subjects) != len(set(subjects)):
        raise ValueError(
            f"{partition_name} subject list contains duplicates: {list(subjects)}"
        )


def validate_split(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Validate and normalize train/validation/test subject partitions.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Normalized subject lists in train, validation, test order.
    """
    train = _as_list(train_subjects, "Train")
    val = _as_list(val_subjects, "Validation")
    test = _as_list(test_subjects, "Test")

    _check_duplicates(train, "Train")
    _check_duplicates(val, "Validation")
    _check_duplicates(test, "Test")

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    overlaps = [
        ("Train/Validation", train_set & val_set),
        ("Train/Test", train_set & test_set),
        ("Validation/Test", val_set & test_set),
    ]

    for name, overlap in overlaps:
        if overlap:
            raise ValueError(f"{name} subject overlap: {sorted(overlap)}")

    return train, val, test


def _array_path(subject: str, name: str) -> Path:
    """Return the standard processed-array path for a subject."""
    if name == "cnn_features":
        return DATA_DIR / "cnn_features" / f"{subject}_cnn_features.npy"

    return DATA_DIR / f"{subject}_{name}.npy"


def load_subject_metadata(
    subject: str,
    *,
    expected_sampling_rate: float = 128.0,
    expected_channels: int = 23,
    expected_samples: int = 512,
) -> pd.DataFrame:
    """
    Load and structurally validate one subject's metadata CSV.

    The defaults match the current CHB-MIT preprocessing pipeline. They
    can be overridden later when another dataset uses a different
    representation.
    """
    path = DATA_DIR / f"{subject}_metadata.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_METADATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{subject}: missing metadata columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{subject}: metadata CSV is empty.")

    if not (df["subject_id"] == subject).all():
        raise ValueError(f"{subject}: metadata contains incorrect subject IDs.")

    if df["dataset_id"].isna().any() or df["edf_id"].isna().any():
        raise ValueError(f"{subject}: dataset_id/edf_id contains missing values.")

    if not (df["end_sec"] > df["start_sec"]).all():
        raise ValueError(f"{subject}: invalid window time intervals.")

    if not np.isclose(
        df["sampling_rate"].to_numpy(),
        expected_sampling_rate,
    ).all():
        raise ValueError(
            f"{subject}: inconsistent sampling rate; expected "
            f"{expected_sampling_rate} Hz."
        )

    if not (df["n_channels"] == expected_channels).all():
        raise ValueError(
            f"{subject}: inconsistent channel count; expected {expected_channels}."
        )

    if not (df["n_samples"] == expected_samples).all():
        raise ValueError(
            f"{subject}: inconsistent sample count; expected {expected_samples}."
        )

    labels = df["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{subject}: metadata labels must be binary 0/1.")

    if df.duplicated(["edf_id", "window_index"]).any():
        raise ValueError(
            f"{subject}: duplicate (edf_id, window_index) metadata rows detected."
        )

    # Every EDF should form one contiguous chronological block. This is
    # required by the existing sequence construction logic.
    edf_change_count = int(df["edf_id"].ne(df["edf_id"].shift()).sum())
    if edf_change_count != int(df["edf_id"].nunique()):
        raise ValueError(
            f"{subject}: at least one EDF appears in multiple non-contiguous blocks."
        )

    # Window indices should restart at zero and increase by one within each EDF.
    for edf_id, group in df.groupby("edf_id", sort=False):
        expected_indices = np.arange(len(group), dtype=group["window_index"].dtype)
        actual_indices = group["window_index"].to_numpy()
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(
                f"{subject}/{edf_id}: window_index is not contiguous from 0."
            )

    return df


def validate_subject_data(
    subject: str,
    *,
    check_fft: bool = True,
    check_cnn_features: bool = True,
    expected_sampling_rate: float = 128.0,
    expected_channels: int = 23,
    expected_samples: int = 512,
    expected_fft_dim: int = 115,
    expected_cnn_dim: int = 128,
) -> pd.DataFrame:
    """
    Validate metadata and alignment of processed arrays for one subject.

    Core arrays:
        <subject>_windows.npy
        <subject>_labels.npy
        <subject>_file_ids.npy

    Optional feature caches are validated when their files exist and the
    corresponding check is enabled.
    """
    df = load_subject_metadata(
        subject,
        expected_sampling_rate=expected_sampling_rate,
        expected_channels=expected_channels,
        expected_samples=expected_samples,
    )

    labels_path = _array_path(subject, "labels")
    file_ids_path = _array_path(subject, "file_ids")
    windows_path = _array_path(subject, "windows")

    for path in (labels_path, file_ids_path, windows_path):
        if not path.exists():
            raise FileNotFoundError(f"{subject}: missing processed array: {path}")

    labels = np.load(labels_path, mmap_mode="r")
    file_ids = np.load(file_ids_path, mmap_mode="r")
    windows = np.load(windows_path, mmap_mode="r")

    n_rows = len(df)

    if labels.ndim != 1 or len(labels) != n_rows:
        raise ValueError(
            f"{subject}: labels shape {labels.shape} does not match "
            f"metadata rows {n_rows}."
        )

    if file_ids.ndim != 1 or len(file_ids) != n_rows:
        raise ValueError(
            f"{subject}: file_ids shape {file_ids.shape} does not match "
            f"metadata rows {n_rows}."
        )

    if windows.ndim != 3 or windows.shape[0] != n_rows:
        raise ValueError(
            f"{subject}: windows shape {windows.shape} does not match "
            f"metadata rows {n_rows}."
        )

    if windows.shape[1:] != (expected_channels, expected_samples):
        raise ValueError(
            f"{subject}: expected windows shape (*, {expected_channels}, "
            f"{expected_samples}), got {windows.shape}."
        )

    if not np.isfinite(np.asarray(labels)).all():
        raise ValueError(f"{subject}: labels contain NaN/Inf.")

    if not np.isin(np.asarray(labels), [0, 1]).all():
        raise ValueError(f"{subject}: array labels must be binary 0/1.")

    # Metadata and labels are row-aligned.
    metadata_labels = df["label"].to_numpy(dtype=np.int8)
    array_labels = np.asarray(labels, dtype=np.int8)
    if not np.array_equal(metadata_labels, array_labels):
        raise ValueError(
            f"{subject}: metadata labels and labels.npy are not row-aligned."
        )

    # Each EDF must correspond to exactly one integer file ID and vice versa.
    edf_to_file = pd.DataFrame(
        {
            "edf_id": df["edf_id"].to_numpy(),
            "file_id": np.asarray(file_ids),
        }
    ).drop_duplicates()

    if edf_to_file.groupby("edf_id")["file_id"].nunique().max() != 1:
        raise ValueError(f"{subject}: an EDF maps to multiple file IDs.")

    if edf_to_file.groupby("file_id")["edf_id"].nunique().max() != 1:
        raise ValueError(f"{subject}: a file ID maps to multiple EDF IDs.")

    if check_fft:
        fft_path = _array_path(subject, "fft_features")
        if fft_path.exists():
            fft = np.load(fft_path, mmap_mode="r")
            if fft.ndim != 2 or fft.shape[0] != n_rows:
                raise ValueError(
                    f"{subject}: FFT feature shape {fft.shape} does not "
                    f"match metadata rows {n_rows}."
                )
            if fft.shape[1] != expected_fft_dim:
                raise ValueError(
                    f"{subject}: expected {expected_fft_dim} FFT features, "
                    f"got {fft.shape[1]}."
                )
            if not np.isfinite(np.asarray(fft)).all():
                raise ValueError(f"{subject}: FFT cache contains NaN/Inf.")

    if check_cnn_features:
        cnn_path = _array_path(subject, "cnn_features")
        if cnn_path.exists():
            cnn = np.load(cnn_path, mmap_mode="r")
            if cnn.ndim != 2 or cnn.shape[0] != n_rows:
                raise ValueError(
                    f"{subject}: CNN feature shape {cnn.shape} does not "
                    f"match metadata rows {n_rows}."
                )
            if cnn.shape[1] != expected_cnn_dim:
                raise ValueError(
                    f"{subject}: expected {expected_cnn_dim} CNN features, "
                    f"got {cnn.shape[1]}."
                )
            if not np.isfinite(np.asarray(cnn)).all():
                raise ValueError(f"{subject}: CNN cache contains NaN/Inf.")

    return df


def validate_subjects(
    subjects: Sequence[str],
    *,
    check_fft: bool = True,
    check_cnn_features: bool = True,
) -> dict[str, pd.DataFrame]:
    """Validate every requested subject and return their metadata."""
    result: dict[str, pd.DataFrame] = {}

    for subject in subjects:
        result[subject] = validate_subject_data(
            subject,
            check_fft=check_fft,
            check_cnn_features=check_cnn_features,
        )

    return result


def summarize_partition(
    name: str,
    subjects: Sequence[str],
    metadata_by_subject: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Print and return window/EDF/seizure statistics for one partition."""
    frames = [metadata_by_subject[subject] for subject in subjects]
    combined = pd.concat(frames, ignore_index=True)

    print(f"\n{name.upper()}")
    print("-" * 60)
    print(f"Subjects        : {len(subjects):,}")
    print(
        "EDFs            : "
        f"{len(combined[['subject_id', 'edf_id']].drop_duplicates()):,}"
    )
    print(f"Windows         : {len(combined):,}")
    print(f"Seizure windows : {int(combined['label'].sum()):,}")
    print(f"Seizure ratio   : {100 * combined['label'].mean():.4f}%")

    print("Subjects:")
    for subject in subjects:
        sub = metadata_by_subject[subject]
        print(
            f"  {subject}: "
            f"{len(sub):,} windows, "
            f"{sub['edf_id'].nunique():,} EDFs, "
            f"{int(sub['label'].sum()):,} seizure windows"
        )

    return combined


def build_partition_subject_indices(
    subjects_order: Sequence[str],
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> dict[str, list[int]]:
    """
    Map ordered subject names to integer positions.

    These indices are useful to model-specific code that stores feature
    arrays as a list ordered like `subjects_order`.
    """
    train, val, test = validate_split(
        train_subjects,
        val_subjects,
        test_subjects,
    )

    subjects_order = list(subjects_order)
    if len(subjects_order) != len(set(subjects_order)):
        raise ValueError("subjects_order contains duplicates.")

    subject_to_index = {
        subject: index
        for index, subject in enumerate(subjects_order)
    }

    requested = set(train + val + test)
    missing_from_order = requested - set(subjects_order)
    if missing_from_order:
        raise ValueError(
            "Split contains subjects missing from subjects_order: "
            f"{sorted(missing_from_order)}"
        )

    return {
        "train": [subject_to_index[s] for s in train],
        "validation": [subject_to_index[s] for s in val],
        "test": [subject_to_index[s] for s in test],
    }


def save_split_manifest(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
    output_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Save the exact subject split as a reproducible JSON manifest."""
    train, val, test = validate_split(
        train_subjects,
        val_subjects,
        test_subjects,
    )

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    manifest = {
        "format_version": 1,
        "split_type": "subject_level",
        "dataset": "CHB-MIT",
        "train": train,
        "validation": val,
        "test": test,
        "all_subjects": train + val + test,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nSplit manifest saved to: {output_path}")
    return output_path


def load_split_manifest(
    input_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, list[str]]:
    """Load and validate a previously saved subject split manifest."""
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_keys = {"train", "validation", "test"}
    missing = required_keys - set(manifest)
    if missing:
        raise ValueError(
            f"Split manifest is missing keys: {sorted(missing)}"
        )

    train, val, test = validate_split(
        manifest["train"],
        manifest["validation"],
        manifest["test"],
    )

    all_subjects = train + val + test
    if "all_subjects" in manifest and list(manifest["all_subjects"]) != all_subjects:
        raise ValueError(
            "Split manifest `all_subjects` does not match its partitions."
        )

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def create_subject_split(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    save_manifest: bool = True,
    check_fft: bool = True,
    check_cnn_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate data, summarize partitions, and optionally persist the split.

    This function is intended to be called after a final subject allocation
    has been chosen. It does not select subjects automatically.
    """
    train, val, test = validate_split(
        train_subjects,
        val_subjects,
        test_subjects,
    )

    all_subjects = train + val + test
    metadata_by_subject = validate_subjects(
        all_subjects,
        check_fft=check_fft,
        check_cnn_features=check_cnn_features,
    )

    train_df = summarize_partition("train", train, metadata_by_subject)
    val_df = summarize_partition("validation", val, metadata_by_subject)
    test_df = summarize_partition("test", test, metadata_by_subject)

    if save_manifest:
        save_split_manifest(
            train,
            val,
            test,
            manifest_path,
        )

    print("\n" + "=" * 60)
    print("SUBJECT-LEVEL SPLIT VALIDATION")
    print("=" * 60)
    print("[PASS] Train/validation/test subjects are disjoint.")
    print("[PASS] No partition contains duplicate subjects.")
    print("[PASS] Metadata/core arrays are aligned.")
    print("[PASS] EDFs are contiguous and have valid window indices.")
    print("[PASS] Sampling rate/channel count/window size are consistent.")
    if check_fft:
        print("[PASS] Available FFT caches are aligned and finite.")
    if check_cnn_features:
        print("[PASS] Available CNN caches are aligned and finite.")

    return train_df, val_df, test_df


if __name__ == "__main__":
    TRAIN_SUBJECTS = [
        "chb01",
        "chb03",
        "chb04",
        "chb06",
        "chb09",
        "chb10",
        "chb11",
        "chb12",
        "chb14",
        "chb15",
    ]

    VAL_SUBJECTS = [
        "chb07",
        "chb08",
    ]

    TEST_SUBJECTS = [
        "chb02",
        "chb05",
        "chb13",
    ]

    create_subject_split(
        TRAIN_SUBJECTS,
        VAL_SUBJECTS,
        TEST_SUBJECTS,
    )

