"""Utilities for exporting timestamped model predictions."""

from pathlib import Path
import numpy as np
import pandas as pd


def _load_metadata(subject, data_dir):
    path = Path(data_dir) / f"{subject}_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")
    df = pd.read_csv(path)
    required = {
        "dataset_id", "subject_id", "edf_id", "window_index",
        "start_sec", "end_sec", "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{subject}: missing metadata columns: {sorted(missing)}")
    return df


def _validate_predictions(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("Prediction arrays have different lengths.")
    if not np.isfinite(y_prob).all():
        raise ValueError("Prediction probabilities contain NaN/Inf.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1.")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0/1.")

    return (
        y_true.astype(np.int8),
        y_pred.astype(np.int8),
        y_prob.astype(np.float64),
    )


def write_window_predictions(
    subjects,
    records,
    y_true,
    y_pred,
    y_prob,
    output_path,
    data_dir="data/processed",
):
    """
    Export predictions for records represented as:
        (subject_index, local_window_index)
    """
    y_true, y_pred, y_prob = _validate_predictions(
        y_true, y_pred, y_prob
    )
    records = list(records)

    if len(records) != len(y_pred):
        raise ValueError(
            f"Record/prediction mismatch: {len(records)} vs {len(y_pred)}"
        )

    metadata = {
        subject: _load_metadata(subject, data_dir)
        for subject in subjects
    }

    rows = []

    for record, true, pred, prob in zip(
        records, y_true, y_pred, y_prob
    ):
        subject_index, local_index = record
        subject = subjects[int(subject_index)]
        df = metadata[subject]

        if not 0 <= int(local_index) < len(df):
            raise IndexError(
                f"{subject}: invalid metadata index {local_index}"
            )

        row = df.iloc[int(local_index)]

        if int(row["label"]) != int(true):
            raise ValueError(
                f"Label mismatch at {subject}, index {local_index}: "
                f"metadata={int(row['label'])}, target={int(true)}"
            )

        rows.append({
            "dataset_id": row["dataset_id"],
            "subject_id": row["subject_id"],
            "edf_id": row["edf_id"],
            "window_index": int(row["window_index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "y_true": int(true),
            "y_pred": int(pred),
            "y_prob": float(prob),
        })

    _finalize_prediction_csv(pd.DataFrame(rows), output_path)


def write_sequence_predictions(
    subjects,
    records,
    sequence_length,
    y_true,
    y_pred,
    y_prob,
    output_path,
    data_dir="data/processed",
):
    """
    Export predictions from chronological sequence records.

    The prediction corresponds to the last/current window:
        target_index = record.start + sequence_length - 1
    """
    y_true, y_pred, y_prob = _validate_predictions(
        y_true, y_pred, y_prob
    )
    records = list(records)

    if len(records) != len(y_pred):
        raise ValueError(
            f"Record/prediction mismatch: {len(records)} vs {len(y_pred)}"
        )
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    metadata = {
        subject: _load_metadata(subject, data_dir)
        for subject in subjects
    }

    rows = []

    for record, true, pred, prob in zip(
        records, y_true, y_pred, y_prob
    ):
        subject = subjects[int(record.subject_index)]
        target_index = int(record.start) + sequence_length - 1
        df = metadata[subject]

        if not 0 <= target_index < len(df):
            raise IndexError(
                f"{subject}: invalid target index {target_index}"
            )

        row = df.iloc[target_index]

        if int(row["label"]) != int(true):
            raise ValueError(
                f"Label mismatch at {subject}, target index {target_index}: "
                f"metadata={int(row['label'])}, target={int(true)}"
            )

        rows.append({
            "dataset_id": row["dataset_id"],
            "subject_id": row["subject_id"],
            "edf_id": row["edf_id"],
            "window_index": int(row["window_index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "y_true": int(true),
            "y_pred": int(pred),
            "y_prob": float(prob),
        })

    _finalize_prediction_csv(pd.DataFrame(rows), output_path)


def _finalize_prediction_csv(df, output_path):
    if df.empty:
        raise ValueError("Prediction table is empty.")

    key = ["subject_id", "edf_id", "window_index"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate prediction rows detected.")

    df = df.sort_values(
        ["subject_id", "edf_id", "start_sec", "window_index"]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} prediction rows to {output_path}")
