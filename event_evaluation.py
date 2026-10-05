"""Event-level seizure evaluation from timestamped predictions.

Initial frozen-style rules:
- Positive prediction windows are grouped when their start-to-start gap is
  <= max_gap_sec. Default max_gap_sec=4.0, which bridges one missing
  2-second prediction step for the current 50%-overlap windowing.
- A predicted event overlapping a true seizure is not a false alarm.
- The first predicted event overlapping a true seizure can detect that event.
- Additional predicted events overlapping the same already-detected seizure
  are classified as duplicate_ictal.
- A predicted event with no overlap with any true seizure is a false alarm.
- Detection time is the end of the first matched positive window.
- Detection latency = detection time - annotated seizure onset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "subject_id",
    "edf_id",
    "window_index",
    "start_sec",
    "end_sec",
    "y_true",
    "y_pred",
    "y_prob",
}


def parse_seizures_for_file(summary_path: Path, target_filename: str):
    """Parse CHB-MIT seizure start/end times for one EDF."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    text = summary_path.read_text(encoding="utf-8")
    chunks = text.split("File Name:")
    seizures = []

    for chunk in chunks:
        if target_filename not in chunk.split("\n")[0]:
            continue

        starts = re.findall(
            r"Seizure Start Time:\s*(\d+)\s*seconds",
            chunk,
        )
        ends = re.findall(
            r"Seizure End Time:\s*(\d+)\s*seconds",
            chunk,
        )

        if not starts:
            starts = re.findall(
                r"Seizure \d+ Start Time:\s*(\d+)\s*seconds",
                chunk,
            )
            ends = re.findall(
                r"Seizure \d+ End Time:\s*(\d+)\s*seconds",
                chunk,
            )

        for start, end in zip(starts, ends):
            seizures.append((float(start), float(end)))

    return seizures


def load_predictions(path: Path):
    """Load and validate timestamped window/sequence predictions."""
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Prediction CSV is empty.")

    key = ["subject_id", "edf_id", "window_index"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate prediction rows detected.")

    for column in ["start_sec", "end_sec", "y_prob"]:
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} contains NaN/Inf.")

    if not (df["end_sec"] > df["start_sec"]).all():
        raise ValueError("Invalid prediction window intervals.")

    if not np.isin(df["y_true"], [0, 1]).all():
        raise ValueError("y_true must contain only 0/1.")

    if not np.isin(df["y_pred"], [0, 1]).all():
        raise ValueError("y_pred must contain only 0/1.")

    return df.sort_values(
        ["subject_id", "edf_id", "start_sec", "window_index"]
    ).reset_index(drop=True)


def build_predicted_events(df, max_gap_sec=4.0):
    """
    Group positive predictions within the same EDF.

    `max_gap_sec` is measured between the START times of adjacent positive
    prediction windows. With 2-second window starts, 4 seconds bridges one
    missing prediction window.
    """
    if max_gap_sec < 2.0:
        raise ValueError(
            "max_gap_sec must be >= the normal 2-second window step."
        )

    events = []
    next_event_id = 0

    for (subject_id, edf_id), group in df.groupby(
        ["subject_id", "edf_id"],
        sort=False,
    ):
        positives = group[
            group["y_pred"] == 1
        ].sort_values(["start_sec", "window_index"])

        current = None

        for row in positives.itertuples(index=False):
            start = float(row.start_sec)
            end = float(row.end_sec)

            if current is None:
                current = {
                    "subject_id": subject_id,
                    "edf_id": edf_id,
                    "start_sec": start,
                    "end_sec": end,
                    "detection_sec": end,
                    "n_positive_windows": 1,
                    "last_positive_start_sec": start,
                }
                continue

            gap = start - current["last_positive_start_sec"]

            if gap <= max_gap_sec + 1e-6:
                current["end_sec"] = max(
                    current["end_sec"],
                    end,
                )
                current["n_positive_windows"] += 1
                current["last_positive_start_sec"] = start
            else:
                current["event_id"] = next_event_id
                events.append(current)
                next_event_id += 1

                current = {
                    "subject_id": subject_id,
                    "edf_id": edf_id,
                    "start_sec": start,
                    "end_sec": end,
                    "detection_sec": end,
                    "n_positive_windows": 1,
                    "last_positive_start_sec": start,
                }

        if current is not None:
            current["event_id"] = next_event_id
            events.append(current)
            next_event_id += 1

    result = pd.DataFrame(events)

    if not result.empty:
        result = result.drop(
            columns=["last_positive_start_sec"],
            errors="ignore",
        )

    return result


def _merge_intervals(intervals):
    """Merge overlapping true seizure intervals."""
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )
        else:
            merged.append([start, end])

    return [
        (float(start), float(end))
        for start, end in merged
    ]


def build_true_events(df, summary_dir):
    """Build true seizure events from original CHB-MIT annotations."""
    records = []

    pairs = df[
        ["subject_id", "edf_id"]
    ].drop_duplicates()

    for row in pairs.itertuples(index=False):
        subject = str(row.subject_id)
        edf_id = str(row.edf_id)

        summary_path = (
            summary_dir
            / subject
            / f"{subject}-summary.txt"
        )

        intervals = parse_seizures_for_file(
            summary_path,
            f"{edf_id}.edf",
        )

        merged = _merge_intervals(intervals)

        for true_event_id, (start, end) in enumerate(merged):
            records.append({
                "subject_id": subject,
                "edf_id": edf_id,
                "true_event_id": true_event_id,
                "start_sec": start,
                "end_sec": end,
            })

    return pd.DataFrame(records)


def _overlap_seconds(
    pred_start,
    pred_end,
    true_start,
    true_end,
):
    return max(
        0.0,
        min(pred_end, true_end)
        - max(pred_start, true_start),
    )


def match_events(predicted_events, true_events):
    """
    Classify every predicted event as:
        detected
        duplicate_ictal
        false_alarm

    The first overlapping predicted event for a true seizure is the
    detection. Later predicted events overlapping that same true seizure
    are duplicate_ictal, not false alarms.

    If a predicted event overlaps multiple true seizures, the true seizure
    with the largest temporal overlap is selected.
    """
    if predicted_events.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "edf_id",
                "pred_event_id",
                "true_event_id",
                "classification",
                "detection_sec",
                "true_onset_sec",
                "true_end_sec",
                "latency_sec",
                "overlap_sec",
            ]
        )

    rows = []
    detected_true = set()

    true_records = list(
        true_events.itertuples(index=False)
    )

    for pred in predicted_events.itertuples(index=False):
        candidates = []

        for true_event in true_records:
            if (
                true_event.subject_id != pred.subject_id
                or true_event.edf_id != pred.edf_id
            ):
                continue

            overlap = _overlap_seconds(
                float(pred.start_sec),
                float(pred.end_sec),
                float(true_event.start_sec),
                float(true_event.end_sec),
            )

            if overlap > 0:
                candidates.append(
                    (overlap, true_event)
                )

        if not candidates:
            rows.append({
                "subject_id": pred.subject_id,
                "edf_id": pred.edf_id,
                "pred_event_id": pred.event_id,
                "true_event_id": np.nan,
                "classification": "false_alarm",
                "detection_sec": pred.detection_sec,
                "true_onset_sec": np.nan,
                "true_end_sec": np.nan,
                "latency_sec": np.nan,
                "overlap_sec": 0.0,
            })
            continue

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1].start_sec,
                item[1].true_event_id,
            )
        )

        overlap, true_event = candidates[0]
        true_key = (
            true_event.subject_id,
            true_event.edf_id,
            int(true_event.true_event_id),
        )

        if true_key in detected_true:
            classification = "duplicate_ictal"
            latency = np.nan
        else:
            classification = "detected"
            detected_true.add(true_key)
            latency = (
                float(pred.detection_sec)
                - float(true_event.start_sec)
            )

        rows.append({
            "subject_id": pred.subject_id,
            "edf_id": pred.edf_id,
            "pred_event_id": pred.event_id,
            "true_event_id": int(
                true_event.true_event_id
            ),
            "classification": classification,
            "detection_sec": float(
                pred.detection_sec
            ),
            "true_onset_sec": float(
                true_event.start_sec
            ),
            "true_end_sec": float(
                true_event.end_sec
            ),
            "latency_sec": latency,
            "overlap_sec": float(overlap),
        })

    return pd.DataFrame(rows)


def _evaluated_non_seizure_hours(df, true_events):
    """
    Calculate non-seizure time inside the intervals covered by predictions.
    """
    hours = 0.0

    for (subject, edf_id), group in df.groupby(
        ["subject_id", "edf_id"]
    ):
        eval_start = float(group["start_sec"].min())
        eval_end = float(group["end_sec"].max())

        true_for_recording = true_events[
            (true_events["subject_id"] == subject)
            & (true_events["edf_id"] == edf_id)
        ]

        intervals = []

        for start, end in zip(
            true_for_recording["start_sec"],
            true_for_recording["end_sec"],
        ):
            clipped_start = max(
                float(start),
                eval_start,
            )
            clipped_end = min(
                float(end),
                eval_end,
            )

            if clipped_start < clipped_end:
                intervals.append(
                    (clipped_start, clipped_end)
                )

        seizure_seconds = sum(
            end - start
            for start, end in _merge_intervals(intervals)
        )

        evaluated_seconds = max(
            eval_end - eval_start,
            0.0,
        )

        hours += max(
            evaluated_seconds - seizure_seconds,
            0.0,
        ) / 3600.0

    return hours


def calculate_metrics(
    df,
    predicted_events,
    true_events,
    matches,
):
    total_true = len(true_events)

    detected_true = (
        int(
            (
                matches["classification"]
                == "detected"
            ).sum()
        )
        if not matches.empty
        else 0
    )

    duplicate_ictal = (
        int(
            (
                matches["classification"]
                == "duplicate_ictal"
            ).sum()
        )
        if not matches.empty
        else 0
    )

    false_alarm_events = (
        int(
            (
                matches["classification"]
                == "false_alarm"
            ).sum()
        )
        if not matches.empty
        else len(predicted_events)
    )

    latencies = (
        matches.loc[
            matches["classification"] == "detected",
            "latency_sec",
        ]
        .dropna()
        .to_numpy()
        if not matches.empty
        else np.array([], dtype=float)
    )

    non_seizure_hours = (
        _evaluated_non_seizure_hours(
            df,
            true_events,
        )
    )

    return {
        "total_true_events": total_true,
        "detected_true_events": detected_true,
        "event_sensitivity": (
            detected_true / total_true
            if total_true
            else 0.0
        ),
        "predicted_events": len(
            predicted_events
        ),
        "duplicate_ictal_events": duplicate_ictal,
        "false_alarm_events": false_alarm_events,
        "evaluated_non_seizure_hours": (
            non_seizure_hours
        ),
        "false_alarms_per_hour": (
            false_alarm_events
            / non_seizure_hours
            if non_seizure_hours > 0
            else float("nan")
        ),
        "mean_detection_latency_sec": (
            float(latencies.mean())
            if len(latencies)
            else float("nan")
        ),
        "median_detection_latency_sec": (
            float(np.median(latencies))
            if len(latencies)
            else float("nan")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        help="Timestamped prediction CSV",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--max-gap-sec",
        type=float,
        default=4.0,
        help=(
            "Maximum start-to-start gap between positive windows "
            "that stays in one predicted event. Default: 4.0s."
        ),
    )
    args = parser.parse_args()

    df = load_predictions(
        Path(args.predictions)
    )

    predicted_events = build_predicted_events(
        df,
        max_gap_sec=args.max_gap_sec,
    )

    true_events = build_true_events(
        df,
        args.summary_dir,
    )

    matches = match_events(
        predicted_events,
        true_events,
    )

    metrics = calculate_metrics(
        df,
        predicted_events,
        true_events,
        matches,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(args.predictions).stem

    predicted_events.to_csv(
        args.output_dir
        / f"{stem}_predicted_events.csv",
        index=False,
    )

    true_events.to_csv(
        args.output_dir
        / f"{stem}_true_events.csv",
        index=False,
    )

    matches.to_csv(
        args.output_dir
        / f"{stem}_event_matches.csv",
        index=False,
    )

    (
        args.output_dir
        / f"{stem}_event_metrics.json"
    ).write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("EVENT-LEVEL EVALUATION")
    print("=" * 60)
    print(
        f"Max prediction gap     : "
        f"{args.max_gap_sec:.1f} sec"
    )
    print(
        f"True seizure events    : "
        f"{metrics['total_true_events']}"
    )
    print(
        f"Detected true events   : "
        f"{metrics['detected_true_events']}"
    )
    print(
        f"Event sensitivity      : "
        f"{metrics['event_sensitivity']:.4f}"
    )
    print(
        f"Predicted events       : "
        f"{metrics['predicted_events']}"
    )
    print(
        f"Duplicate ictal events : "
        f"{metrics['duplicate_ictal_events']}"
    )
    print(
        f"False-alarm events     : "
        f"{metrics['false_alarm_events']}"
    )
    print(
        f"Non-seizure hours      : "
        f"{metrics['evaluated_non_seizure_hours']:.4f}"
    )
    print(
        f"False alarms/hour      : "
        f"{metrics['false_alarms_per_hour']:.4f}"
    )
    print(
        f"Mean latency (sec)     : "
        f"{metrics['mean_detection_latency_sec']:.4f}"
    )
    print(
        f"Median latency (sec)   : "
        f"{metrics['median_detection_latency_sec']:.4f}"
    )

    print("\nSaved:")
    print(
        "  "
        + str(
            args.output_dir
            / f"{stem}_predicted_events.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / f"{stem}_true_events.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / f"{stem}_event_matches.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / f"{stem}_event_metrics.json"
        )
    )


if __name__ == "__main__":
    main()
