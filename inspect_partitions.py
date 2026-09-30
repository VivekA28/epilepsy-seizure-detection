"""
Inspect subject partitions, EDF recording distributions, and dataset metadata.

Aggregates window-level and recording-level statistics across subjects chb01–chb05.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def inspect_subject(subject: str) -> pd.DataFrame:
    path = DATA_DIR / f"{subject}_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    df = pd.read_csv(path)
    seizures = (df["label"] == 1).sum()
    non_seizures = (df["label"] == 0).sum()

    print(f"\n{subject.upper()}:")
    print(f"  Windows: {len(df):,} | EDFs: {df['edf_id'].nunique()} | Seizure: {seizures} | Non-Seizure: {non_seizures}")

    edf_summary = (
        df.groupby("edf_id")
        .agg(windows=("window_index", "count"), seizure_windows=("label", "sum"))
        .reset_index()
    )
    seizure_edfs = edf_summary[edf_summary["seizure_windows"] > 0]
    print(f"  EDFs with seizures ({len(seizure_edfs)}/{len(edf_summary)}): {list(seizure_edfs['edf_id'].values)}")
    return df


def main():
    print("=" * 60)
    print("CHB-MIT PARTITION & EDF METADATA INSPECTION")
    print("=" * 60)

    dfs = [inspect_subject(s) for s in SUBJECTS]
    combined = pd.concat(dfs, ignore_index=True)

    print("\n" + "=" * 60)
    print("DATASET AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Total Windows        : {len(combined):,}")
    print(f"Total Seizure Windows: {(combined['label'] == 1).sum():,}")
    print(f"Total Clean Windows  : {(combined['label'] == 0).sum():,}")
    print(f"Unique Subjects      : {combined['subject_id'].nunique()}")
    print(f"Unique EDF Files     : {combined['edf_id'].nunique()}")

    # Consistency assertions
    assert (combined["end_sec"] > combined["start_sec"]).all(), "Invalid window timestamp range"
    assert (combined["sampling_rate"] == 128.0).all(), "Inconsistent sampling rate"
    assert (combined["n_channels"] == 23).all(), "Inconsistent channel count"
    assert (combined["n_samples"] == 512).all(), "Inconsistent sample count"

    print("\n[PASS] All structural and metadata consistency checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()