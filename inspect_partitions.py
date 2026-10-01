"""
Inspect Subject Partitions, EDF Recording Distributions, and Dataset Metadata.

Aggregates window-level and recording-level statistics across subjects chb01–chb05.

Clinical & Machine Learning Rationale:
  - Data Leakage Prevention: EEG recordings possess high temporal auto-correlation.
    With 50% window overlap (2s step), splitting randomly across windows would place
    heavily correlated adjacent windows in both train and test sets, artificially
    inflating performance. Recording-level and subject-level splits are mandatory.
  - Extreme Class Imbalance: Seizure events constitute < 0.5% of total monitoring time.
    Inspecting the distribution of seizure-containing vs clean EDF files is essential
    for balanced fold stratification and realistic cross-validation splits.
"""

from pathlib import Path
import pandas as pd

# Directory housing standardized metadata CSV files
DATA_DIR = Path("data/processed")
SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05"]


def inspect_subject(subject: str) -> pd.DataFrame:
    """
    Inspect window counts, seizure distributions, and EDF files for one subject.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g., 'chb01').

    Returns
    -------
    pd.DataFrame
        Subject metadata table.

    Raises
    ------
    FileNotFoundError
        If metadata CSV file does not exist on disk.
    """
    path = DATA_DIR / f"{subject}_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata: {path}")

    df = pd.read_csv(path)
    seizures = (df["label"] == 1).sum()
    non_seizures = (df["label"] == 0).sum()

    print(f"\n{subject.upper()}:")
    print(f"  Windows: {len(df):,} | EDFs: {df['edf_id'].nunique()} | Seizure: {seizures} | Non-Seizure: {non_seizures}")

    # Group by individual EDF file to identify which recordings contain ictal activity
    edf_summary = (
        df.groupby("edf_id")
        .agg(windows=("window_index", "count"), seizure_windows=("label", "sum"))
        .reset_index()
    )
    seizure_edfs = edf_summary[edf_summary["seizure_windows"] > 0]
    print(f"  EDFs with seizures ({len(seizure_edfs)}/{len(edf_summary)}): {list(seizure_edfs['edf_id'].values)}")
    return df


def main():
    """Inspect and validate structural metadata consistency across all subjects."""
    print("=" * 60)
    print("CHB-MIT PARTITION & EDF METADATA INSPECTION")
    print("=" * 60)

    # Load and concatenate metadata for all target subjects
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

    # Global dataset consistency assertions:
    # 1. Monotonic temporal interval: end time strictly greater than start time
    assert (combined["end_sec"] > combined["start_sec"]).all(), "Invalid window timestamp range"
    # 2. Invariant sampling frequency across all records: 128 Hz
    assert (combined["sampling_rate"] == 128.0).all(), "Inconsistent sampling rate"
    # 3. Standardized montage channel count: 23 bipolar channels
    assert (combined["n_channels"] == 23).all(), "Inconsistent channel count"
    # 4. Standardized window sample count: 512 samples (4 seconds at 128 Hz)
    assert (combined["n_samples"] == 512).all(), "Inconsistent sample count"

    print("\n[PASS] All structural and metadata consistency checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()