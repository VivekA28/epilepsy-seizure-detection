"""
Preprocessing Pipeline for CHB-MIT Scalp EEG.

Processes continuous EDF recordings for a given subject into standardized,
leakage-free windowed datasets:
  - Downsampling: 256 Hz -> 128 Hz. Preserves physiological frequencies up to
    Nyquist (64 Hz) while cutting memory and compute requirements by half.
  - Powerline Notch Filter: 60 Hz notch filter to eliminate AC mains interference.
  - Bandpass Filter: 0.5–40.0 Hz bandpass to eliminate low-frequency baseline
    drift (<0.5 Hz, e.g., sweating, movement) and high-frequency noise (>40.0 Hz, e.g., EMG).
  - Per-Channel Normalization: Per-recording per-channel z-score standardization.
    Removes inter-electrode amplitude bias without leaking test statistics across recordings
    or confounding explainability (SHAP/LIME attribution).
  - Windowing: 4-second windows (512 samples) with 50% overlap (2-second step) to capture
    sustained electrographic rhythmic seizure discharges with temporal fidelity.
  - Seizure Labeling: Binary label 1 if any portion of the window intersects an
    annotated seizure onset/offset interval from the clinical summary report.
  - Memory-Safe Streaming: Two-phase pipeline that processes one EDF file at a time,
    writes temporary chunks to disk, and consolidates them into a contiguous memory-mapped
    NumPy array to prevent out-of-memory (OOM) failures on large subjects.
  - Metadata Tracking: Retains recording ID, window indices, timestamps, and channel labels
    to ensure patient-specific and recording-specific splits without data leakage.
"""

import sys
import gc
import shutil
import re
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd

# Target subject from CLI arguments (default: 'chb01')
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "chb01"
DATA_DIR = Path(f"data/raw/{SUBJECT}")
SUMMARY_FILE = DATA_DIR / f"{SUBJECT}-summary.txt"
OUT_DIR = Path("data/processed")
TMP_DIR = OUT_DIR / f"_tmp_{SUBJECT}"

# Signal processing hyper-parameters
NOTCH_FREQ = 60.0       # AC powerline frequency in North America (Hz)
BANDPASS_LOW = 0.5      # High-pass cutoff to remove DC drift and sweating artifacts (Hz)
BANDPASS_HIGH = 40.0    # Low-pass cutoff to remove EMG and high-frequency noise (Hz)
WINDOW_SEC = 4          # Window duration in seconds (4s * 128 Hz = 512 samples)
WINDOW_OVERLAP = 0.5    # 50% overlap between successive windows (2.0s step)


def parse_seizures_for_file(summary_path: Path, target_filename: str) -> List[Tuple[int, int]]:
    """
    Parse annotated seizure onset and offset timestamps (in seconds) for one EDF file.

    Parameters
    ----------
    summary_path : Path
        Path to the CHB-MIT clinical summary text file (e.g., chb01-summary.txt).
    target_filename : str
        Base EDF filename being queried (e.g., 'chb01_03.edf').

    Returns
    -------
    list of tuple of (int, int)
        List of (start_sec, end_sec) intervals corresponding to documented seizures.
    """
    text = summary_path.read_text()
    chunks = text.split("File Name:")
    seizures = []
    for chunk in chunks:
        if target_filename not in chunk.split("\n")[0]:
            continue
        # CHB-MIT summary files use either singular or numbered seizure labels
        starts = re.findall(r"Seizure Start Time:\s*(\d+)\s*seconds", chunk)
        ends = re.findall(r"Seizure End Time:\s*(\d+)\s*seconds", chunk)
        if not starts:
            starts = re.findall(r"Seizure \d+ Start Time:\s*(\d+)\s*seconds", chunk)
            ends = re.findall(r"Seizure \d+ End Time:\s*(\d+)\s*seconds", chunk)
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))
    return seizures


def make_windows(
    data: np.ndarray,
    sfreq: float,
    window_sec: float,
    overlap: float,
    seizure_times: List[Tuple[int, int]],
):
    """
    Segment continuous multi-channel recording into fixed-length overlapping windows.

    A window is assigned a binary label of 1 if any part of its time span overlaps
    with an annotated clinical seizure onset/offset interval; otherwise 0.

    Parameters
    ----------
    data : np.ndarray
        Multi-channel EEG array of shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz (typically 128.0).
    window_sec : float
        Window duration in seconds (typically 4.0).
    overlap : float
        Fractional overlap between successive windows (typically 0.5 = 50%).
    seizure_times : list of tuple of (int, int)
        Ground-truth (start_sec, end_sec) intervals for clinical seizures.

    Returns
    -------
    windows : np.ndarray
        Window tensor of shape (n_windows, n_channels, win_len) in float32.
    labels : np.ndarray
        Binary seizure label vector of shape (n_windows,) in int8.
    start_times : np.ndarray
        Window onset timestamp in seconds from recording start.
    end_times : np.ndarray
        Window offset timestamp in seconds from recording start.
    window_indices : np.ndarray
        Sequential window indices (0 to n_windows - 1) within this EDF.
    """
    win_len = int(window_sec * sfreq)
    step = int(win_len * (1 - overlap))
    n_samples = data.shape[1]

    windows, labels = [], []
    start_times, end_times, window_indices = [], [], []

    idx = 0
    for start_sample in range(0, n_samples - win_len + 1, step):
        end_sample = start_sample + win_len
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        # Check for any temporal intersection between window and known seizure intervals
        label = 0
        for s_start, s_end in seizure_times:
            if start_sec < s_end and end_sec > s_start:
                label = 1
                break

        windows.append(data[:, start_sample:end_sample])
        labels.append(label)
        start_times.append(start_sec)
        end_times.append(end_sec)
        window_indices.append(idx)
        idx += 1

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int8),
        np.array(start_times, dtype=np.float32),
        np.array(end_times, dtype=np.float32),
        np.array(window_indices, dtype=np.int32),
    )


def main():
    """Main preprocessing workflow for the selected subject."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(DATA_DIR.glob("*.edf"))
    print(f"Found {len(edf_files)} EDF files for {SUBJECT}")

    reference_n_channels = None
    reference_channel_names = None
    saved_chunks = []
    skipped = []

    # Phase 1: Process each continuous EDF recording individually
    for i, edf_file in enumerate(edf_files, 1):
        try:
            # Read EDF header and signal traces using MNE
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

            # 1. Resample: 256 Hz -> 128 Hz (seizure rhythms lie well below 40 Hz)
            raw.resample(128, verbose=False)
            # 2. Notch filter: Remove 60 Hz AC electrical mains interference
            raw.notch_filter(freqs=NOTCH_FREQ, n_jobs=1, verbose=False)
            # 3. Bandpass filter: Retain 0.5–40 Hz physiological EEG rhythms
            raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, n_jobs=1, verbose=False)

            sfreq = raw.info["sfreq"]
            data = raw.get_data().astype(np.float32)
            channel_names = list(raw.ch_names)

            # 4. Per-channel z-score normalization per recording:
            # Standardizes electrode scales to zero mean and unit variance.
            # Eliminates cross-session impedance differences without leaking data.
            ch_mean = data.mean(axis=1, keepdims=True)
            ch_std = data.std(axis=1, keepdims=True)
            data = (data - ch_mean) / (ch_std + 1e-8)

            # Enforce consistent electrode channel montage across all EDFs of the subject
            if reference_n_channels is None:
                reference_n_channels = data.shape[0]
                reference_channel_names = channel_names
            elif data.shape[0] != reference_n_channels:
                print(f"  [{i}/{len(edf_files)}] {edf_file.name}: SKIPPED (channel count mismatch)")
                skipped.append(edf_file.name)
                del raw, data
                gc.collect()
                continue

            # 5. Extract annotated seizure timestamps and slice into 4s overlapping windows
            seizure_times = parse_seizures_for_file(SUMMARY_FILE, edf_file.name)
            windows, labels, start_times, end_times, window_indices = make_windows(
                data, sfreq, WINDOW_SEC, WINDOW_OVERLAP, seizure_times
            )

            # 6. Cache per-file chunks immediately to disk to minimize memory consumption
            stem = edf_file.stem
            np.save(TMP_DIR / f"{stem}_windows.npy", windows)
            np.save(TMP_DIR / f"{stem}_labels.npy", labels)
            np.save(TMP_DIR / f"{stem}_starts.npy", start_times)
            np.save(TMP_DIR / f"{stem}_ends.npy", end_times)
            np.save(TMP_DIR / f"{stem}_indices.npy", window_indices)
            saved_chunks.append(stem)

            print(f"  [{i}/{len(edf_files)}] {edf_file.name}: {len(windows):,} windows, {labels.sum()} seizure")

            del raw, data, windows, labels
            gc.collect()

        except Exception as e:
            print(f"  [{i}/{len(edf_files)}] {edf_file.name}: SKIPPED (error: {e})")
            skipped.append(edf_file.name)

    # Phase 2: Consolidate cached chunks into a contiguous disk-backed memory-mapped array
    print(f"\nCombining {len(saved_chunks)} saved chunks into memory-mapped array...")
    total_windows = 0
    sample_shape = None
    for stem in saved_chunks:
        w = np.load(TMP_DIR / f"{stem}_windows.npy", mmap_mode="r")
        total_windows += len(w)
        if sample_shape is None:
            sample_shape = w.shape[1:]
        del w

    final_windows_path = OUT_DIR / f"{SUBJECT}_windows.npy"
    final_labels_path = OUT_DIR / f"{SUBJECT}_labels.npy"
    final_file_ids_path = OUT_DIR / f"{SUBJECT}_file_ids.npy"
    final_metadata_path = OUT_DIR / f"{SUBJECT}_metadata.csv"
    final_channel_names_path = OUT_DIR / f"{SUBJECT}_channel_names.txt"

    # Pre-allocate contiguous disk buffer for multi-channel windows
    final_windows = np.lib.format.open_memmap(
        final_windows_path, mode="w+", dtype=np.float32, shape=(total_windows, *sample_shape)
    )
    final_labels = np.zeros(total_windows, dtype=np.int8)
    final_file_ids = np.zeros(total_windows, dtype=np.int32)

    metadata_rows = []
    offset = 0

    # Stream cached chunks into final array and build complete metadata registry
    for file_idx, stem in enumerate(saved_chunks):
        w = np.load(TMP_DIR / f"{stem}_windows.npy")
        l = np.load(TMP_DIR / f"{stem}_labels.npy")
        starts = np.load(TMP_DIR / f"{stem}_starts.npy")
        ends = np.load(TMP_DIR / f"{stem}_ends.npy")
        indices = np.load(TMP_DIR / f"{stem}_indices.npy")
        n = len(w)

        final_windows[offset:offset + n] = w
        final_labels[offset:offset + n] = l
        final_file_ids[offset:offset + n] = file_idx

        for j in range(n):
            metadata_rows.append({
                "dataset_id": "chbmit",
                "subject_id": SUBJECT,
                "edf_id": stem,
                "window_index": int(indices[j]),
                "start_sec": float(starts[j]),
                "end_sec": float(ends[j]),
                "label": int(l[j]),
                "sampling_rate": 128.0,
                "n_channels": int(w.shape[1]),
                "n_samples": int(w.shape[2]),
            })

        offset += n
        del w, l, starts, ends, indices
        gc.collect()

    final_windows.flush()
    np.save(final_labels_path, final_labels)
    np.save(final_file_ids_path, final_file_ids)

    # Persist file list and standardized electrode channel labels
    with open(OUT_DIR / f"{SUBJECT}_file_names.txt", "w") as f:
        f.write("\n".join(saved_chunks))

    if reference_channel_names is not None:
        with open(final_channel_names_path, "w") as f:
            f.write("\n".join(reference_channel_names))

    # Persist structured tabular metadata for reproducible train/val/test partitioning
    pd.DataFrame(metadata_rows).to_csv(final_metadata_path, index=False)
    shutil.rmtree(TMP_DIR)

    print("\n" + "=" * 60)
    print(f"PREPROCESSING COMPLETE - {SUBJECT}")
    print("=" * 60)
    print(f"Files processed: {len(saved_chunks)} / {len(edf_files)}")
    if skipped:
        print(f"Files skipped: {skipped}")
    print(f"Total windows: {total_windows:,} | Seizure windows: {final_labels.sum():,} ({100*final_labels.mean():.2f}%)")
    print(f"Saved: {final_windows_path}")
    print(f"Saved: {final_labels_path}")
    print(f"Saved: {final_metadata_path}")


if __name__ == "__main__":
    main()