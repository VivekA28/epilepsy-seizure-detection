"""
Preprocessing pipeline for CHB-MIT scalp EEG.

Processes continuous EDF recordings for a given subject into standardized,
leakage-free windowed datasets:
  - Downsample: 256 Hz -> 128 Hz
  - Filter: 60 Hz notch filter + 0.5–40 Hz bandpass
  - Normalization: Per-channel z-score per recording
  - Windowing: 4-second windows (512 samples) with 50% overlap (2-second step)
  - Memory-safe streaming: Processes one EDF file at a time, writing to disk via memmap
  - Metadata tracking: Generates channel names and metadata tables for rigorous splits
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

SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "chb01"
DATA_DIR = Path(f"data/raw/{SUBJECT}")
SUMMARY_FILE = DATA_DIR / f"{SUBJECT}-summary.txt"
OUT_DIR = Path("data/processed")
TMP_DIR = OUT_DIR / f"_tmp_{SUBJECT}"

NOTCH_FREQ = 60.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
WINDOW_SEC = 4
WINDOW_OVERLAP = 0.5


def parse_seizures_for_file(summary_path: Path, target_filename: str) -> List[Tuple[int, int]]:
    """Parse annotated seizure onset and offset timestamps (in seconds) for one EDF file."""
    text = summary_path.read_text()
    chunks = text.split("File Name:")
    seizures = []
    for chunk in chunks:
        if target_filename not in chunk.split("\n")[0]:
            continue
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
    Segment multi-channel recording into fixed windows and assign binary labels.
    A window is labeled 1 if it overlaps any annotated seizure interval.
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(DATA_DIR.glob("*.edf"))
    print(f"Found {len(edf_files)} EDF files for {SUBJECT}")

    reference_n_channels = None
    reference_channel_names = None
    saved_chunks = []
    skipped = []

    for i, edf_file in enumerate(edf_files, 1):
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

            # Resample 256Hz -> 128Hz; seizure dynamics live well below 40Hz
            raw.resample(128, verbose=False)
            raw.notch_filter(freqs=NOTCH_FREQ, n_jobs=1, verbose=False)
            raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, n_jobs=1, verbose=False)

            sfreq = raw.info["sfreq"]
            data = raw.get_data().astype(np.float32)
            channel_names = list(raw.ch_names)

            # Per-channel z-score normalization per recording: removes inter-electrode
            # amplitude biases while preserving temporal dynamics for explainability (SHAP).
            ch_mean = data.mean(axis=1, keepdims=True)
            ch_std = data.std(axis=1, keepdims=True)
            data = (data - ch_mean) / (ch_std + 1e-8)

            if reference_n_channels is None:
                reference_n_channels = data.shape[0]
                reference_channel_names = channel_names
            elif data.shape[0] != reference_n_channels:
                print(f"  [{i}/{len(edf_files)}] {edf_file.name}: SKIPPED (channel count mismatch)")
                skipped.append(edf_file.name)
                del raw, data
                gc.collect()
                continue

            seizure_times = parse_seizures_for_file(SUMMARY_FILE, edf_file.name)
            windows, labels, start_times, end_times, window_indices = make_windows(
                data, sfreq, WINDOW_SEC, WINDOW_OVERLAP, seizure_times
            )

            # Cache per-file chunks to disk immediately to conserve RAM
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

    # Combine per-file chunks into subject dataset
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

    final_windows = np.lib.format.open_memmap(
        final_windows_path, mode="w+", dtype=np.float32, shape=(total_windows, *sample_shape)
    )
    final_labels = np.zeros(total_windows, dtype=np.int8)
    final_file_ids = np.zeros(total_windows, dtype=np.int32)

    metadata_rows = []
    offset = 0

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

    with open(OUT_DIR / f"{SUBJECT}_file_names.txt", "w") as f:
        f.write("\n".join(saved_chunks))

    if reference_channel_names is not None:
        with open(final_channel_names_path, "w") as f:
            f.write("\n".join(reference_channel_names))

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