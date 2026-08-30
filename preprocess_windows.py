"""
Preprocessing step 2: filtering + windowing.

Takes the raw EDF signal, applies standard EEG cleanup (notch filter to
remove power-line noise, bandpass filter to keep the frequency range that
matters for seizure activity), then chops the continuous signal into
fixed-length windows, each labeled seizure/non-seizure based on the
annotation times parsed from the summary file.

Run from the project root:
    python preprocess_windows.py
"""

import mne
import re
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_DIR = Path("data/raw/chb01")
EDF_FILE = DATA_DIR / "chb01_03.edf"
SUMMARY_FILE = DATA_DIR / "chb01-summary.txt"

NOTCH_FREQ = 60.0        # power line noise (60Hz US-recorded data; CHB-MIT is 60Hz)
BANDPASS_LOW = 0.5       # Hz - cuts slow drift
BANDPASS_HIGH = 40.0     # Hz - cuts high-frequency muscle/noise artifacts
WINDOW_SEC = 4           # seconds per window
WINDOW_OVERLAP = 0.5     # 50% overlap between consecutive windows

# ---------------------------------------------------------------------
# 1. Load + filter
# ---------------------------------------------------------------------
raw = mne.io.read_raw_edf(EDF_FILE, preload=True, verbose=False)

print("Applying notch filter (removes 60Hz power line noise)...")
raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

print(f"Applying bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, verbose=False)

sfreq = raw.info["sfreq"]
data = raw.get_data()  # shape: (n_channels, n_samples)
print(f"Filtered data shape: {data.shape} (channels, samples)")

# ---------------------------------------------------------------------
# 2. Parse seizure times (same as eda_walkthrough.py)
# ---------------------------------------------------------------------
def parse_seizures_for_file(summary_path: Path, target_filename: str):
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

seizure_times = parse_seizures_for_file(SUMMARY_FILE, EDF_FILE.name)
print(f"Seizure windows in this file: {seizure_times}")

# ---------------------------------------------------------------------
# 3. Window the signal + assign labels
# ---------------------------------------------------------------------
def make_windows(data, sfreq, window_sec, overlap, seizure_times):
    """
    Slices `data` (channels x samples) into fixed-length windows.
    A window is labeled 1 (seizure) if it overlaps at all with any
    annotated seizure period, else 0.
    """
    win_len = int(window_sec * sfreq)
    step = int(win_len * (1 - overlap))
    n_samples = data.shape[1]

    windows = []
    labels = []

    for start_sample in range(0, n_samples - win_len + 1, step):
        end_sample = start_sample + win_len
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        # label = 1 if this window overlaps any seizure period at all
        label = 0
        for s_start, s_end in seizure_times:
            if start_sec < s_end and end_sec > s_start:
                label = 1
                break

        windows.append(data[:, start_sample:end_sample])
        labels.append(label)

    return np.array(windows), np.array(labels)

windows, labels = make_windows(data, sfreq, WINDOW_SEC, WINDOW_OVERLAP, seizure_times)

print(f"\nTotal windows created: {len(windows)}")
print(f"Window shape (each): {windows[0].shape} (channels, samples_per_window)")
print(f"Seizure windows: {labels.sum()} / {len(labels)} ({100 * labels.mean():.2f}%)")

# ---------------------------------------------------------------------
# 4. Save processed windows for later use (model training)
# ---------------------------------------------------------------------
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUT_DIR / "chb01_03_windows.npy", windows)
np.save(OUT_DIR / "chb01_03_labels.npy", labels)

print(f"\nSaved windows to {OUT_DIR / 'chb01_03_windows.npy'}")
print(f"Saved labels to {OUT_DIR / 'chb01_03_labels.npy'}")