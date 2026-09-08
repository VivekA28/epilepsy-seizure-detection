"""
Preprocessing step 2: filtering + windowing across an entire subject
folder (all EDF files for one subject, e.g. all of chb01), combined
into a single dataset.

Processes one file at a time and writes each file's result straight to
disk before moving to the next, rather than keeping every file's data
in memory simultaneously. The final combined dataset is also built on
disk (via a memory-mapped array) instead of in RAM. This keeps memory
usage low regardless of how many files or subjects are processed.

Run from the project root:
    python preprocess_subject.py chb01
"""

import sys
import gc
import shutil
import mne
import re
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "chb01"
DATA_DIR = Path(f"data/raw/{SUBJECT}")
SUMMARY_FILE = DATA_DIR / f"{SUBJECT}-summary.txt"

NOTCH_FREQ = 60.0
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
WINDOW_SEC = 4
WINDOW_OVERLAP = 0.5

TMP_DIR = Path(f"data/processed/_tmp_{SUBJECT}")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Seizure time parser (unchanged)
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

def make_windows(data, sfreq, window_sec, overlap, seizure_times):
    win_len = int(window_sec * sfreq)
    step = int(win_len * (1 - overlap))
    n_samples = data.shape[1]

    windows = []
    labels = []

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

    # Store as float32 (not the numpy default float64) since that's more
    # than enough precision for EEG amplitudes and keeps file sizes/memory
    # usage down. Labels are just 0/1 so int8 is plenty.
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int8)

# ---------------------------------------------------------------------
# Process every EDF file, saving each result to disk immediately
# ---------------------------------------------------------------------
edf_files = sorted(DATA_DIR.glob("*.edf"))
print(f"Found {len(edf_files)} EDF files for {SUBJECT}")

reference_n_channels = None
saved_chunks = []   # tracks which per-file chunks were saved to disk
skipped = []

for i, edf_file in enumerate(edf_files, 1):
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

        # Downsample from 256Hz to 128Hz before filtering. Seizure-relevant
        # activity lives well under 40Hz (our bandpass upper limit anyway),
        # so this loses essentially no information we care about, while
        # halving the amount of data every later step has to hold in memory.
        raw.resample(128, verbose=False)

        # n_jobs=1 keeps mne's filtering single-threaded, which uses
        # noticeably less peak memory than its default parallel mode -
        # slower, but safer on a machine with limited RAM.
        raw.notch_filter(freqs=NOTCH_FREQ, n_jobs=1, verbose=False)
        raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, n_jobs=1, verbose=False)

        sfreq = raw.info["sfreq"]
        data = raw.get_data().astype(np.float32)

        # ---------------------------------------------------------
        # Per-channel z-score normalization (per recording).
        #
        # Different EEG channels have naturally different raw signal
        # amplitudes (e.g. temporal channels near muscle tend to run
        # "louder" than central ones) - unrelated to seizure activity.
        # Without this, downstream explainability (SHAP) tends to just
        # rank high-amplitude channels as "important" regardless of
        # whether they're actually seizure-relevant, since it's picking
        # up on scale rather than signal content.
        #
        # Each channel is normalized independently, using only this
        # recording's own mean/std (not global dataset statistics) -
        # this keeps within-channel temporal changes (e.g. a seizure
        # causing a spike relative to that channel's own baseline)
        # intact, while removing cross-channel scale differences.
        channel_mean = data.mean(axis=1, keepdims=True)
        channel_std = data.std(axis=1, keepdims=True)
        data = (data - channel_mean) / (channel_std + 1e-8)  # epsilon avoids div-by-zero on flat/dead channels
        # ---------------------------------------------------------

        if reference_n_channels is None:
            reference_n_channels = data.shape[0]
        elif data.shape[0] != reference_n_channels:
            print(f"  [{i}/{len(edf_files)}] {edf_file.name}: SKIPPED "
                  f"(channel count {data.shape[0]} != expected {reference_n_channels})")
            skipped.append(edf_file.name)
            del raw, data
            gc.collect()
            continue

        seizure_times = parse_seizures_for_file(SUMMARY_FILE, edf_file.name)
        windows, labels = make_windows(data, sfreq, WINDOW_SEC, WINDOW_OVERLAP, seizure_times)

        # Save this file's windows/labels to a temp folder right away,
        # along with which source file they came from - needed later so
        # train/test splitting can be done by file rather than by
        # individual window (avoids leaking near-duplicate overlapping
        # windows across the train/test boundary)
        w_path = TMP_DIR / f"{edf_file.stem}_windows.npy"
        l_path = TMP_DIR / f"{edf_file.stem}_labels.npy"
        np.save(w_path, windows)
        np.save(l_path, labels)
        saved_chunks.append(edf_file.stem)

        n_seizure = labels.sum()
        print(f"  [{i}/{len(edf_files)}] {edf_file.name}: "
              f"{len(windows)} windows, {n_seizure} seizure "
              f"({'seizure file' if seizure_times else 'clean file'})")

        # Free this file's data now that it's saved, before loading the next
        del raw, data, windows, labels
        gc.collect()

    except Exception as e:
        print(f"  [{i}/{len(edf_files)}] {edf_file.name}: SKIPPED (error: {e})")
        skipped.append(edf_file.name)

# ---------------------------------------------------------------------
# Combine all per-file chunks into one final dataset for this subject
# ---------------------------------------------------------------------
print(f"\nCombining {len(saved_chunks)} saved chunks...")

total_windows = 0
sample_shape = None
for stem in saved_chunks:
    w = np.load(TMP_DIR / f"{stem}_windows.npy", mmap_mode="r")
    total_windows += w.shape[0]
    if sample_shape is None:
        sample_shape = w.shape[1:]
    del w

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
final_windows_path = OUT_DIR / f"{SUBJECT}_windows.npy"
final_labels_path = OUT_DIR / f"{SUBJECT}_labels.npy"

final_windows = np.lib.format.open_memmap(
    final_windows_path, mode="w+", dtype=np.float32,
    shape=(total_windows, *sample_shape)
)
final_labels = np.zeros(total_windows, dtype=np.int8)
final_file_ids = np.zeros(total_windows, dtype=np.int32)

offset = 0
for file_idx, stem in enumerate(saved_chunks):
    w = np.load(TMP_DIR / f"{stem}_windows.npy")
    l = np.load(TMP_DIR / f"{stem}_labels.npy")
    n = w.shape[0]
    final_windows[offset:offset + n] = w
    final_labels[offset:offset + n] = l
    final_file_ids[offset:offset + n] = file_idx
    offset += n
    del w, l
    gc.collect()

final_windows.flush()
np.save(final_labels_path, final_labels)
np.save(OUT_DIR / f"{SUBJECT}_file_ids.npy", final_file_ids)
with open(OUT_DIR / f"{SUBJECT}_file_names.txt", "w") as f:
    f.write("\n".join(saved_chunks))

# ---------------------------------------------------------------------
# Clean up temp per-file chunks
# ---------------------------------------------------------------------
shutil.rmtree(TMP_DIR)

print("\n" + "=" * 60)
print(f"SUMMARY - {SUBJECT}")
print("=" * 60)
print(f"Files processed: {len(saved_chunks)} / {len(edf_files)}")
if skipped:
    print(f"Files skipped: {skipped}")
print(f"Total windows: {total_windows}")
print(f"Seizure windows: {final_labels.sum()} / {total_windows} "
      f"({100 * final_labels.mean():.2f}%)")
print(f"\nSaved to {final_windows_path}")
print(f"Saved to {final_labels_path}")