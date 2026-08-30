"""
First look at CHB-MIT EEG data using mne.
"""

import mne
import re
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load one EDF file
# ---------------------------------------------------------------------
DATA_DIR = Path("data/raw/chb01")
EDF_FILE = DATA_DIR / "chb01_03.edf"

raw = mne.io.read_raw_edf(EDF_FILE, preload=True, verbose=False)

print("=" * 60)
print("BASIC INFO")
print("=" * 60)
print(raw.info)
print("\nChannel names:", raw.ch_names)
print("Sampling frequency (Hz):", raw.info["sfreq"])
print("Recording duration (s):", raw.n_times / raw.info["sfreq"])

# ---------------------------------------------------------------------
# 2. Parse seizure onset/offset from the summary file
# ---------------------------------------------------------------------
SUMMARY_FILE = DATA_DIR / "chb01-summary.txt"

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

print("\n" + "=" * 60)
print("SEIZURE ANNOTATIONS")
print("=" * 60)
if seizure_times:
    for i, (start, end) in enumerate(seizure_times, 1):
        print(f"Seizure {i}: {start}s -> {end}s (duration: {end - start}s)")
else:
    print("No seizures found in this file (check filename/parsing).")

# ---------------------------------------------------------------------
# 3. Plot the raw signal, with seizure window marked
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

data, times = raw[:5, :]
sfreq = raw.info["sfreq"]

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(data.shape[0]):
    ax.plot(times, data[i] * 1e6 + i * 200, label=raw.ch_names[i])

if seizure_times:
    start, end = seizure_times[0]
    ax.axvspan(start, end, color="red", alpha=0.2, label="Seizure window")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (uV, offset per channel)")
ax.set_title("chb01_03.edf - first 5 channels")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("results/chb01_03_preview.png", dpi=150)
print("\nPlot saved to results/chb01_03_preview.png")