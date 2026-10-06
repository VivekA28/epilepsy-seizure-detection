#!/usr/bin/env python3
"""
Standalone read-only script to inspect raw channel order in CHB-MIT EDF files.

Inspects chb01, chb12, chb13, chb14, chb15 in data/raw/<subject>/
Loads raw EDF with MNE without preprocessing, normalization, or canonicalization.
Prints channel order per subject and diff-style summary against chb01.
"""

from pathlib import Path
import warnings

import mne

TARGET_SUBJECTS = ["chb01", "chb12", "chb13", "chb14", "chb15"]
RAW_DIR = Path("data/raw")


def find_first_edf(subject: str) -> Path | None:
    subj_dir = RAW_DIR / subject
    if not subj_dir.exists() or not subj_dir.is_dir():
        return None
    edf_files = sorted(subj_dir.glob("*.edf"))
    return edf_files[0] if edf_files else None


def main():
    channels_by_subject: dict[str, list[str]] = {}
    files_by_subject: dict[str, Path] = {}

    for subject in TARGET_SUBJECTS:
        print("=" * 80)
        print(f"Subject: {subject}")
        print("=" * 80)

        edf_path = find_first_edf(subject)
        if edf_path is None:
            print(f"[SKIPPED] No .edf files found in '{RAW_DIR / subject}' (data not downloaded locally).\n")
            continue

        files_by_subject[subject] = edf_path
        print(f"File: {edf_path}")

        # Load raw EDF with MNE without preprocessing or preload
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        ch_names = list(raw.ch_names)
        channels_by_subject[subject] = ch_names
        print(f"Total Channels: {len(ch_names)}\nChannels (0-based index):")
        for idx, ch in enumerate(ch_names):
            print(f"{idx}: {ch}")
        print()

    # Diff-style summary against chb01
    print("=" * 80)
    print("DIFF-STYLE SUMMARY (Comparison against chb01)")
    print("=" * 80)

    if "chb01" not in channels_by_subject:
        print("ERROR: chb01 was not loaded; cannot perform comparison against chb01.")
        return

    chb01_ch = channels_by_subject["chb01"]
    print(f"Baseline: chb01 ({files_by_subject['chb01']}, {len(chb01_ch)} channels)\n")

    for subject in ["chb12", "chb13", "chb14", "chb15"]:
        if subject not in channels_by_subject:
            print(f"{subject}: SKIPPED (data not downloaded locally in {RAW_DIR / subject}/)")
            continue

        subj_ch = channels_by_subject[subject]
        max_len = max(len(chb01_ch), len(subj_ch))
        diffs = []
        for idx in range(max_len):
            name_01 = chb01_ch[idx] if idx < len(chb01_ch) else "<MISSING>"
            name_sub = subj_ch[idx] if idx < len(subj_ch) else "<MISSING>"
            if name_01 != name_sub:
                diffs.append((idx, name_01, name_sub))

        if not diffs:
            print(f"{subject}: Channel order EXACTLY MATCHES chb01 across all {len(subj_ch)} channels.")
        else:
            print(f"{subject}: {len(diffs)} index mismatch(es) compared to chb01:")
            for idx, name_01, name_sub in diffs:
                print(f"  Index {idx:2d}: chb01 has '{name_01}' | {subject} has '{name_sub}'")
        print()


if __name__ == "__main__":
    main()
