#!/usr/bin/env python3
"""
Semantic channel-order validator across all 15 CHB-MIT subjects.

Verifies the EXACT channel name at every index (0..22), not merely:
    len(channel_names) == 23
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED_CANONICAL = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8"
]


def normalize_ch(ch: str) -> str:
    """Normalize MNE duplicate disambiguators like T8-P8-0 / T8-P8-1 to T8-P8."""
    ch = ch.strip()
    if ch in {"T8-P8-0", "T8-P8-1"}:
        return "T8-P8"
    return ch


def validate_all_subjects():
    print("=" * 80)
    print("SEMANTIC CHANNEL-ORDER VALIDATION (INDEX-BY-INDEX)")
    print("=" * 80)

    all_passed = True
    results = {}

    for i in range(1, 16):
        subject = f"chb{i:02d}"
        chan_path = DATA_DIR / f"{subject}_channel_names.txt"

        if not chan_path.exists():
            print(f"❌ {subject}: File missing ({chan_path})")
            all_passed = False
            results[subject] = "FILE_MISSING"
            continue

        with open(chan_path, "r", encoding="utf-8") as f:
            raw_channels = [line.strip() for line in f if line.strip()]

        if len(raw_channels) != 23:
            print(f"❌ {subject}: Found {len(raw_channels)} channels (expected 23)")
            all_passed = False
            results[subject] = f"BAD_COUNT_{len(raw_channels)}"
            continue

        normalized_channels = [normalize_ch(c) for c in raw_channels]

        mismatches = []
        for idx in range(23):
            actual = normalized_channels[idx]
            expected = EXPECTED_CANONICAL[idx]
            if actual != expected:
                mismatches.append((idx, expected, actual, raw_channels[idx]))

        if mismatches:
            print(f"❌ {subject}: MISMATCH at {len(mismatches)} indices:")
            for idx, exp, norm, raw in mismatches:
                print(f"    Index {idx:2d}: Expected '{exp}', got '{norm}' (raw: '{raw}')")
            all_passed = False
            results[subject] = f"MISMATCH_{len(mismatches)}"
        else:
            print(f"✅ {subject}: All 23 channels match canonical montage index-for-index.")
            results[subject] = "PASS"

    print("=" * 80)
    if all_passed:
        print("SEMANTIC CHANNEL-ORDER VALIDATION: ALL 15 SUBJECTS PASSED ✅")
    else:
        print("SEMANTIC CHANNEL-ORDER VALIDATION: FAILED ❌")
    print("=" * 80)

    return all_passed, results


if __name__ == "__main__":
    passed, _ = validate_all_subjects()
    sys.exit(0 if passed else 1)
