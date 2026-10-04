# ============================================================
# CHB-MIT vs SIENA Channel Compatibility Analysis
# ============================================================

# CHB target bipolar derivations used by the current pipeline.
#
# IMPORTANT:
# CHB contains T8-P8 twice in the target list.
# We preserve that because this is the current CHB representation.

CHB_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
    "T8-P8",
]


# Actual SIENA PN00 EEG channels observed from the EDF.
SIENA_29 = [
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FC1",
    "FC5",
    "CP1",
    "CP5",
    "F9",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FC2",
    "FC6",
    "CP2",
    "CP6",
    "F10",
]


# PN10 has only these 19 EEG channels.
SIENA_19 = [
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FZ",
    "CZ",
    "PZ",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FP2",
]


# Old SIENA naming -> modern CHB-style naming.
NAME_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def canonicalize(channel):
    """Convert SIENA electrode name to common naming."""

    channel = channel.upper().strip()

    return NAME_MAP.get(channel, channel)


def canonical_set(channels):
    """Return canonicalized channel set."""

    return {
        canonicalize(channel)
        for channel in channels
    }


def analyze(chb_channels, siena_channels, dataset_name):

    siena_set = canonical_set(siena_channels)

    reconstructable = []
    impossible = []

    for derivation in chb_channels:

        left, right = derivation.split("-")

        left = canonicalize(left)
        right = canonicalize(right)

        if left in siena_set and right in siena_set:

            reconstructable.append({
                "derivation": derivation,
                "electrode_1": left,
                "electrode_2": right,
                "status": "YES",
            })

        else:

            missing = []

            if left not in siena_set:
                missing.append(left)

            if right not in siena_set:
                missing.append(right)

            impossible.append({
                "derivation": derivation,
                "electrode_1": left,
                "electrode_2": right,
                "missing": ", ".join(missing),
                "status": "NO",
            })

    print()
    print("=" * 90)
    print(f"DATASET: {dataset_name}")
    print("=" * 90)

    print()
    print(f"SIENA EEG channels available : {len(siena_channels)}")
    print(
        f"CHB target channel entries  : "
        f"{len(chb_channels)}"
    )

    print()
    print("-" * 90)
    print("RECONSTRUCTABLE CHANNELS")
    print("-" * 90)

    for item in reconstructable:

        print(
            f"{item['derivation']:<15} "
            f"YES  "
            f"({item['electrode_1']} + "
            f"{item['electrode_2']})"
        )

    print()
    print("-" * 90)
    print("NON-RECONSTRUCTABLE CHANNELS")
    print("-" * 90)

    for item in impossible:

        print(
            f"{item['derivation']:<15} "
            f"NO   "
            f"missing: {item['missing']}"
        )

    print()
    print("-" * 90)
    print("SUMMARY")
    print("-" * 90)

    print(
        f"Reconstructable : "
        f"{len(reconstructable)}/{len(chb_channels)}"
    )

    print(
        f"Not possible    : "
        f"{len(impossible)}/{len(chb_channels)}"
    )

    percentage = (
        len(reconstructable)
        / len(chb_channels)
        * 100
    )

    print(
        f"Coverage        : "
        f"{percentage:.1f}%"
    )


def main():

    print("=" * 90)
    print("CHB-MIT vs SIENA CHANNEL COMPATIBILITY ANALYSIS")
    print("=" * 90)

    analyze(
        CHB_CHANNELS,
        SIENA_29,
        "SIENA 29-channel configuration"
    )

    analyze(
        CHB_CHANNELS,
        SIENA_19,
        "SIENA 19-channel configuration (PN10)"
    )

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()