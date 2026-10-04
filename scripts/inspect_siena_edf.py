from pathlib import Path
import mne


SIENA_DIR = Path("data/siena/raw/PN00")


def main():

    files = sorted(SIENA_DIR.glob("*.edf"))

    print("=" * 100)
    print("SIENA PN00 - ALL EDF FILES INSPECTION")
    print("=" * 100)

    if not files:
        print("ERROR: No EDF files found.")
        return

    print(f"EDF files found: {len(files)}")

    results = []

    for edf_file in files:

        print()
        print("-" * 100)
        print(f"FILE: {edf_file.name}")
        print("-" * 100)

        try:

            raw = mne.io.read_raw_edf(
                edf_file,
                preload=False,
                verbose=False
            )

            sfreq = raw.info["sfreq"]
            n_channels = len(raw.ch_names)
            duration = raw.times[-1]

            channel_types = raw.get_channel_types()

            eeg_channels = [
                ch
                for ch, ch_type in zip(
                    raw.ch_names,
                    channel_types
                )
                if ch_type == "eeg"
            ]

            eog_channels = [
                ch
                for ch, ch_type in zip(
                    raw.ch_names,
                    channel_types
                )
                if ch_type == "eog"
            ]

            ecg_channels = [
                ch
                for ch, ch_type in zip(
                    raw.ch_names,
                    channel_types
                )
                if ch_type == "ecg"
            ]

            misc_channels = [
                ch
                for ch, ch_type in zip(
                    raw.ch_names,
                    channel_types
                )
                if ch_type == "misc"
            ]

            results.append({
                "file": edf_file.name,
                "sampling_rate": sfreq,
                "channels": n_channels,
                "duration_sec": duration,
                "duration_min": duration / 60,
                "eeg_channels": len(eeg_channels),
                "ecg_channels": len(ecg_channels),
                "eog_channels": len(eog_channels),
                "misc_channels": len(misc_channels),
            })

            print(f"Sampling rate : {sfreq} Hz")
            print(f"Total channels: {n_channels}")
            print(f"Duration      : {duration:.2f} sec")
            print(f"Duration      : {duration / 60:.2f} min")

            print()
            print(f"EEG channels  : {len(eeg_channels)}")
            print(f"ECG channels  : {len(ecg_channels)}")
            print(f"EOG channels  : {len(eog_channels)}")
            print(f"Misc channels : {len(misc_channels)}")

            print()
            print("Channel names:")
            print(", ".join(raw.ch_names))

            print()
            print("Annotations:")

            if len(raw.annotations) == 0:
                print("None")
            else:

                for annotation in raw.annotations:

                    print(
                        f"onset={annotation['onset']:.2f}s | "
                        f"duration={annotation['duration']:.2f}s | "
                        f"description={annotation['description']}"
                    )

        except Exception as e:

            print()
            print("ERROR:")
            print(type(e).__name__)
            print(str(e))

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("COMPARISON OF ALL PN00 FILES")
    print("=" * 100)

    if results:

        print(
            f"{'File':<15}"
            f"{'Hz':<10}"
            f"{'Channels':<12}"
            f"{'EEG':<8}"
            f"{'ECG':<8}"
            f"{'Duration(min)':<18}"
        )

        print("-" * 75)

        for r in results:

            print(
                f"{r['file']:<15}"
                f"{r['sampling_rate']:<10.0f}"
                f"{r['channels']:<12}"
                f"{r['eeg_channels']:<8}"
                f"{r['ecg_channels']:<8}"
                f"{r['duration_min']:<18.2f}"
            )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()