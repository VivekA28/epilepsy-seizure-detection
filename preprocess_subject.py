"""



Preprocessing step 2: filtering + windowing across an entire subject



folder (all EDF files for one subject, e.g. all of chb01), combined



into a single dataset.



Processes one file at a time and writes each file's result straight to



disk before moving to the next, rather than keeping every file's data



in memory simultaneously.



The final combined dataset is also built on disk using a memory-mapped



array instead of loading everything into RAM.



Current development subjects:



    chb01



    chb02



    chb03



    chb04



    chb05



Run from the project root:



    python preprocess_subject.py chb01



    python preprocess_subject.py chb02



    ...



The existing preprocessing behaviour is preserved:



    256 Hz -> 128 Hz



    60 Hz notch



    0.5-40 Hz bandpass



    per-channel z-score normalization



    4-second windows



    50% overlap



"""



import sys



import gc



import shutil



import re



import mne



import numpy as np



import pandas as pd



from pathlib import Path



# =====================================================================



# 1. Configuration



# =====================================================================



SUBJECT = (



    sys.argv[1]



    if len(sys.argv) > 1



    else "chb01"



)



DATA_DIR = Path(



    f"data/raw/{SUBJECT}"



)



SUMMARY_FILE = (



    DATA_DIR / f"{SUBJECT}-summary.txt"



)



# Existing preprocessing settings



NOTCH_FREQ = 60.0



BANDPASS_LOW = 0.5



BANDPASS_HIGH = 40.0



WINDOW_SEC = 4



WINDOW_OVERLAP = 0.5



# Temporary directory for per-EDF results



TMP_DIR = Path(



    f"data/processed/_tmp_{SUBJECT}"



)



TMP_DIR.mkdir(



    parents=True,



    exist_ok=True



)



# =====================================================================



# 2. Parse seizure annotations



# =====================================================================



def parse_seizures_for_file(



    summary_path: Path,



    target_filename: str



):



    """



    Read seizure start/end times for one EDF file



    from the CHB-MIT summary file.



    Returns:



        List of tuples:



        [



            (start_seconds, end_seconds),



            ...



        ]



    """



    text = summary_path.read_text()



    chunks = text.split("File Name:")



    seizures = []



    for chunk in chunks:



        first_line = (



            chunk.split("\n")[0]



        )



        if target_filename not in first_line:



            continue



        starts = re.findall(



            r"Seizure Start Time:\s*(\d+)\s*seconds",



            chunk



        )



        ends = re.findall(



            r"Seizure End Time:\s*(\d+)\s*seconds",



            chunk



        )



        # Support alternate CHB-MIT annotation format



        if not starts:



            starts = re.findall(



                r"Seizure \d+ Start Time:\s*(\d+)\s*seconds",



                chunk



            )



            ends = re.findall(



                r"Seizure \d+ End Time:\s*(\d+)\s*seconds",



                chunk



            )



        for s, e in zip(starts, ends):



            seizures.append(



                (



                    int(s),



                    int(e)



                )



            )



    return seizures



# =====================================================================



# 3. Create EEG windows



# =====================================================================



def make_windows(



    data,



    sfreq,



    window_sec,



    overlap,



    seizure_times



):



    """



    Split one preprocessed EDF recording into overlapping windows.



    Input:



        data:



            Shape = (channels, samples)



        sfreq:



            Sampling frequency after preprocessing.



        window_sec:



            Window length in seconds.



        overlap:



            Fractional overlap.



        seizure_times:



            List of seizure intervals.



    Returns:



        windows:



            Shape = (n_windows, channels, samples)



        labels:



            Shape = (n_windows,)



        start_times:



            Start time of every window in seconds.



        end_times:



            End time of every window in seconds.



        window_indices:



            Chronological index of every window.



    """



    # Number of samples in one window



    win_len = int(



        window_sec * sfreq



    )



    # For 50% overlap:



    #



    # 4 sec window



    # 50% overlap



    #



    # stride = 2 sec



    step = int(



        win_len * (1 - overlap)



    )



    n_samples = data.shape[1]



    windows = []



    labels = []



    start_times = []



    end_times = []



    window_indices = []



    # -------------------------------------------------------------



    # Generate windows chronologically



    # -------------------------------------------------------------



    for window_idx, start_sample in enumerate(



        range(



            0,



            n_samples - win_len + 1,



            step



        )



    ):



        end_sample = (



            start_sample + win_len



        )



        start_sec = (



            start_sample / sfreq



        )



        end_sec = (



            end_sample / sfreq



        )



        # ---------------------------------------------------------



        # Determine seizure label



        #



        # A window is labelled seizure if it overlaps



        # with any seizure interval.



        # ---------------------------------------------------------



        label = 0



        for (



            s_start,



            s_end



        ) in seizure_times:



            if (



                start_sec < s_end



                and



                end_sec > s_start



            ):



                label = 1



                break



        # ---------------------------------------------------------



        # Store window



        # ---------------------------------------------------------



        windows.append(



            data[



                :,



                start_sample:end_sample



            ]



        )



        labels.append(



            label



        )



        start_times.append(



            start_sec



        )



        end_times.append(



            end_sec



        )



        window_indices.append(



            window_idx



        )



    # -------------------------------------------------------------



    # Convert to memory-efficient NumPy arrays



    # -------------------------------------------------------------



    windows = np.array(



        windows,



        dtype=np.float32



    )



    labels = np.array(



        labels,



        dtype=np.int8



    )



    start_times = np.array(



        start_times,



        dtype=np.float32



    )



    end_times = np.array(



        end_times,



        dtype=np.float32



    )



    window_indices = np.array(



        window_indices,



        dtype=np.int32



    )



    return (



        windows,



        labels,



        start_times,



        end_times,



        window_indices



    )



# =====================================================================



# 4. Find EDF files



# =====================================================================



edf_files = sorted(



    DATA_DIR.glob("*.edf")



)



print(



    f"Found {len(edf_files)} EDF files "



    f"for {SUBJECT}"



)



# =====================================================================



# 5. Variables used during processing



# =====================================================================



reference_n_channels = None



# Keep track of successfully processed files



saved_chunks = []



# Keep track of skipped files



skipped = []



# Store channel names from the first valid EDF



reference_channel_names = None



# =====================================================================



# 6. Process each EDF file



# =====================================================================



for i, edf_file in enumerate(



    edf_files,



    1



):



    try:



        print(



            f"\n[{i}/{len(edf_files)}] "



            f"Processing {edf_file.name}"



        )



        # ---------------------------------------------------------



        # Load EDF



        # ---------------------------------------------------------



        raw = mne.io.read_raw_edf(



            edf_file,



            preload=True,



            verbose=False



        )

        # ---------------------------------------------------------

        # Remove non-standard extra channels

        #

        # CHB09 contains an additional VNS channel.

        # CHB12-CHB14 contain five EDF channels whose original

        # labels are "-" (MNE renames them to --0 ... --4).

        #

        # The project uses a fixed 23-channel EEG representation.

        # Remove only these documented extra channels.

        # ---------------------------------------------------------



        extra_channels = []



        if "VNS" in raw.ch_names:

            extra_channels.append("VNS")



        dash_channels = [

            ch for ch in raw.ch_names

            if ch == "-" or ch.startswith("--")

        ]



        extra_channels.extend(dash_channels)



        if extra_channels:

            print(

                f"  Removing extra channels from "

                f"{edf_file.name}: {extra_channels}"

            )

            raw.drop_channels(extra_channels)







        # ---------------------------------------------------------



        # Remove extra VNS channel in CHB09



        #



        # CHB09 files after chb09_01 contain the same 23 EEG



        # channels plus an additional VNS channel.



        #



        # For this project, we keep the fixed 23-channel EEG



        # representation and exclude only this extra channel.



        # ---------------------------------------------------------



        if "VNS" in raw.ch_names:



            print(



                f"  Removing extra VNS channel from "



                f"{edf_file.name}"



            )



            raw.drop_channels(["VNS"])



        # ---------------------------------------------------------

        # CHB12-specific channel layout validation

        #

        # CHB12 contains several different recording layouts. We only

        # keep recordings whose remaining EEG channels match the fixed

        # 23-channel bipolar representation used by this project.

        #

        # MNE renames the two duplicate EDF labels "T8-P8" to

        # "T8-P8-0" and "T8-P8-1". These are the two expected

        # T8-P8 positions. We canonicalize those names only for

        # validation; the raw data/channel positions are unchanged.

        #

        # CHB12_32-42 additionally contain LOC-ROC, which is removed

        # because it is not part of the fixed 23-channel EEG layout.

        #

        # CHB12_27 (-CS2 montage) and CHB12_28/29 (single-electrode

        # layout) fail this exact check and are skipped.

        # ---------------------------------------------------------



        if SUBJECT.lower() == "chb12":

            if "LOC-ROC" in raw.ch_names:

                print(

                    f"  Removing auxiliary channel from "

                    f"{edf_file.name}: LOC-ROC"

                )

                raw.drop_channels(["LOC-ROC"])



            chb12_target_channels = [

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



            canonical_channel_names = [

                (

                    "T8-P8"

                    if ch in {"T8-P8-0", "T8-P8-1"}

                    else ch

                )

                for ch in raw.ch_names

            ]



            if canonical_channel_names != chb12_target_channels:

                print(

                    "  SKIPPED: incompatible CHB12 channel "

                    "layout/montage."

                )

                print(

                    f"  Remaining channels: {raw.ch_names}"

                )

                print(

                    "  Expected 23-channel logical layout: "

                    f"{chb12_target_channels}"

                )

                skipped.append(edf_file.name)

                del raw

                gc.collect()

                continue



        # ---------------------------------------------------------



        # ---------------------------------------------------------
        # CHB15-specific channel handling
        #
        # CHB15 recordings contain the same 23-channel scalp EEG
        # layout plus auxiliary/reference channels. Some recordings
        # also contain PZ-OZ, which is not part of our fixed 23-channel
        # target layout. Remove only these verified auxiliary channels.
        # T8-P8 duplicates are canonicalized only for validation; raw
        # channel positions/data are not reordered.
        # ---------------------------------------------------------

        if SUBJECT.lower() == "chb15":

            auxiliary_channels = [
                ch for ch in raw.ch_names
                if ch.endswith("-Ref")
            ]

            if "PZ-OZ" in raw.ch_names:
                auxiliary_channels.append("PZ-OZ")

            if auxiliary_channels:
                print(
                    f"  Removing auxiliary channels from "
                    f"{edf_file.name}: {auxiliary_channels}"
                )
                raw.drop_channels(auxiliary_channels)

            chb15_target_channels = [
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FZ-CZ", "CZ-PZ",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",
            ]

            canonical_channel_names = [
                "T8-P8" if ch in {"T8-P8-0", "T8-P8-1"} else ch
                for ch in raw.ch_names
            ]

            if canonical_channel_names != chb15_target_channels:
                print("  SKIPPED: incompatible CHB15 channel layout/montage.")
                print(f"  Remaining channels: {raw.ch_names}")
                print(
                    "  Expected 23-channel logical layout: "
                    f"{chb15_target_channels}"
                )
                skipped.append(edf_file.name)
                del raw
                gc.collect()
                continue



        # Resample:



        #



        # 256 Hz -> 128 Hz



        # ---------------------------------------------------------



        raw.resample(



            128,



            verbose=False



        )



        # ---------------------------------------------------------



        # Notch filter



        # ---------------------------------------------------------



        raw.notch_filter(



            freqs=NOTCH_FREQ,



            n_jobs=1,



            verbose=False



        )



        # ---------------------------------------------------------



        # Bandpass filter



        #



        # 0.5 Hz -> 40 Hz



        # ---------------------------------------------------------



        raw.filter(



            l_freq=BANDPASS_LOW,



            h_freq=BANDPASS_HIGH,



            n_jobs=1,



            verbose=False



        )



        # Sampling frequency after resampling



        sfreq = raw.info["sfreq"]



        # ---------------------------------------------------------



        # Get channel names



        # ---------------------------------------------------------



        channel_names = list(



            raw.ch_names



        )



        # ---------------------------------------------------------



        # Get EEG data



        #



        # Shape:



        #



        # (channels, samples)



        # ---------------------------------------------------------



        data = raw.get_data().astype(



            np.float32



        )



        # ---------------------------------------------------------



        # Check channel count



        # ---------------------------------------------------------



        if reference_n_channels is None:



            reference_n_channels = (



                data.shape[0]



            )



            reference_channel_names = (



                channel_names



            )



        elif (



            data.shape[0]



            != reference_n_channels



        ):



            print(



                f"  SKIPPED: channel count "



                f"{data.shape[0]} != "



                f"expected "



                f"{reference_n_channels}"



            )



            skipped.append(



                edf_file.name



            )



            del raw



            del data



            gc.collect()



            continue



        # ---------------------------------------------------------



        # Check channel ordering



        #



        # We preserve the existing project's behaviour of



        # using the same number of channels.



        #



        # We also record the actual channel names.



        # ---------------------------------------------------------



        if (



            reference_channel_names is not None



            and



            channel_names



            != reference_channel_names



        ):



            print(



                "  WARNING: channel names/order "



                "differ from the first valid EDF."



            )



        # ---------------------------------------------------------



        # Per-channel z-score normalization



        #



        # This is unchanged from your existing pipeline.



        # ---------------------------------------------------------



        channel_mean = data.mean(



            axis=1,



            keepdims=True



        )



        channel_std = data.std(



            axis=1,



            keepdims=True



        )



        data = (



            data - channel_mean



        ) / (



            channel_std + 1e-8



        )



        # ---------------------------------------------------------



        # Read seizure annotations



        # ---------------------------------------------------------



        seizure_times = (



            parse_seizures_for_file(



                SUMMARY_FILE,



                edf_file.name



            )



        )



        # ---------------------------------------------------------



        # Create windows



        # ---------------------------------------------------------



        (



            windows,



            labels,



            start_times,



            end_times,



            window_indices



        ) = make_windows(



            data,



            sfreq,



            WINDOW_SEC,



            WINDOW_OVERLAP,



            seizure_times



        )



        # ---------------------------------------------------------



        # Save per-file temporary data



        # ---------------------------------------------------------



        w_path = (



            TMP_DIR



            / f"{edf_file.stem}_windows.npy"



        )



        l_path = (



            TMP_DIR



            / f"{edf_file.stem}_labels.npy"



        )



        s_path = (



            TMP_DIR



            / f"{edf_file.stem}_start_times.npy"



        )



        e_path = (



            TMP_DIR



            / f"{edf_file.stem}_end_times.npy"



        )



        wi_path = (



            TMP_DIR



            / f"{edf_file.stem}_window_indices.npy"



        )



        # Save arrays



        np.save(



            w_path,



            windows



        )



        np.save(



            l_path,



            labels



        )



        np.save(



            s_path,



            start_times



        )



        np.save(



            e_path,



            end_times



        )



        np.save(



            wi_path,



            window_indices



        )



        # Record successful file



        saved_chunks.append(



            edf_file.stem



        )



        # ---------------------------------------------------------



        # Print file summary



        # ---------------------------------------------------------



        n_seizure = int(



            labels.sum()



        )



        file_type = (



            "seizure file"



            if seizure_times



            else "clean file"



        )



        print(



            f"  {len(windows)} windows, "



            f"{n_seizure} seizure "



            f"({file_type})"



        )



        # ---------------------------------------------------------



        # Free memory before next EDF



        # ---------------------------------------------------------



        del raw



        del data



        del windows



        del labels



        del start_times



        del end_times



        del window_indices



        gc.collect()



    except Exception as e:



        print(



            f"  SKIPPED "



            f"(error: {e})"



        )



        skipped.append(



            edf_file.name



        )



        # Make sure memory is released



        gc.collect()



# =====================================================================



# 7. Check that at least one file was processed



# =====================================================================



if not saved_chunks:



    print(



        "\nERROR: No EDF files were "



        "successfully processed."



    )



    shutil.rmtree(



        TMP_DIR,



        ignore_errors=True



    )



    sys.exit(1)



# =====================================================================



# 8. Combine all per-file chunks



# =====================================================================



print(



    f"\nCombining "



    f"{len(saved_chunks)} saved chunks..."



)



total_windows = 0



sample_shape = None



# -------------------------------------------------------------



# First pass:



# determine total number of windows



# -------------------------------------------------------------



for stem in saved_chunks:



    w = np.load(



        TMP_DIR



        / f"{stem}_windows.npy",



        mmap_mode="r"



    )



    total_windows += (



        w.shape[0]



    )



    if sample_shape is None:



        sample_shape = (



            w.shape[1:]



        )



    del w



# =====================================================================



# 9. Create output directory



# =====================================================================



OUT_DIR = Path(



    "data/processed"



)



OUT_DIR.mkdir(



    parents=True,



    exist_ok=True



)



# =====================================================================



# 10. Define final output paths



# =====================================================================



final_windows_path = (



    OUT_DIR



    / f"{SUBJECT}_windows.npy"



)



final_labels_path = (



    OUT_DIR



    / f"{SUBJECT}_labels.npy"



)



final_file_ids_path = (



    OUT_DIR



    / f"{SUBJECT}_file_ids.npy"



)



final_file_names_path = (



    OUT_DIR



    / f"{SUBJECT}_file_names.txt"



)



final_metadata_path = (



    OUT_DIR



    / f"{SUBJECT}_metadata.csv"



)



final_channel_names_path = (



    OUT_DIR



    / f"{SUBJECT}_channel_names.txt"



)



# =====================================================================



# 11. Create memory-mapped final window array



# =====================================================================



final_windows = np.lib.format.open_memmap(



    final_windows_path,



    mode="w+",



    dtype=np.float32,



    shape=(



        total_windows,



        *sample_shape



    )



)



# Labels can comfortably stay in RAM



final_labels = np.zeros(



    total_windows,



    dtype=np.int8



)



# File IDs



final_file_ids = np.zeros(



    total_windows,



    dtype=np.int32



)



# =====================================================================



# 12. Combine files + create metadata



# =====================================================================



offset = 0



metadata_rows = []



for file_idx, stem in enumerate(



    saved_chunks



):



    # -------------------------------------------------------------



    # Load per-file arrays



    # -------------------------------------------------------------



    w = np.load(



        TMP_DIR



        / f"{stem}_windows.npy"



    )



    l = np.load(



        TMP_DIR



        / f"{stem}_labels.npy"



    )



    start_times = np.load(



        TMP_DIR



        / f"{stem}_start_times.npy"



    )



    end_times = np.load(



        TMP_DIR



        / f"{stem}_end_times.npy"



    )



    window_indices = np.load(



        TMP_DIR



        / f"{stem}_window_indices.npy"



    )



    n = w.shape[0]



    # -------------------------------------------------------------



    # Copy EEG windows



    # -------------------------------------------------------------



    final_windows[



        offset:offset + n



    ] = w



    # -------------------------------------------------------------



    # Copy labels



    # -------------------------------------------------------------



    final_labels[



        offset:offset + n



    ] = l



    # -------------------------------------------------------------



    # Assign file ID



    #



    # Every window from this EDF receives the same file ID.



    # -------------------------------------------------------------



    final_file_ids[



        offset:offset + n



    ] = file_idx



    # -------------------------------------------------------------



    # Create metadata



    # -------------------------------------------------------------



    for j in range(n):



        metadata_rows.append({



            "dataset_id": "chbmit",



            "subject_id": SUBJECT,



            "edf_id": stem,



            "window_index": int(



                window_indices[j]



            ),



            "start_sec": float(



                start_times[j]



            ),



            "end_sec": float(



                end_times[j]



            ),



            "label": int(



                l[j]



            ),



            "sampling_rate": float(



                sfreq



            ),



            "n_channels": int(



                w.shape[1]



            ),



            "n_samples": int(



                w.shape[2]



            )



        })



    offset += n



    # -------------------------------------------------------------



    # Release memory



    # -------------------------------------------------------------



    del w



    del l



    del start_times



    del end_times



    del window_indices



    gc.collect()



# =====================================================================



# 13. Flush final window array



# =====================================================================



final_windows.flush()



# =====================================================================



# 14. Save labels and file IDs



# =====================================================================



np.save(



    final_labels_path,



    final_labels



)



np.save(



    final_file_ids_path,



    final_file_ids



)



# =====================================================================



# 15. Save EDF file names



# =====================================================================



with open(



    final_file_names_path,



    "w"



) as f:



    f.write(



        "\n".join(



            saved_chunks



        )



    )



# =====================================================================



# 16. Save channel names



# =====================================================================



if reference_channel_names is not None:



    with open(



        final_channel_names_path,



        "w"



    ) as f:



        for channel_name in (



            reference_channel_names



        ):



            f.write(



                channel_name



                + "\n"



            )



# =====================================================================



# 17. Save metadata CSV



# =====================================================================



metadata_df = pd.DataFrame(



    metadata_rows



)



metadata_df.to_csv(



    final_metadata_path,



    index=False



)



# =====================================================================



# 18. Clean up temporary files



# =====================================================================



shutil.rmtree(



    TMP_DIR



)



# =====================================================================



# 19. Final summary



# =====================================================================



print(



    "\n"



    + "=" * 60



)



print(



    f"SUMMARY - {SUBJECT}"



)



print(



    "=" * 60



)



print(



    f"Files processed: "



    f"{len(saved_chunks)} / "



    f"{len(edf_files)}"



)



if skipped:



    print(



        f"Files skipped: "



        f"{skipped}"



    )



print(



    f"Total windows: "



    f"{total_windows}"



)



print(



    f"Seizure windows: "



    f"{final_labels.sum()} / "



    f"{total_windows} "



    f"({100 * final_labels.mean():.2f}%)"



)



print(



    f"\nSaved windows to:"



    f"\n{final_windows_path}"



)



print(



    f"\nSaved labels to:"



    f"\n{final_labels_path}"



)



print(



    f"\nSaved file IDs to:"



    f"\n{final_file_ids_path}"



)



print(



    f"\nSaved file names to:"



    f"\n{final_file_names_path}"



)



print(



    f"\nSaved channel names to:"



    f"\n{final_channel_names_path}"



)



print(



    f"\nSaved metadata to:"



    f"\n{final_metadata_path}"



)



print(



    "\nPreprocessing completed successfully."



)
