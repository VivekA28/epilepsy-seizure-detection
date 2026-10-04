from pathlib import Path
import re
import pandas as pd


SIENA_DIR = Path("data/siena")
SEIZURE_DIR = SIENA_DIR / "seizure_lists"
RECORDS_FILE = SIENA_DIR / "RECORDS"


def normalize_filename(filename):
    """Normalize known SIENA filename inconsistencies."""

    if filename is None:
        return None

    filename = filename.strip()

    # Known typo: PNO6 -> PN06
    filename = filename.replace("PNO6", "PN06")

    # Known typo in PN11 seizure list
    if filename == "PN11-.edf":
        filename = "PN11-1.edf"

    return filename


def read_records():
    """Read official RECORDS file."""

    records = []

    with RECORDS_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("/")

            if len(parts) != 2:
                continue

            subject = parts[0]
            filename = parts[1]

            records.append({
                "subject": subject,
                "filename": filename,
                "path": line,
            })

    return records


def parse_time(time_string):
    """Convert HH.MM.SS or HH:MM:SS into seconds."""

    if time_string is None:
        return None

    time_string = time_string.strip()
    time_string = time_string.replace(":", ".")

    parts = time_string.split(".")

    if len(parts) != 3:
        return None

    try:
        h, m, s = map(int, parts)

        if h > 23 or m > 59 or s > 59:
            return None

        return h * 3600 + m * 60 + s

    except ValueError:
        return None


def extract_filename(block):
    """Extract filename from a seizure block if present."""

    match = re.search(
        r"File name:\s*(.+?)(?:\n|$)",
        block,
        re.IGNORECASE
    )

    if not match:
        return None

    filename = match.group(1).strip()

    return normalize_filename(filename)


def extract_start_time(block):
    """Extract seizure start time."""

    match = re.search(
        r"(?:Seizure start time|Start time):\s*([0-9:.]+)",
        block,
        re.IGNORECASE
    )

    if not match:
        return None

    return match.group(1).strip()


def extract_end_time(block):
    """Extract seizure end time."""

    match = re.search(
        r"(?:Seizure end time|End time):\s*([0-9:.]+)",
        block,
        re.IGNORECASE
    )

    if not match:
        return None

    return match.group(1).strip()


def parse_seizure_list(path):
    """
    Parse one SIENA seizure-list file.

    Handles:
    - File name present
    - File name absent
    - Start time / End time format
    - Seizure start time / Seizure end time format
    """

    text = path.read_text(encoding="utf-8")

    patient_match = re.search(
        r"^\s*(PN\d+)",
        text,
        re.MULTILINE
    )

    if not patient_match:
        return []

    patient = patient_match.group(1)

    # Split at each seizure number.
    blocks = re.split(
        r"(?=Seizure\s+n\s*\d+)",
        text,
        flags=re.IGNORECASE
    )

    seizures = []

    for block in blocks:

        if not re.search(
            r"Seizure\s+n\s*\d+",
            block,
            re.IGNORECASE
        ):
            continue

        filename = extract_filename(block)

        start_time = extract_start_time(block)

        end_time = extract_end_time(block)

        if start_time is None or end_time is None:
            continue

        seizures.append({
            "patient": patient,
            "filename": filename,
            "start_time": start_time,
            "end_time": end_time,
            "start_seconds": parse_time(start_time),
            "end_seconds": parse_time(end_time),
        })

    return seizures


def infer_missing_filename(seizure, records):
    """
    Infer EDF filename when the seizure list does not provide one.

    Important case:
    PN01 has one EDF recording listed in RECORDS,
    while its seizure list contains two seizures without
    individual File name fields.
    """

    if seizure["filename"] is not None:
        return seizure["filename"]

    patient = seizure["patient"]

    patient_records = [
        r for r in records
        if r["subject"] == patient
    ]

    # If the patient has exactly one EDF, use it.
    if len(patient_records) == 1:
        return patient_records[0]["filename"]

    return None


def main():

    print("=" * 90)
    print("SIENA RECORDING / SEIZURE MAPPING")
    print("=" * 90)

    # ---------------------------------------------------------
    # 1. Read RECORDS
    # ---------------------------------------------------------

    records = read_records()

    print()
    print(f"EDF recordings listed in RECORDS : {len(records)}")

    # ---------------------------------------------------------
    # 2. Parse seizure lists
    # ---------------------------------------------------------

    seizure_files = sorted(
        SEIZURE_DIR.glob("Seizures-list-*.txt")
    )

    all_seizures = []

    for path in seizure_files:

        seizures = parse_seizure_list(path)

        all_seizures.extend(seizures)

    print(f"Seizures parsed                  : {len(all_seizures)}")

    # ---------------------------------------------------------
    # 3. Infer missing filenames
    # ---------------------------------------------------------

    for seizure in all_seizures:

        if seizure["filename"] is None:

            inferred = infer_missing_filename(
                seizure,
                records
            )

            seizure["filename"] = inferred

    # ---------------------------------------------------------
    # 4. Create recording lookup
    # ---------------------------------------------------------

    record_lookup = {}

    for record in records:

        key = (
            record["subject"],
            record["filename"].lower()
        )

        record_lookup[key] = record["path"]

    # ---------------------------------------------------------
    # 5. Map seizures to EDF recordings
    # ---------------------------------------------------------

    mapped = []
    unmatched = []

    for seizure in all_seizures:

        patient = seizure["patient"]
        filename = seizure["filename"]

        if filename is None:

            unmatched.append(seizure)
            continue

        key = (
            patient,
            filename.lower()
        )

        record_path = record_lookup.get(key)

        if record_path is None:

            unmatched.append(seizure)

        else:

            mapped.append({
                "patient": patient,
                "filename": filename,
                "record_path": record_path,
                "start_time": seizure["start_time"],
                "end_time": seizure["end_time"],
                "start_seconds": seizure["start_seconds"],
                "end_seconds": seizure["end_seconds"],
            })

    # ---------------------------------------------------------
    # 6. Print mapping
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("SEIZURE → EDF MAPPING")
    print("=" * 90)

    mapping_df = pd.DataFrame(mapped)

    if not mapping_df.empty:

        print(
            mapping_df.to_string(index=False)
        )

    # ---------------------------------------------------------
    # 7. Unmatched seizures
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("UNMATCHED SEIZURES")
    print("=" * 90)

    if unmatched:

        for item in unmatched:

            print(
                f"{item['patient']} | "
                f"{item['filename']} | "
                f"{item['start_time']} → "
                f"{item['end_time']}"
            )

    else:

        print("None")

    # ---------------------------------------------------------
    # 8. Recording summary
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("RECORDING SUMMARY")
    print("=" * 90)

    subjects = sorted(
        set(r["subject"] for r in records)
    )

    for subject in subjects:

        subject_records = [
            r for r in records
            if r["subject"] == subject
        ]

        subject_seizures = [
            s for s in mapped
            if s["patient"] == subject
        ]

        print()
        print(
            f"{subject}: "
            f"{len(subject_records)} EDF recording(s), "
            f"{len(subject_seizures)} seizure(s)"
        )

        for record in subject_records:

            seizure_count = sum(
                1
                for s in subject_seizures
                if s["filename"].lower()
                == record["filename"].lower()
            )

            print(
                f"  {record['filename']:<22} "
                f"seizures={seizure_count}"
            )

    # ---------------------------------------------------------
    # 9. Save mapping
    # ---------------------------------------------------------

    output_file = SIENA_DIR / "siena_seizure_mapping.csv"

    mapping_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 90)
    print("OUTPUT")
    print("=" * 90)

    print(
        f"Saved seizure mapping to: "
        f"{output_file}"
    )

    # ---------------------------------------------------------
    # 10. Final validation
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print("VALIDATION")
    print("=" * 90)

    print(
        f"Total RECORDS EDFs : {len(records)}"
    )

    print(
        f"Total seizures     : {len(all_seizures)}"
    )

    print(
        f"Mapped seizures    : {len(mapped)}"
    )

    print(
        f"Unmatched seizures : {len(unmatched)}"
    )

    if len(all_seizures) == 47 and not unmatched:

        print()
        print(
            "SUCCESS: All 47 SIENA seizures "
            "mapped to EDF recordings."
        )

    else:

        print()
        print(
            "WARNING: Mapping requires "
            "further investigation."
        )


if __name__ == "__main__":
    main()