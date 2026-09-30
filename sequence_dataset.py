"""Chronological EEG sequence indexing and lazy loading for LSTM training."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceRecord:
    subject_index: int
    start: int


class EEGSequenceDataset(Dataset):
    def __init__(self, feature_arrays, labels_arrays, records, sequence_length):
        self.feature_arrays = feature_arrays
        self.labels_arrays = labels_arrays
        self.records = records
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        features = self.feature_arrays[record.subject_index]
        labels = self.labels_arrays[record.subject_index]

        end = record.start + self.sequence_length
        x = np.asarray(features[record.start:end], dtype=np.float32)
        y = np.float32(labels[end - 1])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def build_sequence_records(file_ids, allowed_files, sequence_length, subject_index):
    allowed_files = set(int(x) for x in allowed_files)
    records = []

    boundaries = np.flatnonzero(file_ids[1:] != file_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(file_ids)]))

    for start, end in zip(starts, ends):
        if int(file_ids[start]) not in allowed_files:
            continue
        if end - start < sequence_length:
            continue
        records.extend(
            SequenceRecord(subject_index, i)
            for i in range(start, end - sequence_length + 1)
        )

    return records
