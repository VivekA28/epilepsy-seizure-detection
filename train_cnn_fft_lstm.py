"""Train the local CNN + FFT + LSTM model on cached features."""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DATA_DIR = Path("data/processed")
CNN_DIR = DATA_DIR / "cnn_features"
MODEL_DIR = Path("models")

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

CNN_FEATURE_DIM = 128
FFT_FEATURE_DIM = 115
LSTM_HIDDEN_SIZE = 128
DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 15
LEARNING_RATE = 1e-3


@dataclass(frozen=True)
class SequenceRecord:
    subject_index: int
    start: int


class DualFeatureSequenceDataset(Dataset):
    def __init__(self, cnn_arrays, fft_arrays, labels_arrays, records, sequence_length):
        self.cnn_arrays = cnn_arrays
        self.fft_arrays = fft_arrays
        self.labels_arrays = labels_arrays
        self.records = records
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        end = record.start + self.sequence_length

        cnn = np.array(self.cnn_arrays[record.subject_index][record.start:end], dtype=np.float32, copy=True)
        fft = np.array(self.fft_arrays[record.subject_index][record.start:end], dtype=np.float32, copy=True)
        label = np.float32(self.labels_arrays[record.subject_index][end - 1])

        return torch.from_numpy(cnn), torch.from_numpy(fft), torch.tensor(label, dtype=torch.float32)


class CNNFFTLSTM(nn.Module):
    def __init__(self, cnn_dim=CNN_FEATURE_DIM, fft_dim=FFT_FEATURE_DIM, hidden_size=LSTM_HIDDEN_SIZE):
        super().__init__()
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.fft_norm = nn.LayerNorm(fft_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_dim + fft_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, cnn_features, fft_features):
        cnn_features = self.cnn_norm(cnn_features)
        fft_features = self.fft_norm(fft_features)
        fused = torch.cat([cnn_features, fft_features], dim=-1)
        output, _ = self.lstm(fused)
        return self.classifier(output[:, -1, :])


def load_data(subjects):
    cnn_arrays, fft_arrays, labels_arrays, file_arrays = [], [], [], []

    for subject in subjects:
        labels = np.load(DATA_DIR / f"{subject}_labels.npy", mmap_mode="r")
        file_ids = np.load(DATA_DIR / f"{subject}_file_ids.npy", mmap_mode="r")
        cnn = np.load(CNN_DIR / f"{subject}_cnn_features.npy", mmap_mode="r")
        fft = np.load(DATA_DIR / f"{subject}_fft_features.npy", mmap_mode="r")

        if len(labels) != len(file_ids):
            raise ValueError(f"Label/file-id mismatch for {subject}")
        if cnn.shape != (len(labels), CNN_FEATURE_DIM):
            raise ValueError(f"Bad CNN feature shape for {subject}: {cnn.shape}")
        if fft.shape != (len(labels), FFT_FEATURE_DIM):
            raise ValueError(f"Bad FFT feature shape for {subject}: {fft.shape}")

        cnn_arrays.append(cnn)
        fft_arrays.append(fft)
        labels_arrays.append(labels)
        file_arrays.append(np.asarray(file_ids, dtype=np.int64))

    return cnn_arrays, fft_arrays, labels_arrays, file_arrays


def split_files(file_arrays, labels_arrays):
    global_file_arrays = []
    next_file_id = 0

    for file_ids in file_arrays:
        shifted = file_ids + next_file_id
        global_file_arrays.append(shifted)
        next_file_id = int(shifted.max()) + 1

    all_file_ids = np.concatenate(global_file_arrays)
    all_labels = np.concatenate([np.asarray(labels) for labels in labels_arrays])
    unique_files = np.unique(all_file_ids)
    file_has_seizure = np.array(
        [all_labels[all_file_ids == file_id].max() for file_id in unique_files],
        dtype=np.int8,
    )

    n_seizure = int(file_has_seizure.sum())
    n_clean = len(file_has_seizure) - n_seizure
    if n_seizure < 2 or n_clean < 2:
        raise RuntimeError(f"Not enough seizure/clean EDFs: {n_seizure} seizure, {n_clean} clean.")

    train_files, temp_files, _, temp_labels = train_test_split(
        unique_files,
        file_has_seizure,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=file_has_seizure,
    )

    temp_seizure = int(temp_labels.sum())
    temp_clean = len(temp_labels) - temp_seizure
    if temp_seizure < 2 or temp_clean < 2:
        raise RuntimeError(
            f"Temporary split too small for stratified val/test split: "
            f"{temp_seizure} seizure, {temp_clean} clean."
        )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    return global_file_arrays, train_files, val_files, test_files


def build_sequence_records(global_file_ids, allowed_files, sequence_length, subject_index):
    allowed_files = set(map(int, allowed_files))
    boundaries = np.flatnonzero(global_file_ids[1:] != global_file_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(global_file_ids)]))
    records = []

    for start, end in zip(starts, ends):
        if int(global_file_ids[start]) not in allowed_files:
            continue
        if end - start < sequence_length:
            continue
        records.extend(SequenceRecord(subject_index, i) for i in range(start, end - sequence_length + 1))

    return records


def make_datasets(subjects, sequence_length):
    cnn_arrays, fft_arrays, labels_arrays, file_arrays = load_data(subjects)
    global_file_arrays, train_files, val_files, test_files = split_files(file_arrays, labels_arrays)
    partitions = {"train": train_files, "val": val_files, "test": test_files}
    datasets = {}

    print("\nSequence counts:")
    for partition, allowed_files in partitions.items():
        records = []
        for subject_index, global_file_ids in enumerate(global_file_arrays):
            records.extend(build_sequence_records(global_file_ids, allowed_files, sequence_length, subject_index))

        datasets[partition] = DualFeatureSequenceDataset(
            cnn_arrays, fft_arrays, labels_arrays, records, sequence_length
        )

        positive = sum(
            int(labels_arrays[r.subject_index][r.start + sequence_length - 1])
            for r in records
        )
        print(f"  {partition:5s}: {len(records)} sequences ({positive} seizure targets)")

    return datasets


def make_train_loader(dataset, batch_size):
    target_labels = np.fromiter(
        (
            dataset.labels_arrays[r.subject_index][r.start + dataset.sequence_length - 1]
            for r in dataset.records
        ),
        dtype=np.int64,
        count=len(dataset.records),
    )
    class_counts = np.bincount(target_labels, minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise RuntimeError("Training sequences contain only one class.")

    sample_weights = torch.as_tensor((1.0 / class_counts)[target_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for cnn_batch, fft_batch, y_batch in loader:
            logits = model(cnn_batch.to(device), fft_batch.to(device)).squeeze(-1)
            predictions.append((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
            targets.append(y_batch.numpy().astype(np.int64))

    return np.concatenate(predictions), np.concatenate(targets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+", help="Subjects, e.g. chb01 chb02 ...")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    datasets = make_datasets(args.subjects, args.sequence_length)
    train_loader = make_train_loader(datasets["train"], args.batch_size)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFFTLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nUsing device: {device}")
    print(f"Sequence length: {args.sequence_length} windows")
    print(f"CNN feature dimension: {CNN_FEATURE_DIM}")
    print(f"FFT feature dimension: {FFT_FEATURE_DIM}")
    print(f"Fused feature dimension: {CNN_FEATURE_DIM + FFT_FEATURE_DIM}")
    print(f"LSTM hidden size: {LSTM_HIDDEN_SIZE}")

    best_val_f1, best_state, best_epoch = -1.0, None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for cnn_batch, fft_batch, y_batch in train_loader:
            cnn_batch = cnn_batch.to(device)
            fft_batch = fft_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(cnn_batch, fft_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_batch.size(0)

        avg_loss = total_loss / len(datasets["train"])
        val_preds, val_targets = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_recall = ((val_preds[val_targets == 1] == 1).mean() if (val_targets == 1).any() else 0.0)
        val_precision = ((val_targets[val_preds == 1] == 1).mean() if (val_preds == 1).any() else 0.0)

        print(
            f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f} - "
            f"val recall: {val_recall:.2f} - val precision: {val_precision:.2f} - val f1: {val_f1:.2f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> new best validation checkpoint (f1={best_val_f1:.2f})")

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")

    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint from epoch {best_epoch} (validation f1={best_val_f1:.2f})")
    print("Evaluating TEST set once using the restored validation-selected checkpoint...")

    test_preds, test_targets = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(test_targets, test_preds, target_names=["Non-seizure", "Seizure"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test_targets, test_preds))

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"cnn_fft_lstm_frozen_{'_'.join(args.subjects)}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
