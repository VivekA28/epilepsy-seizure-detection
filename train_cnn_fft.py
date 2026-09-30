"""
Model 3 — CNN + FFT feature fusion ablation.

Combines cached 128-D CNN time-domain features with cached 115-D FFT frequency
band-power features using dual LayerNorm before classification.
"""

import argparse
import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

DATA_DIR = Path("data/processed")
CNN_DIR = DATA_DIR / "cnn_features"
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

CNN_FEATURE_DIM = 128
FFT_FEATURE_DIM = 115
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3


class FusedFeatureDataset(Dataset):
    """Lazy Dataset yielding (cnn_features, fft_features, label) tuples."""

    def __init__(self, cnn_arrays, fft_arrays, labels_arrays, records: List[Tuple[int, int]]):
        self.cnn_arrays = cnn_arrays
        self.fft_arrays = fft_arrays
        self.labels_arrays = labels_arrays
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        subject_idx, window_idx = self.records[index]
        cnn = np.asarray(self.cnn_arrays[subject_idx][window_idx], dtype=np.float32)
        fft = np.asarray(self.fft_arrays[subject_idx][window_idx], dtype=np.float32)
        label = np.float32(self.labels_arrays[subject_idx][window_idx])
        return torch.from_numpy(cnn), torch.from_numpy(fft), torch.tensor(label, dtype=torch.float32)


class CNNFFT(nn.Module):
    """
    Dual-stream feature fusion architecture.
    Normalizes CNN (128-D) and FFT (115-D) independently with LayerNorm before
    concatenating into a 243-D vector to prevent feature scale dominance.
    """

    def __init__(self, cnn_dim: int = CNN_FEATURE_DIM, fft_dim: int = FFT_FEATURE_DIM):
        super().__init__()
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.fft_norm = nn.LayerNorm(fft_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim + fft_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, cnn_features: torch.Tensor, fft_features: torch.Tensor) -> torch.Tensor:
        cnn_normed = self.cnn_norm(cnn_features)
        fft_normed = self.fft_norm(fft_features)
        fused = torch.cat([cnn_normed, fft_normed], dim=1)
        return self.classifier(fused)


def load_subjects(subjects: List[str]):
    """Load memory-mapped arrays and labels for all specified subjects."""
    cnn_arrays, fft_arrays, labels_arrays, file_arrays = [], [], [], []

    for s in subjects:
        labels = np.load(DATA_DIR / f"{s}_labels.npy", mmap_mode="r")
        file_ids = np.load(DATA_DIR / f"{s}_file_ids.npy", mmap_mode="r")
        cnn = np.load(CNN_DIR / f"{s}_cnn_features.npy", mmap_mode="r")
        fft = np.load(DATA_DIR / f"{s}_fft_features.npy", mmap_mode="r")

        if len(labels) != len(file_ids):
            raise ValueError(f"Label/file-id mismatch for {s}")
        if cnn.shape != (len(labels), CNN_FEATURE_DIM):
            raise ValueError(f"Invalid CNN shape for {s}: {cnn.shape}")
        if fft.shape != (len(labels), FFT_FEATURE_DIM):
            raise ValueError(f"Invalid FFT shape for {s}: {fft.shape}")

        cnn_arrays.append(cnn)
        fft_arrays.append(fft)
        labels_arrays.append(labels)
        file_arrays.append(np.asarray(file_ids, dtype=np.int64))

    return cnn_arrays, fft_arrays, labels_arrays, file_arrays


def split_files(file_arrays, labels_arrays):
    """Partition recordings into Train (70%), Val (15%), Test (15%) at file level."""
    global_file_arrays = []
    next_file_id = 0

    for file_ids in file_arrays:
        shifted = file_ids + next_file_id
        global_file_arrays.append(shifted)
        next_file_id = int(shifted.max()) + 1

    all_file_ids = np.concatenate(global_file_arrays)
    all_labels = np.concatenate([np.asarray(l) for l in labels_arrays])
    unique_files = np.unique(all_file_ids)
    file_has_seizure = np.array([all_labels[all_file_ids == f].max() for f in unique_files], dtype=np.int8)

    n_seizure = int(file_has_seizure.sum())
    n_clean = int(len(file_has_seizure) - n_seizure)
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
    temp_clean = int(len(temp_labels) - temp_seizure)
    if temp_seizure < 2 or temp_clean < 2:
        raise RuntimeError(f"Temporary split too small for stratified val/test: {temp_seizure} seizure, {temp_clean} clean.")

    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=RANDOM_STATE,
        stratify=temp_labels,
    )

    allowed = {
        "train": set(train_files.tolist()),
        "val": set(val_files.tolist()),
        "test": set(test_files.tolist()),
    }

    records = {"train": [], "val": [], "test": []}
    for subject_idx, file_ids in enumerate(global_file_arrays):
        for local_idx, g_fid in enumerate(file_ids):
            g_fid = int(g_fid)
            for part in records:
                if g_fid in allowed[part]:
                    records[part].append((subject_idx, local_idx))
                    break

    return records


def make_datasets(subjects: List[str]):
    cnn_arrays, fft_arrays, labels_arrays, file_arrays = load_subjects(subjects)
    records = split_files(file_arrays, labels_arrays)

    datasets = {
        p: FusedFeatureDataset(cnn_arrays, fft_arrays, labels_arrays, records[p])
        for p in records
    }

    print("\nFile-level partition:")
    for p, ds in datasets.items():
        lbls = np.fromiter((labels_arrays[s][i] for s, i in records[p]), dtype=np.int8, count=len(records[p]))
        print(f"  {p:5s}: {len(ds):,} windows ({int(lbls.sum())} seizure)")

    return datasets


def make_train_loader(dataset: FusedFeatureDataset) -> DataLoader:
    targets = np.fromiter((dataset.labels_arrays[s][i] for s, i in dataset.records), dtype=np.int64, count=len(dataset.records))
    counts = np.bincount(targets, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Training set contains only one class.")

    weights = torch.as_tensor((1.0 / counts)[targets], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for cnn_batch, fft_batch, y_batch in loader:
            logits = model(cnn_batch.to(device), fft_batch.to(device)).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            predictions.append(preds)
            targets.append(y_batch.numpy().astype(np.int64))

    return np.concatenate(predictions), np.concatenate(targets)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN + FFT feature fusion model.")
    parser.add_argument("subjects", nargs="+", help="Subjects to train on (e.g. chb01 chb02 ...)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    datasets = make_datasets(args.subjects)
    train_loader = make_train_loader(datasets["train"])
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model = CNNFFT().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for cnn_b, fft_b, y_b in train_loader:
            cnn_b, fft_b, y_b = cnn_b.to(device), fft_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(cnn_b, fft_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_b.size(0)

        avg_loss = total_loss / len(datasets["train"])
        val_preds, val_targets = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_recall = (val_preds[val_targets == 1] == 1).mean() if (val_targets == 1).any() else 0.0
        val_precision = (val_targets[val_preds == 1] == 1).mean() if (val_preds == 1).any() else 0.0

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
        raise RuntimeError("No validation checkpoint produced.")

    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint from epoch {best_epoch} (validation f1={best_val_f1:.2f})")
    print("Evaluating TEST set once using restored validation-selected checkpoint...")

    test_preds, test_targets = evaluate(model, test_loader, device)
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(test_targets, test_preds, target_names=["Non-seizure", "Seizure"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test_targets, test_preds))

    MODEL_DIR.mkdir(exist_ok=True)
    model_name = f"cnn_fft_frozen_{'_'.join(args.subjects)}.pt"
    model_path = MODEL_DIR / model_name
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
