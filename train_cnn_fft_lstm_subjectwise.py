"""
Subject-wise CNN + FFT + LSTM multimodal temporal model for the 15-subject CHB-MIT experiment.

Reads cached 128-D CNN features from the corrected baseline CNN checkpoint and
115-D log-band-power FFT features.
Fuses representations per window using dual LayerNorm (CNN: 128-D, FFT: 115-D -> 243-D fused).
Constructs chronological sequences of 5 consecutive 4-second windows:
  W(t-4), W(t-3), W(t-2), W(t-1), W(t)
Target label is the label of the final window W(t).
Sequences never cross EDF boundaries, subject boundaries, or partition boundaries.

Architecture:
- Dual LayerNorm on CNN (128-D) and FFT (115-D) representations
- Concatenation -> 243-D input
- Single-layer unidirectional LSTM (input_size=243, hidden_size=128)
- Linear classifier head on final timestep representation output[:, -1, :]

Validation F1 selects the checkpoint; the held-out test set is evaluated once at the end.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sequence_dataset import build_sequence_records

DATA_DIR = Path("data/processed")
CNN_DIR = DATA_DIR / "cnn_features"
PARTITION_DIR = Path("data/partitions")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PARTITIONS = {
    "train": PARTITION_DIR / "train_metadata.csv",
    "validation": PARTITION_DIR / "validation_metadata.csv",
    "test": PARTITION_DIR / "test_metadata.csv",
}

SEED = 42
SEQUENCE_LENGTH = 5
CNN_FEATURE_DIM = 128
FFT_FEATURE_DIM = 115
FUSED_FEATURE_DIM = CNN_FEATURE_DIM + FFT_FEATURE_DIM  # 243
LSTM_HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
PIN_MEMORY = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_partitions():
    partitions = {}
    required = {
        "subject_id", "edf_id", "window_index", "start_sec", "end_sec",
        "label", "sampling_rate", "n_channels", "n_samples"
    }

    for split, path in PARTITIONS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/create_subject_split.py first."
            )
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        partitions[split] = df

    subject_sets = {
        split: set(df["subject_id"].astype(str).unique())
        for split, df in partitions.items()
    }

    # Strict subject partition leakage checks
    if subject_sets["train"] & subject_sets["validation"]:
        raise RuntimeError("Subject leakage: train/validation overlap.")
    if subject_sets["train"] & subject_sets["test"]:
        raise RuntimeError("Subject leakage: train/test overlap.")
    if subject_sets["validation"] & subject_sets["test"]:
        raise RuntimeError("Subject leakage: validation/test overlap.")

    print("=" * 90)
    print("SUBJECT-WISE CNN + FFT + LSTM MULTIMODAL TEMPORAL MODEL")
    print("=" * 90)
    for split in ("train", "validation", "test"):
        df = partitions[split]
        subjects = sorted(subject_sets[split])
        print(
            f"{split.capitalize():11s}: {', '.join(subjects)} | "
            f"{len(df):,} windows | {int(df['label'].sum()):,} seizure"
        )

    return partitions, subject_sets


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


def build_split_dataset(subjects, sequence_length):
    """
    Build dual-feature sequence dataset for a specific split given its list of subjects.
    Sequences are built strictly per subject and per EDF file so they never cross boundaries.
    """
    cnn_arrays = []
    fft_arrays = []
    labels_arrays = []
    records = []

    for subj_idx, subject in enumerate(subjects):
        cnn_path = CNN_DIR / f"{subject}_cnn_features.npy"
        fft_path = DATA_DIR / f"{subject}_fft_features.npy"
        label_path = DATA_DIR / f"{subject}_labels.npy"
        fid_path = DATA_DIR / f"{subject}_file_ids.npy"

        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing cached CNN features: {cnn_path}")
        if not fft_path.exists():
            raise FileNotFoundError(f"Missing cached FFT features: {fft_path}")
        if not label_path.exists() or not fid_path.exists():
            raise FileNotFoundError(f"Missing processed labels/file_ids for {subject}")

        cnn = np.load(cnn_path, mmap_mode="r")
        fft = np.load(fft_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        file_ids = np.load(fid_path, mmap_mode="r")

        n = len(labels)
        if len(cnn) != n or len(fft) != n or len(file_ids) != n:
            raise ValueError(f"Shape mismatch for {subject}: cnn={len(cnn)}, fft={len(fft)}, lbl={n}, fid={len(file_ids)}")
        if cnn.shape[1] != CNN_FEATURE_DIM:
            raise ValueError(f"Expected {CNN_FEATURE_DIM}-D CNN features, got {cnn.shape[1]} for {subject}")
        if fft.shape[1] != FFT_FEATURE_DIM:
            raise ValueError(f"Expected {FFT_FEATURE_DIM}-D FFT features, got {fft.shape[1]} for {subject}")

        cnn_arrays.append(cnn)
        fft_arrays.append(fft)
        labels_arrays.append(labels)

        allowed_files = np.unique(file_ids)
        subj_records = build_sequence_records(file_ids, allowed_files, sequence_length, subj_idx)
        records.extend(subj_records)

    ds = DualFeatureSequenceDataset(cnn_arrays, fft_arrays, labels_arrays, records, sequence_length)
    return ds


def make_train_sampler(dataset):
    """Sequence-level WeightedRandomSampler balancing seizure and non-seizure targets."""
    target_labels = np.fromiter(
        (dataset.labels_arrays[r.subject_index][r.start + dataset.sequence_length - 1]
         for r in dataset.records),
        dtype=np.int8,
        count=len(dataset.records),
    )
    class_counts = np.bincount(target_labels, minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise RuntimeError("Training sequences contain only one class.")
    class_weights = 1.0 / class_counts
    weights = torch.as_tensor(class_weights[target_labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), target_labels


def evaluate_split(model, loader, device, criterion=None):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for cnn_batch, fft_batch, y_batch in loader:
            cnn_batch = cnn_batch.to(device)
            fft_batch = fft_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(cnn_batch, fft_batch).squeeze(-1)

            if criterion is not None:
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.int64)

    avg_loss = total_loss / len(targets) if (criterion is not None and len(targets) > 0) else 0.0
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1 = f1_score(targets, preds, zero_division=0)

    metrics = {
        "loss": avg_loss,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "targets": targets,
    }
    return metrics


def main():
    seed_everything(SEED)
    partitions, subject_sets = load_partitions()

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    print("\nBuilding dual-feature sequence datasets (sequence_length = 5)...")
    train_ds = build_split_dataset(train_subjects, SEQUENCE_LENGTH)
    val_ds = build_split_dataset(val_subjects, SEQUENCE_LENGTH)
    test_ds = build_split_dataset(test_subjects, SEQUENCE_LENGTH)

    train_sampler, train_targets = make_train_sampler(train_ds)
    train_seizures = int(train_targets.sum())

    val_targets = np.fromiter(
        (val_ds.labels_arrays[r.subject_index][r.start + val_ds.sequence_length - 1]
         for r in val_ds.records),
        dtype=np.int8,
        count=len(val_ds.records),
    )
    val_seizures = int(val_targets.sum())

    test_targets_pre = np.fromiter(
        (test_ds.labels_arrays[r.subject_index][r.start + test_ds.sequence_length - 1]
         for r in test_ds.records),
        dtype=np.int8,
        count=len(test_ds.records),
    )
    test_seizures = int(test_targets_pre.sum())

    print("\nModel-facing sequence partitions:")
    print(f"  Train      : {len(train_ds):,} sequences ({train_seizures:,} seizure targets)")
    print(f"  Validation : {len(val_ds):,} sequences ({val_seizures:,} seizure targets)")
    print(f"  Test       : {len(test_ds):,} sequences ({test_seizures:,} seizure targets)")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = CNNFFTLSTM(
        cnn_dim=CNN_FEATURE_DIM,
        fft_dim=FFT_FEATURE_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = None
    history = []

    print("\n" + "=" * 90)
    print("TRAINING SUBJECT-WISE CNN + FFT + LSTM MODEL")
    print(f"Hyperparameters: Batch Size={BATCH_SIZE}, LR={LEARNING_RATE}, Epochs={EPOCHS}, Seed={SEED}")
    print(f"Fused Input Dimension: {FUSED_FEATURE_DIM} (CNN: {CNN_FEATURE_DIM} + FFT: {FFT_FEATURE_DIM})")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for cnn_batch, fft_batch, y_batch in train_loader:
            cnn_batch = cnn_batch.to(device)
            fft_batch = fft_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cnn_batch, fft_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        train_loss = total_loss / len(train_ds)

        # Validation evaluation
        val_metrics = evaluate_split(model, val_loader, device, criterion=criterion)
        val_loss = val_metrics["loss"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]
        val_f1 = val_metrics["f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_prec={val_prec:.4f} | "
            f"val_rec={val_rec:.4f} | "
            f"val_F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> new best validation checkpoint (val_F1={best_val_f1:.4f})")

    if best_state is None:
        raise RuntimeError("No validation checkpoint produced.")

    model.load_state_dict(best_state)
    print("\n" + "=" * 90)
    print(f"Restored best checkpoint from epoch {best_epoch} (validation F1={best_val_f1:.4f})")
    print("Evaluating held-out TEST set ONCE using the restored validation-selected checkpoint...")
    print("Test threshold fixed at 0.5")
    print("=" * 90)

    test_metrics = evaluate_split(model, test_loader, device, criterion=criterion)
    test_loss = test_metrics["loss"]
    test_targets = test_metrics["targets"]
    test_preds = test_metrics["preds"]
    test_probs = test_metrics["probs"]

    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_roc_auc = roc_auc_score(test_targets, test_probs)
    test_pr_auc = average_precision_score(test_targets, test_probs)

    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nFINAL SUBJECT-INDEPENDENT CNN + FFT + LSTM TEST RESULTS:")
    print(f"  Test Loss    : {test_loss:.4f}")
    print(f"  Precision    : {test_prec:.4f}")
    print(f"  Recall       : {test_rec:.4f}")
    print(f"  F1 Score     : {test_f1:.4f}")
    print(f"  ROC-AUC      : {test_roc_auc:.4f}")
    print(f"  PR-AUC       : {test_pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TP: {tp:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TN: {tn:,}")
    print(f"  Total Test Positives: {int(test_targets.sum()):,}")
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=["Non-seizure", "Seizure"], digits=4, zero_division=0))

    # Save model checkpoint
    model_save_path = MODEL_DIR / "cnn_fft_lstm_subjectwise_15subjects.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_f1": best_val_f1,
            "train_subjects": train_subjects,
            "validation_subjects": val_subjects,
            "test_subjects": test_subjects,
            "cnn_dim": CNN_FEATURE_DIM,
            "fft_dim": FFT_FEATURE_DIM,
            "fused_dim": FUSED_FEATURE_DIM,
            "hidden_size": LSTM_HIDDEN_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "seed": SEED,
            "history": history,
            "test_metrics": {
                "loss": float(test_loss),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "f1": float(test_f1),
                "roc_auc": float(test_roc_auc),
                "pr_auc": float(test_pr_auc),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "test_positive_count": int(test_targets.sum()),
            },
        },
        model_save_path,
    )
    print(f"Checkpoint saved to {model_save_path}")

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "cnn_fft_lstm_subjectwise_15subjects_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_f1": float(best_val_f1),
                "history": history,
                "test_metrics": {
                    "loss": float(test_loss),
                    "precision": float(test_prec),
                    "recall": float(test_rec),
                    "f1": float(test_f1),
                    "roc_auc": float(test_roc_auc),
                    "pr_auc": float(test_pr_auc),
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "test_positive_count": int(test_targets.sum()),
                },
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
