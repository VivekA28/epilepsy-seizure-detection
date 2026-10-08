"""
Subject-wise baseline CNN for the 15-subject CHB-MIT experiment.

Reads the fixed subject-level partitions from data/partitions/*.csv.
It never performs a new random window/EDF split. Validation F1 selects
the checkpoint; the test set is evaluated once at the end.
"""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data/processed")
PARTITION_DIR = Path("data/partitions")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PARTITIONS = {
    "train": PARTITION_DIR / "train_metadata.csv",
    "validation": PARTITION_DIR / "validation_metadata.csv",
    "test": PARTITION_DIR / "test_metadata.csv",
}

SEED = 42
BATCH_SIZE = 32
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

    if subject_sets["train"] & subject_sets["validation"]:
        raise RuntimeError("Subject leakage: train/validation overlap.")
    if subject_sets["train"] & subject_sets["test"]:
        raise RuntimeError("Subject leakage: train/test overlap.")
    if subject_sets["validation"] & subject_sets["test"]:
        raise RuntimeError("Subject leakage: validation/test overlap.")

    print("=" * 90)
    print("SUBJECT-WISE BASELINE CNN")
    print("=" * 90)
    for split in ("train", "validation", "test"):
        df = partitions[split]
        subjects = sorted(subject_sets[split])
        print(
            f"{split.capitalize():11s}: {', '.join(subjects)} | "
            f"{len(df):,} windows | {int(df['label'].sum()):,} seizure"
        )

    return partitions, subject_sets


def load_subject(subject):
    """Load signal as read-only mmap; labels and metadata stay in RAM."""
    wp = DATA_DIR / f"{subject}_windows.npy"
    lp = DATA_DIR / f"{subject}_labels.npy"
    mp = DATA_DIR / f"{subject}_metadata.csv"

    for p in (wp, lp, mp):
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")

    windows = np.load(wp, mmap_mode="r")
    labels = np.load(lp)
    meta = pd.read_csv(mp)

    if windows.shape[1:] != (23, 512):
        raise ValueError(f"{subject}: unexpected windows shape {windows.shape}")
    if len(windows) != len(labels) or len(windows) != len(meta):
        raise ValueError(f"{subject}: windows/labels/metadata length mismatch")
    if set(meta["subject_id"].astype(str).unique()) != {subject}:
        raise ValueError(f"{subject}: metadata contains another subject")
    if not np.array_equal(labels.astype(meta["label"].dtype), meta["label"].to_numpy()):
        raise ValueError(f"{subject}: labels.npy != metadata.csv labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{subject}: labels are not binary")

    return {"windows": windows, "labels": labels.astype(np.float32), "meta": meta}


def verify_partition_alignment(partitions, sources):
    """Each subject is entirely contained in one partition; verify row order."""
    print("\nChecking partition <-> processed-data alignment...")

    for split, df in partitions.items():
        for subject in sorted(df["subject_id"].astype(str).unique()):
            part = df[df["subject_id"].astype(str) == subject].reset_index(drop=True)
            meta = sources[subject]["meta"].reset_index(drop=True)

            if len(part) != len(meta):
                raise RuntimeError(
                    f"{subject}: partition has {len(part):,} rows, "
                    f"processed metadata has {len(meta):,}"
                )

            for col in ("edf_id", "window_index", "start_sec", "end_sec", "label"):
                a = part[col].to_numpy()
                b = meta[col].to_numpy()
                if col == "edf_id":
                    a, b = a.astype(str), b.astype(str)
                if not np.array_equal(a, b):
                    raise RuntimeError(
                        f"{subject}: partition and processed metadata "
                        f"differ in '{col}'"
                    )

    print("✅ Partition metadata matches processed subject metadata.")


class SubjectWindowDataset(Dataset):
    """Lazy windows from complete subjects in one partition."""

    def __init__(self, sources, subjects):
        self.sources = sources
        self.subjects = list(subjects)

        ids = []
        rows = []
        labels = []

        for sid, subject in enumerate(self.subjects):
            n = len(sources[subject]["labels"])
            ids.append(np.full(n, sid, dtype=np.int16))
            rows.append(np.arange(n, dtype=np.int32))
            labels.append(sources[subject]["labels"])

        self.subject_ids = np.concatenate(ids)
        self.rows = np.concatenate(rows)
        self.labels = np.concatenate(labels).astype(np.float32, copy=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        sid = int(self.subject_ids[i])
        row = int(self.rows[i])
        subject = self.subjects[sid]

        x = torch.from_numpy(
            self.sources[subject]["windows"][row].copy()
        )
        y = self.labels[i]
        return x, torch.tensor(y, dtype=torch.float32)


class SeizureCNN(nn.Module):
    FEATURE_DIM = 128

    def __init__(self, n_channels=23):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, 7, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, 5, padding="same")
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, 3, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def extract_features(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.gap(x).squeeze(-1)

    def forward(self, x):
        x = self.relu(self.fc1(self.extract_features(x)))
        x = self.dropout(x)
        return self.fc2(x)


def predict(model, loader, device):
    model.eval()
    preds, targets, probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=PIN_MEMORY)
            prob = torch.sigmoid(model(x).squeeze(-1))
            p = (prob > 0.5).int()
            probs.append(prob.cpu().numpy().astype(np.float32))
            preds.append(p.cpu().numpy())
            targets.append(y.numpy().astype(np.int64))

    return np.concatenate(preds), np.concatenate(targets), np.concatenate(probs)


def main():
    seed_everything(SEED)
    partitions, subject_sets = load_partitions()

    all_subjects = sorted(set().union(*subject_sets.values()))
    sources = {}

    print("\nLoading processed subjects...")
    for subject in all_subjects:
        sources[subject] = load_subject(subject)
        labels = sources[subject]["labels"]
        print(
            f"  {subject}: {len(labels):,} windows | "
            f"{int(labels.sum()):,} seizure"
        )

    verify_partition_alignment(partitions, sources)

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = SubjectWindowDataset(sources, train_subjects)
    val_ds = SubjectWindowDataset(sources, val_subjects)
    test_ds = SubjectWindowDataset(sources, test_subjects)

    y_train = train_ds.labels
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        raise RuntimeError("Training partition contains only one class.")

    counts = np.bincount(
        y_train.astype(np.int64), minlength=2
    ).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(
        weights[y_train.astype(np.int64)]
    ).double()

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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

    print("\nModel-facing partitions:")
    print(
        f"  Train      : {len(train_ds):,} windows "
        f"({int(train_ds.labels.sum()):,} seizure)"
    )
    print(
        f"  Validation : {len(val_ds):,} windows "
        f"({int(val_ds.labels.sum()):,} seizure)"
    )
    print(
        f"  Test       : {len(test_ds):,} windows "
        f"({int(test_ds.labels.sum()):,} seizure)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SeizureCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = None

    print("\n" + "=" * 90)
    print("TRAINING")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=PIN_MEMORY)
            y = y.to(device, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        val_preds, val_targets, _ = predict(model, val_loader, device)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        avg_loss = total_loss / len(train_ds)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | val F1={val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(
                f"  -> new best validation checkpoint "
                f"(F1={best_val_f1:.3f})"
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint produced.")

    model.load_state_dict(best_state)
    print(
        f"\nRestored epoch {best_epoch} "
        f"(validation F1={best_val_f1:.3f})"
    )

    print("\nEvaluating selected checkpoint on VALIDATION set...")
    val_preds, val_targets, val_probs = predict(model, val_loader, device)
    val_f1_selected = f1_score(val_targets, val_preds, zero_division=0)
    val_roc_auc = roc_auc_score(val_targets, val_probs)
    val_pr_auc = average_precision_score(val_targets, val_probs)

    print("\n" + "=" * 90)
    print("SELECTED CHECKPOINT VALIDATION METRICS")
    print("=" * 90)
    print(f"Validation samples      : {len(val_targets):,}")
    print(f"Validation positives    : {int(val_targets.sum()):,}")
    print(f"Validation F1 (@ 0.5)   : {val_f1_selected:.4f}")
    print(f"Validation ROC-AUC      : {val_roc_auc:.4f}")
    print(f"Validation PR-AUC       : {val_pr_auc:.4f}")

    val_probs_path = RESULTS_DIR / "baseline_cnn_subjectwise_validation_probs.npz"
    np.savez_compressed(val_probs_path, y_true=val_targets, y_prob=val_probs)
    print(f"Saved validation continuous probabilities to: {val_probs_path}")

    print("\nFinal TEST evaluation (running test inference forward exactly once)...")
    test_preds, test_targets, test_probs = predict(model, test_loader, device)

    print("\n" + "=" * 90)
    print("FINAL SUBJECT-INDEPENDENT TEST RESULTS")
    print("=" * 90)
    print(
        classification_report(
            test_targets,
            test_preds,
            target_names=["Non-seizure", "Seizure"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(test_targets, test_preds))

    test_probs_path = RESULTS_DIR / "baseline_cnn_subjectwise_test_probs.npz"
    np.savez_compressed(test_probs_path, y_true=test_targets, y_prob=test_probs)
    print(f"Saved test continuous probabilities to: {test_probs_path}")

    path = MODEL_DIR / "baseline_cnn_subjectwise_15subjects.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_f1": best_val_f1,
            "train_subjects": train_subjects,
            "validation_subjects": val_subjects,
            "test_subjects": test_subjects,
            "input_shape": [23, 512],
            "seed": SEED,
        },
        path,
    )

    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    main()
