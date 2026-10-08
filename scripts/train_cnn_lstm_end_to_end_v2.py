"""
Subject-wise End-to-End Trainable CNN + LSTM temporal model (v2) for 15-subject CHB-MIT experiment.

Protocol v2:
- Seed: 42
- Split: Fixed subject-level partitions from data/partitions/*.csv
- Sequence length: 5 consecutive 4-second windows (T=5)
- Target: Label of the final window W(t)
- No cross-EDF / cross-subject / cross-split sequences
- Input: Raw normalized EEG windows (B, 5, 23, 512) directly from data/processed/{subject}_windows.npy
- Sampler: Sequence-level WeightedRandomSampler balancing seizure and non-seizure targets
- Model:
    * CNN: SeizureCNN feature extractor (warm-started from models/baseline_cnn_subjectwise_15subjects_v2.pt)
    * LSTM: Single-layer unidirectional LSTM (input_size=128, hidden_size=128), initialized randomly (seed 42)
    * Classifier: nn.Linear(128, 1), initialized randomly (seed 42)
- Training:
    * Joint end-to-end backpropagation into CNN + LSTM + Classifier
    * Physical batch size: 32
    * Gradient accumulation steps: 4 (effective batch size: 128)
    * Epochs: 15
    * Optimizer: Adam with differential LR:
        - CNN: 1e-4
        - LSTM: 1e-3
        - Classifier: 1e-3
    * Criterion: BCEWithLogitsLoss
    * NUM_WORKERS=0, PIN_MEMORY=False
- Checkpoint selection: Best epoch by VALIDATION PR-AUC (continuous probabilities)
- Save predictions:
    * results/cnn_lstm_e2e_val_predictions.npz
    * results/cnn_lstm_e2e_test_predictions.npz
- Threshold calibration: Validation only (0.01-0.99 step 0.01, 0.991-0.999 step 0.001)
    * results/cnn_lstm_e2e_val_threshold_sweep.csv
- Test evaluation: Evaluated exactly once at the frozen validation-optimal threshold
"""

import copy
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
PARTITION_DIR = Path("data/partitions")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PARTITIONS = {
    "train": PARTITION_DIR / "train_metadata.csv",
    "validation": PARTITION_DIR / "validation_metadata.csv",
    "test": PARTITION_DIR / "test_metadata.csv",
}

BASELINE_CHECKPOINT_PATH = MODEL_DIR / "baseline_cnn_subjectwise_15subjects_v2.pt"

SEED = 42
SEQUENCE_LENGTH = 5
CNN_FEATURE_DIM = 128
LSTM_HIDDEN_SIZE = 128
PHYSICAL_BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS  # 128
EPOCHS = 15

CNN_LR = 1e-4
LSTM_LR = 1e-3
CLASSIFIER_LR = 1e-3

NUM_WORKERS = 0
PIN_MEMORY = False


def seed_everything(seed=SEED):
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
            raise FileNotFoundError(f"Missing partition file: {path}")
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
    print("SUBJECT-WISE END-TO-END CNN + LSTM TEMPORAL MODEL (v2)")
    print("=" * 90)
    for split in ("train", "validation", "test"):
        df = partitions[split]
        subjects = sorted(subject_sets[split])
        print(
            f"{split.capitalize():11s}: {', '.join(subjects)} | "
            f"{len(df):,} windows | {int(df['label'].sum()):,} seizure"
        )

    return partitions, subject_sets


class RawEEGSequenceDataset(Dataset):
    """Chronological sequences of raw normalized EEG windows (T, 23, 512)."""

    def __init__(self, windows_arrays, labels_arrays, records, sequence_length):
        self.windows_arrays = windows_arrays
        self.labels_arrays = labels_arrays
        self.records = records
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        windows = self.windows_arrays[record.subject_index]
        labels = self.labels_arrays[record.subject_index]

        end = record.start + self.sequence_length
        x = np.array(windows[record.start:end], dtype=np.float32, copy=True)
        y = np.float32(labels[end - 1])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def build_raw_split_dataset(subjects, sequence_length=SEQUENCE_LENGTH):
    """Build sequence dataset directly reading raw windows.npy (zero cached CNN features)."""
    windows_arrays = []
    labels_arrays = []
    records = []

    for subj_idx, subject in enumerate(subjects):
        win_path = DATA_DIR / f"{subject}_windows.npy"
        label_path = DATA_DIR / f"{subject}_labels.npy"
        fid_path = DATA_DIR / f"{subject}_file_ids.npy"

        if not win_path.exists():
            raise FileNotFoundError(f"Missing processed windows: {win_path}")
        if not label_path.exists() or not fid_path.exists():
            raise FileNotFoundError(f"Missing processed labels/file_ids for {subject}")

        windows = np.load(win_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        file_ids = np.load(fid_path, mmap_mode="r")

        if len(windows) != len(labels) or len(labels) != len(file_ids):
            raise ValueError(
                f"Shape mismatch for {subject}: win={len(windows)}, "
                f"lbl={len(labels)}, fid={len(file_ids)}"
            )
        if windows.shape[1:] != (23, 512):
            raise ValueError(f"Expected (23, 512) windows, got {windows.shape[1:]} for {subject}")

        windows_arrays.append(windows)
        labels_arrays.append(labels)

        allowed_files = np.unique(file_ids)
        subj_records = build_sequence_records(file_ids, allowed_files, sequence_length, subj_idx)
        records.extend(subj_records)

    return RawEEGSequenceDataset(windows_arrays, labels_arrays, records, sequence_length)


def make_train_sampler(dataset):
    """Sequence-level WeightedRandomSampler balancing seizure and non-seizure targets."""
    target_labels = np.fromiter(
        (
            dataset.labels_arrays[r.subject_index][r.start + dataset.sequence_length - 1]
            for r in dataset.records
        ),
        dtype=np.int8,
        count=len(dataset.records),
    )
    class_counts = np.bincount(target_labels, minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise RuntimeError("Training sequences contain only one class.")
    class_weights = 1.0 / class_counts
    weights = torch.as_tensor(class_weights[target_labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), target_labels


class CNNFeatureExtractor(nn.Module):
    """Identical SeizureCNN feature extractor architecture producing 128-D GAP features."""
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
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, 23, 512)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.gap(x).squeeze(-1)  # (N, 128)


class EndToEndCNNLSTM(nn.Module):
    """
    Unified end-to-end architecture:
    (B, 5, 23, 512) -> reshape (B*5, 23, 512) -> CNN -> (B*5, 128)
    -> reshape (B, 5, 128) -> LSTM(128, 128) -> final timestep (B, 128)
    -> Linear(128, 1) -> (B, 1)
    """

    def __init__(self, n_channels=23, hidden_size=LSTM_HIDDEN_SIZE):
        super().__init__()
        self.cnn = CNNFeatureExtractor(n_channels=n_channels)
        self.lstm = nn.LSTM(
            input_size=CNNFeatureExtractor.FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        b, t, c, l = x.shape
        x_flat = x.view(b * t, c, l)
        feats_flat = self.cnn(x_flat)
        feats_seq = feats_flat.view(b, t, -1)
        lstm_out, _ = self.lstm(feats_seq)
        logits = self.classifier(lstm_out[:, -1, :])
        return logits


def load_warm_start_weights(model, checkpoint_path=BASELINE_CHECKPOINT_PATH):
    """Warm-start CNN weights from clean Protocol-v2 baseline CNN checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline CNN checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"\n[Warm-Start] Loading baseline checkpoint from: {checkpoint_path}")
    print(f"[Warm-Start] Checkpoint best epoch: {ckpt.get('best_epoch')}")
    print(f"[Warm-Start] Checkpoint val PR-AUC: {ckpt.get('best_val_pr_auc'):.4f}")

    if ckpt.get("best_epoch") != 8:
        raise ValueError(
            f"Expected Protocol-v2 baseline checkpoint Epoch 8, got {ckpt.get('best_epoch')}"
        )

    # Filter out classifier layers (fc1, fc2) and retain conv + bn layers
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("fc")}
    load_res = model.cnn.load_state_dict(sd, strict=True)
    print(f"[Warm-Start] CNN feature extractor weights loaded: {load_res}")


def predict_dataset(model, loader, device):
    """Batched evaluation under torch.no_grad()."""
    model.eval()
    all_logits, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=PIN_MEMORY)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits)

            all_logits.append(logits.cpu().numpy().astype(np.float32))
            all_probs.append(probs.cpu().numpy().astype(np.float32))
            all_targets.append(y.numpy().astype(np.int64))

    return (
        np.concatenate(all_logits),
        np.concatenate(all_probs),
        np.concatenate(all_targets),
    )


def sweep_thresholds(y_true, probabilities):
    grid1 = np.round(np.arange(0.01, 1.00, 0.01), 2)
    grid2 = np.round(np.arange(0.991, 1.000, 0.001), 3)
    thresholds = np.unique(np.concatenate([grid1, grid2]))

    sweep_records = []
    best_th = 0.5
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_cm = None

    for th in thresholds:
        preds = (probabilities >= th).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        sweep_records.append({
            "threshold": float(th),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            best_prec = float(p)
            best_rec = float(r)
            best_cm = cm

    df_sweep = pd.DataFrame(sweep_records)
    return best_th, best_f1, best_prec, best_rec, best_cm, df_sweep


def train_cnn_lstm_end_to_end():
    seed_everything(SEED)
    partitions, subject_sets = load_partitions()

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    print("\nBuilding raw sequence datasets (sequence_length=5)...")
    train_ds = build_raw_split_dataset(train_subjects, SEQUENCE_LENGTH)
    val_ds = build_raw_split_dataset(val_subjects, SEQUENCE_LENGTH)
    test_ds = build_raw_split_dataset(test_subjects, SEQUENCE_LENGTH)

    val_sub_names = np.array([val_subjects[r.subject_index] for r in val_ds.records])
    test_sub_names = np.array([test_subjects[r.subject_index] for r in test_ds.records])

    sampler, _ = make_train_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=PHYSICAL_BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PHYSICAL_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=PHYSICAL_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = EndToEndCNNLSTM(n_channels=23, hidden_size=LSTM_HIDDEN_SIZE)

    # Warm-start CNN from clean baseline
    load_warm_start_weights(model, BASELINE_CHECKPOINT_PATH)
    model.to(device)

    # Differential learning rate optimizer
    optimizer = torch.optim.Adam([
        {"params": model.cnn.parameters(), "lr": CNN_LR},
        {"params": model.lstm.parameters(), "lr": LSTM_LR},
        {"params": model.classifier.parameters(), "lr": CLASSIFIER_LR},
    ])
    criterion = nn.BCEWithLogitsLoss()

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None
    epoch_logs = []

    print("\n" + "=" * 90)
    print("TRAINING END-TO-END CNN + LSTM (Selection criterion: VALIDATION PR-AUC)")
    print(f"Physical Batch: {PHYSICAL_BATCH_SIZE} | Accumulation: {ACCUMULATION_STEPS} | Effective: {EFFECTIVE_BATCH_SIZE}")
    print(f"LR: CNN={CNN_LR}, LSTM={LSTM_LR}, Classifier={CLASSIFIER_LR}")
    print("=" * 90)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=PIN_MEMORY)
            y = y.to(device, non_blocking=PIN_MEMORY)

            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            # Gradient accumulation scaling
            scaled_loss = loss / ACCUMULATION_STEPS
            scaled_loss.backward()

            total_loss += loss.item() * x.size(0)

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = total_loss / len(train_ds)
        val_logits, val_probs, val_targets = predict_dataset(model, val_loader, device)

        val_loss = criterion(torch.from_numpy(val_logits), torch.from_numpy(val_targets).float()).item()
        val_pr_auc = average_precision_score(val_targets, val_probs)
        val_roc_auc = roc_auc_score(val_targets, val_probs)
        val_f1_05 = f1_score(val_targets, (val_probs >= 0.5).astype(int), zero_division=0)
        epoch_duration = time.time() - epoch_start_time

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "val_f1_05": val_f1_05,
            "epoch_duration_sec": epoch_duration,
        })

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{EPOCHS} [{epoch_duration:.1f}s] | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f} | val F1@0.5={val_f1_05:.4f}"
            f"{' -> [BEST]' if is_best else ''}"
        )

    print(f"\nRestoring selected checkpoint from Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)

    ckpt_path = MODEL_DIR / "cnn_lstm_end_to_end_v2.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "epoch_logs": epoch_logs,
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
        "warm_start_checkpoint": str(BASELINE_CHECKPOINT_PATH),
        "warm_start_epoch": 8,
        "cnn_lr": CNN_LR,
        "lstm_lr": LSTM_LR,
        "classifier_lr": CLASSIFIER_LR,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Validation evaluation & threshold sweep
    val_logits, val_probs, val_targets = predict_dataset(model, val_loader, device)
    val_save_path = RESULTS_DIR / "cnn_lstm_e2e_val_predictions.npz"
    np.savez_compressed(
        val_save_path,
        y_true=val_targets,
        logits=val_logits,
        probabilities=val_probs,
        subject_ids=val_sub_names,
    )
    print(f"Saved validation predictions to {val_save_path}")

    val_roc = roc_auc_score(val_targets, val_probs)
    val_pr = average_precision_score(val_targets, val_probs)
    best_th, best_val_f1, best_val_prec, best_val_rec, best_val_cm, df_sweep = sweep_thresholds(
        val_targets, val_probs
    )

    sweep_path = RESULTS_DIR / "cnn_lstm_e2e_val_threshold_sweep.csv"
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Saved threshold sweep to {sweep_path}")
    print(f"Validation PR-AUC: {val_pr:.4f} | ROC-AUC: {val_roc:.4f}")
    print(f"Chosen threshold: {best_th} (Val F1={best_val_f1:.4f}, Prec={best_val_prec:.4f}, Rec={best_val_rec:.4f})")

    # Final single test evaluation
    frozen_th = best_th
    test_logits, test_probs, test_targets = predict_dataset(model, test_loader, device)
    test_save_path = RESULTS_DIR / "cnn_lstm_e2e_test_predictions.npz"
    np.savez_compressed(
        test_save_path,
        y_true=test_targets,
        logits=test_logits,
        probabilities=test_probs,
        subject_ids=test_sub_names,
    )
    print(f"Saved test predictions to {test_save_path}")

    test_preds = (test_probs >= frozen_th).astype(int)
    test_roc = roc_auc_score(test_targets, test_probs)
    test_pr = average_precision_score(test_targets, test_probs)
    test_prec = precision_score(test_targets, test_preds, zero_division=0)
    test_rec = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_cm = confusion_matrix(test_targets, test_preds, labels=[0, 1])
    tn, fp, fn, tp = test_cm.ravel()

    per_sub_metrics = {}
    for s in test_subjects:
        mask = (test_sub_names == s)
        sub_y = test_targets[mask]
        sub_p = test_probs[mask]
        sub_pred = test_preds[mask]
        sub_cm = confusion_matrix(sub_y, sub_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = sub_cm.ravel()

        per_sub_metrics[s] = {
            "samples": len(sub_y),
            "positives": int(sub_y.sum()),
            "roc_auc": roc_auc_score(sub_y, sub_p) if len(np.unique(sub_y)) > 1 else float("nan"),
            "pr_auc": average_precision_score(sub_y, sub_p) if len(np.unique(sub_y)) > 1 else float("nan"),
            "precision": precision_score(sub_y, sub_pred, zero_division=0),
            "recall": recall_score(sub_y, sub_pred, zero_division=0),
            "f1": f1_score(sub_y, sub_pred, zero_division=0),
            "tp": int(s_tp),
            "fp": int(s_fp),
            "fn": int(s_fn),
            "tn": int(s_tn),
        }

    summary = {
        "model": "CNN + LSTM (End-to-End)",
        "best_epoch": best_epoch,
        "val_pr_auc": val_pr,
        "val_roc_auc": val_roc,
        "val_f1_tuned": best_val_f1,
        "chosen_threshold": frozen_th,
        "test_pr_auc": test_pr,
        "test_roc_auc": test_roc,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "fp": int(fp),
        "tp": int(tp),
        "fn": int(fn),
        "tn": int(tn),
        "per_subject": per_sub_metrics,
    }

    import json
    with open(RESULTS_DIR / "cnn_lstm_e2e_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print("END-TO-END CNN + LSTM TRAINING COMPLETE")
    print("=" * 90)
    print(f"Best Epoch: {best_epoch} | Chosen Threshold: {frozen_th}")
    print(f"Val PR-AUC: {val_pr:.4f} | Val ROC-AUC: {val_roc:.4f} | Val F1 (tuned): {best_val_f1:.4f}")
    print(f"Test PR-AUC: {test_pr:.4f} | Test ROC-AUC: {test_roc:.4f}")
    print(f"Test F1: {test_f1:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print("\nPer-subject Test Breakdown:")
    for s, m in per_sub_metrics.items():
        print(
            f"  {s}: ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | "
            f"F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | "
            f"TP={m['tp']} | FP={m['fp']} | FN={m['fn']}"
        )
    return summary


if __name__ == "__main__":
    train_cnn_lstm_end_to_end()
