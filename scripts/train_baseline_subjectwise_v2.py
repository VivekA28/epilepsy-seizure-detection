"""
Subject-wise baseline CNN (v2) for 15-subject CHB-MIT experiment.

Protocol v2:
- Seed: 42
- Split: Fixed subject-level partitions from data/partitions/*.csv
- Sampler: WeightedRandomSampler (same as v1)
- Epochs: 15, Batch size: 32, LR: 1e-3, Adam, BCEWithLogitsLoss
- NUM_WORKERS=0, PIN_MEMORY=False
- Checkpoint selection: Best epoch by VALIDATION PR-AUC (continuous probabilities)
- Save predictions: results/baseline_cnn_val_predictions.npz, results/baseline_cnn_test_predictions.npz
- Threshold sweep: Validation only (0.01-0.99 step 0.01, 0.991-0.999 step 0.001)
- Test evaluation: Exactly once at frozen threshold
"""

import copy
import random
import sys
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

from train_baseline_subjectwise import SeizureCNN, SubjectWindowDataset, load_partitions, load_subject, verify_partition_alignment

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
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


def predict_dataset(model, loader, device):
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


def train_baseline():
    seed_everything(SEED)
    partitions, subject_sets = load_partitions()
    all_subjects = sorted(set().union(*subject_sets.values()))

    print("\nLoading processed subjects for Baseline CNN...")
    sources = {s: load_subject(s) for s in all_subjects}
    verify_partition_alignment(partitions, sources)

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = SubjectWindowDataset(sources, train_subjects)
    val_ds = SubjectWindowDataset(sources, val_subjects)
    test_ds = SubjectWindowDataset(sources, test_subjects)

    val_sub_names = np.array([val_subjects[i] for i in val_ds.subject_ids])
    test_sub_names = np.array([test_subjects[i] for i in test_ds.subject_ids])

    # Sampler
    y_train = train_ds.labels
    counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(weights[y_train.astype(np.int64)]).double()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = SeizureCNN(n_channels=23).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None
    epoch_logs = []

    print("\n" + "=" * 90)
    print("TRAINING BASELINE CNN (Selection criterion: VALIDATION PR-AUC)")
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

        avg_loss = total_loss / len(train_ds)
        _, val_probs, val_targets = predict_dataset(model, val_loader, device)

        val_pr_auc = average_precision_score(val_targets, val_probs)
        val_roc_auc = roc_auc_score(val_targets, val_probs)
        val_f1_05 = f1_score(val_targets, (val_probs >= 0.5).astype(int), zero_division=0)

        epoch_logs.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "val_f1_05": val_f1_05,
        })

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={avg_loss:.4f} | "
            f"val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f} | val F1@0.5={val_f1_05:.4f}"
            f"{' -> [BEST]' if is_best else ''}"
        )

    print(f"\nRestoring selected checkpoint from Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)

    ckpt_path = MODEL_DIR / "baseline_cnn_subjectwise_15subjects_v2.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "epoch_logs": epoch_logs,
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Validation evaluation & threshold sweep
    val_logits, val_probs, val_targets = predict_dataset(model, val_loader, device)
    val_save_path = RESULTS_DIR / "baseline_cnn_val_predictions.npz"
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
    best_th, best_val_f1, best_val_prec, best_val_rec, best_val_cm, df_sweep = sweep_thresholds(val_targets, val_probs)

    sweep_path = RESULTS_DIR / "baseline_cnn_val_threshold_sweep.csv"
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Saved threshold sweep to {sweep_path}")
    print(f"Validation PR-AUC: {val_pr:.4f} | ROC-AUC: {val_roc:.4f}")
    print(f"Chosen threshold: {best_th} (Val F1={best_val_f1:.4f}, Prec={best_val_prec:.4f}, Rec={best_val_rec:.4f})")

    # Final single test evaluation
    frozen_th = best_th
    test_logits, test_probs, test_targets = predict_dataset(model, test_loader, device)
    test_save_path = RESULTS_DIR / "baseline_cnn_test_predictions.npz"
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
        "model": "CNN baseline",
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
    return summary


if __name__ == "__main__":
    train_baseline()
