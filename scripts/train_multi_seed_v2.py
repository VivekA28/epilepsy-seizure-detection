"""
Multi-Seed Controlled Benchmark Runner (Protocol v2)
Seeds: 42, 43, 44

Execution Plan:
1. Seed 43 Chain:
   - Baseline CNN (seed 43)
   - Cache CNN features into data/processed/cnn_features_seed43/
   - CNN + LSTM (seed 43, using cnn_features_seed43)
   - CNN + FFT + LSTM (seed 43, using cnn_features_seed43)
2. Seed 44 Chain:
   - Baseline CNN (seed 44)
   - Cache CNN features into data/processed/cnn_features_seed44/
   - CNN + LSTM (seed 44, using cnn_features_seed44)
   - CNN + FFT + LSTM (seed 44, using cnn_features_seed44)
3. Seed 42 Chain:
   - Baseline CNN (seed 42, reuse checkpoint & sweep extended grid)
   - Cache CNN features into data/processed/cnn_features_seed42/ (from baseline_cnn_subjectwise_15subjects_v2.pt)
   - CNN + LSTM (seed 42, using cnn_features_seed42)
   - CNN + FFT + LSTM (seed 42, using cnn_features_seed42)
4. CNN + FFT (end-to-end raw EEG + FFT) LAST:
   - Seed 42: evaluated on extended grid
   - Seed 43: full train & eval
   - Seed 44: full train & eval
5. Summary:
   - Prints overall mean +/- std table
   - Prints per-subject (chb02, chb05, chb13) mean +/- std table
   - Saves results to results/multi_seed_v2_summary.json & .csv

Features:
- Resumable: skips any step whose output files already exist.
- Extended threshold grid: 0.01-0.99 (step 0.01), 0.991-0.999 (step 0.001), 0.9991-0.9999 (step 0.0001).
- Strict validation-only threshold tuning; test evaluated strictly once at frozen threshold.
- NUM_WORKERS=0, PIN_MEMORY=False.
"""

import argparse
import copy
import gc
import json
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

from sequence_dataset import EEGSequenceDataset, build_sequence_records
from train_baseline_subjectwise import (
    SeizureCNN,
    SubjectWindowDataset,
    load_partitions,
    load_subject as load_subject_base,
    verify_partition_alignment,
)
from train_cnn_fft_subjectwise import (
    CNNFFT,
    SubjectWindowFFTDataset,
    load_subject as load_subject_fft,
)
from train_cnn_lstm_subjectwise import CNNLSTM
from train_cnn_fft_lstm_subjectwise import CNNFFTLSTM, DualFeatureSequenceDataset

DATA_DIR = REPO_ROOT / "data" / "processed"
PARTITION_DIR = REPO_ROOT / "data" / "partitions"
MODEL_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUBJECTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
]
VAL_SUBJECTS = ["chb07", "chb08"]
TEST_SUBJECTS = ["chb02", "chb05", "chb13"]

CNN_FEATURE_DIM = 128
FFT_FEATURE_DIM = 115
LSTM_HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 5

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


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sweep_thresholds(y_true, probabilities):
    grid1 = np.round(np.arange(0.01, 1.00, 0.01), 2)
    grid2 = np.round(np.arange(0.991, 1.000, 0.001), 3)
    grid3 = np.round(np.arange(0.9991, 1.0000, 0.0001), 4)
    thresholds = np.unique(np.concatenate([grid1, grid2, grid3]))

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


def evaluate_predictions_at_threshold(y_true, probabilities, sub_names, threshold, test_subjects=TEST_SUBJECTS):
    preds = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    per_sub = {}
    for s in test_subjects:
        mask = (sub_names == s)
        s_y = y_true[mask]
        s_p = probabilities[mask]
        s_pred = preds[mask]
        s_cm = confusion_matrix(s_y, s_pred, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = s_cm.ravel()

        per_sub[s] = {
            "samples": int(len(s_y)),
            "positives": int(s_y.sum()),
            "pr_auc": float(average_precision_score(s_y, s_p)) if len(np.unique(s_y)) > 1 else float("nan"),
            "roc_auc": float(roc_auc_score(s_y, s_p)) if len(np.unique(s_y)) > 1 else float("nan"),
            "precision": float(precision_score(s_y, s_pred, zero_division=0)),
            "recall": float(recall_score(s_y, s_pred, zero_division=0)),
            "f1": float(f1_score(s_y, s_pred, zero_division=0)),
            "tp": int(s_tp),
            "fp": int(s_fp),
            "fn": int(s_fn),
            "tn": int(s_tn),
        }

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "per_subject": per_sub,
    }


# ==============================================================================
# 1. BASELINE CNN TRAINING & EVALUATION
# ==============================================================================
def predict_cnn_loader(model, loader, device):
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
    return np.concatenate(all_logits), np.concatenate(all_probs), np.concatenate(all_targets)


def run_baseline_cnn_seed(seed, epochs=15, batch_size=32, lr=1e-3):
    model_name = "baseline_cnn"
    ckpt_path = MODEL_DIR / f"baseline_cnn_subjectwise_15subjects_v2_seed{seed}.pt"
    if seed == 42 and not ckpt_path.exists():
        fallback_42 = MODEL_DIR / "baseline_cnn_subjectwise_15subjects_v2.pt"
        if fallback_42.exists():
            ckpt_path = fallback_42

    val_save_path = RESULTS_DIR / f"baseline_cnn_seed{seed}_val_predictions.npz"
    test_save_path = RESULTS_DIR / f"baseline_cnn_seed{seed}_test_predictions.npz"
    sweep_path = RESULTS_DIR / f"baseline_cnn_seed{seed}_val_threshold_sweep.csv"

    # Resumability check
    if test_save_path.exists() and sweep_path.exists():
        print(f"[{model_name.upper()} | SEED {seed}] Output files already exist. Loading saved predictions...")
        val_data = np.load(val_save_path)
        test_data = np.load(test_save_path)
        df_sweep = pd.read_csv(sweep_path)
        best_th = float(df_sweep.loc[df_sweep["f1"].idxmax()]["threshold"])
        test_metrics = evaluate_predictions_at_threshold(
            test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th
        )
        return {
            "model": "CNN baseline",
            "seed": seed,
            "chosen_threshold": best_th,
            "ckpt_path": str(ckpt_path),
            **test_metrics,
        }

    seed_everything(seed)
    partitions, subject_sets = load_partitions()
    all_subjects = sorted(set().union(*subject_sets.values()))
    sources = {s: load_subject_base(s) for s in all_subjects}
    verify_partition_alignment(partitions, sources)

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = SubjectWindowDataset(sources, train_subjects)
    val_ds = SubjectWindowDataset(sources, val_subjects)
    test_ds = SubjectWindowDataset(sources, test_subjects)

    val_sub_names = np.array([val_subjects[i] for i in val_ds.subject_ids])
    test_sub_names = np.array([test_subjects[i] for i in test_ds.subject_ids])

    y_train = train_ds.labels
    counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(weights[y_train.astype(np.int64)]).double()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BASELINE CNN | SEED {seed}] Training on {device} (Epochs={epochs}, LR={lr})")

    model = SeizureCNN(n_channels=23).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None
    epoch_logs = []

    for epoch in range(1, epochs + 1):
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
        _, val_probs, val_targets = predict_cnn_loader(model, val_loader, device)
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

        print(f"  Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f}{' -> [BEST]' if is_best else ''}")

    print(f"Restoring best checkpoint: Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)
    target_ckpt = MODEL_DIR / f"baseline_cnn_subjectwise_15subjects_v2_seed{seed}.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "seed": seed,
    }, target_ckpt)

    # Validation predictions & extended sweep
    val_logits, val_probs, val_targets = predict_cnn_loader(model, val_loader, device)
    np.savez_compressed(val_save_path, y_true=val_targets, logits=val_logits, probabilities=val_probs, subject_ids=val_sub_names)
    best_th, best_val_f1, _, _, _, df_sweep = sweep_thresholds(val_targets, val_probs)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Chosen validation frozen threshold: {best_th:.4f} (Val F1={best_val_f1:.4f})")

    # Single test evaluation
    test_logits, test_probs, test_targets = predict_cnn_loader(model, test_loader, device)
    np.savez_compressed(test_save_path, y_true=test_targets, logits=test_logits, probabilities=test_probs, subject_ids=test_sub_names)

    test_metrics = evaluate_predictions_at_threshold(test_targets, test_probs, test_sub_names, best_th)
    print(f"Test Evaluation: PR-AUC={test_metrics['pr_auc']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f} | F1={test_metrics['f1']:.4f}")

    return {
        "model": "CNN baseline",
        "seed": seed,
        "chosen_threshold": best_th,
        "ckpt_path": str(target_ckpt),
        **test_metrics,
    }


# ==============================================================================
# 2. SEED-SPECIFIC FEATURE CACHING
# ==============================================================================
class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return torch.from_numpy(self.windows[index].copy())


def cache_features_for_seed(seed, ckpt_path):
    cache_dir = DATA_DIR / f"cnn_features_seed{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if all 15 already cached
    all_exist = all((cache_dir / f"{s}_cnn_features.npy").exists() for s in ALL_SUBJECTS)
    if all_exist:
        print(f"\n[CACHE | SEED {seed}] All 15 subject features already cached in {cache_dir.name}. Skipping caching.")
        return cache_dir

    print(f"\n[CACHE | SEED {seed}] Caching 128-D CNN features from {Path(ckpt_path).name} -> {cache_dir.name}/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeizureCNN(n_channels=23).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    t_start = time.time()
    for subject in ALL_SUBJECTS:
        out_p = cache_dir / f"{subject}_cnn_features.npy"
        if out_p.exists():
            continue
        wp = DATA_DIR / f"{subject}_windows.npy"
        windows = np.load(wp, mmap_mode="r")
        n_windows = len(windows)

        memmap = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.float32, shape=(n_windows, 128))
        loader = DataLoader(WindowDataset(windows), batch_size=256, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        offset = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                feat = model.extract_features(batch).cpu().numpy().astype(np.float32)
                end = offset + len(feat)
                memmap[offset:end] = feat
                offset = end
        memmap.flush()
        print(f"  Cached {subject}: {n_windows:,} windows -> {out_p.name}")

    print(f"[CACHE | SEED {seed}] Finished in {(time.time() - t_start):.1f}s.")
    return cache_dir


# ==============================================================================
# 3. CNN + LSTM TRAINING & EVALUATION
# ==============================================================================
def build_lstm_dataset(subjects, sequence_length, cnn_cache_dir):
    feature_arrays = []
    labels_arrays = []
    records = []

    for subj_idx, subject in enumerate(subjects):
        feat_path = cnn_cache_dir / f"{subject}_cnn_features.npy"
        label_path = DATA_DIR / f"{subject}_labels.npy"
        fid_path = DATA_DIR / f"{subject}_file_ids.npy"

        features = np.load(feat_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        file_ids = np.load(fid_path, mmap_mode="r")

        feature_arrays.append(features)
        labels_arrays.append(labels)
        allowed_files = np.unique(file_ids)
        subj_records = build_sequence_records(file_ids, allowed_files, sequence_length, subj_idx)
        records.extend(subj_records)

    return EEGSequenceDataset(feature_arrays, labels_arrays, records, sequence_length)


def make_lstm_sampler(dataset):
    target_labels = np.fromiter(
        (dataset.labels_arrays[r.subject_index][r.start + dataset.sequence_length - 1] for r in dataset.records),
        dtype=np.int64,
        count=len(dataset.records),
    )
    counts = np.bincount(target_labels, minlength=2).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(weights[target_labels]).double()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def predict_lstm_loader(model, loader, device):
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
    return np.concatenate(all_logits), np.concatenate(all_probs), np.concatenate(all_targets)


def run_cnn_lstm_seed(seed, cnn_cache_dir, epochs=15, batch_size=128, lr=1e-3):
    model_name = "cnn_lstm"
    val_save_path = RESULTS_DIR / f"cnn_lstm_seed{seed}_val_predictions.npz"
    test_save_path = RESULTS_DIR / f"cnn_lstm_seed{seed}_test_predictions.npz"
    sweep_path = RESULTS_DIR / f"cnn_lstm_seed{seed}_val_threshold_sweep.csv"
    ckpt_path = MODEL_DIR / f"cnn_lstm_subjectwise_15subjects_v2_seed{seed}.pt"

    # Resumability check
    if test_save_path.exists() and sweep_path.exists():
        print(f"[{model_name.upper()} | SEED {seed}] Output files already exist. Loading saved predictions...")
        val_data = np.load(val_save_path)
        test_data = np.load(test_save_path)
        df_sweep = pd.read_csv(sweep_path)
        best_th = float(df_sweep.loc[df_sweep["f1"].idxmax()]["threshold"])
        test_metrics = evaluate_predictions_at_threshold(
            test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th
        )
        return {
            "model": "CNN + LSTM",
            "seed": seed,
            "chosen_threshold": best_th,
            "ckpt_path": str(ckpt_path),
            **test_metrics,
        }

    seed_everything(seed)
    partitions, subject_sets = load_partitions()
    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = build_lstm_dataset(train_subjects, SEQUENCE_LENGTH, cnn_cache_dir)
    val_ds = build_lstm_dataset(val_subjects, SEQUENCE_LENGTH, cnn_cache_dir)
    test_ds = build_lstm_dataset(test_subjects, SEQUENCE_LENGTH, cnn_cache_dir)

    val_sub_names = np.array([val_subjects[r.subject_index] for r in val_ds.records])
    test_sub_names = np.array([test_subjects[r.subject_index] for r in test_ds.records])

    sampler = make_lstm_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CNN + LSTM | SEED {seed}] Training on {device} (Epochs={epochs}, LR={lr})")

    model = CNNLSTM(input_dim=CNN_FEATURE_DIM, hidden_size=LSTM_HIDDEN_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
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
        _, val_probs, val_targets = predict_lstm_loader(model, val_loader, device)
        val_pr_auc = average_precision_score(val_targets, val_probs)
        val_roc_auc = roc_auc_score(val_targets, val_probs)

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f}{' -> [BEST]' if is_best else ''}")

    print(f"Restoring best checkpoint: Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "seed": seed,
    }, ckpt_path)

    val_logits, val_probs, val_targets = predict_lstm_loader(model, val_loader, device)
    np.savez_compressed(val_save_path, y_true=val_targets, logits=val_logits, probabilities=val_probs, subject_ids=val_sub_names)
    best_th, best_val_f1, _, _, _, df_sweep = sweep_thresholds(val_targets, val_probs)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Chosen validation frozen threshold: {best_th:.4f} (Val F1={best_val_f1:.4f})")

    test_logits, test_probs, test_targets = predict_lstm_loader(model, test_loader, device)
    np.savez_compressed(test_save_path, y_true=test_targets, logits=test_logits, probabilities=test_probs, subject_ids=test_sub_names)

    test_metrics = evaluate_predictions_at_threshold(test_targets, test_probs, test_sub_names, best_th)
    print(f"Test Evaluation: PR-AUC={test_metrics['pr_auc']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f} | F1={test_metrics['f1']:.4f}")

    return {
        "model": "CNN + LSTM",
        "seed": seed,
        "chosen_threshold": best_th,
        "ckpt_path": str(ckpt_path),
        **test_metrics,
    }


# ==============================================================================
# 4. CNN + FFT + LSTM TRAINING & EVALUATION
# ==============================================================================
def build_fft_lstm_dataset(subjects, sequence_length, cnn_cache_dir):
    cnn_arrays = []
    fft_arrays = []
    labels_arrays = []
    records = []

    for subj_idx, subject in enumerate(subjects):
        cnn_path = cnn_cache_dir / f"{subject}_cnn_features.npy"
        fft_path = DATA_DIR / f"{subject}_fft_features.npy"
        label_path = DATA_DIR / f"{subject}_labels.npy"
        fid_path = DATA_DIR / f"{subject}_file_ids.npy"

        cnn = np.load(cnn_path, mmap_mode="r")
        fft = np.load(fft_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        file_ids = np.load(fid_path, mmap_mode="r")

        cnn_arrays.append(cnn)
        fft_arrays.append(fft)
        labels_arrays.append(labels)
        allowed_files = np.unique(file_ids)
        subj_records = build_sequence_records(file_ids, allowed_files, sequence_length, subj_idx)
        records.extend(subj_records)

    return DualFeatureSequenceDataset(cnn_arrays, fft_arrays, labels_arrays, records, sequence_length)


def make_fft_lstm_sampler(dataset):
    target_labels = np.fromiter(
        (dataset.labels_arrays[r.subject_index][r.start + dataset.sequence_length - 1] for r in dataset.records),
        dtype=np.int64,
        count=len(dataset.records),
    )
    counts = np.bincount(target_labels, minlength=2).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(weights[target_labels]).double()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def predict_fft_lstm_loader(model, loader, device):
    model.eval()
    all_logits, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x_cnn, x_fft, y in loader:
            x_cnn = x_cnn.to(device, non_blocking=PIN_MEMORY)
            x_fft = x_fft.to(device, non_blocking=PIN_MEMORY)
            logits = model(x_cnn, x_fft).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_logits.append(logits.cpu().numpy().astype(np.float32))
            all_probs.append(probs.cpu().numpy().astype(np.float32))
            all_targets.append(y.numpy().astype(np.int64))
    return np.concatenate(all_logits), np.concatenate(all_probs), np.concatenate(all_targets)


def run_cnn_fft_lstm_seed(seed, cnn_cache_dir, epochs=15, batch_size=128, lr=1e-3):
    model_name = "cnn_fft_lstm"
    val_save_path = RESULTS_DIR / f"cnn_fft_lstm_seed{seed}_val_predictions.npz"
    test_save_path = RESULTS_DIR / f"cnn_fft_lstm_seed{seed}_test_predictions.npz"
    sweep_path = RESULTS_DIR / f"cnn_fft_lstm_seed{seed}_val_threshold_sweep.csv"
    ckpt_path = MODEL_DIR / f"cnn_fft_lstm_subjectwise_15subjects_v2_seed{seed}.pt"

    # Resumability check
    if test_save_path.exists() and sweep_path.exists():
        print(f"[{model_name.upper()} | SEED {seed}] Output files already exist. Loading saved predictions...")
        val_data = np.load(val_save_path)
        test_data = np.load(test_save_path)
        df_sweep = pd.read_csv(sweep_path)
        best_th = float(df_sweep.loc[df_sweep["f1"].idxmax()]["threshold"])
        test_metrics = evaluate_predictions_at_threshold(
            test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th
        )
        return {
            "model": "CNN + FFT + LSTM",
            "seed": seed,
            "chosen_threshold": best_th,
            "ckpt_path": str(ckpt_path),
            **test_metrics,
        }

    seed_everything(seed)
    partitions, subject_sets = load_partitions()
    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = build_fft_lstm_dataset(train_subjects, SEQUENCE_LENGTH, cnn_cache_dir)
    val_ds = build_fft_lstm_dataset(val_subjects, SEQUENCE_LENGTH, cnn_cache_dir)
    test_ds = build_fft_lstm_dataset(test_subjects, SEQUENCE_LENGTH, cnn_cache_dir)

    val_sub_names = np.array([val_subjects[r.subject_index] for r in val_ds.records])
    test_sub_names = np.array([test_subjects[r.subject_index] for r in test_ds.records])

    sampler = make_fft_lstm_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CNN + FFT + LSTM | SEED {seed}] Training on {device} (Epochs={epochs}, LR={lr})")

    model = CNNFFTLSTM(cnn_dim=CNN_FEATURE_DIM, fft_dim=FFT_FEATURE_DIM, hidden_size=LSTM_HIDDEN_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_cnn, x_fft, y in train_loader:
            x_cnn = x_cnn.to(device, non_blocking=PIN_MEMORY)
            x_fft = x_fft.to(device, non_blocking=PIN_MEMORY)
            y = y.to(device, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_cnn, x_fft).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_cnn.size(0)

        avg_loss = total_loss / len(train_ds)
        _, val_probs, val_targets = predict_fft_lstm_loader(model, val_loader, device)
        val_pr_auc = average_precision_score(val_targets, val_probs)
        val_roc_auc = roc_auc_score(val_targets, val_probs)

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f}{' -> [BEST]' if is_best else ''}")

    print(f"Restoring best checkpoint: Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "seed": seed,
    }, ckpt_path)

    val_logits, val_probs, val_targets = predict_fft_lstm_loader(model, val_loader, device)
    np.savez_compressed(val_save_path, y_true=val_targets, logits=val_logits, probabilities=val_probs, subject_ids=val_sub_names)
    best_th, best_val_f1, _, _, _, df_sweep = sweep_thresholds(val_targets, val_probs)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Chosen validation frozen threshold: {best_th:.4f} (Val F1={best_val_f1:.4f})")

    test_logits, test_probs, test_targets = predict_fft_lstm_loader(model, test_loader, device)
    np.savez_compressed(test_save_path, y_true=test_targets, logits=test_logits, probabilities=test_probs, subject_ids=test_sub_names)

    test_metrics = evaluate_predictions_at_threshold(test_targets, test_probs, test_sub_names, best_th)
    print(f"Test Evaluation: PR-AUC={test_metrics['pr_auc']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f} | F1={test_metrics['f1']:.4f}")

    return {
        "model": "CNN + FFT + LSTM",
        "seed": seed,
        "chosen_threshold": best_th,
        "ckpt_path": str(ckpt_path),
        **test_metrics,
    }


# ==============================================================================
# 5. CNN + FFT END-TO-END TRAINING & EVALUATION
# ==============================================================================
def predict_cnn_fft_loader(model, loader, device):
    model.eval()
    all_logits, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x_raw, x_fft, y in loader:
            x_raw = x_raw.to(device, non_blocking=PIN_MEMORY)
            x_fft = x_fft.to(device, non_blocking=PIN_MEMORY)
            logits = model(x_raw, x_fft).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_logits.append(logits.cpu().numpy().astype(np.float32))
            all_probs.append(probs.cpu().numpy().astype(np.float32))
            all_targets.append(y.numpy().astype(np.int64))
    return np.concatenate(all_logits), np.concatenate(all_probs), np.concatenate(all_targets)


def run_cnn_fft_seed(seed, epochs=15, batch_size=32, lr=1e-3):
    model_name = "cnn_fft"
    val_save_path = RESULTS_DIR / f"cnn_fft_seed{seed}_val_predictions.npz"
    test_save_path = RESULTS_DIR / f"cnn_fft_seed{seed}_test_predictions.npz"
    sweep_path = RESULTS_DIR / f"cnn_fft_seed{seed}_val_threshold_sweep.csv"
    ckpt_path = MODEL_DIR / f"cnn_fft_subjectwise_15subjects_v2_seed{seed}.pt"

    # Resumability check
    if test_save_path.exists() and sweep_path.exists():
        print(f"[{model_name.upper()} | SEED {seed}] Output files already exist. Loading saved predictions...")
        val_data = np.load(val_save_path)
        test_data = np.load(test_save_path)
        df_sweep = pd.read_csv(sweep_path)
        best_th = float(df_sweep.loc[df_sweep["f1"].idxmax()]["threshold"])
        test_metrics = evaluate_predictions_at_threshold(
            test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th
        )
        return {
            "model": "CNN + FFT",
            "seed": seed,
            "chosen_threshold": best_th,
            "ckpt_path": str(ckpt_path),
            **test_metrics,
        }

    # For seed 42, check if original results exist and re-sweep
    if seed == 42:
        orig_val = RESULTS_DIR / "cnn_fft_val_predictions.npz"
        orig_test = RESULTS_DIR / "cnn_fft_test_predictions.npz"
        orig_ckpt = MODEL_DIR / "cnn_fft_subjectwise_15subjects_v2.pt"
        if orig_val.exists() and orig_test.exists() and orig_ckpt.exists():
            print(f"[CNN + FFT | SEED 42] Re-evaluating existing seed 42 predictions under extended grid...")
            val_data = np.load(orig_val)
            test_data = np.load(orig_test)
            best_th, best_val_f1, _, _, _, df_sweep = sweep_thresholds(val_data["y_true"], val_data["probabilities"])
            df_sweep.to_csv(sweep_path, index=False)
            np.savez_compressed(val_save_path, **val_data)
            np.savez_compressed(test_save_path, **test_data)
            test_metrics = evaluate_predictions_at_threshold(test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th)
            return {
                "model": "CNN + FFT",
                "seed": seed,
                "chosen_threshold": best_th,
                "ckpt_path": str(orig_ckpt),
                **test_metrics,
            }

    seed_everything(seed)
    partitions, subject_sets = load_partitions()
    all_subjects = sorted(set().union(*subject_sets.values()))
    sources = {s: load_subject_fft(s) for s in all_subjects}

    train_subjects = sorted(subject_sets["train"])
    val_subjects = sorted(subject_sets["validation"])
    test_subjects = sorted(subject_sets["test"])

    train_ds = SubjectWindowFFTDataset(sources, train_subjects)
    val_ds = SubjectWindowFFTDataset(sources, val_subjects)
    test_ds = SubjectWindowFFTDataset(sources, test_subjects)

    val_sub_names = np.array([val_subjects[i] for i in val_ds.subject_ids])
    test_sub_names = np.array([test_subjects[i] for i in test_ds.subject_ids])

    y_train = train_ds.labels
    counts = np.bincount(y_train.astype(np.int64), minlength=2).astype(float)
    weights = 1.0 / counts
    sample_weights = torch.from_numpy(weights[y_train.astype(np.int64)]).double()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CNN + FFT | SEED {seed}] Training on {device} (Epochs={epochs}, LR={lr})")

    model = CNNFFT(n_channels=23, fft_dim=FFT_FEATURE_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_raw, x_fft, y in train_loader:
            x_raw = x_raw.to(device, non_blocking=PIN_MEMORY)
            x_fft = x_fft.to(device, non_blocking=PIN_MEMORY)
            y = y.to(device, non_blocking=PIN_MEMORY)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_raw, x_fft).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_raw.size(0)

        avg_loss = total_loss / len(train_ds)
        _, val_probs, val_targets = predict_cnn_fft_loader(model, val_loader, device)
        val_pr_auc = average_precision_score(val_targets, val_probs)
        val_roc_auc = roc_auc_score(val_targets, val_probs)

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f} | val PR-AUC={val_pr_auc:.4f} | val ROC-AUC={val_roc_auc:.4f}{' -> [BEST]' if is_best else ''}")

    print(f"Restoring best checkpoint: Epoch {best_epoch} (val PR-AUC={best_val_pr_auc:.4f})")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "seed": seed,
    }, ckpt_path)

    val_logits, val_probs, val_targets = predict_cnn_fft_loader(model, val_loader, device)
    np.savez_compressed(val_save_path, y_true=val_targets, logits=val_logits, probabilities=val_probs, subject_ids=val_sub_names)
    best_th, best_val_f1, _, _, _, df_sweep = sweep_thresholds(val_targets, val_probs)
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Chosen validation frozen threshold: {best_th:.4f} (Val F1={best_val_f1:.4f})")

    test_logits, test_probs, test_targets = predict_cnn_fft_loader(model, test_loader, device)
    np.savez_compressed(test_save_path, y_true=test_targets, logits=test_logits, probabilities=test_probs, subject_ids=test_sub_names)

    test_metrics = evaluate_predictions_at_threshold(test_targets, test_probs, test_sub_names, best_th)
    print(f"Test Evaluation: PR-AUC={test_metrics['pr_auc']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f} | F1={test_metrics['f1']:.4f}")

    return {
        "model": "CNN + FFT",
        "seed": seed,
        "chosen_threshold": best_th,
        "ckpt_path": str(ckpt_path),
        **test_metrics,
    }


# ==============================================================================
# REPORTING & AGGREGATION
# ==============================================================================
def print_summary_tables(results_list):
    print("\n" + "=" * 120)
    print("FINAL MULTI-SEED BENCHMARK EVALUATION (PROTOCOL V2: SEEDS 42, 43, 44)")
    print("=" * 120)

    models_order = ["CNN baseline", "CNN + FFT", "CNN + LSTM", "CNN + FFT + LSTM"]
    by_model = {m: [] for m in models_order}
    for r in results_list:
        if r["model"] in by_model:
            by_model[r["model"]].append(r)

    # 1. Overall Summary Table
    print("\n--- TABLE 1: OVERALL TEST PERFORMANCE (MEAN +/- STD ACROSS SEEDS) ---")
    headers = [
        "Model", "Seeds", "F1 (mean+/-std)", "PR-AUC (mean+/-std)",
        "ROC-AUC (mean+/-std)", "Precision (mean+/-std)", "Recall (mean+/-std)",
        "FP (mean)", "TP (mean)"
    ]

    row_data = []
    for m in models_order:
        entries = by_model[m]
        if not entries:
            continue
        seeds_str = ",".join(str(e["seed"]) for e in sorted(entries, key=lambda x: x["seed"]))
        f1s = [e["f1"] for e in entries]
        prs = [e["pr_auc"] for e in entries]
        rocs = [e["roc_auc"] for e in entries]
        precs = [e["precision"] for e in entries]
        recs = [e["recall"] for e in entries]
        fps = [e["fp"] for e in entries]
        tps = [e["tp"] for e in entries]

        fmt = lambda arr: f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}" if len(arr) > 1 else f"{np.mean(arr):.4f}"
        fmt_int = lambda arr: f"{np.mean(arr):.1f}"

        row_data.append([
            m, seeds_str, fmt(f1s), fmt(prs), fmt(rocs), fmt(precs), fmt(recs), fmt_int(fps), fmt_int(tps)
        ])

    df_overall = pd.DataFrame(row_data, columns=headers)
    print(df_overall.to_string(index=False))

    # 2. Per-Subject Summary Table
    print("\n--- TABLE 2: PER-SUBJECT HELD-OUT TEST PERFORMANCE (MEAN +/- STD ACROSS SEEDS) ---")
    sub_headers = ["Model", "Subject", "F1 (mean+/-std)", "PR-AUC (mean+/-std)", "ROC-AUC (mean+/-std)", "Precision (mean+/-std)", "Recall (mean+/-std)", "FP (mean)", "TP (mean)"]
    sub_rows = []

    for m in models_order:
        entries = by_model[m]
        if not entries:
            continue
        for s in TEST_SUBJECTS:
            s_f1s = [e["per_subject"][s]["f1"] for e in entries if s in e.get("per_subject", {})]
            s_prs = [e["per_subject"][s]["pr_auc"] for e in entries if s in e.get("per_subject", {})]
            s_rocs = [e["per_subject"][s]["roc_auc"] for e in entries if s in e.get("per_subject", {})]
            s_precs = [e["per_subject"][s]["precision"] for e in entries if s in e.get("per_subject", {})]
            s_recs = [e["per_subject"][s]["recall"] for e in entries if s in e.get("per_subject", {})]
            s_fps = [e["per_subject"][s]["fp"] for e in entries if s in e.get("per_subject", {})]
            s_tps = [e["per_subject"][s]["tp"] for e in entries if s in e.get("per_subject", {})]

            fmt = lambda arr: f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}" if len(arr) > 1 else f"{np.mean(arr):.4f}"
            fmt_int = lambda arr: f"{np.mean(arr):.1f}"

            sub_rows.append([
                m, s, fmt(s_f1s), fmt(s_prs), fmt(s_rocs), fmt(s_precs), fmt(s_recs), fmt_int(s_fps), fmt_int(s_tps)
            ])

    df_sub = pd.DataFrame(sub_rows, columns=sub_headers)
    print(df_sub.to_string(index=False))

    # Save summary files
    out_json = RESULTS_DIR / "multi_seed_v2_summary.json"
    with open(out_json, "w") as fp:
        json.dump(results_list, fp, indent=2)
    print(f"\nSaved full results to {out_json}")

    out_csv = RESULTS_DIR / "multi_seed_v2_overall_table.csv"
    df_overall.to_csv(out_csv, index=False)
    out_sub_csv = RESULTS_DIR / "multi_seed_v2_persubject_table.csv"
    df_sub.to_csv(out_sub_csv, index=False)
    print(f"Saved tables to {out_csv} and {out_sub_csv}")


# ==============================================================================
# MAIN EXECUTION DISPATCHER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Controlled Benchmark Runner (Protocol v2)")
    parser.add_argument("--report-only", action="store_true", help="Print summary of existing completed runs only")
    parser.add_argument("--skip-cnn-fft", action="store_true", help="Skip end-to-end CNN+FFT models (seeds 43/44)")
    args = parser.parse_args()

    results_all = []

    # Check which results already exist on disk
    models_prefixes = [
        ("CNN baseline", "baseline_cnn"),
        ("CNN + FFT", "cnn_fft"),
        ("CNN + LSTM", "cnn_lstm"),
        ("CNN + FFT + LSTM", "cnn_fft_lstm"),
    ]

    if args.report_only:
        print("[REPORT ONLY] Scanning completed seed predictions in results/...")
        for display_name, prefix in models_prefixes:
            for seed in [42, 43, 44]:
                test_p = RESULTS_DIR / f"{prefix}_seed{seed}_test_predictions.npz"
                sweep_p = RESULTS_DIR / f"{prefix}_seed{seed}_val_threshold_sweep.csv"
                if test_p.exists() and sweep_p.exists():
                    test_data = np.load(test_p)
                    df_sw = pd.read_csv(sweep_p)
                    best_th = float(df_sw.loc[df_sw["f1"].idxmax()]["threshold"])
                    metrics = evaluate_predictions_at_threshold(test_data["y_true"], test_data["probabilities"], test_data["subject_ids"], best_th)
                    results_all.append({"model": display_name, "seed": seed, "chosen_threshold": best_th, **metrics})
        if results_all:
            print_summary_tables(results_all)
        else:
            print("No completed seed predictions found.")
        return

    print("=" * 100)
    print("STARTING MULTI-SEED BENCHMARK CHAIN: SEEDS 43 -> 44 -> 42 (LSTMs) -> CNN+FFT (43/44)")
    print("=" * 100)
    t_start_all = time.time()

    # --------------------------------------------------------------------------
    # CHAIN 1: SEED 43
    # --------------------------------------------------------------------------
    print("\n" + "#" * 100)
    print("CHAIN 1/4: SEED 43 PIPELINE")
    print("#" * 100)
    clean_gpu()
    r_base_43 = run_baseline_cnn_seed(43)
    results_all.append(r_base_43)

    clean_gpu()
    cache_dir_43 = cache_features_for_seed(43, r_base_43["ckpt_path"])

    clean_gpu()
    r_lstm_43 = run_cnn_lstm_seed(43, cache_dir_43)
    results_all.append(r_lstm_43)

    clean_gpu()
    r_fftlstm_43 = run_cnn_fft_lstm_seed(43, cache_dir_43)
    results_all.append(r_fftlstm_43)

    # --------------------------------------------------------------------------
    # CHAIN 2: SEED 44
    # --------------------------------------------------------------------------
    print("\n" + "#" * 100)
    print("CHAIN 2/4: SEED 44 PIPELINE")
    print("#" * 100)
    clean_gpu()
    r_base_44 = run_baseline_cnn_seed(44)
    results_all.append(r_base_44)

    clean_gpu()
    cache_dir_44 = cache_features_for_seed(44, r_base_44["ckpt_path"])

    clean_gpu()
    r_lstm_44 = run_cnn_lstm_seed(44, cache_dir_44)
    results_all.append(r_lstm_44)

    clean_gpu()
    r_fftlstm_44 = run_cnn_fft_lstm_seed(44, cache_dir_44)
    results_all.append(r_fftlstm_44)

    # --------------------------------------------------------------------------
    # CHAIN 3: SEED 42 RE-RUN
    # --------------------------------------------------------------------------
    print("\n" + "#" * 100)
    print("CHAIN 3/4: SEED 42 CONSISTENCY RE-RUN")
    print("#" * 100)
    clean_gpu()
    r_base_42 = run_baseline_cnn_seed(42)
    results_all.append(r_base_42)

    clean_gpu()
    ckpt_42_baseline = MODEL_DIR / "baseline_cnn_subjectwise_15subjects_v2.pt"
    cache_dir_42 = cache_features_for_seed(42, ckpt_42_baseline)

    clean_gpu()
    r_lstm_42 = run_cnn_lstm_seed(42, cache_dir_42)
    results_all.append(r_lstm_42)

    clean_gpu()
    r_fftlstm_42 = run_cnn_fft_lstm_seed(42, cache_dir_42)
    results_all.append(r_fftlstm_42)

    # --------------------------------------------------------------------------
    # CHAIN 4: CNN + FFT (END-TO-END)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 100)
    print("CHAIN 4/4: CNN + FFT END-TO-END")
    print("#" * 100)
    clean_gpu()
    r_cnnfft_42 = run_cnn_fft_seed(42)
    results_all.append(r_cnnfft_42)

    if not args.skip_cnn_fft:
        clean_gpu()
        r_cnnfft_43 = run_cnn_fft_seed(43)
        results_all.append(r_cnnfft_43)

        clean_gpu()
        r_cnnfft_44 = run_cnn_fft_seed(44)
        results_all.append(r_cnnfft_44)
    else:
        print("[NOTE] --skip-cnn-fft set. Skipping CNN+FFT for seeds 43 & 44.")

    clean_gpu()
    print_summary_tables(results_all)
    print(f"\nTotal elapsed time: {(time.time() - t_start_all) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
