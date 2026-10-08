#!/usr/bin/env python3
"""
Standalone read-only diagnostic script to evaluate per-subject ROC-AUC and performance
for the trained CNN+LSTM model on each test subject individually (chb02, chb05, chb13).
"""

from pathlib import Path

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
from torch.utils.data import DataLoader

from train_cnn_lstm_subjectwise import CNNLSTM, build_split_dataset, evaluate_split

CHECKPOINT_PATH = Path("models/cnn_lstm_subjectwise_15subjects.pt")
TEST_SUBJECTS = ["chb02", "chb05", "chb13"]


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("=" * 85)
    print("PER-SUBJECT ROC-AUC & PERFORMANCE DIAGNOSTIC: CNN + LSTM")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 85)

    # 1. Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    input_dim = ckpt.get("input_dim", 128)
    hidden_size = ckpt.get("hidden_size", 128)
    sequence_length = ckpt.get("sequence_length", 5)

    print(f"Model architecture: CNNLSTM(input_dim={input_dim}, hidden_size={hidden_size})")
    print(f"Sequence length: {sequence_length} windows (20s context)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    model = CNNLSTM(input_dim=input_dim, hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    subject_results = {}

    # 2. Evaluate each subject individually
    for subject in TEST_SUBJECTS:
        print("=" * 85)
        print(f"SUBJECT: {subject}")
        print("=" * 85)

        # Build single-subject dataset using imported build_split_dataset
        ds = build_split_dataset([subject], sequence_length)
        loader = DataLoader(
            ds,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        metrics = evaluate_split(model, loader, device)
        targets = metrics["targets"]
        probs = metrics["probs"]
        preds = metrics["preds"]

        n_seq = len(targets)
        n_seiz = int(targets.sum())
        n_non_seiz = n_seq - n_seiz

        roc_auc = roc_auc_score(targets, probs)
        pr_auc = average_precision_score(targets, probs)
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)

        cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel()

        seiz_probs = probs[targets == 1]
        p_min = float(np.min(seiz_probs)) if len(seiz_probs) > 0 else 0.0
        p25 = float(np.percentile(seiz_probs, 25)) if len(seiz_probs) > 0 else 0.0
        p_med = float(np.median(seiz_probs)) if len(seiz_probs) > 0 else 0.0
        p75 = float(np.percentile(seiz_probs, 75)) if len(seiz_probs) > 0 else 0.0
        p_max = float(np.max(seiz_probs)) if len(seiz_probs) > 0 else 0.0

        non_seiz_probs = probs[targets == 0]
        ns_med = float(np.median(non_seiz_probs)) if len(non_seiz_probs) > 0 else 0.0
        ns_max = float(np.max(non_seiz_probs)) if len(non_seiz_probs) > 0 else 0.0

        subject_results[subject] = {
            "n_seq": n_seq,
            "n_seiz": n_seiz,
            "n_non_seiz": n_non_seiz,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "p_min": p_min,
            "p25": p25,
            "p_med": p_med,
            "p75": p75,
            "p_max": p_max,
            "ns_med": ns_med,
            "ns_max": ns_max,
        }

        print(f"Total Sequences           : {n_seq:,}")
        print(f"Seizure-Target Sequences  : {n_seiz:,} ({n_seiz / n_seq * 100:.3f}%)")
        print(f"Non-Seizure Sequences     : {n_non_seiz:,}")
        print("-" * 50)
        print(f"ROC-AUC                   : {roc_auc:.4f}")
        print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
        print(f"Precision (threshold 0.5) : {prec:.4f}")
        print(f"Recall (threshold 0.5)    : {rec:.4f} ({tp}/{n_seiz})")
        print(f"F1 Score (threshold 0.5)  : {f1:.4f}")
        print("-" * 50)
        print("Confusion Matrix:")
        print(f"  TP: {tp:5d}  |  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  |  TN: {tn:5d}")
        print("-" * 50)
        print("Seizure Probability Distribution (targets == 1):")
        print(f"  Min   : {p_min:.6f}")
        print(f"  p25   : {p25:.6f}")
        print(f"  Median: {p_med:.6f}")
        print(f"  p75   : {p75:.6f}")
        print(f"  Max   : {p_max:.6f}")
        print(f"Non-Seizure Probability (targets == 0): median={ns_med:.6f}, max={ns_max:.6f}")
        print()

    # 3. Side-by-side summary table
    print("=" * 85)
    print("SIDE-BY-SIDE SUMMARY COMPARISON ACROSS TEST SUBJECTS")
    print("=" * 85)
    header = f"{'Metric':<25} | {'chb02':^16} | {'chb05':^16} | {'chb13':^16}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Total Sequences", "{n_seq:,}"),
        ("Seizure Targets", "{n_seiz:,}"),
        ("ROC-AUC", "{roc_auc:.4f}"),
        ("PR-AUC", "{pr_auc:.4f}"),
        ("Precision (0.5)", "{precision:.4f}"),
        ("Recall (0.5)", "{recall:.4f}"),
        ("F1 Score (0.5)", "{f1:.4f}"),
        ("True Positives (TP)", "{tp}"),
        ("False Positives (FP)", "{fp}"),
        ("False Negatives (FN)", "{fn}"),
        ("True Negatives (TN)", "{tn:,}"),
        ("Seizure Prob Min", "{p_min:.6f}"),
        ("Seizure Prob 25th %", "{p25:.6f}"),
        ("Seizure Prob Median", "{p_med:.6f}"),
        ("Seizure Prob 75th %", "{p75:.6f}"),
        ("Seizure Prob Max", "{p_max:.6f}"),
        ("Non-Seiz Prob Median", "{ns_med:.6f}"),
    ]

    for label, fmt in rows:
        val02 = fmt.format(**subject_results["chb02"])
        val05 = fmt.format(**subject_results["chb05"])
        val13 = fmt.format(**subject_results["chb13"])
        print(f"{label:<25} | {val02:^16} | {val05:^16} | {val13:^16}")

    print("=" * 85)


if __name__ == "__main__":
    main()
