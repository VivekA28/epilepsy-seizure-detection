"""
Baseline CNN Evaluation & Threshold Calibration on 15-Subject CHB-MIT Split.

Protocol:
1. Load baseline checkpoint: models/baseline_cnn_subjectwise_15subjects.pt (epoch 13).
2. Run inference on validation (chb07, chb08) and save predictions to:
   results/baseline_15subjects_val_predictions.npz
3. Sweep thresholds 0.01-0.99 (step 0.01) on validation probabilities to find threshold
   maximizing validation F1. Save sweep to:
   results/baseline_15subjects_val_threshold_sweep.csv
4. Freeze that threshold.
5. Run test inference exactly once on test set (chb02, chb05, chb13) and save predictions to:
   results/baseline_15subjects_test_predictions.npz
6. Evaluate test set at frozen threshold (precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC),
   and report per-subject test ROC-AUC and PR-AUC.
"""

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
from torch.utils.data import DataLoader

from train_baseline_subjectwise import SeizureCNN, SubjectWindowDataset, load_subject

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path("models/baseline_cnn_subjectwise_15subjects.pt")

VAL_SUBJECTS = ["chb07", "chb08"]
TEST_SUBJECTS = ["chb02", "chb05", "chb13"]
BATCH_SIZE = 256


def run_inference(model, subjects, device):
    """Run forward pass once across given subjects, preserving per-subject boundaries."""
    sources = {}
    for s in subjects:
        sources[s] = load_subject(s)

    ds = SubjectWindowDataset(sources, subjects)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logits_list = []
    targets_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).squeeze(-1)
            logits_list.append(out.cpu().numpy().astype(np.float32))
            targets_list.append(y.numpy().astype(np.int64))

    logits = np.concatenate(logits_list)
    y_true = np.concatenate(targets_list)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    # Map subject ids string array
    sub_names_array = np.array([subjects[i] for i in ds.subject_ids])

    return y_true, logits, probabilities, sub_names_array, sources


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("15-SUBJECT BASELINE CNN EVALUATION & THRESHOLD CALIBRATION")
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 80)

    # Load model
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SeizureCNN(n_channels=23).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (best epoch: {ckpt.get('best_epoch')}, best val F1: {ckpt.get('best_validation_f1'):.4f})")

    # -------------------------------------------------------------
    # 1. Validation Inference
    # -------------------------------------------------------------
    print("\n--- Running Validation Inference (chb07, chb08) ---")
    val_y_true, val_logits, val_probs, val_sub_ids, _ = run_inference(model, VAL_SUBJECTS, device)
    print(f"Validation samples: {len(val_y_true):,} | Positives: {int(val_y_true.sum()):,}")

    val_save_path = RESULTS_DIR / "baseline_15subjects_val_predictions.npz"
    np.savez_compressed(
        val_save_path,
        y_true=val_y_true,
        logits=val_logits,
        probabilities=val_probs,
        subject_ids=val_sub_ids,
    )
    print(f"Saved validation predictions to: {val_save_path}")

    # -------------------------------------------------------------
    # 2. Validation Threshold Sweep (0.01 - 0.99, step 0.01)
    # -------------------------------------------------------------
    print("\n--- Sweeping Validation Thresholds (0.01 to 0.99) ---")
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep_records = []

    best_thresh = None
    best_val_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_cm = None

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        cm = confusion_matrix(val_y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        p = precision_score(val_y_true, preds, zero_division=0)
        r = recall_score(val_y_true, preds, zero_division=0)
        f1 = f1_score(val_y_true, preds, zero_division=0)

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

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = float(th)
            best_prec = float(p)
            best_rec = float(r)
            best_cm = cm

    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = RESULTS_DIR / "baseline_15subjects_val_threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"Saved threshold sweep to: {sweep_csv_path}")

    val_roc_auc = roc_auc_score(val_y_true, val_probs)
    val_pr_auc = average_precision_score(val_y_true, val_probs)

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Validation ROC-AUC      : {val_roc_auc:.4f}")
    print(f"Validation PR-AUC       : {val_pr_auc:.4f}")
    print(f"Selected Best Threshold : {best_thresh:.2f}")
    print(f"Best Validation F1      : {best_val_f1:.4f}")
    print(f"Precision @ best th     : {best_prec:.4f}")
    print(f"Recall @ best th        : {best_rec:.4f}")
    print("Confusion Matrix @ best th:")
    print(best_cm)

    # -------------------------------------------------------------
    # 3. Test Inference (Run forward pass exactly once)
    # -------------------------------------------------------------
    print("\n--- Running Test Inference (chb02, chb05, chb13) ---")
    test_y_true, test_logits, test_probs, test_sub_ids, _ = run_inference(model, TEST_SUBJECTS, device)
    print(f"Test samples: {len(test_y_true):,} | Positives: {int(test_y_true.sum()):,}")

    test_save_path = RESULTS_DIR / "baseline_15subjects_test_predictions.npz"
    np.savez_compressed(
        test_save_path,
        y_true=test_y_true,
        logits=test_logits,
        probabilities=test_probs,
        subject_ids=test_sub_ids,
    )
    print(f"Saved test predictions to: {test_save_path}")

    # -------------------------------------------------------------
    # 4. Test Evaluation at Frozen Threshold
    # -------------------------------------------------------------
    frozen_th = best_thresh
    test_preds = (test_probs >= frozen_th).astype(int)

    test_roc_auc = roc_auc_score(test_y_true, test_probs)
    test_pr_auc = average_precision_score(test_y_true, test_probs)
    test_prec = precision_score(test_y_true, test_preds, zero_division=0)
    test_rec = recall_score(test_y_true, test_preds, zero_division=0)
    test_f1 = f1_score(test_y_true, test_preds, zero_division=0)
    test_cm = confusion_matrix(test_y_true, test_preds, labels=[0, 1])

    print("\n" + "=" * 80)
    print(f"FINAL TEST EVALUATION AT FROZEN THRESHOLD ({frozen_th:.2f})")
    print("=" * 80)
    print(f"Overall Test ROC-AUC  : {test_roc_auc:.4f} (from continuous probabilities)")
    print(f"Overall Test PR-AUC   : {test_pr_auc:.4f} (from continuous probabilities)")
    print(f"Overall Test Precision: {test_prec:.4f}")
    print(f"Overall Test Recall   : {test_rec:.4f}")
    print(f"Overall Test F1       : {test_f1:.4f}")
    print(f"Confusion Matrix:")
    print(test_cm)

    # -------------------------------------------------------------
    # 5. Per-Subject Test Metrics
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-SUBJECT TEST METRICS")
    print("=" * 80)
    per_sub_rows = []
    for s in TEST_SUBJECTS:
        mask = (test_sub_ids == s)
        sub_y = test_y_true[mask]
        sub_p = test_probs[mask]
        sub_pred = test_preds[mask]

        sub_roc = roc_auc_score(sub_y, sub_p) if len(np.unique(sub_y)) > 1 else float("nan")
        sub_pr = average_precision_score(sub_y, sub_p) if len(np.unique(sub_y)) > 1 else float("nan")
        sub_prec = precision_score(sub_y, sub_pred, zero_division=0)
        sub_rec = recall_score(sub_y, sub_pred, zero_division=0)
        sub_f1 = f1_score(sub_y, sub_pred, zero_division=0)
        sub_cm = confusion_matrix(sub_y, sub_pred, labels=[0, 1])

        per_sub_rows.append({
            "subject": s,
            "samples": int(len(sub_y)),
            "positives": int(sub_y.sum()),
            "roc_auc": sub_roc,
            "pr_auc": sub_pr,
            "precision": sub_prec,
            "recall": sub_rec,
            "f1": sub_f1,
            "tn": int(sub_cm[0, 0]),
            "fp": int(sub_cm[0, 1]),
            "fn": int(sub_cm[1, 0]),
            "tp": int(sub_cm[1, 1]),
        })

        print(f"\nSubject {s.upper()}:")
        print(f"  Windows   : {len(sub_y):,} (Positives: {int(sub_y.sum()):,})")
        print(f"  ROC-AUC   : {sub_roc:.4f}")
        print(f"  PR-AUC    : {sub_pr:.4f}")
        print(f"  Precision : {sub_prec:.4f}")
        print(f"  Recall    : {sub_rec:.4f}")
        print(f"  F1        : {sub_f1:.4f}")
        print(f"  CM        : {sub_cm.tolist()}")

    per_sub_df = pd.DataFrame(per_sub_rows)
    per_sub_csv = RESULTS_DIR / "baseline_15subjects_per_subject_test_metrics.csv"
    per_sub_df.to_csv(per_sub_csv, index=False)
    print(f"\nSaved per-subject test metrics to: {per_sub_csv}")


if __name__ == "__main__":
    main()
