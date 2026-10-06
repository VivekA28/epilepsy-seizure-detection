"""
Diagnostic script: Continuous validation predictions, distribution analysis,
threshold sweep, and calibration curves for 15-subject baseline CNN.

Uses the saved checkpoint:
    models/baseline_cnn_subjectwise_15subjects.pt
Evaluates ONLY the validation subjects:
    chb07, chb08
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
from torch.utils.data import DataLoader

from train_baseline_subjectwise import SeizureCNN, SubjectWindowDataset, load_subject

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path("models/baseline_cnn_subjectwise_15subjects.pt")

VAL_SUBJECTS = ["chb07", "chb08"]
BATCH_SIZE = 128


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SeizureCNN(n_channels=23).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH} (epoch {ckpt.get('best_epoch')})")

    # Load validation subjects only
    print("\nLoading validation subjects:", VAL_SUBJECTS)
    sources = {}
    for s in VAL_SUBJECTS:
        sources[s] = load_subject(s)
        print(f"  {s}: {len(sources[s]['labels']):,} windows | {int(sources[s]['labels'].sum()):,} seizure")

    val_ds = SubjectWindowDataset(sources, VAL_SUBJECTS)
    print(f"Total validation windows: {len(val_ds):,} | Seizures: {int(val_ds.labels.sum()):,}")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Inference: collect raw logits, sigmoid probabilities, true labels
    logits_list = []
    targets_list = []

    print("\nRunning inference on validation partition...")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x).squeeze(-1)
            logits_list.append(out.cpu().numpy())
            targets_list.append(y.numpy().astype(np.float32))

    logits = np.concatenate(logits_list)
    y_true = np.concatenate(targets_list).astype(np.int64)

    # Compute continuous probabilities via sigmoid
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    # Save artifact
    output_npz = RESULTS_DIR / "baseline_15subjects_validation_predictions.npz"
    np.savez_compressed(
        output_npz,
        y_true=y_true,
        logits=logits,
        probabilities=probabilities,
    )
    print(f"Saved continuous predictions to {output_npz}")

    # TASK 2: Proper Validation Metrics
    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    brier = brier_score_loss(y_true, probabilities)
    prevalence = np.mean(y_true)

    print("\n" + "=" * 80)
    print("TASK 2 — CONTINUOUS VALIDATION METRICS")
    print("=" * 80)
    print(f"Validation Prevalence : {prevalence:.6f} ({int(y_true.sum())} / {len(y_true)})")
    print(f"ROC-AUC               : {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"Brier Score           : {brier:.6f}")

    # TASK 3: Probability Distributions
    seizure_mask = y_true == 1
    non_seizure_mask = y_true == 0

    probs_sz = probabilities[seizure_mask]
    probs_non = probabilities[non_seizure_mask]

    percentiles = [5, 25, 50, 75, 95, 99]

    def get_stats(arr):
        stats = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }
        pcts = np.percentile(arr, percentiles)
        for p, val in zip(percentiles, pcts):
            stats[f"p{p}"] = float(val)
        return stats

    stats_sz = get_stats(probs_sz)
    stats_non = get_stats(probs_non)

    print("\n" + "=" * 80)
    print("TASK 3 — PROBABILITY DISTRIBUTIONS")
    print("=" * 80)
    print(f"Group A: Actual Seizure Windows (N = {len(probs_sz):,}):")
    for k, v in stats_sz.items():
        print(f"  {k:8s}: {v:.6f}")

    print(f"\nGroup B: Actual Non-Seizure Windows (N = {len(probs_non):,}):")
    for k, v in stats_non.items():
        print(f"  {k:8s}: {v:.6f}")

    threshold_cutoffs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    print("\nTrue Seizure Windows Exceeding Cutoffs:")
    for cut in threshold_cutoffs:
        count = int(np.sum(probs_sz > cut))
        pct = (count / len(probs_sz)) * 100.0
        print(f"  P > {cut:0.2f}: {count:5d} / {len(probs_sz):5d} ({pct:6.2f}%)")

    # TASK 4: Validation Threshold Sweep
    print("\n" + "=" * 80)
    print("TASK 4 — VALIDATION THRESHOLD SWEEP (Grid: 0.01 to 0.50)")
    print("=" * 80)

    sweep_thresholds = np.linspace(0.01, 0.50, 50)
    sweep_results = []

    best_thresh = None
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_counts = None

    for th in sweep_thresholds:
        preds = (probabilities >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        sweep_results.append({
            "threshold": float(th),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_prec = prec
            best_rec = rec
            best_counts = (tp, fp, fn, tn)

    df_sweep = pd.DataFrame(sweep_results)
    sweep_csv = RESULTS_DIR / "baseline_15subjects_val_threshold_sweep.csv"
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"Saved threshold sweep table to {sweep_csv}")

    # Print summary table at key intervals
    print("\nKey Threshold Steps:")
    print(f"{'Thresh':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'TN':>8}")
    print("-" * 70)
    sample_steps = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for row in sweep_results:
        if any(np.isclose(row["threshold"], s, atol=1e-3) for s in sample_steps) or np.isclose(row["threshold"], best_thresh, atol=1e-3):
            marker = " *" if np.isclose(row["threshold"], best_thresh, atol=1e-3) else ""
            print(f"{row['threshold']:7.3f} | {row['precision']:7.4f} | {row['recall']:7.4f} | {row['f1']:7.4f} | {row['tp']:6d} | {row['fp']:6d} | {row['fn']:6d} | {row['tn']:8d}{marker}")

    print("\nBest Validation Threshold:")
    print(f"  Threshold : {best_thresh:.4f}")
    print(f"  Precision : {best_prec:.4f}")
    print(f"  Recall    : {best_rec:.4f}")
    print(f"  F1 Score  : {best_f1:.4f}")
    print(f"  TP: {best_counts[0]}, FP: {best_counts[1]}, FN: {best_counts[2]}, TN: {best_counts[3]}")

    # TASK 5: Diagnostic Plots
    print("\n" + "=" * 80)
    print("TASK 5 — GENERATING DIAGNOSTIC CURVES")
    print("=" * 80)

    # 1. Precision-Recall Curve
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(rec_curve, prec_curve, color="darkorange", lw=2, label=f"PR curve (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=prevalence, color="navy", linestyle="--", label=f"No-skill baseline ({prevalence:.4f})")
    plt.plot(best_rec, best_prec, "ro", markersize=8, label=f"Best F1 (th={best_thresh:.3f}, F1={best_f1:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curve (CHB-07 & CHB-08)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    pr_curve_path = RESULTS_DIR / "baseline_15subjects_val_pr_curve.png"
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {pr_curve_path}")

    # 2. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkblue", lw=2, label=f"ROC curve (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve (CHB-07 & CHB-08)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    roc_curve_path = RESULTS_DIR / "baseline_15subjects_val_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_curve_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {roc_curve_path}")

    # 3. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, "s-", color="purple", lw=2, label=f"Model (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Empirical)")
    plt.title("Validation Calibration / Reliability Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    cal_curve_path = RESULTS_DIR / "baseline_15subjects_val_calibration_curve.png"
    plt.tight_layout()
    plt.savefig(cal_curve_path, dpi=150)
    plt.close()
    print(f"Saved Calibration curve to {cal_curve_path}")

    # Output summary JSON
    summary_data = {
        "validation_windows": len(y_true),
        "validation_seizures": int(y_true.sum()),
        "validation_prevalence": float(prevalence),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "stats_seizure": stats_sz,
        "stats_non_seizure": stats_non,
        "cutoffs_seizure_exceeded": {
            str(cut): {
                "count": int(np.sum(probs_sz > cut)),
                "pct": float((np.sum(probs_sz > cut) / len(probs_sz)) * 100.0),
            }
            for cut in threshold_cutoffs
        },
        "best_threshold": float(best_thresh),
        "best_precision": float(best_prec),
        "best_recall": float(best_rec),
        "best_f1": float(best_f1),
        "best_tp": int(best_counts[0]),
        "best_fp": int(best_counts[1]),
        "best_fn": int(best_counts[2]),
        "best_tn": int(best_counts[3]),
    }
    summary_json = RESULTS_DIR / "baseline_15subjects_val_diagnostic_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary JSON to {summary_json}")


if __name__ == "__main__":
    main()
