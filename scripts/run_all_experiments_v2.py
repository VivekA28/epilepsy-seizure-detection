"""
Master Sequential Experiment Runner (v2) for 15-Subject CHB-MIT Split.

Runs all 4 models sequentially on the GPU:
1. Baseline CNN (v2)
2. CNN + FFT (v2)
3. CNN + LSTM (v2)
4. CNN + FFT + LSTM (v2)

All models share:
- Seed: 42
- Checkpoint selection: Best epoch by VALIDATION PR-AUC
- Validation threshold sweep: 0.01-0.99 (step 0.01) + 0.991-0.999 (step 0.001)
- Frozen threshold test evaluation executed exactly once
- Per-subject breakdown on test set (chb02, chb05, chb13)
- Sequential execution (NUM_WORKERS=0, PIN_MEMORY=False)
"""

import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from scripts.train_baseline_subjectwise_v2 import train_baseline
from scripts.train_cnn_fft_subjectwise_v2 import train_cnn_fft
from scripts.train_cnn_lstm_subjectwise_v2 import train_cnn_lstm
from scripts.train_cnn_fft_lstm_subjectwise_v2 import train_cnn_fft_lstm

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("MASTER SEQUENTIAL EXPERIMENT RUNNER: 4 ARCHITECTURES UNDER PROTOCOL V2")
    print("=" * 100)
    t_master_start = time.time()

    all_summaries = []

    # 1. Baseline CNN
    print("\n" + "#" * 100)
    print("EXPERIMENT 1/4: CNN BASELINE")
    print("#" * 100)
    t0 = time.time()
    clean_gpu()
    summary_baseline = train_baseline()
    clean_gpu()
    print(f"CNN Baseline completed in {(time.time() - t0)/60:.1f} minutes.")
    all_summaries.append(summary_baseline)

    # 2. CNN + FFT
    print("\n" + "#" * 100)
    print("EXPERIMENT 2/4: CNN + FFT MULTIMODAL")
    print("#" * 100)
    t0 = time.time()
    clean_gpu()
    summary_cnn_fft = train_cnn_fft()
    clean_gpu()
    print(f"CNN + FFT completed in {(time.time() - t0)/60:.1f} minutes.")
    all_summaries.append(summary_cnn_fft)

    # 3. CNN + LSTM
    print("\n" + "#" * 100)
    print("EXPERIMENT 3/4: CNN + LSTM TEMPORAL")
    print("#" * 100)
    t0 = time.time()
    clean_gpu()
    summary_cnn_lstm = train_cnn_lstm()
    clean_gpu()
    print(f"CNN + LSTM completed in {(time.time() - t0)/60:.1f} minutes.")
    all_summaries.append(summary_cnn_lstm)

    # 4. CNN + FFT + LSTM
    print("\n" + "#" * 100)
    print("EXPERIMENT 4/4: CNN + FFT + LSTM MULTIMODAL TEMPORAL")
    print("#" * 100)
    t0 = time.time()
    clean_gpu()
    summary_cnn_fft_lstm = train_cnn_fft_lstm()
    clean_gpu()
    print(f"CNN + FFT + LSTM completed in {(time.time() - t0)/60:.1f} minutes.")
    all_summaries.append(summary_cnn_fft_lstm)

    total_time = (time.time() - t_master_start) / 60
    print("\n" + "=" * 100)
    print(f"ALL 4 EXPERIMENTS COMPLETED IN {total_time:.1f} MINUTES")
    print("=" * 100)

    # Save complete JSON
    json_path = RESULTS_DIR / "all_models_v2_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved raw results JSON to {json_path}")

    # Build Comparison Table
    # model | val PR-AUC | val F1 (tuned thr) | chosen threshold | test PR-AUC | test ROC-AUC | test F1 | test precision | test recall | FP | TP
    table_rows = []
    for s in all_summaries:
        table_rows.append({
            "model": s["model"],
            "val PR-AUC": f"{s['val_pr_auc']:.4f}",
            "val F1 (tuned thr)": f"{s['val_f1_tuned']:.4f}",
            "chosen threshold": f"{s['chosen_threshold']:.3f}",
            "test PR-AUC": f"{s['test_pr_auc']:.4f}",
            "test ROC-AUC": f"{s['test_roc_auc']:.4f}",
            "test F1": f"{s['test_f1']:.4f}",
            "test precision": f"{s['test_precision']:.4f}",
            "test recall": f"{s['test_recall']:.4f}",
            "FP": s["fp"],
            "TP": s["tp"],
        })

    df_table = pd.DataFrame(table_rows)
    csv_path = RESULTS_DIR / "all_models_v2_comparison.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to {csv_path}")

    print("\n" + "=" * 110)
    print("OVERALL MODEL COMPARISON TABLE")
    print("=" * 110)
    print(df_table.to_string(index=False))

    # Print Per-Subject Tables
    for s in all_summaries:
        print("\n" + "-" * 90)
        print(f"PER-SUBJECT BREAKDOWN: {s['model'].upper()} (Frozen Thr: {s['chosen_threshold']:.3f})")
        print("-" * 90)
        sub_rows = []
        for sub, metrics in s["per_subject"].items():
            sub_rows.append({
                "Subject": sub,
                "Windows": metrics["samples"],
                "Seizures": metrics["positives"],
                "PR-AUC": f"{metrics['pr_auc']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TN": metrics["tn"],
            })
        print(pd.DataFrame(sub_rows).to_string(index=False))


if __name__ == "__main__":
    main()
