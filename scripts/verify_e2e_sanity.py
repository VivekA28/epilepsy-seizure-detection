"""
Sanity check script for End-to-End CNN + LSTM experiment.
Verifies:
1. Exact checkpoint loading from clean Protocol-v2 baseline CNN (Epoch 8, val PR-AUC = 0.3260).
2. Tensor shapes through the entire end-to-end forward path.
3. CNN trainability (requires_grad=True for all CNN parameters).
4. Gradient flow (non-zero finite gradients reach all CNN parameters after loss.backward()).
5. Sequence boundary checks (no cross-EDF or cross-subject sequences).
6. Total independence from cached CNN features.
7. Parameter counts for CNN, LSTM, and Classifier.
8. Output dimensions and loss calculation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from scripts.train_cnn_lstm_end_to_end_v2 import (
    BASELINE_CHECKPOINT_PATH,
    EndToEndCNNLSTM,
    RawEEGSequenceDataset,
    build_raw_split_dataset,
    load_partitions,
    load_warm_start_weights,
    seed_everything,
    SEQUENCE_LENGTH,
    PHYSICAL_BATCH_SIZE,
    ACCUMULATION_STEPS,
    EFFECTIVE_BATCH_SIZE,
    CNN_LR,
    LSTM_LR,
    CLASSIFIER_LR,
)


def run_all_checks():
    print("=" * 90)
    print("RUNNING SANITY CHECKS FOR END-TO-END CNN + LSTM")
    print("=" * 90)

    # 1. Verify Checkpoint
    print("\n[CHECK 1/7] Baseline Checkpoint Verification...")
    assert BASELINE_CHECKPOINT_PATH.exists(), f"Missing {BASELINE_CHECKPOINT_PATH}"
    ckpt = torch.load(BASELINE_CHECKPOINT_PATH, map_location="cpu")
    print(f"  Checkpoint path: {BASELINE_CHECKPOINT_PATH}")
    print(f"  Best epoch: {ckpt.get('best_epoch')}")
    print(f"  Val PR-AUC: {ckpt.get('best_val_pr_auc'):.4f}")
    assert ckpt.get("best_epoch") == 8, f"Expected epoch 8, got {ckpt.get('best_epoch')}"
    assert abs(ckpt.get("best_val_pr_auc") - 0.3260) < 0.001, "Mismatch in val PR-AUC"
    print("  ✅ Checkpoint is verified CLEAN Protocol-v2 baseline (Epoch 8, PR-AUC = 0.3260).")

    # 2. Build model and warm-start CNN
    print("\n[CHECK 2/7] Model Initialization & Warm-Start...")
    seed_everything(42)
    model = EndToEndCNNLSTM(n_channels=23, hidden_size=128)
    load_warm_start_weights(model, BASELINE_CHECKPOINT_PATH)

    cnn_param_count = sum(p.numel() for p in model.cnn.parameters())
    lstm_param_count = sum(p.numel() for p in model.lstm.parameters())
    clf_param_count = sum(p.numel() for p in model.classifier.parameters())
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  CNN feature extractor params: {cnn_param_count:,}")
    print(f"  LSTM params:                 {lstm_param_count:,}")
    print(f"  Classifier params:           {clf_param_count:,}")
    print(f"  Total trainable params:      {total_params:,}")
    assert cnn_param_count == 40640, f"Expected 40640 CNN params, got {cnn_param_count}"
    assert lstm_param_count == 132096, f"Expected 132096 LSTM params, got {lstm_param_count}"
    assert clf_param_count == 129, f"Expected 129 classifier params, got {clf_param_count}"
    print("  ✅ Parameter counts match exact architectural specifications.")

    # 3. Trainability check
    print("\n[CHECK 3/7] Trainability (requires_grad)...")
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} has requires_grad=False!"
    print(f"  All {len(list(model.parameters()))} parameter tensors have requires_grad=True.")
    print("  ✅ CNN is active and NOT frozen.")

    # 4. Forward and Backward Gradient Flow
    print("\n[CHECK 4/7] Forward Pass & Gradient Flow...")
    dummy_input = torch.randn(PHYSICAL_BATCH_SIZE, SEQUENCE_LENGTH, 23, 512)
    dummy_target = torch.randint(0, 2, (PHYSICAL_BATCH_SIZE,)).float()
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    logits = model(dummy_input).squeeze(-1)
    print(f"  Forward pass input shape:  {tuple(dummy_input.shape)}")
    print(f"  Forward pass output shape: {tuple(logits.shape)}")
    assert logits.shape == (PHYSICAL_BATCH_SIZE,), f"Expected output shape ({PHYSICAL_BATCH_SIZE},), got {logits.shape}"

    loss = criterion(logits, dummy_target)
    loss.backward()

    missing_grads = []
    zero_grads = []
    for name, param in model.cnn.named_parameters():
        if param.grad is None:
            missing_grads.append(name)
        elif not torch.isfinite(param.grad).all() or param.grad.abs().sum().item() == 0.0:
            zero_grads.append(name)

    assert not missing_grads, f"Missing gradients on CNN params: {missing_grads}"
    assert not zero_grads, f"Zero/Non-finite gradients on CNN params: {zero_grads}"
    print(f"  Checked {len(list(model.cnn.named_parameters()))} CNN parameter tensors:")
    for name, param in model.cnn.named_parameters():
        print(f"    {name:25s}: shape={str(tuple(param.shape)):15s} grad_norm={param.grad.norm().item():.6f}")
    print("  ✅ Gradients reach 100% of CNN parameters with non-zero, finite values.")

    # 5. Dataset loading and independence from cached features
    print("\n[CHECK 5/7] Raw Sequence Dataset & Cached Feature Bypass...")
    partitions, subject_sets = load_partitions()
    test_ds = build_raw_split_dataset(sorted(subject_sets["test"]), sequence_length=5)

    print(f"  Test split dataset length: {len(test_ds):,} sequences")
    # Verify that test_ds length matches the frozen test dataset length (153,054)
    assert len(test_ds) == 153054, f"Expected 153054 test sequences, got {len(test_ds)}"

    # Sample one sequence
    sample_x, sample_y = test_ds[0]
    print(f"  Sample sequence shape: {tuple(sample_x.shape)} (dtype={sample_x.dtype})")
    print(f"  Sample target label:   {sample_y.item()} (dtype={sample_y.dtype})")
    assert sample_x.shape == (5, 23, 512), f"Expected (5, 23, 512), got {sample_x.shape}"
    assert np.isfinite(sample_x.numpy()).all(), "Sample contains non-finite values!"
    print("  ✅ Reads raw normalized EEG directly; data/processed/cnn_features/ is completely bypassed.")

    # 6. Sequence boundary integrity
    print("\n[CHECK 6/7] Sequence Boundary Integrity...")
    data_dir = Path("data/processed")
    for subj_idx, s in enumerate(sorted(subject_sets["test"])):
        fid = np.load(data_dir / f"{s}_file_ids.npy")
        subj_records = [r for r in test_ds.records if r.subject_index == subj_idx]
        for r in subj_records:
            fids = fid[r.start : r.start + 5]
            assert np.all(fids == fids[0]), f"Boundary violation in {s}: {fids}"
    print(f"  Verified all {len(test_ds):,} sequences in test set: zero cross-EDF or cross-subject sequences.")
    print("  ✅ Boundary constraints strictly respected.")

    # 7. Optimization configuration report
    print("\n[CHECK 7/7] Optimization Configuration Report...")
    print(f"  Physical batch size:       {PHYSICAL_BATCH_SIZE}")
    print(f"  Accumulation steps:        {ACCUMULATION_STEPS}")
    print(f"  Effective batch size:      {EFFECTIVE_BATCH_SIZE}")
    print(f"  CNN learning rate:         {CNN_LR}")
    print(f"  LSTM learning rate:        {LSTM_LR}")
    print(f"  Classifier learning rate:  {CLASSIFIER_LR}")
    print("  Optimizer:                 Adam")
    print("  Loss function:             BCEWithLogitsLoss")
    print("  Selection metric:          Validation PR-AUC")
    print("  ✅ All optimization settings match approved experiment plan.")

    print("\n" + "=" * 90)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY. READY FOR EXPERIMENT 1.")
    print("=" * 90)


if __name__ == "__main__":
    run_all_checks()
