"""
SHAP explainability layer for the seizure detection CNN.

Loads a trained model checkpoint, picks a handful of real test windows
(some seizure, some non-seizure), and computes SHAP values showing which
channels (electrodes) and which time regions within each window drove
that specific prediction. This is the project's core original
contribution - turning a black-box "seizure detected" output into
something a clinician could actually inspect and question.

Run from the project root (after train_baseline.py has produced a model):
    python explain_shap.py chb01 chb02 chb03 chb04 chb05
    (subject list must match whatever the model was trained on)
"""

import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Model definition - must match train_baseline.py exactly, since we're
# loading a checkpoint saved from that architecture
# ---------------------------------------------------------------------
class SeizureCNN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding="same")
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["chb01", "chb02", "chb03", "chb04", "chb05"]
DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models") / f"baseline_cnn_{'_'.join(SUBJECTS)}.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 1. Load a manageable sample of data (not the full multi-subject set -
#    SHAP is expensive to compute per-sample, so we only need a
#    background sample plus a handful of examples to actually explain)
# ---------------------------------------------------------------------
print("Loading a sample of data for explanation...")
SAMPLE_SUBJECT = SUBJECTS[0]
windows = np.load(DATA_DIR / f"{SAMPLE_SUBJECT}_windows.npy", mmap_mode="r")
labels = np.load(DATA_DIR / f"{SAMPLE_SUBJECT}_labels.npy")

n_channels = windows.shape[1]
n_samples_per_window = windows.shape[2]

# Background sample: SHAP needs a reference set of "typical" inputs to
# compare against when attributing importance. A random sample of
# mostly-normal windows works well here - doesn't need to be huge.
rng = np.random.default_rng(42)
background_idx = rng.choice(len(windows), size=100, replace=False)
background = np.array(windows[background_idx], dtype=np.float32)
background_t = torch.from_numpy(background).to(device)

# Pick a few real seizure windows and a few real non-seizure windows to
# actually explain
seizure_idx = np.where(labels == 1)[0][:3]
normal_idx = np.where(labels == 0)[0][:3]
explain_idx = np.concatenate([seizure_idx, normal_idx])
explain_windows = np.array(windows[explain_idx], dtype=np.float32)
explain_labels = labels[explain_idx]
explain_t = torch.from_numpy(explain_windows).to(device)

print(f"Explaining {len(seizure_idx)} seizure windows and {len(normal_idx)} non-seizure windows "
      f"from {SAMPLE_SUBJECT}")

# ---------------------------------------------------------------------
# 2. Load the trained model
# ---------------------------------------------------------------------
model = SeizureCNN(n_channels).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# ---------------------------------------------------------------------
# 3. Compute SHAP values
# ---------------------------------------------------------------------
# GradientExplainer works directly with PyTorch models and is far
# cheaper than KernelExplainer for this input size (23 channels x 512
# samples per window) - it uses the model's gradients to attribute
# importance, rather than the model-agnostic (but much slower)
# perturbation approach KernelExplainer uses.
print("Computing SHAP values (this can take a minute)...")
explainer = shap.GradientExplainer(model, background_t)
shap_values = explainer.shap_values(explain_t)

# shap_values shape: (n_explained, n_channels, n_samples_per_window, 1)
# squeeze out the trailing singleton dimension from the single output neuron
shap_values = np.array(shap_values).squeeze(-1)

# ---------------------------------------------------------------------
# 4. Summarize importance per channel (which electrodes mattered most)
# ---------------------------------------------------------------------
CHANNEL_NAMES = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"
][:n_channels]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

for i, (idx, true_label) in enumerate(zip(explain_idx, explain_labels)):
    with torch.no_grad():
        pred_logit = model(explain_t[i:i+1]).item()
    pred_prob = torch.sigmoid(torch.tensor(pred_logit)).item()

    # sum of |shap value| across time, per channel - a simple summary of
    # "how much did this channel matter overall for this prediction"
    channel_importance = np.abs(shap_values[i]).sum(axis=1)
    top_channels = np.argsort(channel_importance)[::-1][:5]

    label_str = "SEIZURE" if true_label == 1 else "non-seizure"
    print(f"\nWindow {idx} (true label: {label_str}, model predicted probability: {pred_prob:.3f})")
    print("Top 5 most influential channels:")
    for ch in top_channels:
        print(f"  {CHANNEL_NAMES[ch]}: importance {channel_importance[ch]:.4f}")

    # Save a bar chart of channel importance for this specific window
    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(channel_importance)[::-1]
    ax.bar(range(len(order)), channel_importance[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CHANNEL_NAMES[c] for c in order], rotation=90)
    ax.set_ylabel("Sum |SHAP value|")
    ax.set_title(f"Channel importance - window {idx} (true: {label_str}, pred prob: {pred_prob:.2f})")
    plt.tight_layout()
    fname = OUT_DIR / f"shap_channel_importance_{label_str}_{idx}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved chart to {fname}")

print("\nDone. SHAP explanation charts saved to results/")