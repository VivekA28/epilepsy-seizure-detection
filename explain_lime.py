"""
LIME explainability for the seizure-detection CNN.

lime's official package has no built-in time-series explainer (only
lime_tabular, lime_image, lime_text), so this implements LIME's core
algorithm directly against EEG windows: segment each window into
channel x time-block "superfeatures", perturb them, observe how the
model's prediction changes, and fit a local weighted linear model
whose coefficients become the importance scores.

The surrogate model is fit against the raw logit (pre-sigmoid), not
the probability - fitting against probability causes the surrogate to
collapse to near-zero importance whenever the model is confidently
predicting near 0 or 1, since the sigmoid saturates and barely moves
even for fairly large input changes.

Explained windows are chosen deterministically (first 3 seizure / first
3 non-seizure windows from the first subject passed on the command
line), matching explain_shap.py's window selection - so SHAP and LIME
results can be directly compared window-for-window.

Run from the project root:
    python explain_lime.py chb01
    python explain_lime.py chb01 chb02 chb03 chb04 chb05
"""

import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Model definition - must match train_baseline.py exactly
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
# 2. Load data
# ---------------------------------------------------------------------
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["chb01"]
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

all_windows, all_labels = [], []
for subject in SUBJECTS:
    w = np.load(DATA_DIR / f"{subject}_windows.npy", mmap_mode="r")
    l = np.load(DATA_DIR / f"{subject}_labels.npy")
    all_windows.append(w)
    all_labels.append(l)
    print(f"  {subject}: {len(w)} windows, {int(l.sum())} seizure")

labels = np.concatenate(all_labels, axis=0)
n_channels = all_windows[0].shape[1]
n_samples = all_windows[0].shape[2]

channel_names_path = DATA_DIR / f"{SUBJECTS[0]}_channel_names.npy"
if channel_names_path.exists():
    channel_names = np.load(channel_names_path, allow_pickle=True)
else:
    channel_names = [f"ch{i}" for i in range(n_channels)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_window(global_idx):
    offset = 0
    for w in all_windows:
        if global_idx < offset + len(w):
            return np.array(w[global_idx - offset], dtype=np.float32)
        offset += len(w)
    raise IndexError(global_idx)

# ---------------------------------------------------------------------
# 3. Load the trained checkpoint
# ---------------------------------------------------------------------
model_name = f"baseline_cnn_{'_'.join(SUBJECTS)}.pt"
model = SeizureCNN(n_channels).to(device)
model.load_state_dict(torch.load(MODEL_DIR / model_name, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_DIR / model_name}")

@torch.no_grad()
def predict_logit(windows_batch):
    """windows_batch: (N, n_channels, n_samples) numpy -> (N,) raw logits (pre-sigmoid)"""
    x = torch.from_numpy(windows_batch).float().to(device)
    return model(x).squeeze(-1).cpu().numpy()

@torch.no_grad()
def predict_proba(windows_batch):
    """windows_batch: (N, n_channels, n_samples) numpy -> (N,) seizure probabilities"""
    logits = predict_logit(windows_batch)
    return 1.0 / (1.0 + np.exp(-logits))

# ---------------------------------------------------------------------
# 4. LIME core: segment, perturb, fit local linear model
# ---------------------------------------------------------------------
N_TIME_SEGMENTS = 4                      # coarse time resolution
SEG_LEN = n_samples // N_TIME_SEGMENTS   # samples per time-segment
N_FEATURES = n_channels * N_TIME_SEGMENTS  # e.g. 23 x 4 = 92 superfeatures
N_PERTURBATIONS = 500                    # samples used to fit the local surrogate

def segment_id(ch, t_bin):
    return ch * N_TIME_SEGMENTS + t_bin

def explain_window(window, rng):
    """
    window: (n_channels, n_samples) - the real EEG window to explain

    Returns:
        importance: (n_channels, N_TIME_SEGMENTS) array of LIME coefficients
        actual_prob: model's real predicted probability on the unperturbed window
    """
    actual_prob = predict_proba(window[None, ...])[0]

    # --- generate perturbations ---
    # Z: binary mask matrix (N_PERTURBATIONS, N_FEATURES) - 1 = superfeature
    # kept as-is, 0 = superfeature replaced with 0 (the "absent" baseline,
    # meaningful here since data is z-score normalized per channel)
    Z = rng.integers(0, 2, size=(N_PERTURBATIONS, N_FEATURES))
    # Ensure the original (all-ones) sample is included, so the surrogate
    # model is anchored at the real point being explained
    Z[0, :] = 1

    perturbed_batch = np.zeros((N_PERTURBATIONS, n_channels, n_samples), dtype=np.float32)
    for p in range(N_PERTURBATIONS):
        sample = window.copy()
        for ch in range(n_channels):
            for t_bin in range(N_TIME_SEGMENTS):
                if Z[p, segment_id(ch, t_bin)] == 0:
                    start = t_bin * SEG_LEN
                    end = start + SEG_LEN if t_bin < N_TIME_SEGMENTS - 1 else n_samples
                    sample[ch, start:end] = 0.0
        perturbed_batch[p] = sample

    # --- get model predictions for every perturbation (in batches, to
    # avoid holding all N_PERTURBATIONS forward passes' memory at once) ---
    # Fit against the raw logit, not the probability - fitting against
    # probability causes the surrogate to collapse to near-zero importance
    # whenever the model is confidently near 0 or 1, since the sigmoid
    # saturates and barely moves even for fairly large input changes.
    preds = []
    batch_size = 64
    for i in range(0, N_PERTURBATIONS, batch_size):
        preds.append(predict_logit(perturbed_batch[i:i + batch_size]))
    y_perturbed = np.concatenate(preds)

    # --- LIME's proximity weighting: perturbations that changed fewer
    # superfeatures (i.e. stayed closer to the original) count more when
    # fitting the local surrogate model ---
    fraction_kept = Z.mean(axis=1)
    kernel_width = 0.25
    distances = 1.0 - fraction_kept  # 0 = identical to original, 1 = everything masked
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # --- fit a local weighted linear (Ridge) model: Z -> y_perturbed ---
    # its coefficients are the LIME importance scores per superfeature
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(Z, y_perturbed, sample_weight=weights)

    importance = surrogate.coef_.reshape(n_channels, N_TIME_SEGMENTS)
    return importance, actual_prob

# ---------------------------------------------------------------------
# 5. Pick sample windows to explain
# ---------------------------------------------------------------------
# Match explain_shap.py exactly: always pull from chb01's own labels
# (subject index 0), and take the first 3 of each class deterministically
# - not a random global sample - so LIME explains the identical windows
# SHAP already explained, making the two directly comparable.
rng = np.random.default_rng(42)
first_subject_labels = all_labels[0]
explain_seizure_idx = np.where(first_subject_labels == 1)[0][:3]
explain_nonseizure_idx = np.where(first_subject_labels == 0)[0][:3]

print(f"\nExplaining {len(explain_seizure_idx)} seizure windows and "
      f"{len(explain_nonseizure_idx)} non-seizure windows from {SUBJECTS[0]}")

def run_and_report(idx, true_label):
    window = get_window(idx)
    importance, actual_prob = explain_window(window, rng)

    print(f"\nWindow {idx} (true label: {true_label}, "
          f"model predicted probability: {actual_prob:.3f})")

    channel_importance = np.abs(importance).sum(axis=1)  # sum over time-segments per channel
    order = np.argsort(channel_importance)[::-1]
    print("Top 5 most influential channels (LIME):")
    for c in order[:5]:
        print(f"  {channel_names[c]}: importance {channel_importance[c]:.4f}")

    # visualize as a channel x time-segment heatmap, matching the SHAP
    # script's visual style for easy side-by-side comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.abs(importance).max()
    vmax = vmax if vmax > 0 else 1.0  # avoid a degenerate 0-0 color range
    im = ax.imshow(importance, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"LIME attribution — window {idx} (true: {true_label}, "
                 f"pred prob: {actual_prob:.2f})")
    ax.set_xlabel("Time segment")
    ax.set_ylabel("EEG channel")
    ax.set_yticks(np.arange(n_channels))
    ax.set_yticklabels(channel_names, fontsize=6)
    ax.set_xticks(np.arange(N_TIME_SEGMENTS))
    fig.colorbar(im, ax=ax, label="LIME coefficient (red = pushes toward seizure)")
    fig.tight_layout()
    out_path = RESULTS_DIR / f"lime_channel_importance_{true_label}_{idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved chart to {out_path}")

for idx in explain_seizure_idx:
    run_and_report(int(idx), "SEIZURE")
for idx in explain_nonseizure_idx:
    run_and_report(int(idx), "non-seizure")

print("\nDone. LIME explanation charts saved to results/")