"""Cache 128-D CNN features for all 15 subjects using the corrected baseline CNN."""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data/processed")
CACHE_DIR = DATA_DIR / "cnn_features"
MODEL_PATH = Path("models/baseline_cnn_subjectwise_15subjects.pt")
BATCH_SIZE = 256

ALL_SUBJECTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
]


class SeizureCNN(nn.Module):
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
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def extract_features(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.gap(x).squeeze(-1)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return torch.from_numpy(self.windows[index].copy())


def cache_subject(subject, model, device):
    windows_path = DATA_DIR / f"{subject}_windows.npy"
    labels_path = DATA_DIR / f"{subject}_labels.npy"
    output_path = CACHE_DIR / f"{subject}_cnn_features.npy"

    if not windows_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing processed data for {subject}")

    windows = np.load(windows_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    n_windows, n_channels, n_samples = windows.shape
    if n_channels != 23 or n_samples != 512:
        raise ValueError(f"Unexpected shape {windows.shape} for {subject}")
    if len(windows) != len(labels):
        raise ValueError(f"Mismatch between windows ({len(windows)}) and labels ({len(labels)}) for {subject}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, SeizureCNN.FEATURE_DIM),
    )

    loader = DataLoader(
        WindowDataset(windows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    t0 = time.time()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = model.extract_features(batch).cpu().numpy().astype(np.float32)
            end = offset + len(features)
            output[offset:end] = features
            offset = end

    output.flush()
    dt = time.time() - t0

    # Verification
    cached = np.load(output_path, mmap_mode="r")
    assert cached.shape == (n_windows, 128), f"Bad cached shape: {cached.shape}"
    # Check sample slice for NaN/Inf
    assert not np.isnan(cached[:1000]).any(), f"NaNs found in {subject}"
    assert not np.isinf(cached[:1000]).any(), f"Infs found in {subject}"

    speed = n_windows / dt if dt > 0 else 0
    print(
        f"  {subject}: {n_windows:,} windows -> {cached.shape} "
        f"({cached.nbytes / 1e6:.1f} MB) in {dt:.1f}s ({speed:.0f} win/s)"
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("CACHING 128-D CNN FEATURES FOR 15-SUBJECT CHB-MIT EXPERIMENT")
    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")
    print("=" * 80)

    model = SeizureCNN(23)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_windows = 0
    total_time = 0.0
    t_start = time.time()

    if len(sys.argv) > 1:
        subjects = sys.argv[1:]
        for sub in subjects:
            if sub not in ALL_SUBJECTS:
                raise ValueError(f"Invalid subject '{sub}'. Must be one of {ALL_SUBJECTS}")
    else:
        subjects = ALL_SUBJECTS

    for subject in subjects:
        t0 = time.time()
        cache_subject(subject, model, device)
        total_time += time.time() - t0

    total_elapsed = time.time() - t_start
    print("=" * 80)
    print(f"Feature caching complete for {len(subjects)} subjects in {total_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
