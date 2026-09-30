"""Cache 128-D CNN features for the current preprocessed EEG windows."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
CACHE_DIR = DATA_DIR / "cnn_features"
BATCH_SIZE = 256


class SeizureCNN(nn.Module):
    FEATURE_DIM = 128

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

    def extract_features(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.global_pool(x).squeeze(-1)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.windows[index], dtype=np.float32))


def cache_subject(subject, model, device, force=False):
    windows_path = DATA_DIR / f"{subject}_windows.npy"
    labels_path = DATA_DIR / f"{subject}_labels.npy"
    output_path = CACHE_DIR / f"{subject}_cnn_features.npy"

    if not windows_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing processed data for {subject}")

    windows = np.load(windows_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if len(windows) != len(labels):
        raise ValueError(f"Window/label length mismatch for {subject}")
    if windows.ndim != 3:
        raise ValueError(f"Expected (N, C, T) windows for {subject}, got {windows.shape}")

    n_windows, n_channels, _ = windows.shape
    model_n_channels = model.conv1.in_channels
    if n_channels != model_n_channels:
        raise ValueError(
            f"Channel mismatch for {subject}: data has {n_channels}, "
            f"model expects {model_n_channels}"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        cached = np.load(output_path, mmap_mode="r")
        expected_shape = (n_windows, SeizureCNN.FEATURE_DIM)
        if cached.shape == expected_shape:
            print(f"{subject}: cache already exists -> {output_path}")
            return
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
        pin_memory=device.type == "cuda",
    )

    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            features = model.extract_features(batch).cpu().numpy().astype(np.float32)
            end = offset + len(features)
            output[offset:end] = features
            offset = end

    output.flush()
    cached = np.load(output_path, mmap_mode="r")
    print(
        f"{subject}: windows={n_windows}, features={cached.shape}, "
        f"seizure={int(labels.sum())} -> {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+", help="Subjects, e.g. chb01 chb02 ...")
    parser.add_argument("--model", default=None, help="CNN checkpoint path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    model_path = (
        Path(args.model)
        if args.model
        else MODEL_DIR / f"baseline_cnn_{'_'.join(args.subjects)}.pt"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {model_path}")

    first_windows = np.load(DATA_DIR / f"{args.subjects[0]}_windows.npy", mmap_mode="r")
    model = SeizureCNN(first_windows.shape[1])
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"CNN checkpoint: {model_path}")

    for subject in args.subjects:
        cache_subject(subject, model, device, args.force)


if __name__ == "__main__":
    main()
