"""
Baseline CNN model for seizure detection on windowed EEG data.

Loads the preprocessed windows/labels for one or more subjects, splits the
recordings into TRAIN / VALIDATION / TEST at the EDF/file level, trains a
simple 1D CNN, selects the best checkpoint using VALIDATION F1, and evaluates
the TEST set only once after training.

Important:
- This local development experiment is file-level, not patient-independent.
- Overlapping windows from the same EDF never cross a partition boundary.
- The final expanded experiment will use subject-level separation.

Uses PyTorch rather than TensorFlow/Keras.

Run from the project root:
    python train_baseline.py chb01
    python train_baseline.py chb01 chb02 chb03 chb04 chb05
"""

import sys
import copy
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["chb01"]
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3

assert abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) < 1e-8

windows_sources = []
all_labels = []
all_file_ids = []
next_file_id = 0
offset = 0

for subject in SUBJECTS:
    windows_path = DATA_DIR / f"{subject}_windows.npy"
    labels_path = DATA_DIR / f"{subject}_labels.npy"
    file_ids_path = DATA_DIR / f"{subject}_file_ids.npy"

    w = np.load(windows_path, mmap_mode="r")
    l = np.load(labels_path)
    f = np.load(file_ids_path)

    f_global = f + next_file_id
    next_file_id = int(f_global.max()) + 1

    windows_sources.append((w, offset))
    offset += len(w)

    all_labels.append(l)
    all_file_ids.append(f_global)

    print(
        f"  {subject}: {len(w)} windows, {int(l.sum())} seizure "
        f"({100 * l.mean():.2f}%)"
    )

if not windows_sources:
    raise RuntimeError("No subjects were provided or loaded.")

labels = np.concatenate(all_labels, axis=0)
file_ids = np.concatenate(all_file_ids, axis=0)
total_windows = len(labels)
y = labels.astype(np.float32)

print(f"\nCombined across {len(SUBJECTS)} subject(s): {total_windows} windows total")
print(f"Seizure windows: {int(labels.sum())} / {total_windows} ({100 * labels.mean():.2f}%)")

def get_window(global_idx):
    """Fetch one window from the appropriate memory-mapped subject array."""
    for w, src_offset in reversed(windows_sources):
        if global_idx >= src_offset:
            return np.array(w[global_idx - src_offset], dtype=np.float32)
    raise IndexError(global_idx)

class SeizureWindowDataset(Dataset):
    """Lazy Dataset over selected global window indices."""

    def __init__(self, indices, labels):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = int(self.indices[i])
        x = get_window(global_idx)
        y_val = self.labels[global_idx]
        return torch.from_numpy(x), torch.tensor(y_val, dtype=torch.float32)

unique_files = np.unique(file_ids)
file_has_seizure = np.array(
    [labels[file_ids == f].max() for f in unique_files],
    dtype=np.int8,
)

n_seizure_files = int(file_has_seizure.sum())
n_clean_files = int(len(file_has_seizure) - n_seizure_files)

if n_seizure_files < 2 or n_clean_files < 2:
    raise RuntimeError(
        "Not enough seizure-containing and clean EDFs for a stratified "
        "70/15/15 file-level split. Load more files/subjects before training. "
        f"Found {n_seizure_files} seizure EDFs and {n_clean_files} clean EDFs."
    )

train_files, temp_files, train_file_labels, temp_file_labels = train_test_split(
    unique_files,
    file_has_seizure,
    test_size=(VALIDATION_FRACTION + TEST_FRACTION),
    random_state=RANDOM_STATE,
    stratify=file_has_seizure,
)

temp_seizure_files = int(temp_file_labels.sum())
temp_clean_files = int(len(temp_file_labels) - temp_seizure_files)
if temp_seizure_files < 2 or temp_clean_files < 2:
    raise RuntimeError(
        "Not enough files remain for a stratified validation/test split. "
        "Load more files/subjects before training. "
        f"Temporary split has {temp_seizure_files} seizure EDFs and "
        f"{temp_clean_files} clean EDFs."
    )

val_files, test_files = train_test_split(
    temp_files,
    test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
    random_state=RANDOM_STATE,
    stratify=temp_file_labels,
)

train_mask = np.isin(file_ids, train_files)
val_mask = np.isin(file_ids, val_files)
test_mask = np.isin(file_ids, test_files)

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print("\nFile-level partition:")
print(
    f"  Train:      {len(train_files)} files, {len(train_idx)} windows "
    f"({int(y_train.sum())} seizure)"
)
print(
    f"  Validation: {len(val_files)} files, {len(val_idx)} windows "
    f"({int(y_val.sum())} seizure)"
)
print(
    f"  Test:       {len(test_files)} files, {len(test_idx)} windows "
    f"({int(y_test.sum())} seizure)"
)
print("  NOTE: this is file-level, not patient-independent.")

if y_train.sum() == 0 or y_train.sum() == len(y_train):
    raise RuntimeError("Training partition contains only one class.")
if y_val.sum() == 0 or y_val.sum() == len(y_val):
    raise RuntimeError("Validation partition contains only one class.")
if y_test.sum() == 0 or y_test.sum() == len(y_test):
    raise RuntimeError("Test partition contains only one class.")

class_sample_count = np.array(
    [len(y_train) - y_train.sum(), y_train.sum()], dtype=np.float64
)
weight_per_class = 1.0 / class_sample_count
sample_weights = np.array(
    [weight_per_class[int(label)] for label in y_train],
    dtype=np.float64,
)
sample_weights = torch.from_numpy(sample_weights).double()

sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

print(
    f"\nUsing weighted oversampling on TRAIN only "
    f"(non-seizure weight: {weight_per_class[0]:.6f}, "
    f"seizure weight: {weight_per_class[1]:.6f})"
)

n_channels = windows_sources[0][0].shape[1]
n_samples = windows_sources[0][0].shape[2]

print(f"\nInput shape per window: ({n_channels}, {n_samples})")

class SeizureCNN(nn.Module):
    """Baseline CNN; `extract_features()` returns the 128-D CNN representation."""

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
        """Return the 128-D representation after convolution + global pooling."""
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = SeizureCNN(n_channels).to(device)
print(model)

train_dataset = SeizureWindowDataset(train_idx, y)
val_dataset = SeizureWindowDataset(val_idx, y)
test_dataset = SeizureWindowDataset(test_idx, y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def collect_predictions(loader):
    """Return binary predictions and labels for one evaluation partition."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            predictions.append(preds)
            targets.append(y_batch.numpy().astype(np.int64))

    return np.concatenate(predictions), np.concatenate(targets)

best_val_f1 = -1.0
best_state = None
best_epoch = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)

    val_preds, val_targets = collect_predictions(val_loader)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    val_recall = (
        (val_preds[val_targets == 1] == 1).mean()
        if (val_targets == 1).any()
        else 0.0
    )
    val_precision = (
        (val_targets[val_preds == 1] == 1).mean()
        if (val_preds == 1).any()
        else 0.0
    )

    print(
        f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f} - "
        f"val recall: {val_recall:.2f} - "
        f"val precision: {val_precision:.2f} - val f1: {val_f1:.2f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        print(f"  -> new best validation checkpoint (f1={best_val_f1:.2f})")

if best_state is None:
    raise RuntimeError("No validation checkpoint was produced.")

model.load_state_dict(best_state)
print(f"\nRestored best checkpoint from epoch {best_epoch} (validation f1={best_val_f1:.2f})")

print("\nEvaluating TEST set once using the restored validation-selected checkpoint...")
final_preds, final_labels = collect_predictions(test_loader)

print("\n" + "=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)
print(
    classification_report(
        final_labels,
        final_preds,
        target_names=["Non-seizure", "Seizure"],
        zero_division=0,
    )
)
print("Confusion matrix:")
print(confusion_matrix(final_labels, final_preds))

model_name = f"baseline_cnn_{'_'.join(SUBJECTS)}.pt"
model_path = MODEL_DIR / model_name
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")
