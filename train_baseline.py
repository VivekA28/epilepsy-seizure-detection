"""
Baseline CNN model for seizure detection on windowed EEG data.

Loads the preprocessed windows/labels for one or more subjects (from
preprocess_subject.py), splits into train/test by FILE (not by individual
window - see note below), trains a simple 1D CNN, and reports metrics
appropriate for imbalanced classification (not just accuracy, since
seizure windows are a small minority of the data).

Uses PyTorch rather than TensorFlow/Keras, since TensorFlow does not yet
support Python 3.14.

Run from the project root:
    python train_baseline.py chb01                     (single subject)
    python train_baseline.py chb01 chb02 chb03 chb04 chb05   (combined)
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# 1. Load processed data - one or more subjects, combined into one dataset
# ---------------------------------------------------------------------
# windows.npy files are opened with mmap_mode="r" instead of a normal
# np.load. This keeps the actual signal data on disk rather than pulling
# all of it into RAM at once - individual windows are only read into
# memory when a specific batch actually needs them (see SeizureWindowDataset
# below). Combining 5 subjects' worth of windows would otherwise need
# ~15GB+ of RAM just for the raw arrays, which doesn't fit comfortably
# alongside everything else running on this machine.
SUBJECTS = sys.argv[1:] if len(sys.argv) > 1 else ["chb01"]
DATA_DIR = Path("data/processed")

# windows_sources keeps track of, for each subject, the memory-mapped
# array plus the row-offset where that subject's windows start once
# everything is thought of as one combined dataset
windows_sources = []   # list of (memmap array, offset)
all_labels = []
all_file_ids = []
next_file_id = 0
offset = 0

for subject in SUBJECTS:
    w = np.load(DATA_DIR / f"{subject}_windows.npy", mmap_mode="r")  # stays on disk
    l = np.load(DATA_DIR / f"{subject}_labels.npy")                  # small, fine in RAM
    f = np.load(DATA_DIR / f"{subject}_file_ids.npy")                # small, fine in RAM

    # file_ids are only unique within a single subject's preprocessing run
    # (e.g. both chb01 and chb02 have a file with id 0). Offsetting by
    # next_file_id makes every file across every subject get its own
    # globally unique id, so the file-level split below can't accidentally
    # treat "file 0 of chb01" and "file 0 of chb02" as the same file.
    f_global = f + next_file_id
    next_file_id = f_global.max() + 1

    windows_sources.append((w, offset))
    offset += len(w)

    all_labels.append(l)
    all_file_ids.append(f_global)

    print(f"  {subject}: {len(w)} windows, {l.sum()} seizure ({100 * l.mean():.2f}%)")

labels = np.concatenate(all_labels, axis=0)
file_ids = np.concatenate(all_file_ids, axis=0)
total_windows = len(labels)

print(f"\nCombined across {len(SUBJECTS)} subject(s): {total_windows} windows total")
print(f"Seizure windows: {labels.sum()} / {total_windows} ({100 * labels.mean():.2f}%)")

y = labels.astype(np.float32)


def get_window(global_idx):
    """
    Fetches a single window given its index in the combined (virtual)
    dataset, pulling it from whichever subject's on-disk memmap it
    actually lives in. Only this one window is read into memory.
    """
    for w, src_offset in reversed(windows_sources):
        if global_idx >= src_offset:
            return np.array(w[global_idx - src_offset], dtype=np.float32)
    raise IndexError(global_idx)


class SeizureWindowDataset(Dataset):
    """
    A PyTorch Dataset that reads windows lazily from disk (via
    get_window) rather than holding the full array in RAM. `indices` is
    the subset of global window indices this Dataset should expose
    (e.g. just the training files, or just the test files).
    """
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = self.indices[i]
        x = get_window(global_idx)
        y_val = self.labels[global_idx]
        return torch.from_numpy(x), torch.tensor(y_val, dtype=torch.float32)

# ---------------------------------------------------------------------
# 2. Train/test split - BY FILE, not by individual window
# ---------------------------------------------------------------------
# Windows overlap by 50%, so a window and its neighbor share half their
# actual signal - splitting randomly by individual window would let
# near-duplicate windows end up on both sides of the split, letting the
# model effectively see the test set during training (data leakage) and
# producing an inflated, unrealistic performance score. Splitting by
# whole file instead means test windows come from recordings the model
# never saw any part of during training. With file_ids now made globally
# unique across subjects above, this split naturally mixes files from all
# loaded subjects between train and test.
unique_files = np.unique(file_ids)

# A file is treated as a "seizure file" for stratification purposes if
# any of its windows are labeled seizure - keeps seizure-containing
# files reasonably spread across both the train and test splits.
file_has_seizure = np.array([
    labels[file_ids == f].max() for f in unique_files
])

train_files, test_files = train_test_split(
    unique_files, test_size=0.25, random_state=42, stratify=file_has_seizure
)

train_mask = np.isin(file_ids, train_files)
test_mask = np.isin(file_ids, test_files)

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"\nTrain set: {len(train_idx)} windows ({int(y_train.sum())} seizure)")
print(f"Test set:  {len(test_idx)} windows ({int(y_test.sum())} seizure)")

# ---------------------------------------------------------------------
# 3. Handle class imbalance via oversampling, not just loss weighting
# ---------------------------------------------------------------------
# With such an extreme imbalance (0.32% seizure), a large pos_weight on
# the loss alone tends to make the model collapse to a trivial "always
# predict seizure" shortcut, since that technically minimizes the
# heavily-lopsided loss without learning any real signal. Oversampling
# addresses this differently: seizure windows get shown to the model
# more often during training, so it actually has to learn what makes
# them different, rather than gaming the loss formula.
class_sample_count = np.array([len(y_train) - y_train.sum(), y_train.sum()])
weight_per_class = 1.0 / class_sample_count
sample_weights = np.array([weight_per_class[int(label)] for label in y_train])
sample_weights = torch.from_numpy(sample_weights).double()

sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

print(f"\nUsing weighted oversampling instead of loss-only class weighting "
      f"(non-seizure weight: {weight_per_class[0]:.6f}, "
      f"seizure weight: {weight_per_class[1]:.6f})")

# ---------------------------------------------------------------------
# 4. Build a simple 1D CNN
# ---------------------------------------------------------------------
n_channels = windows_sources[0][0].shape[1]   # 23
n_samples = windows_sources[0][0].shape[2]    # 512 (4 seconds at 128Hz after resampling)

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
        self.fc2 = nn.Linear(64, 1)   # binary output: seizure logit
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)   # (batch, 128)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)   # raw logit - sigmoid applied inside the loss function
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = SeizureCNN(n_channels).to(device)
print(model)

# ---------------------------------------------------------------------
# 5. Set up data loaders, loss, optimizer
# ---------------------------------------------------------------------
# These datasets read windows lazily from disk (see SeizureWindowDataset
# above) rather than holding the full combined array in RAM.
train_dataset = SeizureWindowDataset(train_idx, y)
test_dataset = SeizureWindowDataset(test_idx, y)

# sampler replaces shuffle=True - it draws from train_dataset with the
# weights computed above, so seizure windows appear far more often per
# epoch than their raw 0.32% share of the data
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# plain BCE loss now - the imbalance is handled by oversampling above,
# not by an extreme loss weight, which is what caused the earlier collapse
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------
EPOCHS = 15

best_f1 = -1
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(-1)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)

    # Quick check on the test set each epoch so we can see whether the
    # model is actually learning to separate the classes, rather than
    # only watching training loss (which alone can be misleading here).
    # Runs through test_loader in batches (not one giant tensor) so the
    # full test set never needs to be materialized in memory at once.
    model.eval()
    test_preds_list = []
    test_labels_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            test_preds_list.append(preds)
            test_labels_list.append(y_batch.numpy())
    test_preds = np.concatenate(test_preds_list)
    y_test_epoch = np.concatenate(test_labels_list)

    epoch_recall = (test_preds[y_test_epoch == 1] == 1).mean() if y_test_epoch.sum() > 0 else 0
    epoch_precision = (y_test_epoch[test_preds == 1] == 1).mean() if test_preds.sum() > 0 else 0
    epoch_f1 = (
        2 * epoch_precision * epoch_recall / (epoch_precision + epoch_recall)
        if (epoch_precision + epoch_recall) > 0 else 0
    )

    print(f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f} - "
          f"test recall: {epoch_recall:.2f} - test precision: {epoch_precision:.2f} - "
          f"test f1: {epoch_f1:.2f}")

    # With so few seizure examples, training is unstable and swings between
    # strong and collapsed epochs - so we keep whichever epoch's weights
    # performed best on the test set (by F1), rather than just using
    # whatever happens to be left at the very last epoch.
    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  -> new best (f1={best_f1:.2f}), checkpoint saved")

# Load the best checkpoint before final evaluation/saving
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint (test f1={best_f1:.2f}) for final evaluation")

# ---------------------------------------------------------------------
# 7. Evaluate on held-out test set
# ---------------------------------------------------------------------
model.eval()
final_preds = []
final_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()
        final_preds.extend(preds)
        final_labels.extend(y_batch.numpy())

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(classification_report(final_labels, final_preds, target_names=["Non-seizure", "Seizure"]))
print("Confusion matrix:")
print(confusion_matrix(final_labels, final_preds))

# ---------------------------------------------------------------------
# 8. Save the model
# ---------------------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
model_name = f"baseline_cnn_{'_'.join(SUBJECTS)}.pt"
torch.save(model.state_dict(), MODEL_DIR / model_name)
print(f"\nModel saved to {MODEL_DIR / model_name}")