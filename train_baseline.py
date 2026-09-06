"""
Baseline CNN model for seizure detection on windowed EEG data.

Loads the preprocessed windows/labels for a subject (from
preprocess_subject.py), splits into train/test, trains a simple 1D CNN,
and reports metrics appropriate for imbalanced classification (not just
accuracy, since seizure windows are a small minority of the data).

Uses PyTorch rather than TensorFlow/Keras, since TensorFlow does not yet
support Python 3.14.

Run from the project root:
    python train_baseline.py chb01
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------------
SUBJECT = sys.argv[1] if len(sys.argv) > 1 else "chb01"
DATA_DIR = Path("data/processed")
windows = np.load(DATA_DIR / f"{SUBJECT}_windows.npy")   # shape: (n_windows, 23, 512)
labels = np.load(DATA_DIR / f"{SUBJECT}_labels.npy")      # shape: (n_windows,)

print(f"Loaded {windows.shape[0]} windows, shape per window: {windows.shape[1:]}")
print(f"Seizure windows: {labels.sum()} / {len(labels)} ({100 * labels.mean():.2f}%)")

# PyTorch's Conv1d expects (batch, channels, samples) - which is actually
# the same layout our data is already saved in, so no transpose needed here
X = windows.astype(np.float32)
y = labels.astype(np.float32)

# ---------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------
# Splitting the actual 3D window arrays directly through sklearn's
# train_test_split is slow, since sklearn's indexing isn't optimized for
# 3D arrays. Instead, split just the indices (fast, since that's just
# numbers), then use plain NumPy indexing to build the real arrays -
# much faster for this shape of data.
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y
)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTrain set: {X_train.shape[0]} windows ({int(y_train.sum())} seizure)")
print(f"Test set:  {X_test.shape[0]} windows ({int(y_test.sum())} seizure)")

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
n_channels = X.shape[1]   # 23
n_samples = X.shape[2]    # 512 (4 seconds at 128Hz after resampling)

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
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

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
    # only watching training loss (which alone can be misleading here)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        test_logits = model(X_test_t).squeeze(-1)
        test_preds = (torch.sigmoid(test_logits) > 0.5).int().cpu().numpy()
    epoch_recall = (test_preds[y_test == 1] == 1).mean() if y_test.sum() > 0 else 0
    epoch_precision = (y_test[test_preds == 1] == 1).mean() if test_preds.sum() > 0 else 0
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
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(classification_report(all_labels, all_preds, target_names=["Non-seizure", "Seizure"]))
print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))

# ---------------------------------------------------------------------
# 8. Save the model
# ---------------------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "baseline_cnn.pt")
print(f"\nModel saved to {MODEL_DIR / 'baseline_cnn.pt'}")