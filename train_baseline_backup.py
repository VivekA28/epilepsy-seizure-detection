"""
Subject-wise CNN training for CHB-MIT seizure detection.

Uses:
    data/partitions/train_metadata.csv
    data/partitions/validation_metadata.csv
    data/partitions/test_metadata.csv

Important:
    - Subject-wise train/validation/test split.
    - EDF ID + window_index used to map metadata to EEG windows.
    - Validation is used for model selection.
    - Test set is used ONLY for final evaluation.
    - EEG arrays remain memory-mapped on disk.
    - Weighted oversampling handles severe class imbalance.

Run:
    python train_baseline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/processed")
PARTITION_DIR = Path("data/partitions")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 90)
print("CHB-MIT SUBJECT-WISE CNN TRAINING")
print("=" * 90)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD PARTITION METADATA
# ============================================================

train_meta = pd.read_csv(
    PARTITION_DIR / "train_metadata.csv"
)

val_meta = pd.read_csv(
    PARTITION_DIR / "validation_metadata.csv"
)

test_meta = pd.read_csv(
    PARTITION_DIR / "test_metadata.csv"
)

print("\nPartition metadata loaded:")
print(f"Train      : {len(train_meta):,} windows")
print(f"Validation : {len(val_meta):,} windows")
print(f"Test       : {len(test_meta):,} windows")


# ============================================================
# REQUIRED COLUMNS CHECK
# ============================================================

required_columns = [
    "subject_id",
    "edf_id",
    "window_index",
    "label",
]

for column in required_columns:
    if column not in train_meta.columns:
        raise ValueError(
            f"Required column '{column}' not found in train metadata."
        )

    if column not in val_meta.columns:
        raise ValueError(
            f"Required column '{column}' not found in validation metadata."
        )

    if column not in test_meta.columns:
        raise ValueError(
            f"Required column '{column}' not found in test metadata."
        )


# ============================================================
# LOAD SUBJECT ARRAYS
# ============================================================

all_subjects = sorted(
    set(train_meta["subject_id"])
    | set(val_meta["subject_id"])
    | set(test_meta["subject_id"])
)

print("\nLoading subject arrays...")

windows_sources = {}
labels_sources = {}
file_ids_sources = {}

for subject in all_subjects:

    windows_path = DATA_DIR / f"{subject}_windows.npy"
    labels_path = DATA_DIR / f"{subject}_labels.npy"
    file_ids_path = DATA_DIR / f"{subject}_file_ids.npy"

    if not windows_path.exists():
        raise FileNotFoundError(windows_path)

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    if not file_ids_path.exists():
        raise FileNotFoundError(file_ids_path)

    windows = np.load(
        windows_path,
        mmap_mode="r"
    )

    labels = np.load(labels_path)

    # IMPORTANT:
    # EDF IDs are strings such as "chb01_01".
    # Convert the entire array to strings once so that
    # metadata and array EDF IDs use the same representation.
    file_ids = np.asarray(
        np.load(file_ids_path)
    ).astype(str)

    if len(windows) != len(labels):
        raise ValueError(
            f"{subject}: windows and labels have different lengths."
        )

    if len(windows) != len(file_ids):
        raise ValueError(
            f"{subject}: windows and file_ids have different lengths."
        )

    windows_sources[subject] = windows
    labels_sources[subject] = labels
    file_ids_sources[subject] = file_ids

    print(
        f"  {subject}: "
        f"{len(windows):,} windows | "
        f"{int(labels.sum()):,} seizure | "
        f"{len(np.unique(file_ids))} EDFs"
    )


# ============================================================
# BUILD EXACT METADATA -> ARRAY MAPPING
# ============================================================

def metadata_to_indices(metadata, split_name):

    indices = []
    labels = []
    mismatches = 0

    for row in metadata.itertuples(index=False):

        subject = str(row.subject_id)
        edf_id = str(row.edf_id)
        window_index = int(row.window_index)
        metadata_label = int(row.label)

        if subject not in windows_sources:
            raise ValueError(
                f"{split_name}: subject {subject} not loaded."
            )

        subject_file_ids = file_ids_sources[subject]

        # ----------------------------------------------------
        # Verify that this EDF actually exists for the subject.
        # EDF IDs are strings such as "chb01_01".
        # ----------------------------------------------------

        matching_indices = np.where(
            subject_file_ids == edf_id
        )[0]

        if len(matching_indices) == 0:
            raise ValueError(
                f"{split_name}: EDF {edf_id} not found for {subject}."
            )

        # ----------------------------------------------------
        # First interpretation:
        # window_index is a global subject-level index.
        # This is the format used by our partition metadata.
        # ----------------------------------------------------

        global_index = window_index

        if global_index < 0:
            raise ValueError(
                f"Invalid window index: {global_index}"
            )

        if global_index >= len(labels_sources[subject]):
            raise ValueError(
                f"{split_name}: window index {global_index} "
                f"out of range for {subject}."
            )

        actual_edf_id = str(
            subject_file_ids[global_index]
        )

        # ----------------------------------------------------
        # If the global interpretation does not match the EDF,
        # fall back to treating window_index as the local
        # position inside that EDF.
        # ----------------------------------------------------

        if actual_edf_id != edf_id:

            edf_indices = np.where(
                subject_file_ids == edf_id
            )[0]

            if window_index >= len(edf_indices):
                raise ValueError(
                    f"{split_name}: cannot map "
                    f"{subject}, EDF {edf_id}, "
                    f"window {window_index}"
                )

            global_index = int(
                edf_indices[window_index]
            )

            actual_edf_id = str(
                subject_file_ids[global_index]
            )

            if actual_edf_id != edf_id:
                raise ValueError(
                    f"{split_name}: EDF mapping failed for "
                    f"{subject}, EDF {edf_id}, "
                    f"window {window_index}"
                )

        # ----------------------------------------------------
        # Verify label.
        # This is critical: metadata label must exactly match
        # the label stored with the EEG window.
        # ----------------------------------------------------

        actual_label = int(
            labels_sources[subject][global_index]
        )

        if actual_label != metadata_label:

            mismatches += 1

            if mismatches <= 10:
                print(
                    "\nWARNING label mismatch:"
                    f"\n  split       = {split_name}"
                    f"\n  subject     = {subject}"
                    f"\n  edf_id      = {edf_id}"
                    f"\n  window      = {window_index}"
                    f"\n  metadata    = {metadata_label}"
                    f"\n  actual      = {actual_label}"
                )

        indices.append(
            (subject, global_index)
        )

        labels.append(actual_label)

    if mismatches > 0:
        raise ValueError(
            f"\nSTOPPING: {mismatches} label mismatches "
            f"found in {split_name}."
        )

    return (
        indices,
        np.asarray(labels, dtype=np.float32)
    )


# ============================================================
# CREATE DATASET INDICES
# ============================================================

print("\n")
print("=" * 90)
print("VERIFYING METADATA -> EEG WINDOW MAPPING")
print("=" * 90)

train_indices, y_train = metadata_to_indices(
    train_meta,
    "TRAIN"
)

val_indices, y_val = metadata_to_indices(
    val_meta,
    "VALIDATION"
)

test_indices, y_test = metadata_to_indices(
    test_meta,
    "TEST"
)


# ============================================================
# HARD EXPECTED LABEL CHECK
# ============================================================

EXPECTED_TRAIN_SEIZURES = 3005
EXPECTED_VAL_SEIZURES = 634
EXPECTED_TEST_SEIZURES = 611
EXPECTED_TOTAL_SEIZURES = 4250

print("\n")
print("=" * 90)
print("PARTITION LABEL VERIFICATION")
print("=" * 90)

print(
    f"\nTrain:"
    f"\n  Windows : {len(y_train):,}"
    f"\n  Seizure : {int(y_train.sum()):,}"
)

print(
    f"\nValidation:"
    f"\n  Windows : {len(y_val):,}"
    f"\n  Seizure : {int(y_val.sum()):,}"
)

print(
    f"\nTest:"
    f"\n  Windows : {len(y_test):,}"
    f"\n  Seizure : {int(y_test.sum()):,}"
)

if int(y_train.sum()) != EXPECTED_TRAIN_SEIZURES:
    raise ValueError(
        "TRAIN seizure count is incorrect. Expected 3005."
    )

if int(y_val.sum()) != EXPECTED_VAL_SEIZURES:
    raise ValueError(
        "VALIDATION seizure count is incorrect. Expected 634."
    )

if int(y_test.sum()) != EXPECTED_TEST_SEIZURES:
    raise ValueError(
        "TEST seizure count is incorrect. Expected 611."
    )

total_seizures = (
    int(y_train.sum())
    + int(y_val.sum())
    + int(y_test.sum())
)

if total_seizures != EXPECTED_TOTAL_SEIZURES:
    raise ValueError(
        "TOTAL seizure count is incorrect. Expected 4250."
    )

print("\nSTATUS: PASS")
print("All partition labels match the expected counts.")
print("Safe to begin CNN training.")


# ============================================================
# DATASET
# ============================================================

class SeizureWindowDataset(Dataset):

    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        subject, global_index = self.indices[i]

        # copy=True removes the read-only memmap warning.
        x = np.array(
            windows_sources[subject][global_index],
            dtype=np.float32,
            copy=True
        )

        y = self.labels[i]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32)
        )


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = int(np.sum(y_train == 0))
positive_count = int(np.sum(y_train == 1))

print("\n")
print("=" * 90)
print("CLASS IMBALANCE")
print("=" * 90)

print(f"\nNon-seizure : {negative_count:,}")
print(f"Seizure     : {positive_count:,}")
print(
    f"Ratio       : "
    f"{negative_count / positive_count:.1f}:1"
)

class_weights = {
    0: 1.0 / negative_count,
    1: 1.0 / positive_count,
}

sample_weights = np.array([
    class_weights[int(label)]
    for label in y_train
])

sample_weights = torch.from_numpy(
    sample_weights
).double()

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = SeizureWindowDataset(
    train_indices,
    y_train
)

val_dataset = SeizureWindowDataset(
    val_indices,
    y_val
)

test_dataset = SeizureWindowDataset(
    test_indices,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# CNN MODEL
# ============================================================

n_channels = windows_sources[
    all_subjects[0]
].shape[1]

n_samples = windows_sources[
    all_subjects[0]
].shape[2]

print("\n")
print("=" * 90)
print("MODEL")
print("=" * 90)

print(f"\nInput channels : {n_channels}")
print(f"Samples/window : {n_samples}")


class SeizureCNN(nn.Module):

    def __init__(self, n_channels):

        super().__init__()

        self.conv1 = nn.Conv1d(
            n_channels,
            32,
            kernel_size=7,
            padding="same"
        )

        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=5,
            padding="same"
        )

        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(
            64,
            128,
            kernel_size=3,
            padding="same"
        )

        self.bn3 = nn.BatchNorm1d(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool1(
            self.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            self.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.relu(
            self.bn3(
                self.conv3(x)
            )
        )

        x = self.global_pool(x).squeeze(-1)

        x = self.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        return self.fc2(x)


model = SeizureCNN(
    n_channels
).to(device)

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device,
                non_blocking=True
            )

            logits = model(
                X_batch
            ).squeeze(-1)

            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )

            labels.append(
                y_batch.numpy()
            )

    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    if len(np.unique(labels)) == 2:

        roc_auc = roc_auc_score(
            labels,
            probabilities
        )

        pr_auc = average_precision_score(
            labels,
            probabilities
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
    }


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 90)
print("TRAINING")
print("=" * 90)

best_val_f1 = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device,
            non_blocking=True
        )

        y_batch = y_batch.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            X_batch
        ).squeeze(-1)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * X_batch.size(0)
        )

    avg_loss = (
        total_loss /
        len(train_dataset)
    )

    # --------------------------------------------------------
    # VALIDATION ONLY
    # --------------------------------------------------------

    val_results = evaluate(
        model,
        val_loader
    )

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"  Loss        : {avg_loss:.4f}")
    print(f"  Val F1      : {val_results['f1']:.4f}")
    print(f"  Val Recall  : {val_results['recall']:.4f}")
    print(f"  Val Prec.   : {val_results['precision']:.4f}")
    print(f"  Val ROC-AUC : {val_results['roc_auc']:.4f}")
    print(f"  Val PR-AUC  : {val_results['pr_auc']:.4f}")

    # --------------------------------------------------------
    # SAVE BEST MODEL USING VALIDATION ONLY
    # --------------------------------------------------------

    if val_results["f1"] > best_val_f1:

        best_val_f1 = val_results["f1"]

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        print(
            f"  -> New best model "
            f"(validation F1 = {best_val_f1:.4f})"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(best_state)

    print(
        f"\nRestored best model "
        f"(validation F1 = {best_val_f1:.4f})"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 90)
print("FINAL TEST SET EVALUATION")
print("=" * 90)

test_results = evaluate(
    model,
    test_loader
)

print(f"\nPrecision : {test_results['precision']:.4f}")
print(f"Recall    : {test_results['recall']:.4f}")
print(f"F1-score  : {test_results['f1']:.4f}")
print(f"ROC-AUC   : {test_results['roc_auc']:.4f}")
print(f"PR-AUC    : {test_results['pr_auc']:.4f}")

print("\nClassification report:")

print(
    classification_report(
        test_results["labels"],
        test_results["predictions"],
        target_names=[
            "Non-seizure",
            "Seizure"
        ],
        zero_division=0
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        test_results["labels"],
        test_results["predictions"]
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "baseline_cnn_subjectwise.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

print(
    f"\nModel saved to:\n{model_path}"
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 90)
print("TRAINING COMPLETE")
print("=" * 90)
