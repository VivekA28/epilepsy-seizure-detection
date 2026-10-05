"""Train/evaluate local CNN + LSTM and export timestamped predictions."""

import argparse
import copy
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from prediction_io import write_sequence_predictions
from sequence_dataset import EEGSequenceDataset, build_sequence_records


DATA_DIR = Path("data/processed")
FEATURE_DIR = DATA_DIR / "cnn_features"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
SEQUENCE_LENGTH = 5
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
CNN_FEATURE_DIM = 128
LSTM_HIDDEN_SIZE = 128


class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim=CNN_FEATURE_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


def split_files(subjects):
    """Create the existing LOCAL file-level split for development."""
    file_arrays = []
    label_arrays = []

    for subject in subjects:
        labels = np.load(
            DATA_DIR / f"{subject}_labels.npy",
            mmap_mode="r",
        )
        file_ids = np.load(
            DATA_DIR / f"{subject}_file_ids.npy",
            mmap_mode="r",
        )

        if len(labels) != len(file_ids):
            raise ValueError(f"Label/file-id mismatch for {subject}")

        file_arrays.append(file_ids)
        label_arrays.append(labels)

    next_file_id = 0
    global_file_ids = []
    global_labels = []

    for file_ids, labels in zip(file_arrays, label_arrays):
        shifted = np.asarray(file_ids, dtype=np.int64) + next_file_id
        global_file_ids.append(shifted)
        global_labels.append(np.asarray(labels))
        next_file_id = int(shifted.max()) + 1

    all_file_ids = np.concatenate(global_file_ids)
    all_labels = np.concatenate(global_labels)

    unique_files = np.unique(all_file_ids)
    file_has_seizure = np.array(
        [all_labels[all_file_ids == file_id].max() for file_id in unique_files],
        dtype=np.int8,
    )

    n_seizure_files = int(file_has_seizure.sum())
    n_clean_files = int(len(file_has_seizure) - n_seizure_files)

    if n_seizure_files < 2 or n_clean_files < 2:
        raise RuntimeError("Not enough seizure/clean EDFs for the split.")

    train_files, temp_files, _, temp_file_labels = train_test_split(
        unique_files,
        file_has_seizure,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=file_has_seizure,
    )

    temp_seizure_files = int(temp_file_labels.sum())
    temp_clean_files = int(len(temp_file_labels) - temp_seizure_files)

    if temp_seizure_files < 2 or temp_clean_files < 2:
        raise RuntimeError(
            "Not enough temporary EDFs for stratified val/test split."
        )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION),
        random_state=RANDOM_STATE,
        stratify=temp_file_labels,
    )

    return (
        file_arrays,
        label_arrays,
        global_file_ids,
        train_files,
        val_files,
        test_files,
    )


def make_datasets(subjects, sequence_length):
    (
        file_arrays,
        label_arrays,
        global_file_ids,
        train_files,
        val_files,
        test_files,
    ) = split_files(subjects)

    feature_arrays = [
        np.load(
            FEATURE_DIR / f"{subject}_cnn_features.npy",
            mmap_mode="r",
        )
        for subject in subjects
    ]

    for subject, features, labels in zip(
        subjects,
        feature_arrays,
        label_arrays,
    ):
        if (
            features.shape[0] != len(labels)
            or features.shape[1] != CNN_FEATURE_DIM
        ):
            raise ValueError(
                f"Bad feature cache for {subject}: {features.shape}"
            )

    partitions = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    datasets = {}
    records_by_partition = {}

    for partition, allowed_files in partitions.items():
        records = []

        for subject_index, file_ids in enumerate(global_file_ids):
            records.extend(
                build_sequence_records(
                    np.asarray(file_ids),
                    allowed_files,
                    sequence_length,
                    subject_index,
                )
            )

        records_by_partition[partition] = records
        datasets[partition] = EEGSequenceDataset(
            feature_arrays,
            label_arrays,
            records,
            sequence_length,
        )

    print("\nSequence counts:")
    for partition, records in records_by_partition.items():
        positive = sum(
            int(
                label_arrays[r.subject_index][
                    r.start + sequence_length - 1
                ]
            )
            for r in records
        )
        print(
            f"  {partition:5s}: {len(records)} sequences "
            f"({positive} seizure targets)"
        )

    return datasets, records_by_partition


def make_train_sampler(dataset):
    target_labels = np.fromiter(
        (
            dataset.labels_arrays[r.subject_index][
                r.start + dataset.sequence_length - 1
            ]
            for r in dataset.records
        ),
        dtype=np.int8,
        count=len(dataset.records),
    )

    class_counts = np.bincount(
        target_labels,
        minlength=2,
    ).astype(np.float64)

    if np.any(class_counts == 0):
        raise RuntimeError(
            "Training sequences contain only one class."
        )

    class_weights = 1.0 / class_counts
    weights = torch.as_tensor(
        class_weights[target_labels],
        dtype=torch.double,
    )

    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )


def evaluate(model, loader, device):
    """Return binary predictions, targets, and sigmoid probabilities."""
    model.eval()
    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits)

            probabilities.append(
                probs.cpu().numpy()
            )
            predictions.append(
                (probs > 0.5).int().cpu().numpy()
            )
            targets.append(
                y_batch.numpy().astype(np.int64)
            )

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def save_prediction_file(
    subjects,
    records,
    y_true,
    y_pred,
    y_prob,
    sequence_length,
    name,
):
    output_path = RESULTS_DIR / name

    write_sequence_predictions(
        subjects=subjects,
        records=records,
        sequence_length=sequence_length,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        output_path=output_path,
        data_dir=DATA_DIR,
    )

    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "subjects",
        nargs="+",
        help="Subjects, e.g. chb01 chb02 ...",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    datasets, records_by_partition = make_datasets(
        args.subjects,
        args.sequence_length,
    )

    train_sampler = make_train_sampler(
        datasets["train"]
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNNLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"\nUsing device: {device}")
    print(f"Sequence length: {args.sequence_length} windows")
    print(f"CNN feature dimension: {CNN_FEATURE_DIM}")
    print(f"LSTM hidden size: {LSTM_HIDDEN_SIZE}")

    best_val_f1 = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * X_batch.size(0)
            )

        avg_loss = total_loss / len(datasets["train"])

        val_preds, val_targets, _ = evaluate(
            model,
            val_loader,
            device,
        )

        val_f1 = f1_score(
            val_targets,
            val_preds,
            zero_division=0,
        )

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
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"val recall: {val_recall:.2f} - "
            f"val precision: {val_precision:.2f} - "
            f"val f1: {val_f1:.2f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            print(
                f"  -> new best validation checkpoint "
                f"(f1={best_val_f1:.2f})"
            )

    if best_state is None:
        raise RuntimeError(
            "No validation checkpoint was produced."
        )

    model.load_state_dict(best_state)

    print(
        f"\nRestored best checkpoint from epoch "
        f"{best_epoch} "
        f"(validation f1={best_val_f1:.2f})"
    )

    # Save validation predictions too. They can be used later for
    # threshold analysis without touching the test set.
    val_preds, val_targets, val_probs = evaluate(
        model,
        val_loader,
        device,
    )

    val_prediction_path = save_prediction_file(
        args.subjects,
        records_by_partition["val"],
        val_targets,
        val_preds,
        val_probs,
        args.sequence_length,
        f"cnn_lstm_{'_'.join(args.subjects)}_val_predictions.csv",
    )

    print(
        "\nEvaluating TEST set once using the "
        "validation-selected checkpoint..."
    )

    test_preds, test_targets, test_probs = evaluate(
        model,
        test_loader,
        device,
    )

    test_prediction_path = save_prediction_file(
        args.subjects,
        records_by_partition["test"],
        test_targets,
        test_preds,
        test_probs,
        args.sequence_length,
        f"cnn_lstm_{'_'.join(args.subjects)}_test_predictions.csv",
    )

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)

    print(
        classification_report(
            test_targets,
            test_preds,
            target_names=[
                "Non-seizure",
                "Seizure",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            test_targets,
            test_preds,
        )
    )

    MODEL_DIR.mkdir(exist_ok=True)

    model_path = (
        MODEL_DIR
        / f"cnn_lstm_frozen_cnn_{'_'.join(args.subjects)}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sequence_length": args.sequence_length,
            "cnn_feature_dim": CNN_FEATURE_DIM,
            "lstm_hidden_size": LSTM_HIDDEN_SIZE,
            "subjects": args.subjects,
            "cnn_checkpoint": (
                f"baseline_cnn_{'_'.join(args.subjects)}.pt"
            ),
            "prediction_threshold": 0.5,
            "validation_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
        model_path,
    )

    print(f"\nModel saved to {model_path}")
    print(f"Validation predictions: {val_prediction_path}")
    print(f"Test predictions: {test_prediction_path}")


if __name__ == "__main__":
    main()
