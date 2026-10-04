import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

PARTITION_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "partitions"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "baseline_cnn_subjectwise.pt"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "shap"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Keep this small initially because the GPU has 6 GB VRAM.
N_BACKGROUND = 50
N_SEIZURE = 10


# ============================================================
# MODEL
# EXACT ARCHITECTURE USED BY THE TRAINED BASELINE CNN
# ============================================================

class BaselineCNN(nn.Module):

    def __init__(self, input_channels=23):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_channels,
            32,
            kernel_size=7,
            padding=3
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        # Global average pooling
        x = self.global_pool(x)

        x = x.squeeze(-1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SHAP EXPLAINABILITY FOR BASELINE CNN")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading model...")

model = BaselineCNN(
    input_channels=23
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle either a raw state_dict or checkpoint dictionary
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD TEST METADATA
# ============================================================

print("\nLoading test metadata...")

test_metadata_path = os.path.join(
    PARTITION_DIR,
    "test_metadata.csv"
)

test_metadata = pd.read_csv(
    test_metadata_path
)

print(
    f"Test windows: "
    f"{len(test_metadata):,}"
)

print(
    f"Test seizure windows: "
    f"{int(test_metadata['label'].sum()):,}"
)


# ============================================================
# LOAD SUBJECT DATA
# ============================================================

print("\nLoading required subject data...")

subjects = sorted(
    test_metadata[
        "subject_id"
    ].unique()
)

print(
    "Subjects:",
    ", ".join(subjects)
)

subject_data = {}


for subject in subjects:

    windows_path = os.path.join(
        DATA_DIR,
        f"{subject}_windows.npy"
    )

    file_ids_path = os.path.join(
        DATA_DIR,
        f"{subject}_file_ids.npy"
    )

    print(
        f"\nLoading {subject}..."
    )

    windows = np.load(
        windows_path,
        mmap_mode="r"
    )

    file_ids = np.load(
        file_ids_path,
        mmap_mode="r"
    )

    subject_data[subject] = {
        "windows": windows,
        "file_ids": file_ids
    }

    print(
        f"  windows: {windows.shape}"
    )

    print(
        f"  file IDs: {file_ids.shape}"
    )


# ============================================================
# BUILD EDF MAPPING
#
# Important:
# EDF filename numbers cannot directly be used as processed
# file IDs because some original EDF files were skipped.
#
# Therefore we reproduce the same positional mapping used in
# the validated training pipeline.
# ============================================================

print(
    "\nBuilding EDF mapping..."
)

edf_name_to_number = {}


for subject in subjects:

    subject_metadata = test_metadata[
        test_metadata["subject_id"] == subject
    ]

    metadata_edfs = sorted(
        subject_metadata[
            "edf_id"
        ].unique()
    )

    actual_file_ids = subject_data[
        subject
    ]["file_ids"]

    processed_edf_numbers = sorted(
        np.unique(
            actual_file_ids
        ).tolist()
    )

    if (
        len(metadata_edfs)
        != len(processed_edf_numbers)
    ):

        raise RuntimeError(
            f"{subject}: EDF count mismatch. "
            f"Metadata={len(metadata_edfs)}, "
            f"processed={len(processed_edf_numbers)}"
        )

    edf_name_to_number[subject] = {
        edf_name: edf_number
        for edf_name, edf_number
        in zip(
            metadata_edfs,
            processed_edf_numbers
        )
    }

    print(
        f"{subject}: "
        f"{len(metadata_edfs)} EDFs mapped"
    )


# ============================================================
# GET ACTUAL EEG WINDOW
# ============================================================

def get_window_from_metadata(row):

    subject = row[
        "subject_id"
    ]

    edf_id = row[
        "edf_id"
    ]

    window_index = int(
        row[
            "window_index"
        ]
    )

    subject_windows = subject_data[
        subject
    ]["windows"]

    subject_file_ids = subject_data[
        subject
    ]["file_ids"]

    if (
        edf_id
        not in edf_name_to_number[
            subject
        ]
    ):

        raise RuntimeError(
            f"EDF {edf_id} not found "
            f"for {subject}"
        )

    edf_number = (
        edf_name_to_number[
            subject
        ][edf_id]
    )

    positions = np.where(
        subject_file_ids
        == edf_number
    )[0]

    if (
        window_index
        >= len(positions)
    ):

        raise RuntimeError(
            f"Invalid window index for "
            f"{subject} {edf_id}: "
            f"{window_index}"
        )

    global_index = positions[
        window_index
    ]

    # Verify EDF mapping
    actual_edf_number = int(
        subject_file_ids[
            global_index
        ]
    )

    if (
        actual_edf_number
        != edf_number
    ):

        raise RuntimeError(
            "EDF mapping verification failed."
        )

    window = np.array(
        subject_windows[
            global_index
        ],
        dtype=np.float32
    )

    # Verify shape
    if window.shape != (
        23,
        512
    ):

        raise RuntimeError(
            f"Unexpected window shape: "
            f"{window.shape}"
        )

    return window


# ============================================================
# SELECT SEIZURE WINDOWS
# ============================================================

print(
    "\nSelecting seizure windows..."
)

seizure_metadata = test_metadata[
    test_metadata[
        "label"
    ] == 1
].copy()

if (
    len(seizure_metadata)
    < N_SEIZURE
):

    raise RuntimeError(
        "Not enough seizure windows."
    )


# Fixed random state gives reproducible SHAP results
selected_seizures = (
    seizure_metadata.sample(
        n=N_SEIZURE,
        random_state=42
    )
)


seizure_windows = []


for _, row in selected_seizures.iterrows():

    window = (
        get_window_from_metadata(
            row
        )
    )

    # Verify metadata label
    if int(row["label"]) != 1:

        raise RuntimeError(
            "Selected window is not a seizure window."
        )

    seizure_windows.append(
        window
    )


seizure_windows = np.stack(
    seizure_windows
)


print(
    f"Selected seizure windows: "
    f"{seizure_windows.shape}"
)


# ============================================================
# SELECT BACKGROUND WINDOWS
# ============================================================

print(
    "\nSelecting background windows..."
)

nonseizure_metadata = test_metadata[
    test_metadata[
        "label"
    ] == 0
].copy()


if (
    len(nonseizure_metadata)
    < N_BACKGROUND
):

    raise RuntimeError(
        "Not enough non-seizure windows."
    )


background_metadata = (
    nonseizure_metadata.sample(
        n=N_BACKGROUND,
        random_state=42
    )
)


background_windows = []


for _, row in background_metadata.iterrows():

    window = (
        get_window_from_metadata(
            row
        )
    )

    if int(row["label"]) != 0:

        raise RuntimeError(
            "Background window has seizure label."
        )

    background_windows.append(
        window
    )


background_windows = np.stack(
    background_windows
)


print(
    f"Background windows: "
    f"{background_windows.shape}"
)


# ============================================================
# CONVERT TO TORCH
# ============================================================

background_tensor = torch.from_numpy(
    background_windows.copy()
).float().to(DEVICE)


seizure_tensor = torch.from_numpy(
    seizure_windows.copy()
).float().to(DEVICE)


print(
    "\nTensor shapes:"
)

print(
    f"Background: {tuple(background_tensor.shape)}"
)

print(
    f"Seizure:    {tuple(seizure_tensor.shape)}"
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print(
    "\nChecking model predictions..."
)

with torch.no_grad():

    logits = model(
        seizure_tensor
    ).squeeze(1)

    probabilities = torch.sigmoid(
        logits
    )


predictions = (
    probabilities
    .cpu()
    .numpy()
)


for i, probability in enumerate(
    predictions
):

    print(
        f"Seizure window {i + 1:02d}: "
        f"prediction probability = "
        f"{probability:.4f}"
    )


# ============================================================
# SHAP GRADIENT EXPLAINER
# ============================================================

print(
    "\nCreating SHAP GradientExplainer..."
)

explainer = shap.GradientExplainer(
    model,
    background_tensor
)


print(
    "Calculating SHAP values..."
)

print(
    "This may take some time."
)


shap_values = explainer.shap_values(
    seizure_tensor
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(
    shap_values,
    list
):

    shap_values = shap_values[0]


shap_values = np.asarray(
    shap_values
)


print(
    "\nRaw SHAP shape:",
    shap_values.shape
)


# ============================================================
# HANDLE POSSIBLE EXTRA OUTPUT DIMENSION
# ============================================================

if shap_values.ndim == 4:

    # Possible shape:
    # (samples, channels, samples, 1)

    if (
        shap_values.shape[-1]
        == 1
    ):

        shap_values = (
            shap_values[..., 0]
        )

    # Possible shape:
    # (samples, 1, channels, samples)

    elif (
        shap_values.shape[1]
        == 1
    ):

        shap_values = (
            shap_values[:, 0, :, :]
        )


# ============================================================
# FINAL SHAP VALIDATION
# ============================================================

if (
    shap_values.shape
    != seizure_windows.shape
):

    raise RuntimeError(
        f"Unexpected SHAP shape: "
        f"{shap_values.shape}. "
        f"Expected: "
        f"{seizure_windows.shape}"
    )


print(
    "Final SHAP shape:",
    shap_values.shape
)


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

shap_output_path = os.path.join(
    RESULTS_DIR,
    "seizure_shap_values.npy"
)

np.save(
    shap_output_path,
    shap_values
)


print(
    f"\nSaved SHAP values:"
)

print(
    shap_output_path
)


# ============================================================
# EEG CHANNEL NAMES
#
# These correspond to the 23-channel CHB-MIT bipolar
# montage used by the preprocessing pipeline.
# ============================================================

channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "FP1-FP2",
    "AF1-AF2",
    "F7-F3",
    "F8-F4",
    "T7-T8"
]


# ============================================================
# GLOBAL CHANNEL IMPORTANCE
# ============================================================

print(
    "\nCalculating channel importance..."
)


# Average absolute SHAP value:
#
# samples × channels × time
#
# average over samples and time
channel_importance = np.mean(
    np.abs(shap_values),
    axis=(0, 2)
)


channel_ranking = pd.DataFrame({

    "channel": channels,

    "mean_abs_shap":
        channel_importance

})


channel_ranking = (
    channel_ranking
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


channel_csv = os.path.join(
    RESULTS_DIR,
    "channel_importance.csv"
)


channel_ranking.to_csv(
    channel_csv,
    index=False
)


print(
    "\nTop EEG channels:"
)

print(
    channel_ranking
    .head(10)
    .to_string(index=False)
)


# ============================================================
# CHANNEL IMPORTANCE BAR PLOT
# ============================================================

plt.figure(
    figsize=(10, 7)
)


top_channels = (
    channel_ranking
    .head(15)
)


plt.barh(
    top_channels[
        "channel"
    ][::-1],

    top_channels[
        "mean_abs_shap"
    ][::-1]
)


plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "EEG Channel"
)

plt.title(
    "SHAP Feature Importance by EEG Channel"
)


plt.tight_layout()


channel_plot = os.path.join(
    RESULTS_DIR,
    "shap_channel_importance.png"
)


plt.savefig(
    channel_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nSaved channel importance plot:"
)

print(
    channel_plot
)


# ============================================================
# SHAP HEATMAP
# ============================================================

print(
    "\nCreating SHAP heatmap..."
)


# Explain the first selected seizure window
sample_index = 0


sample_shap = (
    shap_values[
        sample_index
    ]
)


plt.figure(
    figsize=(16, 8)
)


plt.imshow(
    sample_shap,
    aspect="auto",
    interpolation="nearest"
)


plt.colorbar(
    label="SHAP value"
)


plt.yticks(
    np.arange(
        len(channels)
    ),
    channels
)


plt.xlabel(
    "Time Samples"
)

plt.ylabel(
    "EEG Channel"
)

plt.title(
    "SHAP Explanation for a Seizure EEG Window"
)


plt.tight_layout()


heatmap_path = os.path.join(
    RESULTS_DIR,
    "shap_seizure_heatmap.png"
)


plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"Saved SHAP heatmap:"
)

print(
    heatmap_path
)


# ============================================================
# SAVE EXPLAINED WINDOW INFORMATION
# ============================================================

selected_info = selected_seizures[
    [
        "subject_id",
        "edf_id",
        "window_index",
        "start_sec",
        "end_sec",
        "label"
    ]
].copy()


selected_info[
    "model_probability"
] = predictions


selected_info_path = os.path.join(
    RESULTS_DIR,
    "explained_seizure_windows.csv"
)


selected_info.to_csv(
    selected_info_path,
    index=False
)


print(
    f"\nSaved explained window information:"
)

print(
    selected_info_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SHAP ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    f"Explained seizure windows: "
    f"{N_SEIZURE}"
)

print(
    f"Background windows: "
    f"{N_BACKGROUND}"
)

print(
    f"SHAP values shape: "
    f"{shap_values.shape}"
)


print(
    "\nResults saved in:"
)

print(
    RESULTS_DIR
)


print(
    "\nGenerated files:"
)

print(
    "1. seizure_shap_values.npy"
)

print(
    "2. channel_importance.csv"
)

print(
    "3. shap_channel_importance.png"
)

print(
    "4. shap_seizure_heatmap.png"
)

print(
    "5. explained_seizure_windows.csv"
)

print(
    "\nDone."
)