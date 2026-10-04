# Final Results — Early Epilepsy Seizure Detection

## Dataset

- Total EEG windows: 1,041,166
- Seizure windows: 4,250
- Subjects: 15

## Preprocessing

- Sampling rate: 128 Hz
- Bandpass filter: 0.5–40 Hz
- Notch filter: 60 Hz
- Per-channel Z-score normalization
- Window length: 4 seconds
- Window overlap: 50%

## Subject-wise Data Split

- Training: 10 subjects
- Validation: 2 subjects
- Testing: 3 subjects

No subject was shared between train, validation, and test partitions.

## CNN Baseline

- Input: 23 EEG channels × 512 samples
- Convolutional blocks: 3
- Batch Normalization: Yes
- Max Pooling: Yes
- Global Average Pooling: Yes
- Fully Connected Classifier: Yes

## Test Results

- Precision: 0.2011
- Recall: 0.3044
- F1-score: 0.2422
- ROC-AUC: 0.6678
- PR-AUC: 0.2488

## Explainability — SHAP

- Explainer: SHAP GradientExplainer
- Seizure windows explained: 10
- Background windows: 50
- Input explanation size: 23 × 512
- EEG channels analyzed: 23

### Top SHAP Channels

1. F8-T8
2. F4-C4
3. F8-F4
4. T8-P8
5. FP2-F8
6. FP2-F4
7. P8-O2
8. P4-O2
9. F7-F3
10. FP1-F7

## SHAP Output Files

- `results/shap/seizure_shap_values.npy`
- `results/shap/channel_importance.csv`
- `results/shap/shap_channel_importance.png`
- `results/shap/shap_seizure_heatmap.png`
- `results/shap/explained_seizure_windows.csv`