# Progress Log — Early Epilepsy Seizure Detection

Running log of what's done and what's next. **This is the source of truth.**

---

## Project Goal

Build and evaluate an explainable early seizure-detection pipeline using scalp EEG.

The project is intentionally focused on the supervisor's requested extensions:

- more EEG data
- at least one additional EEG dataset
- FFT / frequency-domain information
- an LSTM / temporal model
- larger training on the university GPU
- SHAP/LIME explainability
- rigorous evaluation without overengineering the architecture

Do **not** add Transformers, attention, BiLSTM, or large ensembles unless a later experiment provides a concrete reason.

---

# 1. Completed Work

## Repository and Project Setup

- [x] Repo created and pushed to GitHub: `VivekA28/epilepsy-seizure-detection`
- [x] Folder scaffolding set up:
  - `data/`
  - `notebooks/`
  - `src/`
  - `models/`
  - `results/`
  - `scripts/`
- [x] `.gitignore` configured to exclude large/raw/model files including:
  - `data/`
  - `models/`
  - `*.edf`
  - `*.pt`
  - `*.h5`
  - `*.pth`
  - `__pycache__/`
  - `venv/`
- [x] `README.md` added with attribution note.
- [x] SSH key authentication configured for project contributors.
- [x] `scripts/download_data.sh` created for dataset download.
- [x] Faster PhysioNet public S3 download path identified and used.

## Initial CHB-MIT Work

- [x] MNE walkthrough completed:
  - EDF loading
  - channel inspection
  - sampling-rate inspection
  - signal plotting
  - seizure annotation parsing
- [x] Memory-safe EEG preprocessing pipeline implemented.
- [x] CHB-MIT preprocessing finalized:
  - 256 Hz → 128 Hz
  - 60 Hz notch filter
  - 0.5–40 Hz bandpass
  - per-channel z-score normalization
  - 4-second windows
  - 50% overlap
- [x] CHB01 processed:
  - 72,951 windows
  - 230 seizure windows
- [x] Window-overlap leakage issue identified and file-level tracking added.
- [x] Existing baseline CNN trained using file-level separation.

## Five-Subject CHB-MIT Dataset

- [x] CHB01–CHB05 processed locally.
- [x] Memory-safe processing used for the expanded dataset.
- [x] Metadata preserved for every processed window.
- [x] Channel names preserved in separate channel-name files.
- [x] Dataset contains 315,686 total windows.
- [x] Dataset contains 843 seizure windows.

### CHB-MIT Dataset Summary

| Subject | Windows | Seizure Windows | Non-Seizure Windows |
|---|---:|---:|---:|
| CHB01 | 72,951 | 230 | 72,721 |
| CHB02 | 63,443 | 90 | 63,353 |
| CHB03 | 68,365 | 210 | 68,155 |
| CHB04 | 40,761 | 26 | 40,735 |
| CHB05 | 70,166 | 287 | 69,879 |
| **Total** | **315,686** | **843** | **314,843** |

---

# 2. Existing CNN Baseline

## Baseline Results

The existing normalized five-subject CNN produced:

- Precision: 0.85
- Recall: 0.89
- F1: 0.87
- Confusion matrix:
  - TN = 89,257
  - FP = 32
  - FN = 22
  - TP = 178

Earlier pre-normalization result:

- Precision ≈ 0.94
- Recall ≈ 0.83
- F1 ≈ 0.88

The normalization experiment produced a similar F1 while changing the precision/recall balance.

## Explainability

- [x] SHAP explainability implemented with `GradientExplainer`.
- [x] SHAP produces input-specific channel importance.
- [x] Custom LIME time-series explainer implemented in `explain_lime.py`.
- [x] LIME divides the input into channel × temporal regions and fits a local surrogate model.
- [x] SHAP/LIME findings reviewed.
- [x] Important channel behaviour varies between windows.
- [x] Repeated FT9-FT10 importance was observed in some seizure windows.
- [x] These observations are treated as model behaviour and **not proof of physiological causality or artifact learning**.
- [x] Project presentation deck created.

---

# 3. Important Methodological Limitation of the Old Baseline

The old training script evaluated the test set every epoch and selected the best checkpoint using test F1.

This is **test-set snooping** and must not be repeated in new experiments.

The corrected protocol must use:

```text
TRAIN
   ↓
weights update

VALIDATION
   ↓
checkpoint/model selection

TEST
   ↓
final evaluation only
