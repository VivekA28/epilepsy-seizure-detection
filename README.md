# Early Epilepsy Seizure Detection & Explainability Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, patient-independent deep learning pipeline for early epileptic seizure detection on continuous scalp EEG, paired with a clinical explainability layer (**SHAP** and **LIME**) to translate black-box neural network predictions into interpretable channel and temporal attributions.

---

## Highlights

* **Patient-Independent Evaluation (Protocol v2)**: Strict subject-level split across 15 CHB-MIT subjects (10 Train, 2 Validation, 3 Test) with zero patient or file leakage.
* **Controlled Multi-Architecture Benchmark**: Evaluates 1D-CNN baseline, multimodal CNN + Spectral FFT, temporal CNN + LSTM, and unified CNN + FFT + LSTM.
* **Clinical Interpretability Layer**: Utilizes SHAP GradientExplainer and LIME perturbation surrogates to verify anatomical channel importance (e.g., temporal/frontotemporal chains) against clinical literature.
* **Pre-trained Checkpoints Included**: All trained PyTorch models (~6 MB total) and benchmark predictions are tracked in this repository for instant reproduction without needing 50 GB of raw data.

---

## ⚡ Quickstart (Clone & Run in 3 Steps)

Anyone cloning this repository can verify the environment, load the pre-trained neural networks, and execute a forward pass immediately:

```bash
# 1. Clone the repository
git clone git@github.com:VivekA28/epilepsy-seizure-detection.git
cd epilepsy-seizure-detection

# 2. Set up virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the instant verification demo (CPU or CUDA)
python run_demo.py
```

### What `run_demo.py` does:
1. Validates all required dependencies (`torch`, `mne`, `pyedflib`, `shap`, `scikit-learn`, etc.).
2. Auto-detects hardware acceleration (CUDA GPU or CPU fallback).
3. Instantiates the 23-channel `SeizureCNN` architecture and loads `models/baseline_cnn_subjectwise_15subjects_v2.pt`.
4. Runs inference across synthetic 23-channel EEG windows with zero runtime errors.
5. Prints the benchmark comparison table and checks explainability artifacts.

---

## 📊 Benchmark Results (Protocol v2)

All models were trained on 10 subjects (`chb01`, `chb03`, `chb04`, `chb06`, `chb09`, `chb10`, `chb11`, `chb12`, `chb14`, `chb15`), tuned on 2 validation subjects (`chb07`, `chb08`), and evaluated **strictly once** on 3 held-out test subjects (`chb02`, `chb05`, `chb13`):

| Architecture | Model Checkpoint | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN baseline** | `models/baseline_cnn_subjectwise_15subjects_v2.pt` | 0.3786 | 0.6629 | **0.4620** | 0.6667 | 0.3535 |
| **CNN + FFT** | `models/cnn_fft_subjectwise_15subjects_v2.pt` | 0.2244 | **0.8160** | 0.3299 | 0.3472 | 0.3142 |
| **CNN + LSTM** | `models/cnn_lstm_subjectwise_15subjects_v2.pt` | 0.3730 | 0.8044 | 0.4542 | **0.7336** | 0.3290 |
| **CNN + FFT + LSTM** | `models/cnn_fft_lstm_subjectwise_15subjects_v2.pt` | **0.3858** | 0.7784 | **0.4665** | 0.6133 | **0.3764** |

*Raw probability arrays and metric logs are archived under [`results/`](./results/).*

---

## 🧠 Clinical Explainability (XAI)

To bridge the gap between deep learning and clinical electroencephalography, model predictions are analyzed via:

1. **SHAP (GradientExplainer)**: Attribution maps identify which electrodes drive seizure classifications. Top identified channels include **F8-T8**, **F4-C4**, **T8-P8**, and **FP2-F8**, corresponding to right temporal/frontotemporal chains commonly involved in focal seizure onsets.
2. **LIME (Local Surrogate Perturbations)**: Channel-by-time perturbation analysis validating temporal onset localization against baseline EEG activity.

*All SHAP heatmaps, channel ranking curves, and LIME perturbation plots are saved under [`results/shap/`](./results/shap/) and [`results/`](./results/).*

---

## 📁 Repository Structure

```text
epilepsy-seizure-detection/
├── run_demo.py                      # Instant verification & supervisor demo script
├── requirements.txt                 # Pinned, tested dependency requirements
├── SUPERVISOR_GUIDE.md              # Detailed walkthrough & troubleshooting guide
├── PROGRESS.md                      # Comprehensive forensic engineering changelog
├── preprocess_subject.py            # Memory-mapped EEG preprocessing (notch + bandpass + z-score)
├── fft_features.py                  # Spectral feature extraction (delta, theta, alpha, beta, gamma)
├── subject_split.py                 # Leakage-free subject-level partition logic
├── sequence_dataset.py              # Chronological sequence window builder for LSTMs
├── explain_shap.py                  # SHAP GradientExplainer implementation
├── explain_lime.py                  # LIME time-series perturbation explainer
│
├── models/                          # Pre-trained PyTorch checkpoints (~6 MB total)
│   ├── baseline_cnn_subjectwise_15subjects_v2.pt
│   ├── cnn_fft_subjectwise_15subjects_v2.pt
│   ├── cnn_lstm_subjectwise_15subjects_v2.pt
│   └── cnn_fft_lstm_subjectwise_15subjects_v2.pt
│
├── results/                         # Precomputed prediction arrays, metrics & plots
│   ├── all_models_v2_comparison.csv
│   └── shap/                        # SHAP attribution heatmaps and rankings
│
└── scripts/                         # Production training & diagnostic scripts
    ├── train_baseline_subjectwise_v2.py
    ├── train_cnn_fft_subjectwise_v2.py
    ├── train_cnn_lstm_subjectwise_v2.py
    ├── train_cnn_fft_lstm_subjectwise_v2.py
    ├── run_all_experiments_v2.py
    └── download_data.sh
```

---

## 🔬 Dataset & Preprocessing Pipeline

The pipeline uses the **CHB-MIT Scalp EEG Database** (PhysioNet):
* **Channels**: 23 bipolar montage channels (International 10–20 system).
* **Sampling Rate**: Downsampled from 256 Hz to 128 Hz.
* **Filtering**: 60 Hz notch filter (powerline removal) + 0.5–40 Hz 4th-order Butterworth bandpass filter.
* **Normalization**: Per-channel z-score standardization ($z = (x - \mu) / \sigma$).
* **Windowing**: 4-second windows (512 samples) with 50% overlap (2-second step).

### Optional: Fetching Raw Data & Retraining
If you wish to download raw EDF files from PhysioNet's open S3 mirror (no AWS account required):
```bash
# Download a specific subject (e.g., chb01)
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/chb01/ data/raw/chb01/

# Run preprocessing
python preprocess_subject.py chb01
```

---

## 📜 Attribution & References

* **CHB-MIT Database**: Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 2000.
* **Baseline Architecture Inspiration**: Adapted and extended from [mkfzdmr/Epileptic-EEG-Classfication-Using-Deep-Learning](https://github.com/mkfzdmr/Epileptic-EEG-Classfication-Using-Deep-Learning).
