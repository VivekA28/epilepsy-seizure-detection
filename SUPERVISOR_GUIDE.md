# Supervisor & Reviewer Technical Guide

This guide provides technical reference for running and verifying the **Early Epilepsy Seizure Detection** pipeline.

---

## 1. Quickstart (Clone & Run in 3 Commands)

```bash
# 1. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run instant verification demo (CPU or CUDA)
python run_demo.py
```

`run_demo.py` will:
- Check all 10 core Python dependencies (`torch`, `mne`, `pyedflib`, `shap`, `sklearn`, `pandas`, `scipy`, `matplotlib`, `tqdm`).
- Detect CUDA GPU availability (or transparently fall back to CPU).
- Instantiate the 23-channel `SeizureCNN` architecture.
- Load the pre-trained Protocol-v2 checkpoint (`models/baseline_cnn_subjectwise_15subjects_v2.pt`).
- Run inference on simulated 23-channel EEG windows and output predictions.
- Print the benchmark comparison table from `results/all_models_v2_comparison.csv`.
- Verify explainability artifacts (`results/shap/` and LIME plots).

---

## 2. Benchmark Summary (Protocol v2)

All architectures evaluated strictly once on held-out test subjects (`chb02`, `chb05`, `chb13`):

| Architecture | Model Checkpoint | Test PR-AUC | Test ROC-AUC | Test F1 | Test Precision | Test Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN baseline** | `models/baseline_cnn_subjectwise_15subjects_v2.pt` | 0.3786 | 0.6629 | **0.4620** | 0.6667 | 0.3535 |
| **CNN + FFT** | `models/cnn_fft_subjectwise_15subjects_v2.pt` | 0.2244 | **0.8160** | 0.3299 | 0.3472 | 0.3142 |
| **CNN + LSTM** | `models/cnn_lstm_subjectwise_15subjects_v2.pt` | 0.3730 | 0.8044 | 0.4542 | **0.7336** | 0.3290 |
| **CNN + FFT + LSTM** | `models/cnn_fft_lstm_subjectwise_15subjects_v2.pt` | **0.3858** | 0.7784 | **0.4665** | 0.6133 | **0.3764** |

---

## 3. Dataset & Pre-trained Models Note

- **Model Checkpoints**: All trained weights are pre-packaged in `models/` (~6 MB total). No lengthy retraining is necessary to inspect or run models.
- **Results & Predictions**: All prediction arrays and metrics are precomputed in `results/` (~23 MB).
- **Raw EEG**: CHB-MIT EDFs are not stored in git due to size (~50 GB). To download raw data for retraining:
  ```bash
  aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/chb01/ data/raw/chb01/
  ```

---

## 4. Troubleshooting & Notes

- **CPU vs GPU**: All scripts automatically select CUDA if available; otherwise they execute on CPU with zero configuration needed.
- **Path Portability**: Paths in `run_demo.py` are dynamically resolved using `pathlib.Path` relative to the repository root.
- **Explainability**: SHAP attribution maps and channel rankings are located in `results/shap/`.
