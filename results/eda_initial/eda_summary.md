# Phase 1 Exploratory Data Analysis: CHB-MIT EEG & STFT Feasibility Summary

**Date**: 2026-10-13  
**Status**: Completed (Read-only, CPU-only, Zero training interference)  
**Output Directory**: `results/eda_initial/`  

---

## 1. Executive Summary & Overview

Phase 1 of exploratory data analysis (EDA) was performed on the processed 15-subject CHB-MIT scalp EEG dataset (`chb01` through `chb15`) to assess dataset integrity, examine signal quality/amplitude distributions across seizure and non-seizure windows, and evaluate the empirical feasibility of Short-Time Fourier Transform (STFT) time-frequency representations before considering model integration.

All operations adhered strictly to safety constraints:
- **Read-only execution**: Zero modifications to dataset files, checkpoints, logs, or running pipelines.
- **CPU-only execution**: `CUDA_VISIBLE_DEVICES=""` with lowest CPU priority (`nice -n 19`), ensuring the GPU multi-seed training pipeline running on the RTX 3050 remained uninterrupted.
- **Memory safety**: Windows arrays were loaded strictly via `np.load(..., mmap_mode="r")`. No full-dataset arrays were brought into RAM.
- **Reproducible sampling**: Fixed random seed (`seed = 42`) across designated training and validation subjects. Test partition subjects (`chb02`, `chb05`, `chb13`) were completely untouched to prevent test-set snooping.

---

## 2. Dataset Overview (Task A)

A complete inventory of processed window arrays, label arrays, metadata tables, and channel definitions was conducted across all 15 subjects.

### Key Metrics Summary
- **Total Windows Across 15 Subjects**: 1,041,166
- **Total Seizure Windows**: 4,250 (0.4082%)
- **Total Non-Seizure Windows**: 1,036,916 (99.5918%)
- **Class Imbalance Ratio**: ~1 seizure window per 244 non-seizure windows (243.98:1)
- **Shape Consistency**: Strictly `(N, 23, 512)` for every subject
- **Channel Consistency**: Exactly 23 bipolar channels per subject, identical canonical ordering
- **Length Consistency**: 100% agreement across `windows.shape[0]`, `labels.shape[0]`, and `metadata.csv` row count for all 15 subjects
- **Missing Files / Corruptions**: None (0 missing files)

### Per-Subject Breakdown Table

| Subject | Partition Split | Total Windows | Seizure Windows | Non-Seizure Windows | Seizure % | Window Array Shape | Channel Count | Consistency Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **chb01** | Train | 72,951 | 230 | 72,721 | 0.3153% | (72951, 23, 512) | 23 | OK |
| **chb02** | Test | 63,443 | 90 | 63,353 | 0.1419% | (63443, 23, 512) | 23 | OK |
| **chb03** | Train | 68,365 | 210 | 68,155 | 0.3072% | (68365, 23, 512) | 23 | OK |
| **chb04** | Train | 40,761 | 26 | 40,735 | 0.0638% | (40761, 23, 512) | 23 | OK |
| **chb05** | Test | 70,166 | 287 | 69,879 | 0.4090% | (70166, 23, 512) | 23 | OK |
| **chb06** | Train | 120,104 | 93 | 120,011 | 0.0774% | (120104, 23, 512) | 23 | OK |
| **chb07** | Validation | 120,674 | 167 | 120,507 | 0.1384% | (120674, 23, 512) | 23 | OK |
| **chb08** | Validation | 35,991 | 467 | 35,524 | 1.2975% | (35991, 23, 512) | 23 | OK |
| **chb09** | Train | 122,146 | 145 | 122,001 | 0.1187% | (122146, 23, 512) | 23 | OK |
| **chb10** | Train | 90,016 | 235 | 89,781 | 0.2611% | (90016, 23, 512) | 23 | OK |
| **chb11** | Train | 62,593 | 407 | 62,186 | 0.6502% | (62593, 23, 512) | 23 | OK |
| **chb12** | Train | 37,216 | 536 | 36,680 | 1.4402% | (37216, 23, 512) | 23 | OK |
| **chb13** | Test | 19,789 | 234 | 19,555 | 1.1825% | (19789, 23, 512) | 23 | OK |
| **chb14** | Train | 46,774 | 98 | 46,676 | 0.2095% | (46774, 23, 512) | 23 | OK |
| **chb15** | Train | 70,177 | 1,025 | 69,152 | 1.4606% | (70177, 23, 512) | 23 | OK |
| **Total** | — | **1,041,166** | **4,250** | **1,036,916** | **0.4082%** | — | — | **All OK** |

*File reference: `results/eda_initial/dataset_overview.csv`*

---

## 3. Sampling Setup & Reproducibility Documentation

To prevent any heavy disk I/O while the multi-seed training was underway, sampling was strictly bounded:
- **Random seed**: `42` (`np.random.default_rng(42)`)
- **Sampled subjects**:
  - Train: `chb01` (primary baseline training subject), `chb03`
  - Validation: `chb07`, `chb08`
  - Test subjects (`chb02`, `chb05`, `chb13`) were strictly excluded from sampling.
- **Sample size**: Exactly 5 randomly sampled seizure windows and 5 randomly sampled non-seizure windows per subject (40 windows total).
- **Representative channel**: Channel index 1 (`F7-T7`), a standard frontotemporal bipolar channel present in all 15 subjects.
- **Montage exploration channels**: `FP1-F7` (ch 0), `F7-T7` (ch 1), `T7-P7` (ch 2), `FZ-CZ` (ch 16), `FT9-FT10` (ch 20).

---

## 4. Signal Quality and Amplitude Analysis (Task B)

Every sampled window was inspected for data-quality defects (NaNs, Infs, clipping, flatlines, extreme amplitude deviations).

### Findings Summary
1. **NaN / Inf Check**:
   - Zero NaN values (0 / 40 windows).
   - Zero Inf values (0 / 40 windows).
2. **Flatline Check**:
   - No channels exhibited near-zero standard deviation (`std < 1e-4`). Minimum channel std across all sampled windows was 0.1818.
3. **Clipping Check**:
   - No flat plateaus or identical repeated values ($\ge 10$ consecutive sample plateaus) were observed in any channel.
4. **Amplitude Distributions & Dynamics**:
   - **Non-seizure baseline windows**: Global standard deviation across all 23 channels ranges from **0.328 to 1.763** z-scores. Values generally sit between -4.5 and +6.0 z-scores.
   - **Seizure windows**: Show markedly higher standard deviation (**1.748 to 7.207** z-scores) with peak excursions reaching **-44.18 to +47.61** in `chb07` (window 120345) and **-16.39 to +20.22** in `chb03` (window 3843).
   - **Normalization behavior**: Because per-channel z-score normalization was applied globally across the continuous recording sessions prior to window slicing, the physiological amplitude burst of ictal events is preserved in the window arrays, rather than artificially squashed to unit variance per window.

### Sample Statistics Excerpt (Representative Channel `F7-T7`)

| Subject | Split | Window Type | Window ID | Global Std | Global Min | Global Max | Ch 1 (F7-T7) Std | Ch 1 Min | Ch 1 Max | Quality Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **chb01** | train | Seizure | 35718 | 2.285 | -13.71 | 14.86 | 3.629 | -13.71 | 6.56 | Clean |
| **chb01** | train | Seizure | 31458 | 3.700 | -15.45 | 15.82 | 3.820 | -13.05 | 9.33 | Clean |
| **chb01** | train | Non-Seizure | 71176 | 1.763 | -11.60 | 11.01 | 1.808 | -4.65 | 6.75 | Clean |
| **chb01** | train | Non-Seizure | 55580 | 0.644 | -2.78 | 2.83 | 0.668 | -1.91 | 1.79 | Clean |
| **chb07** | val | Seizure | 120345 | 7.207 | -44.18 | 47.61 | 3.433 | -9.73 | 12.77 | Clean (High energy) |
| **chb07** | val | Seizure | 78452 | 5.245 | -19.93 | 22.52 | 5.465 | -15.82 | 15.33 | Clean (High energy) |
| **chb07** | val | Non-Seizure | 117661 | 0.450 | -3.01 | 2.85 | 0.626 | -2.11 | 2.85 | Clean |
| **chb07** | val | Non-Seizure | 63442 | 0.424 | -2.42 | 1.88 | 0.472 | -1.74 | 1.62 | Clean |

*Detailed stats for all 40 sampled windows are saved in `results/eda_initial/sampled_windows_stats.csv`.*

---

## 5. Frequency-Domain Exploration & STFT Analysis (Task C)

### STFT Parameter Choices & Justification
To capture intra-window time-frequency dynamics within each 4.0-second window (512 samples at $f_s = 128$ Hz), the following parameters were implemented:
- **Sampling Rate ($f_s$)**: 128 Hz
- **Window Function**: Hann window (`scipy.signal.windows.hann`) to suppress spectral leakage without discontinuity.
- **Segment Length (`nperseg`)**: 128 samples ($1.0\text{ second}$).
  - *Rationale*: Yields $\Delta f = f_s / \text{nperseg} = 1.0\text{ Hz}$ frequency resolution, cleanly resolving physiological bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–40 Hz).
- **Segment Overlap (`noverlap`)**: 112 samples ($87.5\%\text{ overlap}$).
  - *Rationale*: Yields a step size of $\Delta t = 16\text{ samples} = 0.125\text{ seconds}$, providing 25 temporal frames across the 4-second epoch without boundary extrapolation artifacts (`boundary=None`).
- **Physiological Frequency Range**: Truncated to 0.5–40.0 Hz (40 frequency bins), matching the preprocessing bandpass filter.
- **Power Representation**: Power Spectral Density in decibels: $10 \log_{10}(|Z_{xx}|^2 + 10^{-6})$.

### Visible Spectral Differences
1. **Low-Frequency Power Concentration**:
   - In both `chb01` and `chb07`, seizure windows exhibit massive, persistent elevation in power below 5 Hz (delta and low theta).
   - In `chb07` (Window #120345), power at 2.0 Hz and 3.0 Hz is **40× to 65× higher** during seizure than during non-seizure.
   - Mean spectral power across the 0.5–40 Hz band is **-10.3 dB** in seizure vs **-32.2 dB** in non-seizure (a >20 dB difference).
2. **Temporal Rhythmic Evolution vs Static Background**:
   - Non-seizure spectrograms show diffuse, low-amplitude background activity with power falling off smoothly above 8 Hz.
   - Seizure spectrograms exhibit rhythmic pulse-like bands and harmonic ridges extending up into the alpha/beta bands (e.g. 12 Hz harmonic in `chb07`).
   - The STFT captures how rhythmic discharges wax and wane across the 4-second duration. A static whole-window FFT compresses this into a single scalar per band, completely discarding the intra-window timing and phase evolution.

> **Cautionary Scientific Note**: These striking visual spectrogram differences are observed in representative sampled windows. Visual difference in isolated windows does not constitute proof of cross-subject generalization or diagnostic superiority. Ictal patterns vary across patients and seizure types, and non-seizure segments can occasionally contain high-amplitude movement or muscle artifacts.

---

## 6. Justification for Controlled STFT vs Existing FFT Experiment

### Core Comparison: Static FFT vs Time-Frequency STFT
- **Current FFT implementation (`fft_features.py`)**:
  - Takes the full 512-sample window and calculates 5 log band-power scalars per channel: `(n_channels * 5,)` = 115 features.
  - Collapses all within-window time dynamics into a single static average.
- **Candidate STFT representation**:
  - Produces a 2D time-frequency grid per channel (e.g., 40 frequency bins × 25 time frames per channel, or 5 band powers × 25 time frames = 125 features per channel).
  - Preserves intra-window temporal changes (e.g. rhythmic discharge build-up, frequency deceleration).

### Why a Controlled Experiment is Justified
1. **Hypothesis**: Rhythmic seizure progression evolves dynamically over sub-second intervals; time-resolved frequency representations will allow temporal backbones (e.g. 2D CNN or CNN+LSTM) to exploit intra-window spectral modulation that static FFT cannot represent.
2. **Controlled Benchmark Design**:
   - Compare Model 3 (CNN + Static FFT) against an identical architecture receiving time-frequency STFT features (or 2D spectrogram input).
   - Evaluate strictly using the established 15-subject subject-level split (`chb01..15`), with zero hyperparameter tuning on the test partition.
   - Measure whether the incremental computational cost of STFT feature extraction and larger representation dimensionality yields a statistically and clinically meaningful gain in validation F1 and false-positive reduction.

---

## 7. Deliverables Index

All deliverables have been saved to `results/eda_initial/`:
1. `dataset_overview.csv`: Complete subject-by-subject window and label inventory.
2. `sampled_windows_stats.csv`: Full statistical and signal-quality metrics for 40 sampled windows across 4 subjects.
3. `eeg_trace_comparison.png`: Time-domain waveforms comparing seizure vs non-seizure on channel `F7-T7`.
4. `stft_spectrogram_comparison_chb01.png`: Calibrated STFT spectrograms for train subject `chb01`.
5. `stft_spectrogram_comparison_chb07.png`: Calibrated STFT spectrograms for validation subject `chb07`.
6. `multichannel_eeg_sample_preview.png`: 5-channel montage preview (`FP1-F7`, `F7-T7`, `T7-P7`, `FZ-CZ`, `FT9-FT10`).
7. `eda_summary.md`: This comprehensive report.
