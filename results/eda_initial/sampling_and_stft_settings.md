# EDA Sampling and STFT Configuration Documentation

## 1. Random Seed and Reproducibility
- Random Seed: 42
- RNG Generator: `numpy.random.default_rng(42)`
- Sampling Protocol:
  - Exactly 5 seizure windows (label == 1) sampled without replacement per subject.
  - Exactly 5 non-seizure windows (label == 0) sampled without replacement per subject.
  - Total sampled windows across 4 subjects: 40 windows.

## 2. Sampled Subjects and Partitions
- Training Subjects:
  - `chb01`: 230 seizure windows (0.3153% seizure prevalence), 72,951 total windows.
  - `chb03`: 210 seizure windows (0.3072% seizure prevalence), 68,365 total windows.
- Validation Subjects:
  - `chb07`: 167 seizure windows (0.1384% seizure prevalence), 120,674 total windows.
  - `chb08`: 467 seizure windows (1.2975% seizure prevalence), 35,991 total windows.
- Test Partition Subjects:
  - `chb02`, `chb05`, `chb13` were completely untouched and uninspected to preserve test set sanctity and avoid snooping.

## 3. Sampled Window Indices (Seed = 42)
- `chb01`:
  - Seizure sample indices: [35718, 31458, 35692, 5115, 31457]
  - Non-seizure sample indices: [71176, 6884, 38462, 55580, 53733]
- `chb03`:
  - Seizure sample indices: [62475, 3843, 60372, 198, 3842]
  - Non-seizure sample indices: [66701, 6542, 36004, 51999, 50268]
- `chb07`:
  - Seizure sample indices: [120345, 78452, 120327, 72057, 120385]
  - Non-seizure sample indices: [117661, 11348, 63442, 91816, 88756]
- `chb08`:
  - Seizure sample indices: [28051, 10509, 13853, 1375, 10507]
  - Non-seizure sample indices: [35123, 3432, 19034, 27371, 26469]

## 4. Channels Inspected
- Canonical Channel Order: 23 bipolar montage channels.
- Primary Representative Channel:
  - Index 1: `F7-T7` (Left anterior temporal - mid temporal).
- Multi-Channel Montage Subset:
  - Index 0: `FP1-F7`
  - Index 1: `F7-T7`
  - Index 2: `T7-P7`
  - Index 16: `FZ-CZ`
  - Index 20: `FT9-FT10`

## 5. STFT Parameters
- Library: `scipy.signal.stft`
- Sampling Frequency ($f_s$): 128.0 Hz
- Window Duration: 4.0 seconds (512 samples)
- Window Function: Hann window (`scipy.signal.windows.hann`)
- Segment Length (`nperseg`): 128 samples (1.000 second)
- Frequency Resolution ($\Delta f$): 1.000 Hz ($f_s / \text{nperseg}$)
- Segment Overlap (`noverlap`): 112 samples (87.5% overlap)
- Hop Size: 16 samples (0.125 seconds)
- Boundary Handling: `boundary=None` (strictly within-window samples, no zero padding)
- Output Time Frames: 25 time frames ($t = [0.50, 0.625, \dots, 3.50]$ seconds)
- Frequency Band Evaluated: 0.5 Hz – 40.0 Hz (40 frequency bins)
- Color Scaling: Shared vmin / vmax per comparison figure in dB ($10 \log_{10}(P + 10^{-6})$)
