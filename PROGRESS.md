# Progress Log — Early Epilepsy Seizure Detection

Running log of what's done and what's next. **This is the source of truth.**

## Project Goal

Build and evaluate an explainable early seizure-detection pipeline using scalp EEG. The current project is intentionally focused on the supervisor's requested extensions:

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

- [x] Repo created and pushed to GitHub: `VivekA28/epilepsy-seizure-detection`
- [x] Folder scaffolding set up (`data/`, `notebooks/`, `src/`, `models/`, `results/`, `scripts/`)
- [x] `.gitignore` excludes large/raw/model files.
- [x] README added with attribution note.
- [x] MNE walkthrough completed: EDF loading, channel/sampling inspection, signal plotting, seizure annotation parsing.
- [x] Memory-safe preprocessing pipeline implemented.
- [x] CHB-MIT preprocessing finalized: 256 Hz → 128 Hz, 60 Hz notch, 0.5–40 Hz bandpass, per-channel z-score normalization, 4-second windows, 50% overlap.
- [x] chb01 processed: 72,951 windows, 230 seizure windows.
- [x] Window-overlap leakage fixed by tracking EDF/file IDs and splitting by file.
- [x] Five-subject local dataset processed: chb01–chb05, 315,686 windows, 843 seizure windows (~0.27%).
- [x] Normalized five-subject CNN trained.
- [x] SHAP explainability implemented with `GradientExplainer`.
- [x] Custom LIME time-series explainer implemented: 23 channels × 4 temporal segments = 92 superfeatures, 500 perturbations, locality-weighted Ridge regression on CNN logits.
- [x] SHAP/LIME findings reviewed. Channel importance varies across windows; repeated FT9-FT10 importance in two seizure windows is a reason to test cross-subject generalization, but is **not proof** of artifact learning.

## Baseline results

### Historical normalized baseline (old test-selected protocol; not used for new comparisons)

- Precision: 0.85
- Recall: 0.89
- F1: 0.87
- Confusion matrix: TN=89,257, FP=32, FN=22, TP=178

Earlier pre-normalization result: F1≈0.88, precision≈0.94, recall≈0.83.

### Corrected five-subject CNN baseline (validation-selected checkpoint)

- Test seizure precision: 0.98
- Test seizure recall: 0.70
- Test seizure F1: 0.82
- Confusion matrix: `[[55621, 2], [45, 107]]`

This corrected result is the baseline used for the new CNN+LSTM / CNN+FFT / CNN+FFT+LSTM comparisons.

### Important methodological limitation of the old baseline

The old training script evaluated the test set every epoch and selected the best checkpoint using test F1. This is **test-set snooping** and must not be repeated in new experiments.

The corrected protocol below uses a validation set for checkpoint/model selection and keeps the test set untouched until final evaluation.

---

# 2. Split and Generalization Protocol

## Current baseline

The existing five-subject baseline uses a **file-level split**:

```text
EDF file → entirely train OR test
```

This prevents overlapping windows from the same EDF appearing across train/test.

It is **not** a patient-independent evaluation because different EDF files from the same subject can occur in different partitions.

## New experimental protocol

For all new model comparisons, use a proper three-way split:

```text
TRAIN       VALIDATION       TEST
 70%           15%           15%
  ↓              ↓             ↓
weights      checkpoint      FINAL
updates      selection       evaluation
```

The exact percentages can be adjusted if the available number of files/subjects makes a different split more appropriate, but the roles must remain separate.

Rules:

- [x] No checkpoint selection using test performance in all new experiments.
- [x] New experiments select the best checkpoint using validation F1.
- [x] New experiments evaluate the test set once after model selection.
- [x] For the final expanded experiment, use **subject-level separation** so all recordings from a subject remain in exactly one partition.
- [ ] No sequence may cross a subject boundary, EDF boundary, or train/validation/test boundary.

The expected direction is that subject-independent evaluation may be harder than the current file-level baseline. **Do not assume or document a specific future F1 drop before measuring it.**

---

# 3. Planned Model Progression

We will compare the following controlled stages:

```text
Model 1: CNN

Model 2: CNN + LSTM

Model 3: CNN + FFT

Model 4: CNN + FFT + LSTM
```

The CNN+FFT ablation is useful because it separates the effect of frequency-domain information from the effect of temporal modelling.

Do not assume FFT or LSTM improves performance. Improvement must be demonstrated experimentally.

---

# 4. Model 1 — CNN Baseline

Existing architecture:

```text
Input: 23 channels × 512 samples
        ↓
Conv1D 23→32, k=7
BatchNorm + ReLU + MaxPool(4)
        ↓
Conv1D 32→64, k=5
BatchNorm + ReLU + MaxPool(4)
        ↓
Conv1D 64→128, k=3
BatchNorm + ReLU
        ↓
AdaptiveAvgPool1D(1)
        ↓
128-D feature vector
        ↓
Linear 128→64
ReLU + Dropout(0.3)
        ↓
Linear 64→1
```

### Refactor required before LSTM

- [x] CNN backbone separated conceptually from the classifier.
- [x] 128-D feature vector exposed after `AdaptiveAvgPool1d` via `extract_features()` and cached.
- [x] Refactored CNN verified against the original checkpoint behaviour before LSTM integration.

---

# 5. Model 2 — CNN + LSTM

The LSTM must model **chronological consecutive windows**, rather than independent 4-second windows.

Initial design:

- single-layer **unidirectional** LSTM
- hidden size: 128
- initial sequence length: 5 windows
- each window: 4 seconds
- 50% overlap means starts are 2 seconds apart
- 5 windows cover approximately 12 seconds from first-window start to last-window end
- prediction target = label of the last/current window

```text
W(t-4)  W(t-3)  W(t-2)  W(t-1)  W(t)
   │       │       │       │       │
  CNN     CNN     CNN     CNN     CNN
   │       │       │       │       │
 128-D   128-D   128-D   128-D   128-D
   └───────┴───────┴───────┴───────┘
                    ↓
                 LSTM
                    ↓
               classifier
                    ↓
              label of W(t)
```

### Sequence construction rules

- [x] Preserve chronological order.
- [x] Never cross EDF boundaries.
- [x] Local sequence construction keeps each sequence within one subject/EDF.
- [x] Never cross train/validation/test boundaries.
- [x] Valid 5-window sequences are indexed explicitly from EDF/file order.

### Sequence-level imbalance

The current window dataset is ~99.7% non-seizure. The LSTM dataset will also be severely imbalanced.

- [x] Sequence-level weighted sampling implemented.
- [x] Sequence weight is based on the target label of the final/current window.
- [x] Training uses a sequence-level sampler rather than the old window-level sampler.

### Local development strategy

Because local CPU development is limited:

**Phase A — frozen CNN:**

```text
trained CNN
   ↓
extract 128-D feature for each window
   ↓
cache features to disk
   ↓
train LSTM on cached features
```

This is for fast architecture/prototype development.

**Phase B — GPU fine-tuning:**

```text
CNN + LSTM trained jointly
        ↓
end-to-end fine-tuning
```

The university GPU should be used for larger data and final end-to-end experiments.

---

# 6. FFT / Frequency-Domain Features

FFT will be computed from the **same preprocessed EEG windows** used by the CNN.

Current planned parameters:

```text
Sampling rate = 128 Hz
Window length = 512 samples
Duration      = 4 seconds
Δf            = 0.25 Hz
```

### FFT pipeline

```text
preprocessed 4-sec window
        ↓
Hann window
        ↓
RFFT
        ↓
positive frequencies
        ↓
0.5–40 Hz
        ↓
5 bands
        ↓
band-power features
```

Initial bands:

- Delta: 0.5–4 Hz
- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz
- Gamma: 30–40 Hz

For 23 channels × 5 bands:

```text
23 × 5 = 115 FFT features / window
```

### Numerical stability

Use:

```python
log_power = np.log(band_power + 1e-8)
```

to avoid `log(0)` / `-inf` / NaN problems in very-low-power bands.

### Absolute vs relative power

Do not assume relative band power is automatically superior.

Possible experiment:

```text
Variant A: log absolute band power
Variant B: log relative band power
```

Relative band power:

```text
band power / total power
```

Choose based on validation experiments and interpretability.

### FFT validation checklist

- [x] Frequency-bin resolution confirmed as 0.25 Hz for N=512, fs=128 Hz.
- [x] FFT band definitions were implemented and used in the 115-feature representation.
- [x] FFT features used successfully without NaN/inf failures in model training.
- [ ] Inspect spectra and feature distributions more thoroughly before final reporting.
- [x] FFT feature ordering/naming was generated deterministically as channel × band.

---

# 7. Model 3 — CNN + FFT Ablation

This model isolates the contribution of frequency-domain information.

```text
EEG window
   ├───────────────┐
   ↓               ↓
  CNN             FFT
   ↓               ↓
128-D            115-D
   └───────┬───────┘
           ↓
       feature fusion
           ↓
       classifier
```

- [x] Implemented after FFT feature files were generated and shape-checked.
- [x] Compared against CNN using the same file-level 70/15/15 protocol.
- [ ] Keep this ablation if it helps explain the contribution of FFT.

---

# 8. Model 4 — CNN + FFT + LSTM

This is the main combined candidate, not an assumed final winner.

```text
                         EEG
                          │
                    4-sec windows
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        TIME DOMAIN              FREQUENCY DOMAIN
             │                         │
            CNN                       FFT
             │                         │
          128-D                     115-D
             │                         │
         LayerNorm                 LayerNorm
             │                         │
             └────────────┬────────────┘
                          ↓
                    243-D fused vector
                          ↓
                consecutive sequences
                          ↓
                     uni-LSTM
                          ↓
                     classifier
```

### Fusion decision

The CNN and FFT branches have different feature distributions. Before concatenation:

```text
CNN 128-D → LayerNorm ─┐
                       ├→ concatenate → 243-D → LSTM
FFT 115-D → LayerNorm ─┘
```

Start with LayerNorm without adding arbitrary dimensionality reduction. A projection such as 128→64 / 115→64 may be tested later only if there is an experimental reason.

- [ ] Validate fusion dimensions.
- [ ] Check feature distributions after normalization.
- [ ] Train with sequence-level sampling.
- [ ] Compare against CNN, CNN+LSTM, and CNN+FFT under the same evaluation protocol.

---

# 9. Data Expansion and Second Dataset

Supervisor requirement:

- [x] Process substantially more CHB-MIT data.
- [ ] Add at least one additional EEG seizure dataset.

### Candidate second dataset: Siena Scalp EEG Database

Siena is currently a candidate because it is a scalp EEG dataset available through PhysioNet and is potentially compatible with the project's channel-based processing approach.

**Do not assume compatibility before checking the actual recordings.**

Before integration:

- [ ] Verify channel names/montage.
- [ ] Verify sampling rates.
- [ ] Verify seizure annotation format.
- [ ] Verify available subjects/files.
- [ ] Determine channel intersection/alignment with CHB-MIT.
- [ ] Confirm preprocessing/label definitions can be applied consistently.
- [ ] Preserve dataset ID, subject ID and EDF/file ID in processed data.

The additional dataset should be treated as an independent source during evaluation where appropriate; do not blindly merge incompatible montages or label definitions.

---

# 10. University GPU Plan

Local development should **not** wait for the GPU.

Use the local five-subject dataset to validate:

- tensor dimensions
- CNN feature extraction
- sequence construction
- LSTM implementation
- FFT correctness
- fusion architecture
- leakage prevention

Then move larger experiments to the university GPU.

- [ ] Verify CUDA/PyTorch environment.
- [ ] Run a small GPU sanity test.
- [ ] Prepare larger processed dataset efficiently.
- [ ] Run controlled model comparisons.
- [ ] Fine-tune CNN + LSTM / CNN + FFT + LSTM end-to-end.

Do not present local five-subject results as final patient-independent generalization results.

---

# 11. Explainability Plan

Current SHAP/LIME explain the CNN input directly:

```text
23 channels × 512 samples
```

For the final temporal/frequency model, explanations must account for the changed input representation.

## SHAP

Potential explanation levels:

```text
Temporal dimension:
Which of the previous/current windows contributed?

Channel dimension:
Which EEG channels contributed?

Frequency dimension:
Which channel × frequency band features contributed?
```

FFT features have a direct mapping:

```text
feature → channel → band
```

For example:

```text
FT9-FT10 × Theta
```

CNN features are latent representations, so direct biological interpretation should not be claimed.

## LIME

- [ ] Adapt the custom LIME method to the final input representation.
- [ ] Decide whether perturbations should operate on time windows, channel/time regions, or fused features.
- [ ] Preserve the distinction between model attribution and biological causality.

## Final explainability evaluation

- [ ] Explain real samples from multiple subjects.
- [ ] Include seizure and non-seizure examples.
- [ ] Check whether important channels/features remain subject-specific or become more consistent across subjects.
- [ ] Clearly state that SHAP/LIME describe model behaviour, not causal neurophysiology.

---

# 12. Evaluation Metrics

## Window-level metrics

Continue reporting:

- Precision
- Recall / sensitivity
- F1
- Confusion matrix

Accuracy should not be the main metric because of the extreme class imbalance.

## Event-level metrics

Add event-oriented evaluation because overlapping windows do not fully describe clinical detection behaviour.

Initial metrics:

### 1. Seizure event sensitivity

Define a seizure event as detected when the model satisfies a pre-declared rule, e.g. `k` qualifying consecutive positive windows.

Do not choose `k` after seeing test results.

### 2. False alarm rate

Report false alarm events per hour of non-seizure recording.

### 3. Detection latency

Report the time between annotated seizure onset and the first qualifying positive prediction.

The exact event-matching and `k` definition must be fixed before final test evaluation.

---

# 13. Experimental Matrix

Use the same data split, preprocessing, target definition and evaluation protocol for controlled comparisons:

| Model | Time-domain CNN | FFT | LSTM |
|---|---:|---:|---:|
| CNN | ✓ | — | — |
| CNN + LSTM | ✓ | — | ✓ |
| CNN + FFT | ✓ | ✓ | — |
| CNN + FFT + LSTM | ✓ | ✓ | ✓ |

Questions answered by this matrix:

- CNN → CNN+LSTM: does temporal context help?
- CNN → CNN+FFT: does frequency-domain information help?
- CNN+FFT → CNN+FFT+LSTM: does temporal modelling add value after FFT fusion?

Do not select a final model from one metric alone. Consider validation performance, event-level behaviour, generalization and computational cost.

---

# 14. Current Known Limitations / Risks

1. **Current baseline is file-level, not patient-independent.**
2. **Old baseline used test F1 for checkpoint selection.** New experiments must fix this.
3. **Only five CHB-MIT subjects are currently used for local development.**
4. **The second dataset has not yet been integrated.**
5. **CNN+FFT+LSTM is a proposed candidate, not a proven improvement.**
6. **FFT features need empirical validation; relative power is an optional comparison, not an assumption.**
7. **LSTM sequence construction can introduce leakage if boundaries are ignored.**
8. **Event-level metrics require explicit detection/matching rules.**
9. **SHAP/LIME attributions are model explanations, not proof of physiological causality.**
10. **Subject-independent performance is unknown until measured.** Do not predict a specific score in advance.

---

# 15. Work Completed — 2026-10-01

## Evaluation protocol correction

- [x] Replaced the old two-way train/test workflow with a file-level 70% train / 15% validation / 15% test split for new experiments.
- [x] Best checkpoints are selected using validation F1.
- [x] Test data is not used for epoch-by-epoch model selection.
- [x] Added a guard for the two-stage stratified split so small temporary splits fail clearly instead of producing an obscure sklearn error.
- [x] Added reproducibility seeds (`torch`, NumPy, and CUDA when available).

The final patient-independent experiment is still pending; the current local protocol remains file-level because recordings from the same subject can occur in different partitions.

## CNN feature extraction / LSTM pipeline

- [x] Refactored the CNN so its 128-D pooled representation can be extracted independently of the classifier.
- [x] Cached 128-D CNN features for chb01–chb05 under `data/processed/cnn_features/`.
- [x] Built 5-window chronological sequence construction.
- [x] Enforced EDF/file and partition boundaries during sequence construction.
- [x] Used the last/current window as the sequence target.
- [x] Added sequence-level `WeightedRandomSampler` based on the target label.
- [x] Trained the frozen-CNN + LSTM development model.

### CNN + LSTM local result

- Validation best checkpoint: epoch 5, validation F1 = 0.93.
- Test seizure precision = 0.93.
- Test seizure recall = 0.82.
- Test seizure F1 = 0.87.
- Test confusion matrix: `[[55514, 9], [27, 125]]`.

## FFT feature pipeline

- [x] FFT features generated for all five current subjects.
- [x] Each 23-channel window produces 115 features (23 channels × 5 bands).
- [x] Features use the planned Hann-window + RFFT + band-power + log-power representation.
- [x] Numerical safeguard uses `log(power + 1e-8)`.
- [x] Current FFT caches:
  - chb01: `(72951, 115)`
  - chb02: `(63443, 115)`
  - chb03: `(68365, 115)`
  - chb04: `(40761, 115)`
  - chb05: `(70166, 115)`
- [x] CNN and FFT rows were shape-checked against labels/file IDs before training the ablation.

## CNN + FFT ablation

- [x] Built a per-window CNN + FFT classifier without LSTM.
- [x] Applied LayerNorm separately to the 128-D CNN branch and 115-D FFT branch before concatenation.
- [x] Used the corrected validation-selected checkpoint protocol.

### CNN + FFT local result

- Validation best checkpoint: epoch 10, validation F1 = 0.93.
- Test seizure precision = 0.97.
- Test seizure recall = 0.77.
- Test seizure F1 = 0.86.
- Test confusion matrix: `[[55620, 3], [35, 117]]`.

## CNN + FFT + LSTM combined model

- [x] Built the combined 243-D per-window representation (128 CNN + 115 FFT).
- [x] Applied separate LayerNorm to CNN and FFT branches before concatenation.
- [x] Fed 5 chronological fused windows into a single-layer unidirectional LSTM with hidden size 128.
- [x] Used validation F1 for checkpoint selection and evaluated the test set once.

### CNN + FFT + LSTM local result

- Validation best checkpoint: epoch 12, validation F1 = 0.94.
- Test seizure precision = 0.99.
- Test seizure recall = 0.78.
- Test seizure F1 = 0.87.
- Test confusion matrix: `[[55522, 1], [34, 118]]`.

## Controlled local experiment matrix

| Model | Seizure Precision | Seizure Recall | Seizure F1 |
|---|---:|---:|---:|
| CNN (corrected protocol) | 0.98 | 0.70 | 0.82 |
| CNN + LSTM | 0.93 | 0.82 | 0.87 |
| CNN + FFT | 0.97 | 0.77 | 0.86 |
| CNN + FFT + LSTM | 0.99 | 0.78 | 0.87 |

Interpretation for development only:
- LSTM produced the largest recall increase over the corrected CNN baseline in this local experiment.
- FFT improved the CNN-only ablation.
- Adding FFT to CNN + LSTM produced similar F1 to CNN + LSTM but changed the precision/recall trade-off.
- These results are not yet patient-independent final results.

## Code / engineering fixes completed today

- [x] Added dynamic checkpoint filenames to prevent one-subject runs from overwriting multi-subject models.
- [x] Reduced unnecessary vertical formatting/comments in the model scripts.
- [x] Removed the PyTorch warning caused by converting read-only memory-mapped NumPy rows to tensors by copying feature rows before conversion.
- [x] Added separate viva notes for the CNN+LSTM and CNN+FFT+LSTM stages.

---

# 16. Work Remaining

- [ ] More CHB-MIT subjects.
- [ ] Complete/record formal standalone FFT validation and spectrum inspection.
- [ ] Integrate and harmonize a second EEG dataset.
- [x] Build the final subject-level train/validation/test split.
- [ ] Run larger experiments on the university GPU.
- [ ] Fine-tune CNN + LSTM end-to-end.
- [ ] Fine-tune CNN + FFT + LSTM end-to-end.
- [ ] Add event-level sensitivity, false alarms/hour, and detection latency with a pre-declared event-matching rule.
- [ ] Re-run SHAP/LIME on the final selected model using multiple subjects.
- [ ] Finalize the report/presentation after expanded experiments.

---

# 15. Work Order — Next Steps

## Step 1 — Correct evaluation protocol

- [x] Refactor baseline training to use train/validation/test.
- [x] Select checkpoint only from validation performance.
- [x] Keep test untouched until final evaluation.

## Step 2 — Refactor CNN

- [x] Expose 128-D feature vector.
- [x] Verify refactored CNN against baseline.

## Step 3 — CNN + LSTM locally

- [x] Build boundary-safe chronological sequences.
- [x] Add sequence-level imbalance handling.
- [x] Cache frozen CNN features.
- [x] Train LSTM locally.
- [x] Compare with CNN baseline.

## Step 4 — FFT locally

- [x] Implement RFFT + Hann.
- [x] Extract five band powers.
- [x] Add `epsilon` safeguard.
- [x] Validate FFT feature dimensions and numerical behaviour; deeper spectrum inspection remains pending.
- [ ] Optionally compare absolute vs relative band power.

## Step 5 — CNN + FFT ablation

- [x] Trained and evaluated with the same corrected protocol.

## Step 6 — CNN + FFT + LSTM

- [x] CNN and FFT branches normalized separately with LayerNorm before fusion.
- [x] Built 5-window chronological fused sequences.
- [x] Trained locally on the five-subject development dataset.

## Step 7 — Expand data

- [x] Process more CHB-MIT subjects.
- [ ] Integrate and harmonize Siena or another suitable dataset.
- [x] Implement subject-level train/validation/test split.

## Step 8 — University GPU

- [ ] Run larger controlled experiments.
- [ ] Fine-tune CNN + LSTM end-to-end.
- [ ] Fine-tune CNN + FFT + LSTM end-to-end.

## Step 9 — Final evaluation

- [ ] Window-level metrics.
- [ ] Event-level sensitivity.
- [ ] False alarms/hour.
- [ ] Detection latency.
- [ ] Subject-independent generalization.

## Step 10 — Explainability

- [ ] SHAP on final selected model.
- [ ] LIME on final selected model.
- [ ] Multiple subjects.
- [ ] Time/channel/frequency interpretation where technically supported.

## Step 11 — Report / presentation

- [ ] Update architecture diagrams.
- [ ] Update experimental comparison table.
- [ ] Document split and leakage controls.
- [ ] Document FFT design.
- [ ] Document LSTM sequence design.
- [ ] Document SHAP/LIME findings and limitations.
- [ ] Write final conclusions only after experiments are complete.

---

# 16. Current Status in One Line

**Baseline CNN + SHAP/LIME is complete; corrected validation-based CNN evaluation, frozen-CNN + LSTM, CNN + FFT, and frozen-feature CNN + FFT + LSTM local development experiments are complete on chb01–chb05. CHB-MIT is now expanded to chb01–chb15 and a subject-level train/validation/test split is in place and verified. Next major work is end-to-end GPU fine-tuning, event-level metrics, second-dataset integration, final explainability, and report finalization.**

---

# 17. Team Notes

- Two people on this project — Vivek (Arch Linux) and Aishwary (Windows, Git Bash).
- Current work should remain controlled and incremental rather than adding unnecessary architecture complexity.
- Keep experiments reproducible: fixed seeds, recorded split assignments, recorded preprocessing parameters, and saved validation/test results.
- Any architectural change should be recorded with its reason and compared against the relevant baseline.

---

# 2026-10-06 — Critical Finding: Channel Order Mismatch

## Confirmed issue

A channel audit identified a deterministic channel-order mismatch affecting:

- CHB12
- CHB13
- CHB14
- CHB15

The affected processed arrays contain the same 23 channels but do not use the canonical CHB01–CHB11 channel order.

Indices **8–17** are permuted:

```text
Canonical:
8–11   FP2-F4, F4-C4, C4-P4, P4-O2
12–15  FP2-F8, F8-T8, T8-P8, P8-O2
16–17  FZ-CZ, CZ-PZ
```

Current CHB12–CHB15:

```text
8–9    FZ-CZ, CZ-PZ
10–17  FP2-F4, F4-C4, C4-P4, P4-O2,
       FP2-F8, F8-T8, T8-P8, P8-O2
```

Thus 10/23 channel positions have different semantics from the canonical order.

## Exact correction

```python
permutation = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    10, 11, 12, 13,
    14, 15, 16, 17,
    8, 9,
    18, 19, 20, 21, 22
]
```

For `(N, 23, 512)` windows:

```python
corrected_windows = windows[:, permutation, :]
```

The correction is a lossless channel-axis permutation because the audited preprocessing operations are channel-independent.

## Important qualification

The mismatch is a confirmed data-pipeline defect.

It is **not yet proven that it is the sole cause** of the poor subject-wise CNN result.

Latest reported subject-wise result:

```text
Precision : 0.4286
Recall    : 0.0245
F1-score  : 0.0464
ROC-AUC   : 0.3995
PR-AUC    : 0.0212
```

The correction must therefore be tested experimentally rather than assumed to solve the entire problem.

## Next controlled experiment

Only channel order will change.

Keep the following fixed:

- CNN architecture
- subject split
- random seed
- optimizer
- learning rate
- batch size
- loss
- sampler
- checkpoint criterion
- evaluation protocol

### Steps

1. Back up `data/processed`.
2. Reorder CHB12–CHB15 using the fixed permutation.
3. Update their channel-name metadata.
4. Validate exact channel names at every index.
5. Re-run data validation.
6. Confirm labels, metadata, window counts and seizure counts are unchanged.
7. Regenerate derived CNN/FFT features.
8. Retrain the exact same CNN.
9. Compare the result against the current baseline.

## Rules

Do not:

- flip probabilities
- rescue the test result through test-threshold tuning
- use the test set for checkpoint selection
- change several model/training components at once
- train LSTM/FFT before the corrected CNN baseline is established

## Status

- [x] Channel mismatch identified
- [x] Affected subjects identified
- [x] Exact permutation determined
- [x] Lossless correction strategy established
- [x] Back up processed data
- [x] Implement correction
- [x] Validate semantic channel order
- [x] Rebuild derived features
- [x] Retrain baseline (Model 1: Baseline CNN on 15 subjects)
- [x] Train Model 2 (CNN + FFT on 15 subjects)
- [x] Train Model 3 (CNN + LSTM on 15 subjects)
- [x] Train Model 4 (CNN + FFT + LSTM on 15 subjects)

---

# 12. 15-Subject Controlled Benchmark Results

### Split Configuration (Zero Leakage)
- **Train (10 subjects)**: `chb01, chb03, chb04, chb06, chb09, chb10, chb11, chb12, chb14, chb15` (731,103 windows / 3,005 seizure)
- **Validation (2 subjects)**: `chb07, chb08` (156,665 windows / 634 seizure)
- **Test (3 subjects)**: `chb02, chb05, chb13` (153,398 windows / 611 seizure)

### Four-Model Matrix (Evaluated on Unseen Held-Out Test Subjects, Threshold = 0.5)

| Metric | Model 1: Baseline CNN | Model 2: CNN + FFT | Model 3: CNN + LSTM | Model 4: CNN + FFT + LSTM |
| :--- | :--- | :--- | :--- | :--- |
| **Best Val Epoch** | Epoch 3 | Epoch 11 | Epoch 4 | Epoch 11 |
| **Best Val F1** | 0.2539 | **0.3360** | 0.2419 | 0.3100 |
| **Test Precision** | 0.2812 | 0.0045 | **0.4615** | 0.0038 |
| **Test Recall** | 0.0442 | 0.0622 | 0.0196 | **0.1980** |
| **Test F1** | **0.0764** | 0.0084 | 0.0377 | 0.0075 |
| **Test ROC-AUC** | ~0.40 | — | 0.2062 | 0.4015 |
| **Test PR-AUC** | ~0.02 | — | 0.0202 | 0.0133 |
| **True Positives (TP)** | 27 | 38 | 12 | **121** |
| **False Positives (FP)** | 69 | 8,356 | **14** | 31,739 |
| **False Negatives (FN)**| 584 | 573 | 599 | **490** |
| **True Negatives (TN)** | 152,718 | 144,431 | 152,429 | 120,704 |
| **Total Test Positives**| 611 | 611 | 611 | 611 |


