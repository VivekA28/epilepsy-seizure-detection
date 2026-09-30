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
<<<<<<< HEAD
- [x] `.gitignore` in place — excludes `data/`, `models/`, `*.edf`, `*.pt`, `*.h5`, `*.pth`, `__pycache__/`, `venv/`, etc.
- [x] `README.md` added with attribution note (base implementation adapted from `mkfzdmr/Epileptic-EEG-Classification-Using-Deep-Learning`, our contribution is the SHAP/LIME explainability layer)
- [x] SSH key auth set up for Vivek (Arch) — push/pull working with no password prompts
- [x] SSH key auth set up for Aishwary (Windows/Git Bash) — in progress, hit a PATH issue with AWS CLI in Git Bash, resolved by reopening Git Bash / manually appending to PATH
- [x] `scripts/download_data.sh` written — wget-based download script (works, but slow against physionet.org directly, ~40-50 KB/s)
- [x] Found and switched to a faster download path: PhysioNet's public S3 mirror via `aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/<subject>/ data/raw/<subject>/`
- [x] chb01 dataset downloaded (Vivek's machine)
- [x] chb01 dataset download on teammate's machine
- [x] `mne` walkthrough: load an EDF file, inspect channels/sampling rate, plot signal, parse seizure onset/offset from `chbXX-summary.txt`
- [x] Data preprocessing pipeline (`src/preprocessing.py`)
- [x] Processed all 42 files for chb01 (memory-safe pipeline: resampled to 128Hz, float32, per-file streaming to avoid RAM crash) → 72,951 windows, 230 seizure (0.32%)
- [x] Fixed data leakage: added file-level tracking to preprocessing, switched train/test split to be by-file rather than by-window
- [x] Retrained baseline CNN with leak-free split: F1=0.97, precision=0.99, recall=0.96 on chb01 (test set: 18,682 windows, 100 seizure, from held-out files)
- [x] Combined all 5 processed subjects (chb01-chb05) into one training run with memory-safe lazy loading (memmap-backed Dataset, avoids ~15GB+ RAM requirement)
- [x] Trained CNN across 5 subjects, leak-free file-level split: F1=0.88, precision=0.94, recall=0.83 on held-out files/patients (315,686 windows total, 843 seizure, 0.27%)
- [x] Added per-channel z-score normalization to preprocessing (fixes SHAP always ranking the same channels regardless of input, caused by raw amplitude scale differences between electrodes)
- [x] Reprocessed all 5 subjects + retrained CNN with normalization: F1=0.87, precision=0.85, recall=0.89 (confusion matrix [[89257, 32], [22, 178]]) - F1 essentially unchanged vs. pre-normalization, but recall improved (catching more real seizures) at some cost of more false alarms
- [x] SHAP explainability layer working - produces genuinely input-specific channel importance per prediction; consistently ranks the same top channel across all sampled chb01 seizure windows, but that top channel doesn't reproduce as dominant on the 5-subject combined model - suggests the chb01-only model may be partly picking up a patient-specific pattern rather than a universal seizure marker
- [x] LIME explainability layer implemented from scratch (`explain_lime.py`) - no off-the-shelf LIME supports this time-series shape, so it segments each window into channel x time-block regions, perturbs them, and fits a local surrogate model to the real model's output; surrogate fit against raw logits rather than probabilities to avoid sigmoid-saturation collapsing the importance scores on confidently-classified windows
- [x] Cross-validated SHAP against LIME on identical windows: both methods independently agree on the top-ranked channel across every sampled seizure window - two unrelated explanation methods converging is stronger evidence than either alone
- [x] Built project presentation deck (title, problem framing, dataset, pipeline, engineering challenges, results, SHAP explainability, LIME cross-validation, thank you)
=======
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
>>>>>>> 751eb06 (Update progress.md)

## Current baseline result

<<<<<<< HEAD
- [ ] Decide on train/test split strategy (per-subject vs pooled) — informed by the SHAP finding that the chb01-only model may be picking up patient-specific artifacts that don't reproduce in the multi-subject model
- [ ] Confirm the channel-index-to-electrode-name mapping used by `explain_lime.py` (currently falls back to generic ch0..ch22) so LIME results can cite real electrode names in the write-up, matching what `explain_shap.py` already shows
=======
Normalized five-subject CNN:
>>>>>>> 751eb06 (Update progress.md)

- Precision: 0.85
- Recall: 0.89
- F1: 0.87
- Confusion matrix: TN=89,257, FP=32, FN=22, TP=178

Earlier pre-normalization result: F1≈0.88, precision≈0.94, recall≈0.83.

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

- [ ] No checkpoint selection using test performance.
- [ ] Use validation F1 (or another pre-declared validation metric) for best-checkpoint selection.
- [ ] Test set is evaluated only after model selection is complete.
- [ ] For the final expanded experiment, use **subject-level separation** so all recordings from a subject remain in exactly one partition.
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

- [ ] Separate the CNN backbone from the final classifier.
- [ ] Expose the 128-D feature vector after `AdaptiveAvgPool1d`.
- [ ] Verify that the refactored CNN reproduces the existing baseline behaviour before adding the LSTM.

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

- [ ] Preserve chronological order.
- [ ] Never cross EDF boundaries.
- [ ] Never cross subject boundaries.
- [ ] Never cross train/validation/test boundaries.
- [ ] Index valid sequences explicitly rather than creating sequences from an unordered window array.

### Sequence-level imbalance

The current window dataset is ~99.7% non-seizure. The LSTM dataset will also be severely imbalanced.

- [ ] Build sequence-level sampling/balancing.
- [ ] Weight a sequence according to the target label of its final window.
- [ ] Do not simply assume the existing window-level sampler can be reused unchanged.

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

- [ ] Verify frequency-bin resolution.
- [ ] Verify band boundaries against FFT bins.
- [ ] Check for NaN/inf.
- [ ] Inspect spectra and feature distributions before model integration.
- [ ] Confirm feature ordering is deterministic: channel × frequency band.

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

- [ ] Implement after FFT extraction is validated.
- [ ] Compare against CNN under the same train/validation/test protocol.
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

- [ ] Process substantially more CHB-MIT data.
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

# 15. Work Order — Next Steps

## Step 1 — Correct evaluation protocol

- [ ] Refactor baseline training to use train/validation/test.
- [ ] Select checkpoint only from validation performance.
- [ ] Keep test untouched until final evaluation.

## Step 2 — Refactor CNN

- [ ] Expose 128-D feature vector.
- [ ] Verify refactored CNN against baseline.

## Step 3 — CNN + LSTM locally

- [ ] Build boundary-safe chronological sequences.
- [ ] Add sequence-level imbalance handling.
- [ ] Cache frozen CNN features.
- [ ] Train LSTM locally.
- [ ] Compare with CNN baseline.

## Step 4 — FFT locally

- [ ] Implement RFFT + Hann.
- [ ] Extract five band powers.
- [ ] Add `epsilon` safeguard.
- [ ] Validate spectra/features.
- [ ] Optionally compare absolute vs relative band power.

## Step 5 — CNN + FFT ablation

- [ ] Train and evaluate under identical protocol.

## Step 6 — CNN + FFT + LSTM

- [ ] Normalize branches before fusion.
- [ ] Build chronological fused sequences.
- [ ] Train locally first.

## Step 7 — Expand data

- [ ] Process more CHB-MIT subjects.
- [ ] Integrate and harmonize Siena or another suitable dataset.
- [ ] Implement subject-level train/validation/test split.

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

**Baseline CNN + SHAP/LIME is complete on five CHB-MIT subjects with file-level leakage control; the methodology is now being upgraded to validation-based model selection, chronological CNN+LSTM modelling, FFT features, CNN+FFT+LSTM fusion, event-level evaluation, larger multi-subject data, and eventual subject-independent testing on expanded datasets.**

---

# 17. Team Notes

- Two people on this project — Vivek (Arch Linux) and Aishwary (Windows, Git Bash).
- Current work should remain controlled and incremental rather than adding unnecessary architecture complexity.
- Keep experiments reproducible: fixed seeds, recorded split assignments, recorded preprocessing parameters, and saved validation/test results.
- Any architectural change should be recorded with its reason and compared against the relevant baseline.
