# Viva / Interview Notes

Plain-English explanations of what the project does and why, at the depth you actually need to defend it — not full implementation detail. Add to this as we build more.

---

## 1. What's the project, in one line?

Early detection/prediction of epileptic seizures from scalp EEG signals, using a deep learning model, with an added interpretability layer (SHAP/LIME) so the model's predictions can be explained rather than being a black box.

## 2. What dataset are you using, and why?

**CHB-MIT Scalp EEG Database** (PhysioNet) — continuous EEG recordings from pediatric epilepsy patients, with seizure onset/offset times manually annotated by clinicians. It's a standard, widely-used benchmark dataset in seizure detection research, which makes results comparable to existing literature.

## 3. What is an EDF file?

EDF (European Data Format) is a standard file format for storing biological signals like EEG/ECG. Each file contains multiple channels (one per electrode) of continuous time-series voltage data, plus metadata (sampling rate, channel names, recording start time).

## 4. What is `mne` and what does it do here?

`mne-python` is a widely-used Python library for processing EEG/MEG (brain signal) data. In this project it's used to:
- Read EDF files into a structured object (`raw`) containing the signal + metadata
- Access channel names and sampling frequency
- Provide the data as a NumPy array for plotting/further processing

Why not just parse the raw bytes yourself? EDF has a specific binary header format — `mne` handles that parsing correctly and is the standard tool researchers use for this, so results are reproducible and comparable to other work.

## 5. What's in the seizure summary file, and why do you need it?

Each subject folder has a `chbXX-summary.txt` file listing, for every recording, whether it contains a seizure and the exact **start/end time in seconds** (relative to that file) if so.

This matters because: EEG data by itself is unlabeled — you can't tell a model "this is a seizure" without ground-truth timestamps. The summary file *is* the ground truth. Extracting these times programmatically (via the parser script) is how we build the labels needed for supervised learning — i.e., turning raw signal into (signal segment, label) pairs the model can train on.

## 6. Why plot the signal with the seizure window highlighted?

Sanity check before doing any modeling — you want to visually confirm the seizure period actually looks different from the surrounding "normal" signal (typically: higher amplitude, more rhythmic/spiky activity). If it didn't look different at all, that would suggest something's wrong with either the data or the label parsing, before you waste time training a model on bad labels.

## 7. What's the plan for the actual detection model?

Not finalized yet — options being considered are CNN-based (treating EEG segments like images/spectrograms) or a feature-based classical ML approach. Base repo (`mkfzdmr/Epileptic-EEG-Classfication-Using-Deep-Learning`) uses a Synchrosqueezing Transform + CNN approach on time-frequency images — we're adapting from this as a starting point. *(Update this section once the approach is locked in.)*

## 8. What's the actual original contribution, if the base model is adapted from an existing repo?

The interpretability layer — using **SHAP** (SHapley Additive exPlanations) and/or **LIME** (Local Interpretable Model-agnostic Explanations) to explain *why* the model classified a given EEG segment as seizure/non-seizure. This matters clinically: a black-box "seizure detected" alert is much less useful to a doctor than one that also shows which channels/frequency bands drove the prediction. This is disclosed clearly in the README and was communicated to the professor upfront.

*(Once this layer is built, add a section here explaining SHAP/LIME conceptually — what they actually compute and why the output is trustworthy.)*

## 9. Why PhysioNet's S3 mirror instead of direct download?

Same data, just a faster distribution channel — PhysioNet also mirrors CHB-MIT on a public AWS S3 bucket, which is significantly faster than downloading from physionet.org directly (which throttled around 40-50 KB/s in practice). No AWS account needed since it's accessed with `--no-sign-request` (anonymous public access).

---

## Terms you should be able to define on the spot

- **EEG** — Electroencephalogram, records electrical activity of the brain via scalp electrodes
- **Seizure onset/offset** — the timestamps marking when a seizure begins and ends in a recording
- **Sampling frequency (Hz)** — how many data points are recorded per second (CHB-MIT is 256 Hz — 256 readings every second per channel)
- **Channel** — one electrode's signal (CHB-MIT recordings have around 23 channels, standard 10-20 electrode placement system)
- **SHAP/LIME** — post-hoc explainability techniques that show which input features most influenced a specific model prediction
- **Ground truth / labels** — the correct answer for each data point (seizure vs non-seizure), used to train and evaluate the model
