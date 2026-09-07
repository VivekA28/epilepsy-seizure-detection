# Progress Log — Early Epilepsy Seizure Detection

Running log of what's done and what's next. Update this as the project moves — it's the actual source of truth (not chat history).

## Done

- [x] Repo created and pushed to GitHub: `VivekA28/epilepsy-seizure-detection`
- [x] Folder scaffolding set up (`data/`, `notebooks/`, `src/`, `models/`, `results/`, `scripts/`)
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

## In progress

## Next up

- [ ] SHAP/LIME explainability layer (`src/explainability.py`) — this is the core original contribution
- [ ] Decide on train/test split strategy (per-subject vs pooled)
- [ ] Write up results and evaluation metrics

## Team notes

- Two people on this project — Vivek (Arch Linux) and Aishwary (Windows, Git Bash)
- Working style: pair-programming, Vivek driving initially, Aishwary to take over parts of the CLI/Git work as he gets comfortable
- Consider a branch-per-feature + PR workflow instead of both pushing directly to `main`, once active development starts
