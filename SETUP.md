# Setup Guide — Early Epilepsy Seizure Detection

Follow these steps to get the project running from a fresh clone, on either Linux (Arch) or Windows (Git Bash).

## 1. Clone the repo

```bash
git clone git@github.com:VivekA28/epilepsy-seizure-detection.git
cd epilepsy-seizure-detection
```

If SSH isn't set up yet on this machine, see Section 3 first, then come back and clone using the SSH URL above (not HTTPS — HTTPS will keep prompting for a password).

## 2. Create local folders

These are gitignored on purpose (see `.gitignore`), so they don't come with the clone:

```bash
mkdir -p data/raw data/processed notebooks src models results
```

## 3. Set up SSH access to GitHub (one-time, per machine)

Check if a key already exists:
```bash
ls -al ~/.ssh
```

If not, generate one:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter through the prompts (default location; passphrase optional).

Add it to the SSH agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Print your public key and copy the whole line:
```bash
cat ~/.ssh/id_ed25519.pub
```

Add it to **your own** GitHub account: https://github.com/settings/keys → "New SSH key" → paste → save.

Switch the repo remote to SSH if it isn't already:
```bash
git remote set-url origin git@github.com:VivekA28/epilepsy-seizure-detection.git
```

Test:
```bash
ssh -T git@github.com
```
Expected: `Hi <your-username>! You've successfully authenticated, but GitHub does not provide shell access.`

**Note (Windows/Arch):** `ssh-keygen` requires OpenSSH.
- Arch: `sudo pacman -S openssh`
- Windows: comes bundled with Git for Windows — use Git Bash, not plain CMD/PowerShell, for all commands in this doc.

## 4. Install AWS CLI

Needed to pull the dataset from PhysioNet's S3 mirror (much faster than direct wget from physionet.org).

- **Arch:** `sudo pacman -S aws-cli`
- **Windows:** download and run https://awscli.amazonaws.com/AWSCLIV2.msi (standard installer, next-next-finish)

Verify:
```bash
aws --version
```

**Known issue (Windows):** if `aws --version` works in CMD but not Git Bash, close and fully reopen Git Bash (not just the tab). If it still fails, check the install path exists (`C:\Program Files\Amazon\AWSCLIV2\aws.exe`) and add it manually:
```bash
echo 'export PATH="/c/Program Files/Amazon/AWSCLIV2:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 5. Download the dataset (CHB-MIT via S3, no AWS account needed)

```bash
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/chb01/ data/raw/chb01/
```

Replace `chb01` with any other subject folder (`chb02`, `chb03`, ...) as needed. This is much faster than the original `wget -r` approach against physionet.org directly, which can crawl at ~40-50 KB/s.

Each subject folder contains multiple `.edf` files (9–42 depending on subject, usually ~1 hour each) plus a `chbXX-summary.txt` with seizure annotation metadata.

## 6. You're set up

At this point you should have:
- The repo cloned with SSH push/pull working, no password prompts
- Empty working folders (`data/`, `models/`, etc.) created locally
- AWS CLI installed
- At least one subject's EEG data downloaded into `data/raw/`

See `PROGRESS.md` for what's been done on the project so far and what's next.
