# ExecuTorch Export Guide (WSL2/Linux)

This repo exports an ExecuTorch `.pte` with XNNPACK for the E5 query encoder.
ExecuTorch does not provide Windows wheels, so the reliable path is WSL2/Linux.
The generated files are portable back to Windows.

## Prerequisites

- Windows with WSL2 (Ubuntu 22.04 recommended)
- Internet access to download model weights
- 10+ GB free disk space (model + caches)

## Step 1: Install and open WSL2 (Ubuntu)

Run in PowerShell as Administrator (skip if already installed):

```powershell
wsl --install -d Ubuntu-22.04
```

Then open the "Ubuntu" app from the Start menu.

## Step 2: Install Linux dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
```

## Step 3: Go to the repo (Windows drive mount)

```bash
cd /mnt/d/projects/IslamEncycloAI
```

## Step 4: Create a clean virtual environment

```bash
python3 -m venv .venv-et
source .venv-et/bin/activate
pip install --upgrade pip
```

## Step 5: Install Python dependencies

```bash
pip install numpy transformers sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install executorch
```

## Step 6: Run the export

```bash
python scripts/export_e5_query_encoder_executorch.py
```

## Step 7: Verify outputs

```bash
ls output/processed/models
```

Expected files:

- `e5_query_encoder_xnnpack.pte`
- `e5_tokenizer.json`
- `e5_query_encoder_meta.json`

These are written under the Windows path:

- `D:\projects\IslamEncycloAI\output\processed\models\`

## Troubleshooting

### `pip install executorch` fails in WSL

ExecuTorch wheels can vary by platform and PyTorch version. If installation
fails, capture the exact error and use a source build. Ask for a tailored
build recipe with the full error log.

### Model download is slow or fails

Re-run the script. The Hugging Face cache is reused across runs.

### `torch.export.export` not found

You need PyTorch 2.1 or newer. Reinstall torch from the CPU wheel index above.

## Notes

- The script runs a self-test comparing SentenceTransformer and the custom
  module. If it fails, do not ship the artifacts.
- The `.pte` export is CPU/XNNPACK optimized and is the most reliable path
  for low-end devices without accuracy loss.
