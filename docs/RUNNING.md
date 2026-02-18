# Running the ChromaGuide Training (DNABERT -> Adapter -> Mamba)

This document explains how to create the `chromaguide` environment, provide a HuggingFace token, and run the DNABERT→Adapter→Mamba training script with sane defaults.

Prerequisites
- macOS (this repo was developed on macOS)
- mambaforge (or conda)

Create environment (example):

```bash
# create env
/Users/studio/mambaforge/bin/mamba create -n chromaguide python=3.11 -y
# activate and install core deps (adjust channels as needed)
/Users/studio/mambaforge/bin/mamba install -n chromaguide pytorch torchvision torchaudio -c pytorch -y
/Users/studio/mambaforge/bin/mamba install -n chromaguide numpy pandas scipy scikit-learn tqdm transformers -c conda-forge -y
```

Important: Clear or avoid exporting a global `PYTHONPATH` that points to system site-packages when running inside the `chromaguide` env. Example run that ensures `PYTHONPATH` is set to project root only:

```bash
# Run with env python and project PYTHONPATH
PYTHONPATH=$(pwd) /Users/studio/mambaforge/envs/chromaguide/bin/python src/train_dnabert_mamba.py --epochs 10 --batch_size 16 --use_mini --hf_token $HF_TOKEN --trust_remote_code
```

HuggingFace Token
- You can export `HF_TOKEN` in your shell or place a token at `~/.huggingface/token` (script will check both).

Example exports:

```bash
export HF_TOKEN="hf_...your_token_here..."
```

Notes
- For quick validation use `--use_mini` to run a small subset (fast). For final training use the full dataset (do not pass `--use_mini`).
- If DNABERT requires `trust_remote_code=True` (some community repos do), pass `--trust_remote_code` to the training script.
- If you see PyTorch C-extension import errors, ensure you are using the env-specific python and that `PYTHONPATH` is not pointing to the system python site-packages.

Contact
- If you want me to run a 10-epoch mini training now, I can do that (it will use `--use_mini`).
