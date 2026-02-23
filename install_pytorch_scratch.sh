#!/bin/bash

# Install PyTorch to SCRATCH directory (no home quota limits)

set -e

PYTORCH_DIR="/scratch/amird/pytorch-env"
LOG_FILE="/scratch/amird/pytorch_install.log"

mkdir -p "$PYTORCH_DIR"

echo "[$(date)] Starting PyTorch installation to $PYTORCH_DIR" | tee "$LOG_FILE"

# Install torch, torchvision, torchaudio to --target directory
python -m pip install \
    --no-cache-dir \
    --target "$PYTORCH_DIR" \
    torch torchvision torchaudio \
    2>&1 | tee -a "$LOG_FILE"

# Add to PYTHONPATH
export PYTHONPATH="$PYTORCH_DIR:$PYTHONPATH"

# Verify installation
echo "[$(date)] Verifying installation..." | tee -a "$LOG_FILE"
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')" | tee -a "$LOG_FILE"

echo "[$(date)] Installation complete!" | tee -a "$LOG_FILE"
