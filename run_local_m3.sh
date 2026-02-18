#!/bin/bash
# Local Execution Script for Mac Studio M3 Ultra

# 1. Environment Setup
echo "=== CRISPRO-MAMBA-X Local Training (M3 Ultra Optimization) ==="
echo "Date: $(date)"

# Export Optimizations for Apple Silicon
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 disables memory limit (good for 128GB RAM)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 2. Check Device
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')"

# 3. Directory Setup
mkdir -p checkpoints logs/local_training data/processed

# 4. Run Training
# Using --medium_1k for a decent test size (999 samples)
# M3 Ultra can handle this easily.
echo "=== Starting Training (Medium 1k) on MPS ==="

# We pipe output to both console and log file
python3 -m src.train --medium_1k 2>&1 | tee logs/local_training/m3_run_$(date +%s).log

# Check status
if [ $PIPESTATUS -eq 0 ]; then
    echo "✅ Local Training Successful"
else
    echo "❌ Local Training Failed"
    exit 1
fi
