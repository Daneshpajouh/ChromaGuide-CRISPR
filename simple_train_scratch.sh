#!/bin/bash

# Simplified training script - assumes files already present in /scratch/amird/chromaguide

set -euo pipefail

export PYTORCH_DIR="/scratch/amird/pytorch-env"
export WORK_DIR="/scratch/amird/chromaguide"
export PYTHONPATH="$PYTORCH_DIR:$PYTHONPATH"

echo "[$(date)] Starting training script"
echo "Work directory: $WORK_DIR"
echo "PyTorch directory: $PYTORCH_DIR"

# Verify PyTorch
python -c "import torch; print(f'✅ PyTorch {torch.__version__} loaded'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Setup directories
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/checkpoints"
cd "$WORK_DIR"

# Verify training scripts exist
ls -lh scripts/train_*.py || {
    echo "❌ Training scripts not found in $WORK_DIR/scripts/"
    ls -lh scripts/ || echo "scripts/ directory doesn't exist"
    exit 1
}

# Install remaining deps (numpy, pandas, etc.)
python -m pip install --quiet --no-cache-dir numpy pandas scikit-learn scipy 2>&1 | tail -2 || true

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "TRAINING 1: OFF-TARGET WITH FOCAL LOSS"
echo "════════════════════════════════════════════════════════════════"
echo "Target AUROC: 0.99"
echo "Config: 200 epochs, batch 512, lr 0.0005, Focal Loss"
echo ""

OFFTARGET_LOG="logs/off_target_$(date +%Y%m%d_%H%M%S).log"

python -u scripts/train_off_target_focal.py \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005 \
    2>&1 | tee "$OFFTARGET_LOG" || {
    echo "❌ Off-target training failed"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "TRAINING 2: MULTIMODAL WITH DNABERT-2"
echo "════════════════════════════════════════════════════════════════"
echo "Target Rho: 0.911"
echo "Config: 50 epochs, batch 250, lr 5e-4, DNABERT-2"
echo ""

MULTIMODAL_LOG="logs/multimodal_$(date +%Y%m%d_%H%M%S).log"

python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --lr 5e-4 \
    2>&1 | tee "$MULTIMODAL_LOG" || {
    echo "⚠️  Multimodal training had issues, but continuing..."
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] Training complete"
echo "════════════════════════════════════════════════════════════════"
