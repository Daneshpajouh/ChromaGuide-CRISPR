#!/bin/bash

# Master training script using PyTorch installed in SCRATCH
# This bypasses home directory disk quota limitations

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

PYTORCH_DIR="/scratch/amird/pytorch-env"
WORK_DIR="/scratch/amird/chromaguide"
LOG_DIR="$WORK_DIR/logs"
GITHUB_REPO="https://github.com/amirdezfooli/chromaguide.git"

# ============================================================================
# SETUP
# ============================================================================

echo "[$(date)] Starting training script"
echo "Work directory: $WORK_DIR"
echo "PyTorch directory: $PYTORCH_DIR"

# Add SCRATCH PyTorch to PYTHONPATH
export PYTHONPATH="$PYTORCH_DIR:$PYTHONPATH"

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} loaded'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch not available. Waiting 30s for installation to complete..."
    sleep 30
    python -c "import torch; print(f'PyTorch {torch.__version__} loaded')" || exit 1
}

# Setup work directory
mkdir -p "$WORK_DIR" "$LOG_DIR"

# Clone or update repo
if [ ! -d "$WORK_DIR/.git" ]; then
    echo "[$(date)] Cloning chromaguide repository..."
    cd "$WORK_DIR"
    git clone "$GITHUB_REPO" . --depth 1
else
    echo "[$(date)] Updating chromaguide repository..."
    cd "$WORK_DIR"
    git pull origin main
fi

# Install other dependencies (non-PyTorch)
echo "[$(date)] Installing dependencies..."
pip install --quiet --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    scipy \
    transformers \
    biopython \
    dnabertx \
    2>&1 | tail -5 || true

# ============================================================================
# TRAINING 1: OFF-TARGET WITH FOCAL LOSS
# ============================================================================

echo "[$(date)] Starting OFF-TARGET training with Focal Loss..."
OFFTARGET_LOG="$LOG_DIR/off_target_$(date +%Y%m%d_%H%M%S).log"

python -u scripts/train_off_target_focal.py \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005 \
    --device cuda \
    2>&1 | tee "$OFFTARGET_LOG"

OFF_TARGET_AUROC=$(grep "Best AUROC:" "$OFFTARGET_LOG" | tail -1 | grep -oE '[0-9]\.[0-9]+' | tail -1)
echo "✅ Off-target training complete - AUROC: ${OFF_TARGET_AUROC:-pending}" | tee -a "$OFFTARGET_LOG"

# ============================================================================
# TRAINING 2: MULTIMODAL WITH DNABERT-2
# ============================================================================

echo "[$(date)] Starting MULTIMODAL training with DNABERT-2..."
MULTIMODAL_LOG="$LOG_DIR/multimodal_$(date +%Y%m%d_%H%M%S).log"

python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --lr 5e-4 \
    --device cuda \
    2>&1 | tee "$MULTIMODAL_LOG"

MULTIMODAL_RHO=$(grep "Best Rho:" "$MULTIMODAL_LOG" | tail -1 | grep -oE '[0-9]\.[0-9]+' | tail -1)
echo "✅ Multimodal training complete - Rho: ${MULTIMODAL_RHO:-pending}" | tee -a "$MULTIMODAL_LOG"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "TRAINING SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo "Timestamp: $(date)"
echo ""
echo "OFF-TARGET FOCAL LOSS:"
echo "  Target AUROC: 0.99"
echo "  Achieved AUROC: ${OFF_TARGET_AUROC:-unknown}"
echo "  Log: $OFFTARGET_LOG"
echo ""
echo "MULTIMODAL DNABERT-2:"
echo "  Target Rho: 0.911"
echo "  Achieved Rho: ${MULTIMODAL_RHO:-unknown}"
echo "  Log: $MULTIMODAL_LOG"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "[$(date)] Training script complete"
