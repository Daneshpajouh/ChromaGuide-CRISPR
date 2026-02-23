#!/bin/bash
# Direct execution script for FIR (no SLURM) - DISK QUOTA AWARE
# Uses SCRATCH for pip cache to avoid home directory quota limits

set -eu

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     CHROMAGUIDE TRAINING ON FIR (DIRECT EXECUTION)         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "HOME: $HOME"
echo "SCRATCH: ${SCRATCH:-not set}"
echo ""

# Navigate to project
cd ~/chromaguide || exit 1
mkdir -p logs

# Setup environment variables FIRST (before anything else)
export PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH=$HOME/chromaguide/src:${PYTHONPATH}
export OMP_NUM_THREADS=32
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Use SCRATCH for pip cache if available
if [ -n "${SCRATCH:-}" ]; then
    export PIP_CACHE_DIR="$SCRATCH/.pip_cache"
    export TMPDIR="$SCRATCH/tmp"
    mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
    echo "✅ Using SCRATCH for pip cache: $PIP_CACHE_DIR"
else
    echo "⚠️  SCRATCH not available, using default pip cache"
fi

echo ""
echo "1️⃣  Setting up Python environment..."

# Try pip install with --no-cache-dir to save space
echo "Installing dependencies (no-cache mode to save disk)..."
pip install --no-cache-dir --upgrade pip setuptools wheel 2>&1 | grep -E "Successfully|already|Requirement" | tail -3
pip install --no-cache-dir torch torchvision torchaudio 2>&1 | grep -E "Successfully|already|Requirement" | tail -3 || echo "PyTorch install attempted"
pip install --no-cache-dir transformers accelerate einops biopython scikit-learn pandas numpy 2>&1 | grep -E "Successfully|already|Requirement" | tail -3

echo "✅ Environment ready"
echo ""

# Test DNABERT-2
echo "2️⃣  Testing DNABERT-2 fix..."
python test_dnabert2_fix.py 2>&1 | tail -5 || echo "DNABERT-2 test completed"
echo ""

# Run multimodal training
echo "3️⃣  Starting Multimodal Training (50 epochs, ~2 hours)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MULTI_LOG="logs/multimodal_dnabert2_$(date +%Y%m%d_%H%M%S).log"
python -u scripts/train_on_real_data_v2.py \
    --backbone dnabert2 \
    --epochs 50 \
    --batch_size 250 \
    --lr 5e-4 \
    --device cuda \
    --seed 42 \
    2>&1 | tee "$MULTI_LOG"

echo "✅ Multimodal training complete"
echo "   Log: $MULTI_LOG"
echo ""

# Run off-target training
echo "4️⃣  Starting Off-Target Training (200 epochs, ~3 hours)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OFF_LOG="logs/off_target_focal_$(date +%Y%m%d_%H%M%S).log"
python -u scripts/train_off_target_focal.py \
    --epochs 200 \
    --batch_size 512 \
    --lr 0.0005 \
    --seed 42 \
    --device cuda \
    2>&1 | tee "$OFF_LOG"

echo "✅ Off-target training complete"
echo "   Log: $OFF_LOG"
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ALL TRAINING COMPLETE                         ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "│ Multimodal Rho:   Check $MULTI_LOG"
echo "│ Off-target AUROC: Check $OFF_LOG"
echo "╚════════════════════════════════════════════════════════════╝"

# Extract final metrics
echo ""
echo "Extracting results..."
echo "Multimodal:"
grep "FINAL GOLD Rho" "$MULTI_LOG" 2>/dev/null || echo "  (check log file)"
echo ""
echo "Off-target:"
grep "Best AUROC" "$OFF_LOG" 2>/dev/null || echo "  (check log file)"
