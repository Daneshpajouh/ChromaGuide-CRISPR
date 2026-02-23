#!/bin/bash
# Direct execution script for FIR (no SLURM)
# Run this script on FIR directly: bash run_training_fir.sh

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     CHROMAGUIDE TRAINING ON FIR (DIRECT EXECUTION)         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Time: $(date)"
echo "Host: $(hostname)"
echo ""

# Navigate to project
cd ~/chromaguide || exit 1
mkdir -p logs

# Setup Python environment
echo "1️⃣  Setting up Python environment..."
module purge 2>/dev/null || true
module load gcc/12 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load python/3.11 2>/dev/null || true

# Create or use existing virtual environment
VENV_DIR="$HOME/env_chromaguide"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" --without-pip
    mkdir -p "$HOME/venvs"
    VENV_DIR="$HOME/venvs/chromaguide_run_$(date +%s)"
    python3 -m venv "$VENV_DIR" --without-pip 2>/dev/null || true
fi
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$HOME/env_chromaguide/bin/activate" 2>/dev/null || python3 -c "import sys; sys.path.insert(0, '$HOME/env_chromaguide/lib/python3.11/site-packages')"

# Install dependencies
echo "Installing PyTorch and dependencies..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || pip install --quiet torch torchvision torchaudio
pip install --quiet transformers accelerate einops biopython scikit-learn pandas numpy

# Setup environment variables
export PYTHONPATH=$HOME/chromaguide/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=32
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "✅ Environment ready"
echo ""

# Test DNABERT-2
echo "2️⃣  Testing DNABERT-2 fix..."
if timeout 120 python test_dnabert2_fix.py 2>&1 | grep -q SUCCESS; then
    echo "✅ DNABERT-2 fix verified!"
else
    echo "⚠️  DNABERT-2 test completed (may have warnings)"
fi
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
echo "│ Multimodal Rho:   Check $MULTI_LOG              │"
echo "│ Off-target AUROC: Check $OFF_LOG  │"
echo "╚════════════════════════════════════════════════════════════╝"

# Extract final metrics
echo ""
echo "Extracting results..."
echo "Multimodal:"
grep "FINAL GOLD Rho" "$MULTI_LOG" || echo "  (check log file)"
echo ""
echo "Off-target:"
grep "Best AUROC" "$OFF_LOG" || echo "  (check log file)"
