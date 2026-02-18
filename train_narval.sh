#!/bin/bash
#SBATCH --job-name=dnabert_mamba_phase1
#SBATCH --account=def-kwiese
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --mail-user=amird@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# DNABERT-Mamba Phase 1 Training on Narval Cluster
# ============================================================================
# Script: train_narval.sh
# Purpose: Train DNABERT-2 + Mamba model for sgRNA efficiency prediction
# Cluster: Narval (GPU-enabled compute nodes)
# Date: 2026-02-17
# ============================================================================

# Set error handling
set -euo pipefail

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "=========================================="
echo "DNABERT-Mamba Phase 1 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"
echo "=========================================="

# Create logging directory
mkdir -p logs

# Set up Python environment
module load python/3.11
module load cuda/12.2

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing/updating dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# ============================================================================
# DATA PREPARATION
# ============================================================================

echo "Checking data availability..."
if [ ! -d "data/processed" ]; then
    echo "ERROR: Processed data not found. Run download_datasets.py first."
    exit 1
fi

echo "Data files:"
ls -lh data/processed/ | head -10

# ============================================================================
# MODEL TRAINING
# ============================================================================

echo "Starting DNABERT-Mamba Phase 1 training..."
echo "Training configuration:"
echo "  - Model: DNABERT-2 + Mamba"
echo "  - Task: sgRNA efficiency prediction"
echo "  - Batch size: 32"
echo "  - Learning rate: 5e-5"
echo "  - Epochs: 10"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Run training script
python train_phase1.py \
    --model_name "zhihan1996/dnabert-2-117m" \
    --train_data "data/processed/train.csv" \
    --val_data "data/processed/val.csv" \
    --test_data "data/processed/test.csv" \
    --output_dir "checkpoints/phase1_narval_${SLURM_JOB_ID}" \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --logging_steps 100 \
    --save_steps 500 \
    --eval_steps 100 \
    --max_length 512 \
    --use_mamba \
    --mamba_hidden_dim 768 \
    --use_gpu \
    --mixed_precision fp16 2>&1 | tee logs/phase1_${SLURM_JOB_ID}.log

# ============================================================================
# POST-TRAINING EVALUATION
# ============================================================================

echo "Phase 1 Training Complete!"
echo "Results saved to: checkpoints/phase1_narval_${SLURM_JOB_ID}"
echo ""
echo "Next steps:"
echo "  1. Evaluate Phase 1 results"
echo "  2. Prepare Phase 2: Transfer learning"
echo "  3. Submit Phase 2 training job"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
echo "Job Summary:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Status: COMPLETED"
echo "  Log: logs/phase1_${SLURM_JOB_ID}.out"
echo "  Output: checkpoints/phase1_narval_${SLURM_JOB_ID}"
echo "=========================================="

# Deactivate virtual environment
deactivate

exit 0
