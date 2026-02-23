#!/bin/bash
# SLURM job script for FIR - V10 Off-Target Training
# Login: fir.alliancecan.ca
# Request: 1x node with H100 GPU

#SBATCH --job-name=v10_offtarget
#SBATCH --account=def-kwiese
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/v10_offtarget_%j.out
#SBATCH --error=logs/v10_offtarget_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amirdamird@sfu.ca
#SBATCH --mem=64G

set -euo pipefail

echo "=== FIR V10 OFF-TARGET TRAINING ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo ""

# Activate conda environment
# Load conda if available, otherwise use python directly
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate chromaguide
else
    echo "Conda not found, using system Python"
    source ~/chromaguide_venv/bin/activate 2>/dev/null || true
fi

# Navigate to project
cd ~/chromaguide

echo "Current directory: $(pwd)"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Run V10 off-target training
echo "Starting V10 off-target training..."
python3 -u scripts/train_off_target_v10.py \
    --data_path ~/chromaguide/data/raw/crisprofft/CRISPRoffT_all_targets.txt \
    --epochs 8 \
    --batch_size 128 \
    --n_models 5 \
    --output_prefix v10_offtarget

echo ""
echo "=== Training complete ==="
echo "Results saved to models/ and logs/"
