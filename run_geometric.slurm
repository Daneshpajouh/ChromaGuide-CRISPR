#!/bin/bash
#SBATCH --job-name=Geometric_CRISPR
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/geometric_%j.log
#SBATCH --error=logs/geometric_%j.err

echo "=========================================="
echo "GEOMETRIC BIOTHERMODYNAMICS OPTIMIZATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Load modules
module load python/3.11
module load cuda/12

# Activate environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Fix: Uninstall Flash Attention/Triton to avoid version mismatch (trans_b error)
# DNABERT-2 will fall back to standard PyTorch SDPA (slower but stable)
pip uninstall -y flash_attn triton --quiet || true

# Set Cache Dirs to Scratch
export CACHE_DIR=/scratch/amird/CRISPRO-MAMBA-X/cache
export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export PIP_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
mkdir -p $TRITON_CACHE_DIR $XDG_CACHE_HOME $PIP_CACHE_DIR $HF_HOME

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Run Geometric Training
echo "Starting Natural Gradient Descent..."
python train_geometric.py

if [ $? -eq 0 ]; then
    echo "✅ Geometric training completed!"
else
    echo "❌ Training failed"
    exit 1
fi
