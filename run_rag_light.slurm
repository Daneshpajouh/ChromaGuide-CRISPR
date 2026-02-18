#!/bin/bash
#SBATCH --job-name=RAG_Light
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/rag_light_%j.log
#SBATCH --error=logs/rag_light_%j.err

echo "=========================================="
echo "CRISPR RAG-TTA LIGHT (Single H100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Load modules (Exact match to geometric success)
module load python/3.11
module load cuda/12

# Activate environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Fix: Uninstall Flash Attention/Triton (Stability First)
pip uninstall -y flash_attn triton --quiet || true

# Set Cache Dirs to Scratch (Critical for Quota)
export CACHE_DIR=/scratch/amird/CRISPRO-MAMBA-X/cache
export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export PIP_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
mkdir -p $TRITON_CACHE_DIR $XDG_CACHE_HOME $PIP_CACHE_DIR $HF_HOME

# Run RAG Training
# Reduced batch size for single GPU
echo "Starting RAG Training..."
python train_rag_tta.py --batch-size 16 --epochs 10

if [ $? -eq 0 ]; then
    echo "✅ RAG training completed!"
else
    echo "❌ Training failed"
    exit 1
fi
