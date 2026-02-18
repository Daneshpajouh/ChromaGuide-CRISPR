#!/bin/bash
#SBATCH --job-name=CRISPR_RAG_TTA
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/rag_tta_%j.log
#SBATCH --error=logs/rag_tta_%j.err

echo "=========================================="
echo "CRISPR-RAG-TTA Training on H100"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

# Load modules
module load python/3.11
module load cuda/12

# Activate environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Fix: Uninstall Flash Attention/Triton to avoid version mismatch
pip uninstall -y flash_attn triton --quiet || true

# Set Cache Dirs to Scratch
export CACHE_DIR=/scratch/amird/CRISPRO-MAMBA-X/cache
export TRITON_CACHE_DIR=$CACHE_DIR/triton
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export PIP_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
mkdir -p $TRITON_CACHE_DIR $XDG_CACHE_HOME $PIP_CACHE_DIR $HF_HOME

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH"FAISS install skipped"

# Print system info
echo "Python: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "=========================================="
echo "Starting RAG-TTA Training..."
echo "=========================================="

# Train with full dataset
python train_rag_tta.py \
    --data ./data/merged_crispr_data.csv \
    --epochs 50 \
    --batch-size 16

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code $?"
    exit 1
fi

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
