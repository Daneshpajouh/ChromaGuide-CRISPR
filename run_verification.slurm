#!/bin/bash
#SBATCH --job-name=Verify_Geo
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=verify.log
#SBATCH --error=verify.err

cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate

# Cache setup (Vital for HF)
export CACHE_DIR=/scratch/amird/CRISPRO-MAMBA-X/cache
export PID_CACHE_DIR=$CACHE_DIR/pip
export HF_HOME=$CACHE_DIR/huggingface
mkdir -p $HF_HOME

echo "Starting Manual Verification..."
python verify_checkpoint.py
