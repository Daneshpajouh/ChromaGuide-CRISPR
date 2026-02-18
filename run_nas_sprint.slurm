#!/bin/bash
#SBATCH --job-name=NAS_Sprint
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/nas_sprint_%j.log
#SBATCH --error=logs/nas_sprint_%j.err

echo "=========================================="
echo "NAS BASELINE SPRINT"
echo "Job ID: $SLURM_JOB_ID"
echo "Goal: Re-generate Baseline Predictions"
echo "=========================================="

# Load modules
module load python/3.11

# Activate environment
cd /scratch/amird/CRISPRO-MAMBA-X
source venv/bin/activate

# Run NAS Training (Lightweight)
# Using DNABERT-2 as the Evo2 proxy for this sprint
echo "Starting NAS Training..."
python train_rag_tta.py \
    --epochs 3 \
    --batch-size 32 \
    --output_dir ./results_nas \
    --use_tta False \
    --use_rag False

# Note: We reuse train_rag_tta.py but toggle off RAG/TTA to effectively run
# a pure fine-tuning baseline (mimicking the "NAS" output for ensemble)

if [ $? -eq 0 ]; then
    echo "✅ NAS Baseline completed!"
else
    echo "❌ NAS failed"
    exit 1
fi
