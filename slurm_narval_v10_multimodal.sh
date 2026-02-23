#!/bin/bash
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=v10_multimodal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Narval V10 Multimodal (On-Target) Training Script
# Purpose: Train corrected V10 on-target efficacy predictor with per-mark epigenetic gating
# Expected runtime: ~2-3 hours for DeepFusion + per-mark gating

echo "=== NARVAL V10 MULTIMODAL (ON-TARGET) TRAINING ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
module load cuda/12.2
source ~/env/chromaguide/bin/activate

# Set Python path
export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

# Navigate to project
cd ~/chromaguide_experiments

# Verify datasets exist
SPLITS=("split_a_gene_held_out" "split_b_dataset_held_out")
for split in "${SPLITS[@]}"; do
    if [ ! -d "data/processed/$split" ]; then
        echo "ERROR: Dataset not found at data/processed/$split"
        exit 1
    fi
done

echo "Datasets verified:"
find data/processed -name "*_train.csv" | wc -l
echo " cell type × split combinations found"

# Create output directory
mkdir -p logs

# Copy corrected script from local
if [ -f "../../Desktop/PhD/Proposal/scripts/train_on_real_data_v10.py" ]; then
    echo "Updating train_on_real_data_v10.py with corrected architecture..."
    cp ../../Desktop/PhD/Proposal/scripts/train_on_real_data_v10.py scripts/
fi

# Launch training on split A with corrected V10 architecture
echo "Launching V10 multimodal training with corrected PerMarkEpigenicGating..."
python3 -u scripts/train_on_real_data_v10.py \
    --split split_a_gene_held_out \
    > logs/multimodal_v10_corrected_narval.log 2>&1

echo "Training complete. Results saved to logs/"

# Report results
if [ -f "models/multimodal_v10_splitA_ensemble.pt" ]; then
    echo "✓ Ensemble model saved successfully"
else
    echo "✗ Warning: Ensemble model not found"
fi

echo "Job completed: $(date)"
exit 0
