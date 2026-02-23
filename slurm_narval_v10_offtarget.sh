#!/bin/bash
#SBATCH --account=def-kwiese
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=v10_offtarget
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Narval V10 Off-Target Training Script
# Purpose: Train corrected V10 off-target classifier with per-mark epigenetic gating
# Expected runtime: ~1.5-2 hours for 5 models × 8 epochs

echo "=== NARVAL V10 OFF-TARGET TRAINING ==="
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

# Verify dataset exists
if [ ! -f "data/raw/crisprofft/CRISPRoffT_all_targets.txt" ]; then
    echo "ERROR: Dataset not found at data/raw/crisprofft/CRISPRoffT_all_targets.txt"
    exit 1
fi

echo "Dataset verified: $(wc -l < data/raw/crisprofft/CRISPRoffT_all_targets.txt) sequences"

# Create output directory
mkdir -p logs

# Copy corrected script from local
if [ -f "../../Desktop/PhD/Proposal/scripts/train_off_target_v10.py" ]; then
    echo "Updating train_off_target_v10.py with corrected architecture..."
    cp ../../Desktop/PhD/Proposal/scripts/train_off_target_v10.py scripts/
fi

# Launch training with corrected V10 architecture
echo "Launching V10 off-target training with corrected PerMarkEpigenicGating..."
python3 -u scripts/train_off_target_v10.py \
    > logs/off_target_v10_corrected_narval.log 2>&1

echo "Training complete. Results saved to logs/"

# Report results
if [ -f "models/off_target_v10_ensemble.pt" ]; then
    echo "✓ Ensemble model saved successfully"
else
    echo "✗ Warning: Ensemble model not found"
fi

echo "Job completed: $(date)"
exit 0
