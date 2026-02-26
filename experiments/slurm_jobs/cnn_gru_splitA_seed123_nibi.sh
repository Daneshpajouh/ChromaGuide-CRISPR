#!/bin/bash
#SBATCH --job-name=cg-cnn_gr-A-s123
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/cnn_gru_splitA_seed123/slurm-%j.out
#SBATCH --error=results/cnn_gru_splitA_seed123/slurm-%j.err

# ChromaGuide: cnn_gru | Split A | Seed 123
# Cluster: nibi (redistributed from beluga)

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: cnn_gru"
echo "Split: A"
echo "Seed: 123"
echo "===================="

mkdir -p results/cnn_gru_splitA_seed123

cd ~/scratch/chromaguide_v2
source ~/scratch/chromaguide_v2_env/bin/activate
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null

python experiments/train_experiment.py \
    --backbone cnn_gru \
    --split A \
    --split-fold 0 \
    --seed 123 \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb

EXIT_CODE=$?

echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# Print results summary
if [ -f ~/scratch/chromaguide_v2/results/cnn_gru_splitA_seed123/metrics.json ]; then
    python -c "
import json
with open('~/scratch/chromaguide_v2/results/cnn_gru_splitA_seed123/metrics.json') as f:
    r = json.load(f)
print(f'Spearman: {r[\"spearman_rho\"]:.4f}')
print(f'AUROC:    {r[\"auroc\"]:.4f}')
print(f'Time:     {r[\"training_time_seconds\"]/60:.1f} min')
"
fi

exit $EXIT_CODE
