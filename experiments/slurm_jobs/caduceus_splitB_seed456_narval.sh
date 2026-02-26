#!/bin/bash
#SBATCH --job-name=cg-caduce-B-s456
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/caduceus_splitB_seed456/slurm-%j.out
#SBATCH --error=results/caduceus_splitB_seed456/slurm-%j.err

# ChromaGuide: caduceus | Split B | Seed 456
# Cluster: narval (redistributed from beluga)

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: caduceus"
echo "Split: B"
echo "Seed: 456"
echo "===================="

mkdir -p results/caduceus_splitB_seed456

cd ~/scratch/chromaguide_v2
source ~/scratch/chromaguide_v2_env/bin/activate
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null

python experiments/train_experiment.py \
    --backbone caduceus \
    --split B \
    --split-fold 0 \
    --seed 456 \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb

EXIT_CODE=$?

echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# Print results summary
if [ -f ~/scratch/chromaguide_v2/results/caduceus_splitB_seed456/metrics.json ]; then
    python -c "
import json
with open('~/scratch/chromaguide_v2/results/caduceus_splitB_seed456/metrics.json') as f:
    r = json.load(f)
print(f'Spearman: {r[\"spearman_rho\"]:.4f}')
print(f'AUROC:    {r[\"auroc\"]:.4f}')
print(f'Time:     {r[\"training_time_seconds\"]/60:.1f} min')
"
fi

exit $EXIT_CODE
