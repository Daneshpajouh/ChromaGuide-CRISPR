#!/bin/bash
#SBATCH --job-name=cg-dnaber-B-s123
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/dnabert2_splitB_seed123/slurm-%j.out
#SBATCH --error=results/dnabert2_splitB_seed123/slurm-%j.err

# ChromaGuide: dnabert2 | Split B | Seed 123
# Cluster: rorqual (redistributed from killarney)

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: dnabert2"
echo "Split: B"
echo "Seed: 123"
echo "===================="

mkdir -p results/dnabert2_splitB_seed123

cd ~/scratch/chromaguide_v2
source ~/scratch/chromaguide_v2_env/bin/activate
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null

python experiments/train_experiment.py \
    --config configs/dnabert2.yaml \
    --split B \
    --seed 123 \
    --output_dir results/dnabert2_splitB_seed123

EXIT_CODE=$?

echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

# Print results summary
if [ -f results/dnabert2_splitB_seed123/metrics.json ]; then
    python -c "
import json
with open('results/dnabert2_splitB_seed123/metrics.json') as f:
    r = json.load(f)
print(f'Spearman: {r[\"spearman_rho\"]:.4f}')
print(f'AUROC:    {r[\"auroc\"]:.4f}')
print(f'Time:     {r[\"training_time_seconds\"]/60:.1f} min')
"
fi

exit $EXIT_CODE
