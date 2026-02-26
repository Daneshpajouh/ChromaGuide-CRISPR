#!/bin/bash
#SBATCH --job-name=cg-evo-B-s42
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/evo_splitB_seed42/slurm-%j.out
#SBATCH --error=results/evo_splitB_seed42/slurm-%j.err

# ChromaGuide: evo | Split B | Seed 42
# Cluster: narval (redistributed from fir)

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: evo"
echo "Split: B"
echo "Seed: 42"
echo "===================="

mkdir -p results/evo_splitB_seed42

cd ~/scratch/chromaguide_v2
source ~/scratch/chromaguide_v2_env/bin/activate
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null

python experiments/train_experiment.py \
    --backbone evo \
    --split B \
    --split-fold 0 \
    --seed 42 \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb

EXIT_CODE=$?

echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

exit $EXIT_CODE
