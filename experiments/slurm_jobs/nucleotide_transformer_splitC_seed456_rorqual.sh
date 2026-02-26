#!/bin/bash
#SBATCH --job-name=cg-nucleo-C-s456
#SBATCH --account=def-kwiese_gpu
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=results/nucleotide_transformer_splitC_seed456/slurm-%j.out
#SBATCH --error=results/nucleotide_transformer_splitC_seed456/slurm-%j.err

# ChromaGuide: nucleotide_transformer | Split C | Seed 456
# Cluster: rorqual (redistributed from fir)

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "Backbone: nucleotide_transformer"
echo "Split: C"
echo "Seed: 456"
echo "===================="

mkdir -p results/nucleotide_transformer_splitC_seed456

cd ~/scratch/chromaguide_v2
source ~/scratch/chromaguide_v2_env/bin/activate
module load StdEnv/2023 python/3.11.5 scipy-stack 2>/dev/null

python experiments/train_experiment.py \
    --backbone nucleotide_transformer \
    --split C \
    --split-fold 0 \
    --seed 456 \
    --data-dir ~/scratch/chromaguide_v2/data \
    --output-dir ~/scratch/chromaguide_v2/results \
    --no-wandb

EXIT_CODE=$?

echo "===== JOB COMPLETE ====="
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

exit $EXIT_CODE
