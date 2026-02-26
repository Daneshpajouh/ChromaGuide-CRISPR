#!/bin/bash
# Submit all ChromaGuide jobs on fir
# Generated automatically

echo "Submitting 12 jobs on fir..."
cd ~/scratch/chromaguide_v2

echo "Submitting: nucleotide_transformer | Split A | Seed 42"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitA_seed42_fir.sh

echo "Submitting: nucleotide_transformer | Split A | Seed 456"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitA_seed456_fir.sh

echo "Submitting: nucleotide_transformer | Split B | Seed 42"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitB_seed42_fir.sh

echo "Submitting: nucleotide_transformer | Split B | Seed 456"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitB_seed456_fir.sh

echo "Submitting: nucleotide_transformer | Split C | Seed 42"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitC_seed42_fir.sh

echo "Submitting: nucleotide_transformer | Split C | Seed 456"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitC_seed456_fir.sh

echo "Submitting: evo | Split A | Seed 42"
sbatch experiments/slurm_jobs/evo_splitA_seed42_fir.sh

echo "Submitting: evo | Split A | Seed 456"
sbatch experiments/slurm_jobs/evo_splitA_seed456_fir.sh

echo "Submitting: evo | Split B | Seed 42"
sbatch experiments/slurm_jobs/evo_splitB_seed42_fir.sh

echo "Submitting: evo | Split B | Seed 456"
sbatch experiments/slurm_jobs/evo_splitB_seed456_fir.sh

echo "Submitting: evo | Split C | Seed 42"
sbatch experiments/slurm_jobs/evo_splitC_seed42_fir.sh

echo "Submitting: evo | Split C | Seed 456"
sbatch experiments/slurm_jobs/evo_splitC_seed456_fir.sh


echo ""
echo "All 12 jobs submitted on fir."
echo "Check status: squeue -u $USER"
