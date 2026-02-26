#!/bin/bash
# Submit all ChromaGuide jobs on rorqual
# Generated automatically

echo "Submitting 9 jobs on rorqual..."
cd ~/scratch/chromaguide_v2

echo "Submitting: dnabert2 | Split A | Seed 42"
sbatch experiments/slurm_jobs/dnabert2_splitA_seed42_rorqual.sh

echo "Submitting: dnabert2 | Split A | Seed 456"
sbatch experiments/slurm_jobs/dnabert2_splitA_seed456_rorqual.sh

echo "Submitting: dnabert2 | Split B | Seed 42"
sbatch experiments/slurm_jobs/dnabert2_splitB_seed42_rorqual.sh

echo "Submitting: dnabert2 | Split B | Seed 456"
sbatch experiments/slurm_jobs/dnabert2_splitB_seed456_rorqual.sh

echo "Submitting: dnabert2 | Split C | Seed 42"
sbatch experiments/slurm_jobs/dnabert2_splitC_seed42_rorqual.sh

echo "Submitting: dnabert2 | Split C | Seed 456"
sbatch experiments/slurm_jobs/dnabert2_splitC_seed456_rorqual.sh

echo "Submitting: nucleotide_transformer | Split A | Seed 123"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitA_seed123_rorqual.sh

echo "Submitting: nucleotide_transformer | Split B | Seed 123"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitB_seed123_rorqual.sh

echo "Submitting: nucleotide_transformer | Split C | Seed 123"
sbatch experiments/slurm_jobs/nucleotide_transformer_splitC_seed123_rorqual.sh


echo ""
echo "All 9 jobs submitted on rorqual."
echo "Check status: squeue -u $USER"
