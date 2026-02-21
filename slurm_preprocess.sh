#!/bin/bash
#SBATCH --job-name=chromaguide_preprocess
#SBATCH --account=def-fcorry
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

set -e

# Load modules
module load python/3.11
module load bowtie2/2.5.4

# Activate virtualenv
source ~/cg_env/bin/activate

# Paths
INPUT_CSV="merged.csv"
BT2_INDEX="/cvmfs/soft.computecanada.ca/ref/genome/hg38/bowtie2/GRCh38_noalt_as"
BW_DNASE="data/real/raw/encode/dnase.bigWig"
BW_H3K4ME3="data/real/raw/encode/h3k4me3.bigWig"
BW_H3K27AC="data/real/raw/encode/h3k27ac.bigWig"
OUTPUT_H5="data/real/processed/multimodal_data.h5"

mkdir -p data/real/processed

echo "Starting preprocessing at $(date)"
python scripts/preprocess_epigenomics.py \
    --input $INPUT_CSV \
    --bowtie2_index $BT2_INDEX \
    --bw_dnase $BW_DNASE \
    --bw_h3k4me3 $BW_H3K4ME3 \
    --bw_h3k27ac $BW_H3K27AC \
    --output $OUTPUT_H5 \
    --threads $SLURM_CPUS_PER_TASK \
    --window 5000 \
    --bins 100

echo "Finished preprocessing at $(date)"
