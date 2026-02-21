#!/bin/bash
#SBATCH --job-name=preprocess_cg
#SBATCH --account=def-cloze
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

module load bowtie2/2.5.4
module load python/3.10
source ~/cg_env/bin/activate

# Paths
INPUT="/home/amird/chromaguide_experiments/data/real/raw/merged.csv"
BT2_INDEX="/home/amird/chromaguide_experiments/data/reference/GRCh38_noalt_as/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bowtie2"
BW_DNASE="/home/amird/chromaguide_experiments/data/real/raw/encode/ENCFF603BDB_dnase.bigWig"
BW_H3K4ME3="/home/amird/chromaguide_experiments/data/real/raw/encode/ENCFF292HLV_h3k4me3.bigWig"
BW_H3K27AC="/home/amird/chromaguide_experiments/data/real/raw/encode/ENCFF651XQU_h3k27ac.bigWig"
OUTPUT="/home/amird/chromaguide_experiments/data/real/processed/enriched_multimodal.h5"

mkdir -p /home/amird/chromaguide_experiments/data/real/processed
mkdir -p logs

echo "Starting preprocessing at $(date)"
python scripts/preprocess_epigenomics.py \
    --input "$INPUT" \
    --bowtie2_index "$BT2_INDEX" \
    --bw_dnase "$BW_DNASE" \
    --bw_h3k4me3 "$BW_H3K4ME3" \
    --bw_h3k27ac "$BW_H3K27AC" \
    --output "$OUTPUT" \
    --threads 16 \
    --window 5000 \
    --bins 100

echo "Finished preprocessing at $(date)"
