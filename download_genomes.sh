#!/bin/bash
# Download reference genomes for CRISPRO-MAMBA-X on nibi cluster

set -e

GENOME_DIR="/scratch/amird/CRISPRO-MAMBA-X/data/raw/genomes"
mkdir -p $GENOME_DIR/hg38 $GENOME_DIR/mm10

echo "=== Downloading Human Genome (hg38) ==="
cd $GENOME_DIR/hg38

# Download hg38 from UCSC (chromosome by chromosome for reliability)
wget --quiet --show-progress -O hg38.fa.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

echo "=== Downloading Mouse Genome (mm10) ==="
cd $GENOME_DIR/mm10

# Download mm10 from UCSC
wget --quiet --show-progress -O mm10.fa.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz

echo "=== Genome Downloads Complete ==="
ls -lh $GENOME_DIR/hg38/
ls -lh $GENOME_DIR/mm10/

echo "=== Verifying integrity ==="
gunzip -t $GENOME_DIR/hg38/hg38.fa.gz && echo "hg38: OK"
gunzip -t $GENOME_DIR/mm10/mm10.fa.gz && echo "mm10: OK"

echo "=== All genomes ready for training ==="
