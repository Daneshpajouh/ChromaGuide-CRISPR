#!/bin/bash
# Download CRISPR benchmark datasets for ChromaGuide training

set -e

BASE_DIR="$HOME/Desktop/PhD/Proposal"
DATA_DIR="$BASE_DIR/data/benchmarks"

mkdir -p "$DATA_DIR"

echo "=== Downloading CRISPR Benchmark Datasets ==="

# DeepHF datasets (Wang et al. 2019)
echo "[1/4] Downloading DeepHF datasets..."
mkdir -p "$DATA_DIR/deephf"
cd "$DATA_DIR/deephf"

# CRISPRon dataset (2021)
echo "[2/4] Downloading CRISPRon datasets..."
mkdir -p "$DATA_DIR/crispron"
cd "$DATA_DIR/crispron"

# GUIDE-seq off-target data
echo "[3/4] Downloading GUIDE-seq off-target datasets..."
mkdir -p "$DATA_DIR/guideseq"
cd "$DATA_DIR/guideseq"

# CRISPOR benchmark (Haeussler et al.)
echo "[4/4] Downloading CRISPOR benchmark datasets..."
mkdir -p "$DATA_DIR/crispor"
cd "$DATA_DIR/crispor"

echo "=== Dataset Download Instructions ==="
echo ""
echo "DeepHF: github.com/izhangcd/DeepHF/tree/master/data"
echo "CRISPRon: rth.dk/resources/crispr/ (23,902 gRNAs)"
echo "GUIDE-seq: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4320685/"
echo "CRISPOR: crispor.tefor.net"
echo ""
echo "Manual download may be required for some datasets."
echo "Place datasets in: $DATA_DIR"
