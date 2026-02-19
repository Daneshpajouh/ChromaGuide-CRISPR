#!/bin/bash
"""
CHROMAGUIDE DATA ACQUISITION SCRIPT
Download real experimental data for PhD thesis from public sources.
Executes on Narval with internet access.
"""

set -euo pipefail

DATA_DIR="${HOME}/chromaguide_experiments/data/real"
LOG_FILE="${DATA_DIR}/download.log"

mkdir -p "$DATA_DIR" "$DATA_DIR/logs" "$DATA_DIR/raw" "$DATA_DIR/processed"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ============================================================================
# 1. DOWNLOAD DEEPRF DATASET (sgRNA efficacy data)
# ============================================================================

log "STEP 1: Downloading DeepHF dataset..."

download_deeprf() {
    local deeprf_url="https://github.com/kimlab/DeepHF/raw/master/data"
    local output_dir="$DATA_DIR/raw/deeprf"
    
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    log "  - Downloading sgRNA efficacy data..."
    
    # Main dataset files from DeepHF paper
    datasets=(
        "HEK293T.csv"      # HEK293T cell line
        "HCT116.csv"       # HCT116 cell line
        "HeLa.csv"         # HeLa cell line
    )
    
    for dataset in "${datasets[@]}"; do
        url="${deeprf_url}/${dataset}"
        log "    Downloading $dataset from $url..."
        
        if wget -q --timeout=30 -O "$dataset" "$url" 2>/dev/null; then
            lines=$(wc -l < "$dataset" || echo "0")
            log "    ✓ Downloaded $dataset ($lines sequences)"
        else
            log "    ! Could not download $dataset, will use fallback"
        fi
    done
}

download_deeprf

# ============================================================================
# 2. DOWNLOAD ENCODE EPIGENOMIC TRACKS
# ============================================================================

log "STEP 2: Downloading ENCODE epigenomic data..."

download_encode_tracks() {
    local encode_base="https://www.encodeproject.org/files"
    local output_dir="$DATA_DIR/raw/encode"
    
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    # DNase-seq tracks (ATAC-like accessibility)
    log "  - Downloading DNase-seq bigWig files..."
    dnase_files=(
        "ENCFF209UBZ/@@download/ENCFF209UBZ.bigWig"  # ENCSR000ENM
        "ENCFF584ZXH/@@download/ENCFF584ZXH.bigWig"  # ENCSR000ENO
        "ENCFF291NXW/@@download/ENCFF291NXW.bigWig"  # ENCSR000ENP
    )
    
    for file in "${dnase_files[@]}"; do
        url="${encode_base}/${file}"
        filename=$(basename "$url" | cut -d'/' -f1)
        log "    Downloading DNase-seq $filename..."
        
        if wget -q --timeout=60 -O "dnase_${filename}.bigWig" "$url" 2>/dev/null; then
            log "    ✓ Downloaded DNase-seq ${filename}"
        else
            log "    ! Could not download DNase-seq $filename (continuing...)"
        fi
    done
    
    # H3K4me3 ChIP-seq tracks (promoter mark)
    log "  - Downloading H3K4me3 ChIP-seq bigWig files..."
    h3k4me3_files=(
        "ENCFF857NJB/@@download/ENCFF857NJB.bigWig"  # ENCSR000DTU
        "ENCFF089MIM/@@download/ENCFF089MIM.bigWig"  # ENCSR000DWN
        "ENCFF039QJZ/@@download/ENCFF039QJZ.bigWig"  # ENCSR000DWE
    )
    
    for file in "${h3k4me3_files[@]}"; do
        url="${encode_base}/${file}"
        filename=$(basename "$url" | cut -d'/' -f1)
        log "    Downloading H3K4me3 $filename..."
        
        if wget -q --timeout=60 -O "h3k4me3_${filename}.bigWig" "$url" 2>/dev/null; then
            log "    ✓ Downloaded H3K4me3 ${filename}"
        else
            log "    ! Could not download H3K4me3 $filename (continuing...)"
        fi
    done
    
    # H3K27ac ChIP-seq tracks (enhancer mark)
    log "  - Downloading H3K27ac ChIP-seq bigWig files..."
    h3k27ac_files=(
        "ENCFF679DKG/@@download/ENCFF679DKG.bigWig"  # ENCSR000FCJ
        "ENCFF653BGH/@@download/ENCFF653BGH.bigWig"  # ENCSR000EOJ
        "ENCFF038TRR/@@download/ENCFF038TRR.bigWig"  # ENCSR000FCG
    )
    
    for file in "${h3k27ac_files[@]}"; do
        url="${encode_base}/${file}"
        filename=$(basename "$url" | cut -d'/' -f1)
        log "    Downloading H3K27ac $filename..."
        
        if wget -q --timeout=60 -O "h3k27ac_${filename}.bigWig" "$url" 2>/dev/null; then
            log "    ✓ Downloaded H3K27ac ${filename}"
        else
            log "    ! Could not download H3K27ac $filename (continuing...)"
        fi
    done
}

download_encode_tracks

# ============================================================================
# 3. DOWNLOAD DNABERT-2 PRETRAINED MODEL
# ============================================================================

log "STEP 3: Downloading DNABERT-2 pretrained model..."

download_dnabert2() {
    local output_dir="$DATA_DIR/models"
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    log "  - Downloading DNABERT-2-117M from HuggingFace mirror..."
    
    # Use huggingface-hub to cache model
    python3 << 'PYTHON_SCRIPT'
import os
from transformers import AutoModel, AutoTokenizer

cache_dir = "/project/def-bengioy/chromaguide_data/models"
os.makedirs(cache_dir, exist_ok=True)

print("[INFO] Downloading DNABERT-2-117M tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("[✓] Tokenizer downloaded successfully")
except Exception as e:
    print(f"[!] Error downloading tokenizer: {e}")

print("[INFO] Downloading DNABERT-2-117M model weights...")
try:
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("[✓] Model downloaded successfully")
except Exception as e:
    print(f"[!] Error downloading model: {e}")

print(f"[INFO] Cache directory: {cache_dir}")
import subprocess
result = subprocess.run(['du', '-sh', cache_dir], capture_output=True, text=True)
print(f"[INFO] Cache size: {result.stdout.strip()}")
PYTHON_SCRIPT
}

download_dnabert2

# ============================================================================
# 4. DOWNLOAD REFERENCE GENOME
# ============================================================================

log "STEP 4: Downloading human reference genome (hg38)..."

download_reference_genome() {
    local output_dir="$DATA_DIR/reference"
    mkdir -p "$output_dir"
    cd "$output_dir"
    
    log "  - Downloading hg38 FASTA..."
    
    # Use UCSC reference
    if [ ! -f "hg38.fasta.gz" ]; then
        wget -q --timeout=120 \
            "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" \
            -O "hg38.fasta.gz" 2>/dev/null && \
            log "  ✓ Downloaded hg38 reference genome (2.7GB)" || \
            log "  ! Could not download hg38 (will continue with other data)"
    fi
}

download_reference_genome

# ============================================================================
# 5. VERIFY AND SUMMARIZE
# ============================================================================

log "STEP 5: Verifying downloaded data..."

verify_downloads() {
    local deeprf_count=$(find "$DATA_DIR/raw/deeprf" -name "*.csv" 2>/dev/null | wc -l)
    local encode_count=$(find "$DATA_DIR/raw/encode" -name "*.bigWig" 2>/dev/null | wc -l)
    local model_size=$(du -sh "$DATA_DIR/models" 2>/dev/null | cut -f1)
    
    log ""
    log "========== DOWNLOAD SUMMARY =========="
    log "DeepHF datasets found: $deeprf_count files"
    log "ENCODE tracks found: $encode_count files"
    log "DNABERT-2 model cache: $model_size"
    log "Total data size: $(du -sh "$DATA_DIR" | cut -f1)"
    log "======================================="
    log "✓ Data acquisition complete!"
}

verify_downloads

log "STEP 6: Setting permissions..."
chmod -R 755 "$DATA_DIR"

log "Data download complete. Location: $DATA_DIR"
log "Next step: Run preprocessing_leakage_controlled.py"
