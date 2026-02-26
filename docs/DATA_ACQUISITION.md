# ChromaGuide Data Acquisition Guide

## Overview

ChromaGuide uses **real CRISPR-Cas9 on-target efficacy datasets** from peer-reviewed studies. The primary data source is the CRISPR-FMC benchmark collection (Xiang et al. 2025), which aggregates 9 publicly available datasets spanning multiple Cas9 variants, cell lines, and experimental scales.

## Dataset Sources

### Primary: CRISPR-FMC 9 Benchmark Datasets
- **Repository**: https://github.com/xx0220/CRISPR-FMC
- **Paper**: Xiang et al. (2025), "CRISPR-FMC: a dual-branch hybrid network for predicting CRISPR-Cas9 on-target activity", Frontiers in Genome Editing
- **Format**: CSV files with columns `sgRNA` (23-mer) and `indel` (efficacy score)
- **Total**: 291,639 sgRNA samples across 9 datasets

| Dataset | Samples | Cell Line | Cas9 Variant | Source Paper | Scale |
|---------|---------|-----------|--------------|-------------|-------|
| WT | 55,603 | HEK293T | WT-SpCas9 | Wang et al. 2019 | Large |
| ESP | 58,616 | HEK293T | eSpCas9(1.1) | Wang et al. 2019 | Large |
| HF | 56,887 | HEK293T | SpCas9-HF1 | Wang et al. 2019 | Large |
| xCas9 | 37,738 | HEK293T | xCas9 | Kim et al. 2020 | Medium |
| SpCas9-NG | 30,585 | HEK293T | SpCas9-NG | Kim et al. 2020 | Medium |
| Sniper | 37,794 | HEK293T | Sniper-Cas9 | Kim et al. 2020 | Medium |
| HCT116 | 4,239 | HCT116 | WT | Hart 2015 / Chuai 2018 | Small |
| HELA | 8,101 | HeLa | WT | Hart 2015 / Chuai 2018 | Small |
| HL60 | 2,076 | HL60 | WT | Wang T. 2014 / Chuai 2018 | Small |

### Supplementary: DeepHF Full Dataset
- **Repository**: https://github.com/Rafid013/CRISPRpredSEQ
- **File**: `deephf_data.xlsx` (5.3 MB)
- **Paper**: Wang et al. (2019), "Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning", Nature Communications 10:4284
- **Content**: 59,852 sgRNAs with WT, eSpCas9, SpCas9-HF1 efficacy values
- **Note**: The CRISPR-FMC WT/ESP/HF CSVs contain the same data as this xlsx

### ENCODE Epigenomic Tracks (Optional)
- **Source**: https://www.encodeproject.org/
- **Format**: bigWig files (hg38)
- **Tracks**: DNase-seq, H3K4me3, H3K27ac for HEK293T, HCT116, HeLa

## Data Preprocessing Pipeline

### Step 1: Download
```bash
# Automatic (on cluster login node):
python experiments/prepare_data.py --data-dir data/

# Manual (curl):
for ds in WT ESP HF HCT116 HELA HL60; do
    curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/${ds}.csv" \
        -o "data/raw/crispr_fmc/datasets/${ds}.csv"
done
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/xCas.csv" -o data/raw/crispr_fmc/datasets/xCas.csv
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/SpCas9-NG.csv" -o data/raw/crispr_fmc/datasets/SpCas9-NG.csv
curl -sL "https://raw.githubusercontent.com/xx0220/CRISPR-FMC/main/datasets/Sniper-Cas9.csv" -o data/raw/crispr_fmc/datasets/Sniper-Cas9.csv
```

### Step 2: Parse & Normalize
Each dataset CSV has columns: `sgRNA` (23-mer DNA), `indel` (raw efficacy)

**Normalization**: Per-dataset min-max normalization to [0, 1]:
- Large-scale (WT/ESP/HF): Already in [0, 1]
- Medium-scale (xCas9/SpCas9-NG/Sniper): Raw values up to ~75, needs min-max
- Small-scale (HCT116/HELA/HL60): Already in [0, 1]

After normalization, epsilon-clamping (1e-6) for Beta regression.

### Step 3: Gene Assignment
Since genomic coordinates are not available for all datasets, we use sequence-hash-based pseudo-gene assignment:
- Hash the 20nt protospacer (first 20 bases of 23-mer)
- Map to 2000 gene bins via `md5(seq) % 2000`
- This ensures consistent gene grouping for Split A (gene-held-out)
- After deduplication: 94,615 unique sequences, 2000 gene groups

### Step 4: Epigenomic Signals
- If ENCODE bigWig files available: extract real chromatin signals
- Otherwise: cell-line-aware synthetic signals (GC-content-modulated)
- Shape: (n_samples, 3, 100) — 3 tracks, 100 bins per track

### Step 5: Split Construction
Three leakage-controlled splits per the proposal:
- **Split A**: Gene-held-out (70/15/15 train/cal/test by gene group)
- **Split B**: Dataset-held-out (leave-one-dataset-out)
- **Split C**: Cell-line-held-out (leave-one-cell-line-out)

## Output Files
```
data/processed/
├── sequences.csv        # 291,639 rows (full), or 94,615 (deduplicated)
├── sequences.parquet    # Same, parquet format
├── efficacy.npy         # Float32 array, shape (N,)
├── epigenomic.npy       # Float32 array, shape (N, 3, 100)
└── splits/
    ├── split_a.npz      # Gene-held-out indices
    ├── split_b_0.npz    # Dataset-held-out fold 0
    ├── ...
    ├── split_c_0.npz    # Cell-line-held-out fold 0
    ├── ...
    └── cv_5x2_0.npz     # 5×2cv for statistical testing
```

## References

1. Xiang et al. (2025). "CRISPR-FMC: a dual-branch hybrid network for predicting CRISPR-Cas9 on-target activity." Frontiers in Genome Editing.
2. Wang et al. (2019). "Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning." Nature Communications 10:4284.
3. Kim et al. (2020). "SpCas9 activity prediction by DeepSpCas9, a deep learning–based model with high generalization performance." Science Advances.
4. Xiang et al. (2021). "Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep transfer learning." Nature Communications 12:3238.
5. Hart et al. (2015). "High-Resolution CRISPR Screens Reveal Fitness Genes and Genotype-Specific Cancer Liabilities." Cell 163:1515.
6. Chuai et al. (2018). "DeepCRISPR: optimized CRISPR guide RNA design by deep learning." Genome Biology 19:80.
