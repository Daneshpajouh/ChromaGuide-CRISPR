# ChromaGuide v2 — Technical Architecture

## Overview

ChromaGuide is a 3-module framework for CRISPR-Cas9 guide RNA design:

1. **On-Target Efficacy Module** — Predicts cutting efficiency (Spearman ρ ≥ 0.91)
2. **Off-Target Prediction Module** — Classifies off-target risk (AUROC ≥ 0.92)
3. **Integrated Design Score** — Combines on/off-target with conformal calibration

## Module Architecture

```
Input: 23-nt sgRNA sequence + optional epigenomic features
       │
       ▼
┌─────────────────────────────────────────┐
│        Sequence Encoder (Backbone)       │
│  CNN-GRU │ Caduceus │ DNABERT-2 │ NT │ Evo │
└───────────────┬─────────────────────────┘
                │ sequence embedding
                ▼
┌─────────────────────────────────────────┐
│       Epigenomic Encoder (optional)      │
│         MLP on chromatin features        │
└───────────────┬─────────────────────────┘
                │ epigenomic embedding
                ▼
┌─────────────────────────────────────────┐
│          Cross-Attention Fusion          │
│     Fuses sequence + epigenomic info     │
└───────────────┬─────────────────────────┘
                │ fused representation
                ▼
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  On-Target   │  │  Off-Target  │
│  Pred. Head  │  │  Pred. Head  │
│  (Regression)│  │  (Binary)    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
  Efficacy Score    Off-target Risk
       │                 │
       └────────┬────────┘
                ▼
┌─────────────────────────────────────────┐
│       Conformal Prediction Module        │
│  Calibrated uncertainty quantification   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│        Integrated Design Score           │
│   Weighted combination with penalties    │
└─────────────────────────────────────────┘
```

## Backbone Encoders

| Backbone | Parameters | Embedding Dim | Pre-trained Model | Fine-tuning |
|----------|-----------|---------------|-------------------|-------------|
| CNN-GRU | ~2M | 512 | None (trained from scratch) | Full |
| Caduceus | ~7M | 256 | `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` | LoRA r=8 |
| DNABERT-2 | ~117M | 768 | `zhihan1996/DNABERT-2-117M` | LoRA r=16 |
| Nucleotide Transformer | ~500M | 1024 | `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | LoRA r=16 |
| Evo | ~14M | 512 | `togethercomputer/evo-1-131k-base` | LoRA r=8 |

## Source Code Layout

```
chromaguide/
├── configs/           # YAML configs per backbone
├── data/              # Dataset loading, preprocessing, splits
│   ├── acquire.py     # Download DeepHF, CRISPRon, ENCODE
│   ├── dataset.py     # CRISPRDataset (PyTorch)
│   ├── preprocess.py  # Feature engineering
│   └── splits.py      # Gene-held-out cross-validation
├── evaluation/        # Metrics, statistical tests, thesis outputs
├── models/            # ChromaGuide main model, off-target, design score
├── modules/           # Encoder, fusion, prediction, conformal
├── training/          # Trainer, HPO, losses
└── utils/             # Config, logging, reproducibility
```

## Experiment Infrastructure

```
experiments/
├── train_experiment.py        # Master training script
├── prepare_data.py            # Synthetic/real data preparation
├── collect_results.py         # Aggregate results across experiments
├── generate_slurm_jobs.py     # Generate SLURM scripts for all 45 jobs
├── monitor_jobs.sh            # Job monitoring helper
└── slurm_jobs/                # Generated SLURM scripts (45 + submit helpers)
```

## Configuration System

Uses OmegaConf for hierarchical YAML configs. Example:

```yaml
# chromaguide/configs/dnabert2.yaml
backbone: dnabert2
model:
  pretrained: zhihan1996/DNABERT-2-117M
  embedding_dim: 768
  lora_rank: 16
training:
  learning_rate: 2e-5
  batch_size: 32
  max_epochs: 100
  patience: 10
  scheduler: cosine_warmup
```

Override via CLI:
```bash
python experiments/train_experiment.py \
    --config chromaguide/configs/dnabert2.yaml \
    --split A --seed 42 --output_dir results/dnabert2_splitA_seed42
```

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Spearman ρ | ≥ 0.91 | Rank correlation on gene-held-out test set |
| AUROC | ≥ 0.92 | Off-target binary classification |
| ECE | < 0.05 | Expected Calibration Error |
| Conformal Coverage | 90% ± 2% | Prediction interval coverage |
| p-value | < 0.001 | Wilcoxon signed-rank test vs. baselines |

## Datasets

### Phase 1 (Current): Synthetic Data
- 77,902 synthetic CRISPR sequences
- Used for architecture validation and debugging

### Phase 2 (Next): Real Datasets
| Dataset | Size | Task | Source |
|---------|------|------|--------|
| DeepHF | ~50K | On-target efficacy | Wang et al., 2019 |
| CRISPRon | ~15K | On-target efficacy | Xiang et al., 2021 |
| ENCODE | — | Epigenomic features | ENCODE Project |
| GUIDE-seq / CIRCLE-seq | ~500K | Off-target | Tsai et al., 2015/2017 |
