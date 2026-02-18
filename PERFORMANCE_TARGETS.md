# ChromaGuide Performance Targets & Strategy

## MUST SURPASS ALL SOTA MODELS

### Current SOTA Benchmarks (as of 2024-2025)

#### ON-TARGET Prediction:
| Model | Dataset | Spearman | Year |
|-------|---------|----------|------|
| **DeepCRISTL** | DeepHF (HT) | **0.880** | 2024 |
| CRISeq | WT-SpCas9 | 0.877 | 2024 |
| C-RNNCrispr | DeepHF | 0.877 | 2020 |
| DeepMEns | DeepHF | 0.874 | 2024 |
| **DeepCRISTL** | Endogenous | **0.556** | 2024 |

#### OFF-TARGET Prediction:
| Model | Dataset | AUROC | PR-AUC | Year |
|-------|---------|-------|--------|------|
| **CRISPR-MCA** | GUIDE-seq | **0.9853** | **0.8668** | 2024 |
| CRISPR-DNT | Multiple | 0.99+ | 0.7898 | 2024 |
| DNABERT-Epi | GUIDE-seq | 0.99+ | 0.550 | 2025 |

---

## CHROMAGUIDE TARGETS (MUST EXCEED SOTA)

### On-Target Efficacy:
- **High-Throughput (DeepHF)**: Spearman **> 0.90** (vs SOTA 0.880)
- **Endogenous/Functional**: Spearman **> 0.60** (vs SOTA 0.556)
- **AUROC**: **> 0.98**
- **Cohen's D**: **> 1.5** (large effect size)

### Off-Target Specificity:
- **Classification AUROC**: **> 0.99** (vs SOTA 0.985)
- **PR-AUC**: **> 0.90** (vs SOTA 0.867)
- **Specificity Score**: Spearman **> 0.88** (vs SOTA 0.853)
- **F1 Score**: **> 0.80** (balanced precision/recall)

### Conformal Prediction:
- **Coverage**: 90% at 90% confidence level
- **Interval Width**: < 0.15 (tight predictions)
- **Calibration Error**: < 0.05

---

## ARCHITECTURAL ADVANTAGES

### Why ChromaGuide Will Surpass SOTA:

1. **Mamba State Space Model**
   - O(n) complexity vs O(n²) for transformers
   - Better long-range dependencies
   - 3-5x faster training

2. **DNABERT-2 Foundation**
   - Pre-trained on 9 genomes
   - 117M parameters of genomic knowledge
   - Transfer learning advantage

3. **Multi-Modal Epigenomic Integration**
   - ENCODE tracks (H3K4me3, H3K27ac, ATAC-seq)
   - Chromatin accessibility
   - Nucleosome positioning

4. **Beta Regression Head**
   - Proper [0,1] bounded output distribution
   - Better uncertainty quantification
   - Natural for proportion data

5. **MINE Fusion**
   - Maximum Information Coefficient
   - Captures non-linear interactions
   - Better than simple concatenation

6. **Conformal Prediction**
   - Distribution-free uncertainty
   - Calibrated prediction intervals
   - Clinical/therapeutic safety

---

## TRAINING STRATEGY TO SURPASS SOTA

### Phase 1: Foundation (Week 1-2)
- Pre-train on DeepHF (60k+ sgRNAs)
- Target: Spearman > 0.88 (match best SOTA)
- A100 GPU on Narval cluster

### Phase 2: Transfer Learning (Week 3)
- Fine-tune on CRISPRon (23,902 gRNAs)
- Fine-tune on endogenous datasets
- Target: Spearman > 0.60 on functional data

### Phase 3: Multi-Task Learning (Week 4)
- Simultaneous on-target + off-target
- GUIDE-seq integration
- Target: AUROC > 0.99, PR-AUC > 0.90

### Phase 4: Ensemble & Refinement (Week 5)
- 10x ensemble with random initialization
- Hyperparameter optimization
- Ablation studies

---

## VALIDATION PROTOCOL

### Benchmark Datasets:
1. **DeepHF** - Wang et al. 2019 (WT-SpCas9, eSpCas9, HF1)
2. **CRISPRon** - Hoijer et al. 2021 (23,902 gRNAs)
3. **GUIDE-seq** - Tsai et al. 2015 (off-target validation)
4. **CRISPOR** - Haeussler et al. 2018 (12 independent sets)

### Metrics to Report:
- **Spearman** correlation (primary for on-target)
- **Pearson** correlation
- **AUROC** / **PR-AUC** (for classification)
- **Cohen's D** (effect size)
- **MCC** (Matthews Correlation)
- **Calibration** plots for conformal prediction

### Comparison Models:
- DeepCRISTL (current best)
- DeepMEns (ensemble)
- CRISPR-MCA (off-target best)
- DNABERT-Epi (foundation model)
- CFD Score (traditional)

---

## SUCCESS CRITERIA

### Minimum Acceptable Performance:
- Match SOTA on all benchmarks (Spearman ≥ 0.880, AUROC ≥ 0.985)

### Target Performance (Publication Quality):
- **Surpass SOTA by ≥ 2%** on primary metrics
- **On-target**: Spearman ≥ 0.90 on DeepHF
- **Off-target**: AUROC ≥ 0.99, PR-AUC ≥ 0.90
- **Win on ≥ 80% of benchmark datasets**

### Exceptional Performance (Nature/Science tier):
- **Surpass SOTA by ≥ 5%**
- **On-target**: Spearman ≥ 0.92
- **Off-target**: Perfect classification AUROC = 1.0 on some datasets
- **Clinical validation**: Prospective experimental validation

---

## CURRENT PROGRESS

### Local Testing (Mac Studio M3 Ultra, MPS):
- ✅ Simple CNN: Spearman 0.026 (synthetic, 5 epochs)
- ✅ DNABERT+Mamba: Spearman **0.628** (synthetic, 10 epochs, 2M params)
- ⏱️ Training time: ~10 seconds for 10 epochs

### Cluster Deployment (Narval A100):
- ✅ Persistent SSH established (72h)
- ✅ Source code transferred
- ✅ SLURM jobs submitted
- ⏳ Job 56643588 pending (waiting for GPU)

### Next Steps:
1. Download real benchmark datasets (DeepHF, CRISPRon, GUIDE-seq)
2. Train on DeepHF with full ChromaGuide architecture
3. Benchmark against SOTA models
4. Iterate until surpassing all targets
