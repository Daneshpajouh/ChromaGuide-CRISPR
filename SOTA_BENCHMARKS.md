# CRISPR-Cas9 SOTA Benchmarks - Targets to Surpass

## ON-TARGET PREDICTION BENCHMARKS

### High-Throughput Datasets (DeepHF, WT-SpCas9)
| Model | Spearman | Year | Notes |
|-------|----------|------|-------|
| DeepCRISTL (TL) | 0.880 | 2024 | Multi-task + ensemble |
| CRISeq | 0.877 | 2024 | Best on WT-SpCas9 |
| C-RNNCrispr | 0.877 | 2020 | CNN+RNN hybrid |
| DeepMEns | 0.874 | 2024 | Ensemble model |
| CRISPRon | 0.871 | 2024 | Fine-tuned |
| DeepSpCas9 | 0.866 | 2019 | DeepHF baseline |
| DeepCRISPR | 0.845 | 2018 | Original deep learning |

### Endogenous/Functional Datasets
| Model | Avg Spearman | Datasets | Notes |
|-------|--------------|----------|-------|
| DeepCRISTL | 0.556 | HelaLib2 | Transfer learning |
| CRISPRon | 0.518 | Multiple | Fine-tuned |
| DeepMEns | 0.446 | 10 datasets | Ensemble |
| DeepSpCas9 | 0.441 | 10 datasets | Baseline |
| CRISPR-ONT | 0.406 | Benchmark | CNN+Attention |

### Target Metrics for ChromaGuide ON-TARGET:
- **High-throughput**: Spearman > 0.885 (surpass DeepCRISTL)
- **Endogenous**: Spearman > 0.56 (surpass DeepCRISTL)
- **AUROC**: > 0.98

---

## OFF-TARGET PREDICTION BENCHMARKS

### Classification Metrics (GUIDE-seq, CRISPOR)
| Model | AUROC | PR-AUC | F1 | MCC | Year |
|-------|-------|--------|-----|-----|------|
| CRISPR-MCA | 0.9853 | 0.8668 | - | - | 2024 |
| CRISPR-DNT | 0.99+ | 0.7898 | - | - | 2024 |
| DNABERT-Epi | 0.99+ | 0.550 | High | High | 2025 |
| CNN_std | 0.972 | - | - | - | 2018 |
| CFD Score | 0.912 | - | - | - | 2016 |

### CRISOT Specificity
| Model | Spearman (off-target fraction) | Notes |
|-------|-------------------------------|-------|
| CRISOT-Score | -0.853 | Site-seq dataset |

### Target Metrics for ChromaGuide OFF-TARGET:
- **AUROC**: > 0.99
- **PR-AUC**: > 0.87 (surpass CRISPR-MCA)
- **Spearman (specificity)**: > 0.86 absolute

---

## KEY ARCHITECTURE INSIGHTS FROM SOTA

1. **DNABERT-Epi** (2025): Pre-trained DNA foundation + epigenetic features
   - H3K4me3, H3K27ac, ATAC-seq integration
   - PR-AUC: 0.550 on GUIDE-seq

2. **DeepCRISTL** (2024): Transfer learning from high-throughput to functional
   - Multi-task learning across enzymes
   - Ensemble of 10 random initializations

3. **CRISPR-MCA** (2024): Multi-head cross-attention
   - ESB (Efficient Stratified Batch) sampling
   - Best PR-AUC: 0.8668

4. **CCLMoff** (2025): RNA language model for off-target
   - AUROC: 0.8123 on variable length sgRNAs

---

## CHROMAGUIDE TARGETS TO ACHIEVE

### Architecture Advantages:
1. **Mamba SSM** - O(n) complexity vs O(n²) transformers
2. **DNABERT-2 pre-training** - genomic knowledge
3. **Epigenomic integration** - ENCODE tracks
4. **Beta regression** - proper bounded output [0,1]
5. **Conformal prediction** - calibrated uncertainty

### Performance Targets:
| Task | Metric | SOTA | ChromaGuide Target |
|------|--------|------|--------------------|
| On-target (HT) | Spearman | 0.880 | **0.90+** |
| On-target (Endo) | Spearman | 0.556 | **0.60+** |
| Off-target | AUROC | 0.985 | **0.99+** |
| Off-target | PR-AUC | 0.867 | **0.90+** |
| Specificity | Spearman | 0.853 | **0.88+** |

### Dataset Requirements:
- DeepHF (Wang et al. 2019) - 60,000+ sgRNAs
- GUIDE-seq (Tsai et al.) - off-target validation
- CRISPOR benchmark - 12 datasets
- ENCODE epigenomic tracks
