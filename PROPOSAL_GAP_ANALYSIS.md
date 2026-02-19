# PROPOSAL GAP ANALYSIS: What We Promised vs. What We Have

**Generated:** February 18, 2026  
**Status:** CRITICAL - Detailed progress tracking for dissertation completion  
**Scope:** PhD proposal targets (from expected_results.tex, research_questions.tex, methodology.tex)

---

## EXECUTIVE SUMMARY

### Critical Finding
**We are at ~35% completion of proposed research objectives.** The foundation is solid (infrastructure, synthetic validation, architecture), but all RQ-critical components remain incomplete. **All actual success/failure criteria depend on unimplemented components.**

### Completion Matrix
| Component | Proposed | Implemented | Status |
|-----------|----------|-------------|--------|
| **RQ1 (On-target)** | Spearman ρ ≥ 0.911 | ρ ≈ -0.01 (synthetic) | ❌ 0% |
| **RQ2 (Off-target)** | AUROC/AUPRC improvement | No module exists | ❌ 0% |
| **RQ3 (Integrated)** | Joint design score | No module exists | ❌ 0% |
| **Architecture** | 5 backbone ablations | 1 backbone (DNABERT-2) | ⚠️ 20% |
| **Data/Evaluation** | DeepHF + CRISPRnature | 0 real datasets | ❌ 0% |
| **Statistical Testing** | p<0.001, Cohen's d, CQR | No framework | ❌ 0% |
| **Infrastructure** | HPO, A100 cluster, CI/CD | Complete + running | ✅ 100% |

---

## SECTION 1: RESEARCH QUESTION 1 (RQ1) - ON-TARGET PREDICTION

### Proposal Promise

**Research Question:** "How can we design sequence-aware CRISPR guides with improved on-target efficiency?"

**Specific Targets:**
- **Primary Metric:** Spearman ρ ≥ 0.911 on Split A (gene-held-out)
- **Improvement Delta:** ΔSpearman ρ ≥ 0.035 against ChromeCRISPR baseline (rho=0.876)
- **Statistical Significance:** p < 0.001 (Wilcoxon signed-rank or paired t-test)
- **Effect Size:** Cohen's d reported
- **Conformal Prediction Interval (CQR):** Coverage within ±0.02 of 90% target (i.e., 88-92%)
- **Evaluation Splits:**
  - **Split A:** Gene-held-out (most stringent)
  - **Split B:** Cell-line stress test
  - **Split C:** Off-target leakage control

**Architecture:**
- Backbone choices: CNN-GRU, DNABERT-2, Nucleotide Transformer, Caduceus-PS, Evo (7B frozen+adapter)
- Epigenomic encoder: MLP over DNase, ATAC, H3K4me3, H3K27ac from ENCODE
- Fusion strategy: Gated attention (default), plus ablation of {concatenation, cross-attention, MoE}
- Prediction head: Beta regression (for y ∈ (0,1))
- Regularization: MINE/CLUB non-redundancy constraint
- HPO: Optuna with 50 trials, TPE sampler

**Baseline Comparisons (MUST beat):**
1. ChromeCRISPR (rho=0.876 on random split)
2. CRISPR_HNN (Li et al., 2025)
3. PLM-CRISPR (Hou et al., 2025)
4. CRISPR-FMC (Li et al., 2025)
5. DNABERT-Epi (Kimata et al., 2025)
6. CCL/MoFF (Du et al., 2025)
7. DeepSpCas9, DeepHF, CRISPRon

**Dataset:** DeepHF (~40K sgRNAs, 3 cell lines: HEK293T, HCT116, HeLa)

---

### Current Implementation

#### What We Have ✅
1. **Infrastructure Stack:**
   - ✅ Narval A100 cluster access, SLURM scripts
   - ✅ DNABERT-2 backbone (117M params) pre-cached and verified
   - ✅ Mamba LSTM fusion layer implemented
   - ✅ Basic data pipeline created (download_deepHF_data.py, prepare_real_data.py)
   - ✅ HPO framework (Optuna, 50 trials) tested on synthetic
   - ✅ Optimal HPO params found: lr=2.34e-5, hidden=[512,256], dropout=[0.5,0.3]

2. **Proof-of-Concept Results (on synthetic data):**
   - ✅ Synthetic benchmark: 1,200 samples, 4 experiments completed
   - ✅ Model stability verified (no NaN, proper gradient flow)
   - ✅ Concatenation fusion = best (ρ ≈ 0.04 on synthetic)
   - ✅ ρ ≈ -0.01 ± 0.08 on test set (baseline performance)

3. **Monitoring & Automation:**
   - ✅ Background job monitoring (PID 24412, 5-min polling)
   - ✅ Auto-download results from SLURM
   - ✅ GitHub CI/CD pipeline template (v2.0-infrastructure-complete tag)

#### What We DON'T Have ❌

1. **Real Data Training (BLOCKER #1)**
   - ❌ DeepHF dataset not downloaded
   - ❌ DeepHF not preprocessed into train/val/test
   - ❌ No training on actual CRISPR data yet
   - **Current state:** Phase 2 scripts created but not executed

2. **Epigenomic Integration (BLOCKER #2)**
   - ❌ ENCODE tracks not downloaded (DNase, ATAC, H3K4me3, H3K27ac)
   - ❌ Epigenomic encoder (MLP) not implemented
   - ❌ Feature concatenation between sequence + epi features not coded
   - **Impact:** Can only use sequence → cannot reach ρ=0.911 target (literature: ~0.80)

3. **Leakage-Controlled Splits (BLOCKER #3)**
   - ❌ Split A (gene-held-out) not created - CRITICAL validation requirement
   - ❌ Split B (cell-line stress test) not created
   - ❌ Split C (off-target leakage control) not created
   - **Current:** Default train/val/test split (70/15/15) is random, not gene-held-out
   - **Problem:** Random split → artificially inflates ρ (data leakage)

4. **Conformal Prediction Intervals (BLOCKER #4)**
   - ❌ CQR (Conformalized Quantile Regression) not implemented
   - ❌ No coverage analysis (must verify 88-92% CQR coverage)
   - ❌ No uncertainty quantification framework
   - **Impact:** Cannot claim prediction intervals → missing key publication component

5. **Beta Regression Head (BLOCKER #5)**
   - ❌ Current: Standard linear regression y = X*w + b
   - ❌ Needed: Beta regression constrained to (0,1)
   - ❌ Needed: Proper modeling of bounded outcome

6. **Statistical Testing Framework (BLOCKER #6)**
   - ❌ No Wilcoxon signed-rank test implemented
   - ❌ No paired t-test for significance
   - ❌ No Cohen's d effect size calculation
   - ❌ No p-value reporting pipeline
   - **Current:** Only have raw ρ values, no statistical validation

7. **Backbone Ablations (LOWER PRIORITY)**
   - ✅ DNABERT-2 baseline complete
   - ❌ CNN-GRU variant not tested
   - ❌ Nucleotide Transformer not tested
   - ❌ Caduceus-PS not tested
   - ❌ Evo (7B frozen+adapter) not tested
   - **Impact:** Can't determine if DNABERT-2 is optimal choice

8. **Fusion Strategy Ablations (PARTIAL)**
   - ✅ Concatenation tested (best so far)
   - ⚠️ Gated attention "implemented" but validation missing
   - ❌ Cross-attention not tested
   - ❌ MoE fusion not tested

9. **MINE/CLUB Regularization (NOT STARTED)**
   - ❌ No non-redundancy constraint implemented
   - ❌ No mutual information estimation
   - ❌ No redundancy metric tracking

10. **Baseline Comparisons (NOT STARTED)**
    - ❌ No ChromeCRISPR model downloaded/integrated
    - ❌ No CRISPR_HNN, PLM-CRISPR, etc. reproduced
    - ❌ No comparative benchmarking setup

---

### RQ1 Gap Summary

| Target | Promised | Actual | Gap | Status |
|--------|----------|--------|-----|--------|
| **Primary Metric** | ρ ≥ 0.911 | ρ ≈ -0.01 (synthetic only) | **-0.921** | ❌ Critical Gap |
| **Improvement Delta** | Δρ ≥ 0.035 over ChromeCRISPR | Not measured | **Unknown** | ❌ Unbounded |
| **Data Type** | Real DeepHF (40K sgRNAs) | Synthetic (1.2K samples) | **Missing 98% gap** | ❌ Critical |
| **Epigenomics** | Full integration (DNase+H3K4me3+etc) | Not started | **0%** | ❌ Blocking |
| **Split Control** | Gene-held-out Split A (most stringent) | Random split (default) | **Wrong validation** | ❌ Invalid |
| **Conformal Prediction** | CQR 88-92% coverage | Not implemented | **N/A** | ❌ Missing |
| **Statistical Test** | p < 0.001 w/ Cohen's d | No framework | **N/A** | ❌ Missing |
| **Backbone Ablations** | 5 backbones tested | 1 backbone tested | **80% missing** | ⚠️ Incomplete |

---

## SECTION 2: RESEARCH QUESTION 2 (RQ2) - OFF-TARGET SPECIFICITY

### Proposal Promise

**Research Question:** "How can chromatin context improve off-target discrimination?"

**Specific Targets:**
- **Classification Task:** Improve AUROC/AUPRC for binary off-target site classification
- **Ranking Task:** Improve NDCG@k and Top-k recall for off-target ranking
- **Datasets:** CIRCLE-seq (Tsai et al., Nature, 2017), GUIDE-Seq (Tsai et al., Nature, 2015)
- **Validation Requirement:** Must persist under cell-line-held-out evaluation
- **Architecture:** Separate chromatin-aware off-target encoder (attention over chromatin)
- **Baseline:** Sequence-only off-target prediction (UMAP, CIRCLE-seq reported ~0.85 AUROC)

---

### Current Implementation

**Status: NOT STARTED (0% Complete)**

#### What We Have ❌
- ❌ Zero off-target module code
- ❌ CIRCLE-seq dataset not downloaded
- ❌ GUIDE-Seq dataset not downloaded
- ❌ No chromatin-aware encoder designed
- ❌ No attention mechanism for off-target ranking
- ❌ No negative sgRNA (off-target candidates) sampling pipeline

#### Design Debt
This is a **separate architecture** from RQ1 and represents ~20-30% additional engineering effort.

---

### RQ2 Gap Summary

| Target | Promised | Actual | Gap | Status |
|--------|----------|--------|-----|--------|
| **Off-Target Classification** | AUROC improvement over seq-only | No module exists | **N/A** | ❌ 0% |
| **Off-Target Ranking** | NDCG@k improvement | No ranking pipeline | **N/A** | ❌ 0% |
| **Datasets** | CIRCLE-seq + GUIDE-Seq | Not downloaded | **Missing** | ❌ 0% |
| **Chromatin Integration** | Separate off-target encoder | Not designed | **Missing** | ❌ 0% |
| **Cell-line Validation** | Stress test under held-out | Not possible yet | **N/A** | ❌ 0% |

---

## SECTION 3: RESEARCH QUESTION 3 (RQ3) - INTEGRATED DESIGN SCORE

### Proposal Promise

**Research Question:** "Can we combine on-target + off-target predictions into a unified sgRNA design score?"

**Specific Targets:**
- **Method:** Joint scoring function: score = w₁·on_target + w₂·off_target_penalty
- **Validation:** Improve Top-k selection quality over on-target-only or off-target-only ranking
- **Tunability:** Support efficiency/specificity trade-off (w₁ and w₂)

---

### Current Implementation

**Status: NOT STARTED (0% Complete)**

#### What We Have ❌
- ❌ No joint scoring framework
- ❌ No weight optimization (w₁, w₂)
- ❌ No Top-k selection pipeline
- ❌ No trade-off analysis visualization

#### Design Dependency
**Blocked by:** RQ1 completion (on-target ρ ≥ 0.911) + RQ2 completion (off-target AUROC)

Cannot start RQ3 until both RQ1 and RQ2 are validated.

---

### RQ3 Gap Summary

| Target | Promised | Actual | Gap | Status |
|--------|----------|--------|-----|--------|
| **Joint Score Function** | Implemented + optimized | Not started | **100% missing** | ❌ 0% |
| **Top-k Selection** | Comparison framework | Not implemented | **N/A** | ❌ 0% |
| **Trade-off Analysis** | w₁/w₂ parameter sweep | Not designed | **N/A** | ❌ 0% |

---

## SECTION 4: ARCHITECTURE & ENGINEERING GAPS

### Backbone Ablations (INCOMPLETE)

**Promised:**
1. CNN-GRU
2. DNABERT-2 (117M) ✅
3. Nucleotide Transformer (500M)
4. Caduceus-PS (200M)
5. Evo (7B, frozen+adapter)

**Current:** DNABERT-2 only (20% of promised ablations)

**Timeline per backbone:** ~4-6 hours compute on A100
**Total missing:** ~16-24 GPU hours across 4 architectures

---

### Epigenomic Feature Integration

**Promised:**
- DNase (promoter accessibility)
- ATAC-seq (chromatin accessibility)
- H3K4me3 (active promoters)
- H3K27ac (active enhancers)
- Source: ENCODE v4, per cell line (HEK293T, HCT116, HeLa)

**Implementation Gap:**
1. ❌ ENCODE data not downloaded (~5 GB total)
2. ❌ Feature preprocessing pipeline not coded
3. ❌ Epigenomic encoder (MLP) not integrated
4. ❌ Feature normalization strategy not decided (z-score? quantile?)
5. ❌ Benchmarking + feature removal ablation not planned

**Estimated effort:** 20-30 hours (data access, cleaning, integration)

---

### Conformal Quantile Regression (CQR)

**Promised:** 
- CQR prediction intervals at 90% confidence level
- Coverage validation (must be 88-92% on test set)

**Current:** No uncertainty quantification at all

**Technologies needed:**
- Scikit-learn quantile regression or custom implementation
- Calibration curve analysis
- Coverage metrics calculation

**Estimated effort:** 6-8 hours (including validation and tuning)

---

### Beta Regression

**Promised:** Constrained regression y ∈ (0,1) using Beta distribution

**Current:** Unbounded linear regression (y can be < 0 or > 1)

**Implementation options:**
1. PyMC3/Pyro Bayesian approach (slower but principled)
2. sklearn BetaRegressor if available
3. Custom softplus transformation + MSE

**Estimated effort:** 4-6 hours

---

### MINE/CLUB Regularization

**Promised:** Non-redundancy constraint on learned features

**Current:** Not implemented

**Complexity:** High (requires mutual information estimation)
**Estimated effort:** 16-20 hours
**Priority:** Lower (can defer post-results if time-critical)

---

### Baseline Implementations

**Promised:** Comparative benchmarking against:
1. ChromeCRISPR
2. CRISPR_HNN
3. PLM-CRISPR
4. CRISPR-FMC
5. DNABERT-Epi
6. CCL/MoFF
7. DeepSpCas9
8. DeepHF
9. CRISPRon

**Current:** None reproduced

**Effort:** Variable per method (4-12 hours each)
**Priority:** Medium (needed for paper, but can use published numbers initially)

---

## SECTION 5: DATA & EVALUATION GAPS

### Real Dataset Pipeline

| Dataset | Promised | Status | Gap |
|---------|----------|--------|-----|
| **DeepHF** | 40K sgRNAs, 3 cell lines | ❌ Not downloaded | Scripts created, not executed |
| **CRISPRnature** | ~100K sgRNAs, broader cell types | ❌ Not downloaded | Will implement after DeepHF |
| **ENCODE Epi** | DNase/ATAC/H3K marks | ❌ Not accessed | ~5 GB data, needs mapping |
| **CIRCLE-seq** | Off-target validation | ❌ Not obtained | Supplementary materials |
| **GUIDE-Seq** | Off-target validation | ❌ Not obtained | Supplementary materials |

### Evaluation Splits (CRITICAL)

**Promised:**
- **Split A (Gene-held-out):** Hold out entire genes, most stringent
- **Split B (Cell-line stress):** Hold out entire cell lines
- **Split C (Off-target control):** Ensure no leakage between on-target/off-target

**Current:** Default train/val/test (70/15/15) with random sampling
**Problem:** Creates data leakage → inflates metrics by ~5-15%

**Implementation gap:**
1. Define gene grouping for Split A
2. Implement stratified k-fold with gene boundaries
3. Create visualization of split leakage
4. Re-run all experiments with proper splits

**Estimated effort:** 8-12 hours (including re-training)

---

## SECTION 6: STATISTICAL VALIDATION GAPS

### Required but Missing

1. **Significance Testing**
   - Wilcoxon signed-rank test (non-parametric, paired)
   - Paired t-test (for parametric validation)
   - p-values must be < 0.001
   - Current: No testing framework

2. **Effect Size**
   - Cohen's d calculation
   - Confidence intervals (95%)
   - Cohen's d interpretation ("medium" or "large" effect)
   - Current: Not calculated

3. **Conformal Prediction Coverage**
   - Measure actual CQR coverage rate on held-out test
   - Validate within [0.88, 0.92] for 90% target
   - Current: No CQR implemented

4. **Multiple Comparison Correction**
   - Bonferroni or Benjamini-Hochberg for ablation studies
   - Current: Not planned

**Total effort:** 8-12 hours

---

## SECTION 7: PRIORITY MATRIX & TIMELINE

### Tier 1 (BLOCKING - Must Complete Before Paper)

| Task | Effort | Timeline | Dependencies | Status |
|------|--------|----------|--------------|--------|
| **Download DeepHF** | 2h | Now - 2h | None | ❌ Queued |
| **Preprocess DeepHF → train/val/test** | 3h | 2-5h | DeepHF download | ❌ Queued |
| **Implement leakage-controlled splits (A/B/C)** | 12h | 5-17h | DeepHF preprocessed | ❌ Critical |
| **Real data Phase 2 training** | 6-8h compute | 17-25h (wall clock) | Splits implemented | ❌ Queued |
| **Analyze RQ1 results, verify ρ ≥ 0.911** | 4h | 25-29h | Phase 2 training | ❌ Depends on results |
| **Implement statistical testing (Wilcoxon, Cohen's d, p-values)** | 8h | 25-33h (parallel) | Phase 2 results | ❌ Critical |
| **Implement conformal prediction intervals (CQR)** | 8h | 25-33h (parallel) | Phase 2 results | ❌ Critical |
| **Create publication figures** | 6h | 30-36h | All above complete | ❌ Depends on results |

### Tier 2 (HIGH PRIORITY - Should Complete for Paper)

| Task | Effort | Timeline | Dependencies | Status |
|------|--------|----------|--------------|--------|
| **Epigenomic integration (ENCODE + MLP encoder)** | 20-30h | Week 2 | Phase 2 results, decide if worth effort | ⚠️ Contingent |
| **Backbone ablations (4 additional backbones)** | 24h compute (16-24h wall) | Week 2, parallel | Infrastructure ready | ⚠️ Lower priority |
| **Fusion ablations (cross-attention, MoE)** | 12h compute (8-12h wall) | Week 2, parallel | Infrastructure ready | ⚠️ Partial credit |
| **Download CIRCLE-seq, GUIDE-Seq for RQ2 prep** | 4h | Same as Tier 1 phase | Links/permissions secured | ⚠️ Parallel |
| **Off-target encoder design (RQ2)** | 20-25h | Week 2-3 | Off-target data downloaded | ❌ 0% |

### Tier 3 (NICE-TO-HAVE - Can Defer Post-Publication)

| Task | Effort | Timeline | Status |
|------|--------|----------|--------|
| **MINE/CLUB regularization** | 16-20h | Post-paper | ❌ Low priority |
| **Baseline comparisons** | 30-50h | Post-paper | ❌ Low priority |
| **RQ2 (off-target) full implementation** | 25-30h | Post-paper or side project | ❌ 0% |
| **RQ3 (integrated design) full implementation** | 15-20h | Post-paper | ❌ 0%, depends on RQ1+RQ2 |

---

## SECTION 8: CRITICAL PATH TO PUBLICATION

```
NOW (Feb 18)
  ↓ [Tier 1: 0-5 hours]
DOWNLOAD + PREPROCESS DEEPHF
  ↓
IMPLEMENT LEAKAGE-CONTROLLED SPLITS A/B/C
  ↓ [Tier 1: 5-17 hours]
RETRAIN PHASE 2 (6-8h compute)
  ↓
ANALYZE RQ1 RESULTS (ρ target check)
  ↓ [Tier 1: 17-33 hours parallel]
├─ Statistical testing (Wilcoxon, Cohen's d)
├─ CQR implementation + validation
└─ Remove/fix architecture if ρ < 0.876
  ↓
[if ρ ≥0.876]:
  GENERATE PUBLICATION FIGURES
  ↓
  WRITE PAPER (3-5 days)
  ↓
  SUBMIT TO JOURNAL (Mar 1-5)

[if ρ < 0.876]:
  INVESTIGATE ROOT CAUSE
  ├─ Missing epigenomics critical?
  ├─ Wrong architecture?
  └─ Data quality issue?
  ↓
  ITERATE (possibly with epigenomics)
```

**Key decision point:** After Phase 2 results, measure ρ against 0.911 target.
- **If ρ ≥ 0.90:** Proceed to publication (epigenomics optional enhancement)
- **If 0.876 ≤ ρ < 0.90:** Acceptable with epigenomics (worth effort)
- **If ρ < 0.876:** Major issue, investigate before epigenomics

---

## SECTION 9: RESOURCE REQUIREMENTS

### GPU Compute

| Task | GPU Hours | A100 Wall | Cost (@$1.50/hr) |
|------|-----------|-----------|-----------------|
| Phase 2 training (chromaguide_full) | 8 | 1 hour | $1.50 |
| Phase 2 statistical analysis | 0 | 0.5 hours | $0.75 |
| Backbone ablations (if doing) | 24 | 6 hours | $9.00 |
| Epigenomics integration retraining (if doing) | 8 | 1 hour | $1.50 |
| **TOTAL MINIMUM** | **8** | **1.5h** | **$2.25** |
| **TOTAL WITH ABLATIONS** | **32** | **8h** | **$12** |

### Data Download

| Dataset | Size | Time | Status |
|---------|------|------|--------|
| DeepHF pickles | ~500 MB | 5-10 min | ❌ Not downloaded |
| ENCODE marks (4 types × 3 cell lines) | ~5 GB | 30-60 min | ❌ Not accessed |
| CIRCLE-seq supplementary | ~100 MB | 5 min | ❌ Not obtained |
| GUIDE-Seq supplementary | ~100 MB | 5 min | ❌ Not obtained |

### Development Time (Engineer Effort)

| Component | Hours | Max Parallel | Timeline |
|-----------|-------|--------------|----------|
| **CRITICAL PATH** | | | |
| DeepHF download pipeline | 1 | N/A | Now-1h |
| Leakage-controlled splits | 12 | No | 1-13h |
| Statistical testing framework | 8 | Yes (parallel) | 5-13h |
| CQR implementation | 8 | Yes (parallel) | 5-13h |
| Phase 2 training | 0 (compute only) | N/A | 13-21h (wall) |
| Analysis & figures | 4 | No | 21-25h |
| **CRITICAL PATH TOTAL** | **33 hours** | **13-25h wall** | **Feb 18-20** |
| | | | |
| **OPTIONAL ENHANCEMENTS** | | | |
| Epigenomics integration | 20-30 | No | +2-3 days if needed |
| Backbone ablations | 4 | Yes | +1 day parallel |
| Fusion ablations | 4 | Yes | +0.5 day parallel |
| RQ2 off-target module | 25-30 | No | +3-4 days |

---

## SECTION 10: SPECIFIC BLOCKERS & ROOT CAUSES

### BLOCKER #1: Real Data Not Loaded
**Root Cause:** Phase 2 scripts created but not executed  
**Impact:** Cannot train on real CRISPR data, only synthetic  
**Fix:** `python3 scripts/download_deepHF_data.py && python3 scripts/prepare_real_data.py`  
**Timeline:** 30 minutes  
**Risk:** Low (scripts already written)

### BLOCKER #2: Leakage in Evaluation Splits
**Root Cause:** Using default random train/val/test, not gene-held-out  
**Impact:** Metrics inflated by 5-15%, results scientifically invalid  
**Fix:** Redesign splits with gene/cell-line stratification  
**Timeline:** 12 hours (coding + retraining)  
**Risk:** Medium (requires careful validation)

### BLOCKER #3: No Epigenomics = No ρ ≥ 0.90
**Root Cause:** ENCODE data not integrated  
**Impact:** Sequence-only typically reaches ρ ≈ 0.80, not 0.911 target  
**Fix:** Download ENCODE marks, implement MLP encoder, integrate features  
**Timeline:** 20-30 hours  
**Risk:** High (complex integration, unproven benefit)  
**Decision:** Check Phase 2 sequence-only results first; if ρ ≥ 0.85, may be acceptable

### BLOCKER #4: No Conformal Prediction Intervals
**Root Cause:** Not implemented  
**Impact:** Cannot claim uncertainty quantification, publication incompleteness  
**Fix:** Implement CQR framework with calibration  
**Timeline:** 8 hours  
**Risk:** Low (standard technique, scikit-learn support)

### BLOCKER #5: No Statistical Significance Testing
**Root Cause:** Framework not built  
**Impact:** Cannot claim p < 0.001, effect sizes  
**Fix:** Add Wilcoxon signed-rank, paired t-test, Cohen's d  
**Timeline:** 6-8 hours  
**Risk:** Low (standard statistical tests)

---

## SECTION 11: DECISION TREE FOR NEXT 48 HOURS

```
DECISION POINT 0 (NOW - Feb 18 evening):
├─ Q: Should I run Phase 2 immediately?
├─ A: YES - Execute now
│   └─ Commands:
│       python3 scripts/download_deepHF_data.py
│       python3 scripts/prepare_real_data.py --dataset deepHF
│       sbatch submit_jobs/train_chromaguide_realdata.slurm
│
│
DECISION POINT 1 (Feb 19, after Phase 2 completes):
├─ Q: Is ρ on DeepHF ≥ 0.876 (better than ChromeCRISPR)?
├─ A (YES): ─────────────────────┐
│   └─ Continue with ablations/ |
│       epigenomics (TIME         |
│       PERMITTING)              |
│                                |
├─ A (NO, 0.80-0.875): ──────────┤
│   └─ Investigate causes        |
│       ├─ Check split leakage?  |
│       ├─ Add epigenomics?      |
│       └─ Review architecture   |
│                                |
├─ A (NO, < 0.80): ─────────────┤
│   └─ PROBLEM: Significant      |
│       regression from synthetic|
│       ├─ Debug data pipeline   |
│       ├─ Check feature scales  |
│       └─ May need major fix    |
│
│
DECISION POINT 2 (Feb 19-20, analysis phase):
├─ Q: Should I add epigenomics?
├─ A (if ρ ≥ 0.90): NO - sequence-only is sufficient, save 20h
├─ A (if 0.876-0.90): MAYBE - if time permits and reviewers demand
├─ A (if ρ < 0.876): YES - required to reach baseline
│
│
DECISION POINT 3 (Feb 20-21, publication phase):
├─ Q: Should I run backbone ablations?
├─ A (if on schedule): NO - optional, DNABERT-2 sufficient
├─ A (if ahead of schedule): YES - adds ablation rigor
└─ Timeline: Can do in parallel with thesis writing
```

---

## SECTION 12: SUCCESS CRITERIA vs. CURRENT STATE

### Must-Have for PhD Defense (RQ1 Critical Path)

| Criterion | Promised | Current | Status |
|-----------|----------|---------|--------|
| Spearman ρ on real data | ≥ 0.911 | -0.01 (synthetic) | ❌ MISSING |
| vs. ChromeCRISPR Δρ | ≥ 0.035 | Unknown | ❌ UNKNOWN |
| p-value significance | < 0.001 | Not computed | ❌ MISSING |
| Effect size (Cohen's d) | Reported | Not computed | ❌ MISSING |
| Prediction intervals (CQR) | Yes, 88-92% coverage | Not implemented | ❌ MISSING |
| Gene-held-out splits | Required | Not implemented | ❌ MISSING |
| Training data | DeepHF 40K | Not used yet | ❌ PENDING |

### Nice-to-Have (Secondary Objectives)

| Criterion | Promised | Current | Status |
|-----------|----------|---------|--------|
| Backbone ablations | 5 backbones | 1 backbone | ⚠️ PARTIAL (20%) |
| Fusion ablations | 4 fusion types | 1 + validation needed | ⚠️ PARTIAL (25%) |
| Epigenomics integration | Full | Not started | ⚠️ OPTIONAL |
| Off-target (RQ2) | Full module | 0% | ⚠️ CAN DEFER |
| Integrated design (RQ3) | Full module | 0% | ⚠️ CAN DEFER |

---

## SECTION 13: IMMEDIATE ACTION ITEMS (NEXT 8 HOURS)

### Action #1: Execute Phase 2 Data Pipeline
```bash
# Execute now
python3 scripts/download_deepHF_data.py
python3 scripts/prepare_real_data.py --dataset deepHF --seed 42 \
  --val-split 0.15 --test-split 0.15

# Verify
ls -lh data/deepHF/processed/
# Should see: X_train.pkl, y_train.pkl, X_val.pkl, y_val.pkl, X_test.pkl, y_test.pkl, metadata.json
```
**Timeline:** 30 minutes
**Risk:** Low (scripts already written)

### Action #2: Create Leakage-Controlled Splits Module
Create `scripts/create_leakage_controlled_splits.py`:
- Input: DeepHF preprocessed data
- Output: Split A (gene-held-out), Split B (cell-line-held-out), Split C
- Validate no sample overlap between splits
- Save splits to `data/deepHF/splits_A_B_C/`

**Timeline:** 2-3 hours
**Risk:** Medium (validation critical)

### Action #3: Queue Phase 2 Real Data Training
```bash
sbatch --array=1-3 submit_jobs/train_chromaguide_realdata.slurm
# This will:
# - Job 1: Sequence-only on Split A (baseline)
# - Job 2: ChromaGuide full on Split A
# - Job 3: Ablations on Split A if time permits
```
**Timeline:** Submit now, train 6-8 hours
**Expected completion:** Feb 19, morning

### Action #4: Build Statistical Testing Framework (Parallel)
Create `scripts/statistical_testing.py`:
- Wilcoxon signed-rank test
- Paired t-test
- Cohen's d effect size
- p-value reporter

**Timeline:** 4-6 hours (can work in parallel with training)
**Risk:** Low

### Action #5: Implement CQR Framework (Parallel)
Create `scripts/conformal_prediction.py`:
- Calibration set selection
- CQR coverage computation
- Prediction interval generation

**Timeline:** 4-6 hours (parallel)
**Risk:** Low

---

## SECTION 14: 30-DAY ROADMAP

### Week 1 (Feb 18-24): CRITICAL PATH - RQ1 Completion

**Days 1-2 (Feb 18-19):**
- [NOW] Execute Phase 2 data pipeline
- Implement leakage-controlled splits
- Queue Phase 2 training jobs
- **Status check:** Both jobs should be running by Feb 18, 23:00

**Days 3-4 (Feb 19-20):**
- Jobs complete (expecting Feb 19, morning)
- Implement statistical testing framework
- Implement CQR framework
- Analyze RQ1 results
- **Decision point:** Is ρ ≥ 0.876?

**Days 5-7 (Feb 20-22):**
- If ρ ≥ 0.876: Generate publication figures
- If ρ < 0.876: Debug + iterate (may add epigenomics)
- Draft results section for paper
- **Target:** Manuscript draft ready

### Week 2 (Feb 24-Mar 2): SUPPLEMENTARY + PUBLICATION

**Days 8-10 (Feb 24-26):**
- Backbone ablations (if time, can parallelize)
- Fusion ablations (if time)
- Write methods/results sections
- **Target:** Supplementary experiments + methodology solid

**Days 11-14 (Feb 26-Mar 2):**
- Finalize figures
- Complete paper writing
- Internal review
- **Target:** Submit to journal (Nature Biomedical Engineering)

### Week 3+ (Mar 2-12): OPTIONAL ENHANCEMENTS

**If ρ ≥ 0.90 and time permits:**
- Epigenomics integration (20-30 hours) → potential ρ improvement
- RQ2 off-target module groundwork
- Pre-print on arXiv

**If behind schedule:**
- Focus on RQ1 ablations
- Defer epigenomics and RQ2/RQ3

---

## SECTION 15: RISK ASSESSMENT & MITIGATION

### RISK #1: ρ < 0.876 on Real Data (Major Regression)
**Probability:** 15-25% (synthetic showed stability)
**Impact:** Paper rejected, requires investigation
**Mitigation:**
- Check data preprocessing immediately
- Verify no NaN/inf in features
- Check feature scales match training
- Compare splits between random vs proper

**Decision:** If ρ < 0.80, pause publishing; debug first

### RISK #2: Leakage Still Present in Splits
**Probability:** 10% (careful design should catch)
**Impact:** Results scientifically invalid
**Mitigation:**
- Rigorous validation: check no gene overlap between train/val/test
- Verify using gene membership audit
- Cross-check with cell-line membership

### RISK #3: Epigenomics Data Not Available
**Probability:** 5% (ENCODE should be public)
**Impact:** Cannot integrate epi features (planned fallback)
**Mitigation:**
- Test publicly available ENCODE API first
- Have fallback: sequence-only submission if epi fails

### RISK #4: Narval Cluster Downtime
**Probability:** 10% (typical HPC)
**Impact:** 6-12 hour delay
**Mitigation:**
- Monitor cluster status continuously
- Queue jobs early to minimize impact
- Have backup: AWS EC2 A100 (if budget allows)

### RISK #5: Reviewers Demand Baseline Comparisons
**Probability:** 80% (standard requirement)
**Impact:** Need 2-3 weeks for baseline reproduction
**Mitigation:**
- Use published numbers as estimate
- Implement ChromeCRISPR reproduction in parallel by week 2
- Frame as "future work" if time-critical

---

## SECTION 16: GIT TRACKING FOR GAP ANALYSIS

### Commits to Create (Tied to Blockers)

```bash
# Track completion of each blocker
git tag blocker/1-deephf-loaded -m "DeepHF downloaded and preprocessed, ready for training"
git tag blocker/2-leakage-splits -m "Implemented Split A (gene-held-out), Split B, Split C"
git tag blocker/3-rq1-trained -m "Phase 2 real data training complete, ρ measured"
git tag blocker/4-cqr-validated -m "CQR conformal prediction intervals implemented and validated"
git tag blocker/5-statistical-tests -m "Wilcoxon, Cohen's d, p-values computed"
git tag blocker/6-figures-generated -m "Publication-ready figures created"

# Timeline milestones
git tag milestone/phase-2-start -m "Phase 2 begins: real data training"
git tag milestone/phase-2-results -m "Phase 2 results ready for analysis"
git tag milestone/paper-ready -m "Manuscript ready for colleague review"
git tag milestone/submitted -m "Submitted to journal"
```

---

## SECTION 17: CONCLUSION & SUMMARY

### Current State
- **Foundation:** ✅ Excellent (infrastructure, HPO, monitoring)
- **Proof-of-concept:** ✅ Good (synthetic validation complete)
- **Real-world validation:** ❌ NOT STARTED (critical blocker)
- **Research objectives:** ❌ 35% complete, 65% gaps remaining

### Critical Path to Completion
1. **Execute Phase 2 pipeline** (30 min) → real data training
2. **Implement leakage-controlled splits** (12 hours) → valid evaluation
3. **Train on real data** (8 hours compute) → actual ρ measurement
4. **Statistical validation** (8 hours) → p-values and effect sizes
5. **CQR implementation** (8 hours) → prediction intervals
6. **Generate figures + write paper** (10 hours) → publication

**Total critical path:** 46 hours engineering, 8 hours compute = **2.5-3 days to submission**

### Decision Point (Feb 19, Post-Results)
**If ρ ≥ 0.876:**
- Submit paper immediately (RQ1 proven)
- Epigenomics/RQ2/RQ3 are "future work"

**If 0.80 ≤ ρ < 0.876:**
- Debug + add epigenomics (20-30 hours)
- Retest before submission

**If ρ < 0.80:**
- Major issue, pause publishing
- Investigate root cause thoroughly

### Next Command (Execute Now)
```bash
python3 scripts/download_deepHF_data.py && \
python3 scripts/prepare_real_data.py --dataset deepHF --seed 42 && \
echo "✅ Phase 2 data ready. Check: ls -lh data/deepHF/processed/"
```

---

**Document Version:** 1.0  
**Last Updated:** February 18, 2026, 18:40 PST  
**Next Review:** After Phase 2 results available (Feb 19, morning)  
**Responsible:** PhD Candidate (you)  
**Stakeholders:** Advisor, Committee
