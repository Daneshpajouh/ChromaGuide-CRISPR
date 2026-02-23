# V10 QUICK START GUIDE

## What is V10?

**V10** is a second iteration of maximum-power architectures designed to close the gap to PhD proposal targets:
- V9 results: Multimodal Rho 0.7976 (target 0.911), Off-target AUROC 0.9264 (target 0.99)
- V10 strategy: Combine verified reference architectures from published CRISPR ML papers
- Timeline: 3-4 days to final results + evaluation

---

## BEFORE YOU START

### Prerequisites
```bash
# 1. Python environment configured
conda activate cg_train  # or your environment

# 2. PyTorch with GPU support
pip install torch torchvision torchaudio

# 3. HuggingFace transformers (for DNABERT-2)
pip install transformers

# 4. Data available
ls /Users/studio/Desktop/PhD/Proposal/data/processed/split_a/     # Multimodal
ls /Users/studio/Desktop/PhD/Proposal/data/raw/crisprofft/        # Off-target

# 5. Model directory exists
mkdir -p /Users/studio/Desktop/PhD/Proposal/models
mkdir -p /Users/studio/Desktop/PhD/Proposal/logs
```

---

## DEPLOYMENT OPTIONS

### Option 1: Local Training (Mac Studio)
**Best for:** Testing, quick iteration, small models
**Time:** ~8-12 hours each task
**GPU:** MPS (Metal Performance Shaders)

```bash
cd /Users/studio/Desktop/PhD/Proposal

# Multimodal training
python3 scripts/train_on_real_data_v10.py 2>&1 | tee logs/multimodal_v10_local.log

# Off-target training (in separate terminal)
python3 scripts/train_off_target_v10.py 2>&1 | tee logs/off_target_v10_local.log

# Monitor progress
tail -f logs/multimodal_v10_local.log
tail -f logs/off_target_v10_local.log
```

### Option 2: Fir Cluster (Recommended)
**Best for:** Full training, H100 GPUs, faster convergence
**Time:** ~3-4 hours each task
**GPU:** NVIDIA H100 SXM5 80GB

```bash
cd /Users/studio/Desktop/PhD/Proposal

# Single job submission
sbatch slurm_fir_v10_multimodal.sh      # H100:1 (6 hours)
sbatch slurm_fir_v10_off_target.sh      # H100:2 (8 hours)

# Check job status
ssh fir.alliancecan.ca "squeue -u studio"

# Monitor output
ssh fir.alliancecan.ca "tail -f ~/chromaguide/logs/slurm_multimodal_v10_*.log"
```

### Option 3: Parallel Orchestration (AUTO)
**Best for:** Unattended execution on both platforms
**Time:** Simultaneous on Fir + local fallback
**Recommended:** Yes

```bash
# Auto-detects cluster access, submits to Fir, runs locally as backup
python3 deploy_v10.py

# This will:
# 1. Check Fir cluster connectivity
# 2. Submit SLURM jobs if available
# 3. Launch local training in parallel
# 4. Monitor both streams
# 5. Commit results to git when done
```

---

## MONITORING EXECUTION

### Local Process Monitoring
```bash
# Watch multimodal training live
tail -f logs/multimodal_v10_local.log

# Watch off-target training live
tail -f logs/off_target_v10_local.log

# Check GPU usage (Mac)
./scripts/monitor_gpu_mac.sh   # Custom script if available

# Check all Python training processes
ps aux | grep train_
```

### Fir Cluster Monitoring
```bash
# List your jobs
ssh fir.alliancecan.ca "squeue -u studio"

# Check specific job details
ssh fir.alliancecan.ca "sstat -j <job_id> --format=JobID,MaxVMSize,MaxRSS,AveCPU"

# View job output
ssh fir.alliancecan.ca "tail -f ~/chromaguide/logs/slurm_multimodal*.log"

# Cancel a job if needed
ssh fir.alliancecan.ca "scancel <job_id>"
```

### Expected Training Progress

**Multimodal (5 models × 150 epochs):**
- First model: E0-30 in first hour (verify training starting)
- All models: E50+ by hour 3 (should reach 50% accuracy)
- Convergence: E150 by hour 8-10

**Off-target (5 models × 100-200 epochs):**
- First model: E0-30 in first 30 minutes
- All models: E50+ by hour 2
- Convergence: E200 by hour 4-6

---

## AFTER TRAINING COMPLETES

### 1. Check Model Saves
```bash
# Verify multimodal models saved
ls -lah models/multimodal_v10_seed*.pt
# Should have: multimodal_v10_seed0.pt through multimodal_v10_seed4.pt

# Verify off-target models saved
ls -lah models/off_target_v10_seed*.pt
# Should have: off_target_v10_seed0.pt through off_target_v10_seed4.pt
```

### 2. Run Evaluation
```bash
# Evaluate both tasks against targets
python3 scripts/evaluate_v10_models.py

# This will output:
# - Individual model metrics
# - Ensemble metrics
# - Comparison to targets
# - Achievement percentages
# - Gap analysis
```

### 3. Review Results
```bash
# View evaluation results
cat logs/v10_evaluation_results.json | python3 -m json.tool

# Read summary report
cat V10_COMPREHENSIVE_RESULTS.txt

# Check training logs for any anomalies
grep -i "error\|warning\|nan" logs/multimodal_v10*.log
grep -i "error\|warning\|nan" logs/off_target_v10*.log
```

### 4. Commit to Git
```bash
cd /Users/studio/Desktop/PhD/Proposal

git add -A
git commit -m "V10 training complete: [your summary of results]"
git push origin main
```

---

## EXPECTED RESULTS

### Multimodal (On-target Efficacy)
| Metric | V9 | V10 Expected | Target | Success |
|--------|----|----|--------|---------|
| Spearman Rho | 0.7976 | 0.88-0.92 | 0.911 | ✓ Likely |
| p-value | <0.001 | <0.001 | - | ✓ |
| Achievement | 87.6% | 96-101% | 100% | ✓ Very Likely |

### Off-target (Classification)
| Metric | V9 | V10 Expected | Target | Success |
|--------|----|----|--------|---------|
| AUROC | 0.9264 | 0.94-0.97 | 0.99 | ? Uncertain |
| Sensitivity | ? | ~0.85-0.90 | - | ? |
| Specificity | ? | ~0.95-0.99 | - | ? |
| Achievement | ~93% | ~95-98% | 100% | ⚠️ May need V11 |

---

## TROUBLESHOOTING

### Issue: GPU out of memory
**Solution:**
```bash
# Reduce batch size in training script
# Change: batch_size = 50  →  batch_size = 32
# Or submit smaller job: --gpus=h100_mig:1g.10gb

# Restart with reduced memory
python3 train_on_real_data_v10.py --batch_size 32
```

### Issue: DNABERT-2 download fails
**Solution:**
```bash
# Pre-download model
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
print("✓ Models downloaded successfully")
EOF
```

### Issue: Data not found
**Solution:**
```bash
# Verify data paths
ls /Users/studio/Desktop/PhD/Proposal/data/processed/split_a/HCT116_*.csv
ls /Users/studio/Desktop/PhD/Proposal/data/raw/crisprofft/CRISPRoffT_all_targets.txt

# If missing, check data preparation scripts
ls scripts/prepare_*.py
```

### Issue: Fir cluster not accessible
**Solution:**
```bash
# Test connection
ssh -o ConnectTimeout=5 fir.alliancecan.ca "echo OK"

# If fails, run locally instead
# deploy_v10.py will auto-fallback to local training
```

---

## KEY DIFFERENCES: V9 vs V10

### Architectural Improvements
1. **DNABERT-2 vs Custom Transformer**
   - Pretrained on 46GB genomic data
   - BPE tokenization (better than k-mer)
   - 117M parameters with proven generalization

2. **Epigenetic Gating vs No Control**
   - Learnable sigmoid gate to control feature contribution
   - Model learns when to trust sequence vs. epigenomics
   - Reduces overfitting to noisy epigenetic features

3. **Multi-Scale CNN + BiLSTM (Off-target)**
   - Captures patterns at multiple scales (1x1, 3x3, 5x5)
   - BiLSTM for long-range sequence dependencies
   - Three complementary feature extractors

### Training Improvements
1. **Layer-wise learning rates**
   - DNABERT: 2e-5 (gentle fine-tuning)
   - Other modules: 5e-4 to 1e-3 (faster learning)

2. **Better sampling for imbalance**
   - WeightedRandomSampler with dynamic pos_weight
   - Prevents extreme class imbalance convergence

3. **Longer training**
   - Multimodal: 150 epochs (was 50 in early attempts)
   - Off-target: 100-200 epochs (was 200 in V9)

---

## FILE STRUCTURE

```
PhD/Proposal/
├── scripts/
│   ├── train_on_real_data_v10.py      ← V10 Multimodal
│   ├── train_off_target_v10.py        ← V10 Off-target
│   └── evaluate_v10_models.py         ← Evaluation script
├── models/
│   ├── multimodal_v10_seed0.pt        ← Trained models (auto-saved)
│   ├── multimodal_v10_seed1.pt
│   ├── off_target_v10_seed0.pt
│   └── ...
├── logs/
│   ├── multimodal_v10_local.log       ← Training logs
│   ├── off_target_v10_local.log
│   └── v10_evaluation_results.json    ← Results JSON
├── slurm_fir_v10_multimodal.sh        ← SLURM script
├── slurm_fir_v10_off_target.sh        ← SLURM script
├── deploy_v10.py                      ← Orchestration script
├── V10_ARCHITECTURE_GUIDE.md          ← Technical deep-dive
└── V10_QUICK_START_GUIDE.md           ← This file
```

---

## NEXT STEPS AFTER V10

**If V10 achieves targets (Rho ≥ 0.91, AUROC ≥ 0.98):**
1. ✅ Write results section for PhD thesis
2. ✅ Prepare manuscript for publication
3. ✅ Commit final code to GitHub

**If V10 close but not there (90-95% of targets):**
1. ⏳ Plan V11 with ensemble scaling + hyperparameter tuning
2. ⏳ Consider hybrid architectures (RoBERTa + DNABERT-2)
3. ⏳ Implement hard example mining

**If V10 still far short (<90%):**
1. 🔧 Debug training issues
2. 🔧 Check data quality
3. 🔧 Revisit architecture design

---

## CONTACT & SUPPORT

- **Cluster Issues:** Alliance Canada support (https://alliancecan.ca/support)
- **DNABERT-2:** GitHub issues at MAGICS-LAB/DNABERT_2
- **Local Issues:** Check GPU drivers, PyTorch installation

---

**Status:** Ready for deployment
**Created:** February 23, 2026
**Target Completion:** February 26, 2026
