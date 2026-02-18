# Pre-Execution Checklist for ChromaGuide PhD Experiments

## ✅ Pre-Flight Checks

### 1. System Requirements
- [ ] Access to Narval cluster (ssh narval works)
- [ ] SLURM permission (sbatch command available)
- [ ] A100 GPUs available (check `sinfo` on Narval)
- [ ] ~100GB storage for data + models on Narval
- [ ] Python 3.11+ installed on Narval

### 2. Verify Narval Connection
```bash
ssh narval "echo Connected to Narval!"
ssh narval "nvidia-smi"  # Check GPU availability
```
- [ ] SSH connection working
- [ ] GPU availability confirmed

### 3. Check Data Directory Structure
```bash
ssh narval "mkdir -p /project/def-bengioy/chromaguide_data"
ssh narval "mkdir -p /project/def-bengioy/chromaguide_results"
```
- [ ] Base directories created
- [ ] Write permissions verified

### 4. Verify Python Dependencies
Required packages on Narval:
- torch (with CUDA 11.8)
- transformers (for DNABERT)
- pandas (for data handling)
- scipy (for statistics)
- optuna (for HPO)
- matplotlib/seaborn (for figures)
- numpy (for arrays)

```bash
ssh narval "python3 -c 'import torch; print(torch.cuda.is_available())'"
```
- [ ] Python dependencies installed
- [ ] CUDA/GPU accessible from Python

### 5. Local File Verification
```bash
cd /Users/studio/Desktop/PhD/Proposal/scripts
ls -la *.sh *.py *.md
```
- [ ] All 11 script files present
- [ ] All scripts are executable (x permission)
- [ ] README_EXPERIMENTS.md exists
- [ ] V5_SUMMARY.md exists
- [ ] QUICK_START.md exists

---

## 🚀 Execution Phase

### Step 1: Copy Files to Narval
```bash
scp -r /Users/studio/Desktop/PhD/Proposal/scripts narval:/project/def-bengioy/chromaguide/
```
- [ ] Scripts copied to Narval
- [ ] Verify: `ssh narval "ls /project/def-bengioy/chromaguide/scripts/"`

### Step 2: Run Master Orchestration
```bash
cd /Users/studio/Desktop/PhD/Proposal/scripts
bash orchestra_master.sh
```
- [ ] Command submitted successfully
- [ ] No immediate errors in console

### Step 3: Monitor Progress
Open separate terminal:
```bash
ssh narval
watch -n 30 'squeue -u $USER'
```
- [ ] See SLURM jobs submitted
- [ ] All 6 jobs show up in queue
- [ ] Jobs transition from PD (pending) → R (running)

### Step 4: Check Individual Job Logs
```bash
ssh narval "tail -f /project/def-bengioy/chromaguide_results/logs/*.log"
```
- [ ] Logs being written continuously
- [ ] No CUDA out-of-memory errors
- [ ] No module loading failures

### Step 5: Verify Data Download
```bash
ssh narval "ls -la /project/def-bengioy/chromaguide_data/raw/"
```
Expected after data download phase:
- [ ] DeepHF CSV files present (HEK293T.csv, HCT116.csv, HeLa.csv)
- [ ] ENCODE bigWig files present (9 files)
- [ ] Reference genome (optional)

### Step 6: Verify Splits Creation
```bash
ssh narval "ls -la /project/def-bengioy/chromaguide_data/splits/"
```
Expected after preprocessing:
- [ ] split_a_gene_held_out/ directory with train/val/test CSVs
- [ ] split_b_dataset_held_out/ subdirectories with splits
- [ ] split_c_cellline_held_out/ subdirectories with splits

### Step 7: Monitor Training Progress
Check size of results directory (should grow as models train):
```bash
watch -n 60 'ssh narval "du -sh /project/def-bengioy/chromaguide_results/"'
```
- [ ] Results directory growing (indicates training running)
- [ ] New model checkpoints appearing in results/models/

---

## ⏰ Timeline Checkpoints

| Time | Expected Status | Check Command |
|------|---|---|
| T+5 min | Data download started | `ssh narval "tail slurm_logs/data_download.log"` |
| T+30 min | Preprocessing running | `squeue -u $USER \| grep preprocessing` |
| T+1 hour | All 6 jobs submitted | `squeue -u $USER \| wc -l` (should = 6+) |
| T+2-3 hrs | Baseline finishing | `squeue -u $USER \| grep seq_baseline` |
| T+6-8 hrs | ChromaGuide halfway | `tail slurm_logs/chromaguide_full_*.log` |
| T+12-20 hrs | Most jobs running | `squeue -u $USER \| grep -c R` (should = 4-6) |
| T+20-24 hrs | Training complete | All jobs in queue complete |
| T+24-25 hrs | Evaluation running | `cat slurm_logs/evaluation.log` |
| T+25-26 hrs | Figures generated | `ls /project/def-bengioy/chromaguide_results/figures/` |

---

## 🔍 Sanity Checks During Execution

### Every 2 Hours
```bash
# Check all jobs still running
squeue -u $USER

# Check GPU memory usage
nvidia-smi

# Check error logs for warnings
ssh narval "grep -i error slurm_logs/*.err"
```
- [ ] No jobs in ERROR state
- [ ] GPU memory < 40GB (A100 max)
- [ ] No CUDA errors in logs

### If Job Fails
1. Note the job ID and name
2. Check the error log:
   ```bash
   ssh narval "cat slurm_logs/[job_name]_*.err"
   ```
3. Common issues:
   - [ ] Module not loaded → Add module commands
   - [ ] Out of memory → Reduce batch size
   - [ ] Timeout → Increase time in SLURM header
   - [ ] File not found → Check data download completed

### If Data Download Hangs
```bash
ssh narval "ps aux | grep wget"  # See if wget still running
ssh narval "tail -50 slurm_logs/data_download.log"  # Check last lines
```
- [ ] Download process still alive
- [ ] Recent log entries (< 1 hour old)
- [ ] If stuck > 2 hours, kill and restart

---

## ✅ Completion Verification

### After 24-30 Hours (Wait for This!)

#### 1. Check Results Directory
```bash
ssh narval "ls -lR /project/def-bengioy/chromaguide_results/ | head -50"
```
Expected structure:
```
✓ models/               (6 .pt files)
✓ predictions/         (6 results.json files)
✓ splits/              (train/val/test CSVs)
✓ statistics/          (CSV/JSON files)
✓ figures/             (6 PDF files)
✓ evaluation/          (evaluation_report.md)
✓ logs/                (slurm logs)
```
- [ ] All expected directories present
- [ ] No empty directories

#### 2. Verify Figure Files
```bash
ssh narval "file /project/def-bengioy/chromaguide_results/figures/*.pdf"
```
- [ ] 6 PDF files (not empty, actual PDFs)
- [ ] Reasonable file sizes (> 100KB each)

#### 3. Check Evaluation Report
```bash
ssh narval "head -50 /project/def-bengioy/chromaguide_results/evaluation/evaluation_report.md"
```
Should show:
- [ ] Model names
- [ ] Spearman ρ values
- [ ] Statistical test results
- [ ] Confidence intervals

#### 4. Verify Statistics Files
```bash
ssh narval "cat /project/def-bengioy/chromaguide_results/statistics/statistical_summary.json"
```
Should contain:
- [ ] primary_model metrics
- [ ] Comparisons section
- [ ] Ablation section
- [ ] p-values < 0.05

#### 5. Check Git Push
```bash
git log --oneline | head -5
git tag -l | grep v5
```
Should show:
- [ ] v5.0-real-experiments-complete tag
- [ ] Recent commit with "Real PhD thesis experiments"

---

## 📋 Data Quality Checks

### DeepHF Data
```bash
ssh narval "head -5 /project/def-bengioy/chromaguide_data/raw/HEK293T.csv"
```
- [ ] Has columns: sequence, target_gene, intensity
- [ ] ~10,000+ rows per cell line
- [ ] Intensity values in [0, 1] range

### ENCODE Tracks
```bash
ssh narval "ls -lh /project/def-bengioy/chromaguide_data/raw/ENCODE/*.bw"
```
- [ ] 9 bigWig files
- [ ] Each file > 100MB
- [ ] File timestamps recent (from download)

### Split Statistics
```bash
ssh narval "wc -l /project/def-bengioy/chromaguide_data/splits/split_a_gene_held_out/*.csv"
```
Expected ratios:
- [ ] train.csv: ~80% of total
- [ ] validation.csv: ~10% of total
- [ ] test.csv: ~10% of total

---

## 🎓 Results Quality Assessment

### Statistical Significance
Expected results should show:
- [ ] Spearman ρ values > 0.6 (meaningful correlation)
- [ ] p-values < 0.01 (statistically significant)
- [ ] Cohen's d > 0.5 (clinical/practical significance)
- [ ] 95% confidence intervals don't cross zero

### Model Performance
Expected hierarchy:
1. [ ] HPO Best: ρ ≈ 0.82-0.85 (best)
2. [ ] ChromaGuide Full: ρ ≈ 0.78-0.82
3. [ ] Mamba Variant: ρ ≈ 0.76-0.80
4. [ ] Ablation methods: ρ ≈ 0.70-0.77
5. [ ] Sequence Baseline: ρ ≈ 0.65-0.70 (worst)

If results don't match expected order, may indicate:
- [ ] Training not converged (increase epochs in SLURM scripts)
- [ ] Data leakage (verify preprocessing split logic)
- [ ] Hyperparameter suboptimal (check HPO results)

### Figure Quality
When viewing PDFs:
- [ ] No blank/empty plots
- [ ] Axes properly labeled
- [ ] Legend present and readable
- [ ] Title describes the plot
- [ ] No error messages overlaid on figure

---

## 🚨 Emergency Procedures

### If GPU Out of Memory
```bash
# Kill current job (if safe)
scancel $JOB_ID

# Edit SLURM script to reduce batch size:
# Change: batch_size = 32 → batch_size = 16

# Resubmit
sbatch slurm_chromaguide_full.sh
```
- [ ] OOM issue resolved
- [ ] Job resubmitted and running

### If Data Download Hangs
```bash
# Check if wget is running
ps aux | grep wget

# If hung for > 2 hours, kill it
pkill wget

# Re-run data download from scratch
bash download_real_data.sh
```
- [ ] Download process stopped
- [ ] Download restarted manually

### If Results Missing
```bash
# Check if evaluation scripts ran
ssh narval "ls -la /project/def-bengioy/chromaguide_results/evaluation/"

# If missing, run manually
ssh narval "python3 /project/def-bengioy/chromaguide/scripts/evaluation_and_reporting.py"
ssh narval "python3 /project/def-bengioy/chromaguide/scripts/figure_generation.py"
```
- [ ] Evaluation completed
- [ ] Figures generated

---

## Post-Execution: Using Results for Thesis

### 1. Download Results Locally
```bash
scp -r narval:/project/def-bengioy/chromaguide_results ~/thesis_results/
```
- [ ] All results downloaded

### 2. Extract Figures
```bash
cp ~/thesis_results/figures/*.pdf ~/dissertation/images/
```
- [ ] Figures ready for thesis chapters

### 3. Create Bio-Sketch from Report
```bash
cat ~/thesis_results/evaluation/evaluation_report.md
```
- [ ] Copy relevant sections into thesis methods/results

### 4. Reference in Bibliography
Add to .bib file:
```bibtex
@misc{chromaguide2024,
  author = {Your Name},
  title = {ChromaGuide: Multi-Modal Learning for CRISPR},
  url = {https://github.com/your-repo/chromaguide},
  year = {2024},
  note = {Experiment results, GitHub v5.0-real-experiments-complete}
}
```
- [ ] Citation added to thesis bibliography

---

## ✨ Final Status

- [ ] All 6 models trained successfully
- [ ] Statistical tests show significance (p < 0.05)
- [ ] Figures are high-quality PDFs
- [ ] Results match expected benchmarks
- [ ] Code pushed to GitHub with v5.0 tag
- [ ] Ready to include in PhD dissertation! 🎓

---

**You're all set! Run `bash orchestra_master.sh` and check back in ~30 hours for your PhD thesis results.**
