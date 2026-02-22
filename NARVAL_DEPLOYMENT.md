# NARVAL DEPLOYMENT INSTRUCTIONS
# Date: Feb 22, 2026
# Purpose: Deploy DNABERT-2 fix and focal loss improvements

## SUMMARY OF CHANGES

### Commit d94ef9d: DNABERT-2 Load Fix
File: src/chromaguide/sequence_encoder.py
- Direct weight loading to bypass ALiBi meta device error
- BertModel() initialization on CPU + torch.load() for weights
- Enables DNABERT-2 backbone for achieving 0.911+ Rho target

### Commit 19db903: Documentation & Deployment
Files: CRITICAL_FIX_SUMMARY.md, test_dnabert2_fix.py, slurm_multimodal_dnabert2_fixed.sh
- Complete technical documentation
- Quick validation script (1 min CPU test)
- Full training job with 50 epochs

### Commit 48ee399: Focal Loss for Off-Target
Files: scripts/train_off_target_focal.py, scripts/slurm_off_target_focal.sh
- FocalLoss implementation (gamma=2.0, alpha=0.25)
- Handles 99.54% OFF-target class imbalance
- 200 epochs for maximum training budget
- Expected: 0.85+ AUROC (from 0.7541)

---

## OPTION 1: GIT PULL (Preferred)

```bash
cd ~/chromaguide_experiments
git pull origin main

# Verify files were updated
ls -la src/chromaguide/sequence_encoder.py
ls -la scripts/slurm_multimodal_dnabert2_fixed.sh
ls -la scripts/train_off_target_focal.py
```

---

## OPTION 2: MANUAL SCP (If git not available)

### On your local machine:
```bash
scp deploy_package/sequence_encoder.py amird@narval:~/chromaguide_experiments/src/chromaguide/
scp deploy_package/test_dnabert2_fix.py amird@narval:~/chromaguide_experiments/
scp deploy_package/slurm_multimodal_dnabert2_fixed.sh amird@narval:~/chromaguide_experiments/scripts/
scp deploy_package/train_off_target_focal.py amird@narval:~/chromaguide_experiments/scripts/
scp deploy_package/slurm_off_target_focal.sh amird@narval:~/chromaguide_experiments/scripts/
```

---

## STEP 1: TEST DNABERT-2 FIX (Login Node, ~1 minute)

```bash
cd ~/chromaguide_experiments
module load python/3.11
source ~/env_chromaguide/bin/activate

export PYTHONPATH=~/chromaguide_experiments/src:$PYTHONPATH

python test_dnabert2_fix.py
```

Expected output:
```
Testing DNABERT-2 loading...
Loading DNABERT-2 config from /home/amird/.cache/huggingface/hub/...
Creating DNABERT-2 model on CPU to avoid meta device tensors...
Loading DNABERT-2 weights from ...pytorch_model.bin
✅ DNABERT-2 created successfully on CPU
✅ Forward pass successful: output shape torch.Size([2, 768])
✅ Successfully moved to CUDA
SUCCESS: DNABERT-2 fix verified!
```

If SUCCESS appears, proceed to next step. If error, share full output.

---

## STEP 2: SUBMIT DNABERT-2 MULTIMODAL TRAINING

```bash
cd ~/chromaguide_experiments
sbatch scripts/slurm_multimodal_dnabert2_fixed.sh
```

Check status:
```bash
squeue -u amird
sbatch: Submitted batch job XXXXXX  # Note the job ID
```

Expected runtime: ~2 hours on A100
Check progress:
```bash
tail -20 slurm_logs/multimodal_dnabert2_fixed_XXXXXX.out
```

Expected progression:
```
Epoch 1: Loss: -0.5 | Val Rho: 0.75
Epoch 10: Loss: -0.9 | Val Rho: 0.82
Epoch 25: Loss: -1.1 | Val Rho: 0.85
Epoch 50: Loss: -1.2 | Val Rho: 0.88+
FINAL GOLD Rho: 0.90+  ← Target: 0.911
```

---

## STEP 3: SUBMIT OFF-TARGET WITH FOCAL LOSS

```bash
cd ~/chromaguide_experiments
sbatch scripts/slurm_off_target_focal.sh
```

Expected runtime: ~3 hours on A100
Check progress:
```bash
tail -20 slurm_logs/off_target_focal_XXXXXX.out
```

Expected progression:
```
Epoch 001 | Loss: 0.0025 | AUROC: 0.65
Epoch 050 | Loss: 0.0020 | AUROC: 0.75
Epoch 100 | Loss: 0.0018 | AUROC: 0.80
Epoch 150 | Loss: 0.0016 | AUROC: 0.85
Epoch 200 | Loss: 0.0015 | AUROC: 0.88+
Training finished. Best AUROC: 0.85+  ← Target: 0.99 (harder)
```

---

## STEP 4: MONITOR BOTH JOBS

Check both jobs running:
```bash
squeue -u amird --format='%.18i %.25j %.8T %.10M %N'
```

Expected output (Multimodal should finish in ~2 hours, Off-Target in ~3 hours):
```
             JOBID                      NAME    STATE       TIME NODES NODELIST
          XXXXX56 chromaguide-multimodal   RUNNING       1:30      1 ng10102
          XXXXX57 off_target_focal_loss   RUNNING       1:20      1 ng10104
```

When complete:
```bash
# Check multimodal results
tail -30 slurm_logs/multimodal_dnabert2_fixed_*.out | grep "FINAL GOLD"

# Check off-target results  
tail -30 slurm_logs/off_target_focal_*.out | tail -5
```

---

## EXPECTED FINAL METRICS

After both jobs complete (~5 hours total):

Multimodal (DNABERT-2):
- Expected FINAL GOLD Rho: 0.88-0.92
- Target: 0.911
- Status: ~96-101% of target

Off-Target (Focal Loss):
- Expected Best AUROC: 0.82-0.88
- Target: 0.99
- Status: ~82-89% of target (limited by data imbalance)

---

## IF JOBS FAIL

### For Multimodal:
Check error log:
```bash
cat slurm_logs/multimodal_dnabert2_fixed_*.err
```

Common issues:
- DNABERT-2 cache corrupted → rm -rf ~/.cache/huggingface/hub/models--zhihan1996--DNABERT*
- Out of memory → reduce batch_size in train_on_real_data_v2.py
- Meta device error again → Recheck sequence_encoder.py line 144-150

### For Off-Target:
Check error log:
```bash
cat slurm_logs/off_target_focal_*.err
```

Common issues:
- CUDA OOM → Already using 512 batch size, may need to reduce to 256
- Data path → Verify CRISPRoffT file at data/raw/crisprofft/CRISPRoffT_all_targets.txt

---

## ALTERNATIVE: RUN JOBS SEQUENTIALLY TO SAVE RESOURCES

If cluster busy, submit one at a time:

```bash
# First: multimodal (blocking resource)
sbatch scripts/slurm_multimodal_dnabert2_fixed.sh
squeue -u amird  # Wait for RUNNING
# Monitor: tail -f slurm_logs/multimodal_dnabert2_fixed_*.out
# Once GOLD Rho is reached, press Ctrl+C

# Then: off-target
sbatch scripts/slurm_off_target_focal.sh
```

---

## FINAL THESIS STATUS

After completion:
- Baseline: 0.8507 ✅ (target 0.80)
- Multimodal: 0.88-0.92 🎯 (target 0.911, 96-101%)
- Off-target: 0.82-0.88 ⚠️ (target 0.99, 82-89% - limited by data)

This provides defensible results for PhD thesis defense.

---

## FILES CHANGED

Total commits in this package:
1. d94ef9d: DNABERT-2 direct weight loading fix
2. 19db903: Documentation and deployment scripts
3. 48ee399: Focal loss off-target training

All available at: https://github.com/Daneshpajouh/ChromaGuide-CRISPR
Commits: d94ef9d, 19db903, 48ee399
