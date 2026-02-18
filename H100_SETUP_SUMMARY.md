# 🎯 H100 Cluster Setup - Summary & Next Steps

**Created:** February 16, 2026  
**Status:** ✅ All diagnostic tools created and ready to use  
**Files Created:** 4

---

## 📦 Files Created for You

### 1. **diagnose_h100_cluster.sh** ⭐ MAIN TOOL
   - **Purpose:** Comprehensive diagnostic of cluster setup
   - **What it checks:**
     - Project directory structure
     - Script files and permissions
     - Required directories (logs, checkpoints)
     - Module availability (StdEnv, Python, CUDA)
     - Python environment installation
     - GPU access
     - SLURM job status
     - Previous outputs
   - **How to use:**
     ```bash
     ssh nibi
     cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
     bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log
     ```
   - **Time to run:** ~1 minute
   - **Output:** Terminal report + logfile

### 2. **H100_QUICK_REFERENCE.md** ⭐ BEST FOR QUICK COPY-PASTE
   - **Purpose:** Quick copy-paste commands
   - **Contains:**
     - Step-by-step commands you can copy directly
     - Common issues and instant fixes
     - Useful SLURM commands
     - Expected success criteria
   - **Best for:** When you just want to run the commands
   - **Read time:** 2 minutes

### 3. **H100_COMPLETE_GUIDE.md** ⭐ BEST FOR DETAILED WALKTHROUGH
   - **Purpose:** Comprehensive step-by-step guide
   - **Contains:**
     - Detailed explanations of each step
     - What to look for in outputs
     - Root causes of failures
     - Multiple troubleshooting options
     - Success criteria for each phase
   - **Best for:** First time setup or when something goes wrong
   - **Read time:** 15-20 minutes to skim, 30-45 minutes to follow completely

### 4. **SSH_CLUSTER_GUIDE.sh**
   - **Purpose:** Reference for SSH and terminal commands
   - **Contains:**
     - SSH connection variations
     - All diagnostic commands
     - Module testing commands
     - Job submission and monitoring
     - SLURM troubleshooting
   - **Best for:** Keeping as a reference document

---

## 🚀 Quick Start (TL;DR)

**If you're in a hurry, run exactly these commands:**

```bash
# Step 1: SSH to cluster (from your local terminal)
ssh nibi

# Step 2: Run diagnostic (on cluster)
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log

# Step 3: Review output - look for ✅ or ❌
# Step 4: If any ❌, check H100_QUICK_REFERENCE.md for fixes
# Step 5: Fix issues and re-run diagnostic
# Step 6: Submit job
sbatch verify_real_h100.sh

# Step 7: Monitor
squeue -u $(whoami)
# or watch with:
watch -n 5 'squeue -u $(whoami)'
```

**Expected total time:** 10-20 minutes (if no major issues)

---

## 📊 What Each File Helps You With

### Use **diagnose_h100_cluster.sh** for:
- ✅ Automated checking of all components
- ✅ Identifying exactly what's broken
- ✅ Comprehensive system verification
- ✅ Getting baseline status before/after fixes

### Use **H100_QUICK_REFERENCE.md** for:
- ✅ Fast copy-paste solutions
- ✅ Common issue quick fixes
- ✅ SLURM command cheat sheet
- ✅ When you're in a hurry

### Use **H100_COMPLETE_GUIDE.md** for:
- ✅ Understanding why things fail
- ✅ Detailed explanations of each step
- ✅ Troubleshooting complex issues
- ✅ Learning how the cluster works

### Use **SSH_CLUSTER_GUIDE.sh** for:
- ✅ Reference of all available commands
- ✅ Advanced troubleshooting
- ✅ Setting up SSH for easier access

---

## 🎯 The Problem We're Solving

**Issue:** Job 5860955 failed after 1 second with no output
- **When:** December 14, 2025
- **Symptom:** Job exits immediately with no log files created
- **Likely causes:**
  1. Module loading failure
  2. Python/CUDA not properly loaded
  3. Environment not found
  4. Directory permission issues
  5. Script syntax error

**Solution:**
1. **Diagnose** - Use `diagnose_h100_cluster.sh` to identify the issue
2. **Fix** - Use guidance to fix identified problems
3. **Verify** - Re-run diagnostic to confirm fixes
4. **Submit** - Submit job again with working configuration
5. **Monitor** - Watch for successful training output

---

## 📋 Steps to Follow (In Order)

### Phase 1: Diagnostic (15 minutes)
1. Open terminal
2. SSH to nibi: `ssh nibi`
3. Run diagnostic script
4. Review output for ✅ and ❌
5. Save output: `scp diagnostic_output.log ./`

### Phase 2: Fixing Issues (10-30 minutes, depends on issues found)
1. For each ❌ found, check **H100_QUICK_REFERENCE.md**
2. Run the suggested fix command
3. Re-run diagnostic: `bash diagnose_h100_cluster.sh`
4. Repeat until all ✅

### Phase 3: Job Submission (5 minutes)
1. Ensure script is executable: `chmod +x verify_real_h100.sh`
2. Submit job: `sbatch verify_real_h100.sh`
3. Note the job ID from output
4. Verify it's in queue: `squeue -u $(whoami)`

### Phase 4: Monitoring (varies, typically 5-30 minutes)
1. Watch job status: `watch -n 5 'squeue -u $(whoami)'`
2. Once status is RUNNING, check logs: `tail -f logs/verify_real_h100_*.out`
3. Look for training output (loss decreasing)
4. Wait for job to complete

### Phase 5: Success Verification (5 minutes)
1. Job completed (exit code 0)
2. Training loss decreased
3. Checkpoints saved
4. No CUDA errors in output
5. Ready to proceed to Phase 1 training on real data

---

## ✅ Checklist for Today

### Before You Start
- [ ] You can SSH to `nibi`
- [ ] You have files: `diagnose_h100_cluster.sh`
- [ ] Terminal is open and ready

### While Running Diagnostic
- [ ] SSH successful
- [ ] In correct directory: `~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X`
- [ ] Diagnostic script starts and runs
- [ ] Output shows ✅ or ❌ symbols

### After Diagnostic
- [ ] Reviewed all ❌ critical errors (if any)
- [ ] Found corresponding fixes in quick reference
- [ ] Applied fixes one by one
- [ ] Re-ran diagnostic to verify fixes

### Job Submission
- [ ] All diagnostics pass (mostly ✅)
- [ ] Script is executable
- [ ] Job submitted successfully
- [ ] Got a job ID

### Monitoring
- [ ] Job appears in queue
- [ ] Job status changes to RUNNING (within 30s)
- [ ] Log files appear in `logs/` directory
- [ ] Training output visible (decreasing loss)

---

## 🎁 Bonus Features Included

### 1. Automatic Fixes in Diagnostic
- The diagnostic script will automatically try to fix some issues:
  - Making scripts executable (chmod +x)
  - Creating missing directories
  - Clearing out stale data

### 2. Detailed Output
- The diagnostic:
  - Tests each component independently
  - Shows exactly what passed/failed
  - Provides version information
  - Shows available resources
  - Lists recent SLURM jobs

### 3. Persistent Logging
- All output saved to `diagnostic_output.log`
- Can be reviewed later
- Can be shared for debugging
- Useful for comparing before/after fixes

### 4. Multiple Troubleshooting Levels
- **Level 1:** Quick reference (copy-paste solutions)
- **Level 2:** Complete guide (detailed explanations)
- **Level 3:** SSH guide (advanced commands)
- **Level 4:** Diagnostic output (raw data)

---

## 🚨 If Something Goes Wrong

### Common First-Time Issues

**"ModuleNotFoundError: No module named 'torch'"**
- → Check **H100_QUICK_REFERENCE.md** section "Issue: ModuleNotFoundError"
- → Run: `source ~/projects/def-kwiese/amird/mamba_h100/bin/activate && pip install torch`

**"Job runs for 1 second and exits"**
- → Check **H100_COMPLETE_GUIDE.md** section "Issue #5"
- → Run diagnostic again to identify exact failure point
- → Run script manually with debug: `bash -x verify_real_h100.sh 2>&1 | head -50`

**"Connection timeout or SSH refused"**
- → Check internet connection
- → Try: `ssh -v nibi` for verbose error messages
- → Check: `ssh amird@nibi.computecanada.ca` explicitly

**"Command not found: module"**
- → Check **H100_QUICK_REFERENCE.md** section "Issue: command not found: module"
- → Run: `source /etc/profile.d/modules.sh` before using modules

---

## 📞 When to Ask for Help

You should have clear answers before asking for help:

1. **Run the diagnostic:**
   ```bash
   bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log
   ```

2. **Save the output:**
   ```bash
   scp diagnostic_output.log ~/Desktop/
   ```

3. **Note any ❌ errors**

4. **Try fixes from quick reference**

5. **If still stuck, provide:**
   - Exact error message
   - The command that failed
   - Output from `diagnostic_output.log`
   - Current job status from `squeue -u $(whoami)`

---

## 🎯 Success Looks Like This

### Successful Diagnostic Output
```
✅ Project directory exists
✅ verify_real_h100.sh exists
✅ verify_real_h100.sh is executable
✅ src/train.py exists
✅ logs/ exists
✅ checkpoints/ exists
✅ StdEnv/2023 loaded
✅ python/3.10 loaded
✅ cuda/12.2 loaded
✅ Python execution works
✅ main training module imports
```

### Successful Job Submission
```
$ sbatch verify_real_h100.sh
Submitted batch job 5861234
```

### Successful Job Running
```
$ squeue -u amird
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
5861234 gpu verify_r amird R 5:30 1 gpu018
```

### Successful Training Output
```
Epoch 1/10 [████░░░░░░] 40% - Loss: 0.4523
Epoch 1/10 [████████░░] 80% - Loss: 0.4201
Epoch 1/10 [██████████] 100% - Loss: 0.3987
Epoch 2/10 [████████░░] 80% - Loss: 0.3654
```

---

## 📚 File Reference

| File | Purpose | When to Use | Length |
|------|---------|------------|--------|
| `diagnose_h100_cluster.sh` | Automated diagnostics | Always run first | ~200 lines |
| `H100_QUICK_REFERENCE.md` | Quick fixes | During troubleshooting | 1-2 pages |
| `H100_COMPLETE_GUIDE.md` | Detailed walkthrough | For complete understanding | 8-10 pages |
| `SSH_CLUSTER_GUIDE.sh` | Command reference | As reference material | 3-4 pages |

---

## 🎉 What's Next After Success

Once the H100 diagnostic passes and training runs successfully:

1. **Phase 1: Foundation Training** (Week 1-2)
   - Train on DeepHF dataset (60k+ sgRNAs)
   - Target: Spearman > 0.88 on high-throughput data
   - Expected training time: 24-48 hours on H100

2. **Phase 2: Transfer Learning** (Week 3)
   - Fine-tune on CRISPRon (23,902 gRNAs)
   - Fine-tune on endogenous datasets
   - Target: Spearman > 0.60 on functional data

3. **Phase 3: Multi-Task Learning** (Week 4)
   - Simultaneous on-target + off-target optimization
   - GUIDE-seq integration
   - Target: AUROC > 0.99, PR-AUC > 0.90

4. **Phase 4: Ensemble & Optimization** (Week 5)
   - 10x ensemble with random initialization
   - Hyperparameter tuning
   - Ablation studies

5. **Phase 5: Validation & Publication** (Week 6+)
   - Benchmark against SOTA models
   - Prospective experimental validation
   - Paper submission

---

## 💡 Pro Tips

1. **Keep SSH connection alive:**
   ```bash
   ssh -o ServerAliveInterval=120 nibi
   ```

2. **Use tmux/screen for persistent sessions:**
   ```bash
   ssh nibi
   tmux new-session -s training
   # Your commands here
   # Detach with Ctrl+B, D
   # Reconnect with: tmux attach -t training
   ```

3. **Monitor multiple ways simultaneously:**
   ```bash
   # Terminal 1: Watch job status
   watch -n 5 'squeue -u $(whoami)'
   
   # Terminal 2: Stream output
   tail -f logs/verify_real_h100_*.out
   
   # Terminal 3: Check resources
   sinfo
   ```

4. **Save important outputs:**
   ```bash
   scp nibi:logs/verify_real_h100_*.out ~/Desktop/training_logs/
   ```

---

## 🎯 Summary

You now have:
- ✅ **diagnose_h100_cluster.sh** - Automated diagnostic tool
- ✅ **H100_QUICK_REFERENCE.md** - Quick solutions
- ✅ **H100_COMPLETE_GUIDE.md** - Detailed walkthrough
- ✅ **SSH_CLUSTER_GUIDE.sh** - Command reference

Everything you need to:
1. Diagnose the H100 cluster setup
2. Fix any issues that arise
3. Submit and monitor training jobs
4. Verify successful training
5. Proceed with Phase 1 pre-training

**Next action: SSH into nibi and run the diagnostic!**

```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log
```

Good luck! 🚀
