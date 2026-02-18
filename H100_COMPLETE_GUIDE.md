# 🎯 Complete H100 Diagnostic & Recovery Guide

**Date:** February 16, 2026  
**Goal:** Fix job 5860955 failure and validate H100 cluster setup  
**Estimated Time:** 15-30 minutes

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step SSH Connection](#step-by-step-ssh-connection)
3. [Running the Diagnostic](#running-the-diagnostic)
4. [Analyzing Results](#analyzing-results)
5. [Fixing Common Issues](#fixing-common-issues)
6. [Re-Submitting the Job](#re-submitting-the-job)
7. [Monitoring the Job](#monitoring-the-job)

---

## Prerequisites

Before you start, make sure you have:
- ✅ SSH access to `nibi` cluster
- ✅ SSH key configured (or password ready)
- ✅ Terminal/command line access on your local machine
- ✅ Files from the project directory

---

## Step-by-Step SSH Connection

### Step 1.1: Open your terminal
```bash
# On macOS: Open Terminal.app or iTerm2
# On Linux: Open your terminal emulator
# On Windows: Use WSL2 terminal or PuTTY
```

### Step 1.2: Test SSH connectivity
```bash
# First, test if you can reach the cluster
ssh nibi

# You should see output like:
# Last login: Mon Dec 16 10:23:45 2024 from ...
# [amird@nibi2 ~]$
```

**If this fails:**
```bash
# Try with explicit hostname and port
ssh -v amird@nibi.computecanada.ca -p 22

# If you get "Permission denied", check your SSH key
ssh-add ~/.ssh/your_key_file
```

### Step 1.3: Verify you're on the right system
```bash
# After connecting, verify you're on nibi
hostname
# Expected: something like "nibi2" or "nibi3"

# Check your current location
pwd
# Expected: /home/amird (or your username)
```

---

## Running the Diagnostic

### Step 2.1: Copy diagnostic script to cluster

**Option A: Copy from local machine (run on your terminal, not SSH)**
```bash
# From your local machine (not SSH'd):
scp diagnose_h100_cluster.sh nibi:~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/

# Verify copy succeeded
ssh nibi 'ls -lh ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/diagnose_h100_cluster.sh'
```

**Option B: Create script directly on cluster**
```bash
# SSH into cluster first
ssh nibi

# Navigate and create the script
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Paste the diagnostic script content into a file
cat > diagnose_h100_cluster.sh << 'EOF'
# (paste the full diagnostic script here)
EOF

# Make it executable
chmod +x diagnose_h100_cluster.sh
```

### Step 2.2: Run the diagnostic script

```bash
# SSH into cluster (if not already)
ssh nibi

# Go to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Run diagnostic with output saved to a log file
bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log

# Expected runtime: 30-60 seconds
# Look for ✅, ⚠️, and ❌ symbols in output
```

### Step 2.3: Save the output for analysis
```bash
# After diagnostic completes, copy output to local machine
# (Run on your local machine, not SSH'd)
scp nibi:~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/diagnostic_output.log ./

# View the output
cat diagnostic_output.log
```

---

## Analyzing Results

### What to look for:

#### ✅ GREEN: All good (these should all be present)
- Project directory exists
- verify_real_h100.sh exists and is executable
- src/train.py exists
- logs/ directory exists
- Python modules load
- Environment found and can be activated

#### ⚠️ YELLOW: Warnings (usually okay, but worth noting)
- Module load warnings (often just informational)
- Missing packages (can install with pip)
- Environment not yet activated (gets fixed in next section)

#### ❌ RED: Critical Issues (must be fixed)
- Project directory not found
- Script file missing
- Environment directory missing
- Cannot activate Python environment
- CUDA not available (if on compute node)

### Common Diagnostic Outputs

**GOOD OUTPUT (from successful diagnostic):**
```
✅ Project directory exists
✅ verify_real_h100.sh exists
✅ verify_real_h100.sh is executable
✅ src/train.py exists
✅ logs/ exists
✅ checkpoints/ exists
✅ StdEnv/2023 loaded
✅ python/3.10 loaded
✅ scipy-stack loaded
✅ cuda/12.2 loaded
✅ Python found: /usr/bin/python3
✅ Environment found: /home/amird/projects/def-kwiese/amird/mamba_h100
✅ Activation script exists
```

**BAD OUTPUT (examples that need fixing):**
```
❌ Project directory NOT found: /home/amird/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
```

---

## Fixing Common Issues

### Issue #1: Project Directory Not Found

**Symptom:**
```
❌ Project directory NOT found: /home/amird/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
```

**Fix:**
```bash
# SSH into cluster
ssh nibi

# Check if directory exists with different path
ls -la ~
ls -la ~/projects/
ls -la ~/projects/def-kwiese/
ls -la ~/projects/def-kwiese/amird/

# If entire path missing, the project hasn't been synced
# Option A: Sync from local machine
rsync -avz ~/Desktop/PhD/Proposal/src \
  nibi:~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/

# Option B: Contact your advisor - might be on different cluster path
```

### Issue #2: Script Not Executable

**Symptom:**
```
⚠️ verify_real_h100.sh is NOT executable
   Attempting to fix: chmod +x verify_real_h100.sh
```

**Fix (automatic in diagnostic, but manual if needed):**
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
chmod +x verify_real_h100.sh
chmod +x diagnose_h100_cluster.sh

# Verify
ls -la verify_real_h100.sh
# Should show: -rwxr-xr-x (with 'x' for executable)
```

### Issue #3: Environment Not Found

**Symptom:**
```
❌ Environment NOT found at: /home/amird/projects/def-kwiese/amird/mamba_h100
```

**Fix:**
```bash
ssh nibi

# Check what environment toolkit you have
which conda
which mamba

# Create the environment
# Using mamba (preferred):
mamba create -p ~/projects/def-kwiese/amird/mamba_h100 python=3.10 -y

# Or using conda:
conda create -p ~/projects/def-kwiese/amird/mamba_h100 python=3.10 -y

# Activate and install packages
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate
pip install --upgrade pip
pip install torch transformers numpy scipy scikit-learn pandas tqdm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Issue #4: Python Module Not Found

**Symptom:**
```
⚠️ torch not available (may install during training)
```

**Fix:**
```bash
ssh nibi
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate

# Install missing packages
pip install torch  # This will take a few minutes
pip install transformers numpy scipy scikit-learn pandas tqdm

# Verify
python -c "import torch, transformers; print('✓ All imported successfully')"
```

### Issue #5: Job Fails Immediately (1 second runtime)

**Symptom from previous attempt:**
Job ID 5860955 failed in 1 second with no output

**Root Causes & Fixes:**

**Cause A: Module loading failure**
```bash
ssh nibi
# Test module loading manually
source /etc/profile.d/modules.sh
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module list  # Should show all loaded modules
```

**Cause B: Script syntax error**
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Check for syntax errors
bash -n verify_real_h100.sh
# If no output, script is syntactically correct

# Run with debug output
bash -x verify_real_h100.sh 2>&1 | head -50
```

**Cause C: Directory permissions**
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
mkdir -p logs checkpoints
chmod 755 logs checkpoints
chmod 755 .

# Verify directories writable
touch logs/test.txt && rm logs/test.txt
echo "✓ logs directory is writable"
```

**Cause D: SLURM output directory issue**
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Check SLURM configuration
scontrol show config | grep -E "SlurmdLogFile|SlurmctldLogFile"

# Ensure output format is correct in script (look for #SBATCH #SBATCH -o logs/...)
head -20 verify_real_h100.sh | grep "SBATCH"
```

---

## Re-Submitting the Job

### Step 3.1: Verify all issues are fixed
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Do a final check
bash diagnose_h100_cluster.sh 2>&1 | tail -50
# Look for ✅ checks, should be mostly green
```

### Step 3.2: Make script executable again
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
chmod +x verify_real_h100.sh
```

### Step 3.3: Submit the job
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Submit SLURM job
sbatch verify_real_h100.sh

# Expected output:
# Submitted batch job 5861234
# (note the new job ID)

# Save this job ID for monitoring
JOB_ID="5861234"
```

### Step 3.4: Verify job was submitted
```bash
ssh nibi

# Check if job is in queue
squeue -u $(whoami)

# Should show your job with status PENDING or RUNNING
# Example output:
# JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 5861234 gpu ... amird PD       0:00      1 (None)
```

---

## Monitoring the Job

### Option 1: Check Status Once (Quick Check)
```bash
ssh nibi
squeue -u $(whoami)

# Or check specific job
squeue -j 5861234
```

### Option 2: Watch Status Live (Best Option)
```bash
ssh nibi

# This will refresh every 5 seconds until you Ctrl+C
watch -n 5 'squeue -u $(whoami)'
```

### Option 3: Stream Output as it Runs
```bash
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Get the latest log file
tail -f logs/verify_real_h100_*.out

# Or after job ID is known:
tail -f logs/verify_real_h100_5861234.out
```

### Option 4: Non-Interactive Monitoring (For leaving SSH)
```bash
ssh nibi

# Create a monitoring script
cat > monitor_job.sh << 'EOF'
#!/bin/bash
JOB_ID=$1
while squeue -j $JOB_ID > /dev/null 2>&1; do
  echo "$(date) - Job $JOB_ID still running..."
  sleep 30
done
echo "$(date) - Job $JOB_ID completed"
tail -100 ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/logs/verify_real_h100_${JOB_ID}.out
EOF

chmod +x monitor_job.sh
./monitor_job.sh 5861234 > job_monitor.log 2>&1 &
```

---

## What Success Looks Like

### Successful Job Submission
```bash
$ sbatch verify_real_h100.sh
Submitted batch job 5861234
```

### Successful Job Start
```bash
$ squeue -u amird
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
5861234 gpu verify_r amird  R       0:15      1 gpu018
# Status is 'R' (running) and TIME is increasing
```

### Successful Training Output (first few lines of logs)
```bash
$ tail -50 logs/verify_real_h100_5861234.out

====================================================================
Starting ChromaGuide Training
====================================================================
Epoch 1/10 [████░░░░░░] 40% - Loss: 0.4523
Epoch 1/10 [████████░░] 80% - Loss: 0.4201
Epoch 1/10 [██████████] 100% - Loss: 0.3987

Epoch 2/10 [████████░░] 80% - Loss: 0.3654
...
```

### Successful Job Completion
```bash
$ squeue -u amird
# (no output - job has completed)

$ sacct -j 5861234 --format=JobID,JobName,State
       JobID     JobName      State
------------ ---------- ----------
5861234       verify_re  COMPLETED
```

---

## Next Steps After Success

Once the job completes successfully:

1. **Review the output**
   ```bash
   cat ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/logs/verify_real_h100_*.out | tail -100
   ```

2. **Check success criteria**
   - ✅ Job ran for > 10 minutes (not 1 second)
   - ✅ Loss values decreased over time
   - ✅ No CUDA errors
   - ✅ Model saved to checkpoints/

3. **Proceed with Phase 1 training**
   - Start training on full DeepHF dataset
   - Target: Spearman > 0.88

4. **Track metrics**
   - Monitor on-target Spearman
   - Monitor off-target AUROC
   - Compare against SOTA models

---

## Troubleshooting Commands Summary

```bash
# Quick diagnostics (run from local machine)
ssh nibi 'ls -la ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/'
ssh nibi 'squeue -u $(whoami)'
ssh nibi 'cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X && tail -50 logs/verify_real_h100_*.out'

# View all your jobs history
ssh nibi 'sacct -u $(whoami) -S 2024-12-14 --format=JobID,JobName,State,ExitCode,Start,End'

# Kill a stuck job
ssh nibi 'scancel JOBID'

# Create persistent SSH connection (doesn't disconnect on idle)
ssh -o ServerAliveInterval=120 nibi
```

---

## Final Checklist

Before you start:
- [ ] You can SSH to nibi
- [ ] You can run basic commands on the cluster
- [ ] The diagnostic script exists in your project directory

During diagnostic:
- [ ] Diagnostic script completes without errors
- [ ] Most checks show ✅ (some ⚠️ is okay)
- [ ] No ❌ critical errors remain
- [ ] Output saved to diagnostic_output.log

After fixing:
- [ ] Second diagnostic run shows mostly ✅
- [ ] Job submits successfully (sbatch returns job ID)
- [ ] Job status becomes RUNNING (within 30 seconds)
- [ ] Logs appear in logs/ directory

Success criteria:
- [ ] Job runs for > 10 minutes
- [ ] Training loss decreases
- [ ] Checkpoints saved
- [ ] No CUDA errors
- [ ] Job completes with exit code 0

Good luck! 🚀
