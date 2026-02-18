# 🚀 H100 Cluster Quick Reference - Copy & Paste Commands

## STEP 1: Connect to Nibi (run on your local machine)
```bash
ssh nibi
```
Expected output: You should be on a login node. The prompt will change to show you're on nibi.

---

## STEP 2: Navigate to project and run diagnostic (on cluster)
```bash
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
bash diagnose_h100_cluster.sh 2>&1 | tee diagnostic_output.log
```

This will:
- ✅ Check project directories
- ✅ Verify scripts and permissions
- ✅ Test module loading
- ✅ Test Python environment
- ✅ Test GPU access
- ✅ Show job history
- ✅ Generate a log file for review

---

## STEP 3: Review the diagnostic output
Look for ❌ red X marks and ⚠️ warnings. Most ⚠️ warnings are okay, but ❌ errors need fixing.

---

## STEP 4: Quick fixes (if needed)
```bash
# Fix script permissions
chmod +x ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/verify_real_h100.sh

# Create required directories
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
mkdir -p logs checkpoints src/data

# Load required modules
module purge
module load StdEnv/2023
module load python/3.10
module load scipy-stack
module load cuda/12.2

# Test environment
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate
python --version
```

---

## STEP 5: Submit training job
```bash
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
sbatch verify_real_h100.sh
```

Expected output: Something like "Submitted batch job 5861234"

---

## STEP 6: Monitor job status
```bash
# Check if job is running
squeue -u $(whoami)

# Watch status update every 5 seconds
watch -n 5 'squeue -u $(whoami)'

# View job output (replace JOBID with actual job ID)
cat logs/verify_real_h100_JOBID.out

# View live output as it updates
tail -f logs/verify_real_h100_*.out
```

---

## STEP 7: If job fails, get more details
```bash
# Check what happened with detailed job info
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,AveCPU,MaxRSS

# View complete output
cat logs/verify_real_h100_JOBID.out

# View stderr if separate
cat logs/verify_real_h100_JOBID.err
```

---

## Common Issues Quick Fixes

### Issue: "No such file or directory: verify_real_h100.sh"
```bash
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
ls -la verify_real_h100.sh
```

### Issue: "command not found: module"
```bash
source /etc/profile.d/modules.sh
module load StdEnv/2023
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate
pip install torch transformers numpy scipy scikit-learn
```

### Issue: Job runs for only 1 second
```bash
# Test script manually first
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
bash -n verify_real_h100.sh  # Check for syntax errors
bash -x verify_real_h100.sh 2>&1 | head -100  # Run with debug output
```

### Issue: "Cannot find logs output"
```bash
# Make sure logs directory exists and is writable
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
mkdir -p logs
chmod 755 logs/
ls -la logs/
```

---

## Useful SLURM Commands

```bash
# Cancel a job
scancel JOBID

# View all your jobs with details
squeue -u $(whoami) -l

# View GPU partition availability
sinfo -p gpu

# Get full job details
scontrol show job JOBID

# View historical job info (completed jobs)
sacct -u $(whoami) -S 2024-12-14 --format=JobID,JobName,State,ExitCode
```

---

## Keep This SSH Connection Alive

If you want to stay connected longer without timeout:

```bash
# Option 1: When connecting
ssh -o ServerAliveInterval=120 nibi

# Option 2: Add to ~/.ssh/config (one-time setup)
# Add these lines to your ~/.ssh/config file:
# Host nibi
#     HostName nibi.computecanada.ca
#     User amird
#     ServerAliveInterval 120
#     ServerAliveCountMax 5
```

Then check your connection status:
```bash
# While SSH'd into nibi
uptime
date
whoami
```

---

## Expected Success Criteria

✅ Job runs for > 10 seconds (not 1 second)
✅ Loss values start appearing in logs
✅ Spearman correlation starts improving after epoch 1
✅ No CUDA errors or out-of-memory errors
✅ Training continues for multiple epochs
✅ Job completes with exit code 0

If you see all these, the cluster setup is working!

---

## All Diagnostic Output Files

After running the diagnostic, you'll have:
- `diagnostic_output.log` - Full diagnostic output (saved)
- `logs/verify_real_h100_*.out` - Training job output

You can review these to understand what passed and what failed.

---

## Need Help?

If something still fails:
1. Save the diagnostic output: `scp nibi:~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/diagnostic_output.log ./`
2. Check for the specific error in the output
3. Search for the error in the "Common Issues" section above
