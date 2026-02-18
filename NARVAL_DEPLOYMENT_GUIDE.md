# Narval Cluster Deployment Guide

## Overview

This guide walks through deploying the DNABERT-Mamba training pipeline to the Narval cluster.

### Key Information
- **Cluster**: Narval (Alliance Canada)
- **Queue**: GPU-enabled compute nodes
- **Job Manager**: SLURM
- **Time Limit**: 24 hours (adjustable)
- **GPU**: NVIDIA A100 (high-performance)
- **Account**: def-kimballa

---

## Prerequisites

### Local Machine
- ✅ SSH configured with ControlMaster (persistent for 48 hours)
- ✅ VPN connected to SFU/Canadian clusters
- ✅ SLURM job script (`train_narval.sh`)
- ✅ Training script (`train_phase1.py`)
- ✅ Requirements file (`requirements.txt`)

### Narval Cluster
- Narval account activated (access_systems approved)
- MFA enabled and working (Duo)
- Compute allocation available
- Storage quota available

---

## Step-by-Step Deployment Workflow

### Step 1: Establish Persistent SSH Connection

This is critical - you must do this ONCE to create the ControlMaster socket for 48 hours.

```bash
# Terminal 1 - Keep this open for all subsequent operations
ssh narval
# When prompted for MFA:
# - Enter: 1 (for Duo Push)
# - Approve on your iPhone
# - You're now connected to narval login node
# - KEEP THIS SESSION OPEN
```

Connection established. Now use other terminals for commands (they'll use cached socket).

### Step 2: Navigate to Project Directory

```bash
# Terminal 2 (new, uses cached socket - no MFA needed)
ssh narval 'cd $HOME && mkdir -p crispro_project && ls'
```

### Step 3: Transfer Project Files to Cluster

```bash
# From local machine in project directory
# Copy training scripts
scp train_narval.sh narval:~/crispro_project/
scp train_phase1.py narval:~/crispro_project/
scp requirements.txt narval:~/crispro_project/

# Copy model/source code
scp -r src/model narval:~/crispro_project/
scp -r src/api narval:~/crispro_project/
scp -r src/utils narval:~/crispro_project/

# Verify transfer
ssh narval 'ls -la ~/crispro_project/' | head -20
```

### Step 4: Set Up Data on Cluster

```bash
# Check available data directories
ssh narval 'ls -lh /scratch/$USER/ 2>/dev/null || echo "Check home directory"'

# Create data directories
ssh narval 'mkdir -p ~/crispro_project/data/processed'
ssh narval 'mkdir -p ~/crispro_project/logs'
ssh narval 'mkdir -p ~/crispro_project/checkpoints'

# Transfer datasets (if < 5GB)
scp -r data/processed/* narval:~/crispro_project/data/processed/

# OR download on cluster (if datasets available online)
ssh narval 'cd ~/crispro_project && python3 download_datasets.py --dataset deephf'
```

### Step 5: Submit SLURM Job

```bash
# Connect to cluster (uses cached socket)
ssh narval
# OR directly submit:
ssh narval 'cd ~/crispro_project && sbatch train_narval.sh'

# Expected output:
# Submitted batch job 123456
```

**Note the job ID** (e.g., `123456`) for monitoring.

### Step 6: Monitor Job Status

```bash
# Check job status
ssh narval 'squeue -u $USER'

# Watch real-time logs
ssh narval 'tail -f ~/crispro_project/logs/phase1_123456.out'

# Check GPU utilization
ssh narval 'gpustat -i 1'

# Check detailed job info
ssh narval 'sstat -j 123456.batch --format=AveCPU,AveVMSize,MaxVMSize'
```

### Step 7: Retrieve Results

```bash
# Once job completes, download results
scp -r narval:~/crispro_project/checkpoints/phase1_* ./results/narval/

# Download logs
scp narval:~/crispro_project/logs/* ./logs/narval/

# View training metrics
cat results/narval/phase1_*/training_history.json | python -m json.tool
```

---

## Monitoring Commands

### Real-time Job Monitoring

```bash
# All your jobs
ssh narval 'squeue -u $(whoami)'

# Specific job details
ssh narval 'sstat -j 123456.batch --format=JobID,Elapsed,CPUTime,AveCPU,AveVMSize,MaxVMSize'

# GPU status
ssh narval 'nvidia-smi'

# Cluster load
ssh narval 'sinfo -p gpu -o "PARTITION AVAILABLE TIMELIMIT NODES STATE"'
```

### Log Monitoring

```bash
# Standard output
ssh narval 'tail -50 ~/crispro_project/logs/phase1_123456.out'

# Training progress
ssh narval 'grep -E "Epoch|Loss|Spearman" ~/crispro_project/logs/phase1_123456.log'

# Error messages
ssh narval 'tail -50 ~/crispro_project/logs/phase1_123456.err'

# Training metrics (after completion)
ssh narval 'cat ~/crispro_project/checkpoints/phase1_*/training_history.json | python -m json.tool'
```

### Job Control

```bash
# Cancel job
ssh narval 'scancel 123456'

# Check job history
ssh narval 'sacct -j 123456'

# Estimate job priority
ssh narval 'squeue -u $(whoami) -o "JOBID PRIORITY"'
```

---

## Troubleshooting

### Problem: SSH times out with "Connection refused"

**Solution**: Reconnect with fresh MFA approval:
```bash
# Kill existing socket (forces new auth)
rm ~/.ssh/sockets/amird@narval.alliancecan.ca:22

# Reconnect (needs MFA)
ssh narval
```

### Problem: SLURM job fails to start

**Checks**:
```bash
# Verify account access
ssh narval 'saucctrl --list'

# Check resource allocation
ssh narval 'salloc --account=def-kimballa -t 5 exit'

# Check queue limitations
ssh narval 'sinfo -p gpu -o "PARTITION TIMELIMIT NODES STATE"'
```

### Problem: Job exceeds memory or time

**Solutions**:
- Increase memory: `#SBATCH --mem=128G`
- Increase time: `#SBATCH --time=48:00:00`
- Reduce batch size: `--batch_size=16` in train_phase1.py

### Problem: GPU not available

**Check**:
```bash
ssh narval 'nvidia-smi'                    # See which GPUs available
ssh narval 'sinfo -p gpu --Node'           # Check node availability
ssh narval 'squeue -p gpu -o "JOBID NODES" | head'  # See what's running
```

---

## ControlMaster Socket Management

### Check Socket Status

```bash
# List active sockets
ls -la ~/.ssh/sockets/

# Test socket (should be instant)
time ssh narval 'hostname'

# Check how long socket has been active
stat ~/.ssh/sockets/amird@narval.alliancecan.ca:22

# Socket expiration
echo "Socket expires at: $(date -d '+48 hours' '+%Y-%m-%d %H:%M:%S')"
```

### Reset Socket (requires new MFA)

```bash
# Remove old socket to force re-authentication
rm ~/.ssh/sockets/amird@narval.alliancecan.ca:22

# Next SSH command triggers MFA again
ssh narval 'echo "Re-authenticated"'
```

---

## Complete Workflow Summary

```bash
# 1. One-time ControlMaster setup (Terminal 1, keep open for 48h)
ssh narval
# [Approve Duo MFA]
# Keep this session open

# 2. Deploy to cluster (Terminal 2, uses cached socket)
scp train_narval.sh narval:~/crispro_project/
scp train_phase1.py narval:~/crispro_project/
scp requirements.txt narval:~/crispro_project/
scp -r src/model narval:~/crispro_project/
scp -r data/processed narval:~/crispro_project/data/

# 3. Submit job (Terminal 2)
ssh narval 'cd ~/crispro_project && sbatch train_narval.sh'
# Returns: Submitted batch job 123456

# 4. Monitor (Terminal 2, repeated)
ssh narval 'squeue -u $USER'
ssh narval 'tail -f ~/crispro_project/logs/phase1_123456.out'

# 5. Retrieve results (Terminal 2, when complete)
scp -r narval:~/crispro_project/checkpoints/phase1_* ./results/

# Terminal 1 remains open, keeping socket alive for 48 hours
```

---

## Recommended Schedule

- **0:00** - Start ssh narval session, approve MFA
- **0:10** - Deploy files and submit job
- **0:15** - Monitor job startup
- **1:00 - 24:00** - Job training progress
- **24:30** - Check completion or resubmit if needed
- **25:00** - Download results
- **48:00** - ControlMaster socket expires (re-authenticate if needed)

---

## Next Steps After Phase 1

1. **Evaluate results**: Analyze training_history.json
2. **Fine-tune hyperparameters** if needed
3. **Prepare Phase 2**: Transfer learning on larger dataset
4. **Submit Phase 2 job**: Modify train_narval.sh for Phase 2 configuration

---

## Quick Command Reference

```bash
# All in one: Deploy and submit
scp train_narval.sh train_phase1.py requirements.txt narval:~/crispro_project/ && \
ssh narval 'cd ~/crispro_project && sbatch train_narval.sh'

# Monitor job (replace 123456 with actual job ID)
watch -n 10 'ssh narval "squeue -j 123456 && nvidia-smi"'

# Real-time training logs
ssh narval 'tail -f ~/crispro_project/logs/phase1_*.out'

# Download everything after job completes
scp -r narval:~/crispro_project/checkpoints ./results/ && \
scp -r narval:~/crispro_project/logs ./results/
```

---

## Support & Documentation

- **Alliance Canada Docs**: https://docs.alliancecan.ca/wiki/Home
- **Narval-specific**: https://docs.alliancecan.ca/wiki/Narval
- **SLURM Guide**: https://docs.alliancecan.ca/wiki/Running_jobs
- **Multifactor Auth**: https://docs.alliancecan.ca/wiki/Multifactor_authentication

---

**Created**: 2026-02-17  
**Last Updated**: 2026-02-17  
**Status**: Ready for deployment
