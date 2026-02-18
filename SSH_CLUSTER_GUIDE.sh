#!/bin/bash
###############################################################################
# SSH Connection Guide for Nibi H100 Cluster
# Purpose: Connect to the cluster and run diagnostic scripts
###############################################################################

# ============================================================================
# STEP 1: SSH to Nibi Login Node (Run on your local machine)
# ============================================================================

# Option A: Direct SSH with hostname
ssh nibi

# Option B: Full SSH command with explicit details
ssh -v amird@nibi.computecanada.ca

# Option C: If using specific key or port
ssh -i ~/.ssh/your_key -p 22 amird@nibi.computecanada.ca

# Note: Replace 'amird' with your actual username if different

# ============================================================================
# STEP 2: Once Connected to Nibi (Run on cluster login node)
# ============================================================================

# Navigate to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Verify you're in the right location
pwd
# Output should be: /home/<username>/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# ============================================================================
# STEP 3: Run the Diagnostic Script (Run on cluster)
# ============================================================================

# Option A: If you want to run the script from your local laptop
# First, copy the diagnostic script to the cluster:
scp diagnose_h100_cluster.sh nibi:~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/

# Then SSH and run it:
ssh nibi
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
bash diagnose_h100_cluster.sh

# Option B: Run diagnostic commands directly (one by one for debugging)
# ============================================================================

# Check 1: Verify project structure
ls -lah ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/

# Check 2: Verify script exists and is executable
ls -l ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/verify_real_h100.sh

# Check 3: Load modules
module purge
module load StdEnv/2023
module load python/3.10
module load scipy-stack
module load cuda/12.2

# Verify modules loaded
module list

# Check 4: Verify environment
ls -lah ~/projects/def-kwiese/amird/mamba_h100/

# Check 5: Test environment activation
source ~/projects/def-kwiese/amird/mamba_h100/bin/activate

# Check 6: Test Python
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check 7: Try to import training module
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
python -c "from src import train; print('✓ Import successful')"

# ============================================================================
# STEP 4: Check Job Status and Logs (Run on cluster)
# ============================================================================

# View recent jobs
squeue -u $(whoami) -l

# Check specific job
squeue -j 5860955

# View SLURM output for a job (replace JOBID)
cat logs/verify_real_h100_JOBID.out

# View recent SLURM errors
cd logs/
ls -lat | head -20  # View most recently modified files

# ============================================================================
# STEP 5: If Something Fails, Debug Further
# ============================================================================

# Test GPU access without SLURM (on login node - may not work)
nvidia-smi

# Test script execution manually (dry-run without SLURM)
bash -x ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/verify_real_h100.sh 2>&1 | head -50

# Check if there are any syntax errors in the script
bash -n ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X/verify_real_h100.sh

# ============================================================================
# STEP 6: Re-submit Job After Fixing Issues
# ============================================================================

# Navigate to project directory
cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X

# Make script executable again (if needed)
chmod +x verify_real_h100.sh

# Submit SLURM job
sbatch verify_real_h100.sh

# Watch job status in real-time
watch -n 5 'squeue -u $(whoami)'

# Or use this command to maintain connection and monitor
while true; do 
  clear
  echo "=== $(date) ==="
  squeue -u $(whoami) --format="%.18i %.50j %.8T %.10M %.3C"
  sleep 5
done

# ============================================================================
# HELPFUL SLURM COMMANDS
# ============================================================================

# Cancel a job
scancel 5860955

# Get job information
sinfo

# Check GPU partition specifically
sinfo -p gpu

# Get detailed job information
sacct -j 5860955 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed

# View job submission line
scontrol show job 5860955

# Stream job output in real-time (if available)
tail -f logs/verify_real_h100_*.out

# ============================================================================
# EXPECTED OUTPUT FROM DIAGNOSTIC SCRIPT
# ============================================================================

# You should see:
# ✅ Project directory exists
# ✅ verify_real_h100.sh exists
# ✅ src/train.py exists
# ✅ logs/ exists
# ✅ checkpoints/ exists
# ✅ StdEnv/2023 loaded
# ✅ python/3.10 loaded
# ✅ cuda/12.2 loaded
# ✅ Environment found at /home/.../mamba_h100
# ✅ Python execution works

# If you see ❌ errors, those need to be fixed before submitting training job

# ============================================================================
# COMMON ISSUES AND FIXES
# ============================================================================

# Issue: "command not found: module"
# Fix: The module command is usually available after loading StdEnv
#      Try: source /etc/profile.d/modules.sh

# Issue: "Environment not found at ~/projects/def-kwiese/amird/mamba_h100"
# Fix: Create the environment:
#      mamba create -p ~/projects/def-kwiese/amird/mamba_h100 python=3.10
#      mamba activate ~/projects/def-kwiese/amird/mamba_h100
#      pip install torch transformers

# Issue: "Python: command not found"
# Fix: Ensure python/3.10 module is loaded:
#      module load python/3.10

# Issue: "SLURM job fails immediately (1 second runtime)"
# Fix: Check these in order:
#      1. Module loading failures (try manually)
#      2. Environment activation failures
#      3. Script syntax errors (bash -n script.sh)
#      4. Directory permission issues (chmod 755 logs/)
#      5. SLURM output redirection issues

# ============================================================================
# MAKING SSH EASIER (Optional - on your local machine)
# ============================================================================

# Add to ~/.ssh/config file:
# ----
# Host nibi
#     HostName nibi.computecanada.ca
#     User amird
#     IdentityFile ~/.ssh/your_key_if_using_one
#     ServerAliveInterval 120
#     ServerAliveCountMax 5
# ----

# Then you can just run:
ssh nibi

# ============================================================================
