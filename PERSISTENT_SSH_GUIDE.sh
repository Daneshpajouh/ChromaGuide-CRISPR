#!/bin/bash

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           PERSISTENT SSH CONNECTION - COMPLETE SETUP GUIDE                 ║
║                                                                              ║
║    One-time MFA authentication that stays connected for 72 hours           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ QUICK START (Copy & Paste These Commands)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# STEP 1: Establish persistent connection (ONCE at start)
bash establish_persistent_ssh.sh

# You'll see an MFA prompt - authenticate ONCE
# After successful authentication, you never type MFA again for 72 hours


# STEP 2: Use Narval without additional authentication (as many times as you want)

# Option A: Interactive shell
ssh narval

# Option B: Submit DeepHF job
ssh narval 'cd ~/chromaguide_experiments && sbatch scripts/slurm_train_v2_deephf.sh'

# Option C: Submit both jobs
ssh narval 'cd ~/chromaguide_experiments && sbatch scripts/slurm_train_v2_*.sh'

# Option D: Check job status
ssh narval squeue -u daneshpajouh

# Option E: Copy results back
scp -r narval:~/chromaguide_experiments/results/* .


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 HOW IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SSH ControlMaster allows connection multiplexing:

  1. First time: `ssh narval` creates MASTER connection
     ↓
     First SSH session STAYS OPEN in background
     ↓
     You authenticate with MFA ONCE

  2. Subsequent times: All SSH commands reuse MASTER connection
     ↓
     No additional authentication needed
     ↓
     Instant connection (already authenticated)

  3. Connection persistence: 72 hours (3 days)
     ↓
     Even if you close all SSH windows
     ↓
     Master socket remains active in ~/.ssh/control-*
     ↓
     New connections instantly reuse the authenticated session


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 DETAILED SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your SSH config has been updated with:

  Host narval
      HostName narval.computecanada.ca
      User daneshpajouh
      ControlMaster auto        ← Enable connection multiplexing
      ControlPath ~/.ssh/control-%h-%p-%r  ← Where to store master socket
      ControlPersist 72h        ← Keep connection alive for 72 hours
      ServerAliveInterval 60    ← Keep-alive ping every 60 seconds
      ServerAliveCountMax 1440  ← Allow 1440 lost pings (24 hours)

This means:
  ✓ First SSH to 'narval' initiates master connection (requires MFA)
  ✓ All subsequent SSH to 'narval' reuse the master (instant, no MFA)
  ✓ Connection persists even if you close all terminal windows
  ✓ Automatic keep-alive packets prevent timeout


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ EXECUTE NOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Run the setup script:

   bash establish_persistent_ssh.sh

   This will:
   ✓ Create master SSH connection to Narval
   ✓ Prompt for MFA authentication (ONLY ONCE)
   ✓ Keep connection open in background
   ✓ Verify connection is working

2. When complete, you'll see:

   ✅ PERSISTENT CONNECTION ESTABLISHED SUCCESSFULLY
   
   Connection Details:
     Host:    narval.computecanada.ca
     User:    daneshpajouh
     Socket:  ~/.ssh/control-narval.computecanada.ca-22-daneshpajouh
     Persist: 72 hours


3. Then use any of these commands (NO MORE MFA PROMPTS):

   # Interactive shell
   ssh narval
   
   # Submit jobs
   ssh narval 'sbatch scripts/slurm_train_v2_deephf.sh'
   
   # Check status
   ssh narval 'squeue -u daneshpajouh'


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❓ TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Script fails when I run it
A: Check SSH key is configured correctly:
   ssh-add -l  (should show your key)
   If not:  ssh-add ~/.ssh/id_rsa

Q: MFA still appears after setup
A: Master connection may have been closed:
   ls -la ~/.ssh/control-*  (check if socket exists)
   If missing: run establish_persistent_ssh.sh again

Q: "Permission denied (publickey,gssapi-keyex)"
A: Check SSH key:
   ssh-keyscan narval.computecanada.ca >> ~/.ssh/known_hosts
   ssh-add -K ~/.ssh/id_rsa  (on macOS)

Q: Changes to ~/.ssh/config not working
A: Kill existing connections and restart:
   ssh -O exit narval
   bash establish_persistent_ssh.sh

Q: Connection drops after a few hours
A: The 72-hour persistence may have a server-side limit:
   Just re-run setup script to re-authenticate:
   bash establish_persistent_ssh.sh


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 FOR YOUR TRAINING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

With persistent SSH, you can now:

# 1. Submit both jobs simultaneously (no MFA re-prompt)
ssh narval << 'BATCH'
cd ~/chromaguide_experiments
sbatch scripts/slurm_train_v2_deephf.sh
sbatch scripts/slurm_train_v2_crispron.sh
echo "Jobs submitted successfully"
BATCH

# 2. Monitor in real-time without re-authenticating
watch -n 5 'ssh narval squeue -u daneshpajouh'

# 3. Pull results every hour without re-authenticating
while true; do
  scp -r narval:~/chromaguide_experiments/results/* local_results/
  sleep 3600
done

# 4. Everything from this file works seamlessly
execute_chromaguide_v2_automated.sh  # No MFA interruptions


═══════════════════════════════════════════════════════════════════════════════

                         Ready? Execute this now:

                      bash establish_persistent_ssh.sh

═══════════════════════════════════════════════════════════════════════════════
EOF
