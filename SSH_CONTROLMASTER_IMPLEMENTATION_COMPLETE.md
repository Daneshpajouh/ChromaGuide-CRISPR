# SSH ControlMaster Implementation - Complete Setup

## Status: ✅ READY FOR FIRST AUTHENTICATION

The SSH ControlMaster infrastructure is now fully configured and ready for use. All you need to do is authenticate once per cluster, and subsequent commands will run without requiring MFA.

---

## Quick Start

### Step 1: Authenticate to Cluster (One-time per persistence window)

```bash
ssh nibi
# OR
ssh hulk
```

When prompted, complete the Duo MFA authentication:
- Check your iPhone for Duo push notification
- Approve the login request
- OR enter your Duo passcode

This creates a persistent socket for 48-72 hours.

### Step 2: Use Cached Connection (No MFA required after Step 1)

After initial authentication, all commands work instantly without re-authentication:

```bash
# Run a quick command
ssh nibi 'hostname'

# Run a diagnostic
ssh nibi 'bash diagnose_h100_cluster.sh'

# Interactive shell (also cached)
ssh nibi
```

---

## Configuration Details

### Configured Clusters

| Cluster | Hostname | Persistence | Purpose |
|---------|----------|-------------|---------|
| **nibi** | nibi.alliancecan.ca | 48h | GPU testing (H100) |
| **fir** | fir.alliancecan.ca | 72h | Primary compute |
| **narval** | narval.alliancecan.ca | 48h | Backup GPU |
| **hulk** | hulk.alliancecan.ca | 10h | Development |

### SSH Socket Directory

```
~/.ssh/controlmasters/
├── amird@nibi.alliancecan.ca:22
├── amird@fir.alliancecan.ca:22
├── amird@narval.alliancecan.ca:22
└── amird@hulk.alliancecan.ca:22
```

Each socket is created after first successful authentication.

### SSH Config Settings Applied

All clusters configured with:
- **ControlMaster auto** - Automatically use/create master socket
- **ControlPath ~/.ssh/controlmasters/%r@%h:%p** - Socket location
- **StrictHostKeyChecking accept-new** - Trust new hosts automatically
- **ServerAliveInterval 120** - Keep connection alive (2-min intervals)
- **ConnectTimeout 30** - 30-second timeout for initial connection
- **Compression yes** - Compress data for faster transfer

---

## Workflow Examples

### Example 1: Monitor Phase 1 Training on Cluster

**First terminal** (authenticate once):
```bash
ssh nibi
# [Enter MFA]
# Now you're in an interactive shell on nibi
```

**Other terminals** (no MFA needed):
```bash
# Check training progress
ssh nibi 'tail -f ~/training_logs/phase1_metrics.txt'

# Check GPU memory
ssh nibi 'nvidia-smi'

# Run diagnostics
ssh nibi 'bash diagnose_h100_cluster.sh'
```

### Example 2: Run Multiple Commands Without Interactive Shell

```bash
# All of these work instantly after first authentication
ssh nibi 'nvidia-smi'
ssh nibi 'hostname'
ssh nibi 'python3 check_dataset_status.py'
ssh nibi 'sbatch train_phase1.sh'
```

### Example 3: Transfer Files via Cached Connection

```bash
# After initial authentication, use scp instantly
scp local_config.yaml nibi:~/config/
scp nibi:~/results/metrics.csv ./local_results/
```

---

## Verifying ControlMaster is Working

### Check 1: Socket Created

After first authentication:
```bash
ls ~/.ssh/controlmasters/
# Should show: amird@nibi.alliancecan.ca:22
```

### Check 2: Fast Subsequent Connections

```bash
time ssh nibi 'echo "Connected"'
# Should take <1 second (much faster than MFA each time)
```

### Check 3: Max Control Connections

```bash
ssh -O check nibi
# Shows control master status and number of open connections
```

---

## Duration of Persistence

Once authenticated, subsequent commands work without MFA for:
- **nibi**: 48 hours
- **fir**: 72 hours  
- **narval**: 48 hours
- **hulk**: 10 hours

After persistence window expires, you'll need to authenticate once again.

---

## Next Steps

### Immediate Actions Required

1. **Authenticate to nibi (or any cluster)**
   ```bash
   ssh nibi
   # Complete MFA
   ```

2. **Verify socket created**
   ```bash
   ls ~/.ssh/controlmasters/
   ```

3. **Test cached connection**
   ```bash
   ssh nibi 'hostname'
   # Should work instantly without MFA
   ```

### Once SSH is Working

1. **Run H100 diagnostics**
   ```bash
   ssh nibi 'bash diagnose_h100_cluster.sh'
   ```

2. **Check dataset availability**
   ```bash
   ssh nibi 'bash check_sota_status.sh'
   ```

3. **Prepare Phase 1 training**
   ```bash
   # Download datasets locally
   python3 download_datasets.py --dataset deephf
   
   # Create training job script
   vim train_phase1.sh
   
   # Submit to cluster
   scp train_phase1.sh nibi:~/
   ssh nibi 'sbatch ~/train_phase1.sh'
   ```

---

## Troubleshooting

### Issue: "ssh: connect to host nibi.alliancecan.ca port 22: Connection timed out"

**Cause**: Network connectivity or cluster temporarily down

**Solution**:
```bash
# Test basic connectivity
ping -c 3 nibi.alliancecan.ca

# Try with increased timeout
ssh -o ConnectTimeout=60 nibi 'hostname'

# Check if cluster is up
ssh fir 'module list'  # Try different cluster
```

### Issue: MFA Prompt Doesn't Appear

**Cause**: SSH trying to use password authentication instead of key-based auth

**Solution**:
```bash
# Verify your SSH key exists
ls -la ~/.ssh/id_rsa

# Test with verbose output
ssh -vv nibi

# Explicitly use key
ssh -i ~/.ssh/id_rsa nibi
```

### Issue: Socket Exists But Still Asking for MFA

**Cause**: Socket may have expired or ControlMaster not working

**Solution**:
```bash
# Remove old socket
rm ~/.ssh/controlmasters/amird@nibi.alliancecan.ca:22

# Re-authenticate
ssh nibi
```

### Issue: "Permission denied (keyboard-interactive,hostbased)"

**Cause**: Normal behavior when MFA is required - this is expected

**Solution**:
```bash
# This is correct - respond to Duo MFA prompt
# After successful auth, socket is created
# Subsequent commands work without this prompt
```

---

## Documentation Files

- **[CONTROLMASTER_SETUP_GUIDE.md](CONTROLMASTER_SETUP_GUIDE.md)** - Comprehensive technical guide
- **[H100_SETUP_SUMMARY.md](H100_SETUP_SUMMARY.md)** - H100 cluster configuration
- **[SSH_CONTROLMASTER_IMPLEMENTATION_COMPLETE.md](SSH_CONTROLMASTER_IMPLEMENTATION_COMPLETE.md)** - This file

---

## Summary

✅ **What's Done:**
- SSH config updated for all clusters (nibi, fir, narval, hulk)
- ControlMaster settings optimized for each cluster
- Socket directory created (~/.ssh/controlmasters/)
- Comprehensive documentation provided

⏳ **What's Needed:**
- You: Run `ssh nibi` and complete ONE MFA authentication
- After that: All commands work instantly for 48-72 hours

🚀 **Result:**
- Unlimited automated SSH commands without additional MFA
- Enables Phase 1 training monitoring, diagnostics, job submission
- Persistence windows allow workflow execution across multiple sessions

---

## Quick Reference Commands

```bash
# First authentication (MFA required once)
ssh nibi

# Check socket created
ls ~/.ssh/controlmasters/

# Subsequent commands (instant, no MFA)
ssh nibi 'nvidia-smi'
ssh nibi 'bash diagnose_h100_cluster.sh'

# Interactive shell after initial auth
ssh nibi

# File transfer via cached connection
scp local_file nibi:~/
scp nibi:~/results/* ./local_results/

# Check ControlMaster status
ssh -O check nibi

# Force new connection (new MFA required)
ssh -O exit nibi
ssh nibi
```

---

**Status**: Ready for production use  
**Last Updated**: 2026-02-17  
**Git Commit**: b1870ca (CONTROLMASTER_SETUP_GUIDE.md added)
