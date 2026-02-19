#!/bin/bash

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            BACKGROUND MONITORING - QUICK REFERENCE                           ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

🎯 STATUS RIGHT NOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Job ID:        56715343
  Status:        ✅ RUNNING (on node ng31103)
  Started:       ~20 seconds ago
  Log File:      training_monitor_56715343.log
  Monitor PID:   (check with: ps aux | grep monitor)
  
Expected Duration:  12 hours


📋 COMMANDS YOU CAN RUN NOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# View live monitoring log (auto-updates every 60 seconds)
tail -f training_monitor_56715343.log

# Or more colorful:
bash monitor_training_bg.sh logs

# Check monitor status any time
bash monitor_training_bg.sh status

# Quick job status check via SSH (no MFA needed)
ssh narval "squeue -j 56715343"

# View training logs on Narval (once job outputs them)
ssh narval "tail -f ~/chromaguide_experiments/slurm_logs/slurm-56715343.out"

# Stop background monitor (when done)
bash monitor_training_bg.sh stop


🚀 MONITORING FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The background monitor automatically:
  ✓ Checks job status every 60 seconds
  ✓ Captures job accounting details
  ✓ Tails the latest training logs
  ✓ Tracks disk usage during training
  ✓ Saves everything to: training_monitor_56715343.log
  ✓ Runs completely in background (no terminal needed)


⏱️ TIMELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  T+0 min      Job submitted (done)
  T+20s        Job started (🟢 RUNNING)
  T+6h         Midpoint - check logs for progress
  T+12h        Training completes
  T+18h        Results + SOTA comparison ready


📊 YOU CAN KEEP WORKING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The monitor runs continuously in background. You can:
  • Close this terminal without stopping the monitor
  • Come back anytime to check status
  • Check logs remotely: tail -f training_monitor_56715343.log
  • SSH into Narval anytime (persistent connection active 72h)
  • Start writing/editing other files


💡 RECOMMENDED NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Leave the monitor running (it uses minimal resources)
2. Periodically check: tail -f training_monitor_56715343.log
3. In ~12h, check for completed results
4. Extract results and update GitHub


═════════════════════════════════════════════════════════════════════════════════

           Everything is ready. Job is running. You're all set! 🎉

═════════════════════════════════════════════════════════════════════════════════
EOF
