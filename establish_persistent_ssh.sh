#!/bin/bash
################################################################################
# ESTABLISH PERSISTENT SSH CONNECTION TO NARVAL
#
# This script creates a persistent SSH connection that:
# - Only requires MFA authentication ONCE
# - Remains connected for 72 hours
# - Automatically keeps alive with keep-alive packets
# - Allows all subsequent commands to reuse the connection
#
# Run this ONCE at the start of your work session:
#   bash establish_persistent_ssh.sh
#
# Then use any of these commands (no additional MFA needed):
#   ssh narval "command here"
#   ssh narval  (interactive shell)
#   sftp narval
################################################################################

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  NARVAL PERSISTENT SSH CONNECTION SETUP                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will establish a persistent SSH connection to Narval."
echo "You will be prompted for MFA authentication ONCE."
echo "After that, all connections reuse the authenticated session."
echo ""

# Check if control socket already exists
CONTROL_SOCKET=~/.ssh/control-narval.computecanada.ca-22-daneshpajouh

if [ -S "$CONTROL_SOCKET" ]; then
    echo "✅ Persistent connection already active!"
    echo "   Socket: $CONTROL_SOCKET"
    ssh narval "echo 'Connection verified at:' && date"
    exit 0
fi

echo "🔐 Initiating persistent SSH connection to Narval..."
echo "   (You will see an MFA prompt below)"
echo ""

# Establish master connection with MFA
# Using -N to not execute command, just establish connection
# Using -M to start as master
ssh -M -N narval &
SSH_PID=$!

# Give it a moment to authenticate
sleep 2

# Check if connection succeeded
if [ -S "$CONTROL_SOCKET" ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "✅ PERSISTENT CONNECTION ESTABLISHED SUCCESSFULLY"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Connection Details:"
    echo "  Host:    narval.computecanada.ca"
    echo "  User:    daneshpajouh"
    echo "  Socket:  $CONTROL_SOCKET"
    echo "  Persist: 72 hours"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Your MFA authentication is COMPLETE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Now you can use Narval without additional MFA:"
    echo ""
    echo "  📌 Interactive shell:"
    echo "     ssh narval"
    echo ""
    echo "  📌 Run single command:"
    echo "     ssh narval 'cd ~/chromaguide_experiments && sbatch scripts/slurm_train_v2_deephf.sh'"
    echo ""
    echo "  📌 Copy files:"
    echo "     scp -r narval:~/chromaguide_experiments/results/* ."
    echo ""
    echo "  📌 Submit jobs directly:"
    echo "     ssh narval sbatch scripts/slurm_train_v2_deephf.sh"
    echo ""
    echo "The connection will remain active for 72 hours."
    echo "If it drops, just run this script again."
    echo ""
else
    echo "❌ Connection failed. Check your credentials and try again."
    exit 1
fi
