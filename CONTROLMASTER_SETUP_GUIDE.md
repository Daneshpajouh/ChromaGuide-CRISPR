#!/usr/bin/env bash
################################################################################
# SSH ControlMaster Setup Guide
# Purpose: Persistent connections with multi-factor authentication
# Date: February 17, 2026
################################################################################

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                     SSH ControlMaster Setup Complete                         ║
║                   Persistent Connection Sharing Guide                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ SETUP COMPLETE
═══════════════════════════════════════════════════════════════════════════════

Configuration file:  ~/.ssh/config
Socket directory:    ~/.ssh/controlmasters/
Permissions:         700 (user-only)
Status:              READY

═══════════════════════════════════════════════════════════════════════════════

📋 CONFIGURED CLUSTERS
═══════════════════════════════════════════════════════════════════════════════

The following clusters are configured with ControlMaster:

  ✓ FIR       (fir.alliancecan.ca)           - 72 hour persistence
  ✓ NIBI      (nibi.alliancecan.ca)          - 48 hour persistence  
  ✓ NARVAL    (narval.alliancecan.ca)        - 48 hour persistence
  ✓ RORQUAL   (rorqual.alliancecan.ca)       - Auto-enabled
  ✓ CEDAR     (cedar.alliancecan.ca)         - Auto-enabled
  ✓ GRAHAM    (graham.alliancecan.ca)        - Auto-enabled
  ✓ HULK      (hulk.alliancecan.ca)          - 10 hour persistence

═══════════════════════════════════════════════════════════════════════════════

🔐 HOW CONTROLMASTER WORKS
═══════════════════════════════════════════════════════════════════════════════

1. FIRST CONNECTION (with MFA)
   $ ssh nibi
   
   What happens:
   • First SSH connection is initiated
   • ControlMaster auto-creates a master socket
   • Duo MFA is prompted
   • You authenticate (push/passcode)
   • Connection is established and cached
   • Socket file created: ~/.ssh/controlmasters/amird@nibi.alliancecan.ca:22

2. SUBSEQUENT CONNECTIONS (within persistence window)
   $ ssh nibi 'ls -la'
   
   What happens:
   • SSH checks for existing master socket
   • Socket is found and connection reused
   • NO MFA prompt (socket already authenticated)
   • Command executes immediately
   • No additional authentication needed

3. PERSISTENCE DURATION
   FIR:     72 hours
   NIBI:    48 hours
   NARVAL:  48 hours
   HULK:    10 hours
   
   After persistence window expires, next SSH will require MFA again.

═══════════════════════════════════════════════════════════════════════════════

🚀 FIRST TIME: Establish Master Connection
═══════════════════════════════════════════════════════════════════════════════

To establish the initial authenticated connection (ONCE per persistence window):

OPTION A: Interactive Shell (Recommended for First Time)
  $ ssh nibi
  
  • Prompts for Duo authentication
  • You respond to push/passcode
  • Connection remains open
  • ControlMaster socket created

OPTION B: Single Command Connection
  $ ssh nibi 'echo "Establishing master connection"'
  
  • Same MFA prompt
  • Command executes after auth
  • Socket created
  • Connection closes after command

OPTION C: Just Create the Master (No Command)
  $ ssh -N nibi &
  
  • Connects and creates socket
  • Runs in background with -N flag
  • No command execution
  • Connection stays open

EXAMPLE - First Time Setup:
────────────────────────────
$ ssh nibi

(Your local machine connects to nibi.alliancecan.ca)

Multifactor authentication is now mandatory:
Duo two-factor login for amird

Enter a passcode or select one of the following options:
1. Duo Push to Amir's iPhone (iOS)

Passcode or option (1-1): 1
[Your phone gets a push notification]
[You approve on your phone]

(amird@nibi:~)$ 
[You're now connected to the login node]

(amird@nibi:~)$ exit
[Connection closes, but socket persists]

═══════════════════════════════════════════════════════════════════════════════

✨ AFTER THAT: Use Socket Without MFA
═══════════════════════════════════════════════════════════════════════════════

Once the master socket is established, ALL subsequent SSH commands work
WITHOUT requiring MFA authentication:

$ ssh nibi 'hostname'
nibi1

$ ssh nibi 'uptime'
 10:45:17 up 128 days, 3:12, 0 users, load average: 0.42, 0.38, 0.36

$ ssh nibi 'ls -la ~/projects'
total 48
drwxr-xr-x   8 amird  def-group   256 Feb 17 10:30 ./
drwxrwx---+ 20 amird  def-group   640 Feb 17 10:45 ../

$ scp nibi:~/data/file.txt ./file.txt
file.txt           100% 1234MB   45MB/s   00:27

No MFA prompts! No authentication required!

═══════════════════════════════════════════════════════════════════════════════

🔍 CHECK SOCKET STATUS
═══════════════════════════════════════════════════════════════════════════════

View active ControlMaster sockets:

$ ls -la ~/.ssh/controlmasters/
  total 0
  drwx------     3 studio  staff     96 Feb 17 10:45 ./
  drwx------    11 studio  staff    352 Feb 17 10:40 ../
  srwx------     1 studio  staff      0 Feb 17 10:45 amird@nibi.alliancecan.ca:22

Socket exists = Master connection is active
Socket removed = Master connection has expired

Test if socket is active:

$ ssh -O check nibi
Master running (pid=12345)

Test if socket is inactive:

$ ssh -O check nibi
Master not running

═══════════════════════════════════════════════════════════════════════════════

⏰ PERSISTENCE WINDOWS
═══════════════════════════════════════════════════════════════════════════════

FIR:
  ServerAliveInterval: 60 seconds
  ControlPersist: 72 hours
  Use for: Long-running large training jobs
  
NIBI:
  ServerAliveInterval: 60 seconds
  ControlPersist: 48 hours
  Use for: Multi-day training
  
NARVAL:
  ServerAliveInterval: 60 seconds
  ControlPersist: 48 hours
  Use for: Multi-day training

HULK:
  ServerAliveInterval: 120 seconds
  ControlPersist: 10 hours
  Use for: Interactive session work
  
═══════════════════════════════════════════════════════════════════════════════

🛠️ TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

PROBLEM: "Permission denied (keyboard-interactive,hostbased)"
SOLUTION:
  This is normal on first connection - it means MFA is needed.
  Provide the MFA response (Duo Push or passcode).

PROBLEM: "Control socket does not exist"
SOLUTION:
  First establish the master socket:
  $ ssh nibi  (authenticate with MFA)
  Then subsequent commands will use the socket.

PROBLEM: "Socket already open by another process"
SOLUTION:
  Sockets can only be used by one process at a time.
  Solution: Use -S flag for a different socket:
  $ ssh -S ~/.ssh/controlmasters/new_socket nibi

PROBLEM: "Connection reset by peer"
SOLUTION:
  The socket may have expired.
  Establish a new master connection:
  $ ssh nibi  (authenticate with MFA)

═══════════════════════════════════════════════════════════════════════════════

📝 WORKFLOW EXAMPLE: Phase 1 Training
═══════════════════════════════════════════════════════════════════════════════

Day 1 - Setup:
─────────────
$ ssh nibi
[Authenticate with Duo MFA]
(amird@nibi:~)$ cd ~/projects/CRISPRO-MAMBA-X
(amird@nibi:~)$ python3 src/train.py --dataset deephf &
(amird@nibi:~)$ exit
[Master socket remains active for 48 hours]


Days 1-48 - Check Progress (NO MFA):
───────────────────────────────────
$ ssh nibi 'cat training.log | tail -20'
[Output appears immediately - no MFA!]

$ ssh nibi 'ps aux | grep train'
[Check if training process is running]

$ ssh nibi 'df -h'
[Check disk usage]

$ scp nibi:checkpoints/latest.pt ./latest.pt
[Copy checkpoint - no MFA needed!]


Day 49+ - Reconnect (MFA Again):
────────────────────────────────
$ ssh nibi
[Now MFA is required - socket expired]
[Authenticate with Duo MFA]
[New 48-hour window starts]


═══════════════════════════════════════════════════════════════════════════════

💡 BEST PRACTICES
═══════════════════════════════════════════════════════════════════════════════

✓ Establish master during work hours when you can respond to MFA
✓ Establish master when starting a new job or long-running task
✓ Check socket status before long SSH operations
✓ Increase ControlPersist for longer experiments (up to 168h = 1 week)
✓ Use -N flag for background master: ssh -N nibi &
✓ Monitor ~/.ssh/controlmasters/ for active sockets

═══════════════════════════════════════════════════════════════════════════════

🔗 NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

1. Establish master connection to nibi:
   $ ssh nibi
   [Authenticate with Duo MFA]
   [Type exit when done]

2. Verify socket created:
   $ ls ~/.ssh/controlmasters/

3. Test without MFA:
   $ ssh nibi 'hostname'
   [Should work without authentication prompt]

4. Use for cluster diagnostics:
   $ ssh nibi 'bash diagnose_h100_cluster.sh'

5. Monitor Phase 1 training when ready:
   $ ssh nibi 'tail -f training.log'

═══════════════════════════════════════════════════════════════════════════════

SSH Config File: ~/.ssh/config
ControlMasters Directory: ~/.ssh/controlmasters/
Status: ✅ READY TO USE

Next: Establish first master connection with "ssh nibi"

═══════════════════════════════════════════════════════════════════════════════

EOF
