#!/bin/bash
# Deploy and run monitoring on Narval cluster
# Usage: ./deploy_monitoring.sh [interval_seconds] [max_iterations]

set -e

NARVAL_USER="amird"
NARVAL_HOST="narval.alliancecan.ca"
INTERVAL=${1:-300}  # Default 5 minutes
MAX_ITER=${2:-0}    # 0 = infinite

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ChromaGuide Monitoring Deployment to Narval${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Step 1: Create remote monitoring directory
echo -e "\n${YELLOW}Step 1: Creating monitoring directory on Narval...${NC}"
ssh "${NARVAL_USER}@${NARVAL_HOST}" "mkdir -p /home/amird/monitoring && ls -ld /home/amird/monitoring"

# Step 2: Copy monitoring script to Narval
echo -e "\n${YELLOW}Step 2: Uploading monitoring script...${NC}"
scp scripts/monitor_narval_jobs.py "${NARVAL_USER}@${NARVAL_HOST}:/home/amird/monitoring/"
ssh "${NARVAL_USER}@${NARVAL_HOST}" "chmod +x /home/amird/monitoring/monitor_narval_jobs.py && ls -lh /home/amird/monitoring/"

# Step 3: Start monitoring in background
echo -e "\n${YELLOW}Step 3: Starting monitoring in background...${NC}"
ssh "${NARVAL_USER}@${NARVAL_HOST}" << 'REMOTE_CMD'
cd /home/amird/monitoring

# Create wrapper script that handles continuous monitoring
cat > run_monitoring.sh << 'EOF'
#!/bin/bash
echo "[$(date)] Starting continuous monitoring with interval: 300s" >> monitoring.log
python3 monitor_narval_jobs.py --interval 300 --iterations 0 >> monitoring.log 2>&1 &
MONITOR_PID=$!
echo "[$(date)] Monitoring started with PID: $MONITOR_PID" >> monitoring.log
echo $MONITOR_PID > monitoring.pid
EOF

chmod +x run_monitoring.sh

# Start monitoring
./run_monitoring.sh

# Verify it's running
sleep 2
if [ -f monitoring.pid ]; then
  PID=$(cat monitoring.pid)
  if ps -p $PID > /dev/null 2>&1; then
    echo "[$(date)] ✓ Monitoring process $PID is running" >> monitoring.log
  else
    echo "[$(date)] ✗ Monitoring process failed to start" >> monitoring.log
  fi
fi

REMOTE_CMD

# Step 4: Verify monitoring is running
echo -e "\n${YELLOW}Step 4: Verifying monitoring process...${NC}"
ssh "${NARVAL_USER}@${NARVAL_HOST}" "tail -5 /home/amird/monitoring/monitoring.log && echo '---' && ps aux | grep monitor_narval_jobs.py | grep -v grep || echo 'Checking...'"

# Step 5: Create status check script
echo -e "\n${YELLOW}Step 5: Creating quick status check script...${NC}"
ssh "${NARVAL_USER}@${NARVAL_HOST}" << 'STATUS_SCRIPT'
cat > /home/amird/check_monitoring.sh << 'EOF'
#!/bin/bash
echo "Monitoring Status Check - $(date)"
echo "════════════════════════════════════"

# Check if monitoring process running
if [ -f /home/amird/monitoring/monitoring.pid ]; then
  PID=$(cat /home/amird/monitoring/monitoring.pid)
  if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Monitoring process running (PID: $PID)"
  else
    echo "✗ Monitoring process stopped"
  fi
else
  echo "? No monitoring PID file found"
fi

echo ""
echo "Recent monitoring log entries:"
tail -10 /home/amird/monitoring/monitoring.log

echo ""
echo "Current job status:"
squeue -u amird --format='%i %T %M' 2>/dev/null | head -7
EOF

chmod +x /home/amird/check_monitoring.sh
echo "Status check script created"
STATUS_SCRIPT

# Step 6: Create local monitoring check command
echo -e "\n${YELLOW}Step 6: Creating local monitoring check command...${NC}"
cat > /tmp/check_narval_monitoring.sh << 'LOCAL_CHECK'
#!/bin/bash
echo "Checking Narval monitoring status..."
ssh amird@narval.alliancecan.ca "bash /home/amird/check_monitoring.sh"
LOCAL_CHECK
chmod +x /tmp/check_narval_monitoring.sh

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Monitoring deployment COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo -e "\n${BLUE}Next steps:${NC}"
echo "1. Check monitoring status anytime:"
echo "   ssh narval 'bash /home/amird/check_monitoring.sh'"
echo ""
echo "2. View live monitoring logs:"
echo "   ssh narval 'tail -f /home/amird/monitoring/monitoring.log'"
echo ""
echo "3. Stop monitoring if needed:"
echo "   ssh narval 'kill \$(cat /home/amird/monitoring/monitoring.pid)'"
echo ""
echo "4. Restart monitoring:"
echo "   ssh narval 'cd /home/amird/monitoring && ./run_monitoring.sh'"
echo ""
echo "5. Download monitoring reports locally:"
echo "   scp -r narval:/home/amird/monitoring/narval_monitoring.log ./"
echo ""

echo -e "${YELLOW}Monitoring is now running on Narval and will:${NC}"
echo "  • Check job status every 5 minutes"
echo "  • Track GPU utilization"
echo "  • Monitor data download progress"
echo "  • Download intermediate results automatically"
echo "  • Log everything to narval_monitoring.log"
echo ""
echo "Your training jobs are being monitored 24/7! 🎯"
