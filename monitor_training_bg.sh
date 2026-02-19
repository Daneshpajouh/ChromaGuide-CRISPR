#!/bin/bash
################################################################################
# Background Job Monitor for ChromaGuide Training
#
# Monitors Narval job 56715343 continuously in the background
# Logs status and output to: training_monitor_JOBID.log
#
# Start:  bash monitor_training_bg.sh start 56715455
# Stop:   bash monitor_training_bg.sh stop 56715455
# Status: bash monitor_training_bg.sh status 56715455
# View:   tail -f training_monitor_56715455.log
################################################################################

COMMAND=$1
JOB_ID=${2:-56715455}
LOG_FILE="training_monitor_${JOB_ID}.log"
PID_FILE="/tmp/chromaguide_monitor_${JOB_ID}.pid"
CHECK_INTERVAL=60  # Check every 60 seconds

################################################################################
# Start Background Monitor
################################################################################
start_monitor() {
    if [ -f "$PID_FILE" ]; then
        echo "❌ Monitor already running (PID: $(cat $PID_FILE))"
        exit 1
    fi

    echo "🔍 Starting background monitor for job $JOB_ID..."
    echo "📝 Logs: $LOG_FILE"
    echo ""

    # Start background process
    (
        # Redirect output to log file
        exec >> "$LOG_FILE" 2>&1

        echo "╔════════════════════════════════════════════════════════════════════╗"
        echo "║ ChromaGuide V2 Training - Background Monitor Started              ║"
        echo "║ Job ID: $JOB_ID                                                   ║"
        echo "║ Start Time: $(date)                                              ║"
        echo "╚════════════════════════════════════════════════════════════════════╝"
        echo ""

        ITERATION=0
        while true; do
            ITERATION=$((ITERATION + 1))
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "Check #$ITERATION ($TIMESTAMP)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # Get job status
            echo ""
            echo "Job Status:"
            ssh narval "squeue -j $JOB_ID --format='%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R'" 2>/dev/null || echo "  (SSH connection issue)"

            # Get job accounting info
            echo ""
            echo "Job Details:"
            ssh narval "sacct -j $JOB_ID --format=jobid,jobname,state,elapsed,totalcpu,maxrss 2>/dev/null | head -3" || echo "  (Not yet available)"

            # Check if log file exists and show tail
            echo ""
            echo "Recent Log Output:"
            REMOTE_LOG="~/chromaguide_experiments/slurm_logs/slurm-${JOB_ID}.out"
            LOG_EXISTS=$(ssh narval "[ -f $REMOTE_LOG ] && echo 1 || echo 0" 2>/dev/null)

            if [ "$LOG_EXISTS" = "1" ]; then
                echo "  (Last 15 lines of slurm output:)"
                ssh narval "tail -15 $REMOTE_LOG" 2>/dev/null | sed 's/^/    /'
            else
                echo "  (Waiting for job to start...)"
            fi

            # Check disk usage if job is running
            JOB_STATE=$(ssh narval "squeue -j $JOB_ID -h --format=%T" 2>/dev/null)
            if [ "$JOB_STATE" = "RUNNING" ]; then
                echo ""
                echo "Disk Usage:"
                ssh narval "du -sh ~/chromaguide_experiments/{checkpoints,results} 2>/dev/null" | sed 's/^/    /'
            fi

            echo ""
            echo "Next check in ${CHECK_INTERVAL}s... ($(date '+%H:%M:%S + '$CHECK_INTERVAL's'))"
            echo ""

            sleep $CHECK_INTERVAL
        done
    ) &

    BG_PID=$!
    echo $BG_PID > "$PID_FILE"
    echo "✅ Monitor started (PID: $BG_PID)"
    echo "   Watch logs: tail -f $LOG_FILE"
    echo "   Stop monitor: bash monitor_training_bg.sh stop"
}

################################################################################
# Stop Background Monitor
################################################################################
stop_monitor() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Monitor not running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping monitor (PID: $PID)..."
    kill $PID 2>/dev/null
    rm -f "$PID_FILE"

    echo "✅ Monitor stopped"
    echo "Final log saved to: $LOG_FILE"
}

################################################################################
# Status Check
################################################################################
status_monitor() {
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ Monitor not running"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Monitor is running (PID: $PID)"
        echo "   Log file: $LOG_FILE"
        echo "   Lines in log: $(wc -l < $LOG_FILE)"
        echo ""
        echo "Recent entries:"
        tail -5 "$LOG_FILE" | sed 's/^/   /'
    else
        echo "❌ Monitor process not found (PID was: $PID)"
        rm -f "$PID_FILE"
    fi
}

################################################################################
# View Logs
################################################################################
view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "❌ No log file found: $LOG_FILE"
        exit 1
    fi

    echo "📝 Tailing log file (Ctrl+C to stop)..."
    echo ""
    tail -f "$LOG_FILE"
}

################################################################################
# Main
################################################################################
case "${1:-start}" in
    start)
        start_monitor
        ;;
    stop)
        stop_monitor
        ;;
    status)
        status_monitor
        ;;
    logs|view)
        view_logs
        ;;
    *)
        cat << EOF
Usage: bash monitor_training_bg.sh [COMMAND]

Commands:
  start              Start background monitoring (default)
  stop               Stop background monitoring
  status             Show monitor status
  logs/view          View live log output (tail -f)

Examples:
  bash monitor_training_bg.sh start
  bash monitor_training_bg.sh status
  bash monitor_training_bg.sh logs
  bash monitor_training_bg.sh stop

Job ID: $JOB_ID
Log file: $LOG_FILE
EOF
        exit 1
        ;;
esac
