#!/bin/bash
# Continuous monitoring script for off-target training
# Check every 5 minutes for 50 minutes

echo "Starting off-target monitoring (5-minute intervals)..."
echo "Expected training time: ~45-50 minutes for ~100 epochs"
echo ""

for i in {1..10}; do
    echo "=== CHECK $i (Time: $(date '+%H:%M:%S')) ==="

    # Get last epoch line
    LAST_LINE=$(tail -3 ~/chromaguide_experiments/slurm_logs/off_target_v4_enhanced_56835933.out 2>/dev/null | grep -E "Epoch|Training finished" | tail -1)

    if [[ -z "$LAST_LINE" ]]; then
        echo "⏳ Job starting or no output yet..."
    elif echo "$LAST_LINE" | grep -q "Training finished"; then
        echo "✅ TRAINING COMPLETED!"
        echo "$LAST_LINE"
        tail -20 ~/chromaguide_experiments/slurm_logs/off_target_v4_enhanced_56835933.out
        break
    else
        echo "$LAST_LINE"
        # Extract AUROC if available
        if echo "$LAST_LINE" | grep -q "AUROC:"; then
            AUROC=$(echo "$LAST_LINE" | grep -o "AUROC: [0-9.]*" | cut -d' ' -f2)
            echo "→ Current AUROC: $AUROC"
        fi
    fi

    echo ""

    if [ $i -lt 10 ]; then
        sleep 300  # 5 minutes
    fi
done

echo ""
echo "=== FINAL STATUS ===="
if grep -q "Training finished" ~/chromaguide_experiments/slurm_logs/off_target_v4_enhanced_56835933.out; then
    echo "✅ Off-target training completed successfully"
    tail -15 ~/chromaguide_experiments/slurm_logs/off_target_v4_enhanced_56835933.out
else
    echo "⏳ Still running... Final 10 lines:"
    tail -10 ~/chromaguide_experiments/slurm_logs/off_target_v4_enhanced_56835933.out
fi
