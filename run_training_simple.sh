#!/bin/bash
set -e

cd /Users/studio/Desktop/PhD/Proposal

echo "🚀 Starting V10 OFF-TARGET training..."
conda run -n cg_train python3 -u scripts/train_off_target_v10.py 2>&1 | tee logs/off_target_v10.log &
OFF_PID=$!

sleep 15

echo "🚀 Starting V10 MULTIMODAL training..."
conda run -n cg_train python3 -u scripts/train_on_real_data_v10.py 2>&1 | tee logs/multimodal_v10.log &
MULTI_PID=$!

echo "✅ Both training processes started in background"
echo "   Off-target PID: $OFF_PID"
echo "   Multimodal PID: $MULTI_PID"
echo ""
echo "Monitor with: tail -f logs/off_target_v10.log"
echo "             tail -f logs/multimodal_v10.log"

# Wait for both to complete
wait $OFF_PID
wait $MULTI_PID

echo "✅ Training complete!"
