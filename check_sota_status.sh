#!/bin/bash
# Check SOTA Benchmark Status

echo "=== SOTA QUAD CROWN MONITOR ==="
echo "Time: $(date)"

# 1. Check Queue
echo -e "\n[1] Queue Status (Job 5966974):"
ssh nibi "squeue -u amird"

# 2. Check Logs
echo -e "\n[2] Latest Logs:"
ssh nibi "ls -lth ~/projects/def-kwiese/amird/Proposal/logs/quad_crown_5966974*.out 2>/dev/null | head -5"
ssh nibi "tail -n 3 ~/projects/def-kwiese/amird/Proposal/logs/quad_crown_5966974*.out 2>/dev/null"

# 3. Check Results
echo -e "\n[3] Results So Far:"
ssh nibi "cat ~/projects/def-kwiese/amird/Proposal/task1_ontarget_result.json 2>/dev/null" && echo ""
ssh nibi "cat ~/projects/def-kwiese/amird/Proposal/task2_offtarget_result.json 2>/dev/null" && echo ""
ssh nibi "cat ~/projects/def-kwiese/amird/Proposal/task3_generative_result.json 2>/dev/null" && echo ""
ssh nibi "cat ~/projects/def-kwiese/amird/Proposal/task4_transfer_result.json 2>/dev/null" && echo ""

echo -e "\n=== END REPORT ==="
