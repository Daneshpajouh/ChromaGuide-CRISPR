#!/bin/bash
# Monitor ChromaGuide jobs on this cluster
# Usage: bash monitor_jobs.sh

echo "==========================================="
echo "ChromaGuide Job Monitor - $(hostname)"
echo "Time: $(date)"
echo "==========================================="

# Show all running/pending jobs
echo ""
echo "=== SLURM Queue ==="
squeue -u $USER -o "%.10i %.30j %.8T %.10M %.6D %R" 2>/dev/null || echo "squeue not available"

# Count jobs by status
echo ""
echo "=== Job Status Summary ==="
RUNNING=$(squeue -u $USER -t RUNNING -h 2>/dev/null | wc -l)
PENDING=$(squeue -u $USER -t PENDING -h 2>/dev/null | wc -l)
echo "Running: $RUNNING"
echo "Pending: $PENDING"

# Check completed results
echo ""
echo "=== Completed Results ==="
cd ~/scratch/chromaguide_v2 2>/dev/null || exit 1

TOTAL=0
COMPLETED=0
FAILED=0

for dir in results/*/; do
    if [ -d "$dir" ]; then
        TOTAL=$((TOTAL + 1))
        if [ -f "${dir}results.json" ]; then
            COMPLETED=$((COMPLETED + 1))
            # Extract key metrics
            BACKBONE=$(python3 -c "import json; r=json.load(open('${dir}results.json')); print(r['backbone'])" 2>/dev/null)
            SPLIT=$(python3 -c "import json; r=json.load(open('${dir}results.json')); print(r['split'])" 2>/dev/null)
            SEED=$(python3 -c "import json; r=json.load(open('${dir}results.json')); print(r['seed'])" 2>/dev/null)
            SPEARMAN=$(python3 -c "import json; r=json.load(open('${dir}results.json')); print(f\"{r['test_metrics']['spearman']:.4f}\")" 2>/dev/null)
            echo "  ✓ ${BACKBONE} | Split ${SPLIT} | Seed ${SEED} | ρ=${SPEARMAN}"
        elif ls ${dir}slurm-*.err 2>/dev/null | head -1 | xargs -I{} grep -l "Error\|FAILED\|OOM" {} > /dev/null 2>&1; then
            FAILED=$((FAILED + 1))
            echo "  ✗ ${dir} (FAILED)"
        fi
    fi
done

echo ""
echo "Total experiments: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "In progress/pending: $((TOTAL - COMPLETED - FAILED))"

# GPU usage
echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available (login node)"

echo ""
echo "=== Disk Usage ==="
du -sh ~/scratch/chromaguide_v2/results/ 2>/dev/null
du -sh ~/scratch/chromaguide_v2/data/ 2>/dev/null
echo "Scratch quota:"
diskusage_report 2>/dev/null | head -5 || echo "diskusage_report not available"
