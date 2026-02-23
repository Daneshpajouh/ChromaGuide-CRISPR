#!/bin/bash
# Real-time monitoring script for V9 training targets

echo "=================================="
echo "V9 TRAINING MONITOR"
echo "=================================="
echo ""
echo "To continuously monitor, run:"
echo "  while true; do clear; bash MONITOR_V9.sh; sleep 30; done"
echo ""

# Check if processes are running
echo "STATUS:"
if pgrep -f "train_on_real_data_v9" > /dev/null; then
  echo "  ✅ Multimodal V9 training: RUNNING"
else
  echo "  ⚠️  Multimodal V9: NOT running"
fi

if pgrep -f "train_off_target_v9" > /dev/null; then
  echo "  ✅ Off-target V9 training: RUNNING"
else
  echo "  ⚠️  Off-target V9: NOT running"
fi

echo ""
echo "MULTIMODAL V9 (target: Rho ≥ 0.911):"
echo "---"
tail -20 logs/multimodal_v9.log 2>/dev/null | grep -E "MODEL|Training|E  [0-9]|Rho|RESULTS|Individual"

echo ""
echo "OFF-TARGET V9 (target: AUROC ≥ 0.99):"
echo "---"
tail -20 logs/off_target_v9.log 2>/dev/null | grep -E "MODELS|Model|Val AUROC|AUROC|Individual"

echo ""
echo "LATEST EVENTS:"
echo "---"
echo "Multimodal:"
tail -1 logs/multimodal_v9.log 2>/dev/null
echo ""
echo "Off-target:"
tail -1 logs/off_target_v9.log 2>/dev/null
