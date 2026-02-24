#!/bin/bash
# SLURM submission for pre-caching DNABERT-2 on FIR
#SBATCH --job-name=cache_dnabert
#SBATCH --account=def-kwiese
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/cache_dnabert_%j.out
#SBATCH --error=logs/cache_dnabert_%j.err

set -e

echo "=== PRE-CACHE DNABERT-2 FOR COMPUTE NODE ACCESS ==="
echo "Node: $(hostname -f)"
echo "Time: $(date)"
echo ""

# Initialize environment
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
fi

# Try loading modules if available
if command -v module >/dev/null 2>&1; then
    module load python/3.11.5 2>/dev/null || true
    module load cuda/12.2 2>/dev/null || true
fi

# Create virtualenv in home (shared filesystem)
VENV_PATH="$HOME/.cache/dnabert_venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtualenv at $VENV_PATH..."
    python -m venv "$VENV_PATH" --system-site-packages
fi

# Activate virtualenv
source "$VENV_PATH/bin/activate"

# Install/upgrade requirements
echo "Installing dependencies..."
pip install --upgrade pip -q 2>/dev/null || true
pip install transformers torch -q 2>/dev/null || true

# Download and cache DNABERT-2
echo ""
echo "Downloading DNABERT-2 from HuggingFace..."
python << 'PYEOF'
import os
import sys
from pathlib import Path

os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
cache_dir = os.environ['HF_HOME']
cache_hub = Path(cache_dir) / 'hub'

print(f"Cache directory: {cache_dir}")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(cache_hub, exist_ok=True)

try:
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    print("✓ Tokenizer cached")

    print("Loading model...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
    print("✓ Model cached")

    # Verify cache
    model_files = list(cache_hub.glob("**/models*DNABERT*/**"))
    print(f"\n✅ SUCCESS: DNABERT-2 cached for compute node access")
    print(f"   Cache location: {cache_dir}")
    print(f"   Files cached: {len(model_files)} items")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYEOF

RET=$?
deactivate

if [ $RET -eq 0 ]; then
    echo ""
    echo "✅ Pre-caching complete - model ready for training jobs"
    ls -lh ~/.cache/huggingface/hub 2>/dev/null | head -10 || echo "Cache directory contents"
    exit 0
else
    echo "❌ Pre-caching failed"
    exit 1
fi
