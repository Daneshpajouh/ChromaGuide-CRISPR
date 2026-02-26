#!/bin/bash
# ============================================================
# ChromaGuide v2: Multi-Cluster Setup Script
#
# Supported clusters:
#   - Narval  (A100-40GB × 4/node, AMD EPYC, Calcul Québec)
#   - Béluga  (V100-16GB × 4/node, Intel Skylake) [→ Rorqual replacement]
#   - Fir     (A100-80GB, SFU, installation in progress)
#   - Killarney (H100-80GB × 8/node OR L40S-48GB × 4/node, UofT/Vector)
#
# Usage: bash scripts_v2/setup_cluster.sh [cluster_name]
#   e.g.: bash scripts_v2/setup_cluster.sh narval
# ============================================================

set -e

CLUSTER="${1:-auto}"

# Auto-detect cluster from hostname
if [ "$CLUSTER" = "auto" ]; then
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    case "$HOSTNAME" in
        *narval*)    CLUSTER="narval" ;;
        *beluga*)    CLUSTER="beluga" ;;
        *rorqual*)   CLUSTER="rorqual" ;;
        *fir*)       CLUSTER="fir" ;;
        *killarney*) CLUSTER="killarney" ;;
        *nibi*)      CLUSTER="nibi" ;;
        *)           CLUSTER="generic"
                     echo "WARNING: Unknown cluster ($HOSTNAME), using generic setup" ;;
    esac
fi

echo "============================================"
echo "ChromaGuide v2: Cluster Setup"
echo "Cluster: $CLUSTER"
echo "User: $(whoami)"
echo "Date: $(date)"
echo "============================================"

# 1. Load modules (cluster-specific)
echo "--- Loading modules ---"
case "$CLUSTER" in
    narval)
        module load StdEnv/2023
        module load python/3.11
        module load cuda/12.2
        module load arrow/17.0.0
        ;;
    beluga|rorqual)
        module load StdEnv/2023
        module load python/3.11
        module load cuda/12.2
        ;;
    fir)
        # FIR uses CVMFS profile
        if [ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]; then
            source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
        fi
        module load python/3.11
        module load cuda/12.2
        ;;
    killarney)
        module load python/3.11
        module load cuda/12.2
        ;;
    *)
        module load python/3.11 2>/dev/null || true
        module load cuda/12.2 2>/dev/null || true
        ;;
esac

module load git 2>/dev/null || true

# 2. Create virtual environment
echo "--- Creating Python environment ---"
VENV_DIR="$HOME/chromaguide_v2_env"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate? (y/N): " RECREATE
    if [ "$RECREATE" = "y" ]; then
        rm -rf "$VENV_DIR"
        python -m venv "$VENV_DIR"
    fi
else
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 3. Install PyTorch (CUDA 12.x)
echo "--- Installing PyTorch ---"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
echo "--- Installing ChromaGuide v2 ---"
cd "$(dirname "$0")/.."
pip install -r requirements.txt
pip install -e .

# 5. Optional: Mamba SSM + LoRA (may fail on some clusters)
echo "--- Installing optional packages ---"
pip install mamba-ssm causal-conv1d 2>/dev/null || echo "mamba-ssm not available (will use GRU fallback)"
pip install peft 2>/dev/null || echo "peft not available (will use manual LoRA)"

# 6. Verify
echo "--- Verification ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import chromaguide
print(f'ChromaGuide: {chromaguide.__version__}')

from chromaguide.utils import load_config
from chromaguide.models.chromaguide import ChromaGuideModel
from chromaguide.utils.reproducibility import count_parameters
cfg = load_config()
model = ChromaGuideModel(cfg)
print(f'Model: {count_parameters(model)[\"trainable_M\"]} params')
print('All checks passed!')
"

# 7. Create directories
mkdir -p results/{figures,tables,checkpoints,logs,cv_results}
mkdir -p data/{raw,processed/splits}

# 8. Print cluster-specific info
echo ""
echo "============================================"
echo "Setup complete for: $CLUSTER"
echo "Activate env:  source $VENV_DIR/bin/activate"
echo ""

case "$CLUSTER" in
    narval)
        echo "GPU: 4× A100-40GB per node"
        echo "Account: --account=def-kwiese (or your allocation)"
        echo "GPU flag: --gres=gpu:a100:1"
        echo "Max cores/GPU: 12"
        echo "Max mem/GPU: ~125G"
        ;;
    beluga|rorqual)
        echo "GPU: 4× V100-16GB per node (Béluga) / H100-80GB (Rorqual)"
        echo "Account: --account=def-kwiese"
        echo "GPU flag: --gres=gpu:1 (or --gres=gpu:v100:1)"
        echo "Max cores/GPU: 10"
        echo "Max mem/GPU: ~46G"
        ;;
    fir)
        echo "GPU: A100-80GB"
        echo "Account: --account=def-kwiese"
        echo "GPU flag: --gres=gpu:a100:1"
        ;;
    killarney)
        echo "GPU: 4× L40S-48GB (Standard) or 8× H100-80GB (Performance)"
        echo "Account: --account=<your-aip-account>"
        echo "Standard GPU: --gres=gpu:l40s:1"
        echo "Performance GPU: --gres=gpu:h100:1"
        ;;
esac

echo ""
echo "Next: python scripts_v2/run_all_experiments.py --cluster $CLUSTER"
echo "============================================"
