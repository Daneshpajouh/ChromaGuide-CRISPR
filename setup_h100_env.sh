#!/bin/bash
# Nibi Cluster: H100 Stability Environment Setup
# Based on Definitive Root Cause Analysis (PyTorch 2.5 + H100 PTX)

echo "=== Setting up H100-Compatible Mamba Environment ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# 1. Clean Slate
module purge
module load python/3.10 cuda/12.2 scipy-stack

# 2. Create Fresh Virtual Environment
# Using --no-download to ensure we control exactly what goes in
cd ~/projects/def-kwiese/amird
if [ -d "mamba_h100" ]; then
    echo "Backing up existing mamba_h100 environment..."
    mv mamba_h100 mamba_h100_backup_$(date +%s)
fi

echo ">>> Creating virtualenv 'mamba_h100'..."
virtualenv --no-download mamba_h100
source mamba_h100/bin/activate

# 3. Install PyTorch 2.5 (CRITICAL DOWNGRADE)
# Nibi wheelhouse might have 2.5.1, or we use pip
# Checking local wheels first is faster, but user provided specific instructions
echo ">>> Installing PyTorch 2.5.1 (Downgrading from 2.6)..."
# Try wheelhouse first for speed/optimization
pip install --no-index torch==2.5.1 torchvision || \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install --no-index numpy pandas scipy scikit-learn tqdm

# 4. Set Compilation Flags for Hopper (sm_90)
echo ">>> Setting H100 Compilation Flags..."
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=8
export TMPDIR=~/projects/def-kwiese/amird/tmp
mkdir -p $TMPDIR

# 5. Build Mamba from Source
echo ">>> Building causal-conv1d (sm_90)..."
pip install packaging ninja
pip install --no-cache-dir --no-binary causal-conv1d "causal-conv1d>=1.5.0" --verbose --no-build-isolation

echo ">>> Building mamba-ssm (sm_90)..."
pip install --no-cache-dir --no-binary mamba-ssm "mamba-ssm>=2.2.4" --verbose --no-build-isolation

echo "=== Setup Complete ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('Mamba Kernels Loaded Successfully')"
