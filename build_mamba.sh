#!/bin/bash
# Rebuild Mamba-SSM for H100 (Hopper)
# Run on Login Node (or interactive job if needed)

echo "=== Rebuilding Mamba-SSM for H100 ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Load Modules (Must match what we use for training)
module load python/3.10 scipy-stack cuda/12.2

# Activate Environment
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate

# Set TMPDIR to project space to avoid Home Quota Exceeded
export TMPDIR=~/projects/def-kwiese/amird/tmp
mkdir -p $TMPDIR
echo "Using TMPDIR=$TMPDIR"

# 1. Uninstall existing binaries
echo ">>> Uninstalling existing mamba-ssm and causal-conv1d..."
pip uninstall -y mamba-ssm causal-conv1d

# 2. Set H100 Compilation Flags
echo ">>> Setting Compilation Flags for Compute Capability 9.0 (H100)..."
export TORCH_CUDA_ARCH_LIST="9.0"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS=8

# 3. Install Causal Conv1d (Dependency)
echo ">>> Building causal-conv1d from source..."
# Using --no-binary to force compilation
pip install --no-cache-dir --no-binary causal-conv1d "causal-conv1d>=1.4.0" --verbose --no-build-isolation

# 4. Install Mamba-SSM
echo ">>> Building mamba-ssm from source..."
pip install --no-cache-dir --no-binary mamba-ssm "mamba-ssm>=2.2.4" --verbose --no-build-isolation

echo "=== Build Complete ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import mamba_ssm; print(f'Mamba SSM Version: {mamba_ssm.__version__}')"
