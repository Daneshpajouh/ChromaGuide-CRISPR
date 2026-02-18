#!/bin/bash
###############################################################################
# H100 Cluster Diagnostic Script
# Purpose: Diagnose why job 5860955 failed and verify cluster setup
# Usage: bash diagnose_h100_cluster.sh
###############################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "ChromaGuide H100 Cluster Diagnostic"
echo "Date: $(date)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Get the base project path
PROJECT_BASE="$HOME/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X"

echo "📋 CONFIGURATION"
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Project Base: $PROJECT_BASE"
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "Current Dir: $(pwd)"
echo ""

# ============================================================================
# CHECK 1: Project Directory Structure
# ============================================================================
echo "✓ CHECK 1: Project Directory Structure"
echo "────────────────────────────────────────────────────────────────────────────────"
if [ -d "$PROJECT_BASE" ]; then
    echo "✅ Project directory exists: $PROJECT_BASE"
    echo ""
    echo "Directory contents:"
    ls -lah "$PROJECT_BASE" | head -20
    echo ""
else
    echo "❌ Project directory NOT found: $PROJECT_BASE"
    echo "ERROR: Cannot continue without project directory"
    exit 1
fi
echo ""

# ============================================================================
# CHECK 2: Script Files
# ============================================================================
echo "✓ CHECK 2: Script Files"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -f "$PROJECT_BASE/verify_real_h100.sh" ]; then
    echo "✅ verify_real_h100.sh exists"
    if [ -x "$PROJECT_BASE/verify_real_h100.sh" ]; then
        echo "✅ verify_real_h100.sh is executable"
    else
        echo "⚠️  verify_real_h100.sh is NOT executable"
        echo "   Attempting to fix: chmod +x verify_real_h100.sh"
        chmod +x "$PROJECT_BASE/verify_real_h100.sh"
    fi
else
    echo "❌ verify_real_h100.sh NOT found"
fi

if [ -f "$PROJECT_BASE/src/train.py" ]; then
    echo "✅ src/train.py exists"
else
    echo "❌ src/train.py NOT found"
fi
echo ""

# ============================================================================
# CHECK 3: Directories
# ============================================================================
echo "✓ CHECK 3: Required Directories"
echo "────────────────────────────────────────────────────────────────────────────────"

DIRS_TO_CHECK=("logs" "checkpoints" "src" "data")
for dir in "${DIRS_TO_CHECK[@]}"; do
    if [ -d "$PROJECT_BASE/$dir" ]; then
        echo "✅ $dir/ exists"
    else
        echo "⚠️  $dir/ NOT found, creating..."
        mkdir -p "$PROJECT_BASE/$dir"
        echo "   Created: $PROJECT_BASE/$dir"
    fi
done
echo ""

# ============================================================================
# CHECK 4: Module Availability
# ============================================================================
echo "✓ CHECK 4: Module Availability"
echo "────────────────────────────────────────────────────────────────────────────────"

# Try to load modules
set +e  # Don't exit on module errors
module purge 2>/dev/null
echo "Attempting to load StdEnv/2023..."
module load StdEnv/2023 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ StdEnv/2023 loaded"
else
    echo "⚠️  StdEnv/2023 load had warnings (may be okay)"
fi

echo "Attempting to load python/3.10..."
module load python/3.10 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ python/3.10 loaded"
else
    echo "⚠️  python/3.10 load had warnings"
fi

echo "Attempting to load scipy-stack..."
module load scipy-stack 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ scipy-stack loaded"
else
    echo "⚠️  scipy-stack load had warnings"
fi

echo "Attempting to load cuda/12.2..."
module load cuda/12.2 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ cuda/12.2 loaded"
else
    echo "⚠️  cuda/12.2 load had warnings"
fi

set -e  # Re-enable exit on error
echo ""

# ============================================================================
# CHECK 5: Python and Environment
# ============================================================================
echo "✓ CHECK 5: Python Environment"
echo "────────────────────────────────────────────────────────────────────────────────"

# Check python availability
PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -n "$PYTHON_CMD" ]; then
    echo "✅ Python found: $PYTHON_CMD"
    echo "   Version: $($PYTHON_CMD --version)"
else
    echo "⚠️  Python not found in PATH"
fi

# Check conda/mamba availability
if command -v conda &> /dev/null; then
    echo "✅ conda found: $(conda --version)"
elif command -v mamba &> /dev/null; then
    echo "✅ mamba found: $(mamba --version)"
else
    echo "⚠️  Neither conda nor mamba found"
fi
echo ""

# ============================================================================
# CHECK 6: Environment Activation
# ============================================================================
echo "✓ CHECK 6: Environment Activation"
echo "────────────────────────────────────────────────────────────────────────────────"

ENV_PATH="$HOME/projects/def-kwiese/amird/mamba_h100"
if [ -d "$ENV_PATH" ]; then
    echo "✅ Environment found: $ENV_PATH"
    echo "   Contents:"
    ls -lah "$ENV_PATH" | head -10
    
    # Check if activation script exists
    if [ -f "$ENV_PATH/bin/activate" ]; then
        echo "✅ Activation script exists"
    else
        echo "⚠️  Activation script not found at expected location"
    fi
else
    echo "❌ Environment NOT found at: $ENV_PATH"
    echo "   Try running: conda create -p $ENV_PATH python=3.10"
    exit 1
fi
echo ""

# ============================================================================
# CHECK 7: GPU Availability
# ============================================================================
echo "✓ CHECK 7: GPU Availability"
echo "────────────────────────────────────────────────────────────────────────────────"

if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
else
    echo "❌ nvidia-smi NOT found (not on GPU node)"
    echo "   Note: This is expected on login node. H100 test runs on compute nodes."
fi
echo ""

# ============================================================================
# CHECK 8: Python Import Test
# ============================================================================
echo "✓ CHECK 8: Python Import Test"
echo "────────────────────────────────────────────────────────────────────────────────"

cd "$PROJECT_BASE"
if [ -f "$ENV_PATH/bin/activate" ]; then
    # Source the environment
    source "$ENV_PATH/bin/activate" 2>/dev/null || true
    
    # Try to import main modules
    echo "Testing Python imports..."
    
    if python -c "import sys; print(f'Python: {sys.version}')" 2>/dev/null; then
        echo "✅ Python execution works"
    else
        echo "⚠️  Python execution had issues"
    fi
    
    # Try importing key packages
    IMPORTS=("torch" "numpy" "transformers" "mamba_ssm")
    for pkg in "${IMPORTS[@]}"; do
        if python -c "import $pkg; print(f'  {$pkg}.__version__={$pkg.__version__}')" 2>/dev/null; then
            echo "✅ $pkg available"
        else
            echo "⚠️  $pkg not installed (may install during training)"
        fi
    done
    
    # Try importing src
    if python -c "from src import train; print('✅ src.train imports successfully')" 2>/dev/null; then
        echo "✅ Main training module imports"
    else
        echo "⚠️  src.train import failed (may need pip install)"
    fi
else
    echo "⚠️  Cannot test imports without environment"
fi
cd - > /dev/null
echo ""

# ============================================================================
# CHECK 9: SLURM Status
# ============================================================================
echo "✓ CHECK 9: SLURM Job Status"
echo "────────────────────────────────────────────────────────────────────────────────"

if command -v squeue &> /dev/null; then
    echo "Recent jobs for current user:"
    squeue -u "$(whoami)" --format="%.18i %.50j %.8T %.10M %.3C" | head -10
    echo ""
    
    echo "Job 5860955 status (if exists):"
    if squeue -j 5860955 2>/dev/null | grep -q 5860955; then
        squeue -j 5860955 --format="%.18i %.50j %.8T %.10M %.3C"
    else
        echo "Job 5860955 not found (completed or failed)"
    fi
    
    echo ""
    echo "GPU partition status:"
    sinfo -p gpu --format="%.20P %.10D %.15T" 2>/dev/null | head -5
else
    echo "⚠️  squeue not found (not on SLURM system)"
fi
echo ""

# ============================================================================
# CHECK 10: SLURM Output Files
# ============================================================================
echo "✓ CHECK 10: SLURM Output Files"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d "$PROJECT_BASE/logs" ]; then
    echo "Contents of logs/ directory:"
    ls -lah "$PROJECT_BASE/logs/" 2>/dev/null | head -20 || echo "  (empty or not accessible)"
    echo ""
    
    echo "Checking for job output files matching pattern 'verify_real_h100*':"
    find "$PROJECT_BASE/logs" -name "*verify_real_h100*" -o -name "*5860955*" 2>/dev/null | head -10 || echo "  (none found)"
else
    echo "logs/ directory not found"
fi
echo ""

# ============================================================================
# CHECK 11: Script Content Review
# ============================================================================
echo "✓ CHECK 11: Verify Script Content"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -f "$PROJECT_BASE/verify_real_h100.sh" ]; then
    echo "First 30 lines of verify_real_h100.sh:"
    head -30 "$PROJECT_BASE/verify_real_h100.sh"
    echo ""
    echo "... (script continues)"
else
    echo "Script not found"
fi
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 DIAGNOSTIC SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "✓ If all checks passed, the environment is ready for training"
echo ""
echo "Next steps:"
echo "  1. Review any ⚠️  warnings above"
echo "  2. Install missing packages if needed: pip install torch transformers"
echo "  3. Submit test job: sbatch verify_real_h100.sh"
echo "  4. Monitor with: squeue -u $(whoami) -l"
echo "  5. Check output: cat logs/verify_real_h100_<JOBID>.out"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
