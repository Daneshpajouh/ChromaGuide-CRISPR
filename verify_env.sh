#!/bin/bash
# Verify all dependencies and fix any import issues
echo "=== Verifying Python Environment ==="
conda activate cg_train

# Check Python version
python --version

# Test all critical imports
echo ""
echo "=== Testing Critical Imports ==="
python -c "
import sys
print('Testing imports...')
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
except Exception as e:
    print(f'✗ PyTorch error: {e}')
    sys.exit(1)

try:
    import numpy
    print(f'✓ NumPy {numpy.__version__}')
except Exception as e:
    print(f'✗ NumPy error: {e}')
    sys.exit(1)

try:
    import pandas
    print(f'✓ Pandas {pandas.__version__}')
except Exception as e:
    print(f'✗ Pandas error: {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'✓ scikit-learn {sklearn.__version__}')
except Exception as e:
    print(f'✗ scikit-learn error: {e}')
    sys.exit(1)

try:
    from imblearn.over_sampling import SMOTE
    print(f'✓ imbalanced-learn SMOTE available')
except Exception as e:
    print(f'✗ imbalanced-learn error: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except Exception as e:
    print(f'✗ Transformers error: {e}')

try:
    from PIL import Image
    print(f'✓ Pillow available')
except Exception as e:
    print(f'✗ Pillow error: {e}')

print('')
print('MPS GPU available:', torch.backends.mps.is_available())
"

if [ $? -ne 0 ]; then
    echo "Import test failed!"
    exit 1
fi

echo ""
echo "=== All imports successful! ==="
