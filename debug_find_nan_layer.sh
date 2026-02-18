#!/bin/bash
# Debug which specific layer has NaN in initialization

cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Finding Which Layer Has NaN ==="
python3 << 'EOF'
import torch
from src.model.crispro import CRISPROModel

device = "cuda"
print(f"Device: {device}")

# Initialize model
model = CRISPROModel(d_model=256, n_layers=4, n_modalities=6).to(device)

# Check each parameter
print("\n=== Checking Each Parameter ===")
for name, param in model.named_parameters():
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    if has_nan or has_inf:
        print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}, shape={param.shape}")
        print(f"   Values: {param.flatten()[:10]}")
    else:
        param_min = param.min().item()
        param_max = param.max().item()
        print(f"✅ {name}: range=[{param_min:.6f}, {param_max:.6f}], shape={param.shape}")

print("\n=== Checking Buffers ===")
for name, buffer in model.named_buffers():
    if buffer.numel() > 0:
        has_nan = torch.isnan(buffer).any().item()
        has_inf = torch.isinf(buffer).any().item()
        if has_nan or has_inf:
            print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"✅ {name}: OK")

EOF
