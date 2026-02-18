#!/bin/bash
# Test forward pass step-by-step

cd ~/projects/def-kwiese/amird/thesis_project/CRISPRO-MAMBA-X
module load StdEnv/2023 python/3.10 scipy-stack cuda/12.2
source ~/projects/def-kwiese/amird/pytorch_env/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "=== Step-by-Step Forward Pass Debugging ==="
python3 << 'EOF'
import torch
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from torch.utils.data import DataLoader, Subset

device = "cuda"
print(f"Device: {device}")

# Load data
dataset = CRISPRoffTDataset(use_mini=True, context_window=4096)
subset = Subset(dataset, [0])  # Just one sample
loader = DataLoader(subset, batch_size=1)
batch = next(iter(loader))

seqs = batch['sequence'].to(device)
epi = batch['epigenetics'].to(device)

print(f"\nInput shapes: seqs={seqs.shape}, epi={epi.shape}")

# Initialize model
model = CRISPROModel(d_model=256, n_layers=4, n_modalities=6).to(device)

# Check dt_bias specifically
print("\n=== Checking dt_bias values ===")
for i in range(4):
    dt_bias = model.encoder.layers[i].mamba.dt_bias
    print(f"Layer {i} dt_bias: {dt_bias}")
    print(f"  Range: [{dt_bias.min():.6e}, {dt_bias.max():.6e}]")
    print(f"  Has NaN: {torch.isnan(dt_bias).any()}")
    print(f"  Has Inf: {torch.isinf(dt_bias).any()}")

# Forward with hooks
print("\n=== Forward Pass with Hooks ===")
def check_output(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    has_nan = torch.isnan(o).any().item()
                    has_inf = torch.isinf(o).any().item()
                    if has_nan or has_inf:
                        print(f"  ❌ {name} output[{i}]: NaN={has_nan}, Inf={has_inf}")
                    else:
                        print(f"  ✅ {name} output[{i}]: OK")
        elif isinstance(output, torch.Tensor):
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            if has_nan or has_inf:
                print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
                print(f"     Range: [{output.min():.6e}, {output.max():.6e}]")
            else:
                print(f"  ✅ {name}: OK")
    return hook

# Register hooks
model.dna_emb.register_forward_hook(check_output("DNA Embedding"))
model.epi_fusion.register_forward_hook(check_output("Epi Fusion"))
for i in range(4):
    model.encoder.layers[i].register_forward_hook(check_output(f"Encoder Layer {i}"))

try:
    with torch.no_grad():
        pred_cls, pred_reg = model(seqs, epi)
    print(f"\nFinal predictions - cls NaN: {torch.isnan(pred_cls).any()}, reg NaN: {torch.isnan(pred_reg).any()}")
except Exception as e:
    print(f"\n❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

EOF
