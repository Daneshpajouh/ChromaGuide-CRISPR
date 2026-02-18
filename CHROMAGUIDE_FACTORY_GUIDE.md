"""
CHROMAGUIDE MODEL FACTORY - Complete Usage Guide

This document covers the unified model factory system for ChromaGuide.
It replaces 40+ scattered model loading scripts with a clean, consistent interface.

Table of Contents:
  1. Quick Start
  2. Creating Models
  3. Loading Models
  4. Model Composition
  5. Training Integration
  6. Advanced Usage
  7. Troubleshooting
"""

# ==============================================================================
# 1. QUICK START
# ==============================================================================

"""
The simplest way to create and use a model:
"""

from src.model import create_model, load_model, save_model

# Create a model
model = create_model('crispro', d_model=256, n_layers=4, device='cuda')

# Use it
output = model(sequence_input, epigenomics_input)

# Save it
save_model(model, 'checkpoints/crispro.pt', model_name='crispro')

# Load it back
model = load_model('checkpoints/crispro.pt', 'crispro', device='cuda')


# ==============================================================================
# 2. CREATING MODELS
# ==============================================================================

"""
Get available models:
"""

from src.model import list_available_models, model_info

# See all models
models = list_available_models()
for name, desc in models.items():
    print(f"{name}: {desc}")

# Get detailed info about a model
info = model_info('chromaguide')
print(info)
# Output: {
#   'class': ChromaGuideModel,
#   'config': {...default config...},
#   'description': '...',
#   'approx_params': 45000000
# }


"""
Create models with different configs:
"""

# Option 1: Use defaults
model1 = create_model('crispro')

# Option 2: Override specific parameters
model2 = create_model('crispro', d_model=512, n_layers=6, device='cuda')

# Option 3: Provide full config dict
config = {
    'd_model': 256,
    'n_layers': 4,
    'use_quantum': True,
    'use_topo': True,
}
model3 = create_model('crispro_mamba_x', config=config)

# Option 4: Mix config dict and kwargs
model4 = create_model('dnabert_mamba', config=config, adapter_out_dim=512)


"""
Using the ModelFactory directly (advanced):
"""

from src.model import ModelFactory

factory = ModelFactory(device='cuda')

# Create models
m1 = factory.create('crispro')
m2 = factory.create('chromaguide', use_epigenomics=True)

# Create multiple models
models = factory.create_ensemble(['crispro', 'deepmens', 'chromaguide'])

# Change device
factory.to('cpu')


# ==============================================================================
# 3. LOADING & SAVING MODELS
# ==============================================================================

"""
Simple save/load:
"""

from src.model import save_model, load_model

# Save with metadata
save_model(
    model,
    'checkpoints/best.pt',
    model_name='crispro',
    config={'d_model': 256, 'n_layers': 4}
)

# Load (metadata is used to reconstruct model)
model, checkpoint = factory.load_checkpoint('checkpoints/best.pt')


"""
Minimal load (faster for inference):
"""

# Don't load metadata, just the weights
model = load_model_minimal('checkpoints/best.pt', 'crispro', d_model=256, n_layers=4)


"""
Save/load with optimizer state (for training resumption):
"""

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Save
checkpoint_data = {
    'model_name': 'crispro',
    'epoch': 10,
    'loss': 0.042,
}
factory.save_checkpoint(
    model,
    'checkpoints/epoch_10.pt',
    metadata=checkpoint_data,
    optimizer_state=optimizer.state_dict(),
)

# Load and resume
model, checkpoint = factory.load_checkpoint('checkpoints/epoch_10.pt', model_name='crispro')
optimizer.load_state_dict(checkpoint['optimizer_state'])
epoch = checkpoint['metadata']['epoch']


# ==============================================================================
# 4. MODEL COMPOSITION
# ==============================================================================

"""
Build custom architectures from components:
"""

from src.model.utils import compose_models

# Get individual components
seq_encoder = create_model('...')  # Some encoder
fusion_module = create_model('...')  # Fusion layer
pred_head = create_model('...')  # Prediction head

# Compose them
model = compose_models(
    sequence_encoder=seq_encoder,
    fusion_module=fusion_module,
    prediction_head=pred_head,
)


# ==============================================================================
# 5. TRAINING INTEGRATION
# ==============================================================================

"""
Using with training config:
"""

from src.config.chromaguide_config import TrainConfig
from src.model import create_model

# Create config
config = TrainConfig(
    model='crispro',
    epochs=50,
    batch_size=32,
    lr=2e-4,
    device='cuda',
)

# Ensure paths exist
config.resolve_paths()

# Create model using the config
model = create_model(
    config.resolve_model_name(),
    **config.model_kwargs()
)


"""
Training loop pattern:
"""

import torch
import torch.optim as optim

factory = ModelFactory(device='cuda')
model = factory.create('chromaguide', use_epigenomics=True)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(50):
    for batch in train_loader:
        outputs = model(batch['seq'], batch['epi'])
        loss = compute_loss(outputs, batch['target'])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Save checkpoint
    factory.save_checkpoint(
        model,
        'checkpoints/epoch_{epoch}.pt',
        metadata={
            'model_name': 'chromaguide',
            'epoch': epoch,
            'loss': loss.item(),
        }
    )


# ==============================================================================
# 6. ADVANCED USAGE
# ==============================================================================

"""
Parameter groups with differential learning rates:
"""

from src.model.utils import get_learning_rate_groups

groups = get_learning_rate_groups(
    model,
    base_lr=1e-4,
    groups={
        'backbone': 0.1,      # 10x lower LR for backbone
        'head': 1.0,          # Normal LR for head
        'fusion': 0.5,        # 50% LR for fusion
    }
)

optimizer = optim.AdamW(groups)


"""
Freezing and unfreezing:
"""

from src.model.utils import freeze_backbone, unfreeze_backbone

# Freeze everything except the head
freeze_backbone(model, freeze_all_but=['head', 'pred_head'])

# Unfreeze everything
unfreeze_backbone(model)


"""
Parameter counting:
"""

from src.model.utils import count_parameters

total = count_parameters(model, trainable_only=False)
trainable = count_parameters(model, trainable_only=True)

print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")


"""
Model summary:
"""

from src.model.utils import print_model_summary

print_model_summary(model)


"""
Configuration validation (dry-run):
"""

from src.model import validate_model_config

try:
    validate_model_config('crispro', {'d_model': 256, 'n_layers': 4})
    print("✓ Config is valid")
except ValueError as e:
    print(f"✗ Invalid config: {e}")


"""
List all models with descriptions:
"""

from src.model import ModelRegistry

for name in ModelRegistry.list_models():
    desc = ModelRegistry.get_description(name)
    config = ModelRegistry.get_config(name)
    print(f"{name:20} | {desc}")
    print(f"  Default config: {config}")


"""
Register custom models:
"""

from src.model import ModelRegistry
import torch.nn as nn

@ModelRegistry.register(
    'my_custom_model',
    'My custom CRISPR model'
)
class MyCustomModel(nn.Module):
    def __init__(self, d_model=256, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, x):
        return self.linear(x)

# Now can use it
model = create_model('my_custom_model', d_model=512)


# ==============================================================================
# 7. TROUBLESHOOTING
# ==============================================================================

"""
ERROR: Model 'xyz' not found
  Solution: Check available models with list_available_models()

ERROR: Config invalid for model
  Solution: Use validate_model_config() to test config before creating

ERROR: Out of memory when loading
  Solution: Use load_model_minimal() with smaller dtype, or load on CPU first

ERROR: Checkpoint incompatible
  Solution: Specify model_name and config when loading
  load_checkpoint(..., model_name='crispro', config={...})
"""


# ==============================================================================
# SUMMARY TABLE: Available Models
# ==============================================================================

"""
Model Name              | Description                              | Key Parameters
-----------------------|------------------------------------------|--------------------
chromaguide            | Multi-modal (seq + epi) with Beta regr   | encoder_type, use_epigenomics
crispro                | Mamba-2 SSM for efficiency prediction    | d_model, n_layers, use_causal
crispro_mamba_x        | CRISPRO + quantum + topological          | d_model, n_layers, use_quantum
dnabert_mamba          | DNABERT foundation + Mamba adapter       | dnabert_name, adapter_out_dim
deepmens               | Multi-branch CNN (seq+shape+position)    | seq_len, num_shape_features
deepmens_ensemble      | Ensemble of 5 DeepMENS models            | models_list
rnagenesis_vae         | VAE for synthetic guide generation       | latent_dim, hidden_dim
rnagenesis_diffusion   | Diffusion for synthetic generation       | noise_steps, hidden_dim
conformal              | Conformal prediction wrapper             | coverage_level (optional)
off_target             | CNN-based off-target risk prediction     | seq_len, hidden_dim
ot_rag                 | RAG for off-target with retriever        | retriever_top_k (optional)
"""
