"""ChromaGuide Model Zoo

Unified model factory and registry for all model architectures.

Quick Start:
    from src.model import create_model, load_model, ModelFactory
    
    # Create model
    model = create_model('crispro', d_model=256, n_layers=4)
    
    # Load from checkpoint
    model = load_model('checkpoints/best.pt', 'crispro')
    
    # Or use factory directly
    factory = ModelFactory(device='cuda')
    model = factory.create('chromaguide', use_epigenomics=True)
    factory.save_checkpoint(model, 'model.pt')
"""

# Registry and Factory
from .registry import ModelRegistry, is_registered, list_registered_models
from .factory import ModelFactory, create_factory

# Utilities
from .utils import (
    get_factory,
    create_model,
    load_model,
    load_model_minimal,
    save_model,
    list_available_models,
    model_info,
    count_parameters,
    freeze_backbone,
    unfreeze_backbone,
    get_trainable_parameters,
    get_learning_rate_groups,
    validate_model_config,
    print_model_summary,
    compose_models,
)

# Model Zoo (auto-registers all models)
from . import model_zoo

__all__ = [
    # Registry
    'ModelRegistry',
    'is_registered',
    'list_registered_models',
    
    # Factory
    'ModelFactory',
    'create_factory',
    'get_factory',
    
    # Utilities
    'create_model',
    'load_model',
    'load_model_minimal',
    'save_model',
    'list_available_models',
    'model_info',
    'count_parameters',
    'freeze_backbone',
    'unfreeze_backbone',
    'get_trainable_parameters',
    'get_learning_rate_groups',
    'validate_model_config',
    'print_model_summary',
    'compose_models',
]
