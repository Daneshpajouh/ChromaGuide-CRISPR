"""Utility functions for model creation and management.

Convenience functions for common model operations:
- Loading pretrained models
- Creating model compositions
- Model analysis and introspection
"""

import logging
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from .factory import ModelFactory
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


# Global factory instance
_global_factory: Optional[ModelFactory] = None


def get_factory(device: Optional[str] = None) -> ModelFactory:
    """Get or create global model factory.
    
    Args:
        device: Optional device to initialize factory with
        
    Returns:
        Global ModelFactory instance
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = ModelFactory(device=device)
    elif device is not None:
        _global_factory.to(device)
    
    return _global_factory


def create_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """Convenience function to create a model.
    
    Args:
        model_name: Model identifier
        config: Configuration dictionary
        device: Device to place model on
        **kwargs: Additional arguments
        
    Returns:
        Instantiated model
    """
    factory = get_factory(device)
    return factory.create(model_name, config=config, **kwargs)


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: Optional[str] = None,
    **model_kwargs
) -> nn.Module:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_name: Model identifier
        device: Device to load on
        **model_kwargs: Model configuration
        
    Returns:
        Loaded model
    """
    factory = get_factory(device)
    model, _ = factory.load_checkpoint(checkpoint_path, model_name=model_name, **model_kwargs)
    return model


def load_model_minimal(
    checkpoint_path: str,
    model_name: str,
    device: Optional[str] = None,
    **model_kwargs
) -> nn.Module:
    """Minimal checkpoint load (faster, for inference).
    
    Args:
        checkpoint_path: Path to checkpoint
        model_name: Model identifier
        device: Device to load on
        **model_kwargs: Model configuration
        
    Returns:
        Loaded model in eval mode
    """
    factory = get_factory(device)
    return factory.load_checkpoint_minimal(checkpoint_path, model_name, **model_kwargs)


def save_model(
    model: nn.Module,
    checkpoint_path: str,
    model_name: Optional[str] = None,
    config: Optional[Dict] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Output path
        model_name: Model identifier (added to metadata)
        config: Model configuration (added to metadata)
        metadata: Additional metadata
    """
    factory = get_factory()
    
    # Build metadata
    save_metadata = metadata or {}
    if model_name is not None:
        save_metadata['model_name'] = model_name
    if config is not None:
        save_metadata['config'] = config
    
    factory.save_checkpoint(model, checkpoint_path, metadata=save_metadata)


def list_available_models() -> Dict[str, str]:
    """Get all registered models with descriptions.
    
    Returns:
        Dict mapping model names to descriptions
    """
    return ModelRegistry.list_with_descriptions()


def model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Dict with model info
    """
    factory = get_factory()
    return factory.get_model_info(model_name)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: Model instance
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module, freeze_all_but: List[str] = None) -> None:
    """Freeze all model parameters except specified modules.
    
    Args:
        model: Model instance
        freeze_all_but: List of module names to keep trainable (if None, freeze all)
    """
    freeze_all_but = freeze_all_but or []
    
    for name, param in model.named_parameters():
        # Check if this parameter belongs to a module we want to keep trainable
        should_freeze = True
        for keep_name in freeze_all_but:
            if keep_name in name:
                should_freeze = False
                break
        
        param.requires_grad = not should_freeze
    
    logger.info(f"Froze backbone. Trainable modules: {freeze_all_but or 'none'}")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters.
    
    Args:
        model: Model instance
    """
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Unfroze all parameters")


def get_trainable_parameters(model: nn.Module) -> List[torch.nn.Parameter]:
    """Get list of trainable parameters.
    
    Args:
        model: Model instance
        
    Returns:
        List of trainable parameters
    """
    return [p for p in model.parameters() if p.requires_grad]


def get_learning_rate_groups(
    model: nn.Module,
    base_lr: float = 1e-4,
    decay_factor: float = 0.1,
    groups: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Create parameter groups for differential learning rates.
    
    Useful for fine-tuning where you want different LR for different layers.
    
    Args:
        model: Model instance
        base_lr: Base learning rate
        decay_factor: Factor to decay LR for earlier layers
        groups: Custom group definitions {module_name: lr_multiplier}
        
    Returns:
        List of parameter group dicts for optimizer
    """
    groups = groups or {}
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter matches any custom group
        group_name = 'default'
        group_lr = base_lr
        
        for pattern, lr_mult in groups.items():
            if pattern in name:
                group_name = pattern
                group_lr = base_lr * lr_mult
                break
        
        if group_name not in param_groups:
            param_groups[group_name] = {'params': [], 'lr': group_lr}
        
        param_groups[group_name]['params'].append(param)
    
    return list(param_groups.values())


def validate_model_config(model_name: str, config: Dict[str, Any]) -> bool:
    """Validate model configuration (dry-run test).
    
    Args:
        model_name: Model identifier
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError or TypeError with error details
    """
    factory = get_factory('cpu')  # Use CPU to avoid memory issues
    return factory.validate_config(model_name, config)


def print_model_summary(model: nn.Module, input_shape: Optional[tuple] = None) -> None:
    """Print model architecture summary.
    
    Args:
        model: Model instance
        input_shape: Optional input shape for shape inference
    """
    print(f"\n{'='*70}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*70}")
    print(model)
    print(f"{'='*70}")
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters:    {total_params - trainable_params:,}")
    print(f"{'='*70}\n")


def compose_models(
    sequence_encoder: nn.Module,
    fusion_module: Optional[nn.Module] = None,
    prediction_head: Optional[nn.Module] = None,
    uncertainty_estimator: Optional[nn.Module] = None,
) -> nn.Module:
    """Compose multiple model components into a unified model.
    
    Useful for building custom architectures from components.
    
    Args:
        sequence_encoder: Sequence encoding module
        fusion_module: Optional fusion module (e.g., multimodal fusion)
        prediction_head: Optional prediction head
        uncertainty_estimator: Optional uncertainty module
        
    Returns:
        Composite model (nn.Sequential or custom wrapper)
    """
    # For now, return a simple Sequential
    # In a real implementation, would create a proper composite class
    modules = [sequence_encoder]
    
    if fusion_module is not None:
        modules.append(fusion_module)
    
    if prediction_head is not None:
        modules.append(prediction_head)
    
    if uncertainty_estimator is not None:
        modules.append(uncertainty_estimator)
    
    logger.info(f"Composed model with {len(modules)} components")
    
    return nn.Sequential(*modules)
