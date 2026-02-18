"""Unified model factory for ChromaGuide.

This module provides a factory pattern for creating, loading, and managing
model instances with configuration validation and device handling.

Example:
    Create a model:
    >>> factory = ModelFactory()
    >>> model = factory.create('crispro', d_model=256, n_layers=4)
    >>> model.to('cuda')
    
    Load from checkpoint:
    >>> model = factory.load_checkpoint('checkpoints/best.pt', device='cuda')
    
    Save checkpoint:
    >>> factory.save_checkpoint(model, 'checkpoints/model.pt')
"""
import os
import json
import logging
from typing import Dict, Optional, Any, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn

from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for unified model creation, loading, and management.
    
    Handles:
    - Model instantiation from registry
    - Configuration management
    - Device placement
    - Checkpoint saving/loading
    - Model composition (encoder + fusion + head, etc.)
    """
    
    def __init__(self, device: Optional[str] = None, default_dtype: torch.dtype = torch.float32):
        """Initialize the factory.
        
        Args:
            device: Device to place models on ('cuda', 'cpu', 'mps', or None for auto-detect)
            default_dtype: Default torch dtype for models
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.default_dtype = default_dtype
        self.dtype_str = str(default_dtype).split('.')[-1]  # 'float32', 'float16', etc.
        logger.info(f"ModelFactory initialized: device={self.device}, dtype={self.dtype_str}")
    
    def create(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> nn.Module:
        """Create a model instance from registry.
        
        Args:
            model_name: Registered model name (e.g., 'crispro', 'chromaguide')
            config: Optional configuration dictionary to override defaults
            **kwargs: Additional arguments that override config values
            
        Returns:
            Instantiated model on the configured device
            
        Raises:
            ValueError: If model not found in registry
            TypeError: If configuration is invalid
        """
        # Get model class
        model_class = ModelRegistry.get(model_name)
        if model_class is None:
            available = ModelRegistry.list_models()
            raise ValueError(
                f"Model '{model_name}' not found. Available models:\n  " +
                "\n  ".join(available)
            )
        
        # Merge configurations: defaults -> config -> kwargs
        merged_config = ModelRegistry.get_config(model_name).copy()
        if config is not None:
            merged_config.update(config)
        merged_config.update(kwargs)
        
        logger.info(f"Creating model '{model_name}' with config: {merged_config}")
        
        try:
            model = model_class(**merged_config)
            model = model.to(self.device)
            if hasattr(model, 'double'):
                model = model.to(dtype=self.default_dtype)
            
            logger.info(f"✓ Created model '{model_name}' on device={self.device}")
            return model
        
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate '{model_name}' with config {merged_config}.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error creating model '{model_name}': {e}")
            raise
    
    def create_ensemble(
        self,
        model_names: list,
        configs: Optional[Dict[str, Dict]] = None,
        **common_kwargs
    ) -> nn.ModuleList:
        """Create multiple models as an ensemble.
        
        Args:
            model_names: List of model names (can include duplicates with different configs)
            configs: Dict mapping model names to their configs
            **common_kwargs: Arguments applied to all models
            
        Returns:
            nn.ModuleList of models
        """
        configs = configs or {}
        models = nn.ModuleList()
        
        for model_name in model_names:
            model_config = configs.get(model_name, {})
            model = self.create(model_name, config=model_config, **common_kwargs)
            models.append(model)
        
        logger.info(f"Created ensemble of {len(models)} models")
        return models
    
    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            path: Output path
            metadata: Optional metadata (e.g., training info, model name, config)
            optimizer_state: Optional optimizer state dict
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state': model.state_dict(),
            'dtype': str(self.default_dtype),
            'device': self.device,
        }
        
        # Add metadata
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Add optimizer state
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        
        try:
            torch.save(checkpoint, path)
            logger.info(f"✓ Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model_name: Optional[str] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        strict: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_name: Model name (from checkpoint metadata if not provided)
            config: Model config (from checkpoint if not provided)
            device: Device to load on (uses factory default if not provided)
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Tuple of (model, checkpoint_dict)
            
        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If model_name cannot be determined
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        device = device or self.device
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
        
        # Extract metadata
        metadata = checkpoint.get('metadata', {})
        model_config = config or metadata.get('config', {})
        
        # Determine model name
        if model_name is None:
            model_name = metadata.get('model_name')
            if model_name is None:
                raise ValueError(
                    "model_name not provided and not found in checkpoint metadata. "
                    "Please specify model_name when loading."
                )
        
        # Create model
        model = self.create(model_name, config=model_config)
        
        # Load state dict
        try:
            model.load_state_dict(checkpoint['model_state'], strict=strict)
            logger.info(f"✓ Loaded model state (strict={strict})")
        except RuntimeError as e:
            if strict:
                logger.error(f"Failed to load state dict with strict=True: {e}")
                raise
            else:
                logger.warning(f"Partial state dict load: {e}")
        
        return model, checkpoint
    
    def load_checkpoint_minimal(
        self,
        checkpoint_path: str,
        model_name: str,
        **model_kwargs
    ) -> nn.Module:
        """Minimal checkpoint loading (quick inference).
        
        Creates model and loads weights without metadata validation.
        
        Args:
            checkpoint_path: Checkpoint path
            model_name: Model name
            **model_kwargs: Model configuration
            
        Returns:
            Loaded model
        """
        device = self.device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = self.create(model_name, **model_kwargs)
        model.load_state_dict(checkpoint['model_state'], strict=False)
        model.eval()
        
        return model
    
    def validate_config(self, model_name: str, config: Dict) -> bool:
        """Validate configuration for a model (dry-run instantiation).
        
        Args:
            model_name: Model name
            config: Configuration to validate
            
        Returns:
            True if config is valid
            
        Raises:
            ValueError or TypeError with details
        """
        try:
            # Try to instantiate on CPU with small dtype to avoid memory issues
            original_device = self.device
            original_dtype = self.default_dtype
            
            self.device = 'cpu'
            self.default_dtype = torch.float32
            
            model = self.create(model_name, config=config)
            del model
            
            # Restore
            self.device = original_device
            self.default_dtype = original_dtype
            
            logger.info(f"✓ Config validated for model '{model_name}'")
            return True
        
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Dict with keys: class, config, description, parameters
        """
        info = ModelRegistry.info(model_name)
        
        # Count parameters
        try:
            model = self.create(model_name)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            info['approx_params'] = num_params
            del model
        except Exception as e:
            logger.warning(f"Could not count parameters for '{model_name}': {e}")
        
        return info
    
    def to(self, device: str) -> None:
        """Change factory device.
        
        Args:
            device: Target device ('cuda', 'cpu', 'mps')
        """
        self.device = device
        logger.info(f"Factory device changed to: {device}")
    
    def __str__(self) -> str:
        """String representation."""
        models = ModelRegistry.list_models()
        return (
            f"ModelFactory(device={self.device}, dtype={self.dtype_str}, "
            f"models={len(models)})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


def create_factory(
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32
) -> ModelFactory:
    """Convenience function to create a ModelFactory instance.
    
    Args:
        device: Target device
        dtype: Target dtype
        
    Returns:
        ModelFactory instance
    """
    return ModelFactory(device=device, default_dtype=dtype)
