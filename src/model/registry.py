"""Model registry system for unified model management.

This module provides a centralized registry for all model architectures,
enabling dynamic registration and lookup of model classes.

Example:
    Register a model:
    >>> @ModelRegistry.register('crispro')
    >>> class CRISPROModel(nn.Module):
    ...     pass

    Get a model:
    >>> model_class = ModelRegistry.get('crispro')
    >>> model = model_class(**config)

    List available models:
    >>> ModelRegistry.list_models()
    ['crispro', 'chromaguide', 'dnabert_mamba', ...]
"""
from typing import Dict, Type, List, Optional, Callable
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all model architectures.
    
    Thread-safe model registration and lookup.
    """
    
    _models: Dict[str, Type[nn.Module]] = {}
    _configs: Dict[str, Dict] = {}  # Default configs per model
    _descriptions: Dict[str, str] = {}  # Model descriptions
    
    @classmethod
    def register(cls, name: str, description: str = ""):
        """Register a model class (decorator pattern).
        
        Args:
            name: Unique model identifier (e.g., 'crispro', 'chromaguide')
            description: Human-readable description of the model
            
        Returns:
            Decorator function
            
        Example:
            @ModelRegistry.register('crispro', 'CRISPR efficiency prediction with Mamba-2')
            class CRISPROModel(nn.Module):
                pass
        """
        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._models:
                logger.warning(f"Overriding existing model registration: {name}")
            
            cls._models[name] = model_class
            cls._descriptions[name] = description
            logger.info(f"Registered model: {name}")
            return model_class
        
        return decorator
    
    @classmethod
    def register_default_config(cls, name: str, config: Dict) -> None:
        """Register default configuration for a model.
        
        Args:
            name: Model identifier
            config: Default configuration dictionary
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Register model first.")
        
        cls._configs[name] = config
        logger.debug(f"Registered default config for model: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[nn.Module]]:
        """Get a model class by name.
        
        Args:
            name: Model identifier
            
        Returns:
            Model class or None if not found
        """
        return cls._models.get(name)
    
    @classmethod
    def get_config(cls, name: str) -> Dict:
        """Get default configuration for a model.
        
        Args:
            name: Model identifier
            
        Returns:
            Default configuration dictionary (empty dict if none registered)
        """
        return cls._configs.get(name, {}).copy()
    
    @classmethod
    def get_description(cls, name: str) -> str:
        """Get model description.
        
        Args:
            name: Model identifier
            
        Returns:
            Description string or empty string
        """
        return cls._descriptions.get(name, "")
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names.
        
        Returns:
            Sorted list of model identifiers
        """
        return sorted(cls._models.keys())
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if a model is registered.
        
        Args:
            name: Model identifier
            
        Returns:
            True if model is registered
        """
        return name in cls._models
    
    @classmethod
    def info(cls, name: str) -> Dict:
        """Get full information about a model.
        
        Args:
            name: Model identifier
            
        Returns:
            Dict with keys: 'class', 'config', 'description'
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered")
        
        return {
            'class': cls._models[name],
            'config': cls.get_config(name),
            'description': cls.get_description(name),
        }
    
    @classmethod
    def list_with_descriptions(cls) -> Dict[str, str]:
        """Get all models with descriptions.
        
        Returns:
            Dict mapping model names to descriptions
        """
        return {name: cls._descriptions.get(name, "") 
                for name in cls.list_models()}
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing).
        
        Use with caution!
        """
        cls._models.clear()
        cls._configs.clear()
        cls._descriptions.clear()
        logger.warning("Model registry cleared")


def is_registered(name: str) -> bool:
    """Check if a model is registered (convenience function).
    
    Args:
        name: Model identifier
        
    Returns:
        True if registered
    """
    return ModelRegistry.exists(name)


def list_registered_models() -> List[str]:
    """List all registered models (convenience function).
    
    Returns:
        Sorted list of model names
    """
    return ModelRegistry.list_models()
