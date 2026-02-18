"""Unified configuration for ChromaGuide training and model selection.

Integrates with the ModelFactory for clean model creation and training.

Example:
    from src.config.chromaguide_config import TrainConfig
    from src.model import create_model
    
    config = TrainConfig(model='crispro', epochs=50, batch_size=32)
    model = create_model(config.model, **config.model_kwargs())
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

# Map of short aliases to actual model names in registry
MODEL_ALIASES = {
    'baseline': 'chromaguide',
    'crispr': 'crispro',
    'mamba': 'crispro',
    'dnabert': 'dnabert_mamba',
    'hybrid': 'dnabert_mamba',
    'deepmens': 'deepmens',
    'ensemble': 'deepmens_ensemble',
}

# Dynamically get registered models (import delayed to avoid circular deps)
def get_available_models() -> list:
    try:
        from src.model import list_registered_models
        return list_registered_models()
    except Exception:
        # Fallback if registry not available
        return list(MODEL_ALIASES.values())


@dataclass
class TrainConfig:
    """Training configuration compatible with ModelFactory.
    
    Key Features:
    - model: Model name (resolved through aliases)
    - Supports loading from JSON
    - Integrates with ModelFactory for training
    - Automatic path creation and resolution
    """
    
    model: str = 'chromaguide'
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 5
    device: Optional[str] = None
    checkpoint: str = "checkpoints/best.pt"
    seed: int = 42
    log_dir: str = "logs/chromaguide"
    run_name: Optional[str] = None
    metrics_csv: Optional[str] = None
    
    # Model-specific kwargs (passed to create_model)
    model_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate model name and configuration.
        
        Raises:
            ValueError: If model not recognized
        """
        available = get_available_models()
        resolved_model = self.resolve_model_name()
        
        if resolved_model not in available:
            logger.warning(
                f"Model '{self.model}' (resolved: {resolved_model}) not in registry. "
                f"Available: {available}"
            )

    def resolve_model_name(self) -> str:
        """Resolve model name through aliases.
        
        Returns:
            Actual model name in registry
        """
        return MODEL_ALIASES.get(self.model, self.model)

    def model_kwargs(self) -> Dict[str, Any]:
        """Get model creation kwargs including custom config.
        
        Returns:
            Dictionary to pass to create_model(**kwargs)
        """
        return self.model_config.copy()

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        """Load config from JSON file.
        
        Args:
            path: Path to JSON config file
            
        Returns:
            TrainConfig instance
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        cfg = TrainConfig(**data)
        cfg.validate()
        return cfg

    def to_json(self, path: str) -> None:
        """Save config to JSON file.
        
        Args:
            path: Output JSON path
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved config to {path}")

    def merge_overrides(self, overrides: Dict[str, Any]) -> "TrainConfig":
        """Merge command-line overrides.
        
        Args:
            overrides: Dict of values to override
            
        Returns:
            Self (for chaining)
        """
        for k, v in overrides.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        self.validate()
        return self

    def resolve_paths(self) -> None:
        """Create output directories.
        
        Creates log_dir and resolves metrics_csv path.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        if self.metrics_csv is None and self.run_name:
            self.metrics_csv = os.path.join(self.log_dir, f"{self.run_name}_metrics.csv")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"TrainConfig(model={self.model}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, lr={self.lr}, device={self.device})"
        )
