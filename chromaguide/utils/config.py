"""Configuration management using OmegaConf."""
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load YAML configuration with optional CLI overrides.
    
    Priority (highest → lowest):
        1. CLI overrides (dot notation, e.g., model.sequence_encoder.type=caduceus)
        2. Experiment config (if provided)
        3. Default config (chromaguide/configs/default.yaml)
    
    Args:
        config_path: Path to a YAML config file (overrides defaults).
        overrides: List of "key=value" strings for CLI overrides.
    
    Returns:
        Merged OmegaConf DictConfig.
    """
    # Locate default config
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    
    base_cfg = OmegaConf.load(default_path)
    
    # Merge experiment-specific config
    if config_path is not None:
        exp_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, exp_cfg)
    
    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    # Resolve any interpolations
    OmegaConf.resolve(base_cfg)
    
    return base_cfg


def save_config(cfg: DictConfig, path: str) -> None:
    """Save a config to YAML for experiment reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
