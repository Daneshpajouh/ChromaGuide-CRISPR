"""ChromaGuide Training Configuration"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Supported model architectures
SUPPORTED_MODELS = [
    'chromaguide',
    'dnabert_mamba',
    'dnabert_only',
    'mamba_only',
    'baseline_cnn_bilstm',
    'hybrid_dnabert_mamba'
]

DEFAULT_MODEL = 'chromaguide'

@dataclass
class TrainingConfig:
    """Training configuration for ChromaGuide models"""
    # Model settings
    model_type: str = DEFAULT_MODEL
    seq_len: int = 23
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    patience: int = 10
    
    # Data settings
    data_path: str = 'data/crispr_data.csv'
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Output settings
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # Device settings
    device: str = 'auto'
    num_workers: int = 4
    
    def validate(self):
        """Validate configuration"""
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_type}")
        if self.train_split + self.val_split + self.test_split != 1.0:
            raise ValueError("Data splits must sum to 1.0")
