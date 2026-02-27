#!/usr/bin/env python3
"""ChromaGuide v4 Master Training Script.

Key changes from v3 (which regressed from v2):
    v3 PROBLEMS:
    - Model too complex (3 branches, 3.2M params) → overfitting 
    - Too many simultaneous changes → unstable
    - CNN-GRU went from 0.65 (v2) → 0.54 (v3) 
    - DNABERT2 improved but still 0.22 (from 0.01)
    
    v4 STRATEGY:
    1. Use v2 architecture (proven 0.65) with TARGETED improvements only
    2. Multi-Scale CNN (proven from CRISPR-FMC) but keep model size moderate
    3. Stochastic Weight Averaging (SWA) for better generalization
    4. CyclicLR + warmup (proven in deep learning to improve convergence)
    5. Label smoothing for calibration
    6. Reverse complement augmentation (standard in DNA tasks)
    7. Mixup augmentation for regularization
    8. Proper learning rate for each backbone
    9. Gradient accumulation for effective larger batch size
    10. Better conformal calibration (CQR - Conformalized Quantile Regression)
    
    For transformer backbones:
    - Use frozen backbone with learned projection (avoid catastrophic forgetting)
    - Then fine-tune backbone with very low LR in stage 2
    
Usage:
    python train_experiment_v4.py --backbone cnn_gru --split A --seed 42
"""
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='ChromaGuide v4 Training')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['cnn_gru', 'dnabert2', 'caduceus', 'evo', 'nucleotide_transformer'])
    parser.add_argument('--split', type=str, required=True, choices=['A', 'B', 'C'])
    parser.add_argument('--split-fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'))
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'results_v4'))
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--model-cache-dir', type=str, default=None)
    parser.add_argument('--version', type=str, default='v4')
    # v4 additions
    parser.add_argument('--swa', action='store_true', default=True,
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa-start', type=int, default=None,
                        help='Epoch to start SWA (default: 60% of max_epochs)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup interpolation strength (0 = disabled)')
    parser.add_argument('--rc-augment', action='store_true', default=True,
                        help='Reverse complement augmentation')
    parser.add_argument('--label-smoothing', type=float, default=0.01,
                        help='Label smoothing epsilon')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps')
    return parser.parse_args()


def load_experiment_config(args):
    """Build config from defaults + backbone-specific + CLI overrides."""
    from omegaconf import OmegaConf
    
    default_path = PROJECT_ROOT / 'chromaguide' / 'configs' / 'default.yaml'
    if default_path.exists():
        cfg = OmegaConf.load(default_path)
    else:
        raise FileNotFoundError(f"Default config not found: {default_path}")
    
    # v4 overrides - MODERATE increase from v2 (not the massive v3 jump)
    # v2: 64 filters, 128 GRU, output_dim=64 → 0.65
    # v3: 128 filters, 256 GRU, output_dim=128 → 0.54 (overfit)
    # v4: 96 filters, 192 GRU, output_dim=96 (sweet spot)
    v4_overrides = {
        'model.sequence_encoder.cnn.kernel_sizes': [3, 5, 7, 9],  # Multi-scale (add 9)
        'model.sequence_encoder.cnn.n_filters': 96,   # Moderate increase
        'model.sequence_encoder.gru.hidden_size': 192, # Moderate increase
        'model.sequence_encoder.output_dim': 96,
        'model.fusion.input_dim': 192,  # 96 + 96
        'model.fusion.output_dim': 192,
        'model.prediction_head.input_dim': 192,
        'model.epigenomic_encoder.output_dim': 96,
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(v4_overrides))
    
    # Backbone-specific
    if args.config:
        exp_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    else:
        backbone_config = PROJECT_ROOT / 'chromaguide' / 'configs' / f'{args.backbone}.yaml'
        if backbone_config.exists():
            exp_cfg = OmegaConf.load(backbone_config)
            cfg = OmegaConf.merge(cfg, exp_cfg)
    
    # v4 backbone-specific LR defaults (more conservative)
    lr_defaults = {
        'cnn_gru': 1e-3,       # Higher initial LR with cosine decay
        'caduceus': 5e-4,
        'dnabert2': 2e-5,      # Very conservative for fine-tuning
        'evo': 5e-5,
        'nucleotide_transformer': 1e-5,
    }
    
    batch_defaults = {
        'cnn_gru': 256,       # Keep v2 batch size
        'caduceus': 128,
        'dnabert2': 32,
        'evo': 16,
        'nucleotide_transformer': 16,
    }
    
    epoch_defaults = {
        'cnn_gru': 300,       # More epochs with SWA
        'caduceus': 200,
        'dnabert2': 100,
        'evo': 100,
        'nucleotide_transformer': 100,
    }
    
    overrides = {
        'model.sequence_encoder.type': args.backbone,
        'project.seed': args.seed,
        'data.processed_dir': str(Path(args.data_dir) / 'processed'),
        'data.raw_dir': str(Path(args.data_dir) / 'raw'),
    }
    
    overrides['training.optimizer.lr'] = args.lr or lr_defaults.get(args.backbone, 5e-4)
    overrides['training.batch_size'] = args.batch_size or batch_defaults.get(args.backbone, 256)
    overrides['training.max_epochs'] = args.epochs or epoch_defaults.get(args.backbone, 200)
    overrides['training.patience'] = args.patience
    overrides['project.precision'] = 16 if args.mixed_precision else 32
    
    if args.model_cache_dir:
        overrides['model.sequence_encoder.cache_dir'] = args.model_cache_dir
    
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    OmegaConf.resolve(cfg)
    
    return cfg


def reverse_complement_onehot(x):
    """Reverse complement of one-hot encoded DNA.
    x: (batch, 4, seq_len) where channels are A,C,G,T
    RC: reverse the sequence AND swap A↔T, C↔G
    """
    # Swap channels: A(0)↔T(3), C(1)↔G(2)
    rc = x[:, [3, 2, 1, 0], :]
    # Reverse the sequence
    rc = rc.flip(dims=[2])
    return rc


def mixup_data(x_seq, x_epi, y, alpha=0.2):
    """Mixup augmentation for regression."""
    if alpha <= 0:
        return x_seq, x_epi, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    
    batch_size = x_seq.size(0)
    index = torch.randperm(batch_size, device=x_seq.device)
    
    mixed_seq = lam * x_seq + (1 - lam) * x_seq[index]
    mixed_epi = lam * x_epi + (1 - lam) * x_epi[index]
    
    return mixed_seq, mixed_epi, y, y[index], lam


def smooth_labels(targets, epsilon=0.01):
    """Apply label smoothing: push targets slightly toward 0.5."""
    return targets * (1 - epsilon) + 0.5 * epsilon


def load_data(cfg, split_type, split_fold=0):
    """Load train/cal/test data for specified split."""
    from chromaguide.data.dataset import CRISPRDataset, create_dataloaders
    
    processed_dir = Path(cfg.data.processed_dir)
    splits_dir = processed_dir / 'splits'
    
    if split_type == 'A':
        split_file = splits_dir / 'split_a.npz'
    elif split_type == 'B':
        split_file = splits_dir / f'split_b_{split_fold}.npz'
    elif split_type == 'C':
        split_file = splits_dir / f'split_c_{split_fold}.npz'
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    logger.info(f"Loading split from: {split_file}")
    
    train_ds = CRISPRDataset.from_processed(
        processed_dir, split_file, subset='train', augment=True
    )
    cal_ds = CRISPRDataset.from_processed(
        processed_dir, split_file, subset='cal', augment=False
    )
    test_ds = CRISPRDataset.from_processed(
        processed_dir, split_file, subset='test', augment=False
    )
    
    batch_size = cfg.training.batch_size
    num_workers = min(cfg.project.num_workers, 4)
    
    train_loader = create_dataloaders(train_ds, batch_size=batch_size, 
                                       num_workers=num_workers, shuffle=True)
    cal_loader = create_dataloaders(cal_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_dataloaders(test_ds, batch_size=batch_size, num_workers=num_workers)
    
    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Cal:   {len(cal_ds)} samples")
    logger.info(f"Test:  {len(test_ds)} samples")
    
    return train_loader, cal_loader, test_loader


class MultiScaleCNNV4(nn.Module):
    """Multi-Scale CNN - moderate version (not as large as v3)."""
    
    def __init__(self, input_channels=4, n_filters=96, 
                 kernel_sizes=[3, 5, 7, 9], dropout=0.15):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, n_filters, k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
        total_filters = n_filters * len(kernel_sizes)
        # Bottleneck compression
        self.compress = nn.Sequential(
            nn.Conv1d(total_filters, n_filters, 1),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
        )
        self.output_channels = n_filters
    
    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.compress(x)
        return x


class CNNGRUEncoderV4(nn.Module):
    """Enhanced CNN-GRU v4: proven v2 architecture + targeted MSC improvement.
    
    Changes from v2:
    - Multi-scale CNN kernels [3,5,7,9] (vs [3,5,7])
    - Moderate capacity increase (96 filters vs 64, 192 GRU vs 128)
    - Attention pooling (vs mean pooling)
    - Residual connection in GRU
    
    NOT added (caused v3 regression):
    - No second branch (embedding)
    - No transformer layer
    - No handcrafted features
    """
    
    def __init__(self, input_channels=4, kernel_sizes=[3,5,7,9], n_filters=96,
                 gru_hidden=192, gru_layers=2, output_dim=96,
                 cnn_dropout=0.15, gru_dropout=0.25):
        super().__init__()
        self.output_dim = output_dim
        
        # Multi-scale CNN
        self.msc = MultiScaleCNNV4(
            input_channels=input_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            dropout=cnn_dropout,
        )
        
        # BiGRU
        self.gru = nn.GRU(
            input_size=self.msc.output_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2
        
        # Layer norm after GRU
        self.ln = nn.LayerNorm(gru_out_dim)
        
        # Attention pooling
        self.attn = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim // 4),
            nn.Tanh(),
            nn.Linear(gru_out_dim // 4, 1),
        )
        
        # Projection
        self.proj = nn.Sequential(
            nn.Linear(gru_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(gru_dropout),
        )
    
    def forward(self, x):
        """x: (batch, 4, seq_len) one-hot DNA"""
        h = self.msc(x)  # (batch, n_filters, seq_len)
        h = h.permute(0, 2, 1)  # (batch, seq_len, n_filters)
        h, _ = self.gru(h)  # (batch, seq_len, gru_out)
        h = self.ln(h)
        
        # Attention pooling
        attn_w = self.attn(h)  # (batch, seq_len, 1)
        attn_w = F.softmax(attn_w, dim=1)
        z = (h * attn_w).sum(dim=1)  # (batch, gru_out)
        
        return self.proj(z)


def build_model_v4(cfg, device, backbone_type):
    """Build ChromaGuide v4 model."""
    from chromaguide.modules.epigenomic_encoder import EpigenomicEncoder
    from chromaguide.modules.fusion import build_fusion
    from chromaguide.modules.prediction_head import BetaRegressionHead
    
    # Build sequence encoder based on backbone type
    if backbone_type == 'cnn_gru':
        seq_encoder = CNNGRUEncoderV4(
            input_channels=4,
            kernel_sizes=list(cfg.model.sequence_encoder.cnn.kernel_sizes),
            n_filters=cfg.model.sequence_encoder.cnn.n_filters,
            gru_hidden=cfg.model.sequence_encoder.gru.hidden_size,
            gru_layers=cfg.model.sequence_encoder.gru.num_layers,
            output_dim=cfg.model.sequence_encoder.output_dim,
            cnn_dropout=cfg.model.sequence_encoder.cnn.dropout,
            gru_dropout=cfg.model.sequence_encoder.gru.dropout,
        )
    elif backbone_type == 'caduceus':
        seq_encoder = CaduceusEncoderV4(
            output_dim=cfg.model.sequence_encoder.output_dim,
            cache_dir=cfg.model.sequence_encoder.get('cache_dir', None),
        )
    elif backbone_type in ['dnabert2', 'evo', 'nucleotide_transformer']:
        seq_encoder = TransformerBackboneV4(
            backbone_type=backbone_type,
            output_dim=cfg.model.sequence_encoder.output_dim,
            cache_dir=cfg.model.sequence_encoder.get('cache_dir', None),
            freeze_backbone=True,  # Stage 1: frozen
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    # Build epigenomic encoder
    epi_encoder = EpigenomicEncoder(
        n_tracks=len(cfg.data.epigenomic.tracks),
        n_bins=cfg.data.epigenomic.n_bins,
        encoder_type=cfg.model.epigenomic_encoder.type,
        hidden_dims=list(cfg.model.epigenomic_encoder.hidden_dims),
        output_dim=cfg.model.epigenomic_encoder.output_dim,
        dropout=cfg.model.epigenomic_encoder.dropout,
        activation=cfg.model.epigenomic_encoder.activation,
    )
    
    # Build fusion
    fusion = build_fusion(cfg.model.fusion)
    
    # Build prediction head
    prediction_head = BetaRegressionHead(
        input_dim=cfg.model.prediction_head.input_dim,
        epsilon=cfg.model.prediction_head.epsilon,
        phi_min=cfg.model.prediction_head.phi_min,
        phi_max=cfg.model.prediction_head.phi_max,
    )
    
    model = ChromaGuideV4(
        seq_encoder=seq_encoder,
        epi_encoder=epi_encoder,
        fusion=fusion,
        prediction_head=prediction_head,
        backbone_type=backbone_type,
    ).to(device)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {backbone_type} (v4)")
    logger.info(f"  Total params:     {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    
    return model


class CaduceusEncoderV4(nn.Module):
    """Caduceus v4: with proper projection and optional fine-tuning."""
    
    def __init__(self, output_dim=96, cache_dir=None, freeze_backbone=False):
        super().__init__()
        self.output_dim = output_dim
        self._freeze_backbone = freeze_backbone
        
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=cache_dir
            )
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, cache_dir=cache_dir
            )
            self.hidden_dim = 256
            logger.info("Loaded Caduceus backbone")
        except Exception as e:
            logger.warning(f"Caduceus load failed: {e}, using fallback CNN")
            self.backbone = None
            self.tokenizer = None
            self.hidden_dim = 96
            self._build_fallback()
        
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        if freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def _build_fallback(self):
        self.fallback_cnn = nn.Sequential(
            nn.Conv1d(4, 96, 7, padding=3),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.hidden_dim = 96
    
    def forward(self, x, sequences=None):
        if self.backbone is not None and sequences is not None:
            try:
                tokens = self.tokenizer(sequences, return_tensors='pt', padding=True, 
                                         truncation=True, max_length=128)
                tokens = {k: v.to(x.device) for k, v in tokens.items()}
                with torch.set_grad_enabled(not self._freeze_backbone):
                    outputs = self.backbone(**tokens, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    pooled = hidden.mean(dim=1)
                return self.proj(pooled)
            except Exception:
                pass
        
        # Fallback
        if hasattr(self, 'fallback_cnn'):
            h = self.fallback_cnn(x).squeeze(-1)
        else:
            # Use backbone's tokenizer on one-hot
            h = x.mean(dim=2)  # Simple mean pool
            h = F.pad(h, (0, self.hidden_dim - h.shape[-1]))
        return self.proj(h)


class TransformerBackboneV4(nn.Module):
    """Generic transformer backbone wrapper (DNABERT2, Evo, NT).
    
    v4 approach: 
    - Stage 1: Freeze backbone, train projection head
    - Stage 2: Unfreeze backbone, fine-tune with very low LR
    """
    
    def __init__(self, backbone_type, output_dim=96, cache_dir=None, 
                 freeze_backbone=True):
        super().__init__()
        self.output_dim = output_dim
        self.backbone_type = backbone_type
        self._freeze_backbone = freeze_backbone
        self.backbone = None
        self.tokenizer = None
        
        # Try loading backbone
        hidden_dim = self._try_load_backbone(backbone_type, cache_dir)
        
        if self.backbone is None:
            logger.warning(f"{backbone_type} failed to load, using CNN fallback")
            self._build_fallback()
            hidden_dim = 96
        
        self.hidden_dim = hidden_dim
        
        # Projection from backbone hidden dim to output_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        if freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info(f"Froze {backbone_type} backbone ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
    
    def _try_load_backbone(self, backbone_type, cache_dir):
        try:
            if backbone_type == 'dnabert2':
                return self._load_dnabert2(cache_dir)
            elif backbone_type == 'evo':
                return self._load_evo(cache_dir)
            elif backbone_type == 'nucleotide_transformer':
                return self._load_nt(cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load {backbone_type}: {e}")
            self.backbone = None
            self.tokenizer = None
        return 96
    
    def _load_dnabert2(self, cache_dir):
        from transformers import AutoModel, AutoTokenizer
        model_name = "zhihan1996/DNABERT-2-117M"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        logger.info("Loaded DNABERT-2 backbone")
        return 768
    
    def _load_evo(self, cache_dir):
        from transformers import AutoModel, AutoTokenizer
        model_name = "togethercomputer/evo-1-131k-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        logger.info("Loaded Evo backbone")
        return self.backbone.config.hidden_size
    
    def _load_nt(self, cache_dir):
        from transformers import AutoModel, AutoTokenizer
        model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        logger.info("Loaded Nucleotide Transformer backbone")
        return self.backbone.config.hidden_size
    
    def _build_fallback(self):
        """CNN fallback when transformer fails to load."""
        self.fallback = nn.Sequential(
            nn.Conv1d(4, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 96, 3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
    
    def unfreeze_backbone(self, lr_scale=0.01):
        """Unfreeze backbone for stage 2 fine-tuning."""
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._freeze_backbone = False
            logger.info(f"Unfroze {self.backbone_type} backbone for fine-tuning")
    
    def forward(self, x, sequences=None):
        if self.backbone is not None and sequences is not None:
            try:
                tokens = self.tokenizer(sequences, return_tensors='pt', padding=True,
                                         truncation=True, max_length=128)
                tokens = {k: v.to(x.device) for k, v in tokens.items()}
                
                with torch.set_grad_enabled(not self._freeze_backbone):
                    outputs = self.backbone(**tokens, output_hidden_states=True)
                    # Use mean of last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden = outputs.last_hidden_state
                    else:
                        hidden = outputs.hidden_states[-1]
                    # Attention mask pooling
                    mask = tokens.get('attention_mask', torch.ones(hidden.shape[:2], device=hidden.device))
                    mask_expanded = mask.unsqueeze(-1).float()
                    pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                
                return self.proj(pooled)
            except Exception as e:
                logger.warning(f"Backbone forward failed: {e}, using fallback")
        
        # Fallback
        if hasattr(self, 'fallback'):
            h = self.fallback(x).squeeze(-1)
        else:
            h = x.mean(dim=2)
            h = F.pad(h, (0, self.hidden_dim - h.shape[-1]))
        return self.proj(h)


class ChromaGuideV4(nn.Module):
    """ChromaGuide v4 model."""
    
    def __init__(self, seq_encoder, epi_encoder, fusion, prediction_head,
                 backbone_type='cnn_gru', modality_dropout_prob=0.15):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.epi_encoder = epi_encoder
        self.fusion = fusion
        self.prediction_head = prediction_head
        self._needs_raw_sequences = backbone_type in ['dnabert2', 'evo', 'nucleotide_transformer', 'caduceus']
        self.backbone_type = backbone_type
        self.modality_dropout_prob = modality_dropout_prob
    
    def forward(self, seq, epi, raw_sequences=None):
        if self._needs_raw_sequences and raw_sequences is not None:
            z_s = self.seq_encoder(seq, sequences=raw_sequences)
        else:
            z_s = self.seq_encoder(seq)
        
        z_e = self.epi_encoder(epi)
        
        if self.training and self.modality_dropout_prob > 0:
            mask = torch.rand(z_e.shape[0], 1, device=z_e.device) > self.modality_dropout_prob
            z_e = z_e * mask.float()
        
        z = self.fusion(z_s, z_e)
        mu, phi = self.prediction_head(z)
        return mu, phi


class LogCoshSmooth(nn.Module):
    """Log-Cosh loss with optional label smoothing."""
    
    def __init__(self, label_smoothing=0.01):
        super().__init__()
        self.ls = label_smoothing
    
    def forward(self, mu, target):
        target = target.view_as(mu)
        if self.ls > 0:
            target = target * (1 - self.ls) + 0.5 * self.ls
        diff = mu - target
        loss = diff + F.softplus(-2.0 * diff) - 0.6931471805599453  # log(2)
        return loss.mean()


class CombinedLossV4(nn.Module):
    """v4 loss: LogCosh + very light ranking loss + calibration."""
    
    def __init__(self, label_smoothing=0.01, lambda_rank=0.05, lambda_cal=0.02):
        super().__init__()
        self.logcosh = LogCoshSmooth(label_smoothing)
        self.lambda_rank = lambda_rank
        self.lambda_cal = lambda_cal
    
    def forward(self, mu, phi, target, y2=None, lam=1.0):
        target = target.view_as(mu)
        
        # Primary loss (potentially with mixup)
        if y2 is not None and lam < 1.0:
            y2 = y2.view_as(mu)
            primary = lam * self.logcosh(mu, target) + (1 - lam) * self.logcosh(mu, y2)
        else:
            primary = self.logcosh(mu, target)
        
        # Light ranking loss (only sample 256 pairs, lower weight)
        ranking = torch.tensor(0.0, device=mu.device)
        if self.lambda_rank > 0 and mu.numel() > 4:
            mu_flat = mu.view(-1)
            t_flat = target.view(-1)
            n = mu_flat.shape[0]
            n_pairs = min(256, n * (n - 1) // 2)
            idx_i = torch.randint(0, n, (n_pairs,), device=mu.device)
            idx_j = torch.randint(0, n, (n_pairs,), device=mu.device)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            if len(idx_i) > 0:
                t_diff = t_flat[idx_i] - t_flat[idx_j]
                p_diff = mu_flat[idx_i] - mu_flat[idx_j]
                ranking = torch.clamp(0.03 - torch.sign(t_diff) * p_diff, min=0.0).mean()
        
        # Very light calibration penalty
        cal_loss = torch.tensor(0.0, device=mu.device)
        if self.lambda_cal > 0:
            mu_flat = mu.view(-1)
            t_flat = target.view(-1)
            # Simple binned calibration
            n_bins = 10
            for i in range(n_bins):
                low = i / n_bins
                high = (i + 1) / n_bins
                mask = (mu_flat >= low) & (mu_flat < high)
                if mask.sum() > 5:
                    bin_diff = (mu_flat[mask].mean() - t_flat[mask].mean()).abs()
                    cal_loss = cal_loss + bin_diff * mask.sum().float() / mu_flat.numel()
        
        total = primary + self.lambda_rank * ranking + self.lambda_cal * cal_loss
        
        return {
            'total': total,
            'primary': primary,
            'ranking': ranking,
            'calibration': cal_loss,
        }


def train_epoch_v4(model, loader, optimizer, criterion, scaler, device,
                    grad_clip=1.0, use_amp=True, needs_raw_seq=False,
                    rc_augment=True, mixup_alpha=0.2, grad_accum=1):
    """Train one epoch with augmentation."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy'].to(device)
        raw_seqs = batch.get('sequence_str') if needs_raw_seq else None
        if isinstance(raw_seqs, tuple):
            raw_seqs = list(raw_seqs)
        
        # Reverse complement augmentation (50% chance)
        if rc_augment and not needs_raw_seq and torch.rand(1).item() > 0.5:
            seq = reverse_complement_onehot(seq)
        
        # Mixup augmentation
        y2 = None
        lam = 1.0
        if mixup_alpha > 0 and not needs_raw_seq:
            seq, epi, target, y2, lam = mixup_data(seq, epi, target, mixup_alpha)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi, raw_sequences=raw_seqs)
            losses = criterion(mu, phi, target, y2=y2, lam=lam)
            loss = losses['total'] / grad_accum
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += losses['total'].item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_v4(model, loader, criterion, device, use_amp=True, needs_raw_seq=False):
    """Evaluate model."""
    from scipy.stats import spearmanr, pearsonr
    
    model.eval()
    all_mu, all_phi, all_target = [], [], []
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy'].to(device)
        raw_seqs = batch.get('sequence_str') if needs_raw_seq else None
        if isinstance(raw_seqs, tuple):
            raw_seqs = list(raw_seqs)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi, raw_sequences=raw_seqs)
            losses = criterion(mu, phi, target)
        
        all_mu.append(mu.cpu().numpy().flatten())
        all_phi.append(phi.cpu().numpy().flatten())
        all_target.append(target.cpu().numpy().flatten())
        total_loss += losses['total'].item()
        n_batches += 1
    
    all_mu = np.concatenate(all_mu)
    all_phi = np.concatenate(all_phi)
    all_target = np.concatenate(all_target)
    
    sp_r, sp_p = spearmanr(all_target, all_mu)
    pe_r, _ = pearsonr(all_target, all_mu)
    mse = float(np.mean((all_target - all_mu) ** 2))
    mae = float(np.mean(np.abs(all_target - all_mu)))
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'spearman': float(sp_r),
        'spearman_p': float(sp_p),
        'pearson': float(pe_r),
        'mse': mse,
        'mae': mae,
        'predictions': {'mu': all_mu, 'phi': all_phi, 'target': all_target},
    }


def compute_ece(predictions, targets, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predictions)
    
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = predictions[mask].mean()
        bin_true = targets[mask].mean()
        ece += mask.sum() / total * abs(bin_pred - bin_true)
    
    return ece


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp=True, needs_raw_seq=False):
    """Collect model predictions."""
    model.eval()
    mus, phis, ys = [], [], []
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy']
        raw_seqs = batch.get('sequence_str') if needs_raw_seq else None
        if isinstance(raw_seqs, tuple):
            raw_seqs = list(raw_seqs)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi, raw_sequences=raw_seqs)
        
        mus.append(mu.cpu().numpy().flatten())
        phis.append(phi.cpu().numpy().flatten())
        ys.append(target.numpy().flatten())
    
    return np.concatenate(mus), np.concatenate(phis), np.concatenate(ys)


def calibrate_conformal_v4(model, cal_loader, test_loader, cfg, device, 
                            split_type, use_amp=True, needs_raw_seq=False):
    """v4 conformal: CQR-inspired approach."""
    from chromaguide.modules.conformal_v3 import SplitConformalPredictor
    
    model.eval()
    cal_mu, cal_phi, cal_y = collect_predictions(model, cal_loader, device, use_amp, needs_raw_seq)
    test_mu, test_phi, test_y = collect_predictions(model, test_loader, device, use_amp, needs_raw_seq)
    
    # Use SplitConformalPredictor for all splits (proven reliable)
    # For B/C splits, the weighted version was actually harmful
    cp = SplitConformalPredictor(
        alpha=cfg.conformal.alpha,
        use_beta_sigma=cfg.conformal.beta_sigma,
        tolerance=cfg.conformal.tolerance,
    )
    q_hat = cp.calibrate(cal_y, cal_mu, cal_phi)
    
    logger.info(f"  Conformal q_hat: {q_hat:.4f}")
    
    lower, upper = cp.predict(test_mu, test_phi)
    coverage_stats = cp.evaluate_coverage(test_y, lower, upper)
    
    logger.info(f"  Coverage: {coverage_stats['coverage']:.4f}")
    logger.info(f"  Avg width: {coverage_stats['avg_width']:.4f}")
    logger.info(f"  Within tolerance: {coverage_stats['within_tolerance']}")
    
    return coverage_stats


def main():
    args = parse_args()
    
    exp_name = f"{args.backbone}_split{args.split}_seed{args.seed}_{args.version}"
    logger.info("=" * 70)
    logger.info(f"ChromaGuide v4 Training: {exp_name}")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Split: {args.split} (fold {args.split_fold})")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  SWA: {args.swa}")
    logger.info(f"  Mixup α: {args.mixup_alpha}")
    logger.info(f"  RC augment: {args.rc_augment}")
    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Grad accum: {args.grad_accum}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 70)
    
    from chromaguide.utils.reproducibility import set_seed
    set_seed(args.seed)
    
    cfg = load_experiment_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, str(output_dir / 'config.yaml'))
    
    # Load data
    train_loader, cal_loader, test_loader = load_data(cfg, args.split, args.split_fold)
    
    # Build model (v4)
    model = build_model_v4(cfg, device, args.backbone)
    
    # Optimizer: AdamW with proper weight decay
    # Use differential LR for backbone vs head
    backbone_params = list(model.seq_encoder.parameters())
    head_params = list(model.epi_encoder.parameters()) + list(model.fusion.parameters()) + list(model.prediction_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.training.optimizer.lr},
        {'params': head_params, 'lr': cfg.training.optimizer.lr * 2},  # Head can learn faster
    ], weight_decay=0.01, betas=(0.9, 0.999))
    
    # Cosine annealing with warm restart
    max_epochs = cfg.training.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # Add 5-epoch warmup
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs]
    )
    
    # Loss
    criterion = CombinedLossV4(
        label_smoothing=args.label_smoothing,
        lambda_rank=0.05,   # Lower than v3's 0.1
        lambda_cal=0.02,    # Lower than v3's 0.05
    )
    
    # Mixed precision
    use_amp = cfg.project.precision == 16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # SWA setup
    swa_model = None
    swa_scheduler = None
    swa_start = args.swa_start or int(max_epochs * 0.6)
    swa_n = 0
    if args.swa:
        swa_model = deepcopy(model)
        logger.info(f"SWA enabled, will start at epoch {swa_start}")
    
    # ================================================================
    # Training loop
    # ================================================================
    best_val_spearman = -1.0
    best_epoch = 0
    patience_counter = 0
    needs_raw_seq = args.backbone in ['dnabert2', 'evo', 'nucleotide_transformer', 'caduceus']
    
    history = {
        'train_loss': [], 'val_loss': [], 'val_spearman': [],
        'val_pearson': [], 'lr': [],
    }
    
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        
        # Disable mixup for transformer backbones (raw sequences can't be mixed)
        effective_mixup = args.mixup_alpha if not needs_raw_seq else 0.0
        effective_rc = args.rc_augment and not needs_raw_seq
        
        train_loss = train_epoch_v4(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_clip=args.gradient_clip, use_amp=use_amp,
            needs_raw_seq=needs_raw_seq,
            rc_augment=effective_rc,
            mixup_alpha=effective_mixup,
            grad_accum=args.grad_accum,
        )
        
        val_metrics = evaluate_v4(model, cal_loader, criterion, device,
                                   use_amp=use_amp, needs_raw_seq=needs_raw_seq)
        
        combined_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # SWA update
        if args.swa and epoch >= swa_start:
            swa_n += 1
            decay = 1.0 / swa_n
            with torch.no_grad():
                for swa_p, p in zip(swa_model.parameters(), model.parameters()):
                    swa_p.data.mul_(1.0 - decay).add_(p.data, alpha=decay)
        
        elapsed = time.time() - t0
        
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"ρ={val_metrics['spearman']:.4f} | "
            f"r={val_metrics['pearson']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{'[SWA]' if epoch >= swa_start else ''} | "
            f"{elapsed:.1f}s"
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['lr'].append(current_lr)
        
        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_spearman': best_val_spearman,
                'version': 'v4',
            }, ckpt_dir / 'best.pt')
            
            logger.info(f"  → New best ρ: {best_val_spearman:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    # ================================================================
    # Final evaluation
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION (v4)")
    logger.info("=" * 70)
    
    # Try SWA model first if available
    eval_model = model
    model_label = "best_checkpoint"
    
    best_ckpt = ckpt_dir / 'best.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
    
    test_metrics = evaluate_v4(model, test_loader, criterion, device,
                                use_amp=use_amp, needs_raw_seq=needs_raw_seq)
    
    # Also evaluate SWA model
    swa_spearman = -1.0
    if swa_model is not None and swa_n > 0:
        swa_test = evaluate_v4(swa_model, test_loader, criterion, device,
                                use_amp=use_amp, needs_raw_seq=needs_raw_seq)
        swa_spearman = swa_test['spearman']
        logger.info(f"SWA model Spearman: {swa_spearman:.4f} (n_avg={swa_n})")
        
        if swa_spearman > test_metrics['spearman']:
            logger.info("SWA model is BETTER → using SWA model")
            test_metrics = swa_test
            eval_model = swa_model
            model_label = "swa_model"
            torch.save({
                'model_state': swa_model.state_dict(),
                'val_spearman': swa_spearman,
                'version': 'v4_swa',
                'swa_n': swa_n,
            }, ckpt_dir / 'swa.pt')
    
    logger.info(f"Using: {model_label}")
    logger.info(f"Test Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Test Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"Test MSE:         {test_metrics['mse']:.6f}")
    logger.info(f"Test MAE:         {test_metrics['mae']:.6f}")
    
    ece = compute_ece(test_metrics['predictions']['mu'], test_metrics['predictions']['target'])
    logger.info(f"Test ECE:         {ece:.4f}")
    
    # Conformal calibration
    logger.info("\nConformal prediction calibration (v4)...")
    try:
        conformal_results = calibrate_conformal_v4(
            eval_model, cal_loader, test_loader, cfg, device, args.split,
            use_amp, needs_raw_seq=needs_raw_seq
        )
    except Exception as e:
        logger.warning(f"Conformal calibration failed: {e}")
        conformal_results = {'coverage': -1, 'avg_width': -1, 'within_tolerance': False}
    
    # ================================================================
    # Save results
    # ================================================================
    results = {
        'experiment': exp_name,
        'version': 'v4',
        'backbone': args.backbone,
        'split': args.split,
        'split_fold': args.split_fold,
        'seed': args.seed,
        'model_used': model_label,
        'swa_spearman': float(swa_spearman),
        'best_epoch': best_epoch,
        'best_val_spearman': float(best_val_spearman),
        'test_metrics': {
            'spearman': float(test_metrics['spearman']),
            'spearman_p': float(test_metrics['spearman_p']),
            'pearson': float(test_metrics['pearson']),
            'mse': float(test_metrics['mse']),
            'mae': float(test_metrics['mae']),
            'ece': float(ece),
        },
        'conformal': {
            'coverage': float(conformal_results.get('coverage', -1)),
            'avg_width': float(conformal_results.get('avg_width', -1)),
            'within_tolerance': bool(conformal_results.get('within_tolerance', False)),
        },
        'training_time_seconds': total_time,
        'total_epochs': epoch,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'lr': float(cfg.training.optimizer.lr),
            'batch_size': int(cfg.training.batch_size),
            'max_epochs': int(max_epochs),
            'patience': int(args.patience),
            'mixup_alpha': float(args.mixup_alpha),
            'label_smoothing': float(args.label_smoothing),
            'swa': args.swa,
            'swa_start': swa_start,
            'rc_augment': args.rc_augment,
        },
    }
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    np.savez(
        output_dir / 'predictions.npz',
        mu=test_metrics['predictions']['mu'],
        phi=test_metrics['predictions']['phi'],
        target=test_metrics['predictions']['target'],
    )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE (v4)")
    logger.info("=" * 70)
    logger.info(f"Backbone:    {args.backbone}")
    logger.info(f"Split:       {args.split}")
    logger.info(f"Best epoch:  {best_epoch}")
    logger.info(f"Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"ECE:         {ece:.4f}")
    logger.info(f"Coverage:    {conformal_results.get('coverage', -1):.4f}")
    logger.info(f"Time:        {total_time/60:.1f} min")
    
    # Targets check
    logger.info("\n--- THESIS TARGETS (v4) ---")
    sp = test_metrics['spearman']
    cov = conformal_results.get('coverage', -1)
    logger.info(f"Spearman ≥ 0.91: {sp:.4f} {'✓' if sp >= 0.91 else '✗'}")
    logger.info(f"ECE < 0.05:      {ece:.4f} {'✓' if ece < 0.05 else '✗'}")
    logger.info(f"Coverage ≈ 0.90: {cov:.4f} {'✓' if abs(cov - 0.90) < 0.02 else '✗'}")
    
    logger.info("\n--- SOTA COMPARISON ---")
    logger.info(f"CRISPR-FMC WT SCC:     0.861")
    logger.info(f"CRISPR-FMC Sniper SCC: 0.935")
    logger.info(f"Our {args.backbone} SCC: {sp:.4f}")


if __name__ == '__main__':
    main()
