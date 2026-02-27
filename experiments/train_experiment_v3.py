#!/usr/bin/env python3
"""ChromaGuide v3 Master Training Script.

Major improvements over v2:
    1. Log-Cosh loss + pairwise ranking loss (from CRISPR-FMC SOTA)
    2. Multi-scale CNN + Transformer + BiGRU architecture
    3. Dual-branch encoding with handcrafted features
    4. Weighted conformal prediction for domain-shift splits
    5. Ensemble support (3 seeds)
    6. Proper transformer backbone handling with fallbacks
    7. Adamax optimizer (from CRISPR-FMC)
    8. Differential learning rates

Usage:
    python train_experiment_v3.py --backbone cnn_gru --split A --seed 42
    python train_experiment_v3.py --backbone dnabert2 --split A --seed 42
    python train_experiment_v3.py --backbone caduceus --split B --split-fold 0 --seed 42
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
    parser = argparse.ArgumentParser(description='ChromaGuide v3 Training')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['cnn_gru', 'dnabert2', 'caduceus', 'evo', 'nucleotide_transformer'],
                        help='Sequence encoder backbone')
    parser.add_argument('--split', type=str, required=True,
                        choices=['A', 'B', 'C'],
                        help='Split type (A=gene-held-out, B=dataset-held-out, C=cell-line-held-out)')
    parser.add_argument('--split-fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'))
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'results_v3'))
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    # V3 additions
    parser.add_argument('--loss-type', type=str, default='logcosh',
                        choices=['logcosh', 'beta_nll', 'mse'],
                        help='Primary loss function')
    parser.add_argument('--lambda-rank', type=float, default=0.1,
                        help='Ranking loss weight')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        choices=['adamax', 'adamw', 'adam'],
                        help='Optimizer (adamax from CRISPR-FMC SOTA)')
    parser.add_argument('--model-cache-dir', type=str, default=None,
                        help='Cache directory for pretrained models')
    parser.add_argument('--version', type=str, default='v3')
    return parser.parse_args()


def load_experiment_config(args):
    """Build config from defaults + backbone-specific + CLI overrides."""
    from omegaconf import OmegaConf
    
    default_path = PROJECT_ROOT / 'chromaguide' / 'configs' / 'default.yaml'
    if default_path.exists():
        cfg = OmegaConf.load(default_path)
    else:
        raise FileNotFoundError(f"Default config not found: {default_path}")
    
    # V3 config overrides
    v3_overrides = {
        'model.sequence_encoder.output_dim': 128,  # Wider
        'model.sequence_encoder.cnn.kernel_sizes': [1, 3, 5, 7],  # Multi-scale
        'model.sequence_encoder.cnn.n_filters': 128,  # More filters
        'model.sequence_encoder.gru.hidden_size': 256,  # Wider GRU
        'model.fusion.input_dim': 256,  # 128 + 128
        'model.fusion.output_dim': 256,
        'model.prediction_head.input_dim': 256,
        'model.epigenomic_encoder.output_dim': 128,
    }
    cfg = OmegaConf.merge(cfg, OmegaConf.create(v3_overrides))
    
    # Backbone-specific
    if args.config:
        exp_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    else:
        backbone_config = PROJECT_ROOT / 'chromaguide' / 'configs' / f'{args.backbone}.yaml'
        if backbone_config.exists():
            exp_cfg = OmegaConf.load(backbone_config)
            cfg = OmegaConf.merge(cfg, exp_cfg)
    
    # V3 backbone-specific LR defaults
    lr_defaults = {
        'cnn_gru': 3e-4,       # From CRISPR-FMC
        'caduceus': 3e-4,
        'dnabert2': 5e-5,      # Fine-tuning
        'evo': 1e-4,
        'nucleotide_transformer': 2e-5,
    }
    
    batch_defaults = {
        'cnn_gru': 512,        # Larger batch (from CRISPR-FMC: 4096)
        'caduceus': 256,
        'dnabert2': 64,
        'evo': 32,
        'nucleotide_transformer': 16,
    }
    
    overrides = {
        'model.sequence_encoder.type': args.backbone,
        'project.seed': args.seed,
        'data.processed_dir': str(Path(args.data_dir) / 'processed'),
        'data.raw_dir': str(Path(args.data_dir) / 'raw'),
    }
    
    overrides['training.optimizer.lr'] = args.lr or lr_defaults.get(args.backbone, 3e-4)
    overrides['training.batch_size'] = args.batch_size or batch_defaults.get(args.backbone, 256)
    
    if args.epochs:
        overrides['training.max_epochs'] = args.epochs
    else:
        overrides['training.max_epochs'] = 200  # More epochs (from CRISPR-FMC)
    
    overrides['training.patience'] = args.patience
    overrides['project.precision'] = 16 if args.mixed_precision else 32
    
    if args.model_cache_dir:
        overrides['model.sequence_encoder.cache_dir'] = args.model_cache_dir
    
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    OmegaConf.resolve(cfg)
    
    return cfg


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
    
    train_loader = create_dataloaders(train_ds, batch_size=batch_size, num_workers=num_workers)
    cal_loader = create_dataloaders(cal_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_dataloaders(test_ds, batch_size=batch_size, num_workers=num_workers)
    
    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Cal:   {len(cal_ds)} samples")
    logger.info(f"Test:  {len(test_ds)} samples")
    
    return train_loader, cal_loader, test_loader


def build_model_v3(cfg, device):
    """Build ChromaGuide v3 model with enhanced architecture."""
    from chromaguide.modules.sequence_encoders_v3 import build_sequence_encoder_v3
    from chromaguide.modules.epigenomic_encoder import EpigenomicEncoder
    from chromaguide.modules.fusion import build_fusion
    from chromaguide.modules.prediction_head import BetaRegressionHead
    
    # Build sequence encoder (v3)
    seq_encoder = build_sequence_encoder_v3(cfg.model.sequence_encoder)
    
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
    
    # Combine into a model
    model = ChromaGuideV3(
        seq_encoder=seq_encoder,
        epi_encoder=epi_encoder,
        fusion=fusion,
        prediction_head=prediction_head,
        backbone_type=cfg.model.sequence_encoder.type,
        modality_dropout_prob=cfg.model.modality_dropout.prob if cfg.model.modality_dropout.enabled else 0.0,
    ).to(device)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.sequence_encoder.type} (v3)")
    logger.info(f"  Total params:     {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    
    return model


class ChromaGuideV3(nn.Module):
    """ChromaGuide v3 model wrapper."""
    
    def __init__(self, seq_encoder, epi_encoder, fusion, prediction_head,
                 backbone_type='cnn_gru', modality_dropout_prob=0.15):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.epi_encoder = epi_encoder
        self.fusion = fusion
        self.prediction_head = prediction_head
        self._needs_raw_sequences = backbone_type in ['dnabert2', 'evo', 'nucleotide_transformer']
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


def train_epoch(model, loader, optimizer, criterion, scaler, device, 
                grad_clip=1.0, use_amp=True, needs_raw_sequences=False):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_primary = 0.0
    epoch_ranking = 0.0
    n_batches = 0
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy'].to(device)
        raw_seqs = batch.get('sequence_str') if needs_raw_sequences else None
        # Convert tuple to list if needed (DataLoader collation)
        if isinstance(raw_seqs, tuple):
            raw_seqs = list(raw_seqs)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi, raw_sequences=raw_seqs)
            losses = criterion(mu, phi, target)
        
        scaler.scale(losses['total']).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += losses['total'].item()
        epoch_primary += losses['primary'].item()
        epoch_ranking += losses.get('ranking', torch.tensor(0.0)).item()
        n_batches += 1
    
    return {
        'total': epoch_loss / max(n_batches, 1),
        'primary': epoch_primary / max(n_batches, 1),
        'ranking': epoch_ranking / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True, needs_raw_sequences=False):
    """Evaluate model and compute metrics."""
    from scipy.stats import spearmanr, pearsonr
    
    model.eval()
    all_mu, all_phi, all_target = [], [], []
    total_loss = 0.0
    n_batches = 0
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy'].to(device)
        raw_seqs = batch.get('sequence_str') if needs_raw_sequences else None
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
    
    spearman_r, spearman_p = spearmanr(all_target, all_mu)
    pearson_r, _ = pearsonr(all_target, all_mu)
    mse = float(np.mean((all_target - all_mu) ** 2))
    mae = float(np.mean(np.abs(all_target - all_mu)))
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'spearman': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson': float(pearson_r),
        'mse': mse,
        'mae': mae,
        'predictions': {'mu': all_mu, 'phi': all_phi, 'target': all_target},
    }


def calibrate_conformal_v3(model, cal_loader, test_loader, cfg, device, split_type,
                            use_amp=True, needs_raw_sequences=False):
    """Calibrate conformal predictor (v3: domain-aware for Splits B/C)."""
    from chromaguide.modules.conformal_v3 import (
        SplitConformalPredictor,
        WeightedConformalPredictor,
    )
    
    model.eval()
    
    cal_mu, cal_phi, cal_y = collect_predictions(model, cal_loader, device, use_amp, needs_raw_sequences)
    test_mu, test_phi, test_y = collect_predictions(model, test_loader, device, use_amp, needs_raw_sequences)
    
    if split_type == 'A':
        # Standard conformal for IID split
        cp = SplitConformalPredictor(
            alpha=cfg.conformal.alpha,
            use_beta_sigma=cfg.conformal.beta_sigma,
            tolerance=cfg.conformal.tolerance,
        )
        q_hat = cp.calibrate(cal_y, cal_mu, cal_phi)
    else:
        # Weighted conformal for domain-shift splits
        cp = WeightedConformalPredictor(
            alpha=cfg.conformal.alpha,
            use_beta_sigma=cfg.conformal.beta_sigma,
            tolerance=cfg.conformal.tolerance,
            domain_weight_method="adaptive",
        )
        q_hat = cp.calibrate(cal_y, cal_mu, cal_phi,
                              test_mu=test_mu, test_phi=test_phi)
    
    logger.info(f"  Conformal q_hat: {q_hat:.4f}")
    
    lower, upper = cp.predict(test_mu, test_phi)
    coverage_stats = cp.evaluate_coverage(test_y, lower, upper)
    
    logger.info(f"  Coverage: {coverage_stats['coverage']:.4f} (target: {1 - cfg.conformal.alpha:.2f})")
    logger.info(f"  Avg width: {coverage_stats['avg_width']:.4f}")
    logger.info(f"  Within tolerance: {coverage_stats['within_tolerance']}")
    
    return coverage_stats


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp=True, needs_raw_sequences=False):
    """Collect model predictions."""
    model.eval()
    mus, phis, ys = [], [], []
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy']
        raw_seqs = batch.get('sequence_str') if needs_raw_sequences else None
        if isinstance(raw_seqs, tuple):
            raw_seqs = list(raw_seqs)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi, raw_sequences=raw_seqs)
        
        mus.append(mu.cpu().numpy().flatten())
        phis.append(phi.cpu().numpy().flatten())
        ys.append(target.numpy().flatten())
    
    return np.concatenate(mus), np.concatenate(phis), np.concatenate(ys)


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


def main():
    args = parse_args()
    
    # Setup
    exp_name = f"{args.backbone}_split{args.split}_seed{args.seed}_{args.version}"
    logger.info("=" * 70)
    logger.info(f"ChromaGuide v3 Training: {exp_name}")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Split: {args.split} (fold {args.split_fold})")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Loss: {args.loss_type}")
    logger.info(f"  Optimizer: {args.optimizer}")
    logger.info(f"  Ranking weight: {args.lambda_rank}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else "")
    logger.info("=" * 70)
    
    # Seed
    from chromaguide.utils.reproducibility import set_seed
    set_seed(args.seed)
    
    # Config
    cfg = load_experiment_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output directory
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, str(output_dir / 'config.yaml'))
    
    # Load data
    train_loader, cal_loader, test_loader = load_data(cfg, args.split, args.split_fold)
    
    # Build model (v3)
    model = build_model_v3(cfg, device)
    
    # Optimizer (Adamax from CRISPR-FMC SOTA)
    if args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
        )
    
    # Scheduler
    from chromaguide.training.trainer import CosineWarmupScheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        min_lr=cfg.training.scheduler.min_lr,
    )
    
    # Loss (v3: Log-Cosh + Ranking)
    from chromaguide.training.losses_v3 import CalibratedLossV3
    criterion = CalibratedLossV3(
        primary_type=args.loss_type,
        lambda_rank=args.lambda_rank,
        lambda_cal=0.05,
        lambda_nr=0.01 if cfg.model.non_redundancy.enabled else 0.0,
        use_nr=cfg.model.non_redundancy.enabled,
    )
    
    # Mixed precision
    use_amp = cfg.project.precision == 16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # ================================================================
    # Training loop
    # ================================================================
    best_val_spearman = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [], 'val_spearman': [], 
        'val_pearson': [], 'lr': [], 'train_ranking': [],
    }
    
    needs_raw_seq = args.backbone in ['dnabert2', 'evo', 'nucleotide_transformer']
    
    start_time = time.time()
    
    for epoch in range(1, cfg.training.max_epochs + 1):
        t0 = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_clip=args.gradient_clip, use_amp=use_amp,
            needs_raw_sequences=needs_raw_seq
        )
        
        val_metrics = evaluate(model, cal_loader, criterion, device, use_amp=use_amp,
                                needs_raw_sequences=needs_raw_seq)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        elapsed = time.time() - t0
        
        logger.info(
            f"Epoch {epoch:3d}/{cfg.training.max_epochs} | "
            f"Train: {train_metrics['total']:.4f} (rank: {train_metrics['ranking']:.4f}) | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"ρ={val_metrics['spearman']:.4f} | "
            f"r={val_metrics['pearson']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )
        
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['lr'].append(current_lr)
        history['train_ranking'].append(train_metrics['ranking'])
        
        if val_metrics['spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['spearman']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_spearman': best_val_spearman,
                'config': OmegaConf.to_container(cfg),
                'version': 'v3',
            }, ckpt_dir / 'best.pt')
            
            logger.info(f"  → New best Spearman: {best_val_spearman:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break
    
    total_time = time.time() - start_time
    
    # ================================================================
    # Final evaluation
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION (v3)")
    logger.info("=" * 70)
    
    best_ckpt = ckpt_dir / 'best.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
    
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp,
                             needs_raw_sequences=needs_raw_seq)
    
    logger.info(f"Test Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Test Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"Test MSE:         {test_metrics['mse']:.6f}")
    logger.info(f"Test MAE:         {test_metrics['mae']:.6f}")
    
    ece = compute_ece(test_metrics['predictions']['mu'], test_metrics['predictions']['target'])
    logger.info(f"Test ECE:         {ece:.4f}")
    
    # Conformal calibration (v3: domain-aware)
    logger.info("\nConformal prediction calibration (v3: domain-aware)...")
    try:
        conformal_results = calibrate_conformal_v3(
            model, cal_loader, test_loader, cfg, device, args.split,
            use_amp, needs_raw_sequences=needs_raw_seq
        )
    except Exception as e:
        logger.warning(f"Conformal calibration failed: {e}")
        conformal_results = {'coverage': -1, 'avg_width': -1, 'within_tolerance': False}
    
    # ================================================================
    # Save results
    # ================================================================
    results = {
        'experiment': exp_name,
        'version': 'v3',
        'backbone': args.backbone,
        'split': args.split,
        'split_fold': args.split_fold,
        'seed': args.seed,
        'loss_type': args.loss_type,
        'optimizer': args.optimizer,
        'lambda_rank': args.lambda_rank,
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
    }
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save predictions
    np.savez(
        output_dir / 'predictions.npz',
        mu=test_metrics['predictions']['mu'],
        phi=test_metrics['predictions']['phi'],
        target=test_metrics['predictions']['target'],
    )
    
    # ================================================================
    # Summary
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE (v3)")
    logger.info("=" * 70)
    logger.info(f"Backbone:    {args.backbone}")
    logger.info(f"Split:       {args.split}")
    logger.info(f"Loss:        {args.loss_type} + ranking (λ={args.lambda_rank})")
    logger.info(f"Best epoch:  {best_epoch}")
    logger.info(f"Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"ECE:         {ece:.4f}")
    logger.info(f"Coverage:    {conformal_results.get('coverage', -1):.4f}")
    logger.info(f"Time:        {total_time/60:.1f} min")
    
    # Targets check
    logger.info("\n--- THESIS TARGETS (v3) ---")
    sp = test_metrics['spearman']
    cov = conformal_results.get('coverage', -1)
    logger.info(f"Spearman ≥ 0.91: {sp:.4f} {'✓' if sp >= 0.91 else '✗'}")
    logger.info(f"ECE < 0.05:      {ece:.4f} {'✓' if ece < 0.05 else '✗'}")
    logger.info(f"Coverage ≈ 0.90: {cov:.4f} {'✓' if abs(cov - 0.90) < 0.02 else '✗'}")
    
    # SOTA comparison
    logger.info("\n--- SOTA COMPARISON ---")
    logger.info(f"CRISPR-FMC WT SCC:     0.861")
    logger.info(f"CRISPR-FMC Sniper SCC: 0.935")
    logger.info(f"CrnnCrispr avg SCC:    0.701")
    logger.info(f"Our {args.backbone} SCC: {sp:.4f}")
    if sp > 0.935:
        logger.info(">>> SURPASSED CRISPR-FMC SOTA! <<<")
    elif sp > 0.861:
        logger.info(">>> Above CRISPR-FMC WT baseline, approaching SOTA")
    elif sp > 0.701:
        logger.info(">>> Above CrnnCrispr baseline")
    
    if not args.no_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
