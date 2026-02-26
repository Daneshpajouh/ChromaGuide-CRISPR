#!/usr/bin/env python3
"""Master training script for ChromaGuide experiments.

Trains a single backbone on a single split with a single seed.
Handles: data loading, training, conformal calibration, evaluation, saving.

Usage:
    python train_experiment.py --backbone cnn_gru --split A --seed 42
    python train_experiment.py --backbone dnabert2 --split A --seed 42 --config configs/dnabert2.yaml
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
    parser = argparse.ArgumentParser(description='ChromaGuide Training Experiment')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=['cnn_gru', 'dnabert2', 'caduceus', 'evo', 'nucleotide_transformer'],
                        help='Sequence encoder backbone')
    parser.add_argument('--split', type=str, required=True,
                        choices=['A', 'B', 'C'],
                        help='Split type (A=gene-held-out, B=dataset-held-out, C=cell-line-held-out)')
    parser.add_argument('--split-fold', type=int, default=0,
                        help='Fold index for split B/C (0-based)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiment config YAML (overrides defaults)')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'),
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'results'),
                        help='Output directory')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Use mixed precision training')
    return parser.parse_args()


def load_experiment_config(args):
    """Build config from defaults + backbone-specific + CLI overrides."""
    from omegaconf import OmegaConf
    
    # Load default config
    default_path = PROJECT_ROOT / 'chromaguide' / 'configs' / 'default.yaml'
    if default_path.exists():
        cfg = OmegaConf.load(default_path)
    else:
        raise FileNotFoundError(f"Default config not found: {default_path}")
    
    # Merge backbone-specific config
    if args.config:
        exp_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    else:
        # Auto-detect backbone config
        backbone_config = PROJECT_ROOT / 'chromaguide' / 'configs' / f'{args.backbone}.yaml'
        if backbone_config.exists():
            exp_cfg = OmegaConf.load(backbone_config)
            cfg = OmegaConf.merge(cfg, exp_cfg)
    
    # Apply CLI overrides
    overrides = {
        'model.sequence_encoder.type': args.backbone,
        'project.seed': args.seed,
        'data.processed_dir': str(Path(args.data_dir) / 'processed'),
        'data.raw_dir': str(Path(args.data_dir) / 'raw'),
    }
    if args.epochs:
        overrides['training.max_epochs'] = args.epochs
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.lr:
        overrides['training.optimizer.lr'] = args.lr
    if args.patience:
        overrides['training.patience'] = args.patience
    
    overrides['project.precision'] = 16 if args.mixed_precision else 32
    
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


def build_model(cfg, device):
    """Build ChromaGuide model."""
    from chromaguide.models.chromaguide import ChromaGuideModel
    
    model = ChromaGuideModel(cfg).to(device)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model.sequence_encoder.type}")
    logger.info(f"  Total params:     {total:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    
    return model


def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip=1.0, use_amp=True):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_primary = 0.0
    n_batches = 0
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi)
            losses = criterion(mu, phi, target)
        
        scaler.scale(losses['total']).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += losses['total'].item()
        epoch_primary += losses['primary'].item()
        n_batches += 1
    
    return {
        'total': epoch_loss / max(n_batches, 1),
        'primary': epoch_primary / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
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
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi)
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


def calibrate_conformal(model, cal_loader, test_loader, cfg, device, use_amp=True):
    """Calibrate conformal predictor and evaluate coverage."""
    from chromaguide.modules.conformal import SplitConformalPredictor
    
    model.eval()
    
    # Collect calibration predictions
    cal_mu, cal_phi, cal_y = collect_predictions(model, cal_loader, device, use_amp)
    test_mu, test_phi, test_y = collect_predictions(model, test_loader, device, use_amp)
    
    cp = SplitConformalPredictor(
        alpha=cfg.conformal.alpha,
        use_beta_sigma=cfg.conformal.beta_sigma,
        tolerance=cfg.conformal.tolerance,
    )
    
    q_hat = cp.calibrate(cal_y, cal_mu, cal_phi)
    logger.info(f"  Conformal q_hat: {q_hat:.4f}")
    
    lower, upper = cp.predict(test_mu, test_phi)
    coverage_stats = cp.evaluate_coverage(test_y, lower, upper)
    
    logger.info(f"  Coverage: {coverage_stats['coverage']:.4f} (target: {1 - cfg.conformal.alpha:.2f})")
    logger.info(f"  Avg width: {coverage_stats['avg_width']:.4f}")
    logger.info(f"  Within tolerance: {coverage_stats['within_tolerance']}")
    
    return coverage_stats


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp=True):
    """Collect model predictions."""
    model.eval()
    mus, phis, ys = [], [], []
    
    for batch in loader:
        seq = batch['sequence'].to(device)
        epi = batch['epigenomic'].to(device)
        target = batch['efficacy']
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, phi = model(seq, epi)
        
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
    exp_name = f"{args.backbone}_split{args.split}_seed{args.seed}"
    logger.info("=" * 70)
    logger.info(f"ChromaGuide Training: {exp_name}")
    logger.info(f"  Backbone: {args.backbone}")
    logger.info(f"  Split: {args.split} (fold {args.split_fold})")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
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
    
    # Build model
    model = build_model(cfg, device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
    )
    
    # Scheduler
    from chromaguide.training.trainer import CosineWarmupScheduler
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.training.scheduler.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        min_lr=cfg.training.scheduler.min_lr,
    )
    
    # Loss
    from chromaguide.training.losses import CalibratedLoss
    criterion = CalibratedLoss(
        primary_type=cfg.training.loss.primary,
        lambda_cal=cfg.training.loss.lambda_cal,
        lambda_nr=cfg.training.loss.get('lambda_nr', 0.01),
        use_nr=cfg.model.non_redundancy.enabled,
    )
    
    # Mixed precision
    use_amp = cfg.project.precision == 16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # W&B (optional)
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project='chromaguide',
                name=exp_name,
                config=OmegaConf.to_container(cfg),
                tags=[args.backbone, f'split_{args.split}', f'seed_{args.seed}'],
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            args.no_wandb = True
    
    # ================================================================
    # Training loop
    # ================================================================
    best_val_spearman = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_spearman': [], 'val_pearson': [], 'lr': []}
    
    start_time = time.time()
    
    for epoch in range(1, cfg.training.max_epochs + 1):
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_clip=args.gradient_clip, use_amp=use_amp
        )
        
        # Validate
        val_metrics = evaluate(model, cal_loader, criterion, device, use_amp=use_amp)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        elapsed = time.time() - t0
        
        # Log
        logger.info(
            f"Epoch {epoch:3d}/{cfg.training.max_epochs} | "
            f"Train: {train_metrics['total']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"ρ={val_metrics['spearman']:.4f} | "
            f"r={val_metrics['pearson']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )
        
        # Track history
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_spearman'].append(val_metrics['spearman'])
        history['val_pearson'].append(val_metrics['pearson'])
        history['lr'].append(current_lr)
        
        # W&B
        if not args.no_wandb:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['total'],
                    'val/loss': val_metrics['loss'],
                    'val/spearman': val_metrics['spearman'],
                    'val/pearson': val_metrics['pearson'],
                    'val/mse': val_metrics['mse'],
                    'lr': current_lr,
                })
            except Exception:
                pass
        
        # Best model
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
            }, ckpt_dir / 'best.pt')
            
            logger.info(f"  → New best Spearman: {best_val_spearman:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break
    
    total_time = time.time() - start_time
    
    # ================================================================
    # Final evaluation
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    
    # Load best checkpoint
    best_ckpt = ckpt_dir / 'best.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
    
    logger.info(f"Test Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Test Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"Test MSE:         {test_metrics['mse']:.6f}")
    logger.info(f"Test MAE:         {test_metrics['mae']:.6f}")
    
    # ECE
    ece = compute_ece(test_metrics['predictions']['mu'], test_metrics['predictions']['target'])
    logger.info(f"Test ECE:         {ece:.4f}")
    
    # Conformal calibration
    logger.info("\nConformal prediction calibration...")
    try:
        conformal_results = calibrate_conformal(model, cal_loader, test_loader, cfg, device, use_amp)
    except Exception as e:
        logger.warning(f"Conformal calibration failed: {e}")
        conformal_results = {'coverage': -1, 'avg_width': -1, 'within_tolerance': False}
    
    # ================================================================
    # Save results
    # ================================================================
    results = {
        'experiment': exp_name,
        'backbone': args.backbone,
        'split': args.split,
        'split_fold': args.split_fold,
        'seed': args.seed,
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
    history_file = output_dir / 'history.json'
    with open(history_file, 'w') as f:
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
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Backbone:    {args.backbone}")
    logger.info(f"Split:       {args.split}")
    logger.info(f"Seed:        {args.seed}")
    logger.info(f"Best epoch:  {best_epoch}")
    logger.info(f"Spearman ρ:  {test_metrics['spearman']:.4f}")
    logger.info(f"Pearson r:   {test_metrics['pearson']:.4f}")
    logger.info(f"ECE:         {ece:.4f}")
    logger.info(f"Coverage:    {conformal_results.get('coverage', -1):.4f}")
    logger.info(f"Time:        {total_time/60:.1f} min")
    logger.info(f"Output:      {output_dir}")
    
    # Targets check
    logger.info("\n--- THESIS TARGETS ---")
    target_spearman = 0.91
    target_ece = 0.05
    target_coverage = 0.90
    cov = conformal_results.get('coverage', -1)
    
    logger.info(f"Spearman ≥ {target_spearman}: {test_metrics['spearman']:.4f} {'✓' if test_metrics['spearman'] >= target_spearman else '✗'}")
    logger.info(f"ECE < {target_ece}:      {ece:.4f} {'✓' if ece < target_ece else '✗'}")
    logger.info(f"Coverage ≈ {target_coverage}:  {cov:.4f} {'✓' if abs(cov - target_coverage) < 0.02 else '✗'}")
    
    if not args.no_wandb:
        try:
            import wandb
            wandb.log({
                'test/spearman': test_metrics['spearman'],
                'test/pearson': test_metrics['pearson'],
                'test/mse': test_metrics['mse'],
                'test/ece': ece,
                'test/coverage': conformal_results.get('coverage', -1),
            })
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
