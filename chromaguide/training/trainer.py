"""Training loop for ChromaGuide.

Handles:
    - Mixed-precision training (AMP)
    - Gradient clipping
    - Cosine warmup LR scheduler
    - Early stopping
    - W&B logging
    - Checkpoint saving/loading
    - Conformal calibration after training
"""
from __future__ import annotations
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig

from chromaguide.models.chromaguide import ChromaGuideModel
from chromaguide.data.dataset import CRISPRDataset, create_dataloaders
from chromaguide.training.losses import CalibratedLoss
from chromaguide.modules.conformal import SplitConformalPredictor
from chromaguide.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            import math
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cos_factor
                for base_lr in self.base_lrs
            ]


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False
    
    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """Main training loop for ChromaGuide."""
    
    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device = None,
        split_type: str = "A",
        use_wandb: bool = True,
        logger: logging.Logger = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_type = split_type
        self.use_wandb = use_wandb
        self.logger = logger or logging.getLogger(__name__)
        
        # Build model
        self.model = ChromaGuideModel(cfg).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=tuple(cfg.training.optimizer.betas),
        )
        
        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.training.scheduler.warmup_epochs,
            max_epochs=cfg.training.max_epochs,
            min_lr=cfg.training.scheduler.min_lr,
        )
        
        # Loss
        self.criterion = CalibratedLoss(
            primary_type=cfg.training.loss.primary,
            lambda_cal=cfg.training.loss.lambda_cal,
            lambda_nr=cfg.training.loss.get("lambda_nr", 0.01),
            use_nr=cfg.model.non_redundancy.enabled,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=cfg.training.patience)
        
        # Mixed precision
        self.use_amp = cfg.project.precision == 16
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Gradient clipping
        self.grad_clip = cfg.training.gradient_clip
        
        # Checkpoint directory
        self.ckpt_dir = Path("results/checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_spearman = 0.0
        self.train_losses = []
        self.val_losses = []
        
        # W&B
        self._wandb_run = None
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project="chromaguide",
                name=f"{self.cfg.model.sequence_encoder.type}_split{self.split_type}",
                config=dict(self.cfg),
                tags=[self.cfg.model.sequence_encoder.type, f"split_{self.split_type}"],
            )
        except Exception as e:
            self.logger.warning(f"W&B init failed: {e}. Continuing without logging.")
            self.use_wandb = False
    
    def _load_data(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Load train/cal/test dataloaders."""
        processed_dir = Path(self.cfg.data.processed_dir)
        splits_dir = processed_dir / "splits"
        
        if self.split_type == "A":
            split_file = splits_dir / "split_a.npz"
        else:
            split_file = splits_dir / f"split_{self.split_type.lower()}_0.npz"
        
        train_ds = CRISPRDataset.from_processed(
            processed_dir, split_file, subset="train", augment=True,
        )
        cal_ds = CRISPRDataset.from_processed(
            processed_dir, split_file, subset="cal", augment=False,
        )
        test_ds = CRISPRDataset.from_processed(
            processed_dir, split_file, subset="test", augment=False,
        )
        
        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.project.num_workers
        
        train_loader = create_dataloaders(train_ds, batch_size=batch_size, num_workers=num_workers)
        cal_loader = create_dataloaders(cal_ds, batch_size=batch_size, num_workers=num_workers)
        test_loader = create_dataloaders(test_ds, batch_size=batch_size, num_workers=num_workers)
        
        return train_loader, cal_loader, test_loader
    
    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "primary": 0.0, "calibration": 0.0}
        n_batches = 0
        
        for batch in loader:
            seq = batch["sequence"].to(self.device)
            epi = batch["epigenomic"].to(self.device)
            target = batch["efficacy"].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                mu, phi = self.model(seq, epi)
                losses = self.criterion(mu, phi, target)
            
            self.scaler.scale(losses["total"]).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1
        
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Validate and compute metrics."""
        self.model.eval()
        all_mu, all_phi, all_target = [], [], []
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            seq = batch["sequence"].to(self.device)
            epi = batch["epigenomic"].to(self.device)
            target = batch["efficacy"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                mu, phi = self.model(seq, epi)
                losses = self.criterion(mu, phi, target)
            
            all_mu.append(mu.cpu().numpy())
            all_phi.append(phi.cpu().numpy())
            all_target.append(target.cpu().numpy())
            total_loss += losses["total"].item()
            n_batches += 1
        
        all_mu = np.concatenate(all_mu).flatten()
        all_phi = np.concatenate(all_phi).flatten()
        all_target = np.concatenate(all_target).flatten()
        
        from scipy.stats import spearmanr, pearsonr
        spearman_r, spearman_p = spearmanr(all_target, all_mu)
        pearson_r, _ = pearsonr(all_target, all_mu)
        mse = float(np.mean((all_target - all_mu) ** 2))
        mae = float(np.mean(np.abs(all_target - all_mu)))
        
        return {
            "val_loss": total_loss / max(n_batches, 1),
            "spearman": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson": float(pearson_r),
            "mse": mse,
            "mae": mae,
        }
    
    def _save_checkpoint(self, is_best: bool = False) -> str:
        """Save model checkpoint."""
        encoder_type = self.cfg.model.sequence_encoder.type
        ckpt = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_spearman": self.best_val_spearman,
            "config": dict(self.cfg),
        }
        
        path = self.ckpt_dir / f"{encoder_type}_split{self.split_type}_epoch{self.epoch}.pt"
        torch.save(ckpt, path)
        
        if is_best:
            best_path = self.ckpt_dir / f"{encoder_type}_split{self.split_type}_best.pt"
            torch.save(ckpt, best_path)
        
        return str(path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.epoch = ckpt["epoch"]
        self.best_val_spearman = ckpt["best_val_spearman"]
    
    def fit(self) -> dict:
        """Full training loop.
        
        Returns:
            Dict with training history and final metrics.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Training ChromaGuide ({self.cfg.model.sequence_encoder.type})")
        self.logger.info(f"Split: {self.split_type}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)
        
        # Load data
        train_loader, cal_loader, test_loader = self._load_data()
        self.logger.info(f"Train: {len(train_loader.dataset)}, Cal: {len(cal_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        # Count parameters
        from chromaguide.utils.reproducibility import count_parameters
        params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {params['trainable_M']} trainable")
        
        history = {"train_loss": [], "val_loss": [], "spearman": []}
        
        for self.epoch in range(1, self.cfg.training.max_epochs + 1):
            t0 = time.time()
            
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate on calibration set
            val_metrics = self._validate(cal_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {self.epoch:3d}/{self.cfg.training.max_epochs} | "
                f"Train loss: {train_metrics['total']:.4f} | "
                f"Val loss: {val_metrics['val_loss']:.4f} | "
                f"Spearman: {val_metrics['spearman']:.4f} | "
                f"Pearson: {val_metrics['pearson']:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )
            
            # Track history
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["spearman"].append(val_metrics["spearman"])
            
            # W&B logging
            if self.use_wandb and self._wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch": self.epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": self.scheduler.get_last_lr()[0],
                })
            
            # Save best checkpoint
            if val_metrics["spearman"] > self.best_val_spearman:
                self.best_val_spearman = val_metrics["spearman"]
                self._save_checkpoint(is_best=True)
                self.logger.info(f"  → New best Spearman: {self.best_val_spearman:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics["val_loss"]):
                self.logger.info(f"Early stopping at epoch {self.epoch}")
                break
        
        # Final evaluation on test set
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Final evaluation on test set")
        
        # Load best checkpoint
        best_ckpt = self.ckpt_dir / f"{self.cfg.model.sequence_encoder.type}_split{self.split_type}_best.pt"
        if best_ckpt.exists():
            self.load_checkpoint(str(best_ckpt))
        
        test_metrics = self._validate(test_loader)
        self.logger.info(f"Test Spearman: {test_metrics['spearman']:.4f}")
        self.logger.info(f"Test Pearson: {test_metrics['pearson']:.4f}")
        self.logger.info(f"Test MSE: {test_metrics['mse']:.6f}")
        
        # Conformal calibration
        self.logger.info("\nCalibrating conformal prediction...")
        conformal_results = self._calibrate_conformal(cal_loader, test_loader)
        
        results = {
            "history": history,
            "best_val_spearman": self.best_val_spearman,
            "test_metrics": test_metrics,
            "conformal": conformal_results,
        }
        
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.finish()
        
        return results
    
    def _calibrate_conformal(
        self, cal_loader: DataLoader, test_loader: DataLoader
    ) -> dict:
        """Calibrate conformal predictor and evaluate coverage."""
        self.model.eval()
        
        # Get calibration predictions
        cal_mu, cal_phi, cal_y = self._collect_predictions(cal_loader)
        test_mu, test_phi, test_y = self._collect_predictions(test_loader)
        
        # Build conformal predictor
        cp = SplitConformalPredictor(
            alpha=self.cfg.conformal.alpha,
            use_beta_sigma=self.cfg.conformal.beta_sigma,
            tolerance=self.cfg.conformal.tolerance,
        )
        
        q_hat = cp.calibrate(cal_y, cal_mu, cal_phi)
        self.logger.info(f"  Conformal q_hat: {q_hat:.4f}")
        
        lower, upper = cp.predict(test_mu, test_phi)
        coverage_stats = cp.evaluate_coverage(test_y, lower, upper)
        
        self.logger.info(f"  Coverage: {coverage_stats['coverage']:.3f} (target: {1 - self.cfg.conformal.alpha})")
        self.logger.info(f"  Avg width: {coverage_stats['avg_width']:.4f}")
        self.logger.info(f"  Within tolerance: {coverage_stats['within_tolerance']}")
        
        return coverage_stats
    
    @torch.no_grad()
    def _collect_predictions(self, loader: DataLoader):
        """Collect model predictions into numpy arrays."""
        self.model.eval()
        mus, phis, ys = [], [], []
        
        for batch in loader:
            seq = batch["sequence"].to(self.device)
            epi = batch["epigenomic"].to(self.device)
            target = batch["efficacy"]
            
            mu, phi = self.model(seq, epi)
            mus.append(mu.cpu().numpy().flatten())
            phis.append(phi.cpu().numpy().flatten())
            ys.append(target.numpy().flatten())
        
        return np.concatenate(mus), np.concatenate(phis), np.concatenate(ys)
