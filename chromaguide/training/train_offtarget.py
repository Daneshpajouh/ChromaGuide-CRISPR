"""Off-target model training pipeline."""
from __future__ import annotations
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from omegaconf import DictConfig

from chromaguide.models.offtarget import OffTargetModel, OffTargetSiteScorerCNN
from chromaguide.data.dataset import OffTargetDataset
from chromaguide.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def _generate_synthetic_offtarget_data(n_pairs: int = 50000, seed: int = 42):
    """Generate synthetic off-target data for development.
    
    Returns guides, targets, labels, chromatin arrays.
    """
    np.random.seed(seed)
    bases = list("ACGT")
    
    guides, targets, labels = [], [], []
    
    for _ in range(n_pairs):
        guide = "".join(np.random.choice(bases, 23))
        
        # Create off-target site with some mismatches
        n_mismatches = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
        target_list = list(guide)
        mismatch_positions = np.random.choice(23, n_mismatches, replace=False)
        for pos in mismatch_positions:
            target_list[pos] = np.random.choice([b for b in bases if b != target_list[pos]])
        target = "".join(target_list)
        
        # Label: fewer mismatches → higher chance of cleavage
        p_cleave = max(0, 1.0 - 0.3 * n_mismatches + np.random.normal(0, 0.1))
        label = 1 if np.random.random() < p_cleave else 0
        
        guides.append(guide)
        targets.append(target)
        labels.append(label)
    
    labels = np.array(labels, dtype=np.float32)
    chromatin = np.random.exponential(1.0, (n_pairs, 3)).astype(np.float32)
    
    return guides, targets, labels, chromatin


def train_offtarget(cfg: DictConfig, device: torch.device) -> dict:
    """Train the off-target prediction module.
    
    Uses binary cross-entropy with positive class weighting
    to handle the severe class imbalance (most sites are not cleaved).
    """
    logger.info("=" * 60)
    logger.info("Training Off-Target Prediction Module")
    logger.info("=" * 60)
    
    set_seed(cfg.project.seed)
    
    # Load data (or generate synthetic)
    processed_dir = Path(cfg.data.processed_dir)
    offtarget_file = processed_dir / "offtarget_pairs.parquet"
    
    if offtarget_file.exists():
        import pandas as pd
        df = pd.read_parquet(offtarget_file)
        guides = df["guide"].tolist()
        targets = df["target"].tolist()
        labels = df["label"].values
        chromatin = df[["dnase", "h3k4me3", "h3k27ac"]].values if "dnase" in df.columns else None
    else:
        logger.warning("Off-target data not found. Using synthetic data.")
        guides, targets, labels, chromatin = _generate_synthetic_offtarget_data()
    
    # Create dataset
    dataset = OffTargetDataset(
        guides=guides,
        targets=targets,
        labels=labels,
        chromatin=chromatin,
    )
    
    # Train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    
    batch_size = cfg.offtarget_model.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Build model (site scorer only; aggregation tested separately)
    model = OffTargetSiteScorerCNN(
        kernel_sizes=list(cfg.offtarget_model.scoring.kernel_sizes),
        n_filters=cfg.offtarget_model.scoring.n_filters,
        fc_dims=list(cfg.offtarget_model.scoring.fc_dims),
        use_chromatin=cfg.offtarget_model.scoring.use_chromatin,
    ).to(device)
    
    # Loss with positive weight for class imbalance
    pos_weight = torch.tensor([cfg.offtarget_model.training.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.offtarget_model.training.lr)
    
    logger.info(f"Train: {n_train}, Val: {n_val}")
    logger.info(f"Positive rate: {labels.mean():.3f}")
    
    best_auroc = 0.0
    
    for epoch in range(1, cfg.offtarget_model.training.epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            alignment = batch["alignment"].to(device)
            label = batch["label"].to(device)
            chrom = batch.get("chromatin")
            if chrom is not None:
                chrom = chrom.to(device)
            
            pred = model(alignment, chrom).squeeze(-1)
            # Remove sigmoid since BCEWithLogitsLoss applies it
            loss = criterion(pred.logit() if False else pred, label)  # pred is already sigmoid'd
            
            optimizer.zero_grad()
            # Use BCELoss instead since model outputs probabilities
            loss = nn.functional.binary_cross_entropy(pred.squeeze(), label, reduction="mean")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                alignment = batch["alignment"].to(device)
                chrom = batch.get("chromatin")
                if chrom is not None:
                    chrom = chrom.to(device)
                
                pred = model(alignment, chrom).squeeze(-1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch["label"].numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auroc = roc_auc_score(all_labels, all_preds)
            auprc = average_precision_score(all_labels, all_preds)
        except ValueError:
            auroc = 0.5
            auprc = 0.5
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{cfg.offtarget_model.training.epochs} | "
                f"Loss: {total_loss / len(train_loader):.4f} | "
                f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}"
            )
        
        if auroc > best_auroc:
            best_auroc = auroc
            ckpt_dir = Path("results/checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "offtarget_best.pt")
    
    logger.info(f"\nBest AUROC: {best_auroc:.4f}")
    
    return {"best_auroc": best_auroc}
