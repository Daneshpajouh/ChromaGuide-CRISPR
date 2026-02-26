"""Run evaluation on trained models."""
from __future__ import annotations
import json
import logging
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig

from chromaguide.models.chromaguide import ChromaGuideModel
from chromaguide.data.dataset import CRISPRDataset, create_dataloaders
from chromaguide.evaluation.metrics import compute_metrics
from chromaguide.modules.conformal import SplitConformalPredictor

logger = logging.getLogger(__name__)


def run_evaluation(
    cfg: DictConfig,
    checkpoint_path: str,
    split_type: str,
    device: torch.device,
) -> dict:
    """Full evaluation pipeline for a trained model."""
    
    # Load model
    model = ChromaGuideModel(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    # Load data
    processed_dir = Path(cfg.data.processed_dir)
    splits_dir = processed_dir / "splits"
    
    if split_type == "A":
        split_file = splits_dir / "split_a.npz"
    else:
        split_file = splits_dir / f"split_{split_type.lower()}_0.npz"
    
    cal_ds = CRISPRDataset.from_processed(processed_dir, split_file, "cal")
    test_ds = CRISPRDataset.from_processed(processed_dir, split_file, "test")
    
    batch_size = cfg.training.batch_size
    cal_loader = create_dataloaders(cal_ds, batch_size=batch_size)
    test_loader = create_dataloaders(test_ds, batch_size=batch_size)
    
    # Collect predictions
    def collect(loader):
        mus, phis, ys = [], [], []
        with torch.no_grad():
            for batch in loader:
                seq = batch["sequence"].to(device)
                epi = batch["epigenomic"].to(device)
                mu, phi = model(seq, epi)
                mus.append(mu.cpu().numpy().flatten())
                phis.append(phi.cpu().numpy().flatten())
                ys.append(batch["efficacy"].numpy().flatten())
        return np.concatenate(mus), np.concatenate(phis), np.concatenate(ys)
    
    cal_mu, cal_phi, cal_y = collect(cal_loader)
    test_mu, test_phi, test_y = collect(test_loader)
    
    # Compute all metrics
    metrics = compute_metrics(
        test_y, test_mu, test_phi,
        metric_types=["primary", "secondary", "ranking", "calibration"],
    )
    
    # Conformal prediction
    cp = SplitConformalPredictor(
        alpha=cfg.conformal.alpha,
        use_beta_sigma=cfg.conformal.beta_sigma,
    )
    cp.calibrate(cal_y, cal_mu, cal_phi)
    lower, upper = cp.predict(test_mu, test_phi)
    coverage = cp.evaluate_coverage(test_y, lower, upper)
    
    metrics.update({f"conformal_{k}": v for k, v in coverage.items() if isinstance(v, (int, float, bool))})
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    encoder_type = cfg.model.sequence_encoder.type
    result_file = results_dir / f"eval_{encoder_type}_split{split_type}.json"
    
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nEvaluation results for {encoder_type} (Split {split_type}):")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    
    # Also save raw predictions for downstream analysis
    np.savez(
        results_dir / f"predictions_{encoder_type}_split{split_type}.npz",
        test_mu=test_mu, test_phi=test_phi, test_y=test_y,
        cal_mu=cal_mu, cal_phi=cal_phi, cal_y=cal_y,
        lower=lower, upper=upper,
    )
    
    return metrics
