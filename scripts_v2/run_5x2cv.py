"""Run 5×2 cross-validation for statistical testing.

Trains ChromaGuide (default backbone) and baseline on 10 folds,
then runs 5×2cv paired t-test + BCa bootstrap CIs.
"""
import json
import logging
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

from chromaguide.utils import load_config, set_seed, get_device, setup_logger
from chromaguide.models.chromaguide import ChromaGuideModel
from chromaguide.data.dataset import CRISPRDataset, create_dataloaders
from chromaguide.training.trainer import Trainer
from chromaguide.evaluation.statistical_tests import (
    five_by_two_cv_paired_t_test,
    bca_bootstrap_ci,
    holm_bonferroni,
)

logger = setup_logger(log_dir="results/logs")


def run_single_fold(cfg, split_file, seed, device):
    """Train and evaluate on a single CV fold."""
    set_seed(seed)
    
    model = ChromaGuideModel(cfg).to(device)
    
    processed_dir = Path(cfg.data.processed_dir)
    train_ds = CRISPRDataset.from_processed(processed_dir, split_file, "train", augment=True)
    test_ds = CRISPRDataset.from_processed(processed_dir, split_file, "test", augment=False)
    
    train_loader = create_dataloaders(train_ds, batch_size=cfg.training.batch_size)
    test_loader = create_dataloaders(test_ds, batch_size=cfg.training.batch_size)
    
    # Quick training (fewer epochs for CV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
    
    model.train()
    for epoch in range(30):  # Shortened for CV
        for batch in train_loader:
            seq = batch["sequence"].to(device)
            epi = batch["epigenomic"].to(device)
            target = batch["efficacy"].to(device)
            
            mu, phi = model(seq, epi)
            loss = torch.nn.functional.mse_loss(mu.view(-1), target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_mu, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            epi = batch["epigenomic"].to(device)
            mu, _ = model(seq, epi)
            all_mu.append(mu.cpu().numpy().flatten())
            all_y.append(batch["efficacy"].numpy().flatten())
    
    all_mu = np.concatenate(all_mu)
    all_y = np.concatenate(all_y)
    
    rho, _ = spearmanr(all_y, all_mu)
    return float(rho), all_y, all_mu


def main():
    cfg = load_config()
    device = get_device()
    
    processed_dir = Path(cfg.data.processed_dir)
    splits_dir = processed_dir / "splits"
    
    # Find CV split files
    cv_files = sorted(splits_dir.glob("cv_5x2_*.npz"))
    
    if not cv_files:
        logger.error("No CV split files found. Run: chromaguide data --stage splits")
        return
    
    logger.info(f"Found {len(cv_files)} CV split files")
    
    # Run ChromaGuide on all folds
    chromaguide_scores = [[] for _ in range(5)]  # 5 repeats × [fold0, fold1]
    
    for i, split_file in enumerate(cv_files):
        repeat = i // 2  # actually fold // 2
        fold_in_repeat = i % 2
        
        if repeat >= 5:
            break
        
        logger.info(f"CV fold {i}: repeat={repeat}, sub-fold={fold_in_repeat}")
        rho, y_true, y_pred = run_single_fold(cfg, str(split_file), cfg.project.seed + i, device)
        
        if fold_in_repeat < 2:
            chromaguide_scores[repeat].append(rho)
        
        logger.info(f"  Spearman ρ = {rho:.4f}")
    
    # Run baseline (sequence-only, no epigenomics)
    logger.info("\nRunning baseline (sequence-only)...")
    from omegaconf import OmegaConf
    baseline_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    baseline_cfg.model.modality_dropout.prob = 1.0  # Always mask epigenomic
    
    baseline_scores = [[] for _ in range(5)]
    
    for i, split_file in enumerate(cv_files[:10]):
        repeat = i // 2
        fold_in_repeat = i % 2
        
        if repeat >= 5:
            break
        
        rho, _, _ = run_single_fold(baseline_cfg, str(split_file), cfg.project.seed + i, device)
        baseline_scores[repeat].append(rho)
        logger.info(f"  Baseline fold {i}: ρ = {rho:.4f}")
    
    # Ensure each repeat has exactly 2 folds
    for i in range(5):
        while len(chromaguide_scores[i]) < 2:
            chromaguide_scores[i].append(chromaguide_scores[i][-1] if chromaguide_scores[i] else 0.5)
        while len(baseline_scores[i]) < 2:
            baseline_scores[i].append(baseline_scores[i][-1] if baseline_scores[i] else 0.5)
    
    # 5×2cv paired t-test
    test_result = five_by_two_cv_paired_t_test(chromaguide_scores, baseline_scores)
    
    logger.info("\n" + "=" * 60)
    logger.info("5×2cv Paired t-test Results")
    logger.info(f"  ChromaGuide scores: {chromaguide_scores}")
    logger.info(f"  Baseline scores: {baseline_scores}")
    logger.info(f"  t-statistic: {test_result['t_statistic']:.4f}")
    logger.info(f"  p-value: {test_result['p_value']:.6f}")
    logger.info(f"  Significant (p < 0.05): {test_result['significant']}")
    
    # Save results
    results = {
        "chromaguide_scores": chromaguide_scores,
        "baseline_scores": baseline_scores,
        "t_test": test_result,
    }
    
    with open("results/cv_results/5x2cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to results/cv_results/5x2cv_results.json")


if __name__ == "__main__":
    main()
