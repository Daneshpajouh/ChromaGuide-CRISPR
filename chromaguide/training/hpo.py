"""Hyperparameter optimization using Optuna.

Searches over:
    - Learning rate (log-uniform)
    - Dropout rates
    - Hidden dimensions
    - Batch size
    - Gate bias initialization

Uses TPE sampler with early pruning via Hyperband.
"""
from __future__ import annotations
import logging
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from chromaguide.training.trainer import Trainer
from chromaguide.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, cfg: DictConfig, split_type: str = "A") -> float:
    """Optuna objective function.
    
    Suggests hyperparameters and returns validation Spearman ρ.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gate_bias = trial.suggest_float("gate_bias", -1.0, 1.0)
    lambda_cal = trial.suggest_float("lambda_cal", 0.01, 0.5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 10)
    
    # Apply to config
    hpo_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    hpo_cfg.training.optimizer.lr = lr
    hpo_cfg.training.optimizer.weight_decay = weight_decay
    hpo_cfg.training.batch_size = batch_size
    hpo_cfg.training.scheduler.warmup_epochs = warmup_epochs
    hpo_cfg.training.loss.lambda_cal = lambda_cal
    hpo_cfg.model.sequence_encoder.cnn.dropout = dropout
    hpo_cfg.model.sequence_encoder.gru.dropout = dropout
    hpo_cfg.model.epigenomic_encoder.dropout = dropout
    
    # Shorter training for HPO
    hpo_cfg.training.max_epochs = 30
    hpo_cfg.training.patience = 8
    
    set_seed(cfg.project.seed)
    
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(
        hpo_cfg, device=device, split_type=split_type,
        use_wandb=False, logger=logger,
    )
    
    try:
        results = trainer.fit()
        spearman = results["best_val_spearman"]
        
        # Report intermediate values for pruning
        for epoch, val in enumerate(results["history"]["spearman"]):
            trial.report(val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return spearman
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def run_hpo(cfg: DictConfig, n_trials: int = 50, split_type: str = "A") -> dict:
    """Run full HPO study.
    
    Args:
        cfg: Base configuration.
        n_trials: Number of Optuna trials.
        split_type: Which split to optimize on.
    
    Returns:
        Dict with best params and study results.
    """
    logger.info("=" * 60)
    logger.info(f"ChromaGuide Hyperparameter Optimization")
    logger.info(f"Encoder: {cfg.model.sequence_encoder.type}")
    logger.info(f"Split: {split_type}")
    logger.info(f"Trials: {n_trials}")
    logger.info("=" * 60)
    
    # Create study
    study = optuna.create_study(
        study_name=f"chromaguide_{cfg.model.sequence_encoder.type}",
        direction="maximize",  # Maximize Spearman ρ
        sampler=TPESampler(seed=cfg.project.seed),
        pruner=HyperbandPruner(min_resource=5, max_resource=30),
        storage=f"sqlite:///results/optuna_{cfg.model.sequence_encoder.type}.db",
        load_if_exists=True,
    )
    
    study.optimize(
        lambda trial: objective(trial, cfg, split_type),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True,
    )
    
    # Results
    logger.info("\n" + "=" * 60)
    logger.info("HPO Complete!")
    logger.info(f"Best Spearman: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Save best params
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    import json
    with open(results_dir / f"hpo_best_{cfg.model.sequence_encoder.type}.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }, f, indent=2)
    
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study": study,
    }
