"""CLI entry point for ChromaGuide."""
import click
from pathlib import Path


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ChromaGuide: Multi-Modal CRISPR-Cas9 sgRNA Design Framework."""
    pass


@main.command()
@click.option("--config", "-c", type=str, default=None, help="Path to experiment config YAML")
@click.option("--stage", type=click.Choice(["download", "preprocess", "splits"]), required=True)
def data(config, stage):
    """Data acquisition and preprocessing pipeline."""
    from chromaguide.utils import load_config
    cfg = load_config(config)
    
    if stage == "download":
        from chromaguide.data.acquire import download_all
        download_all(cfg)
    elif stage == "preprocess":
        from chromaguide.data.preprocess import preprocess_all
        preprocess_all(cfg)
    elif stage == "splits":
        from chromaguide.data.splits import build_all_splits
        build_all_splits(cfg)


@main.command()
@click.option("--config", "-c", type=str, default=None, help="Path to experiment config YAML")
@click.option("--backbone", type=str, default=None, help="Override sequence encoder type")
@click.option("--seed", type=int, default=None, help="Override random seed")
@click.option("--split", type=click.Choice(["A", "B", "C"]), default="A")
@click.option("--wandb/--no-wandb", default=True, help="Enable W&B logging")
@click.argument("overrides", nargs=-1)
def train(config, backbone, seed, split, wandb, overrides):
    """Train ChromaGuide model."""
    from chromaguide.utils import load_config, set_seed, get_device, setup_logger
    
    override_list = list(overrides)
    if backbone:
        override_list.append(f"model.sequence_encoder.type={backbone}")
    if seed:
        override_list.append(f"project.seed={seed}")
    
    cfg = load_config(config, override_list if override_list else None)
    set_seed(cfg.project.seed)
    logger = setup_logger(log_dir="results/logs")
    device = get_device()
    
    from chromaguide.training.trainer import Trainer
    trainer = Trainer(cfg, device=device, split_type=split, use_wandb=wandb, logger=logger)
    trainer.fit()


@main.command()
@click.option("--config", "-c", type=str, default=None)
@click.option("--checkpoint", type=str, required=True, help="Path to model checkpoint")
@click.option("--split", type=click.Choice(["A", "B", "C"]), default="A")
def evaluate(config, checkpoint, split):
    """Evaluate a trained ChromaGuide model."""
    from chromaguide.utils import load_config, set_seed, get_device
    cfg = load_config(config)
    set_seed(cfg.project.seed)
    device = get_device()
    
    from chromaguide.evaluation.evaluate import run_evaluation
    run_evaluation(cfg, checkpoint, split, device)


@main.command()
@click.option("--config", "-c", type=str, default=None)
@click.option("--n-trials", type=int, default=50)
@click.option("--split", type=click.Choice(["A", "B", "C"]), default="A")
def hpo(config, n_trials, split):
    """Run Optuna hyperparameter optimization."""
    from chromaguide.utils import load_config, set_seed
    cfg = load_config(config)
    set_seed(cfg.project.seed)
    
    from chromaguide.training.hpo import run_hpo
    run_hpo(cfg, n_trials=n_trials, split_type=split)


@main.command()
@click.option("--config", "-c", type=str, default=None)
def offtarget(config):
    """Train the off-target prediction module."""
    from chromaguide.utils import load_config, set_seed, get_device
    cfg = load_config(config)
    set_seed(cfg.project.seed)
    device = get_device()
    
    from chromaguide.training.train_offtarget import train_offtarget
    train_offtarget(cfg, device)


@main.command()
@click.option("--results-dir", type=str, default="results/")
def thesis(results_dir):
    """Generate thesis-ready figures, tables, and statistical analysis."""
    from chromaguide.evaluation.thesis_outputs import generate_all
    generate_all(results_dir)


if __name__ == "__main__":
    main()
