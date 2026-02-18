"""
Advanced Hyperparameter Tuning + NAS for CRISPRO-MAMBA-X on H100 Cluster
Implements comprehensive architecture search with Optuna
"""
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from scipy.stats import spearmanr
import json

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel

def objective(trial):
    """
    Comprehensive search space including:
    - Learning rate & optimizer settings
    - Model architecture (NAS)
    - Training hyperparameters
    - Regularization strategies
    """

    # === ARCHITECTURE SEARCH (NAS) ===
    d_model = trial.suggest_categorical("d_model", [128, 256, 384, 512])
    n_layers = trial.suggest_int("n_layers", 2, 8)

    # Advanced architectural choices
    use_causal = trial.suggest_categorical("use_causal", [True, False])
    use_quantum = trial.suggest_categorical("use_quantum", [True, False])
    use_topo = trial.suggest_categorical("use_topo", [True, False])
    use_thermo = trial.suggest_categorical("use_thermo", [True, False])

    # MoM (Mixture of Mamba) Architecture
    use_moe = trial.suggest_categorical("use_moe", [True, False])
    n_experts = trial.suggest_categorical("n_experts", [4, 8, 16])
    expert_capacity = trial.suggest_float("expert_capacity", 0.5, 2.0)

    if not use_moe:
        n_experts = 8  # Default
        expert_capacity = 1.0

    # === OPTIMIZER & LEARNING ===
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RAdam"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Learning rate schedule
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "linear", "exponential"])

    # === REGULARIZATION ===
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    gradient_clip = trial.suggest_float("gradient_clip", 0.5, 5.0)

    # === TRAINING CONFIG ===
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])  # Reduced to avoid OOM

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Architecture Search")
    print(f"  Model: D={d_model}, L={n_layers}")
    print(f"  Features: Causal={use_causal}, Quantum={use_quantum}, Topo={use_topo}, Thermo={use_thermo}")
    print(f"  MoM: Enabled={use_moe}" + (f", Experts={n_experts}, Capacity={expert_capacity:.2f}" if use_moe else ""))
    print(f"  Optimizer: {optimizer_name}, LR={lr:.2e}, WD={weight_decay:.2e}")
    print(f"  Regularization: Dropout={dropout:.3f}, GradClip={gradient_clip:.2f}")
    print(f"{'='*60}\n")

    # === SETUP ===
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load mini dataset for fast trials
    data_path = "/scratch/amird/CRISPRO-MAMBA-X/data/mini/crisprofft/mini_crisprofft.txt"
    dataset = CRISPRoffTDataset(data_path_override=data_path, context_window=4096)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === MODEL ===
    model = CRISPROModel(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=23,
        n_modalities=6,
        use_causal=use_causal,
        use_quantum=use_quantum,
        use_topo=use_topo
    ).to(device)

    # === OPTIMIZER ===
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # RAdam
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # === SCHEDULER ===
    if use_scheduler:
        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        elif scheduler_type == "linear":
            scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=5)
        else:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # === LOSS ===
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # === TRAINING LOOP (5 epochs for pruning) ===
    n_epochs = 5
    best_spearman = -1.0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            if batch is None:
                continue

            seq = batch['sequence'].to(device)
            epi = batch['epigenetics'].to(device)
            target = batch['efficiency'].to(device, dtype=torch.float32)
            bio = batch.get('biophysics', None)
            if bio is not None:
                bio = bio.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(seq, epigenetics=epi, biophysics=bio, causal=use_causal)

            pred_cls = outputs['classification']
            pred_reg = outputs['regression']

            # Loss
            target_cls = (target > 0).float().view(-1, 1)
            target_reg = target.view(-1, 1)

            loss_cls = criterion_cls(pred_cls.view(-1, 1), target_cls)
            loss_reg = criterion_reg(pred_reg.view(-1, 1), target_reg)

            loss = loss_cls + loss_reg

            # Add thermodynamic loss if enabled
            if use_thermo and 'latent' in outputs:
                entropy_loss = torch.mean(outputs['latent'] ** 2) * 0.01
                loss += entropy_loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                seq = batch['sequence'].to(device)
                epi = batch['epigenetics'].to(device)
                target = batch['efficiency'].to(device, dtype=torch.float32)
                bio = batch.get('biophysics', None)
                if bio is not None:
                    bio = bio.to(device)

                outputs = model(seq, epigenetics=epi, biophysics=bio, causal=use_causal)
                pred_reg = outputs['regression']

                preds.extend(pred_reg.view(-1).cpu().numpy())
                targets.extend(target.view(-1).cpu().numpy())

        # Compute Spearman correlation
        try:
            spearman, _ = spearmanr(targets, preds)
            if np.isnan(spearman):
                spearman = -1.0
        except:
            spearman = -1.0

        best_spearman = max(best_spearman, spearman)

        # Report to Optuna for pruning
        trial.report(spearman, epoch)

        # Early stopping
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Step scheduler
        if use_scheduler:
            scheduler.step()

        print(f"  Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss/len(train_loader):.4f}, Spearman={spearman:.4f}")

    return best_spearman


def run_study(n_trials=100, study_name="crispro_nas"):
    """Run comprehensive HP tuning study"""

    # Use SQLite storage for persistence
    storage = f"sqlite:////scratch/amird/CRISPRO-MAMBA-X/optuna_{study_name}.db"

    # Create study with advanced sampler and pruner
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    print(f"\n{'='*80}")
    print(f"STARTING COMPREHENSIVE NAS + HYPERPARAMETER OPTIMIZATION")
    print(f"Study: {study_name}")
    print(f"Target: {n_trials} trials")
    print(f"Storage: {storage}")
    print(f"{'='*80}\n")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # === RESULTS ===
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE!")
    print(f"{'='*80}\n")

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best Spearman: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_file = f"/scratch/amird/CRISPRO-MAMBA-X/results_{study_name}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study_name': study_name
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print top 5 trials
    print(f"\nTop 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)
    for i, trial in enumerate(trials_sorted[:5]):
        print(f"  {i+1}. Trial {trial.number}: {trial.value:.4f}")
        print(f"     Params: D={trial.params.get('d_model')}, L={trial.params.get('n_layers')}, LR={trial.params.get('lr'):.2e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--study_name", type=str, default="crispro_nas_v1", help="Study name")
    args = parser.parse_args()

    run_study(n_trials=args.n_trials, study_name=args.study_name)
