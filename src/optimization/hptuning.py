"""
Hyperparameter Tuning for CRISPRO-MAMBA-X using Optuna.
Objective: Maximize Spearman Correlation on Validation Set.
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from src.utils.loss import CombinedLoss, HybridLoss
import numpy as np
from scipy.stats import spearmanr
import argparse
import sys
import os

def objective(trial):
    # --- Hyperparameters to Tune ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    gradient_clip = trial.suggest_float("gradient_clip", 0.1, 2.0)

    # Optional: Architecture search parameters
    # d_state = trial.suggest_categorical("d_state", [16, 32, 64])

    print(f"\nTrial {trial.number}: LR={lr}, D={d_model}, L={n_layers}, Clip={gradient_clip}")

    # --- Setup ---
    # Add project root to path
    if 'src' not in sys.modules:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if root_dir not in sys.path:
            sys.path.append(root_dir)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Use mini dataset for speed during tuning
    # Assuming standard mini path
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/mini/crisprofft/mini_crisprofft.txt'))
    dataset = CRISPRoffTDataset(data_path_override=data_path, context_window=4096)

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model
    model = CRISPROModel(
        d_model=d_model,
        n_layers=n_layers,
        n_modalities=6,
        vocab_size=23,
        use_causal=True, # Enable advanced features for tuning
        use_quantum=True,
        use_topo=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Loss
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    # Pruning
    # Train for limited epochs (e.g., 5) to check promise
    n_epochs = 3 # Fast tuning

    best_spearman = -1.0

    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            if batch is None: continue
            seq = batch['sequence'].to(device)
            epi = batch['epigenetics'].to(device)
            target = batch['efficiency'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(seq, epigenetics=epi, causal=True)

            p_cls = outputs['classification']
            p_reg = outputs['regression']

            # Simple loss for tuning
            loss = criterion_reg(p_reg.view(-1,1), target.view(-1,1)) + criterion_cls(p_cls.view(-1,1), (target>0).float().view(-1,1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Validation
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(device)
                epi = batch['epigenetics'].to(device)
                target = batch['efficiency'].to(device, dtype=torch.float32)

                outputs = model(seq, epigenetics=epi, causal=True)
                p_reg = outputs['regression']

                preds.extend(p_reg.view(-1).cpu().numpy())
                targets.extend(target.view(-1).cpu().numpy())

        try:
            spearman, _ = spearmanr(targets, preds)
        except:
            spearman = 0.0

        # Report to Optuna
        trial.report(spearman, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_spearman = max(best_spearman, spearman)

    return best_spearman

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters: ", study.best_params)
    print("Best Spearman: ", study.best_value)
