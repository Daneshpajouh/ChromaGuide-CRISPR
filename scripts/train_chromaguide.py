#!/usr/bin/env python3
"""Starter training script for ChromaGuide smoke tests.

Features:
- Small synthetic dataset generator for smoke testing
- Trains the baseline `BaselineCNNBiGRU` encoder + regression head
- Supports model selection: baseline, dnabert, mamba, hybrid, evo2, ensemble (with safe fallbacks)
- Evaluation with Spearman, Cohen's d, AUROC from `src.evaluation.metrics`
- Checkpointing and early stopping

Usage examples:
  python scripts/train_chromaguide.py --model baseline --epochs 10

"""
from __future__ import annotations
import argparse
import csv
import os
import random
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise ImportError("PyTorch is required to run training. Install torch before running this script.") from e

import numpy as np

from src.evaluation.metrics import spearman_correlation, cohen_d, auroc
from src.config.chromaguide_config import TrainConfig, SUPPORTED_MODELS, DEFAULT_MODEL
from src.utils.device import DeviceManager, get_best_device, warn_if_pythonpath

LOG = logging.getLogger("train_chromaguide")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_synthetic_dataset(n_samples: int = 200, seq_len: int = 23, vocab_size: int = 5) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor]:
    """Create a small synthetic dataset suitable for baseline encoder.

    Returns:
        input_ids: LongTensor (N, L) values in {1..vocab_size-1} (0 reserved for padding)
        targets: FloatTensor (N,) regression targets in [0,1]
        binary_labels: LongTensor (N,) derived from targets for AUROC
    """
    N = n_samples
    L = seq_len
    # integer tokens 1..(vocab_size-1)
    input_ids = np.random.randint(1, vocab_size, size=(N, L), dtype=np.int64)

    # create a synthetic signal: weighted sum of a few positions + noise
    weights = np.zeros(L)
    informative_positions = list(range(4, 10))
    for i, p in enumerate(informative_positions):
        weights[p] = 1.0 + 0.2 * i

    # map tokens to simple numeric contributions (A,C,G,T -> 1..4)
    token_vals = input_ids.astype(float)
    scores = (token_vals * weights).sum(axis=1)
    # normalize to [0,1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # add small noise
    scores = scores * 0.9 + 0.05 * np.random.randn(N)
    scores = np.clip(scores, 0.0, 1.0)

    binary = (scores > 0.5).astype(np.int64)

    return torch.from_numpy(input_ids).long(), torch.from_numpy(scores.astype(np.float32)), torch.from_numpy(binary)


class RegressionModel(nn.Module):
    def __init__(self, encoder: nn.Module, out_dim: int = 1):
        super().__init__()
        self.encoder = encoder
        enc_dim = getattr(encoder, 'out_dim', None)
        if enc_dim is None:
            # try run a dummy through encoder if possible
            enc_dim = 64
        self.head = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x, attention_mask=None):
        z = self.encoder(x, attention_mask=attention_mask)
        out = self.head(z).squeeze(-1)
        return out


def predict_on_target(model, xb):
    out = model(xb)
    if isinstance(out, dict):
        if 'on_target' in out:
            return out['on_target']
        if 'on' in out:
            return out['on']
        if 'pred' in out:
            return out['pred']
        raise KeyError("Model output dict missing on_target/on/pred keys")
    return out


def train_one_epoch(model, dataloader, optimizer, device_manager):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for xb, yb in dataloader:
        # move inputs to device via DeviceManager
        try:
            xb = device_manager.tensor_to_device(xb)
            yb = device_manager.tensor_to_device(yb)
        except Exception:
            # fallback if a raw device was passed
            if hasattr(device_manager, 'type'):
                xb = xb.to(device_manager)
                yb = yb.to(device_manager)
        optimizer.zero_grad()
        preds = predict_on_target(model, xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device_manager):
    model.eval()
    preds_list = []
    ys = []
    with torch.no_grad():
        for xb, yb in dataloader:
            try:
                xb = device_manager.tensor_to_device(xb)
            except Exception:
                if hasattr(device_manager, 'type'):
                    xb = xb.to(device_manager)
            out = predict_on_target(model, xb)
            preds_list.append(out.detach().cpu().numpy())
            ys.append(yb.numpy())

    preds = np.concatenate(preds_list, axis=0).ravel()
    ys = np.concatenate(ys, axis=0).ravel()

    # spearman
    s = spearman_correlation(ys, preds)

    # cohen's d between top and bottom quartile of predictions
    q1 = np.percentile(preds, 25)
    q3 = np.percentile(preds, 75)
    low_idx = preds <= q1
    high_idx = preds >= q3
    if low_idx.sum() < 2 or high_idx.sum() < 2:
        d = 0.0
    else:
        d = cohen_d(ys[high_idx], ys[low_idx])

    # auroc requires binary labels. Create binary labels from ys (threshold 0.5)
    binary = (ys > 0.5).astype(int)
    try:
        a = auroc(binary, preds)
    except Exception:
        a = 0.0

    return dict(spearman=s, cohen_d=d, auroc=a, preds=preds, y=ys)


def save_checkpoint(model, optimizer, epoch, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch}, path)


def setup_logging(log_dir: str, run_name: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path)]
    logging.basicConfig(level=logging.INFO, handlers=handlers, format='%(asctime)s | %(levelname)s | %(message)s')
    return log_path


def init_metrics_csv(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "spearman", "auroc", "cohen_d"])


def append_metrics_csv(path: str, epoch: int, train_loss: float, val_loss: float, metrics: Dict[str, float]):
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{metrics['spearman']:.6f}", f"{metrics['auroc']:.6f}", f"{metrics['cohen_d']:.6f}"])


class MultiTaskFromEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        enc_dim = getattr(encoder, 'out_dim', 64)
        self.on_head = nn.Sequential(nn.Linear(enc_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.off_head = nn.Sequential(nn.Linear(enc_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.indel_head = nn.Sequential(nn.Linear(enc_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, attention_mask=None):
        z = self.encoder(x, attention_mask=attention_mask)
        on = self.on_head(z).squeeze(-1)
        off = torch.sigmoid(self.off_head(z).squeeze(-1))
        indel = self.indel_head(z).squeeze(-1)
        return {"on_target": on, "off_target": off, "indel": indel}


def build_encoder(model_name: str) -> nn.Module:
    if model_name == 'baseline':
        from src.models.baseline_cnn_bigru import BaselineCNNBiGRU
        return BaselineCNNBiGRU()

    if model_name == 'dnabert':
        from src.models.dnabert_encoder import DNABERTEncoder
        enc = DNABERTEncoder()

        class _Wrap(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc
                self.out_dim = 64
                self.proj = nn.LazyLinear(self.out_dim)

            def forward(self, input_ids, attention_mask=None):
                seqs = ["".join(['ACGT'[int(t-1)] if 1 <= int(t) <= 4 else 'N' for t in row.tolist()]) for row in input_ids]
                emb = self.enc.embed(seqs)
                if torch.is_tensor(emb):
                    emb = emb.to(next(self.parameters()).device)
                    return self.proj(emb)
                return torch.randn((len(seqs), self.out_dim), device=next(self.parameters()).device)

        return _Wrap(enc)

    if model_name == 'mamba':
        from src.models.mamba_encoder import MambaEncoder
        enc = MambaEncoder()

        class _WrapM(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc
                self.out_dim = 64
                self.proj = nn.LazyLinear(self.out_dim)

            def forward(self, input_ids, attention_mask=None):
                B, L = input_ids.shape
                x = torch.randn(B, L, 256, device=input_ids.device)
                out = self.enc.forward(x)
                if isinstance(out, torch.Tensor) and out.ndim == 3:
                    out = out.mean(dim=1)
                if not torch.is_tensor(out):
                    out = torch.randn(B, 256, device=input_ids.device)
                return self.proj(out)

        return _WrapM(enc)

    raise ValueError(f"Unknown encoder model: {model_name}")


def build_model(model_name: str) -> nn.Module:
    if model_name in ('baseline', 'dnabert', 'mamba'):
        encoder = build_encoder(model_name)
        return RegressionModel(encoder)

    if model_name == 'hybrid':
        from src.models.hybrid_dnabert_mamba import HybridDNABERTMamba
        return HybridDNABERTMamba()

    if model_name == 'evo2':
        from src.models.evo2_adapter import Evo2AdapterModel
        return Evo2AdapterModel()

    if model_name == 'ensemble':
        from src.models.ensemble_meta_learner import EnsembleMetaLearner
        base_models = [MultiTaskFromEncoder(build_encoder(n)) for n in ('baseline', 'dnabert', 'mamba')]
        return EnsembleMetaLearner(base_models)

    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--metrics-csv', type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config:
        cfg = TrainConfig.from_json(args.config)

    cfg.merge_overrides({
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "log_dir": args.log_dir,
        "run_name": args.run_name,
        "metrics_csv": args.metrics_csv,
    })

    if cfg.model is None:
        cfg.model = DEFAULT_MODEL

    if cfg.device is None:
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.run_name is None:
        cfg.run_name = f"chromaguide_{cfg.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cfg.resolve_paths()

    log_path = setup_logging(cfg.log_dir, cfg.run_name)
    LOG.info(f"Starting training: model={cfg.model} device={cfg.device}")
    LOG.info(f"Run name: {cfg.run_name} | Log file: {log_path}")

    # warn about PYTHONPATH which can break environment isolation
    try:
        warn_if_pythonpath()
    except Exception:
        pass

    set_seed(cfg.seed)

    # create synthetic data
    X, y_reg, y_bin = make_synthetic_dataset(n_samples=400, seq_len=23, vocab_size=6)
    # split
    n = X.size(0)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y_reg[train_idx]
    X_val = X[val_idx]
    y_val = y_reg[val_idx]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # create device manager (auto-detect or use cfg.device)
    device_manager = DeviceManager(cfg.device)
    device = device_manager.device if hasattr(device_manager, 'device') else torch.device(cfg.device)

    # model selection with safe fallback to baseline
    try:
        model = build_model(cfg.model)
    except Exception as e:
        LOG.warning(f"Failed to build model '{cfg.model}': {e}. Falling back to baseline.")
        model = build_model('baseline')

    # move model to device using DeviceManager
    try:
        model = device_manager.to_device(model)
    except Exception:
        try:
            model.to(device)
        except Exception:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    if cfg.metrics_csv:
        init_metrics_csv(cfg.metrics_csv)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device_manager)
        metrics = evaluate(model, val_loader, device_manager)
        val_loss = ((metrics['y'] - metrics['preds']) ** 2).mean()

        LOG.info(f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={metrics['spearman']:.4f} auroc={metrics['auroc']:.4f} cohen_d={metrics['cohen_d']:.4f}")

        if cfg.metrics_csv:
            append_metrics_csv(cfg.metrics_csv, epoch, train_loss, val_loss, metrics)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, cfg.checkpoint)
            LOG.info(f"Saved new best checkpoint at epoch {epoch} -> {cfg.checkpoint}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            LOG.info(f"Early stopping (no improvement for {cfg.patience} epochs). Best epoch: {best_epoch}")
            break

    LOG.info("Training finished.")


if __name__ == '__main__':
    main()
