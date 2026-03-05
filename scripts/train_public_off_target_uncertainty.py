#!/usr/bin/env python3
"""Train a public off-target regression + uncertainty model on staged CHANGE-seq table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def one_hot_encode(seq: str, length: int = 23) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = (seq or "").upper()[:length]
    for j, c in enumerate(s):
        idx = mapping.get(c)
        if idx is not None:
            encoded[idx, j] = 1.0
    return encoded


def encode_pair(guide_seq: str, target_seq: str, length: int = 23) -> tuple[np.ndarray, np.ndarray]:
    g = one_hot_encode(guide_seq, length=length)
    t = one_hot_encode(target_seq, length=length)
    g_idx = np.argmax(g, axis=0)
    t_idx = np.argmax(t, axis=0)
    g_valid = g.sum(axis=0) > 0
    t_valid = t.sum(axis=0) > 0
    mismatch = ((g_idx != t_idx) & g_valid & t_valid).astype(np.float32)[None, :]
    length_gap = float(abs(len((guide_seq or "")) - len((target_seq or ""))) > 0)
    pair = np.concatenate([g, t, mismatch], axis=0).astype(np.float32)
    aux = np.array([float(mismatch.sum()) / 23.0, length_gap], dtype=np.float32)
    return pair, aux


class UncertaintyOffTargetCNN(nn.Module):
    def __init__(
        self,
        base_channels: int = 192,
        fc_hidden: int = 256,
        conv_dropout: float = 0.25,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        mid = max(64, base_channels // 2)
        high = max(base_channels, base_channels * 2)
        self.conv1 = nn.Conv1d(9, base_channels, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(base_channels, high, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(high, base_channels, kernel_size=8, padding=3)
        self.conv4 = nn.Conv1d(base_channels, mid, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.bn2 = nn.BatchNorm1d(high)
        self.bn3 = nn.BatchNorm1d(base_channels)
        self.bn4 = nn.BatchNorm1d(mid)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(conv_dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.backbone = nn.Sequential(
            nn.Linear(mid + 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, max(64, fc_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
        )
        head_dim = max(64, fc_hidden // 2)
        self.mu_head = nn.Linear(head_dim, 1)
        self.logvar_head = nn.Linear(head_dim, 1)

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.gap(x).squeeze(-1)
        x = torch.cat([x, aux], dim=1)
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), min=-6.0, max=6.0)
        return mu, logvar


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (target - mu) ** 2 + logvar).mean()


def group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.20))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_groups = set(uniq[:n_train])
    val_groups = set(uniq[n_train : n_train + n_val])
    test_groups = set(uniq[n_train + n_val :])
    if not test_groups:
        test_groups = {uniq[-1]}
        val_groups.discard(uniq[-1])
    train_idx = np.array([i for i, g in enumerate(groups) if g in train_groups], dtype=np.int64)
    val_idx = np.array([i for i, g in enumerate(groups) if g in val_groups], dtype=np.int64)
    test_idx = np.array([i for i, g in enumerate(groups) if g in test_groups], dtype=np.int64)
    return train_idx, val_idx, test_idx


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def interval_coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z: float) -> float:
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def load_data(path: Path, max_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    pairs = []
    aux = []
    y = []
    groups = []
    stats = {
        "rows_scanned": 0,
        "rows_loaded": 0,
        "rows_skipped_missing_activity": 0,
        "rows_skipped_bad_activity": 0,
        "rows_skipped_missing_sequence": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["rows_scanned"] += 1
            guide = row.get("sgRNA_seq", "")
            off = row.get("off_seq", "")
            if not guide or not off:
                stats["rows_skipped_missing_sequence"] += 1
                continue

            raw_activity = (row.get("activity_log1p_read") or "").strip()
            if raw_activity == "":
                stats["rows_skipped_missing_activity"] += 1
                continue
            try:
                activity = float(raw_activity)
            except ValueError:
                stats["rows_skipped_bad_activity"] += 1
                continue

            pair, aux_row = encode_pair(guide, off, length=23)
            pairs.append(pair)
            aux.append(aux_row)
            y.append(activity)
            groups.append(row.get("sgRNA_id", "") or guide)
            stats["rows_loaded"] += 1
            if max_rows > 0 and stats["rows_loaded"] >= max_rows:
                break
    if not pairs:
        raise RuntimeError("No rows loaded from staged CHANGE-seq table.")
    return (
        np.stack(pairs).astype(np.float32),
        np.stack(aux).astype(np.float32),
        np.array(y, dtype=np.float32),
        np.array(groups),
        stats,
    )


def evaluate(
    model: nn.Module,
    x_pair: np.ndarray,
    x_aux: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    preds_mu = []
    preds_sigma = []
    ys = []
    ds = DataLoader(
        TensorDataset(
            torch.from_numpy(x_pair),
            torch.from_numpy(x_aux),
            torch.from_numpy(y),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for xb_pair, xb_aux, yb in ds:
            xb_pair = xb_pair.to(device)
            xb_aux = xb_aux.to(device)
            yb = yb.to(device)
            mu, logvar = model(xb_pair, xb_aux)
            sigma = torch.exp(0.5 * logvar)
            preds_mu.append(mu.cpu().numpy())
            preds_sigma.append(sigma.cpu().numpy())
            ys.append(yb.cpu().numpy())

    pred_mu = np.concatenate(preds_mu)
    pred_sigma = np.maximum(np.concatenate(preds_sigma), 1e-6)
    y_true = np.concatenate(ys)
    rmse = float(np.sqrt(np.mean((pred_mu - y_true) ** 2)))
    nll = float(np.mean(0.5 * (((y_true - pred_mu) / pred_sigma) ** 2 + 2 * np.log(pred_sigma))))
    out = {
        "pearson": pearson_corr(pred_mu, y_true),
        "spearman": spearman_corr(pred_mu, y_true),
        "rmse": rmse,
        "nll": nll,
        "coverage_68": interval_coverage(y_true, pred_mu, pred_sigma, 1.0),
        "coverage_90": interval_coverage(y_true, pred_mu, pred_sigma, 1.64),
        "coverage_95": interval_coverage(y_true, pred_mu, pred_sigma, 1.96),
        "pred_mean": float(np.mean(pred_mu)),
        "pred_std_mean": float(np.mean(pred_sigma)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-csv",
        default="data/public_benchmarks/off_target/secondary_change_seq/CHANGE_seq_processed_table.csv",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=192)
    parser.add_argument("--fc-hidden", type=int, default=256)
    parser.add_argument("--conv-dropout", type=float, default=0.25)
    parser.add_argument("--fc-dropout", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--output-json",
        default="results/public_benchmarks/public_off_target_uncertainty_change_seq.json",
    )
    parser.add_argument(
        "--model-out",
        default="results/public_benchmarks/public_off_target_uncertainty_change_seq.pt",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data_csv).resolve()
    out_path = Path(args.output_json).resolve()
    model_path = Path(args.model_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_pair, x_aux, y, groups, load_stats = load_data(data_path, args.max_rows)
    train_idx, val_idx, test_idx = group_split(groups, seed=args.seed)

    x_pair_train, x_aux_train, y_train = x_pair[train_idx], x_aux[train_idx], y[train_idx]
    x_pair_val, x_aux_val, y_val = x_pair[val_idx], x_aux[val_idx], y[val_idx]
    x_pair_test, x_aux_test, y_test = x_pair[test_idx], x_aux[test_idx], y[test_idx]

    device = resolve_device(args.device)
    model = UncertaintyOffTargetCNN(
        base_channels=args.base_channels,
        fc_hidden=args.fc_hidden,
        conv_dropout=args.conv_dropout,
        fc_dropout=args.fc_dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_pair_train),
            torch.from_numpy(x_aux_train),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb_pair, xb_aux, yb in train_loader:
            xb_pair = xb_pair.to(device)
            xb_aux = xb_aux.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            mu, logvar = model(xb_pair, xb_aux)
            loss = gaussian_nll(mu, logvar, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, x_pair_val, x_aux_val, y_val, args.batch_size, device)
        val_nll = float(val_metrics["nll"])
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_nll": val_nll})

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, x_pair_train, x_aux_train, y_train, args.batch_size, device)
    val_metrics = evaluate(model, x_pair_val, x_aux_val, y_val, args.batch_size, device)
    test_metrics = evaluate(model, x_pair_test, x_aux_test, y_test, args.batch_size, device)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, model_path)
    payload = {
        "data_csv": str(data_path),
        "seed": args.seed,
        "device": str(device),
        "rows": {
            "scanned": int(load_stats["rows_scanned"]),
            "total": int(len(y)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "skipped_missing_activity": int(load_stats["rows_skipped_missing_activity"]),
            "skipped_bad_activity": int(load_stats["rows_skipped_bad_activity"]),
            "skipped_missing_sequence": int(load_stats["rows_skipped_missing_sequence"]),
        },
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "base_channels": args.base_channels,
            "fc_hidden": args.fc_hidden,
            "conv_dropout": args.conv_dropout,
            "fc_dropout": args.fc_dropout,
        },
        "history": history,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "calibration_target_reference": {
            "coverage_68_expected": 0.68,
            "coverage_90_expected": 0.90,
            "coverage_95_expected": 0.95,
        },
        "notes": [
            "This run uses the staged CHANGE-seq proxy table.",
            "Use only for exploratory uncertainty tracking unless primary-source parity is verified.",
        ],
        "model_path": str(model_path),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metrics"]["test"], indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
