#!/usr/bin/env python3
"""Matched off-target benchmark: pair-aware model vs sequence-only baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
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


def one_hot_encode(seq: str, length: int = 32) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros((4, length), dtype=np.float32)
    s = seq.upper()[:length]
    for j, c in enumerate(s):
        if c in mapping:
            out[mapping[c], j] = 1.0
    return out


def encode_pair(guide_seq: str, target_seq: str, length: int = 32) -> tuple[np.ndarray, np.ndarray]:
    g = one_hot_encode(guide_seq, length)
    t = one_hot_encode(target_seq, length)
    g_idx = np.argmax(g, axis=0)
    t_idx = np.argmax(t, axis=0)
    g_valid = g.sum(axis=0) > 0
    t_valid = t.sum(axis=0) > 0
    mismatch = ((g_idx != t_idx) & g_valid & t_valid).astype(np.float32)[None, :]
    pair = np.concatenate([g, t, mismatch], axis=0)
    aux = np.array([mismatch.sum() / 16.0, 0.0], dtype=np.float32)
    return pair, aux


def load_data(data_path: str, max_rows: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    guide_only, pair_feats, aux_feats, labels = [], [], [], []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 36:
                continue
            guide_seq = parts[21]
            target_seq = parts[22]
            ident = parts[33]
            if ident not in {"ON", "OFF"}:
                continue
            if not guide_seq or not target_seq or len(guide_seq) < 20 or len(target_seq) < 20:
                continue
            label = 1.0 if ident == "OFF" else 0.0
            g = one_hot_encode(guide_seq, 32)
            p, aux = encode_pair(guide_seq, target_seq, 32)
            bulge_raw = parts[35]
            if bulge_raw != "NULL":
                try:
                    aux[1] = float(bulge_raw) / 4.0
                except Exception:
                    aux[1] = 0.0
            guide_only.append(g)
            pair_feats.append(p)
            aux_feats.append(aux)
            labels.append(label)
            if max_rows > 0 and len(labels) >= max_rows:
                break
    return (
        np.asarray(guide_only, dtype=np.float32),
        np.asarray(pair_feats, dtype=np.float32),
        np.asarray(aux_feats, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


class GuideOnlyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class PairAwareCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 256, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256 + 2, 192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(192, 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(torch.cat([h, aux], dim=1))


@dataclass
class TrainResult:
    best_val_auroc: float
    best_epoch: int
    test_auroc: float
    test_auprc: float
    test_pred: np.ndarray


def _eval_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pair_aware: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if pair_aware:
                x, aux, y = batch
                x = x.to(device)
                aux = aux.to(device)
                logits = model(x, aux)
            else:
                x, y = batch
                x = x.to(device)
                logits = model(x)
            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds.extend(prob.tolist())
            labels.extend(y.numpy().tolist())
    return np.asarray(preds, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def train_guide_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TrainResult:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    nw = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = GuideOnlyCNN().to(device)
    pos = max(1.0, float((y_train == 1).sum()))
    neg = max(1.0, float((y_train == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    patience, bad = 10, 0
    torch.manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_pred, val_lab = _eval_logits(model, val_loader, device, pair_aware=False)
        val_auc = float(roc_auc_score(val_lab, val_pred))
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred, test_lab = _eval_logits(model, test_loader, device, pair_aware=False)
    return TrainResult(
        best_val_auroc=best_auc,
        best_epoch=best_epoch,
        test_auroc=float(roc_auc_score(test_lab, test_pred)),
        test_auprc=float(average_precision_score(test_lab, test_pred)),
        test_pred=test_pred,
    )


def train_pair_aware(
    x_train: np.ndarray,
    aux_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    aux_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    aux_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TrainResult:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(aux_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(aux_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(aux_test), torch.from_numpy(y_test))
    nw = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = PairAwareCNN().to(device)
    pos = max(1.0, float((y_train == 1).sum()))
    neg = max(1.0, float((y_train == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = None
    best_auc = -1.0
    best_epoch = 0
    patience, bad = 10, 0
    torch.manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, ab, yb in train_loader:
            xb = xb.to(device)
            ab = ab.to(device)
            yb = yb.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, ab), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_pred, val_lab = _eval_logits(model, val_loader, device, pair_aware=True)
        val_auc = float(roc_auc_score(val_lab, val_pred))
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred, test_lab = _eval_logits(model, test_loader, device, pair_aware=True)
    return TrainResult(
        best_val_auroc=best_auc,
        best_epoch=best_epoch,
        test_auroc=float(roc_auc_score(test_lab, test_pred)),
        test_auprc=float(average_precision_score(test_lab, test_pred)),
        test_pred=test_pred,
    )


def paired_bootstrap_delta(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_auc = []
    delta_ap = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        if len(np.unique(yb)) < 2:
            continue
        delta_auc.append(float(roc_auc_score(yb, pb) - roc_auc_score(yb, pa)))
        delta_ap.append(float(average_precision_score(yb, pb) - average_precision_score(yb, pa)))
    da = np.asarray(delta_auc, dtype=np.float64)
    dp = np.asarray(delta_ap, dtype=np.float64)
    return {
        "delta_auroc_mean": float(da.mean()),
        "delta_auroc_ci_low": float(np.quantile(da, 0.025)),
        "delta_auroc_ci_high": float(np.quantile(da, 0.975)),
        "delta_auroc_p_leq_0": float((da <= 0).mean()),
        "delta_auprc_mean": float(dp.mean()),
        "delta_auprc_ci_low": float(np.quantile(dp, 0.025)),
        "delta_auprc_ci_high": float(np.quantile(dp, 0.975)),
        "delta_auprc_p_leq_0": float((dp <= 0).mean()),
        "n_boot_effective": int(len(da)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/raw/crisprofft/CRISPRoffT_all_targets.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--output_json", type=str, default="results/runs/offtarget_matched_eval.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)

    x_guide, x_pair, x_aux, y = load_data(args.data_path, max_rows=args.max_rows)
    print(f"Loaded N={len(y)} | ON={(y==0).sum()} OFF={(y==1).sum()}")

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y)
    tr2_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=args.seed, stratify=y[tr_idx])

    g_res = train_guide_only(
        x_guide[tr2_idx],
        y[tr2_idx],
        x_guide[va_idx],
        y[va_idx],
        x_guide[te_idx],
        y[te_idx],
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
    )
    p_res = train_pair_aware(
        x_pair[tr2_idx],
        x_aux[tr2_idx],
        y[tr2_idx],
        x_pair[va_idx],
        x_aux[va_idx],
        y[va_idx],
        x_pair[te_idx],
        x_aux[te_idx],
        y[te_idx],
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed + 1,
    )

    boot = paired_bootstrap_delta(y[te_idx], g_res.test_pred, p_res.test_pred, args.n_boot, args.seed)
    out = {
        "dataset": os.path.abspath(args.data_path),
        "seed": args.seed,
        "device": str(device),
        "n_total": int(len(y)),
        "n_train": int(len(tr2_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_on_total": int((y == 0).sum()),
        "n_off_total": int((y == 1).sum()),
        "guide_only": {
            "best_val_auroc": g_res.best_val_auroc,
            "best_epoch": g_res.best_epoch,
            "test_auroc": g_res.test_auroc,
            "test_auprc": g_res.test_auprc,
        },
        "pair_aware": {
            "best_val_auroc": p_res.best_val_auroc,
            "best_epoch": p_res.best_epoch,
            "test_auroc": p_res.test_auroc,
            "test_auprc": p_res.test_auprc,
        },
        "paired_bootstrap": boot,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out["guide_only"], indent=2))
    print(json.dumps(out["pair_aware"], indent=2))
    print(json.dumps(out["paired_bootstrap"], indent=2))
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
