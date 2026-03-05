#!/usr/bin/env python3
"""Train a public off-target model on the staged CCLMoff CSV.

This supports both exploratory method filters and deterministic claim-frame
execution via manifest-defined split rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


UNRESOLVED_METHOD = "UNRESOLVED_BLANK"


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


def parse_csv_set(raw: str) -> set[str]:
    values = set()
    for item in (raw or "").split(","):
        txt = item.strip()
        if not txt:
            continue
        if txt == "__BLANK__":
            txt = UNRESOLVED_METHOD
        values.add(txt)
    return values


def one_hot_encode(seq: str, length: int = 23) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((4, length), dtype=np.float32)
    s = (seq or "").upper()[:length]
    for j, c in enumerate(s):
        if c in mapping:
            encoded[mapping[c], j] = 1.0
    return encoded


def encode_pair(guide_seq: str, target_seq: str, length: int = 23) -> tuple[np.ndarray, float, float]:
    g = one_hot_encode(guide_seq, length=length)
    t = one_hot_encode(target_seq, length=length)
    g_idx = np.argmax(g, axis=0)
    t_idx = np.argmax(t, axis=0)
    g_valid = g.sum(axis=0) > 0
    t_valid = t.sum(axis=0) > 0
    mismatch = ((g_idx != t_idx) & g_valid & t_valid).astype(np.float32)[None, :]
    length_gap = float(abs(len((guide_seq or "")) - len((target_seq or ""))) > 0)
    return np.concatenate([g, t, mismatch], axis=0), float(mismatch.sum()), length_gap


class OffTargetCNN(nn.Module):
    def __init__(
        self,
        base_channels: int = 256,
        fc_hidden: int = 256,
        conv_dropout: float = 0.4,
        fc_dropout: float = 0.3,
    ):
        super().__init__()
        mid_channels = max(64, base_channels // 2)
        high_channels = max(base_channels, base_channels * 2)
        self.conv1 = nn.Conv1d(9, base_channels, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(base_channels, high_channels, kernel_size=8, padding=3)
        self.conv3 = nn.Conv1d(high_channels, base_channels, kernel_size=8, padding=3)
        self.conv4 = nn.Conv1d(base_channels, mid_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(conv_dropout)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.bn2 = nn.BatchNorm1d(high_channels)
        self.bn3 = nn.BatchNorm1d(base_channels)
        self.bn4 = nn.BatchNorm1d(mid_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(mid_channels + 2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(conv_dropout),
            nn.Linear(fc_hidden, max(64, fc_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(max(64, fc_hidden // 2), 1),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
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
        return self.fc(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * ce_loss).mean()


def default_method_aliases() -> dict[str, str]:
    return {
        "": UNRESOLVED_METHOD,
        "DIS-seq": "DISCOVER-seq",
        "DISplus-seq": "DISCOVER-seq+",
    }


def load_method_aliases(path: str) -> dict[str, str]:
    aliases = default_method_aliases()
    if not path:
        return aliases
    payload = json.loads(Path(path).read_text())
    entries = payload.get("methods", [])
    for entry in entries:
        raw_method = entry.get("raw_method", "")
        normalized = entry.get("normalized_method")
        if normalized:
            aliases[raw_method] = normalized
    return aliases


def normalize_method(raw_method: str, aliases: dict[str, str]) -> str:
    raw = (raw_method or "").strip()
    if raw in aliases:
        return aliases[raw]
    return raw if raw else UNRESOLVED_METHOD


def build_guide_folds(unique_groups: list[str], fold_count: int, seed: int) -> list[list[str]]:
    groups = list(unique_groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    folds: list[list[str]] = []
    base = len(groups) // fold_count
    rem = len(groups) % fold_count
    start = 0
    for fold_idx in range(fold_count):
        size = base + (1 if fold_idx < rem else 0)
        end = start + size
        folds.append(groups[start:end])
        start = end
    return folds


def load_frame_manifest(args: argparse.Namespace) -> dict:
    if not args.manifest_json:
        return {}
    payload = json.loads(Path(args.manifest_json).read_text())
    status = payload.get("status", "ready")
    if status != "ready":
        reason = payload.get("blocked_reason", f"manifest status={status}")
        raise RuntimeError(f"Manifest is not runnable: {reason}")
    return payload


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    manifest = load_frame_manifest(args)
    manifest_splits = manifest.get("splits", [])
    selected_split = {}
    requested_fold_index = args.fold_index
    effective_fold_index = requested_fold_index if requested_fold_index >= 0 else int(manifest.get("fold_index_default", 0))
    if manifest_splits:
        split_index = effective_fold_index
        if not (0 <= split_index < len(manifest_splits)):
            raise RuntimeError(
                f"Requested split index {split_index} is out of range for manifest with {len(manifest_splits)} splits."
            )
        selected_split = manifest_splits[split_index]
    effective = {
        "frame_name": manifest.get("frame_name", ""),
        "manifest_json": args.manifest_json,
        "data_path": manifest.get("data_path", args.data_path),
        "method_map_json": manifest.get("method_map_json", args.method_map_json),
        "split_mode": manifest.get("split_mode", args.split_mode),
        "fold_count": int(manifest.get("fold_count", len(manifest_splits) or args.fold_count)),
        "fold_index": effective_fold_index,
        "methods": set(manifest.get("include_methods", sorted(parse_csv_set(args.methods)))),
        "exclude_methods": set(manifest.get("exclude_methods", [])),
        "train_methods": set(
            selected_split.get("train_methods", manifest.get("train_methods", sorted(parse_csv_set(args.train_methods))))
        ),
        "test_methods": set(
            selected_split.get("test_methods", manifest.get("test_methods", sorted(parse_csv_set(args.test_methods))))
        ),
        "guide_folds": manifest.get("guide_folds", []),
        "seed": int(manifest.get("seed", args.seed)),
        "selected_split": selected_split,
    }
    if effective["split_mode"] == "manifest":
        raise RuntimeError("Manifest split_mode cannot recursively be 'manifest'.")
    if args.split_mode == "manifest" and not args.manifest_json:
        raise RuntimeError("--split-mode manifest requires --manifest-json.")
    return effective


def load_cclmoff_data(
    data_path: str,
    method_aliases: dict[str, str],
    include_methods: set[str],
    exclude_methods: set[str],
    max_rows: int,
    negative_keep_prob: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    pair_feats: list[np.ndarray] = []
    aux_feats: list[list[float]] = []
    labels: list[float] = []
    groups: list[str] = []
    methods: list[str] = []

    rows_seen = 0
    with open(data_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canonical_method = normalize_method(row.get("Method", ""), method_aliases)
            if include_methods and canonical_method not in include_methods:
                continue
            if canonical_method in exclude_methods:
                continue

            label = float(row["label"])
            if label <= 0.0 and rng.random() > negative_keep_prob:
                continue

            sg = row.get("sgRNA_seq", "") or row.get("sgRNA_type", "")
            off = row.get("off_seq", "")
            if not sg or not off:
                continue

            pair_ch, mismatch_count, bulge_flag = encode_pair(sg, off, length=23)
            pair_feats.append(pair_ch.astype(np.float32))
            aux_feats.append([mismatch_count / 23.0, bulge_flag])
            labels.append(1.0 if label > 0.0 else 0.0)
            groups.append(row.get("sgRNA_type", sg))
            methods.append(canonical_method)
            rows_seen += 1
            if max_rows > 0 and rows_seen >= max_rows:
                break

    if not pair_feats:
        raise RuntimeError("No rows loaded from CCLMoff CSV with current filters.")

    return (
        np.stack(pair_feats).astype(np.float32),
        np.array(aux_feats, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(groups),
        np.array(methods),
    )


def split_by_holdout(groups: np.ndarray, seed: int, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray, dict]:
    unique = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique)
    cut = max(1, min(len(unique) - 1, int(len(unique) * train_frac)))
    train_groups = set(unique[:cut])
    train_idx = np.array([i for i, group in enumerate(groups) if group in train_groups], dtype=np.int64)
    val_idx = np.array([i for i, group in enumerate(groups) if group not in train_groups], dtype=np.int64)
    meta = {
        "split_mode": "guide_holdout",
        "train_groups": sorted(train_groups),
        "val_groups": sorted(set(unique) - train_groups),
    }
    return train_idx, val_idx, meta


def split_by_kfold(
    groups: np.ndarray,
    fold_count: int,
    fold_index: int,
    seed: int,
    manifest_folds: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    if fold_count < 2:
        raise RuntimeError("guide_kfold requires fold_count >= 2")
    if not (0 <= fold_index < fold_count):
        raise RuntimeError(f"fold_index={fold_index} is out of range for fold_count={fold_count}")

    unique = sorted(set(groups.tolist()))
    if manifest_folds:
        folds = [list(fold) for fold in manifest_folds]
    else:
        folds = build_guide_folds(unique, fold_count, seed)
    if len(folds) != fold_count:
        raise RuntimeError("Manifest fold count does not match requested fold_count.")

    val_groups = set(folds[fold_index])
    train_groups = set(unique) - val_groups
    train_idx = np.array([i for i, group in enumerate(groups) if group in train_groups], dtype=np.int64)
    val_idx = np.array([i for i, group in enumerate(groups) if group in val_groups], dtype=np.int64)
    meta = {
        "split_mode": "guide_kfold",
        "fold_count": fold_count,
        "fold_index": fold_index,
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "guide_folds": folds,
    }
    return train_idx, val_idx, meta


def split_by_methods(
    methods: np.ndarray,
    train_methods: set[str],
    test_methods: set[str],
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not train_methods or not test_methods:
        raise RuntimeError(f"{split_mode} requires both train_methods and test_methods.")
    train_idx = np.array([i for i, item in enumerate(methods) if item in train_methods], dtype=np.int64)
    val_idx = np.array([i for i, item in enumerate(methods) if item in test_methods], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(f"{split_mode} produced an empty train or validation split.")
    meta = {
        "split_mode": split_mode,
        "train_methods": sorted(train_methods),
        "val_methods": sorted(test_methods),
    }
    return train_idx, val_idx, meta


def build_split(
    groups: np.ndarray,
    methods: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    split_mode = config["split_mode"]
    if split_mode == "guide_holdout":
        return split_by_holdout(groups, config["seed"])
    if split_mode == "guide_kfold":
        return split_by_kfold(
            groups,
            fold_count=config["fold_count"],
            fold_index=config["fold_index"],
            seed=config["seed"],
            manifest_folds=config["guide_folds"],
        )
    if split_mode in {"train_test_methods", "lodo"}:
        return split_by_methods(methods, config["train_methods"], config["test_methods"], split_mode)
    raise RuntimeError(f"Unsupported split_mode: {split_mode}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_feats, batch_aux, batch_labels in train_loader:
        batch_feats = batch_feats.to(device)
        batch_aux = batch_aux.to(device)
        batch_labels = batch_labels.to(device).view(-1, 1)
        optimizer.zero_grad()
        logits = model(batch_feats, batch_aux)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)
    return total_loss / len(train_loader.dataset)


def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch_feats, batch_aux, batch_labels in val_loader:
            logits = model(batch_feats.to(device), batch_aux.to(device))
            preds = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.numpy().flatten().tolist())

    if len(set(all_labels)) < 2:
        raise RuntimeError("Validation split must contain both positive and negative labels.")
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    return float(auroc), float(auprc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/public_benchmarks/off_target/primary_cclmoff/09212024_CCLMoff_dataset.csv",
    )
    parser.add_argument(
        "--methods",
        default="CIRCLE-seq,__BLANK__",
        help="Comma-separated include filter for exploratory runs.",
    )
    parser.add_argument("--method-map-json", default="")
    parser.add_argument(
        "--split-mode",
        default="guide_holdout",
        choices=["guide_holdout", "guide_kfold", "train_test_methods", "lodo", "manifest"],
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=-1)
    parser.add_argument("--train-methods", default="")
    parser.add_argument("--test-methods", default="")
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--negative_keep_prob", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--base_channels", type=int, default=256)
    parser.add_argument("--fc_hidden", type=int, default=256)
    parser.add_argument("--conv_dropout", type=float, default=0.4)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=4)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--model_out", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = resolve_runtime_config(args)
    if args.split_mode == "manifest":
        config["split_mode"] = json.loads(Path(args.manifest_json).read_text())["split_mode"]

    method_aliases = load_method_aliases(config["method_map_json"])
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    X, aux, y, groups, methods = load_cclmoff_data(
        config["data_path"],
        method_aliases=method_aliases,
        include_methods=config["methods"],
        exclude_methods=config["exclude_methods"],
        max_rows=args.max_rows,
        negative_keep_prob=args.negative_keep_prob,
        seed=config["seed"],
    )
    print(
        f"Loaded rows: {len(y)} | positives: {int(y.sum())} | negatives: {int((y == 0).sum())}"
    )

    train_idx, val_idx, split_meta = build_split(groups, methods, config)
    X_train, X_val = X[train_idx], X[val_idx]
    aux_train, aux_val = aux[train_idx], aux[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")
    print(f"Train guides: {len(set(groups[train_idx].tolist()))} | Val guides: {len(set(groups[val_idx].tolist()))}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(aux_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(aux_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = OffTargetCNN(
        base_channels=args.base_channels,
        fc_hidden=args.fc_hidden,
        conv_dropout=args.conv_dropout,
        fc_dropout=args.fc_dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    best_auroc = 0.0
    best_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    model_out_path = args.model_out.strip() or "best_public_off_target_cclmoff.pt"

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_auroc, val_auprc = validate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | "
            f"AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f} | LR: {lr_now:.6f}"
        )
        scheduler.step(val_auroc)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_auprc = val_auprc
            best_epoch = epoch + 1
            patience_counter = 0
            model_dir = os.path.dirname(model_out_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_out_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    out_json = args.output_json.strip() or f"results/public_benchmarks/off_target_cclmoff_seed{args.seed}.json"
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "seed": config["seed"],
        "frame_name": config["frame_name"],
        "manifest_json": config["manifest_json"],
        "data_path": config["data_path"],
        "method_map_json": config["method_map_json"],
        "split_mode": config["split_mode"],
        "methods": sorted(config["methods"]),
        "exclude_methods": sorted(config["exclude_methods"]),
        "train_methods": sorted(config["train_methods"]),
        "test_methods": sorted(config["test_methods"]),
        "fold_count": config["fold_count"],
        "fold_index": config["fold_index"],
        "max_rows": args.max_rows,
        "negative_keep_prob": args.negative_keep_prob,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "fc_hidden": args.fc_hidden,
        "conv_dropout": args.conv_dropout,
        "fc_dropout": args.fc_dropout,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "best_epoch": best_epoch,
        "best_auroc": float(best_auroc),
        "best_auprc": float(best_auprc),
        "model_out": model_out_path,
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "n_train_guides": int(len(set(groups[train_idx].tolist()))),
        "n_val_guides": int(len(set(groups[val_idx].tolist()))),
        "train_method_counts": {
            method: int(np.sum(methods[train_idx] == method))
            for method in sorted(set(methods[train_idx].tolist()))
        },
        "val_method_counts": {
            method: int(np.sum(methods[val_idx] == method))
            for method in sorted(set(methods[val_idx].tolist()))
        },
        "split_meta": split_meta,
    }
    Path(out_json).write_text(json.dumps(payload, indent=2))

    print(f"Training finished. Best AUROC: {best_auroc:.4f} | Best AUPRC: {best_auprc:.4f}")
    print(f"Model saved to {model_out_path}")
    print(f"Metrics saved to {out_json}")


if __name__ == "__main__":
    main()
