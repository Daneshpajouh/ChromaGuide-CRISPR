#!/usr/bin/env python3
"""Run upstream CRISPR-FMC on frozen public on-target folds.

This script keeps claim hygiene by evaluating CRISPR-FMC on the same local split
artifacts used by our public benchmark harness, then writing per-fold and summary
JSON outputs for direct comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


CANONICAL_9 = [
    "WT",
    "ESP",
    "HF",
    "xCas9",
    "SpCas9-NG",
    "Sniper-Cas9",
    "HCT116",
    "HELA",
    "HL60",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CRISPR-FMC on public on-target folds.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--datasets", default=",".join(CANONICAL_9), help="Comma-separated dataset names")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--embed-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=220)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional split cap for smoke runs")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--cache-dir", default="data/public_benchmarks/on_target/rnafm_cache")
    ap.add_argument("--output-root", default="results/public_benchmarks/sota_crispr_fmc")
    ap.add_argument("--summary-json", default="")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda:0")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_split_rows(path: Path, max_rows: int = 0) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence"].strip().upper()
            y = float(row["efficiency"])
            rows.append((seq, y))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def onehot_23nt(seq: str) -> np.ndarray:
    code = {
        "A": [1.0, 0.0, 0.0, 0.0],
        "T": [0.0, 1.0, 0.0, 0.0],
        "C": [0.0, 0.0, 1.0, 0.0],
        "G": [0.0, 0.0, 0.0, 1.0],
        "U": [0.0, 1.0, 0.0, 0.0],
        "N": [0.0, 0.0, 0.0, 0.0],
    }
    arr = np.zeros((23, 4), dtype=np.float32)
    s = seq[:23]
    if len(s) < 23:
        s = s + "N" * (23 - len(s))
    for i, ch in enumerate(s):
        arr[i] = np.array(code.get(ch, code["N"]), dtype=np.float32)
    return arr


class LogCoshLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(pred - target + 1e-12)).mean()


class CrisprFMCDataset(Dataset):
    def __init__(self, onehot: np.ndarray, rnafm: np.ndarray, y: np.ndarray) -> None:
        self.onehot = torch.tensor(onehot, dtype=torch.float32)
        self.rnafm = torch.tensor(rnafm, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.onehot[idx]
        if x1.ndim == 3 and x1.shape[0] == 1:
            x1 = x1.squeeze(0)
        x1 = x1.permute(1, 0).unsqueeze(-1)  # [23,4] -> [4,23,1]
        return x1, self.rnafm[idx], self.y[idx]


@dataclass
class FeatureCache:
    cache_dir: Path
    embed_batch_size: int
    device: torch.device

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._seq_to_embed: dict[str, np.ndarray] = {}
        self._seq_to_onehot: dict[str, np.ndarray] = {}
        self._fm_model = None
        self._batch_converter = None

    def _ensure_fm(self) -> None:
        if self._fm_model is not None:
            return
        from fm import pretrained

        model, alphabet = pretrained.rna_fm_t12()
        model.eval()
        model.to(self.device)
        self._fm_model = model
        self._batch_converter = alphabet.get_batch_converter()

    def _cache_path(self, dataset_name: str) -> Path:
        return self.cache_dir / f"{dataset_name}_rnafm_cache.npz"

    def prime_from_dataset(self, dataset_name: str, sequences: list[str]) -> None:
        cpath = self._cache_path(dataset_name)
        uniq = []
        seen = set()
        for s in sequences:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        missing: list[str] = uniq
        if cpath.exists():
            data = np.load(cpath, allow_pickle=True)
            keys = data["keys"].tolist()
            embeds = data["embeds"]
            for i, seq in enumerate(keys):
                self._seq_to_embed[seq] = embeds[i]
                self._seq_to_onehot[seq] = onehot_23nt(seq)
            missing = [s for s in uniq if s not in self._seq_to_embed]
            if not missing:
                return

        self._ensure_fm()
        all_embeds: list[np.ndarray] = []

        for i in range(0, len(missing), self.embed_batch_size):
            batch = missing[i:i + self.embed_batch_size]
            # RNA-FM tokenization expects RNA alphabet; map T->U for embedding branch.
            batch_tuples = [(f"seq_{j}", seq.replace("T", "U")) for j, seq in enumerate(batch)]
            _, _, toks = self._batch_converter(batch_tuples)
            toks = toks.to(self.device)
            with torch.no_grad():
                out = self._fm_model(toks, repr_layers=[12])
                reps = out["representations"][12]
                # Remove BOS/EOS when present.
                if reps.shape[1] >= 25:
                    reps = reps[:, 1:-1, :]
                pooled = reps.mean(dim=1)
                pooled = pooled[:, :640].detach().cpu().numpy().astype(np.float32)
            all_embeds.extend(list(pooled))

        embed_arr = np.stack(all_embeds, axis=0)
        for i, seq in enumerate(missing):
            self._seq_to_embed[seq] = embed_arr[i]
            self._seq_to_onehot[seq] = onehot_23nt(seq)

        all_keys = np.array(sorted(self._seq_to_embed.keys()), dtype=object)
        all_embeds_np = np.stack([self._seq_to_embed[k] for k in all_keys], axis=0)
        np.savez_compressed(cpath, keys=all_keys, embeds=all_embeds_np)

    def arrays_for_rows(self, rows: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x1 = np.stack([self._seq_to_onehot[s] for s, _ in rows], axis=0)
        x2 = np.stack([self._seq_to_embed[s] for s, _ in rows], axis=0)
        y = np.array([v for _, v in rows], dtype=np.float32)
        return x1, x2, y


def eval_corr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.nan_to_num(y_true, nan=float(np.nanmean(y_true)))
    y_pred = np.nan_to_num(y_pred, nan=float(np.nanmean(y_pred)))
    if spearmanr is not None and pearsonr is not None:
        scc = float(spearmanr(y_true, y_pred)[0])
        pcc = float(pearsonr(y_true, y_pred)[0])
        return scc, pcc

    # Fallback: rank-based corr via numpy.
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        return ranks

    r1 = _rank(y_true)
    r2 = _rank(y_pred)
    scc = float(np.corrcoef(r1, r2)[0, 1])
    pcc = float(np.corrcoef(y_true, y_pred)[0, 1])
    return scc, pcc


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            p = model(x1, x2).view(-1).detach().cpu().numpy()
            preds.append(p)
            labels.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return eval_corr(y_true, y_pred)


def import_upstream_model(repo_root: Path):
    src = repo_root / "data" / "public_benchmarks" / "sources" / "CRISPR-FMC"
    if not src.exists():
        raise FileNotFoundError(f"CRISPR-FMC source not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from CRISPR_FMC_model import CRISPR_FMC  # type: ignore

    return CRISPR_FMC


def dataset_split_paths(repo_root: Path, dataset: str, fold_idx: int) -> tuple[Path, Path, Path]:
    base = repo_root / "data" / "public_benchmarks" / "on_target" / "folds" / dataset / f"fold_{fold_idx}"
    tr = base / f"{dataset}_train.csv"
    va = base / f"{dataset}_validation.csv"
    te = base / f"{dataset}_test.csv"
    return tr, va, te


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
    feature_cache: FeatureCache,
    output_root: Path,
    CRISPR_FMC,
) -> dict:
    fold_metrics = []

    # Load split rows first so smoke runs can prime only the rows they actually use.
    split_rows: list[tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]]]] = []
    seqs_for_priming: list[str] = []
    for fold in range(args.folds):
        tr_path, va_path, te_path = dataset_split_paths(repo_root, dataset, fold)
        tr_rows = read_split_rows(tr_path, max_rows=args.max_rows)
        va_rows = read_split_rows(va_path, max_rows=args.max_rows)
        te_rows = read_split_rows(te_path, max_rows=args.max_rows)
        split_rows.append((tr_rows, va_rows, te_rows))
        seqs_for_priming.extend([s for s, _ in tr_rows])
        seqs_for_priming.extend([s for s, _ in va_rows])
        seqs_for_priming.extend([s for s, _ in te_rows])

    feature_cache.prime_from_dataset(dataset, seqs_for_priming)

    for fold, (tr_rows, va_rows, te_rows) in enumerate(split_rows):
        x1_tr, x2_tr, y_tr = feature_cache.arrays_for_rows(tr_rows)
        x1_va, x2_va, y_va = feature_cache.arrays_for_rows(va_rows)
        x1_te, x2_te, y_te = feature_cache.arrays_for_rows(te_rows)

        train_ds = CrisprFMCDataset(x1_tr, x2_tr, y_tr)
        val_ds = CrisprFMCDataset(x1_va, x2_va, y_va)
        test_ds = CrisprFMCDataset(x1_te, x2_te, y_te)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = CRISPR_FMC().to(device)
        criterion = LogCoshLoss()
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_scc = -1e9
        start = time.time()

        for epoch in range(args.epochs):
            model.train()
            for x1, x2, y in train_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x1, x2).view(-1)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            val_scc, _ = evaluate(model, val_loader, device)
            if val_scc > best_val_scc:
                best_val_scc = val_scc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_scc, test_pcc = evaluate(model, test_loader, device)
        duration = time.time() - start

        fold_result = {
            "dataset": dataset,
            "fold": fold,
            "test_scc": float(test_scc),
            "test_pcc": float(test_pcc),
            "best_val_scc": float(best_val_scc),
            "train_rows": int(len(tr_rows)),
            "val_rows": int(len(va_rows)),
            "test_rows": int(len(te_rows)),
            "duration_sec": float(duration),
            "epochs": int(args.epochs),
        }
        fold_metrics.append(fold_result)

        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"fold_{fold}.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    sccs = [m["test_scc"] for m in fold_metrics]
    pccs = [m["test_pcc"] for m in fold_metrics]
    return {
        "dataset": dataset,
        "folds": fold_metrics,
        "mean_scc": float(np.mean(sccs)),
        "std_scc": float(np.std(sccs)),
        "mean_pcc": float(np.mean(pccs)),
        "std_pcc": float(np.std(pccs)),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    set_seed(args.seed)
    device = select_device(args.device)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in CANONICAL_9:
            raise ValueError(f"Unknown dataset: {d}")

    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    feature_cache = FeatureCache(
        cache_dir=(repo_root / args.cache_dir).resolve(),
        embed_batch_size=args.embed_batch_size,
        device=device,
    )
    CRISPR_FMC = import_upstream_model(repo_root)

    dataset_summaries = {}
    for dataset in datasets:
        dataset_summaries[dataset] = run_dataset(
            dataset=dataset,
            args=args,
            repo_root=repo_root,
            device=device,
            feature_cache=feature_cache,
            output_root=output_root,
            CRISPR_FMC=CRISPR_FMC,
        )

    summary = {
        "model": "CRISPR-FMC-upstream",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "datasets": datasets,
        "config": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embed_batch_size": args.embed_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_rows": args.max_rows,
            "seed": args.seed,
        },
        "dataset_summaries": dataset_summaries,
        "mean_scc_across_datasets": float(np.mean([dataset_summaries[d]["mean_scc"] for d in datasets])),
        "mean_pcc_across_datasets": float(np.mean([dataset_summaries[d]["mean_pcc"] for d in datasets])),
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_root / "SUMMARY.json"
    if not summary_path.is_absolute():
        summary_path = (repo_root / summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
