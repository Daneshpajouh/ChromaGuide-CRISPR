#!/usr/bin/env python3
"""DeepHF-style on-target trainer (Embedding + BiLSTM + epigenomic MLP)."""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dir_for(split: str) -> Path:
    mapping = {
        "A": "split_a_gene_held_out",
        "B": "split_b_dataset_held_out",
        "C": "split_c_cellline_held_out",
    }
    return Path("data/processed") / mapping[split]


def resolve_pretrain_data_dir(pretrain_data_dir_arg: str) -> Path:
    """Resolve DeepHF upstream data directory across local and cluster layouts."""
    if pretrain_data_dir_arg:
        return Path(pretrain_data_dir_arg).expanduser()

    env_dir = os.environ.get("DEEPHF_UPSTREAM_DATA_DIR")
    candidates = [
        Path(env_dir).expanduser() if env_dir else None,
        Path(__file__).resolve().parents[2] / "DeepHF_upstream_2" / "data",
        Path.cwd().parent / "DeepHF_upstream_2" / "data",
        Path.home() / "scratch" / "DeepHF_upstream_2" / "data",
        Path("/scratch/amird/DeepHF_upstream_2/data"),
        Path("/Users/studio/Desktop/Projects/PhD/DeepHF_upstream_2/data"),
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c

    return Path("/scratch/amird/DeepHF_upstream_2/data")


def _read_split_partition(sdir: Path, partition: str) -> pd.DataFrame:
    dfs = []
    for f in sorted(sdir.glob(f"*_{partition}.csv")):
        cell_line = f.name.replace(f"_{partition}.csv", "")
        df = pd.read_csv(f)
        df["cell_line"] = cell_line
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No files found for partition '{partition}' in {sdir}")
    return pd.concat(dfs).reset_index(drop=True)


def _add_cellline_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    base_idx: int = 11,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    cell_lines = sorted(set(train["cell_line"]).union(val["cell_line"]).union(test["cell_line"]))
    extra_cols = []
    for i, cl in enumerate(cell_lines):
        col = f"feat_{base_idx + i}"
        extra_cols.append(col)
        train[col] = (train["cell_line"] == cl).astype(np.float32)
        val[col] = (val["cell_line"] == cl).astype(np.float32)
        test[col] = (test["cell_line"] == cl).astype(np.float32)
    return train, val, test, extra_cols


def load_split_data(split: str, use_cellline_feature: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    sdir = split_dir_for(split)
    train = _read_split_partition(sdir, "train")
    val = _read_split_partition(sdir, "validation")
    test = _read_split_partition(sdir, "test")

    feat_cols = [f"feat_{i}" for i in range(11)]
    if use_cellline_feature:
        train, val, test, extra_cols = _add_cellline_features(train, val, test)
        feat_cols.extend(extra_cols)

    return train, val, test, feat_cols


def seqs_to_tokens(seqs: list[str]) -> np.ndarray:
    # DeepHF tokenization: PAD=0, START=1, A=2, T=3, C=4, G=5
    m = {"A": 2, "T": 3, "C": 4, "G": 5}
    out = np.zeros((len(seqs), 22), dtype=np.int64)
    out[:, 0] = 1
    for i, s in enumerate(seqs):
        s = str(s).upper()
        for j, ch in enumerate(s[:21], start=1):
            out[i, j] = m.get(ch, 0)
    return out


def load_pretrain_arrays_hfesp(upstream_data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, fs, ys = [], [], []
    for name in ("hf_seq_data_array.pkl", "esp_seq_data_array.pkl"):
        with open(upstream_data_dir / name, "rb") as f:
            x, feats, y = pickle.load(f)
        xs.append(np.asarray(x, dtype=np.int64))
        fs.append(np.asarray(feats, dtype=np.float32))
        ys.append(np.asarray(y, dtype=np.float32))
    x_all = np.concatenate(xs, axis=0)
    f_all = np.concatenate(fs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return x_all, f_all, y_all


def pairwise_rank_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = pred.view(-1)
    t = y.view(-1)
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dt = t.unsqueeze(1) - t.unsqueeze(0)
    mask = dt != 0
    if mask.sum() == 0:
        return torch.zeros([], device=pred.device, dtype=pred.dtype)
    labels = (dt > 0).float()
    loss = F.binary_cross_entropy_with_logits(dp, labels, reduction="none")
    return loss[mask].mean()


class DeepHFStyle(nn.Module):
    def __init__(
        self,
        epi_dim: int,
        em_dim: int = 44,
        lstm_hidden: int = 60,
        fc_units: int = 320,
        fc_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(7, em_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=em_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        seq_dim = 22 * 2 * lstm_hidden
        in_dim = seq_dim + epi_dim
        layers = []
        for _ in range(fc_layers):
            layers.extend([nn.Linear(in_dim, fc_units), nn.ELU(), nn.Dropout(dropout)])
            in_dim = fc_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, seq_tokens: torch.Tensor, epi: torch.Tensor) -> torch.Tensor:
        x = self.embedding(seq_tokens)
        x = self.emb_drop(x)
        x, _ = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, epi], dim=1)
        x = self.mlp(x)
        return self.head(x)


def make_arrays(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = seqs_to_tokens(df["sequence"].tolist())
    epi = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["efficiency"].to_numpy(dtype=np.float32)
    return seq, epi, y


def run_epoch(
    model: DeepHFStyle,
    optimizer: torch.optim.Optimizer,
    seq: np.ndarray,
    epi: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    shuffle_seed: int,
    rank_lambda: float,
) -> tuple[float, float, float]:
    model.train()
    rng = np.random.default_rng(shuffle_seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)

    total = 0.0
    total_reg = 0.0
    total_rank = 0.0
    n_batches = 0

    for i in range(0, len(idx), batch_size):
        bi = idx[i : i + batch_size]
        s = torch.tensor(seq[bi], dtype=torch.long, device=device)
        e = torch.tensor(epi[bi], dtype=torch.float32, device=device)
        t = torch.tensor(y[bi], dtype=torch.float32, device=device).unsqueeze(1)

        pred = model(s, e)
        reg_loss = F.smooth_l1_loss(pred, t)
        r_loss = pairwise_rank_loss(pred, t)
        loss = reg_loss + rank_lambda * r_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())
        total_reg += float(reg_loss.item())
        total_rank += float(r_loss.item())
        n_batches += 1

    d = max(1, n_batches)
    return total / d, total_reg / d, total_rank / d


@torch.no_grad()
def predict(model: DeepHFStyle, seq: np.ndarray, epi: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(seq), batch_size):
        s = torch.tensor(seq[i : i + batch_size], dtype=torch.long, device=device)
        e = torch.tensor(epi[i : i + batch_size], dtype=torch.float32, device=device)
        p = model(s, e)
        out.extend(p.cpu().numpy().flatten())
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser(description="DeepHF-style on-target training")
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--em_dim", type=int, default=44)
    p.add_argument("--lstm_hidden", type=int, default=60)
    p.add_argument("--fc_units", type=int, default=320)
    p.add_argument("--fc_layers", type=int, default=3)
    p.add_argument("--rank_lambda", type=float, default=0.2)
    p.add_argument("--holdout_frac", type=float, default=0.05)
    p.add_argument("--finetune_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--pretrain", choices=["none", "hfesp"], default="hfesp")
    p.add_argument("--pretrain_epochs", type=int, default=5)
    p.add_argument("--pretrain_max_rows", type=int, default=0, help="If >0, subsample pretrain rows for faster sweeps")
    p.add_argument(
        "--pretrain_data_dir",
        type=str,
        default="",
        help="Path to DeepHF_upstream_2/data (auto-resolved if omitted).",
    )
    p.add_argument("--use_cellline_feature", dest="use_cellline_feature", action="store_true")
    p.add_argument("--no_cellline_feature", dest="use_cellline_feature", action="store_false")
    p.set_defaults(use_cellline_feature=False)
    p.add_argument("--output_prefix", type=str, default="results/runs_fixedmap/deephf_style")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df, val_df, test_df, feat_cols = load_split_data(args.split, args.use_cellline_feature)
    trainval = pd.concat([train_df, val_df]).reset_index(drop=True)
    holdout_idx = trainval.sample(frac=args.holdout_frac, random_state=args.seed).index
    holdout_df = trainval.loc[holdout_idx].reset_index(drop=True)
    fit_df = trainval.drop(holdout_idx).reset_index(drop=True)

    fit_seq, fit_epi, fit_y = make_arrays(fit_df, feat_cols)
    hold_seq, hold_epi, hold_y = make_arrays(holdout_df, feat_cols)
    test_seq, test_epi, test_y = make_arrays(test_df, feat_cols)

    model = DeepHFStyle(
        epi_dim=len(feat_cols),
        em_dim=args.em_dim,
        lstm_hidden=args.lstm_hidden,
        fc_units=args.fc_units,
        fc_layers=args.fc_layers,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Using device: {device}", flush=True)
    print(
        f"Config: split={args.split}, seed={args.seed}, pretrain={args.pretrain}, "
        f"rank_lambda={args.rank_lambda}, use_cellline_feature={args.use_cellline_feature}",
        flush=True,
    )
    print(f"Sizes: fit={len(fit_df)} holdout={len(holdout_df)} test={len(test_df)}", flush=True)

    if args.pretrain == "hfesp":
        pretrain_data_dir = resolve_pretrain_data_dir(args.pretrain_data_dir)
        print(f"Pretrain data dir: {pretrain_data_dir}", flush=True)
        x_pre, f_pre, y_pre = load_pretrain_arrays_hfesp(pretrain_data_dir)
        if args.pretrain_max_rows > 0 and args.pretrain_max_rows < len(y_pre):
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(len(y_pre), size=args.pretrain_max_rows, replace=False)
            x_pre = x_pre[keep]
            f_pre = f_pre[keep]
            y_pre = y_pre[keep]
        # Add zero columns when training uses extra features (e.g., cell-line one-hot).
        if f_pre.shape[1] < len(feat_cols):
            pad = np.zeros((len(f_pre), len(feat_cols) - f_pre.shape[1]), dtype=np.float32)
            f_pre = np.concatenate([f_pre, pad], axis=1)
        print(f"Pretrain rows: {len(y_pre)}", flush=True)
        for epoch in range(1, args.pretrain_epochs + 1):
            total, reg, rank = run_epoch(
                model,
                opt,
                x_pre,
                f_pre,
                y_pre,
                args.batch_size,
                device,
                args.seed + epoch,
                rank_lambda=0.0,
            )
            print(
                f"Pretrain epoch {epoch} | total={total:.4f} | reg={reg:.4f} | rank={rank:.4f}",
                flush=True,
            )

    best_hold = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, args.finetune_epochs + 1):
        total, reg, rank = run_epoch(
            model,
            opt,
            fit_seq,
            fit_epi,
            fit_y,
            args.batch_size,
            device,
            1000 + args.seed + epoch,
            rank_lambda=args.rank_lambda,
        )
        hold_pred = predict(model, hold_seq, hold_epi, args.batch_size, device)
        hold_rho = float(spearmanr(hold_y, hold_pred)[0])
        if not np.isfinite(hold_rho):
            hold_rho = -1.0

        history.append(
            {
                "epoch": epoch,
                "total_loss": total,
                "reg_loss": reg,
                "rank_loss": rank,
                "holdout_rho": hold_rho,
            }
        )

        print(
            f"Finetune epoch {epoch} | total={total:.4f} | reg={reg:.4f} | "
            f"rank={rank:.4f} | holdout_rho={hold_rho:.4f}",
            flush=True,
        )

        if hold_rho > best_hold:
            best_hold = hold_rho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred = predict(model, test_seq, test_epi, args.batch_size, device)
    gold_rho = float(spearmanr(test_y, test_pred)[0])

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_prefix.with_suffix(".json")
    pred_path = out_prefix.with_name(out_prefix.name + "_predictions.csv")

    payload = {
        "split": args.split,
        "seed": args.seed,
        "model": "deephf_style_bilstm",
        "pretrain": args.pretrain,
        "rank_lambda": args.rank_lambda,
        "use_cellline_feature": args.use_cellline_feature,
        "feature_cols": feat_cols,
        "best_holdout_rho": best_hold,
        "gold_rho": gold_rho,
        "history": history,
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(
        {
            "sequence": test_df["sequence"].values,
            "efficiency": test_y,
            "prediction": test_pred,
        }
    ).to_csv(pred_path, index=False)

    print(f"GOLD rho: {gold_rho:.6f}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Saved predictions: {pred_path}", flush=True)


if __name__ == "__main__":
    main()
