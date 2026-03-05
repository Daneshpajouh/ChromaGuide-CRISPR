#!/usr/bin/env python3
"""
On-target training with ranking-aware objective.

This script keeps the base ChromaGuide beta-regression loss and adds a
pairwise ranking loss (RankNet-style BCE on score differences) so training
aligns better with Spearman rho optimization.
"""

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
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from chromaguide.chromaguide_model import ChromaGuideModel

NT_LOOKUP = np.full(256, 4, dtype=np.int64)
for ch, idx in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("N", 4)):
    NT_LOOKUP[ord(ch)] = idx
    NT_LOOKUP[ord(ch.lower())] = idx
ONEHOT_TABLE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25, 0.25],
    ],
    dtype=np.float32,
)


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


def decode_sequences(encoded: np.ndarray) -> list[str]:
    # DeepHF tokenization: PAD=0, START=1, A=2, T=3, C=4, G=5
    bases = {2: "A", 3: "T", 4: "C", 5: "G"}
    out = []
    for s in encoded:
        core = s[1:22] if len(s) >= 22 else s[:21]
        out.append("".join(bases.get(int(b), "N") for b in core))
    return out


def load_pretrain_df_hfesp(upstream_data_dir: Path, seed: int) -> pd.DataFrame:
    dfs = []
    for name in ("hf_seq_data_array.pkl", "esp_seq_data_array.pkl"):
        with open(upstream_data_dir / name, "rb") as f:
            x, feats, y = pickle.load(f)
        d = {"sequence": decode_sequences(x), "efficiency": y}
        for i in range(feats.shape[1]):
            d[f"feat_{i}"] = feats[:, i]
        dfs.append(pd.DataFrame(d))
    return pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare_seq(seqs: list[str], device: torch.device) -> torch.Tensor:
    L = 21
    idx = np.full((len(seqs), L), 4, dtype=np.int64)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)[:L]
        if b.size:
            idx[i, : b.size] = NT_LOOKUP[b]
    arr = ONEHOT_TABLE[idx]
    arr = np.transpose(arr, (0, 2, 1))
    return torch.from_numpy(arr).to(device=device)


def prepare_epi(batch: pd.DataFrame, feature_cols: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(batch[feature_cols].values.astype(np.float32), device=device).unsqueeze(2)


def pairwise_rank_loss(pred_mu: torch.Tensor, y: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    """RankNet-style pairwise BCE on sampled comparable pairs."""
    p = pred_mu.view(-1)
    t = y.view(-1)
    bsz = int(p.numel())
    if bsz < 2:
        return torch.zeros([], device=pred_mu.device, dtype=pred_mu.dtype)

    n_pairs = min(max_pairs, bsz * (bsz - 1))
    i = torch.randint(0, bsz, (n_pairs,), device=pred_mu.device)
    j = torch.randint(0, bsz, (n_pairs,), device=pred_mu.device)
    keep = i != j
    if keep.sum() == 0:
        return torch.zeros([], device=pred_mu.device, dtype=pred_mu.dtype)
    i = i[keep]
    j = j[keep]

    diff_t = t[i] - t[j]
    mask = diff_t != 0
    if mask.sum() == 0:
        return torch.zeros([], device=pred_mu.device, dtype=pred_mu.dtype)

    diff_p = p[i] - p[j]
    labels = (diff_t > 0).float()
    return F.binary_cross_entropy_with_logits(diff_p[mask], labels[mask])


def run_epoch(
    model: ChromaGuideModel,
    optimizer: torch.optim.Optimizer,
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    shuffle_seed: int,
    rank_lambda: float,
    feature_cols: list[str],
) -> tuple[float, float, float]:
    model.train()
    cur = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    total_loss = 0.0
    total_beta = 0.0
    total_rank = 0.0

    for i in range(0, len(cur), batch_size):
        b = cur.iloc[i : i + batch_size]
        if len(b) < 2:
            continue
        y = torch.tensor(b["efficiency"].values, dtype=torch.float32, device=device).unsqueeze(1)
        output = model(
            prepare_seq(b["sequence"].tolist(), device),
            epi_tracks=prepare_epi(b, feature_cols, device),
        )

        beta_loss = model.compute_loss(output, y)["total_loss"]
        if rank_lambda > 0:
            rank_loss = pairwise_rank_loss(output["mu"], y)
        else:
            rank_loss = torch.zeros([], device=device, dtype=beta_loss.dtype)
        loss = beta_loss + rank_lambda * rank_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_beta += float(beta_loss.item())
        total_rank += float(rank_loss.item())

    denom = max(1, len(cur) / batch_size)
    return total_loss / denom, total_beta / denom, total_rank / denom


@torch.no_grad()
def predict_df(
    model: ChromaGuideModel,
    df: pd.DataFrame,
    batch_size: int,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(df), batch_size):
        b = df.iloc[i : i + batch_size]
        output = model(
            prepare_seq(b["sequence"].tolist(), device),
            epi_tracks=prepare_epi(b, feature_cols, device),
        )
        preds.extend(output["mu"].cpu().numpy().flatten())
    return np.asarray(preds)


def main() -> None:
    p = argparse.ArgumentParser(description="Ranking-aware on-target train+val")
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--fusion", choices=["gate", "concat"], default="gate")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--holdout_frac", type=float, default=0.05)
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--rank_lambda", type=float, default=0.5)
    p.add_argument("--pretrain", choices=["none", "hfesp"], default="hfesp")
    p.add_argument("--pretrain_epochs", type=int, default=4)
    p.add_argument("--pretrain_max_rows", type=int, default=0, help="If >0, subsample pretrain rows for faster sweeps")
    p.add_argument(
        "--pretrain_data_dir",
        type=str,
        default="",
        help="Path to DeepHF_upstream_2/data (auto-resolved if omitted).",
    )
    p.add_argument("--output_prefix", type=str, default="results/runs/ranked_experiment")
    p.add_argument("--use_cellline_feature", dest="use_cellline_feature", action="store_true")
    p.add_argument("--no_cellline_feature", dest="use_cellline_feature", action="store_false")
    p.set_defaults(use_cellline_feature=True)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(
        f"Using device: {device}\n"
        f"Config: split={args.split}, seed={args.seed}, d_model={args.d_model}, "
        f"fusion={args.fusion}, rank_lambda={args.rank_lambda}, pretrain={args.pretrain}, "
        f"use_cellline_feature={args.use_cellline_feature}",
        flush=True,
    )

    train_df, val_df, test_df, feature_cols = load_split_data(args.split, args.use_cellline_feature)
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    holdout_idx = trainval_df.sample(frac=args.holdout_frac, random_state=args.seed).index
    holdout_df = trainval_df.loc[holdout_idx].reset_index(drop=True)
    finetune_df = trainval_df.drop(holdout_idx).reset_index(drop=True)
    print(f"Sizes: finetune={len(finetune_df)} holdout={len(holdout_df)} test={len(test_df)}", flush=True)

    model = ChromaGuideModel(
        encoder_type="cnn_gru",
        d_model=args.d_model,
        seq_len=21,
        num_epi_tracks=len(feature_cols),
        num_epi_bins=1,
        use_epigenomics=True,
        use_gate_fusion=(args.fusion == "gate"),
        fusion_type=args.fusion,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrain == "hfesp":
        pretrain_data_dir = resolve_pretrain_data_dir(args.pretrain_data_dir)
        print(f"Pretrain data dir: {pretrain_data_dir}", flush=True)
        pretrain_df = load_pretrain_df_hfesp(pretrain_data_dir, args.seed)
        if args.pretrain_max_rows > 0 and args.pretrain_max_rows < len(pretrain_df):
            pretrain_df = pretrain_df.sample(n=args.pretrain_max_rows, random_state=args.seed).reset_index(drop=True)
        for c in feature_cols:
            if c not in pretrain_df.columns:
                pretrain_df[c] = 0.0
        print(f"Pretrain rows: {len(pretrain_df)}", flush=True)
        for epoch in range(1, args.pretrain_epochs + 1):
            total, beta, rank = run_epoch(
                model,
                optimizer,
                pretrain_df,
                args.batch_size,
                device,
                args.seed + epoch,
                rank_lambda=0.0,  # pretrain on base beta objective
                feature_cols=feature_cols,
            )
            print(
                f"Pretrain epoch {epoch} | total={total:.4f} | beta={beta:.4f} | rank={rank:.4f}",
                flush=True,
            )

    best_holdout = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, args.finetune_epochs + 1):
        total, beta, rank = run_epoch(
            model,
            optimizer,
            finetune_df,
            args.batch_size,
            device,
            1000 + args.seed + epoch,
            rank_lambda=args.rank_lambda,
            feature_cols=feature_cols,
        )

        holdout_preds = predict_df(model, holdout_df, args.batch_size, feature_cols, device)
        holdout_rho = float(spearmanr(holdout_df["efficiency"].values, holdout_preds)[0])
        if not np.isfinite(holdout_rho):
            holdout_rho = -1.0

        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total),
                "beta_loss": float(beta),
                "rank_loss": float(rank),
                "holdout_rho": holdout_rho,
            }
        )
        print(
            f"Finetune epoch {epoch} | total={total:.4f} | beta={beta:.4f} | "
            f"rank={rank:.4f} | holdout_rho={holdout_rho:.4f}",
            flush=True,
        )

        if holdout_rho > best_holdout:
            best_holdout = holdout_rho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_preds = predict_df(model, test_df, args.batch_size, feature_cols, device)
    gold_rho = float(spearmanr(test_df["efficiency"].values, test_preds)[0])

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    pred_path = out_prefix.with_name(out_prefix.name + "_predictions.csv")

    result = {
        "split": args.split,
        "seed": args.seed,
        "d_model": args.d_model,
        "fusion": args.fusion,
        "rank_lambda": args.rank_lambda,
        "pretrain": args.pretrain,
        "use_cellline_feature": args.use_cellline_feature,
        "feature_cols": feature_cols,
        "best_holdout_rho": best_holdout,
        "gold_rho": gold_rho,
        "history": history,
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(
        {
            "sequence": test_df["sequence"].values,
            "efficiency": test_df["efficiency"].values,
            "prediction": test_preds,
        }
    ).to_csv(pred_path, index=False)

    print(f"GOLD rho: {gold_rho:.6f}", flush=True)
    print(f"Saved metrics: {json_path}", flush=True)
    print(f"Saved predictions: {pred_path}", flush=True)


if __name__ == "__main__":
    main()
