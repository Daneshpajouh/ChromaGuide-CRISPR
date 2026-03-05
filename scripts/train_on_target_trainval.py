#!/usr/bin/env python3
"""
Train ChromaGuide on train+validation with a small internal holdout for early stopping.

Supports optional transfer pretraining on DeepHF HF+ESP datasets, then fine-tuning
and evaluation on a target split (A/B/C). Saves metrics JSON + test predictions CSV.
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

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
        [1.0, 0.0, 0.0, 0.0],  # A
        [0.0, 1.0, 0.0, 0.0],  # C
        [0.0, 0.0, 1.0, 0.0],  # G
        [0.0, 0.0, 0.0, 1.0],  # T
        [0.25, 0.25, 0.25, 0.25],  # N/ambiguous
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


def split_dir_for(split: str, split_dir: Optional[str] = None) -> Path:
    if split_dir:
        return Path(split_dir)
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

    # Fallback path for clearer error reporting downstream.
    return Path("/scratch/amird/DeepHF_upstream_2/data")


def _read_split_partition(sdir: Path, partition: str) -> pd.DataFrame:
    dfs = []
    for f in sorted(sdir.glob(f"*_{partition}.csv")):
        # Ignore AppleDouble metadata files accidentally introduced by cross-platform copies.
        if f.name.startswith("._"):
            continue
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


def load_split_data(
    split: str,
    use_cellline_feature: bool,
    split_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    sdir = split_dir_for(split, split_dir=split_dir)
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
    arr = ONEHOT_TABLE[idx]  # (B, L, 4)
    arr = np.transpose(arr, (0, 2, 1))  # (B, 4, L)
    return torch.from_numpy(arr).to(device=device)


def prepare_epi(batch: pd.DataFrame, feature_cols: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(batch[feature_cols].values.astype(np.float32), device=device).unsqueeze(2)


def run_epoch(
    model: ChromaGuideModel,
    optimizer: torch.optim.Optimizer,
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    shuffle_seed: int,
    feature_cols: list[str],
    loss_type: str = "beta",
) -> float:
    model.train()
    cur = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    total = 0.0
    for i in range(0, len(cur), batch_size):
        b = cur.iloc[i : i + batch_size]
        if len(b) < 2:
            continue
        y = torch.tensor(b["efficiency"].values, dtype=torch.float32, device=device).unsqueeze(1)
        output = model(
            prepare_seq(b["sequence"].tolist(), device),
            epi_tracks=prepare_epi(b, feature_cols, device),
        )
        if loss_type == "mse":
            loss = F.mse_loss(output["mu"], y)
        else:
            loss = model.compute_loss(output, y)["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(cur) / batch_size)


def build_mixed_finetune_df(
    finetune_df: pd.DataFrame,
    pretrain_df: Optional[pd.DataFrame],
    mix_frac: float,
    mix_decay: float,
    epoch: int,
    seed: int,
    max_pretrain_rows: int,
) -> pd.DataFrame:
    if pretrain_df is None or mix_frac <= 0:
        return finetune_df

    frac_now = mix_frac * (mix_decay ** max(0, epoch - 1))
    if frac_now <= 0:
        return finetune_df

    n_pre = int(round(len(finetune_df) * frac_now))
    if max_pretrain_rows > 0:
        n_pre = min(n_pre, max_pretrain_rows)
    if n_pre <= 0:
        return finetune_df

    replace = n_pre > len(pretrain_df)
    aux = pretrain_df.sample(n=n_pre, replace=replace, random_state=seed + epoch).reset_index(drop=True)
    merged = pd.concat([finetune_df, aux], ignore_index=True)
    return merged.sample(frac=1.0, random_state=seed + 10000 + epoch).reset_index(drop=True)


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
    p = argparse.ArgumentParser(description="Train on-target train+val + optional transfer pretraining")
    p.add_argument("--split", choices=["A", "B", "C"], default="A")
    p.add_argument("--split_dir", type=str, default="", help="Optional explicit split directory path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--encoder_type", choices=["cnn_gru", "mamba", "dnabert2"], default="cnn_gru")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--fusion", choices=["gate", "concat", "cross_attention"], default="gate")
    p.add_argument("--loss_type", choices=["beta", "mse"], default="beta")
    p.add_argument("--use_mi_regularizer", action="store_true")
    p.add_argument("--mi_lambda", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--pretrain_lr", type=float, default=0.0, help="If >0, LR used only during pretraining")
    p.add_argument("--finetune_lr", type=float, default=0.0, help="If >0, LR used during finetuning")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--holdout_frac", type=float, default=0.05)
    p.add_argument(
        "--use_explicit_validation_holdout",
        action="store_true",
        help="Use provided validation partition as holdout (recommended for strict leakage-controlled regimes)",
    )
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--pretrain", choices=["none", "hfesp"], default="none")
    p.add_argument("--pretrain_epochs", type=int, default=6)
    p.add_argument("--pretrain_max_rows", type=int, default=0, help="If >0, subsample pretrain rows for faster sweeps")
    p.add_argument(
        "--finetune_pretrain_mix",
        type=float,
        default=0.0,
        help="Extra pretrain rows per finetune row (e.g., 0.5 = +50 percent)",
    )
    p.add_argument("--finetune_pretrain_mix_decay", type=float, default=1.0, help="Per-epoch decay for finetune_pretrain_mix")
    p.add_argument("--finetune_pretrain_max_rows", type=int, default=0, help="Cap sampled pretrain rows mixed per finetune epoch")
    p.add_argument("--dnabert_freeze", dest="dnabert_freeze", action="store_true", help="Freeze DNABERT2 backbone when available")
    p.add_argument("--dnabert_unfreeze", dest="dnabert_freeze", action="store_false", help="Fine-tune DNABERT2 backbone when available")
    p.add_argument("--dnabert_backbone_lr_factor", type=float, default=0.1, help="Backbone LR multiplier when DNABERT2 backbone is unfrozen")
    p.add_argument(
        "--pretrain_data_dir",
        type=str,
        default="",
        help="Path to DeepHF_upstream_2/data (auto-resolved if omitted).",
    )
    p.add_argument("--output_prefix", type=str, default="results/runs/trainval_experiment")
    p.add_argument("--use_cellline_feature", dest="use_cellline_feature", action="store_true")
    p.add_argument("--no_cellline_feature", dest="use_cellline_feature", action="store_false")
    p.set_defaults(use_cellline_feature=True)
    p.set_defaults(dnabert_freeze=True)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(
        f"Config: split={args.split}, seed={args.seed}, pretrain={args.pretrain}, "
        f"encoder={args.encoder_type}, d_model={args.d_model}, fusion={args.fusion}, "
        f"loss={args.loss_type}, mi_reg={args.use_mi_regularizer}, "
        f"use_cellline_feature={args.use_cellline_feature}, "
        f"dnabert_freeze={args.dnabert_freeze}",
        flush=True,
    )

    train_df, val_df, test_df, feature_cols = load_split_data(
        args.split,
        args.use_cellline_feature,
        split_dir=(args.split_dir or None),
    )
    if args.use_explicit_validation_holdout:
        finetune_df = train_df.reset_index(drop=True)
        holdout_df = val_df.reset_index(drop=True)
    else:
        trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        holdout_idx = trainval_df.sample(frac=args.holdout_frac, random_state=args.seed).index
        holdout_df = trainval_df.loc[holdout_idx].reset_index(drop=True)
        finetune_df = trainval_df.drop(holdout_idx).reset_index(drop=True)
    print(
        f"Sizes: finetune={len(finetune_df)} holdout={len(holdout_df)} test={len(test_df)}",
        flush=True,
    )

    model = ChromaGuideModel(
        encoder_type=args.encoder_type,
        d_model=args.d_model,
        seq_len=21,
        num_epi_tracks=len(feature_cols),
        num_epi_bins=1,
        use_epigenomics=True,
        use_gate_fusion=(args.fusion == "gate"),
        fusion_type=args.fusion,
        use_mi_regularizer=args.use_mi_regularizer,
        mi_lambda=args.mi_lambda,
        dropout=args.dropout,
        dnabert_freeze=args.dnabert_freeze,
    ).to(device)
    pretrain_lr = args.pretrain_lr if args.pretrain_lr > 0 else args.lr
    finetune_lr = args.finetune_lr if args.finetune_lr > 0 else args.lr
    if (
        args.encoder_type == "dnabert2"
        and hasattr(model.seq_encoder, "use_bert")
        and getattr(model.seq_encoder, "use_bert")
        and not args.dnabert_freeze
        and hasattr(model.seq_encoder, "backbone")
    ):
        backbone_params = list(model.seq_encoder.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": head_params, "lr": pretrain_lr},
                {"params": backbone_params, "lr": pretrain_lr * args.dnabert_backbone_lr_factor},
            ],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)

    pretrain_df = None
    pretrain_data_dir = resolve_pretrain_data_dir(args.pretrain_data_dir)
    if args.pretrain == "hfesp":
        print(f"Pretrain data dir: {pretrain_data_dir}", flush=True)
    if args.pretrain == "hfesp":
        pretrain_df = load_pretrain_df_hfesp(pretrain_data_dir, args.seed)
        for c in feature_cols:
            if c not in pretrain_df.columns:
                pretrain_df[c] = 0.0
        if args.pretrain_max_rows > 0 and args.pretrain_max_rows < len(pretrain_df):
            pretrain_df = pretrain_df.sample(n=args.pretrain_max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Pretrain rows: {len(pretrain_df)}", flush=True)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = run_epoch(
                model,
                optimizer,
                pretrain_df,
                args.batch_size,
                device,
                args.seed + epoch,
                feature_cols,
                loss_type=args.loss_type,
            )
            print(f"Pretrain epoch {epoch} | loss={loss:.4f}", flush=True)

    for g in optimizer.param_groups:
        g["lr"] = finetune_lr

    best_holdout = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, args.finetune_epochs + 1):
        epoch_train_df = build_mixed_finetune_df(
            finetune_df=finetune_df,
            pretrain_df=pretrain_df,
            mix_frac=args.finetune_pretrain_mix,
            mix_decay=args.finetune_pretrain_mix_decay,
            epoch=epoch,
            seed=args.seed,
            max_pretrain_rows=args.finetune_pretrain_max_rows,
        )
        loss = run_epoch(
            model,
            optimizer,
            epoch_train_df,
            args.batch_size,
            device,
            1000 + args.seed + epoch,
            feature_cols,
            loss_type=args.loss_type,
        )
        holdout_preds = predict_df(model, holdout_df, args.batch_size, feature_cols, device)
        holdout_rho = float(spearmanr(holdout_df["efficiency"].values, holdout_preds)[0])
        if not np.isfinite(holdout_rho):
            holdout_rho = -1.0
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "holdout_rho": holdout_rho,
                "epoch_train_rows": int(len(epoch_train_df)),
            }
        )
        print(
            f"Finetune epoch {epoch} | rows={len(epoch_train_df)} | loss={loss:.4f} | holdout_rho={holdout_rho:.4f}",
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
        "split_dir": args.split_dir,
        "use_explicit_validation_holdout": args.use_explicit_validation_holdout,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "holdout_frac": args.holdout_frac,
        "finetune_epochs": args.finetune_epochs,
        "patience": args.patience,
        "pretrain": args.pretrain,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_max_rows": args.pretrain_max_rows,
        "pretrain_data_dir_resolved": str(pretrain_data_dir),
        "pretrain_lr": pretrain_lr,
        "finetune_lr": finetune_lr,
        "finetune_pretrain_mix": args.finetune_pretrain_mix,
        "finetune_pretrain_mix_decay": args.finetune_pretrain_mix_decay,
        "finetune_pretrain_max_rows": args.finetune_pretrain_max_rows,
        "loss_type": args.loss_type,
        "use_mi_regularizer": args.use_mi_regularizer,
        "mi_lambda": args.mi_lambda,
        "encoder_type": args.encoder_type,
        "d_model": args.d_model,
        "fusion": args.fusion,
        "use_cellline_feature": args.use_cellline_feature,
        "feature_cols": feature_cols,
        "best_holdout_rho": best_holdout,
        "gold_rho": gold_rho,
        "finetune_rows": len(finetune_df),
        "holdout_rows": len(holdout_df),
        "test_rows": len(test_df),
        "argv": sys.argv,
        "cli_args": vars(args),
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
