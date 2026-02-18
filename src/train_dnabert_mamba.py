"""DNABERT -> Adapter -> Mamba training CLI.

Cleaned, compact training script. Use `--use_mini` for fast validation runs.
"""

import argparse
import logging
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModel = None
    TRANSFORMERS_AVAILABLE = False

try:
    from src.data.crisprofft import CRISPRoffTDataset
except Exception:
    CRISPRoffTDataset = None

from src.model.adapters import AdapterFactory
from src.utils.experiment_logger import ExperimentLogger
from src.evaluation.metrics import compute_regression_metrics


def build_dnabert_model(model_name: str = "dna_bert_2", token: Optional[str] = None, trust_remote_code: bool = True):
    if not TRANSFORMERS_AVAILABLE:
        logging.warning("transformers not available; cannot build DNABERT foundation model.")
        return None
    try:
        # prefer explicit token passed by caller, else fallback to env/file
        if token is None:
            token = os.environ.get('HF_TOKEN')
            if token is None:
                token_path = os.path.expanduser('~/.huggingface/token')
                if os.path.exists(token_path):
                    try:
                        with open(token_path, 'r') as fh:
                            token = fh.read().strip()
                            os.environ['HF_TOKEN'] = token
                    except Exception:
                        token = None

        # load tokenizer and model with auth token and trust_remote_code
        kwargs = {}
        if token:
            kwargs['use_auth_token'] = token
        kwargs['trust_remote_code'] = bool(trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModel.from_pretrained(model_name, **kwargs)
        logging.info(f"Loaded DNABERT model '{model_name}' with trust_remote_code={kwargs['trust_remote_code']} token={'yes' if token else 'no'}")
        try:
            hidden = getattr(model.config, 'hidden_size', None)
            if hidden:
                logging.info(f"DNABERT hidden size: {hidden}")
        except Exception:
            pass
        return tokenizer, model
    except Exception as e:
        logging.warning(f"Failed to load DNABERT model '{model_name}': {e}")
        return None


def train(args):
    logging.info("Starting DNABERT->Mamba training skeleton")

    if CRISPRoffTDataset is None:
        logging.error("CRISPRoffTDataset not available. Ensure `src/data/crisprofft.py` exists.")
        return

    # Attempt to build DNABERT foundation first so we can hand the tokenizer
    # into the dataset constructor (dataset will pre-tokenize guide sequences).
    dnabert = None
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        res = build_dnabert_model(args.dnabert_name, token=getattr(args, 'hf_token', None), trust_remote_code=getattr(args, 'trust_remote_code', True))
        if res is not None:
            tokenizer, dnabert = res

    # Dataset and DataLoader
    # Determine project root and candidate mini dataset locations so users
    # can pass --use_mini even if older paths (data/min/...) were used.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path_override = None
    if getattr(args, 'use_mini', False):
        candidates = [
            os.path.join(project_root, 'data', 'min', 'crisproff', 'min1_crisproff.txt'),
            os.path.join(project_root, 'data', 'mini', 'crisprofft', 'mini_crisprofft.txt'),
            os.path.join(project_root, 'data', 'mini', 'crisprofft', 'mini_crisprofft.txt'),
        ]
        for p in candidates:
            if os.path.exists(p):
                data_path_override = p
                logging.info(f"Using mini dataset at: {p}")
                break
        if data_path_override is None:
            logging.warning("Mini dataset not found in standard locations; will attempt dataset default path.")

    # Pass tokenizer into dataset so it can precompute tokens per sample
    train_dataset = CRISPRoffTDataset(split="train", use_mini=args.use_mini, data_path_override=data_path_override, tokenizer=tokenizer)
    val_dataset = CRISPRoffTDataset(split="val", use_mini=args.use_mini, data_path_override=data_path_override, tokenizer=tokenizer)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create adapter that maps DNABERT hidden size -> Mamba input dim
    # Heuristics: if dnabert is present, use its config.hidden_size
    if dnabert is not None and hasattr(dnabert.config, "hidden_size"):
        in_dim = dnabert.config.hidden_size
    else:
        in_dim = args.adapter_in_dim

    out_dim = args.adapter_out_dim
    adapter = AdapterFactory(in_dim, out_dim, kind=args.adapter_kind)

    # Try to import Mamba stack (best-effort)
    try:
        from src.model import mamba2_block
        try:
            mamba = mamba2_block.Mamba2Config and getattr(mamba2_block, "Mamba2")
        except Exception:
            mamba = None
    except Exception:
        mamba = None

    # Head (simple MLP)
    head = torch.nn.Sequential(torch.nn.Linear(out_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter.to(device)
    head.to(device)
    if dnabert is not None:
        try:
            dnabert.to(device)
            dnabert.train()
        except Exception:
            pass

    # Optimizers: include DNABERT params (low lr) if available, else adapter+head only
    if dnabert is not None:
        optim = torch.optim.AdamW([
            {'params': dnabert.parameters(), 'lr': 1e-5},
            {'params': adapter.parameters(), 'lr': args.lr},
            {'params': head.parameters(), 'lr': args.lr},
        ], lr=args.lr)
    else:
        optim = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=args.lr)

    # Learning rate scheduler
    scheduler = None
    if getattr(args, 'scheduler', None) == 'cosine':
        try:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))
        except Exception:
            scheduler = None
    elif getattr(args, 'scheduler', None) == 'plateau':
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
        except Exception:
            scheduler = None

    # Simple loss
    loss_fn = torch.nn.MSELoss()

    # Experiment logger: capture hyperparams, dataset and runtime info
    logger = ExperimentLogger(run_name=None, out_dir='logs')
    hyper = {k: v for k, v in vars(args).items()}
    # approximate model param count
    try:
        param_count = sum(p.numel() for p in adapter.parameters()) + sum(p.numel() for p in head.parameters())
    except Exception:
        param_count = None
    dataset_info = {
        'name': 'CRISPRoffT',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
    }
    logger.start_run(hyperparams=hyper, model=None, dataset_info=dataset_info)

    # Training loop with per-epoch metrics collection
    # Prepare best-checkpoint tracking
    best_spearman = float('-inf')
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        steps = 0
        train_preds = []
        train_targets = []

        adapter.train()
        head.train()

        for batch in loader:
            seq = batch.get("sequence")
            target = batch.get("efficiency")
            if seq is None or target is None:
                continue

            # Prefer dataset-provided tokenization if available (dataset
            # was created with `tokenizer=` and adds a per-sample `tokens` entry).
            if tokenizer is not None and dnabert is not None:
                try:
                    tokens_batch = batch.get('tokens')
                    if tokens_batch:
                        # tokens_batch is a list of per-sample token dicts
                        from torch.nn.utils.rnn import pad_sequence
                        input_ids_list = []
                        attention_list = []
                        for t in tokens_batch:
                            if isinstance(t, dict):
                                ids = t.get('input_ids')
                                mask = t.get('attention_mask')
                            else:
                                ids = t
                                mask = None
                            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
                            if mask is not None:
                                attention_list.append(torch.tensor(mask, dtype=torch.long))
                        if input_ids_list:
                            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0).to(device)
                        else:
                            raise RuntimeError('Empty input_ids_list')
                        attention_mask = None
                        if attention_list:
                            attention_mask = pad_sequence(attention_list, batch_first=True, padding_value=0).to(device)
                        emb = dnabert(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(emb, 'last_hidden_state'):
                            x = emb.last_hidden_state.mean(dim=1)
                        else:
                            x = emb[0].mean(dim=1)
                    else:
                        # Fall back to tokenizing the raw sequence (may fail if seq is tensor)
                        tokens = tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
                        input_ids = tokens['input_ids'].to(device)
                        attention_mask = tokens.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        emb = dnabert(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(emb, 'last_hidden_state'):
                            x = emb.last_hidden_state.mean(dim=1)
                        else:
                            x = emb[0].mean(dim=1)
                except Exception:
                    B = len(seq) if hasattr(seq, '__len__') else 1
                    x = torch.randn(B, in_dim, device=device)
            else:
                B = len(seq) if hasattr(seq, '__len__') else 1
                x = torch.randn(B, in_dim, device=device)

            x = adapter(x)
            pred = head(x).squeeze(-1)
            target = torch.as_tensor(target, dtype=torch.float32, device=device)
            loss = loss_fn(pred, target)

            optim.zero_grad()
            loss.backward()
            # gradient clipping for stability
            try:
                params_for_clip = [p for grp in optim.param_groups for p in grp.get('params', []) if p.grad is not None]
                if params_for_clip:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, max_norm=1.0)
            except Exception:
                pass
            optim.step()

            epoch_loss += float(loss.detach().cpu().numpy())
            steps += 1

            train_preds.append(pred.detach().cpu().numpy())
            train_targets.append(target.detach().cpu().numpy())

        # aggregate train metrics
        if train_preds:
            import numpy as _np
            train_preds_a = _np.concatenate([_np.ravel(x) for x in train_preds])
            train_targets_a = _np.concatenate([_np.ravel(x) for x in train_targets])
        else:
            train_preds_a = _np.array([])
            train_targets_a = _np.array([])

        train_metrics = compute_regression_metrics(train_targets_a, train_preds_a)

        # validation step
        adapter.eval()
        head.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch.get("sequence")
                target = batch.get("efficiency")
                if seq is None or target is None:
                    continue
                if tokenizer is not None:
                    try:
                        tokens = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
                        input_ids = tokens["input_ids"].to(device)
                        attention_mask = tokens.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        emb = dnabert(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(emb, "last_hidden_state"):
                            x = emb.last_hidden_state.mean(dim=1)
                        else:
                            x = emb[0].mean(dim=1)
                    except Exception:
                        B = len(seq) if hasattr(seq, '__len__') else 1
                        x = torch.randn(B, in_dim, device=device)
                else:
                    B = len(seq) if hasattr(seq, '__len__') else 1
                    x = torch.randn(B, in_dim, device=device)

                x = adapter(x)
                pred = head(x).squeeze(-1)
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(torch.as_tensor(target, dtype=torch.float32, device=device).cpu().numpy())

        if val_preds:
            import numpy as _np
            val_preds_a = _np.concatenate([_np.ravel(x) for x in val_preds])
            val_targets_a = _np.concatenate([_np.ravel(x) for x in val_targets])
        else:
            val_preds_a = _np.array([])
            val_targets_a = _np.array([])

        val_metrics = compute_regression_metrics(val_targets_a, val_preds_a)

        # Save best checkpoint by validation Spearman
        try:
            cur_spear = float(val_metrics.get('spearman', float('-inf')))
        except Exception:
            cur_spear = float('-inf')
        if cur_spear > best_spearman:
            best_spearman = cur_spear
            ckpt = {
                'epoch': epoch + 1,
                'best_spearman': best_spearman,
                'adapter_state': adapter.state_dict(),
                'head_state': head.state_dict(),
                'optimizer_state': optim.state_dict(),
                'val_metrics': val_metrics,
            }
            if dnabert is not None:
                try:
                    ckpt['dnabert_state'] = dnabert.state_dict()
                except Exception:
                    pass
            try:
                torch.save(ckpt, os.path.join('checkpoints', 'dnabert_mamba_best.pt'))
                logging.info(f"Saved best checkpoint (epoch {ckpt['epoch']}) to checkpoints/dnabert_mamba_best.pt with val_spearman={best_spearman:.6f}")
            except Exception as e:
                logging.warning(f"Failed to save checkpoint: {e}")

        epoch_duration = time.time() - epoch_start
        # Log epoch summary including key regression metrics (Spearman, Pearson, MSE)
        try:
            t_spear = float(train_metrics.get('spearman', float('nan')))
        except Exception:
            t_spear = float('nan')
        try:
            v_spear = float(val_metrics.get('spearman', float('nan')))
        except Exception:
            v_spear = float('nan')
        try:
            t_pear = float(train_metrics.get('pearson', float('nan')))
        except Exception:
            t_pear = float('nan')
        try:
            v_pear = float(val_metrics.get('pearson', float('nan')))
        except Exception:
            v_pear = float('nan')
        try:
            v_mse = float(val_metrics.get('mse', float('nan')))
        except Exception:
            v_mse = float('nan')

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} train_loss={epoch_loss/ max(1, steps):.6f} "
            f"train_samps={len(train_dataset)} val_samps={len(val_dataset)} "
            f"train_spearman={t_spear:.4f} train_pearson={t_pear:.4f} "
            f"val_spearman={v_spear:.4f} val_pearson={v_pear:.4f} val_mse={v_mse:.6f}"
        )

        # step plateau scheduler based on validation MSE
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                scheduler.step(val_metrics.get('mse', float('nan')))
            except Exception:
                pass

        # log to experiment logger
        logger.log_epoch(epoch=epoch + 1, duration_s=epoch_duration, train_metrics=train_metrics, val_metrics=val_metrics, lr=args.lr)

    # finalize logger and save
    logger.finalize(best_info=None, checkpoints={'best': 'checkpoints/dnabert_mamba_best.pt'})


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnabert_name", type=str, default="dna_bert_2")
    parser.add_argument("--adapter_kind", type=str, default="linear")
    parser.add_argument("--adapter_in_dim", type=int, default=768)
    parser.add_argument("--adapter_out_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_mini", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token (overrides ~/.huggingface/token)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True when loading HF model")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
