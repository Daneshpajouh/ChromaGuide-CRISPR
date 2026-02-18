"""DNABERT + Mamba integration class.

Provides a tight DNABERT->Adapter->Mamba class with training-stage helpers and
Spearman logging. This is a conservative implementation that attempts to import
optional dependencies (transformers, scipy) but degrades gracefully when they
are not available.
"""
from typing import Optional, Dict, Any, List
import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoModel = None
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    spearmanr = None
    SCIPY_AVAILABLE = False

from src.model.adapters import AdapterFactory

logger = logging.getLogger(__name__)


class DNABERTMamba(nn.Module):
    """DNABERT -> Adapter -> Mamba multi-head model.

    Outputs three heads:
      - efficiency (regression, 0-1)
      - off_target_prob (binary probability)
      - indel_prob (regression, 0-1)

    The class provides helpers for freezing/unfreezing and a compact
    training loop is provided by `train()` below.
    """

    def __init__(self,
                 dnabert_name: Optional[str] = None,
                 tokenizer: Optional[Any] = None,
                 adapter_kind: str = 'linear',
                 adapter_out_dim: int = 256,
                 mamba_cfg: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Foundation model (optional)
        self.dnabert = None
        self.tokenizer = tokenizer
        if dnabert_name and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(dnabert_name)
                self.dnabert = AutoModel.from_pretrained(dnabert_name)
                logger.info(f"Loaded DNABERT foundation '{dnabert_name}'")
            except Exception as e:
                logger.warning(f"Failed to load DNABERT '{dnabert_name}': {e}")

        # Determine adapter input dim
        if self.dnabert is not None and hasattr(self.dnabert.config, 'hidden_size'):
            in_dim = self.dnabert.config.hidden_size
        else:
            in_dim = adapter_out_dim

        # Adapter projects foundation hidden -> adapter_out_dim
        self.adapter = AdapterFactory(in_dim, adapter_out_dim, kind=adapter_kind)

        # Mamba stack: try to use existing block, fallback to Transformer encoder
        try:
            from src.model import mamba2_block
            if hasattr(mamba2_block, 'Mamba2'):
                cfg = mamba_cfg or {}
                # If Mamba2 expects a config object, tests will need adaptation
                try:
                    self.mamba = mamba2_block.Mamba2(d_model=adapter_out_dim, **cfg)
                except Exception:
                    # Try default constructor
                    self.mamba = mamba2_block.Mamba2()
            else:
                self.mamba = None
        except Exception:
            self.mamba = None

        if self.mamba is None:
            encoder_layer = nn.TransformerEncoderLayer(d_model=adapter_out_dim, nhead=4)
            self.mamba = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Multi-head outputs
        self.head_eff = nn.Sequential(nn.Linear(adapter_out_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.head_off = nn.Sequential(nn.Linear(adapter_out_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_indel = nn.Sequential(nn.Linear(adapter_out_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        # Apply to device
        self.to(self.device)

    def forward(self, inputs: Dict[str, Any]):
        """Forward accepts a dict with one of:
           - 'tokens': dict from tokenizer (input_ids, attention_mask)
           - 'embeddings': precomputed tensor
           - 'sequence': integer-encoded torch tensor
        Returns a dict with keys: 'efficiency', 'off_target', 'indel'
        """
        # 1) Obtain embedding vector x (B, D)
        if 'embeddings' in inputs and inputs['embeddings'] is not None:
            x = inputs['embeddings'].to(self.device)
        elif 'tokens' in inputs and self.dnabert is not None:
            t = inputs['tokens']
            # Handle both dict-like tokenizer outputs and pre-padded lists
            if isinstance(t, dict):
                # Convert lists -> tensors where necessary
                input_ids = torch.tensor(t.get('input_ids')).to(self.device)
                attention_mask = None
                if 'attention_mask' in t and t['attention_mask'] is not None:
                    attention_mask = torch.tensor(t.get('attention_mask')).to(self.device)
            else:
                input_ids = torch.tensor(t).to(self.device)
                attention_mask = None

            # Run foundation model in eval/no-grad unless fine-tuning
            with torch.no_grad():
                if attention_mask is not None:
                    emb = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    emb = self.dnabert(input_ids=input_ids)
                # prefer last_hidden_state
                if hasattr(emb, 'last_hidden_state'):
                    x = emb.last_hidden_state.mean(dim=1)
                else:
                    x = emb[0].mean(dim=1)
        elif 'sequence' in inputs:
            seq = inputs['sequence']
            if isinstance(seq, list):
                seq = torch.tensor(seq, dtype=torch.long)
            x = seq.to(self.device).float()
        else:
            raise ValueError('No valid input provided to DNABERTMamba.forward')

        # 2) Adapter
        x = self.adapter(x)

        # 3) Mamba expects sequence-like input; if x is (B, D) expand a sequence dim
        if x.dim() == 2:
            x_in = x.unsqueeze(1)  # (B, S=1, D)
        else:
            x_in = x

        m_out = self.mamba(x_in)
        if m_out.dim() == 3:
            pooled = m_out.mean(dim=1)
        else:
            pooled = m_out

        # 4) Heads
        eff = self.head_eff(pooled).squeeze(-1)
        off = torch.sigmoid(self.head_off(pooled).squeeze(-1))
        ind = torch.sigmoid(self.head_indel(pooled).squeeze(-1))

        return {"efficiency": eff, "off_target": off, "indel": ind}

    def freeze_foundation(self):
        if self.dnabert is not None:
            for p in self.dnabert.parameters():
                p.requires_grad = False

    def unfreeze_foundation(self):
        if self.dnabert is not None:
            for p in self.dnabert.parameters():
                p.requires_grad = True

    def compute_spearman(self, y_true: List[float], y_pred: List[float]) -> float:
        try:
            if SCIPY_AVAILABLE:
                rho, _ = spearmanr(y_true, y_pred)
                return float(rho)
            else:
                # Fallback: compute ranks and Pearson manually
                import numpy as _np
                yt = _np.array(y_true)
                yp = _np.array(y_pred)
                # handle constant arrays
                if yt.std() == 0 or yp.std() == 0:
                    return 0.0
                # convert to ranks
                def rank(a):
                    return _np.argsort(_np.argsort(a)).astype(float)
                r1 = rank(yt)
                r2 = rank(yp)
                rho = ((r1 - r1.mean()) * (r2 - r2.mean())).mean() / (r1.std() * r2.std())
                return float(rho)
        except Exception:
            return 0.0


def collate_batch(batch: List[Dict[str, Any]], tokenizer=None):
    """Collate a list of dataset samples into a batch suitable for the model.

    Each sample is expected to be a dict with keys: 'sequence', 'epigenetics', 'efficiency', optionally 'tokens'.
    If `tokenizer` is provided and samples contain 'tokens' as dicts, use tokenizer to pad.
    """
    batch_out = {}
    # Efficiency / indel / off-target may not all be present in raw dataset; handle gracefully
    effs = []
    offs = []
    inds = []
    tokens_list = []
    sequences = []
    for s in batch:
        effs.append(float(s.get('efficiency', 0.0)))
        # Off-target and indel may be absent; default to 0
        offs.append(float(s.get('off_target', 0.0)))
        inds.append(float(s.get('indel', 0.0)))
        tokens_list.append(s.get('tokens', None))
        sequences.append(s.get('sequence', None))

    batch_out['efficiency'] = torch.tensor(effs, dtype=torch.float32)
    batch_out['off_target'] = torch.tensor(offs, dtype=torch.float32)
    batch_out['indel'] = torch.tensor(inds, dtype=torch.float32)

    # Prefer tokenizer-based padding
    if tokenizer is not None and any(isinstance(t, dict) for t in tokens_list if t is not None):
        # Build a list of dicts for tokenizer.pad
        tdicts = [t for t in tokens_list if t is not None]
        try:
            # tokenizer.pad returns dict of tensors if return_tensors provided; we keep lists
            padded = tokenizer.pad(tdicts, return_tensors='pt')
            batch_out['tokens'] = {k: v for k, v in padded.items()}
        except Exception:
            # Fallback: convert input_ids to padded tensor
            input_ids = [torch.tensor(t.get('input_ids')) if isinstance(t, dict) and t.get('input_ids') is not None else torch.tensor([]) for t in tokens_list]
            try:
                from torch.nn.utils.rnn import pad_sequence
                input_ids = [x for x in input_ids if x.numel() > 0]
                if len(input_ids) > 0:
                    batch_out['tokens'] = {'input_ids': pad_sequence(input_ids, batch_first=True, padding_value=0)}
            except Exception:
                batch_out['tokens'] = None
    else:
        # Fallback: pad integer sequences if present
        seq_tensors = [torch.tensor(s, dtype=torch.long) if s is not None else torch.tensor([]) for s in sequences]
        seq_tensors = [x for x in seq_tensors if x.numel() > 0]
        if len(seq_tensors) > 0:
            from torch.nn.utils.rnn import pad_sequence
            batch_out['sequence'] = pad_sequence(seq_tensors, batch_first=True, padding_value=5)
        else:
            batch_out['sequence'] = None

    return batch_out


def train_loop(model: DNABERTMamba,
               train_loader: DataLoader,
               val_loader: Optional[DataLoader] = None,
               epochs: int = 3,
               lr: float = 1e-4,
               adapter_stage_epochs: int = 1,
               device: Optional[torch.device] = None,
               checkpoint_path: Optional[str] = None,
               tokenizer=None,
               early_stopping_rounds: int = 3):
    device = device or model.device
    model.to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # loss components
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    best_spearman = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs} - training")
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            b = collate_batch(batch, tokenizer=tokenizer)
            inputs = {}
            if b.get('tokens') is not None:
                inputs['tokens'] = {k: v.to(device) for k, v in b['tokens'].items()}
            elif b.get('sequence') is not None:
                inputs['sequence'] = b['sequence'].to(device)

            targets_eff = b['efficiency'].to(device)
            targets_off = b['off_target'].to(device)
            targets_ind = b['indel'].to(device)

            preds = model(inputs)

            loss_eff = mse(preds['efficiency'], targets_eff)
            loss_off = bce(preds['off_target'], targets_off)
            loss_ind = mse(preds['indel'], targets_ind)

            # Weighted sum (configurable later)
            loss = loss_eff + 0.5 * loss_off + 0.5 * loss_ind

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.detach().cpu().numpy())
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else float('nan')
        logger.info(f"Epoch {epoch} training loss: {avg_loss:.6f}")

        # Validation
        if val_loader is not None:
            logger.info("Running validation...")
            model.eval()
            all_true = []
            all_pred = []
            with torch.no_grad():
                for vb in val_loader:
                    bv = collate_batch(vb, tokenizer=tokenizer)
                    vin = {}
                    if bv.get('tokens') is not None:
                        vin['tokens'] = {k: v.to(device) for k, v in bv['tokens'].items()}
                    elif bv.get('sequence') is not None:
                        vin['sequence'] = bv['sequence'].to(device)

                    vpred = model(vin)
                    all_true.extend(bv['efficiency'].tolist())
                    all_pred.extend(vpred['efficiency'].detach().cpu().tolist())

            rho = model.compute_spearman(all_true, all_pred)
            logger.info(f"Validation Spearman rho: {rho:.4f}")

            # Early stopping + checkpoint
            if rho > best_spearman:
                best_spearman = rho
                no_improve = 0
                if checkpoint_path:
                    try:
                        torch.save({'model_state': model.state_dict(), 'optimizer_state': optim.state_dict(), 'epoch': epoch}, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
            else:
                no_improve += 1
                if no_improve >= early_stopping_rounds:
                    logger.info("Early stopping triggered")
                    break

    return model, best_spearman


def cli():
    import argparse
    parser = argparse.ArgumentParser(description="Train DNABERT->Mamba multi-head model")
    parser.add_argument('--dnabert_name', type=str, default='dna_bert_2')
    parser.add_argument('--adapter_kind', type=str, default='linear')
    parser.add_argument('--adapter_out_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_mini', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='dnabert_mamba.pt')
    parser.add_argument('--early_stop', type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build tokenizer and dataset
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.dnabert_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer '{args.dnabert_name}': {e}")

    # Lazy import dataset to avoid heavy imports at module load
    try:
        from src.data.crisprofft import CRISPRoffTDataset
    except Exception as e:
        logger.error(f"Failed to import dataset: {e}")
        return

    train_ds = CRISPRoffTDataset(split='train', use_mini=args.use_mini, tokenizer=tokenizer)
    val_ds = CRISPRoffTDataset(split='val', use_mini=args.use_mini, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    model = DNABERTMamba(dnabert_name=args.dnabert_name, tokenizer=tokenizer, adapter_kind=args.adapter_kind, adapter_out_dim=args.adapter_out_dim)

    # Stage 1: adapter-only training (freeze foundation)
    model.freeze_foundation()
    model, best = train_loop(model, train_loader, val_loader, epochs=1, lr=args.lr, checkpoint_path=args.checkpoint, tokenizer=tokenizer, early_stopping_rounds=args.early_stop)

    # Stage 2: fine-tune entire stack
    model.unfreeze_foundation()
    model, best = train_loop(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr * 0.2, checkpoint_path=args.checkpoint, tokenizer=tokenizer, early_stopping_rounds=args.early_stop)


if __name__ == '__main__':
    cli()

