"""Training utilities: Trainer class with early stopping and checkpointing.

This is a lightweight trainer designed to work with models and datasets in
this repo. It expects PyTorch models and DataLoader instances. If torch is not
available, the module raises at import-time when attempting to train.
"""
from typing import Optional, Dict, Any
import logging
import time
import os

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None
    DataLoader = None

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device: Optional[str] = None, checkpoint_path: Optional[str] = None):
        if torch is None:
            raise RuntimeError('PyTorch is required for Trainer')
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10, early_stopping: int = 3, tokenizer=None, metrics_callback=None):
        best_metric = None
        no_improve = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                # Expect batch to be pre-collated dict of tensors or token dict
                self.optimizer.zero_grad()
                # Delegate forward to model (models should accept dicts as implemented earlier)
                try:
                    batch = self._move_batch_to_device(batch)
                    preds = self.model(batch)
                except Exception as e:
                    logger.error(f"Model forward failed: {e}")
                    raise

                # compute loss if labels present in batch
                labels = batch.get('efficiency') or batch.get('label')
                if labels is None:
                    logger.debug('No labels found in batch; skipping loss step')
                    continue
                loss = self.loss_fn(preds['efficiency'], labels.to(self.device).float())
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.detach().cpu().numpy())
                steps += 1

            avg_loss = total_loss / steps if steps > 0 else float('nan')
            t1 = time.time()
            logger.info(f"Epoch {epoch} finished. train_loss={avg_loss:.6f} time={(t1-t0):.1f}s")

            # Validation
            if val_loader is not None and metrics_callback is not None:
                metric = metrics_callback(self.model, val_loader, tokenizer=tokenizer, device=self.device)
                logger.info(f"Validation metric: {metric}")
                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    no_improve = 0
                    if self.checkpoint_path:
                        self._save_checkpoint(epoch)
                else:
                    no_improve += 1
                    if no_improve >= early_stopping:
                        logger.info('Early stopping triggered')
                        break

        return best_metric

    def _save_checkpoint(self, epoch: int):
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save({'epoch': epoch, 'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()}, self.checkpoint_path)
            logger.info(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(batch, dict):
            return batch
        moved = {}
        for k, v in batch.items():
            try:
                if isinstance(v, dict):
                    moved[k] = {kk: vv.to(self.device) if hasattr(vv, 'to') else vv for kk, vv in v.items()}
                elif hasattr(v, 'to'):
                    moved[k] = v.to(self.device)
                else:
                    moved[k] = v
            except Exception:
                moved[k] = v
        return moved
