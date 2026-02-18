
import os
import sys
import time
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from src.utils.ranknet_loss import HybridRankLoss
from src.utils.loss import CombinedLoss
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, roc_auc_score
import pandas as pd
import argparse
from torch.utils.data import Dataset

# Disable TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="RAG-TTA SOTA Training")
parser.add_argument('--batch_size', type=int, default=64, help="Effective batch size")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
parser.add_argument('--lr', type=float, default=2e-4, help="Learning Rate")
parser.add_argument('--dataset', type=str, default="train_val_set.csv", help="Path to Training Data (NOT Gold Set)")
parser.add_argument('--rag', action='store_true', help="Use RAG-TTA Architecture")
# Legacy args for compat if needed (though we overwrite logic)
parser.add_argument('--production', action='store_true', help="Ignored in SOTA mode")
args = parser.parse_args()

BATCH_SIZE = 4
ACCUM_STEPS = max(1, args.batch_size // BATCH_SIZE)
LEARNING_RATE = args.lr
EPOCHS = args.epochs
CHECKPOINT_DIR = "checkpoints"

def train():
    # 1. Hardware
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    print(f"Device: {device.upper()}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Data Loading (The SOTA Way)
    print(f"Loading Training Data from: {args.dataset}")
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset {args.dataset} not found. Run generate_gold_test_set.py first.")
        # Fallback to merged if just testing
        if os.path.exists("merged_crispr_data.csv"):
             print("⚠️ Fallback to merged_crispr_data.csv")
             args.dataset = "merged_crispr_data.csv"
        else:
             return

    # Initialize Dataset and inject specific CSV
    # We rely on CRISPRoffTDataset logic but force the data source
    full_dataset = CRISPRoffTDataset(use_mini=True, data_path_override=args.dataset)
    print(f"Total Training Candidates: {len(full_dataset)}")

    # Standard Split 90/10 (Validation is allowed, Testing is NOT)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # -------------------------------------------------------------------------
    # SOTA INTERVENTION: 50/50 Balanced Sampling
    # -------------------------------------------------------------------------
    print("⚖️  Calculating Balanced Weights...")
    # Map indices to targets
    train_indices = train_dataset.indices
    all_targets = full_dataset.data.iloc[train_indices]['efficiency'].fillna(0).values # Assuming 'efficiency' is target
    all_targets_bin = (all_targets > 0).astype(int)

    class_counts = np.bincount(all_targets_bin)
    if len(class_counts) < 2:
        class_counts = np.array([1, 1]) # Prevent crash if empty

    # Weight = 1 / Count
    weight_0 = 1.0 / (class_counts[0] + 1e-6)
    weight_1 = 1.0 / (class_counts[1] + 1e-6)

    sample_weights = np.array([weight_1 if t > 0 else weight_0 for t in all_targets_bin])
    sample_weights = torch.from_numpy(sample_weights).double()

    print(f"    Class 0 (Inactive): {class_counts[0]}")
    print(f"    Class 1 (Active):   {class_counts[1] if len(class_counts) > 1 else 0}")

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    print("    Sampler ENABLED (50/50 Ratio enforced)")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # 3. Model
    print("Initializing Model...")
    from src.model.crispro import CRISPROModel
    # Optionally load RAG if requested, but defaulting to CRISPRO for stable baseline first
    model = None
    if args.rag:
        try:
            from src.model.crispr_rag_tta import CRISPR_RAG_TTA
            model = CRISPR_RAG_TTA(k_neighbors=10)
            print(">>> Using CRISPR_RAG_TTA")
        except Exception as e:
             print(f">>> RAG import failed: {e}, falling back to CRISPRO")
             model = CRISPROModel(vocab_size=5)
    else:
        model = CRISPROModel(vocab_size=5)

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Slightly tighter decay

    # 4. Loss Functions (New RankNet + Dice/Focal)
    # RankNet for Regression (On-Target), Dice+Focal for Classification (Off-Target)
    criterion_reg = HybridRankLoss(rank_weight=0.9, mse_weight=0.1)

    # SOTA Strategy: Dice Loss (for 98% imbalance) + Focal Loss (for hard examples)
    # Weights: 1.0 each (Balanced)
    criterion_cls = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0)

    # 5. Loop
    best_spearman = -1.0

    print(f"Starting SOTA Sprint for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, batch in enumerate(pbar):
            if batch is None: continue

            seqs = batch['sequence'].to(device)
            epi = batch['epigenetics'].to(device)
            targets = batch['efficiency'].to(device, dtype=torch.float32)

            # Forward
            if args.rag and hasattr(model, 'forward_rag'):
                 # RAG pathway
                 outputs = model(seqs, epi) # wrapper handles it
            else:
                 outputs = model(seqs, epi)

            # Unpack
            if isinstance(outputs, dict):
                pred_cls = outputs['classification']
                pred_reg = outputs['regression']
            elif isinstance(outputs, tuple):
                pred_cls, pred_reg = outputs[0], outputs[1]
            else:
                pred_cls = torch.zeros_like(targets) # dummy
                pred_reg = outputs

            # Losses
            # Ensure shapes
            pred_reg = pred_reg.view(-1, 1)
            targets_reg = targets.view(-1, 1)
            pred_cls = pred_cls.view(-1, 1)
            targets_cls = (targets > 0).float().view(-1, 1)

            l_reg = criterion_reg(pred_reg, targets_reg)
            l_cls = criterion_cls(pred_cls, targets_cls)

            loss = (l_reg + l_cls) / ACCUM_STEPS
            loss.backward()

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS
            steps += 1
            pbar.set_postfix({'Loss': loss.item()*ACCUM_STEPS, 'Reg': l_reg.item()})

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                seqs = batch['sequence'].to(device)
                epi = batch['epigenetics'].to(device)
                targets = batch['efficiency'].to(device)

                out = model(seqs, epi)
                if isinstance(out, dict): score = out['regression']
                elif isinstance(out, tuple): score = out[1]
                else: score = out

                all_preds.extend(score.cpu().flatten().numpy())
                all_targets.extend(targets.cpu().flatten().numpy())

        # Metrics
        sp = spearmanr(all_targets, all_preds)[0]
        pr = pearsonr(all_targets, all_preds)[0]

        print(f"Epoch {epoch+1} Valid: Spearman={sp:.4f}, Pearson={pr:.4f}")

        if sp > best_spearman:
            best_spearman = sp
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("⭐ New Best Model Saved!")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pth"))

def collate_fn(batch):
    batch = [x for x in batch if x['efficiency'] is not None and not np.isnan(x['efficiency'])]
    if len(batch) == 0: return None

    # Pad sequences if needed (Mamba requirement typically % 8 or 256)
    # Simplified here for stability
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    train()
