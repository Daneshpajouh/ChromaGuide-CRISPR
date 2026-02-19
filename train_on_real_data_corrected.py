import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from scipy.stats import spearmanr
from pathlib import Path
import json
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR

# Disable Torch compilation if it causes issues on cluster
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Ensure output is flushed for cluster logging
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, name):
       return getattr(self.stream, name)

sys.stdout = Unbuffered(sys.stdout)

# Setup DNABERT-2 path and environment
MODEL_PATH = "zhihan1996/DNABERT-2-117M"

class BetaRegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.fc(x)
        alpha = self.softplus(out[:, 0:1]) + 1.01
        beta = self.softplus(out[:, 1:2]) + 1.01
        return alpha, beta

def beta_nll_loss(alpha, beta, target):
    dist = torch.distributions.Beta(alpha, beta)
    target = torch.clamp(target, 0.001, 0.999)
    return -dist.log_prob(target).mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = "/home/amird/chromaguide_experiments/data/real/merged.csv"
    gold_path = "/home/amird/chromaguide_experiments/test_set_GOLD.csv"

    if not Path(data_path).exists():
        print(f"ERROR: Missing merged.csv at {data_path}")
        return
    if not Path(gold_path).exists():
        print(f"ERROR: Missing gold set at {gold_path}")
        return

    # 1. Leakage Protection
    all_df = pd.read_csv(data_path)
    test_df = pd.read_csv(gold_path)

    gold_sequences = set(test_df['sequence'].tolist())
    train_val_df = all_df[~all_df['sequence'].isin(gold_sequences)].copy()

    print(f"Total merged samples: {len(all_df)}")
    print(f"GOLD samples: {len(test_df)}")
    print(f"Clean Train/Val samples: {len(train_val_df)}")

    # Split train/val
    train_df = train_val_df.sample(frac=0.9, random_state=42)
    val_df = train_val_df.drop(train_df.index)

    # 2. Initialize Model/Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Apply monkey-patches
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        BertEncoder = get_class_from_dynamic_module("bert_layers.BertEncoder", MODEL_PATH)

        original_rebuild = BertEncoder.rebuild_alibi_tensor
        def patched_rebuild(self, size, device=None):
            if device is None:
                device = torch.device('cpu')
            return original_rebuild(self, size, device)
        BertEncoder.rebuild_alibi_tensor = patched_rebuild

        bert_layers_module = sys.modules[BertEncoder.__module__]
        bert_layers_module.flash_attn_qkvpacked_func = None
        print("Successfully patched DNABERT-2.")
    except Exception as e:
        print(f"Warning: Could not apply monkey-patch: {e}")

    backbone = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
    head = BetaRegressionHead(768).to(device)
    backbone = backbone.to(device)

    # We unfreeze the backbone for fine-tuning to reach rho >= 0.911
    backbone.train()

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': 5e-5},
        {'params': head.parameters(), 'lr': 5e-4}
    ], weight_decay=0.01)

    # 3. Training Loop
    epochs = 50
    batch_size = 64 # Reduced for fine-tuning memory
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_rho = -1
    patience = 15
    no_improve = 0

    print("Starting production training with ChromaGuide methodology...")
    for epoch in range(epochs):
        head.train()
        total_loss = 0
        curr_train = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(curr_train), batch_size):
            batch = curr_train.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

            tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)

            # Gathers gradients for backbone too
            outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
            hidden = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]).mean(dim=1)

            alpha, beta = head(hidden)
            loss = beta_nll_loss(alpha, beta, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i // batch_size) % 50 == 0:
                print(f"E{epoch+1} B{i//batch_size} Loss: {loss.item():.4f}")

        scheduler.step()

        # Validation
        backbone.eval()
        head.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i : i+batch_size]
                seqs = batch['sequence'].tolist()
                labels = batch['efficiency'].values

                tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
                outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
                hidden = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]).mean(dim=1)

                alpha, beta = head(hidden)
                preds = (alpha / (alpha + beta)).cpu().numpy()
                val_preds.extend(preds.flatten())
                val_labels.extend(labels)

        val_rho, _ = spearmanr(val_preds, val_labels)
        print(f"Epoch {epoch+1} | Loss: {total_loss/(len(train_df)/batch_size):.4f} | Val Rho: {val_rho:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            # Save both for complete recovery
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': head.state_dict(),
            }, "best_model_full.pt")
            print(f"--> Saved New Best Model (Rho: {val_rho:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Switch back to train for next epoch
        backbone.train()
        head.train()

    # 4. Final Evaluation on GOLD set
    print("-" * 30)
    print("Evaluating on GOLD test set...")
    if Path("best_model_full.pt").exists():
        checkpoint = torch.load("best_model_full.pt")
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])

    backbone.eval()
    head.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            labels = batch['efficiency'].values

            tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
            outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
            hidden = (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]).mean(dim=1)

            alpha, beta = head(hidden)
            preds = (alpha / (alpha + beta)).cpu().numpy()
            test_preds.extend(preds.flatten())
            test_labels.extend(labels)

    test_rho, _ = spearmanr(test_preds, test_labels)
    print(f"FINAL GOLD Spearman Rho: {test_rho:.4f}")

    results = {
        "final_gold_rho": float(test_rho),
        "best_val_rho": float(best_val_rho),
        "target_reached": bool(test_rho >= 0.911),
        "n_train": len(train_df),
        "n_test_gold": len(test_df)
    }
    with open("results_production.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Production results saved to results_production.json")

if __name__ == "__main__":
    main()
