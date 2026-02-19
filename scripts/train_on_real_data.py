import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from pathlib import Path
import json
import logging
import sys

# Add src to path for ChromaGuide imports
sys.path.insert(0, '/home/amird/chromaguide_experiments/src')

try:
    from chromaguide.chromaguide_model import ChromaGuideModel
    USE_CHROMAGUIDE = True
    print("Successfully imported ChromaGuideModel - using multimodal architecture")
except ImportError as e:
    print(f"ChromaGuideModel import failed: {e}")
    print("Falling back to DNABERT-2 + BetaRegression")
    USE_CHROMAGUIDE = False

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
MAMBA_ENV = "/home/amird/env_chromaguide"
LOCAL_CACHE = "/home/amird/.cache/huggingface/hub"
os.environ["HF_HOME"] = LOCAL_CACHE
MODEL_PATH = "zhihan1996/DNABERT-2-117M"

class BetaRegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
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

    # Load data - absolute paths for Narval
    data_path = "/home/amird/chromaguide_experiments/data/real/merged.csv"
    gold_path = "/home/amird/chromaguide_experiments/test_set_GOLD.csv"

    if not Path(data_path).exists():
        print(f"ERROR: Missing merged.csv at {data_path}")
        return
    if not Path(gold_path).exists():
        print(f"ERROR: Missing gold set at {gold_path}")
        return

    # 1. Leakage Protection: Remove GOLD set samples from train/val
    all_df = pd.read_csv(data_path)
    test_df = pd.read_csv(gold_path)

    # Filter out any sequence that exists in the GOLD set
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
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Apply monkey-patch to fix DNABERT-2 'meta' device bug and Triton compatibility
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        BertEncoder = get_class_from_dynamic_module("bert_layers.BertEncoder", MODEL_PATH)

        # Patch 1: Fix 'meta' device bug
        original_rebuild = BertEncoder.rebuild_alibi_tensor
        def patched_rebuild(self, size, device=None):
            if device is None:
                device = torch.device('cpu') # Force CPU for initialization logic
            return original_rebuild(self, size, device)
        BertEncoder.rebuild_alibi_tensor = patched_rebuild

        # Patch 2: Disable Triton flash attention (fixes dot() trans_b error in newer Triton)
        bert_layers_module = sys.modules[BertEncoder.__module__]
        bert_layers_module.flash_attn_qkvpacked_func = None

        print("Successfully patched DNABERT-2: Alibi tensor logic & Triton compatibility fixed.")
    except Exception as e:
        print(f"Warning: Could not apply monkey-patch: {e}")

    # Model initialization
    if USE_CHROMAGUIDE:
        # Use full ChromaGuide model with multimodal fusion
        model = ChromaGuideModel(
            encoder_type='dnabert',  # Use DNABERT-2 as backbone
            d_model=768,  # DNABERT-2 hidden dimension
            seq_len=23,
            num_epi_tracks=4,  # Number of epigenomic tracks
            num_epi_bins=100,  # Bins per track
            use_epigenomics=True,  # Enable multimodal fusion
            use_gate_fusion=True,  # Use GatedAttentionFusion
            use_mi_regularizer=False,  # Disable MI regularizer for now
            dropout=0.1,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        print("Using ChromaGuideModel with GatedAttentionFusion")

    else:
        # Fallback: DNABERT-2 + BetaRegressionHead
        backbone = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True, low_cpu_mem_usage=False)
        head = BetaRegressionHead(768).to(device)
        backbone = backbone.to(device)

        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        print("Using fallback DNABERT-2 + BetaRegressionHead")

    # 3. Training Loop
    epochs = 10
    batch_size = 128
    best_val_rho = -1
    patience = 3
    no_improve = 0

    print("Starting production training...")
    for epoch in range(epochs):
        if USE_CHROMAGUIDE:
            model.train()
        else:
            head.train()
        total_loss = 0

        # Shuffle train
        curr_train = train_df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(curr_train), batch_size):
            batch = curr_train.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            labels = torch.tensor(batch['efficiency'].values, dtype=torch.float32).to(device).unsqueeze(1)

            # Forward pass - handle both ChromaGuide and fallback models
            if USE_CHROMAGUIDE:
                # Convert sequences to one-hot encoding for ChromaGuideModel
                seq_tensor = torch.zeros(len(seqs), 4, 23, device=device)
                for j, seq in enumerate(seqs):
                    for k, nucleotide in enumerate(seq[:23]):  # Limit to max length
                        if nucleotide == 'A': seq_tensor[j, 0, k] = 1
                        elif nucleotide == 'C': seq_tensor[j, 1, k] = 1
                        elif nucleotide == 'G': seq_tensor[j, 2, k] = 1
                        elif nucleotide == 'T': seq_tensor[j, 3, k] = 1

                # No epigenomic data available yet, use sequence-only mode
                output = model(seq_tensor, epi_tracks=None, epi_mask=None)
                loss_dict = model.compute_loss(output, labels)
                loss = loss_dict['total_loss']

            else:
                # Fallback: DNABERT-2 + BetaRegressionHead
                tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
                with torch.no_grad():
                    outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
                    hidden = outputs.last_hidden_state.mean(dim=1)

                alpha, beta = head(hidden)
                loss = beta_nll_loss(alpha, beta, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i // batch_size) % 100 == 0:
                print(f"Epoch {epoch+1} Batch {i//batch_size} Loss: {loss.item():.4f}")

        # Validation
        if USE_CHROMAGUIDE:
            model.eval()
        else:
            head.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for i in range(0, len(val_df), batch_size):
                batch = val_df.iloc[i : i+batch_size]
                seqs = batch['sequence'].tolist()
                labels = batch['efficiency'].values

                if USE_CHROMAGUIDE:
                    # Convert sequences to one-hot for ChromaGuideModel
                    seq_tensor = torch.zeros(len(seqs), 4, 23, device=device)
                    for j, seq in enumerate(seqs):
                        for k, nucleotide in enumerate(seq[:23]):
                            if nucleotide == 'A': seq_tensor[j, 0, k] = 1
                            elif nucleotide == 'C': seq_tensor[j, 1, k] = 1
                            elif nucleotide == 'G': seq_tensor[j, 2, k] = 1
                            elif nucleotide == 'T': seq_tensor[j, 3, k] = 1

                    output = model(seq_tensor, epi_tracks=None, epi_mask=None)
                    preds = output['mu'].cpu().numpy().flatten()

                else:
                    # Fallback: DNABERT-2 + BetaRegressionHead
                    tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
                    outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
                    hidden = outputs.last_hidden_state.mean(dim=1)

                    alpha, beta = head(hidden)
                    preds = (alpha / (alpha + beta)).cpu().numpy().flatten()

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_rho, _ = spearmanr(val_preds, val_labels)
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/(len(train_df)/batch_size):.4f} | Val Rho: {val_rho:.4f}")

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            if USE_CHROMAGUIDE:
                torch.save(model.state_dict(), "best_chromaguide_model.pt")
            else:
                torch.save(head.state_dict(), "best_head_production.pt")
            print(f"--> Saved New Best Model (Rho: {val_rho:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # 4. Final Evaluation on GOLD set
    print("-" * 30)
    print("Evaluating on GOLD test set...")
    # Load best checkpoint
    if USE_CHROMAGUIDE:
        if Path("best_chromaguide_model.pt").exists():
            model.load_state_dict(torch.load("best_chromaguide_model.pt"))
        model.eval()
    else:
        if Path("best_head_production.pt").exists():
            head.load_state_dict(torch.load("best_head_production.pt"))
        head.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i : i+batch_size]
            seqs = batch['sequence'].tolist()
            labels = batch['efficiency'].values

            if USE_CHROMAGUIDE:
                # Convert sequences to one-hot for ChromaGuideModel
                seq_tensor = torch.zeros(len(seqs), 4, 23, device=device)
                for j, seq in enumerate(seqs):
                    for k, nucleotide in enumerate(seq[:23]):
                        if nucleotide == 'A': seq_tensor[j, 0, k] = 1
                        elif nucleotide == 'C': seq_tensor[j, 1, k] = 1
                        elif nucleotide == 'G': seq_tensor[j, 2, k] = 1
                        elif nucleotide == 'T': seq_tensor[j, 3, k] = 1

                output = model(seq_tensor, epi_tracks=None, epi_mask=None)
                preds = output['mu'].cpu().numpy().flatten()

            else:
                # Fallback: DNABERT-2 + BetaRegressionHead
                tokens = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=30).to(device)
                outputs = backbone(tokens['input_ids'], tokens['attention_mask'])
                hidden = outputs.last_hidden_state.mean(dim=1)

                alpha, beta = head(hidden)
                preds = (alpha / (alpha + beta)).cpu().numpy().flatten()

            test_preds.extend(preds)
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
    with open("results_v2_production.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Production results saved to results_v2_production.json")

if __name__ == "__main__":
    main()
