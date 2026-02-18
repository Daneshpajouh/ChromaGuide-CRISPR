#!/usr/bin/env python3
"""
Complete RAG-TTA Training Script for CRISPR
Deploy this to cluster and run immediately
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, TrainerCallback
from sklearn.model_selection import train_test_split
import argparse

# Add project to path
sys.path.insert(0, '/scratch/amird/CRISPRO-MAMBA-X')

from src.model.crispr_rag_tta import CRISPR_RAG_TTA, thermodynamic_loss

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = logs.copy()
            entry['epoch'] = state.epoch
            entry['step'] = state.global_step
            self.history.append(entry)
            pd.DataFrame(self.history).to_csv(self.log_path, index=False)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    foundation_model = "zhihan1996/DNABERT-2-117M"
    k_neighbors = 50
    max_seq_len = 512

    # Training
    batch_size = 128
    epochs = 50
    learning_rate = 3e-4
    warmup_ratio = 0.1

    # Data
    data_path = "/scratch/amird/CRISPRO-MAMBA-X/data/merged_crispr_data.csv"
    min_samples = 10000  # Minimum dataset size

    # Output
    output_dir = "./results_rag_tta"
    checkpoint_dir = f"{output_dir}/checkpoints"

    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16

# ============================================================================
# DATASET
# ============================================================================

class CRISPRDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, efficiencies, tokenizer, max_len=512):
        self.sequences = sequences
        self.efficiencies = efficiencies
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx]).upper()
        eff = float(self.efficiencies[idx])

        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(eff, dtype=torch.float32)
        }

# ============================================================================
# TRAINING
# ============================================================================

def build_memory_bank(model, train_dataset, tokenizer, device):
    """
    Phase 1: Build RAG memory bank by encoding all training samples
    """
    print("\n" + "="*80)
    print("PHASE 1: Building RAG Memory Bank")
    print("="*80)

    model.eval()
    all_embeddings = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Get BERT embeddings
            bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(bert_out, 'last_hidden_state'):
                last_hidden_state = bert_out.last_hidden_state
            else:
                last_hidden_state = bert_out[0]
            embeddings = last_hidden_state[:, 0, :]  # [CLS]

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu().unsqueeze(-1))

            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx * 32}/{len(train_dataset)} samples...")

    # Concatenate all
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Initialize memory
    model.rag_head.initialize_memory(all_embeddings, all_labels)

    print(f"✓ Memory bank built: {len(all_embeddings)} guides indexed")
    return model

def compute_metrics(pred):
    """Evaluation metrics"""
    if isinstance(pred.predictions, tuple):
        predictions = pred.predictions[0]
    else:
        predictions = pred.predictions
    predictions = predictions.flatten()
    labels = pred.label_ids.flatten()

    mask = ~(np.isnan(predictions) | np.isnan(labels))
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) < 2:
        return {"error": "insufficient_data"}

    spearman_rho, _ = spearmanr(labels, predictions)

    return {
        "spearman_rho": float(spearman_rho),
        "mae": float(np.abs(predictions - labels).mean())
    }

def train_rag_tta(args):
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("CRISPR-RAG-TTA TRAINING PIPELINE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    if not os.path.exists(Config.data_path):
        raise FileNotFoundError(f"CRITICAL: Real data not found at {Config.data_path}. Synthetic data fallback is DISABLED.")
    else:
        df = pd.read_csv(Config.data_path)
        print(f"✓ Loaded {len(df)} samples")

    # Z-score normalize
    from scipy.stats import zscore
    df['efficiency_norm'] = zscore(df['efficiency'])

    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.foundation_model, trust_remote_code=True)

    # Datasets
    train_dataset = CRISPRDataset(
        train_df['sequence'].tolist(),
        train_df['efficiency_norm'].tolist(),
        tokenizer,
        Config.max_seq_len
    )

    val_dataset = CRISPRDataset(
        val_df['sequence'].tolist(),
        val_df['efficiency_norm'].tolist(),
        tokenizer,
        Config.max_seq_len
    )

    # Model
    print("\nInitializing model...")
    model = CRISPR_RAG_TTA(
        foundation_model=Config.foundation_model,
        k_neighbors=Config.k_neighbors,
        use_tta=False  # Enable after training
    ).to(Config.device)

    # Build memory bank
    model = build_memory_bank(model, train_dataset, tokenizer, Config.device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        learning_rate=Config.learning_rate,
        warmup_ratio=Config.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="spearman_rho",
        greater_is_better=True,
        bf16=True,  # H100 supports BF16
        dataloader_num_workers=Config.num_workers,
        remove_unused_columns=False,
    )

    # Custom Trainer
    class RAGTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Forward pass
            outputs = model(input_ids, attention_mask, return_components=True)
            predictions = outputs['final_prediction'].squeeze()

            # Loss
            loss = nn.MSELoss()(predictions, labels)

            return (loss, outputs) if return_outputs else loss

    # Trainer
    trainer = RAGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CSVLoggerCallback(f"{Config.output_dir}/training_log.csv")]
    )

    # Train
    print("\n" + "="*80)
    print(f"PHASE 2: Training ({Config.epochs} epochs)")
    print("="*80)

    trainer.train()

    # Evaluate with TTA
    print("\n" + "="*80)
    print("FINAL EVALUATION (With Test-Time Adaptation)")
    print("="*80)
    model.use_tta = True
    results = trainer.evaluate()

    print("\n📊 RESULTS:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Save
    model.save_pretrained(f"{Config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{Config.output_dir}/final_model")

    spearman = results.get('eval_spearman_rho', 0)
    print(f"\n🎯 Spearman ρ = {spearman:.4f}")

    if spearman > 0.85:
        print("   ✅ SUCCESS! Beat SOTA threshold!")
    elif spearman > 0.75:
        print("   ✅ Good! Matched baseline, try ensemble")
    else:
        print("   ⚠️  Below target, check data quality")

    return model, results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=Config.data_path)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    args = parser.parse_args()

    # Override config
    Config.data_path = args.data
    Config.epochs = args.epochs
    Config.batch_size = args.batch_size

    # Create output dir
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)

    # Train
    model, results = train_rag_tta(args)

    print("\n✅ Training complete!")
    print(f"   Model saved to: {Config.output_dir}/final_model")
