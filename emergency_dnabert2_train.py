#!/usr/bin/env python3
"""
Emergency DNABERT-2 Training Script with LoRA
Target: Achieve Spearman ρ > 0.75 baseline
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn as nn

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MAX_LEN = 512  # DNABERT-2 max sequence length
BATCH_SIZE = 16
EPOCHS = 50  # Much longer than previous 5 epochs
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Paths
DATA_PATH = "/Users/studio/Desktop/PhD/Proposal/data/merged_crispr_data.csv"  # Will create this
OUTPUT_DIR = "./results_dnabert2_emergency"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

# ============================================================================
# DATASET CLASS
# ============================================================================
class CRISPRDataset(Dataset):
    """Dataset for CRISPR guide efficiency prediction"""

    def __init__(self, sequences, labels, tokenizer, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = str(self.sequences[idx]).upper()
        label = float(self.labels[idx])

        # Tokenize
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
            'labels': torch.tensor(label, dtype=torch.float32)
        }

# ============================================================================
# MODEL WITH REGRESSION HEAD
# ============================================================================
class DNABERTForRegression(nn.Module):
    """DNABERT-2 with regression head for efficiency prediction"""

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)

        # Regression head
        hidden_size = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.regressor(pooled)
        return logits

# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(pred):
    """Compute comprehensive evaluation metrics"""
    predictions = pred.predictions.flatten()
    labels = pred.label_ids.flatten()

    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(labels))
    predictions = predictions[mask]
    labels = labels[mask]

    if len(predictions) < 2:
        return {"error": "insufficient_data"}

    # Compute metrics
    spearman_rho, _ = spearmanr(labels, predictions)
    pearson_r, _ = pearsonr(labels, predictions)
    r2 = r2_score(labels, predictions)
    mae = mean_absolute_error(labels, predictions)

    return {
        "spearman_rho": float(spearman_rho),
        "pearson_r": float(pearson_r),
        "r2": float(r2),
        "mae": float(mae)
    }

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_and_prepare_data(data_path):
    """Load data with proper normalization"""

    print(f"\n{'='*80}")
    print(f"LOADING DATA: {data_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(data_path):
        print(f"⚠️  Data file not found: {data_path}")
        print("Creating dummy dataset for testing...")

        # Create dummy data for testing
        n_samples = 1000
        df = pd.DataFrame({
            'sequence': ['ATCG' * 5 + 'AGG' for _ in range(n_samples)],  # 23bp guide + PAM
            'efficiency': np.random.rand(n_samples)
        })
    else:
        df = pd.read_csv(data_path)

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check for required columns
    if 'sequence' not in df.columns or 'efficiency' not in df.columns:
        raise ValueError("Data must have 'sequence' and 'efficiency' columns")

    # Clean data
    df = df.dropna(subset=['sequence', 'efficiency'])
    print(f"After dropping NaN: {len(df)} samples")

    # Z-score normalization of efficiency (CRITICAL!)
    from scipy.stats import zscore
    df['efficiency_normalized'] = zscore(df['efficiency'])

    print(f"\nEfficiency statistics (raw):")
    print(f"  Mean: {df['efficiency'].mean():.4f}")
    print(f"  Std:  {df['efficiency'].std():.4f}")
    print(f"  Min:  {df['efficiency'].min():.4f}")
    print(f"  Max:  {df['efficiency'].max():.4f}")

    return df

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print(f"\n{'='*80}")
    print(f"EMERGENCY DNABERT-2 TRAINING")
    print(f"Target: Spearman ρ > 0.75")
    print(f"{'='*80}\n")

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load data
    df = load_and_prepare_data(DATA_PATH)

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"\nTrain: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")

    # Create datasets
    train_dataset = CRISPRDataset(
        train_df['sequence'].tolist(),
        train_df['efficiency_normalized'].tolist(),
        tokenizer,
        MAX_LEN
    )

    val_dataset = CRISPRDataset(
        val_df['sequence'].tolist(),
        val_df['efficiency_normalized'].tolist(),
        tokenizer,
        MAX_LEN
    )

    # Initialize model
    print(f"\nInitializing {MODEL_NAME}...")
    model = DNABERTForRegression(MODEL_NAME)

    # Apply LoRA for efficient fine-tuning
    print("\nApplying LoRA (Low-Rank Adaptation)...")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"]  # BERT attention layers
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LR,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="spearman_rho",
        greater_is_better=True,
        fp16=device == "cuda",  # Mixed precision on CUDA
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Custom Trainer
    class RegressionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            loss = nn.MSELoss()(outputs.squeeze(), labels)
            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING ({EPOCHS} epochs)")
    print(f"{'='*80}\n")

    trainer.train()

    # Final evaluation
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION")
    print(f"{'='*80}\n")

    results = trainer.evaluate()

    print("\n📊 FINAL RESULTS:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save model
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    print(f"\n✅ Model saved to: {OUTPUT_DIR}/final_model")
    print("\n🎯 TARGET CHECK:")
    spearman = results.get('eval_spearman_rho', 0)
    if spearman > 0.75:
        print(f"   ✅ SUCCESS! Spearman ρ = {spearman:.4f} > 0.75")
    else:
        print(f"   ⚠️  Below target: Spearman ρ = {spearman:.4f} < 0.75")
        print("      Try: More data, longer training, or ensemble")

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    main()
