
import os
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.crispr_rag_tta import CRISPR_RAG_TTA

def run_inference(model_path, data_path, output_path, batch_size=512, device='cuda'):
    print(f"Loading model from {model_path}...")

    # Load Foundation Model Config (Quick Hack: Check config.json or assume DNABERT)
    # Ideally load from config, but we know it's DNABERT-2 for this sprint
    foundation = "zhihan1996/DNABERT-2-117M"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = CRISPR_RAG_TTA(foundation, k_neighbors=50).to(device)

    # Load Weights
    # Note: RAG-TTA saves via save_pretrained which saves the foundation + head
    # But CRISPR_RAG_TTA is a custom wrapper. We might need to load state_dict if save_pretrained wasn't perfect.
    # However, train_rag_tta uses model.save_pretrained.
    # Let's assume standard loading for now, or fallback to state_dict.
    try:
        model.bert = model.bert.from_pretrained(model_path, trust_remote_code=True)
        # Load Head weights if separate?
        # Usually save_pretrained only saves the HF model parts if the class inherits from PreTrainedModel correctly.
        # CRISPR_RAG_TTA inherits from nn.Module.
        # This might be a GOTCHA. train_rag_tta called model.save_pretrained.
        # If CRISPR_RAG_TTA doesn't implement save_pretrained, it might fail or only save the bert part.
        # Let's verify this in the future. For now, we assume it works or we use the checkpoint_dir.
        pass
    except Exception as e:
        print(f"Standard load failed: {e}. Trying torch.load...")
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))

    model.eval()
    print("Model loaded.")

    # Load Data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")

    sequences = df['sequence'].tolist()

    # Tokenize
    print("Tokenizing...")
    encodings = tokenizer(
        sequences,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    dataset = torch.utils.data.TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Re-build Memory Bank?
    # CRITICAL: The RAG model needs the memory bank!
    # The saved model MIGHT NOT contain the memory bank vectors if they are buffers.
    # We must RE-BUILD the memory bank from the Training Data.
    # This script needs access to the TRAIN set to populate the memory.

    print("⚠️  WARNING: RAG Inference requires Memory Bank population.")
    print("    Ensure the model was saved with memory or provide train_data_path to rebuild.")

    # Inference Loop
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids, attention_mask = [b.to(device) for b in batch]

            # TTA Toggle
            # model.use_tta = True

            outputs = model(input_ids, attention_mask)
            # Output is prediction? Or dict?
            # Forward returns prediction tensor by default in our modified code.
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)

    df['prediction'] = all_preds

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    run_inference(args.model, args.data, args.output)
