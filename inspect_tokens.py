import pandas as pd
from transformers import AutoTokenizer
import os
import sys

# Define path
data_path = "/scratch/amird/CRISPRO-MAMBA-X/data/merged_crispr_data.csv"
model_name = "zhihan1996/DNABERT-2-117M"

def inspect():
    print(f"Checking data at {data_path}...")
    if not os.path.exists(data_path):
        print("❌ Data file not found!")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")

    # Check for NaN efficiency
    print(f"NaN Efficiency Count: {df['efficiency'].isna().sum()}")

    # Sample sequences
    samples = df['sequence'].head(5).tolist()

    print("\nLoading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return

    print("\nVerifying Tokenization...")
    for seq in samples:
        tokens = tokenizer(seq, return_tensors='pt')['input_ids'][0]
        print(f"\nSeq: {seq[:20]}...")
        print(f"Tokens: {tokens.tolist()[:10]}... (Len: {len(tokens)})")

        # Check if tokens are all identical or all [UNK]
        unique_tokens = len(set(tokens.tolist()))
        if unique_tokens < 5:
             print("⚠️ WARNING: Low token diversity! (Possible [UNK] issue)")
        else:
             print("✅ Token diversity OK.")

if __name__ == "__main__":
    inspect()
