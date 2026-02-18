import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Add project root to path
sys.path.append('/scratch/amird/CRISPRO-MAMBA-X')
from transformers import AutoModel

class CRISPRRegressionModel(torch.nn.Module):
    def __init__(self, foundation_model):
        super().__init__()
        self.bert = AutoModel.from_pretrained(foundation_model, trust_remote_code=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        if hasattr(self.bert, 'bert'):
             utils = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
             utils = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(utils, 'last_hidden_state'):
            last_hidden_state = utils.last_hidden_state
        else:
            last_hidden_state = utils[0]
        pooler_output = last_hidden_state[:, 0, :]
        return self.head(pooler_output)

def verify():
    print("Loading data...")
    df = pd.read_csv('/scratch/amird/CRISPRO-MAMBA-X/data/merged_crispr_data.csv')
    df = df.sample(1000) # Small validation set
    print(f"Sampled {len(df)} rows")

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    print("Loading Model...")
    model = CRISPRRegressionModel("zhihan1996/DNABERT-2-117M")
    state_dict = torch.load('/scratch/amird/CRISPRO-MAMBA-X/results_geometric/best_geometric_model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Inference
    preds = []
    targets = []

    print("Running Inference...")
    with torch.no_grad():
        for i, row in df.iterrows():
            seq = str(row['sequence']).upper()
            target = float(row['efficiency'])

            inputs = tokenizer(seq, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            out = model(inputs['input_ids'], inputs['attention_mask'])
            preds.append(out.item())
            targets.append(target)

    preds = np.array(preds)
    targets = np.array(targets)

    rho, p = spearmanr(preds, targets)
    print(f"Spearman Rho: {rho:.4f}")
    print(f"Pred Std: {np.std(preds):.4f}")
    print(f"Target Std: {np.std(targets):.4f}")

    if np.std(preds) < 1e-6:
        print("WARNING: Mode Collapse detected (Const Preds)")

if __name__ == "__main__":
    verify()
