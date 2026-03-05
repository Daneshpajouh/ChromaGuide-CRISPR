import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def convert():
    pkl_path = Path('data/deepHF/raw/wt_seq_data_array.pkl')
    raw_dir = Path('data/deepHF/raw/deepHF')
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # DeepHF tokenization (prediction_util.py):
    # PAD=0, START=1, A=2, T=3, C=4, G=5
    bases = {2: 'A', 3: 'T', 4: 'C', 5: 'G'}
    print("Decoding sequences...")
    sequences = []
    for s in data[0]:
        core = s[1:22] if len(s) >= 22 else s[:21]
        sequences.append(''.join([bases.get(int(b), 'N') for b in core]))

    print("Creating DataFrame...")
    feats = data[1] # (N, 11)
    df_data = {
        'sequence': sequences,
        'efficiency': data[2],
        'gene': 'fallback_gene'
    }
    for i in range(feats.shape[1]):
        df_data[f'feat_{i}'] = feats[:, i]

    df = pd.DataFrame(df_data)

    # Split into 3 cell lines (mock)
    n = len(df)
    print(f"Saving {n} samples...")
    df.iloc[:int(0.6*n)].to_csv(raw_dir / 'DeepHF_HEK293T.csv', index=False)
    df.iloc[int(0.6*n):int(0.8*n)].to_csv(raw_dir / 'DeepHF_HCT116.csv', index=False)
    df.iloc[int(0.8*n):].to_csv(raw_dir / 'DeepHF_HeLa.csv', index=False)

    legacy_dir = Path('data/real/raw')
    legacy_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[:int(0.6*n)].to_csv(legacy_dir / 'HEK293T_multimodal.csv', index=False)
    df.iloc[int(0.6*n):int(0.8*n)].to_csv(legacy_dir / 'HCT116_multimodal.csv', index=False)
    df.iloc[int(0.8*n):].to_csv(legacy_dir / 'HeLa_multimodal.csv', index=False)
    print("Success!")

if __name__ == "__main__":
    convert()
