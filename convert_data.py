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

    bases = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}
    print("Decoding sequences...")
    sequences = [''.join([bases.get(b, 'N') for b in s[:21]]) for s in data[0]]

    print("Creating DataFrame...")
    feats = data[1] # (N, 11)
    df_data = {
        'sequence': sequences,
        'efficiency': data[2],
        'gene': 'placeholder_gene'
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
