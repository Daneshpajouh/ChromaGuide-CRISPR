import torch
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.model.deepmens import DeepMEnsExact
from src.model.ensemble import DeepMEnsEnsemble
from src.data.dna_shape import shape_featurizer
import torch.nn.functional as F

def load_ensemble(model_dir="models/deepmens", num_models=5, device="cpu"):
    models = []
    print(f"Loading {num_models} models from {model_dir}...")
    for seed in range(num_models):
        path = os.path.join(model_dir, f"deepmens_seed_{seed}.pt")
        if not os.path.exists(path):
            print(f"⚠️ Warning: Model {path} not found. Skipping.")
            continue

        model = DeepMEnsExact(seq_len=23)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    if len(models) == 0:
        raise ValueError("No models loaded! Train ensemble first.")

    ensemble = DeepMEnsEnsemble(models)
    return ensemble

def predict_single(ensemble, sequence, device="cpu"):
    """
    Predict efficiency for a single sgRNA string.
    """
    # Preprocess
    # 1. Sequence
    vocab = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
    seq_ints = torch.tensor([vocab.get(c, 5) for c in sequence], dtype=torch.long)
    seq_onehot = F.one_hot(seq_ints.clamp(0, 4), num_classes=5)[:, :4]
    seq_onehot = seq_onehot.permute(1, 0).float().unsqueeze(0) # (1, 4, L)

    # 2. Shape
    shape_feat = shape_featurizer.get_shape(sequence).unsqueeze(0) # (1, 4, L)

    # 3. Position
    pos_ints = torch.arange(len(sequence), dtype=torch.long).unsqueeze(0) # (1, L)

    # Predict
    with torch.no_grad():
        mean, std = ensemble(
            seq_onehot.to(device),
            shape_feat.to(device),
            pos_ints.to(device)
        )

    return mean.item(), std.item()

def run_predictions(input_file, output_file, model_dir="models/deepmens"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = load_ensemble(model_dir, device=device)

    print(f"Computing predictions for {input_file}...")

    # Load Input
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith(".txt"):
        df = pd.read_csv(input_file, sep="\t")
    else:
        # Assume raw text, one sequence per line
        logger = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    logger.append({'Guide_sequence': line.strip()})
        df = pd.DataFrame(logger)

    # Find sequence column
    if 'Guide_sequence' in df.columns:
        seq_col = 'Guide_sequence'
    elif 'seq' in df.columns:
        seq_col = 'seq'
    else:
        seq_col = df.columns[0] # Assume first column

    predictions = []
    uncertainties = []

    for seq in tqdm(df[seq_col]):
        # Ensure length 23
        if len(seq) < 23:
            # Pad or skip? For CRISPR, usually 20+PAM.
            # If 20bp, pad with NNN?
            # DeepMEns expects 23.
            # Assuming input includes PAM.
            pass

        m, s = predict_single(ensemble, seq, device)
        predictions.append(m)
        uncertainties.append(s)

    df['DeepMEns_Efficiency'] = predictions
    df['Uncertainty_Std'] = uncertainties

    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV/TXT")
    parser.add_argument("--output", type=str, default="results.csv", help="Output file")
    parser.add_argument("--model_dir", type=str, default="models/deepmens", help="Model directory")

    args = parser.parse_args()

    # Special: if input is a literal sequence
    if not os.path.exists(args.input) and set(args.input).issubset(set('ACGTN')):
        print(f"Predicting for single sequence: {args.input}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ensemble = load_ensemble(args.model_dir, device=device)
        m, s = predict_single(ensemble, args.input, device)
        print(f"Efficiency: {m:.4f} ± {s:.4f}")
    else:
        run_predictions(args.input, args.output, args.model_dir)
