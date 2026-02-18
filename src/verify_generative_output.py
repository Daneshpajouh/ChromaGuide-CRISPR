import torch
import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from src.model.rnagenesis_vae import RNAGenesisVAE

def verify_generative_output():
    print("🧪 Verifying RNAGenesis Output Validity...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE if exists, else init random
    vae = RNAGenesisVAE(seq_len=23, latent_dim=256).to(device)
    path = "models/rnagenesis/prod/vae.pt"
    if os.path.exists(path):
        print(f"Loading VAE from {path}")
        vae.load_state_dict(torch.load(path, map_location=device))
    else:
        print("⚠️ VAE not found (Training in progress?). Testing with Random Init for structural check.")

    vae.eval()

    # Sample Latents
    z = torch.randn(10, 256).to(device)

    # Decode
    with torch.no_grad():
        logits = vae.decode(z) # (B, 4, 23)
        probs = torch.softmax(logits, dim=1)
        indices = torch.argmax(probs, dim=1) # (B, 23)

    # Map to ACTG
    vocab = ['A', 'C', 'G', 'T']
    sequences = []
    valid_count = 0

    for i in range(10):
        seq_idx = indices[i].cpu().numpy()
        seq_str = "".join([vocab[idx] for idx in seq_idx])
        sequences.append(seq_str)

        # Check Validity (length 23, ACGT only)
        if len(seq_str) == 23 and all(c in vocab for c in seq_str):
            valid_count += 1

    print(f"\nGenerated {valid_count}/10 Valid Sequences:")
    for i, s in enumerate(sequences):
        print(f"{i+1}: {s}")

    if valid_count == 10:
        print("\n✅ Generative Pipeline Structural Check PASSED.")
    else:
        print("\n❌ Generative Pipeline FAILED.")

if __name__ == "__main__":
    verify_generative_output()
