import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.deepmens import DeepMEnsExact
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper
import os

def refine_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Refining DeepMEns on Normalized CRISPRoffT | Device: {device}")

    # 1. Load Data (Partial for speed)
    train_base = CRISPRoffTDataset(split='train', use_mini=False, max_samples=5000)
    train_dataset = DeepMEnsDatasetWrapper(train_base)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Load Model
    model = DeepMEnsExact(seq_len=23).to(device)
    path = "models/deepmens/deepmens_seed_0.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print("Loaded existing weights for refinement.")

    # 3. Refine (1 Epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Lower LR for refinement
    criterion = nn.MSELoss()

    model.train()
    for seq, shape, pos, label in train_loader:
        seq, shape, pos, label = seq.to(device), shape.to(device), pos.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(seq, shape, pos).squeeze()
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    # 4. Save
    os.makedirs("models/deepmens", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Refined model saved to {path}")

if __name__ == "__main__":
    refine_model()
