"""
Train Off-Target Specificity Model (Task 2)
SOTA Target: CCLMoff

 PR-AUC = 0.810 (June 2025)

Quick implementation using existing DeepMEns backbone with binary classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from tqdm import tqdm

from src.model.deepmens import DeepMEnsExact
from src.data.crisprofft import CRISPRoffTDataset
from src.train_deepmens import DeepMEnsDatasetWrapper

class OffTargetClassifier(nn.Module):
    """Binary classifier for off-target activity"""
    def __init__(self):
        super().__init__()
        self.backbone = DeepMEnsExact(seq_len=23)
        # Replace final layer with binary classification
        self.backbone.output = nn.Linear(128, 1)  # Logits for BCE

    def forward(self, x_seq, x_shape, x_pos):
        return self.backbone(x_seq, x_shape, x_pos)

def train_offtarget_quick(epochs=5):
    print("=== TASK 2: OFF-TARGET TRAINING ===")
    print("SOTA: CCLMoff (PR-AUC = 0.810)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data - CRISPRoffT has off-target labels
    print("Loading CRISPRoffT (off-target data)...")
    train_data = CRISPRoffTDataset(split='train', use_mini=False)
    train_wrapped = DeepMEnsDatasetWrapper(train_data)
    train_loader = DataLoader(train_wrapped, batch_size=64, shuffle=True)

    # Model
    model = OffTargetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        losses = []

        for seq, shape, pos, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            seq, shape, pos = seq.to(device), shape.to(device), pos.to(device)
            target = target.to(device)

            # Binary labels: active (>0.5) = 1, inactive = 0
            binary_target = (target > 0.5).float()

            optimizer.zero_grad()
            logits = model(seq, shape, pos).squeeze()
            loss = criterion(logits, binary_target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f}")

    # Save
    torch.save(model.state_dict(), "models/offtarget/offtarget_classifier.pt")
    print("Saved to models/offtarget/offtarget_classifier.pt")

    return model

def evaluate_offtarget(model):
    """Evaluate with PR-AUC metric"""
    print("\\nEvaluating off-target model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data = CRISPRoffTDataset(split='test', use_mini=False)
    test_wrapped = DeepMEnsDatasetWrapper(test_data)
    test_loader = DataLoader(test_wrapped, batch_size=64, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, shape, pos, target in tqdm(test_loader):
            seq, shape, pos = seq.to(device), shape.to(device), pos.to(device)

            logits = model(seq, shape, pos).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = (target.numpy() > 0.5).astype(int)

            all_preds.extend(probs)
            all_labels.extend(labels)

    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    print(f"\\nOff-Target PR-AUC: {pr_auc:.4f}")
    print(f"SOTA (CCLMoff): 0.810")
    print(f"Status: {'✅ BEAT SOTA' if pr_auc > 0.810 else '❌ BELOW SOTA'}")

    return pr_auc

if __name__ == "__main__":
    # Quick training
    model = train_offtarget_quick(epochs=3)
    pr_auc = evaluate_offtarget(model)

    print(f"\\nFinal PR-AUC: {pr_auc:.4f}")
