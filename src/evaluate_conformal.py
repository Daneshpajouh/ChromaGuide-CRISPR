import torch
from torch.utils.data import DataLoader, random_split
from src.data.crisprofft import CRISPRoffTDataset
from src.model.crispro import CRISPROModel
from src.model.conformal import MondrianConformalPredictor
import numpy as np

def evaluate():
    print("=== CRISPRO-MAMBA-X: Conformal Evaluation ===")

    # 1. Config
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"

    # 2. Data
    # For Eval, we use the same loader logic
    full_dataset = CRISPRoffTDataset(use_mini=False)

    # Split: Train(80) / Cal(10) / Test(10)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    cal_len = int(0.1 * total_len)
    test_len = total_len - train_len - cal_len

    _, cal_dataset, test_dataset = random_split(
        full_dataset, [train_len, cal_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Calibration Set: {len(cal_dataset)}")
    print(f"Test Set:        {len(test_dataset)}")

    # Collate & Tokenize Helpers (Reused)
    def collate_fn(batch):
        clean = []
        for x in batch:
            e = x.get('efficiency')
            if e is None:
                continue
            # avoid copying tensors into torch.tensor(...) (can warn); use as_tensor or pass-through
            e_t = e if torch.is_tensor(e) else torch.as_tensor(e)
            if not torch.isnan(e_t):
                clean.append(x)
        if not clean:
            return None
        return torch.utils.data.dataloader.default_collate(clean)

    vocab = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
    def tokenize(seq_list):
        max_len = max([len(s) for s in seq_list])
        batch = torch.zeros(len(seq_list), max_len, dtype=torch.long)
        for i, s in enumerate(seq_list):
            for j, char in enumerate(s):
                batch[i, j] = vocab.get(char.upper(), 0)
        return batch.to(device)

    cal_loader = DataLoader(cal_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # 3. Load Model
    model = CRISPROModel(d_model=256, n_layers=4, n_modalities=6).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
        print("Loaded Trained Model.")
    except:
        print("Warning: Could not load checkpoint. Using random weights (Metrics will be poor).")

    # 4. Conformal Prediction
    cp = MondrianConformalPredictor(model, alpha=0.1) # 90% Confidence

    # A. Calibrate
    # We need a wrapper to feed the loader correctly to cp.calibrate
    # The CP class expects batches. We'll modify the loop inside CP or wrap here.
    # Let's adjust CP.calibrate to accept a generator or we handle batch prep here.
    # For now, let's just manually feed calibration scores to CP if the class allows,
    # OR simpler: we iterate here and call a helper in CP.

    # Let's rewrite CP usage slightly to be robust:
    # We will pass the Validation Scores directly or let CP handle it.
    # CP.calibrate iterates the loader. But CP doesn't know how to tokenize.
    # FIX: We will do a loop here to collect scores, then compute Q.
    # Actually, simpler: Let's just tokenize in the loader wrapper.

    class TokenizedLoader:
        def __init__(self, loader):
            self.loader = loader
            self.device = device
        def __iter__(self):
            for batch in self.loader:
                if batch is None: continue
                yield {
                    'sequence': tokenize(batch['sequence']),
                    'epigenetics': batch['epigenetics'].to(self.device),
                    'efficiency': batch['efficiency'].to(self.device)
                }
        def __len__(self): return len(self.loader)

    print("Calibrating...")
    cp.calibrate(TokenizedLoader(cal_loader), device=device)

    # B. Test / Evaluate
    print("Evaluating Coverage...")
    total = 0
    covered = 0

    predictions = []
    actuals = []

    for batch in test_loader:
        if batch is None: continue
        seqs = tokenize(batch['sequence'])
        epi = batch['epigenetics'].to(device)
        targets = batch['efficiency'].to(device)

        preds, lowers, uppers = cp.predict(seqs, epi)

        # Check coverage
        # lower <= target <= upper
        is_covered = (targets.cpu() >= lowers) & (targets.cpu() <= uppers)
        covered += is_covered.sum().item()
        total += len(targets)

        predictions.extend(preds.cpu().tolist())
        actuals.extend(targets.cpu().tolist())

    coverage = covered / total if total > 0 else 0
    print(f"Empirical Coverage (Target 0.90): {coverage:.4f}")

    # Spearman Correlation
    from scipy.stats import spearmanr
    corr, p = spearmanr(predictions, actuals)
    print(f"Spearman Correlation: {corr:.4f}")

if __name__ == "__main__":
    evaluate()
