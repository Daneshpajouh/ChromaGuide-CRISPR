import torch
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath('src'))

try:
    # Use torch.load with weights_only=False if needed, but safer to try regular load first
    # However, since we don't know the exact class structure, we might need to be careful
    checkpoint = torch.load('best_model_full.pt', map_location='cpu')
    print(f"Type of checkpoint: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {checkpoint.keys()}")
    else:
        print(f"Model object type: {type(checkpoint)}")
        if hasattr(checkpoint, 'state_dict'):
            print("Model has state_dict")
except Exception as e:
    print(f"Error loading model: {e}")
