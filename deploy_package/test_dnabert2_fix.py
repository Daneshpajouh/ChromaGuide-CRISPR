#!/usr/bin/env python3
"""Quick test of DNABERT-2 loading fix"""
import sys
sys.path.insert(0, 'src')
import torch

print('Testing DNABERT-2 loading...')
try:
    from chromaguide.sequence_encoder import DNABERT2Encoder
    print('✅ Import successful')

    encoder = DNABERT2Encoder(d_model=768)
    print('✅ DNABERT-2 created successfully on CPU')

    # Do a quick forward pass
    dummy_input = torch.randint(0, 30, (2, 23), dtype=torch.long)
    output = encoder(dummy_input)
    print(f'✅ Forward pass successful: output shape {output.shape}')

    encoder = encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'✅ Successfully moved to {"CUDA" if torch.cuda.is_available() else "CPU"}')
    print('')
    print('SUCCESS: DNABERT-2 fix verified!')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
