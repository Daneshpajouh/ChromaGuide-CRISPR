#!/usr/bin/env python3
"""
Example: Using ModelFactory for inference

This script demonstrates:
1. Loading trained models
2. Running inference
3. Batch prediction
4. Saving results

Usage:
    python examples/inference_with_factory.py \
        --checkpoint checkpoints/best.pt \
        --model crispro \
        --sequences data/test_sequences.fasta
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model, load_model_minimal


def one_hot_encode(sequence: str, vocab_size: int = 4) -> np.ndarray:
    """One-hot encode DNA sequence.
    
    Args:
        sequence: DNA sequence string (ACGT)
        vocab_size: Number of nucleotide types (4 for ACGT)
        
    Returns:
        One-hot encoded array (vocab_size, seq_len)
    """
    # Map A, C, G, T to indices
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Convert sequence to indices
    indices = np.array([char_to_idx.get(c, 0) for c in sequence.upper()])
    
    # One-hot encode
    one_hot = np.zeros((vocab_size, len(sequence)))
    one_hot[indices, np.arange(len(sequence))] = 1
    
    return one_hot


def create_batch(sequences: list, pad_to: int = 23) -> torch.Tensor:
    """Create batch of one-hot encoded sequences.
    
    Args:
        sequences: List of DNA sequences
        pad_to: Pad all sequences to this length
        
    Returns:
        Batch tensor (batch_size, 4, seq_len)
    """
    batch = []
    
    for seq in sequences:
        # Ensure sequence is correct length
        if len(seq) < pad_to:
            seq = seq + 'N' * (pad_to - len(seq))
        elif len(seq) > pad_to:
            seq = seq[:pad_to]
        
        one_hot = one_hot_encode(seq)
        batch.append(one_hot)
    
    return torch.from_numpy(np.stack(batch)).float()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='ChromaGuide inference example')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='crispro',
        help='Model name'
    )
    parser.add_argument(
        '--sequences',
        type=str,
        help='Sequences file (FASTA or one per line)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda, cpu, mps). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions'
    )
    
    return parser.parse_args()


def load_sequences_from_file(filepath: str) -> list:
    """Load sequences from FASTA or text file.
    
    Args:
        filepath: Path to sequence file
        
    Returns:
        List of sequences
    """
    sequences = []
    
    if filepath.endswith('.fasta') or filepath.endswith('.fa'):
        # FASTA format
        with open(filepath) as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(''.join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append(''.join(current_seq))
    else:
        # Plain text (one sequence per line)
        with open(filepath) as f:
            for line in f:
                seq = line.strip()
                if seq and not seq.startswith('#'):
                    sequences.append(seq)
    
    return sequences


def main():
    """Main inference loop."""
    args = parse_args()
    
    logger.info(f"{'='*70}")
    logger.info(f"ChromaGuide Inference")
    logger.info(f"{'='*70}")
    
    # =========================================================================
    # 1. Load Model
    # =========================================================================
    logger.info(f"\nLoading checkpoint from: {args.checkpoint}")
    
    # Use minimal loading for faster inference
    model = load_model_minimal(
        args.checkpoint,
        args.model,
        device=args.device
    )
    
    logger.info(f"✓ Model loaded on device: {model.parameters().__next__().device}")
    
    # =========================================================================
    # 2. Load Sequences
    # =========================================================================
    if args.sequences:
        logger.info(f"\nLoading sequences from: {args.sequences}")
        sequences = load_sequences_from_file(args.sequences)
        logger.info(f"Loaded {len(sequences)} sequences")
    else:
        # Demo sequences
        logger.info(f"Using demo sequences")
        sequences = [
            'ACGTACGTACGTACGTACGTACG',
            'GCTAGCTAGCTAGCTAGCTAGCTA',
            'AAAATTTTCCCCGGGGACGTACGT',
        ]
    
    # =========================================================================
    # 3. Run Inference
    # =========================================================================
    logger.info(f"\nRunning inference...")
    
    all_predictions = []
    num_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(sequences))
            
            batch_seqs = sequences[start_idx:end_idx]
            
            # Prepare batch
            batch_tensor = create_batch(batch_seqs)
            batch_tensor = batch_tensor.to(device)
            
            # Forward pass
            outputs = model(batch_tensor)
            
            # Extract predictions
            if isinstance(outputs, dict):
                # Multi-output model
                if 'regression' in outputs:
                    pred = outputs['regression'].squeeze(-1)
                elif 'mu' in outputs:
                    pred = outputs['mu'].squeeze(-1)
                else:
                    # Get first output
                    first_key = list(outputs.keys())[0]
                    pred = outputs[first_key]
            else:
                # Simple output
                pred = outputs.squeeze(-1)
            
            # Store predictions
            all_predictions.extend(pred.cpu().numpy().tolist())
            
            logger.info(f"  Batch {batch_idx+1}/{num_batches}: {len(batch_seqs)} sequences")
    
    # =========================================================================
    # 4. Display and Save Results
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"Results")
    logger.info(f"{'='*70}")
    
    print(f"\n{'Sequence':<30} | {'Efficiency Score':>18}")
    print(f"{'-'*50}")
    
    for seq, pred in zip(sequences, all_predictions):
        # Clamp to [0, 1] if needed
        score = float(pred)
        if isinstance(score, (list, tuple)):
            score = score[0]
        score = np.clip(score, 0, 1)
        
        print(f"{seq:<30} | {score:>18.6f}")
    
    # =========================================================================
    # 5. Save Results (Optional)
    # =========================================================================
    if args.output:
        logger.info(f"\nSaving results to: {args.output}")
        
        with open(args.output, 'w') as f:
            f.write("sequence,efficiency_score\n")
            for seq, pred in zip(sequences, all_predictions):
                score = float(pred)
                if isinstance(score, (list, tuple)):
                    score = score[0]
                score = np.clip(score, 0, 1)
                f.write(f"{seq},{score:.6f}\n")
        
        logger.info(f"✓ Results saved")
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    all_predictions = np.array(all_predictions)
    all_predictions = np.clip(all_predictions, 0, 1)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Summary Statistics")
    logger.info(f"{'='*70}")
    logger.info(f"Mean Efficiency:   {all_predictions.mean():.6f}")
    logger.info(f"Std Efficiency:    {all_predictions.std():.6f}")
    logger.info(f"Min Efficiency:    {all_predictions.min():.6f}")
    logger.info(f"Max Efficiency:    {all_predictions.max():.6f}")
    logger.info(f"Num High Eff (>0.5): {(all_predictions > 0.5).sum()}")
    
    logger.info(f"\n✓ Inference complete!")


if __name__ == '__main__':
    main()
