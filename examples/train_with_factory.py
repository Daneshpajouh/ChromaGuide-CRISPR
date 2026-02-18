#!/usr/bin/env python3
"""
Example: Using ModelFactory for chromaGuide training

This script demonstrates:
1. Creating models with the factory
2. Configuring training with TrainConfig
3. Saving/loading checkpoints
4. Integrating with training loop

Usage:
    python examples/train_with_factory.py --model crispro --epochs 10 --batch-size 32
"""

import argparse
import logging
import os
import torch
import torch.optim as optim
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModelFactory, create_model, save_model
from src.config.chromaguide_config import TrainConfig
from src.model.utils import count_parameters, print_model_summary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ChromaGuide model with ModelFactory'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='chromaguide',
        help='Model name (e.g., crispro, chromaguide, dnabert_mamba)'
    )
    
    # Training config
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda, cpu, mps). Auto-detects if not specified.'
    )
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-interval', type=int, default=5, help='Save every N epochs')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--run-name', type=str, default='experiment_01')
    
    # Data
    parser.add_argument('--data-path', type=str, default='data/mini/crisprofft/mini_crisprofft.txt')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to load')
    
    return parser.parse_args()


def create_simple_dataloader(batch_size: int, samples: int):
    """Create dummy dataloader for demonstration."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    sequences = torch.randn(samples, 4, 23)  # (batch, channels=4, seq_len=23)
    epigenomics = torch.randn(samples, 4, 100)  # (batch, tracks=4, bins=100)
    labels = torch.rand(samples, 1)  # (batch, 1) - efficiency scores
    
    dataset = TensorDataset(sequences, epigenomics, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def train_epoch(model, loader, optimizer, device, loss_fn=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (seq, epi, labels) in enumerate(loader):
        seq = seq.to(device)
        epi = epi.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(seq, epi)
        
        # Compute loss (dummy MSE for demo)
        if isinstance(outputs, dict):
            # Extract appropriate output
            if 'regression' in outputs:
                pred = outputs['regression']
            elif 'mu' in outputs:
                pred = outputs['mu']
            else:
                pred = outputs[list(outputs.keys())[0]]
        else:
            pred = outputs
        
        loss = torch.nn.functional.mse_loss(pred, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def main():
    """Main training loop."""
    args = parse_args()
    
    # =========================================================================
    # 1. Setup
    # =========================================================================
    logger.info(f"{'='*70}")
    logger.info(f"ChromaGuide Training with ModelFactory")
    logger.info(f"{'='*70}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # =========================================================================
    # 2. Initialize Configuration
    # =========================================================================
    config = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        checkpoint=os.path.join(args.checkpoint_dir, 'best.pt'),
        log_dir=args.log_dir,
        run_name=args.run_name,
    )
    config.resolve_paths()
    
    logger.info(f"Config: {config}")
    
    # =========================================================================
    # 3. Create Model using Factory
    # =========================================================================
    factory = ModelFactory(device=config.device)
    logger.info(f"Factory: {factory}")
    
    model_name = config.resolve_model_name()
    logger.info(f"Creating model '{model_name}'...")
    
    model = factory.create(model_name, **config.model_kwargs())
    logger.info(f"✓ Model created")
    
    # Print summary
    print_model_summary(model)
    
    # =========================================================================
    # 4. Setup Training
    # =========================================================================
    device = factory.device
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    logger.info(f"Optimizer: AdamW (lr={config.lr})")
    logger.info(f"Device: {device}")
    
    # =========================================================================
    # 5. Create Data Loaders
    # =========================================================================
    logger.info(f"Creating dataloaders...")
    train_loader = create_simple_dataloader(
        batch_size=config.batch_size,
        samples=100 if args.max_samples is None else args.max_samples
    )
    val_loader = create_simple_dataloader(
        batch_size=config.batch_size,
        samples=20
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch+1}/{config.epochs}")
        logger.info(f"{'-'*70}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, epi, labels in val_loader:
                seq = seq.to(device)
                epi = epi.to(device)
                labels = labels.to(device)
                
                outputs = model(seq, epi)
                if isinstance(outputs, dict):
                    if 'regression' in outputs:
                        pred = outputs['regression']
                    elif 'mu' in outputs:
                        pred = outputs['mu']
                    else:
                        pred = outputs[list(outputs.keys())[0]]
                else:
                    pred = outputs
                
                loss = torch.nn.functional.mse_loss(pred, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Val Loss:   {val_loss:.6f}")
        
        # Step scheduler
        scheduler.step()
        
        # =====================================================================
        # 7. Checkpoint Management
        # =====================================================================
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            
            # Save best checkpoint
            factory.save_checkpoint(
                model,
                config.checkpoint,
                metadata={
                    'model_name': model_name,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config.to_dict(),
                }
            )
            logger.info(f"✓ Saved best checkpoint (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f'epoch_{epoch+1}.pt'
            )
            factory.save_checkpoint(
                model,
                ckpt_path,
                metadata={
                    'model_name': model_name,
                    'epoch': epoch,
                    'val_loss': val_loss,
                }
            )
        
        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # =========================================================================
    # 8. Final Summary
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Best Checkpoint: {config.checkpoint}")
    logger.info(f"Best Val Loss: {best_loss:.6f}")
    logger.info(f"Model Parameters: {count_parameters(model):,}")
    
    # =========================================================================
    # 9. Save Final Training Config
    # =========================================================================
    config_path = os.path.join(args.log_dir, f"{args.run_name}_config.json")
    config.to_json(config_path)
    logger.info(f"Saved config to {config_path}")
    
    logger.info(f"✓ Done!")


if __name__ == '__main__':
    main()
