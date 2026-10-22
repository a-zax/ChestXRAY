#!/usr/bin/env python3
"""
Training script for Cognitive Report Generator
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys
import os

# Ensure we can import modules from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import ChestXrayDataset, get_transforms

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, accumulation_steps=1):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad() # Initialize gradients once at start
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device).float() # Ensure float for Vit
        indication_ids = batch['indication_ids'].to(device)
        indication_mask = batch['indication_mask'].to(device)
        report_ids = batch['report_ids'].to(device)
        report_mask = batch['report_mask'].to(device)
        
        # Mixed precision forward
        with autocast():
            outputs = model(
                images, 
                indication_ids, 
                indication_mask, 
                report_ids,
                report_mask
            )
            loss = outputs['loss']
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward (Scaling)
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Update parameters
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics (undo normalization for display)
        current_loss = loss.item() * accumulation_steps
        total_loss += current_loss
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device).float()
            indication_ids = batch['indication_ids'].to(device)
            indication_mask = batch['indication_mask'].to(device)
            report_ids = batch['report_ids'].to(device)
            report_mask = batch['report_mask'].to(device)
            
            with autocast():
                outputs = model(
                    images, 
                    indication_ids, 
                    indication_mask, 
                    report_ids,
                    report_mask
                )
                loss = outputs['loss']
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize model
    print("Initializing model...")
    model = CognitiveReportGenerator(
        visual_encoder=args.visual_encoder,
        text_encoder_name=args.text_encoder,
        decoder_name=args.decoder,
        num_diseases=14,
        hidden_dim=512
    )
    model = model.to(device)
    
    # Create datasets
    print("Loading datasets...")
    
    if args.use_mock:
        print("WARNING: Using Mock Dataset for verification!")
        from models.dataset import MockChestXrayDataset
        train_dataset = MockChestXrayDataset(
            num_samples=100, 
            img_size=args.img_size,
            enc_tokenizer_name=args.text_encoder,
            dec_tokenizer_name=args.decoder
        )
        val_dataset = MockChestXrayDataset(
            num_samples=20, 
            img_size=args.img_size,
            enc_tokenizer_name=args.text_encoder,
            dec_tokenizer_name=args.decoder
        )
    else:
        # Adjust paths relative to project root or use provided args
        data_root = Path(args.data_dir)
        
        # Check if files exist
        # Determine processed data path
        dataset_name = args.dataset
        if dataset_name == 'mimic_cxr':
            processed_dir = data_root / 'processed_mimic'
            img_dir = data_root / 'raw/mimic_cxr' # Adjust if nested
            
            # Kaggle override
            if not processed_dir.exists() and Path('/kaggle/input').exists():
                print("Kaggle detected. Please ensure you have run preprocessing first.")
        else:
            processed_dir = data_root / 'processed'
            img_dir = data_root / 'raw/iu_xray/images'

        # Special check: If we are in Kaggle and processed data is missing, we still need to run preprocessing.
        # But images can be pointed to /kaggle/input
        if not img_dir.exists() and Path('/kaggle/input').exists():
             kaggle_iu = Path('/kaggle/input/chest-xrays-indiana-university')
             kaggle_mimic = Path('/kaggle/input/mimic-cxr-dataset')
             if dataset_name == 'mimic_cxr' and kaggle_mimic.exists():
                 img_dir = kaggle_mimic
             elif dataset_name == 'iu_xray' and kaggle_iu.exists():
                 img_dir = kaggle_iu

        train_csv = processed_dir / 'train.csv'
        val_csv = processed_dir / 'val.csv'
        
        if not train_csv.exists():
            print(f"Error: {train_csv} not found. Running preprocessing...")
            # Ideally run script here, but for now just warn
            print("Please run preprocessing script.")
            sys.exit(1)
        
        train_dataset = ChestXrayDataset(
            csv_file=train_csv,
            img_dir=img_dir,
            transform=get_transforms(is_train=True, img_size=args.img_size),
            enc_tokenizer_name=args.text_encoder,
            dec_tokenizer_name=args.decoder,
            max_text_len=args.max_text_len
        )
        
        val_dataset = ChestXrayDataset(
            csv_file=val_csv,
            img_dir=img_dir,
            transform=get_transforms(is_train=False, img_size=args.img_size),
            enc_tokenizer_name=args.text_encoder,
            dec_tokenizer_name=args.decoder,
            max_text_len=args.max_text_len
        )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Mixed precision scaler
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    history = []
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            device, 
            epoch,
            accumulation_steps=args.accumulation_steps
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save latest
        latest_path = checkpoint_dir / 'latest_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }, latest_path)
    
    # Save training history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset', type=str, default='iu_xray', help='iu_xray or mimic_cxr')
    
    # Model
    parser.add_argument('--visual_encoder', type=str, default='vit_base_patch16_224')
    parser.add_argument('--text_encoder', type=str, default='distilbert-base-uncased')
    parser.add_argument('--decoder', type=str, default='distilgpt2')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--max_text_len', type=int, default=256)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_mock', action='store_true', help='Use mock data for verification')
    
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
