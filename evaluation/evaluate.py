#!/usr/bin/env python3
"""
Comprehensive evaluation script
"""

from torch.cuda.amp import autocast
import transformers

# Silence warnings
transformers.logging.set_verbosity_error()

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu
import os

# Ensure we can import from parent
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import ChestXrayDataset, get_transforms

def evaluate_all(model, dataloader, tokenizer, device):
    """
    Evaluate all metrics
    """
    model.eval()
    
    generated_reports = []
    reference_reports = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device).float()
            indication_ids = batch['indication_ids'].to(device)
            indication_mask = batch['indication_mask'].to(device)
            
            # Generate with Mixed Precision
            with autocast():
                generated_ids, disease_probs = model.generate(
                    images, 
                    indication_ids, 
                    indication_mask,
                    max_length=256
                )
            
            # Decode generated reports
            for i in range(len(generated_ids)):
                gen_text = tokenizer.decode(
                    generated_ids[i], 
                    skip_special_tokens=True
                )
                ref_text = tokenizer.decode(
                    batch['report_ids'][i], 
                    skip_special_tokens=True
                )
                
                generated_reports.append(gen_text)
                reference_reports.append(ref_text)
    
    # Load tokenizer (same as used in training/dataset)
    # Re-loading tokenizer here is redundant if passed in, but the original code had it.
    # Actually, let's just use the passed tokenizer if we don't need to rebuild it.
    # The original code structure had evaluate_all separated.
    # Let's fix the structure properly.
    
    # 2. BLEU-4
    # NLTK expects list of list of tokens for refs, list of tokens for hyp
    print("Calculating BLEU-4...")
    references = [[ref.split()] for ref in reference_reports]
    hypotheses = [gen.split() for gen in generated_reports]
    
    try:
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu4 = 0.0
    
    results = {
        'bleu4': bleu4,
        'num_samples': len(generated_reports)
    }
    
    return results, generated_reports, reference_reports

def main(args):
    
    # CRITICAL: For generation with decoder-only models (GPT), padding should be on the LEFT
    # otherwise generated tokens might be overwritten or attention might be wrong.
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
            # Decode generated  reports
        for i in range(len(generated_ids)):
            gen_text = tokenizer.decode(
                generated_ids[i], 
                skip_special_tokens=True
            )
            ref_text = tokenizer.decode(
                batch['report_ids'][i], 
                skip_special_tokens=True
            )
                
            generated_reports.append(gen_text)
            reference_reports.append(ref_text)
            
            # Here you would typically collect disease predictions if you had ground truth labels
            # for classification evaluation.
            # predicted_diseases.append((disease_probs > 0.5).cpu().numpy())
    
    # 2. BLEU-4
    # NLTK expects list of list of tokens for refs, list of tokens for hyp
    print("Calculating BLEU-4...")
    references = [[ref.split()] for ref in reference_reports]
    hypotheses = [gen.split() for gen in generated_reports]
    
    try:
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        bleu4 = 0.0
    
    results = {
        'bleu4': bleu4,
        'num_samples': len(generated_reports)
    }
    
    return results, generated_reports, reference_reports

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = CognitiveReportGenerator(
        visual_encoder=args.visual_encoder,
        text_encoder_name=args.text_encoder,
        decoder_name=args.decoder,
        num_diseases=14,
        hidden_dim=512
    )
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer (same as used in training/dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    
    # CRITICAL: For generation with decoder-only models (GPT), padding should be on the LEFT
    # otherwise generated tokens might be overwritten or attention might be wrong.
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test dataset
    print(f"Loading test dataset ({args.dataset})...")
    data_root = Path(args.data_dir)
    
    if args.dataset == 'mimic_cxr':
        processed_dir = data_root / 'processed_mimic'
        img_dir = data_root / 'raw/mimic_cxr'
        
        # Kaggle Override for MIMIC
        kaggle_mimic = Path('/kaggle/input/mimic-cxr-dataset')
        if not img_dir.exists() and kaggle_mimic.exists():
            print(f"Kaggle detected. using MIMIC data from {kaggle_mimic}")
            img_dir = kaggle_mimic
            
    else:
        processed_dir = data_root / 'processed'
        img_dir = data_root / 'raw/iu_xray/images'
        
        # Kaggle Override for IU-Xray
        kaggle_iu = Path('/kaggle/input/chest-xrays-indiana-university')
        if not img_dir.exists() and kaggle_iu.exists():
            print(f"Kaggle detected. using IU-Xray data from {kaggle_iu}")
            img_dir = kaggle_iu
        
    test_csv = processed_dir / 'test.csv'
    if not test_csv.exists():
        print(f"Error: {test_csv} not found.")
        return

    test_dataset = ChestXrayDataset(
        csv_file=test_csv,
        img_dir=img_dir,
        transform=get_transforms(is_train=False),
        max_text_len=256
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results, generated, references = evaluate_all(
        model, test_loader, tokenizer, device
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"BLEU-4: {results['bleu4']:.4f}")
    print(f"Samples evaluated: {results['num_samples']}")
    print("="*50)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'generated_reports.txt', 'w') as f:
        for report in generated:
            f.write(report.replace('\n', ' ') + '\n')
    
    with open(output_dir / 'reference_reports.txt', 'w') as f:
        for report in references:
            f.write(report.replace('\n', ' ') + '\n')
    
    print(f"\n✓ Results saved to {output_dir}")
    print("\nNext steps for Official Metrics:")
    print("1. Run RadGraph evaluation manually using the generated text files.")
    print("2. Calculate CIDEr using pycocoevalcap.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--dataset', type=str, default='iu_xray', help='iu_xray or mimic_cxr')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model configs matching training
    parser.add_argument('--visual_encoder', type=str, default='vit_base_patch16_224')
    parser.add_argument('--text_encoder', type=str, default='distilbert-base-uncased')
    parser.add_argument('--decoder', type=str, default='distilgpt2')
    
    args = parser.parse_args()
    main(args)
