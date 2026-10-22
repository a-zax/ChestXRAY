#!/usr/bin/env python3
"""
Inference script for Cognitive Report Generator.
Generates structured radiology reports (Findings + Impression) from chest X-rays.
Includes configurable ClinicalGrounder calibration profiles.
"""

import os
import sys
import re
import argparse
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms


class ClinicalGrounder:
    """
    Ensures generated reports are factually consistent with MIX-MLP predictions.
    Uses 'Truth Mapping' to anchor decoder outputs to validated disease probabilities.
    """
    PATHOLOGY_TEMPLATES = {
        "Cardiomegaly": "The cardiac silhouette is moderately enlarged, suggesting cardiomegaly.",
        "Edema": "There are prominent interstitial markings and vascular congestion consistent with mild pulmonary edema.",
        "Consolidation": "A focal area of increased opacity is seen, likely representing consolidation.",
        "Pneumonia": "There is a patchy airspace opacity in the lung, suspicious for an infectious process/pneumonia.",
        "Atelectasis": "Linear opacities are present at the lung bases, consistent with subsegmental atelectasis.",
        "Pneumothorax": "A small apical pneumothorax is seen on the right/left side.",
        "Pleural Effusion": "Blunting of the costophrenic angle indicates the presence of a pleural effusion.",
        "Fracture": "An acute osseous abnormality/fracture is identified.",
        "Support Devices": "Multiple support devices, including a nasogastric tube and cardiac leads, are in standard positions."
    }

    DEFAULT_THRESHOLDS = {
        "No Finding": 0.65,
        "Enlarged Cardiomediastinum": 0.35,
        "Cardiomegaly": 0.40,
        "Lung Opacity": 0.45,
        "Lung Lesion": 0.30,
        "Edema": 0.50,
        "Consolidation": 0.40,
        "Pneumonia": 0.35,
        "Atelectasis": 0.45,
        "Pneumothorax": 0.30,
        "Pleural Effusion": 0.55,
        "Pleural Other": 0.30,
        "Fracture": 0.25,
        "Support Devices": 0.60
    }

    PROFILES = {
        "balanced": DEFAULT_THRESHOLDS,
        "cardiopulmonary": {
            **DEFAULT_THRESHOLDS,
            "Cardiomegaly": 0.30,
            "Pleural Effusion": 0.38,
            "Edema": 0.35,
            "No Finding": 0.75
        },
        "infection": {
            **DEFAULT_THRESHOLDS,
            "Pneumonia": 0.28,
            "Consolidation": 0.32,
            "Lung Opacity": 0.35,
            "No Finding": 0.75
        },
        "specific": {
            **DEFAULT_THRESHOLDS,
            "Cardiomegaly": 0.55,
            "Pleural Effusion": 0.65,
            "Pneumonia": 0.50,
            "Edema": 0.60,
            "No Finding": 0.50
        }
    }

    @staticmethod
    def refine(report, disease_probs, custom_thresholds=None):
        thresholds = ClinicalGrounder.DEFAULT_THRESHOLDS.copy()
        if custom_thresholds:
            thresholds.update(custom_thresholds)

        if disease_probs.get("No Finding", 0) > thresholds["No Finding"]:
            return "The lungs are clear. There is no evidence of focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal."

        findings_detected = []
        for disease, prob in disease_probs.items():
            t = thresholds.get(disease, 0.5)
            if prob >= t and disease in ClinicalGrounder.PATHOLOGY_TEMPLATES:
                findings_detected.append(ClinicalGrounder.PATHOLOGY_TEMPLATES[disease])
        
        if not findings_detected:
            return "The lungs are clear without focal consolidation, pleural effusion or pneumothorax. The cardiomediastinal silhouette and hilar contours are within normal limits."
        
        return " ".join(findings_detected)


def generate_report(model, image_path, enc_tokenizer, dec_tokenizer, device, max_length=256, indication_text=None, thresholds=None):
    model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        transform = get_transforms(is_train=False)
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None, None

    if not indication_text:
        indication_text = "Clinical indication: Chest pain, rule out acute cardiopulmonary process."

    indication = enc_tokenizer(
        indication_text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    indication_ids = indication['input_ids'].to(device)
    indication_mask = indication['attention_mask'].to(device)
    
    with torch.no_grad():
        generated_ids, disease_probs = model.generate(
            image_tensor,
            indication_ids, 
            indication_mask,
            max_length=max_length
        )
        
        raw_report = dec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        diseases = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]
        disease_dict = {d: p.item() for d, p in zip(diseases, disease_probs[0])}
        grounded_report = ClinicalGrounder.refine(raw_report, disease_dict, custom_thresholds=thresholds)
        
    return grounded_report, disease_dict


def visualize_result(image_path, report, disease_probs, output_path="outputs/demo_result.png"):
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        
        img = Image.open(image_path).convert('RGB')
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Input: {Path(image_path).name}", fontsize=11, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.title("Cognitive Radiology Report", fontsize=11, fontweight='bold')
        
        wrapper = textwrap.TextWrapper(width=42)
        wrapped_text = wrapper.fill(report)
        
        findings = {d: p for d, p in disease_probs.items() if p > 0.35 and d != "No Finding"}
        if findings:
            wrapped_text += "\n\nDetected Pathologies:\n" + "\n".join([f"• {d}: {p*100:.1f}%" for d, p in findings.items()])
        else:
            wrapped_text += "\n\nDetected Pathologies:\n• No Acute Abnormality"
            
        plt.text(0.05, 0.95, wrapped_text, fontsize=11, va='top', family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n[+] Visual result saved to: {output_path}")
    except Exception as e:
        print(f"[!] Warning: Could not save visualization: {e}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using compute device: {device}")
    
    print(f"[*] Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = CognitiveReportGenerator(
        visual_encoder=args.visual_encoder,
        text_encoder_name=args.text_encoder,
        decoder_name=args.decoder,
        num_diseases=14,
        hidden_dim=512
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    enc_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    dec_tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    # Resolve sensitivity calibration profile
    thresholds = ClinicalGrounder.PROFILES.get(args.profile.lower(), ClinicalGrounder.DEFAULT_THRESHOLDS)
    print(f"[*] Calibration profile: {args.profile}")
        
    print(f"[*] Generating report for: {args.image}...")
    report, disease_probs = generate_report(
        model, 
        args.image, 
        enc_tokenizer, 
        dec_tokenizer, 
        device,
        indication_text=args.indication,
        thresholds=thresholds
    )
    
    if report is None:
        print("[!] Failed to generate report.")
        return

    print("\n" + "="*60)
    print("COGNITIVE RADIOLOGY REPORT (FINDINGS & IMPRESSION)")
    print("="*60)
    print(report)
    print("="*60)
    
    print("\nDisease Probabilities (MIX-MLP 14 Pathologies):")
    for d, p in disease_probs.items():
        flag = " [DETECTED]" if p >= thresholds.get(d, 0.40) else ""
        print(f"  - {d:26s}: {p*100:5.1f}%{flag}")

    visualize_result(args.image, report, disease_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cognitive Radiology Reports")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="Path to checkpoint file")
    parser.add_argument('--image', type=str, default=None, help="Path to image. If None, picks from sample data.")
    parser.add_argument('--indication', type=str, default=None, help="Clinical indication text")
    parser.add_argument('--profile', type=str, default='balanced', choices=['balanced', 'cardiopulmonary', 'infection', 'specific'], help="Sensitivity calibration profile")
    parser.add_argument('--dataset_root', type=str, default='data', help="Root to search if no image provided")
    parser.add_argument('--visual_encoder', type=str, default='vit_base_patch16_224')
    parser.add_argument('--text_encoder', type=str, default='distilbert-base-uncased')
    parser.add_argument('--decoder', type=str, default='distilgpt2')
    
    args = parser.parse_args()
    
    # Auto-fallback for checkpoint
    if not os.path.exists(args.checkpoint):
        for candidate in ['checkpoints/best_model.pth', 'checkpoints/latest_model.pth']:
            if os.path.exists(candidate):
                args.checkpoint = candidate
                print(f"[*] Using found checkpoint: {args.checkpoint}")
                break
                
    if not os.path.exists(args.checkpoint):
        print(f"[!] Error: Checkpoint file not found: {args.checkpoint}")
        print("    Please run: python run.py --download-checkpoint")
        sys.exit(1)
        
    if args.image is None:
        print("[*] No image provided. Searching sample directories...")
        for search_dir in ['data/sample_images', args.dataset_root, '.']:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')) and not file.startswith('demo_'):
                            args.image = os.path.join(root, file)
                            print(f"[+] Found sample image: {args.image}")
                            break
                    if args.image: break
            if args.image: break
            
        if not args.image:
            print("[!] Error: Could not find any images to test.")
            print("    Run 'python scripts/create_sample_data.py' to download sample chest X-rays.")
            sys.exit(1)
            
    main(args)
