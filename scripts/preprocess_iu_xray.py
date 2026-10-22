#!/usr/bin/env python3
"""
Preprocess IU-Xray dataset for training
Creates train/val/test splits and cleans text
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    """Clean report text"""
    if pd.isna(text):
        return ""
    # Remove excess whitespace
    text = ' '.join(str(text).split())
    # Keep original case (medical terms are case-sensitive, though often lowercased in models)
    return text

def main():
    # Paths
    # Assuming code is run from project root or scripts dir, adjusting for relative path
    # We will use absolute paths relative to execution or fixed relative paths
    
    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    data_dir = project_root / 'data' / 'raw' / 'iu_xray'
    
    # Kaggle Compatibility: Look for data in /kaggle/input if local raw folder is empty
    kaggle_path = Path('/kaggle/input/chest-xrays-indiana-university')
    if not (data_dir / 'indiana_projections.csv').exists() and kaggle_path.exists():
        print(f"Kaggle detected. Using data from {kaggle_path}")
        data_dir = kaggle_path

    out_dir = project_root / 'data' / 'processed'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Project Root: {project_root}")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {out_dir}")

    # Check if data exists
    if not (data_dir / 'indiana_projections.csv').exists():
        print(f"Error: Data not found in {data_dir}")
        print("Please download the IU-Xray dataset or 'Add Data' in Kaggle.")
        return

    # Load data
    print("Loading data...")
    projections = pd.read_csv(data_dir / 'indiana_projections.csv')
    reports = pd.read_csv(data_dir / 'indiana_reports.csv')
    
    # Merge
    data = projections.merge(reports, on='uid')
    print(f"Total samples (raw): {len(data)}")
    
    # Filter: Only Frontal views (PA) for simplicity and consistency
    data = data[data['projection'] == 'Frontal'].copy()
    print(f"After filtering Frontal views: {len(data)}")
    
    # Clean text fields
    print("Cleaning text...")
    data['findings'] = data['findings'].apply(clean_text)
    data['impression'] = data['impression'].apply(clean_text)
    data['indication'] = data['indication'].apply(clean_text)
    
    # Remove empty reports (need at least findings or impression)
    data = data[
        (data['findings'].str.len() > 3) | 
        (data['impression'].str.len() > 3)
    ].copy()
    print(f"After removing empty reports: {len(data)}")
    
    # Create combined report field
    data['report'] = data.apply(
        lambda x: f"FINDINGS: {x['findings']} IMPRESSION: {x['impression']}", 
        axis=1
    )
    
    # Reset index
    data = data.reset_index(drop=True)

    # Split: 80% train, 10% val, 10% test
    train_data, temp_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42
    )
    
    # Save splits
    train_data.to_csv(out_dir / 'train.csv', index=False)
    val_data.to_csv(out_dir / 'val.csv', index=False)
    test_data.to_csv(out_dir / 'test.csv', index=False)
    
    print(f"\n✓ Data preprocessing complete!")
    print(f"Train: {len(train_data)} samples")
    print(f"Val:   {len(val_data)} samples")
    print(f"Test:  {len(test_data)} samples")
    print(f"\nSaved to: {out_dir}")

if __name__ == "__main__":
    main()
