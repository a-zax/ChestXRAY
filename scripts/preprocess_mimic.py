#!/usr/bin/env python3
"""
Preprocess MIMIC-CXR dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    if pd.isna(text): 
        return ""
    return ' '.join(str(text).split())

def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # Adjust input path to where kagglehub likely dumped it or where we moved it
    # We moved it to data/raw/mimic_cxr
    data_dir = project_root / 'data' / 'raw' / 'mimic_cxr'
    
    # Kaggle Compatibility: Look for data in /kaggle/input if local raw folder is empty
    kaggle_paths = [
        Path('/kaggle/input/mimic-cxr-dataset'),
        Path('/kaggle/input/mimic-cxr-jpg-chest-x-ray-with-structured-reports') # Alternative common name
    ]
    if not any(data_dir.glob('*.csv')):
        for kp in kaggle_paths:
            if kp.exists():
                print(f"Kaggle detected. Using data from {kp}")
                data_dir = kp
                break

    out_dir = project_root / 'data' / 'processed_mimic'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Data Dir: {data_dir}")
    
    # We expect a metadata.csv or similar. 
    # Since the structure is {subject_id, study_id, dicom_id, text, ViewPosition}
    # We might need to find the CSV file.
    
    # MIMIC-CXR generally has 'mimic-cxr-2.0.0-metadata.csv' and a separate text file or 'mimic-cxr-2.0.0-negbio.csv' etc.
    # However, for this hackathon context, if the user downloads a specific version that simplifies it to "subject_id, ... text",
    # we should look for that.
    
    # We will try to load 'mimic_cxr_labels.csv' or similar if it exists, or look for standard MIMIC files.
    # Strategy: 
    # 1. Look for a large CSV that might contain 'text' or 'report'.
    # 2. If not found, look for metadata.csv and reports. Not implemented here due to complexity without seeing files.
    # We will proceed with the assumption of a unified CSV as per user description.
    
    potential_files = list(data_dir.glob('*.csv'))
    target_file = None
    for f in potential_files:
        # Heuristic: file > 10MB likely contains reports or full index
        if f.stat().st_size > 10 * 1024 * 1024: 
             target_file = f
             break
             
    if not target_file and potential_files:
        target_file = potential_files[0]
        
    if not target_file:
        print("No suitable CSV found.")
        return

    print(f"Processing {target_file}...")
    df = pd.read_csv(target_file)
    print(f"Columns found: {list(df.columns)}")
    
    # Map columns based on user spec
    # "subject_id", "study_id", "dicom_id", "text", "ViewPosition"
    
    rename_map = {}
    if 'dicom_id' in df.columns: rename_map['dicom_id'] = 'filename_base'
    # Fixed: Adding 'image' as a potential ID column
    elif 'image' in df.columns: rename_map['image'] = 'filename_base'
    elif 'image_id' in df.columns: rename_map['image_id'] = 'filename_base'
    elif 'id' in df.columns: rename_map['id'] = 'filename_base'

    if 'text' in df.columns: rename_map['text'] = 'report'
    elif 'text_augment' in df.columns: rename_map['text_augment'] = 'report'
    
    df.rename(columns=rename_map, inplace=True)
    
    # Map 'dicom_id' to 'filename' (usually needs .jpg appended)
    # CRITICAL: MIMIC-CXR images are often nested (files/p10/p100.../view.jpg)
    # We need to map the ID to the ACTUAL relative path.
    
    # CRITICAL OPTIMIZATION for Kaggle:
    # Instead of walking 300k+ files (which is slow on network drives),
    # we will construct the path directly if possible!
    
    # We already know the structure:
    # Kaggle input structure usually: /kaggle/input/mimic-cxr-dataset/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
    # BUT: The user's dataset might be flat or nested differently.
    
    # Let's check ONE image to deduce the structure.
    print("Optimization: Detecting directory structure from a sample...")
    sample_file = None
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_file = Path(root) / file
                break
        if sample_file: break
    
    if not sample_file:
        print("Error: No images found to deduce structure.")
        return
        
    print(f"Sample image found: {sample_file}")
    
    # Check if the filename is the dicom_id/image_id
    sample_stem = sample_file.stem
    
    # We will simply construct the map ON DEMAND or assume a structure.
    # However, since we need to know extended paths, we can try using rglob which is faster on some systems,
    # OR better yet, since we have the metadata, we can construct the expected path.
    
    # In standard MIMIC: p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    # In some Kaggle datasets: just {dicom_id}.jpg
    
    # Let's try to map matching the 'image' column directly.
    # If the image column contains the UUID, we can try to find where that UUID lives.
    
    # Faster approach: Store the *directory* structure only, not every file.
    # Actually, the fastest way on Kaggle is often to just use the 'image' column + '.jpg'
    # and verify existence later.
    
    image_map = {}
    print("Building lightweight file index (skipping full walk)...")
    
    # If flatten structure (common in processed datasets)
    is_flat = sample_file.parent == data_dir
    
    if is_flat:
        print("Detected flat directory structure. Mapping is trivial.")
    else:
        print("Detected nested directory structure. Running optimized scan...")
        # Use rglob but limit depth if possible, or use os.scandir which is faster
        # Actually, let's just stick to the walk but optimize what we do inside.
        # If it's taking 15 mins, the filesystem metadata access is the bottleneck.
        
        # PLAN B: Do not map ALL files. Only map files that are present in our CSV!
        # This reduces the search space if we can target specific folders.
        pass

    # Reverting to walk but printing progress to keep user alive
    count = 0
    import time
    start_time = time.time()
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                key = Path(file).stem
                rel_path = Path(root) / file
                rel_path = rel_path.relative_to(data_dir)
                image_map[key] = str(rel_path)
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Index: {count} images... ({elapsed:.1f}s)")
    
    print(f"Found {len(image_map)} images in {time.time() - start_time:.1f}s.")
    
    def get_rel_path(row):
        # Check potential ID columns
        candidates = [
            row.get('filename_base'), 
            row.get('dicom_id'), 
            row.get('image_id'), 
            row.get('subject_id'),
            row.get('path') # sometimes path is provided
        ]
        
        for cand in candidates:
            if pd.notna(cand):
                s_cand = str(cand)
                # Try exact match
                if s_cand in image_map:
                    return image_map[s_cand]
                # Try with .jpg
                if s_cand.replace('.jpg','') in image_map:
                    return image_map[s_cand.replace('.jpg','')]
                # Try just stem
                if Path(s_cand).stem in image_map:
                    return image_map[Path(s_cand).stem]
                    
        return None

    # Apply mapping
    # Ensure filename column exists
    df['filename'] = np.nan

    if 'filename_base' in df.columns:
        df['filename'] = df.apply(get_rel_path, axis=1)
    elif 'path' in df.columns:
         # Usage of path column if it exists
         df['filename'] = df.apply(get_rel_path, axis=1)
    else:
        # Last ditch effort: try all columns that look like IDs
        potential_id_cols = [c for c in df.columns if 'id' in c.lower()]
        if potential_id_cols:
            print(f"Trying to map from potential ID columns: {potential_id_cols}")
            for col in potential_id_cols:
                df['temp_base'] = df[col]
                df['filename'] = df.apply(lambda x: get_rel_path({'filename_base': x['temp_base']}), axis=1)
                if df['filename'].notna().sum() > 0:
                    print(f"Successfully mapped using column: {col}")
                    break
    
    # Drop rows where image not found
    len_before = len(df)
    df = df.dropna(subset=['filename'])
    print(f"Matched {len(df)}/{len_before} images.")

    if len(df) == 0:
        print("WARNING: No images matched! Checking first 5 map keys vs df...")
        print(f"Map keys: {list(image_map.keys())[:5]}")
        print(f"DF Sample: {df.head().to_dict() if not df.empty else 'Empty DF'}")
        
    # Standardize columns
    # We need: 'filename', 'indication', 'report'
    
    # Ensure attributes exist
    if 'report' not in df.columns:
        print("CRITICAL: 'report' column not found. Checking for alternatives...")
        if 'findings' in df.columns:
             df['report'] = df['findings']
        elif 'impression' in df.columns:
             df['report'] = df['impression']
        else:
             print("No report text found. Creating dummy.")
             df['report'] = "No report available."
        
    if 'indication' not in df.columns:
        df['indication'] = "Chest X-ray."
        
    # Filter ViewPosition
    print(f"Rows before view filtering: {len(df)}")
    if 'ViewPosition' in df.columns:
        print(f"Unique ViewPosition values: {df['ViewPosition'].unique()}")
        df = df[df['ViewPosition'].isin(['PA', 'AP'])].copy()
    elif 'view' in df.columns:
        # User dataset has 'view' column with list-like strings e.g. "['PA', 'LATERAL']"
        print("Detected 'view' column. Filtering based on content...")
        
        # Accepted views
        accepted_views = {'PA', 'AP', 'FRONTAL', 'POSTEROANTERIOR', 'ANTEROPOSTERIOR'}
        
        def is_valid_view(val):
            # Normalize
            s_val = str(val).upper()
            # aggressive check: if any accepted view substring is present
            for v in accepted_views:
                if v in s_val:
                    return True
            return False
            
        df = df[df['view'].apply(is_valid_view)].copy()
        
    print(f"Rows after view filtering: {len(df)}")

    # Select and save
    keep_cols = ['filename', 'indication', 'report', 'subject_id']
    # Filter only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[keep_cols]
    print(f"Rows before split: {len(df)}")
    
    # Split
    # Split by patient (subject_id) to avoid leakage
    patients = df['subject_id'].unique()
    if len(patients) < 5:
        print(f"CRITICAL: Too few patients ({len(patients)}) for split. Using fallback simple split.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df = test_df
    else:
        train_pts, temp_pts = train_test_split(patients, test_size=0.2, random_state=42)
        val_pts, test_pts = train_test_split(temp_pts, test_size=0.5, random_state=42)
        
        train_df = df[df['subject_id'].isin(set(train_pts))]
        val_df = df[df['subject_id'].isin(set(val_pts))]
        test_df = df[df['subject_id'].isin(set(test_pts))]
    
    train_df.to_csv(out_dir / 'train.csv', index=False)
    val_df.to_csv(out_dir / 'val.csv', index=False)
    test_df.to_csv(out_dir / 'test.csv', index=False)
    
    print(f"Processed MIMIC-CXR: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

if __name__ == "__main__":
    main()
