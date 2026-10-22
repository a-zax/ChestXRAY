import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer

class ChestXrayDataset(Dataset):
    def __init__(
        self, 
        csv_file, 
        img_dir, 
        transform=None, 
        enc_tokenizer_name='distilbert-base-uncased',
        dec_tokenizer_name='distilgpt2',
        max_text_len=256,
        max_indication_len=64
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        # Auto-detect subdirectory structure
        # Kaggle Compatibility: Look for images in /kaggle/input if not found locally
        if not self.img_dir.exists() or not any(self.img_dir.iterdir()):
             kaggle_iu = Path('/kaggle/input/chest-xrays-indiana-university')
             kaggle_mimic = Path('/kaggle/input/mimic-cxr-dataset')
             
             # Try to be smart: Check which one actually has files?
             # Or just check if the path being requested looks like one or the other?
             # Since we can't easily know, we trust the Caller (train.py/evaluate.py) to pass the right one.
             # This fallback is a last resort.
             if kaggle_iu.exists() and any(kaggle_iu.iterdir()): 
                 # Only default to IU if we aren't clearly looking for MIMIC
                 if 'mimic' not in str(self.img_dir).lower():
                    self.img_dir = kaggle_iu
             
             if kaggle_mimic.exists() and any(kaggle_mimic.iterdir()):
                 if 'mimic' in str(self.img_dir).lower() or not kaggle_iu.exists():
                    self.img_dir = kaggle_mimic

        if (self.img_dir / 'images_normalized').exists():
            self.img_dir = self.img_dir / 'images_normalized' # IU-Xray
        elif (self.img_dir / 'files').exists():
             self.img_dir = self.img_dir / 'files' # MIMIC-CXR often has 'files'
            
        self.transform = transform
        self.transform = transform
        self.max_text_len = max_text_len
        self.max_indication_len = max_indication_len
        
        # Initialize tokenizers
        # Encoder tokenizer for Clinical Indication (Context)
        self.enc_tokenizer = AutoTokenizer.from_pretrained(enc_tokenizer_name)
        
        # Decoder tokenizer for Report Generation
        self.dec_tokenizer = AutoTokenizer.from_pretrained(dec_tokenizer_name)
        if self.dec_tokenizer.pad_token is None:
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.img_dir / row['filename']
        try:
            image = Image.open(img_path).convert('RGB')
            # Optimization: Resize to a smaller size before converting to numpy 
            # to avoid large memory allocations for high-res images
            if max(image.size) > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Helper for handling NaN/empty strings
        def sanitize_text(text):
            return str(text) if pd.notna(text) else ""

        indication_text = sanitize_text(row.get('indication', ''))
        report_text = sanitize_text(row.get('report', ''))

        # Tokenize indication (Encoder)
        indication = self.enc_tokenizer(
            indication_text,
            max_length=self.max_indication_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize report (Decoder)
        report = self.dec_tokenizer(
            report_text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        disease_labels = torch.zeros(14, dtype=torch.float32)
        
        return {
            'image': image,
            'indication_ids': indication['input_ids'].squeeze(0),
            'indication_mask': indication['attention_mask'].squeeze(0),
            'report_ids': report['input_ids'].squeeze(0),
            'report_mask': report['attention_mask'].squeeze(0),
            'disease_labels': disease_labels,
            'filename': row['filename']
        }

# Define transforms
def get_transforms(is_train=True, img_size=224):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=5, 
                p=0.3
            ),
            A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

class MockChestXrayDataset(Dataset):
    """
    Mock dataset for pipeline verification without real data.
    """
    def __init__(self, num_samples=100, img_size=224, 
                 enc_tokenizer_name='distilbert-base-uncased',
                 dec_tokenizer_name='distilgpt2'):
        self.num_samples = num_samples
        self.img_size = img_size
        
        self.enc_tokenizer = AutoTokenizer.from_pretrained(enc_tokenizer_name)
        self.dec_tokenizer = AutoTokenizer.from_pretrained(dec_tokenizer_name)
        if self.dec_tokenizer.pad_token is None:
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random image
        image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Apply transform (normalize)
        transform = get_transforms(is_train=True, img_size=self.img_size)
        augmented = transform(image=image)
        image = augmented['image']
        
        # Random inputs respecting vocab sizes
        indication_ids = torch.randint(0, self.enc_tokenizer.vocab_size, (64,))
        indication_mask = torch.ones(64)
        
        report_ids = torch.randint(0, self.dec_tokenizer.vocab_size, (256,))
        report_mask = torch.ones(256)
        
        disease_labels = torch.zeros(14, dtype=torch.float32)
        
        return {
            'image': image,
            'indication_ids': indication_ids,
            'indication_mask': indication_mask,
            'report_ids': report_ids,
            'report_mask': report_mask,
            'disease_labels': disease_labels,
            'filename': f'mock_{idx}.png'
        }
