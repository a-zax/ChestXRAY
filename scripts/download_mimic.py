import kagglehub
import shutil
import os
from pathlib import Path

def main():
    print("Downloading MIMIC-CXR dataset using kagglehub...")
    # Kaggle Compatibility
    kaggle_paths = [
        Path('/kaggle/input/mimic-cxr-dataset'),
        Path('/kaggle/input/mimic-cxr-jpg-chest-x-ray-with-structured-reports')
    ]
    for kp in kaggle_paths:
        if kp.exists():
            print(f"Kaggle detected. MIMIC-CXR already available at {kp}")
            print("Skipping download/copy to save space.")
            return

    # Download dataset
    try:
        path = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
        print("Dataset downloaded to:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Define target directory
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / 'data' / 'raw' / 'mimic_cxr'
    
    print(f"Target directory: {target_dir}")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("Copying files to target directory...")
    source_path = Path(path)
    
    # Iterate over files in the downloaded path and copy them
    for item in source_path.iterdir():
        dest = target_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                os.remove(dest)
        
        print(f"Copying {item} to {dest}")
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
        
    print("✓ MIMIC-CXR dataset successfully placed in data/raw/mimic_cxr")

if __name__ == "__main__":
    main()
