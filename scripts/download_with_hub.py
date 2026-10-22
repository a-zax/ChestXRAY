import kagglehub
import shutil
import os
from pathlib import Path

def main():
    print("Downloading dataset using kagglehub...")
    # Kaggle Compatibility
    kaggle_path = Path('/kaggle/input/chest-xrays-indiana-university')
    if kaggle_path.exists():
        print(f"Kaggle detected. IU-Xray already available at {kaggle_path}")
        print("Skipping download/copy to save space.")
        return

    # Download latest version
    try:
        path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
        print("Dataset downloaded to:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Define target directory relative to this script
    # script is in /scripts, we want /data/raw/iu_xray
    target_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'iu_xray'
    # The script is in d:/braindead/scripts
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / 'data' / 'raw' / 'iu_xray'
    
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
        
    print("✓ Dataset successfully placed in data/raw/iu_xray")
    
    # Verify
    if (target_dir / 'indiana_reports.csv').exists():
        print("Verification successful: indiana_reports.csv found.")
    else:
        print("Warning: indiana_reports.csv not found in target directory.")

if __name__ == "__main__":
    main()
