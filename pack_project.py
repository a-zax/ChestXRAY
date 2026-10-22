import zipfile
import os
from pathlib import Path

def zip_project(output_filename="project_codebase.zip"):
    # Folders and files to include
    include_dirs = ["models", "training", "evaluation", "scripts", "configs"]
    include_files = ["requirements.txt", "README.md", "submission.txt", "Run_on_Colab.ipynb", "Run_on_Kaggle.ipynb"]
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_dirs:
            if os.path.exists(item):
                for root, dirs, files in os.walk(item):
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith(".pyc"):
                            continue
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
                        print(f"Added: {file_path}")
        
        for item in include_files:
            if os.path.exists(item):
                zipf.write(item, item)
                print(f"Added: {item}")

    print(f"\nSuccessfully created {output_filename}")

if __name__ == "__main__":
    zip_project()
