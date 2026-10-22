import zipfile
import os
from pathlib import Path

def zip_project(output_filename="project_codebase.zip"):
    include_dirs = ["models", "training", "evaluation", "scripts", "configs", "assets", "docs", "tests"]
    include_files = [
        "requirements.txt", "README.md", "AGENT_GUIDE.md", "run.py", "run.ps1", "run.sh"
    ]
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_dirs:
            if os.path.exists(item):
                for root, dirs, files in os.walk(item):
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith((".pyc", ".part", ".pth")):
                            continue
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
        
        for item in include_files:
            if os.path.exists(item):
                zipf.write(item, item)

    size_kb = os.path.getsize(output_filename) // 1024
    print(f"\n[+] Successfully created {output_filename} ({size_kb} KB)")

if __name__ == "__main__":
    zip_project()
