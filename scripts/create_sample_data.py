#!/usr/bin/env python3
"""
Downloads a set of sample public chest X-ray images for immediate inference and testing.
Allows anyone cloning the project to test without downloading full 50GB MIMIC-CXR.
"""

import os
import sys
from pathlib import Path
import requests

SAMPLE_IMAGES = {
    "sample_normal_adult_pa.png": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
    "sample_pediatric.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/72/Boy_chest_x-ray_pic.jpg",
    "sample_pleural_mass.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/eb/Chest_radiograph_showing_fibrous_tumor_of_the_pleura.jpg"
}


def download_sample_images(output_dir: str = "data/sample_images"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "MedicalReportBot/1.0 (test@domain.com)"}
    print(f"[*] Preparing sample chest X-rays in: {out_path.resolve()}")

    for filename, url in SAMPLE_IMAGES.items():
        file_path = out_path / filename
        if file_path.exists() and file_path.stat().st_size > 1000:
            print(f"[+] Sample image already exists: {filename} ({file_path.stat().st_size:,} bytes)")
            continue

        print(f"[*] Downloading {filename}...")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(resp.content)
            print(f"[+] Downloaded: {filename} ({len(resp.content):,} bytes)")
        except Exception as e:
            print(f"[!] Warning: Failed to download {filename}: {e}")

    # Fallback synthetic generator if no network
    downloaded_files = list(out_path.glob("*.[jp][pn]g"))
    if not downloaded_files:
        print("[*] Generating fallback synthetic chest X-ray...")
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (224, 224), color=(20, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.ellipse([30, 40, 100, 180], fill=(60, 60, 60), outline=(100, 100, 100))
            draw.ellipse([124, 40, 194, 180], fill=(60, 60, 60), outline=(100, 100, 100))
            draw.ellipse([80, 100, 144, 170], fill=(120, 120, 120))
            fallback_path = out_path / "sample_synthetic_xray.png"
            img.save(fallback_path)
            print(f"[+] Created synthetic fallback image: {fallback_path}")
        except Exception as ex:
            print(f"[!] Could not create synthetic fallback: {ex}")

    print("[+] Sample images ready!")


if __name__ == "__main__":
    download_sample_images()
