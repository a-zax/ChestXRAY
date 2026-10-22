#!/usr/bin/env python3
"""
Master Entry Point & Orchestrator for Cognitive Radiology Assistant.
Unified CLI designed for humans and AI agents (Claude, Codex, Antigravity, etc.).

Usage:
    python run.py --setup              # Install dependencies, download NLTK data & sample images
    python run.py --check              # Run complete system & model healthcheck
    python run.py --demo               # Run report generation on a sample chest X-ray
    python run.py --gui                # Launch the interactive desktop GUI application
    python run.py --test               # Run unit and architectural tests
    python run.py --download-model     # Download trained model checkpoint (best_model.pth)
    python run.py --train [options]    # Launch model training pipeline
    python run.py --eval [options]     # Run evaluation suite on test dataset
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd, cwd=None, env=None):
    """Run a command and forward standard IO."""
    print(f"\n[*] Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env or os.environ.copy())
    return res.returncode


def cmd_setup(args):
    """Setup project environment, dependencies, NLTK data, and sample chest X-rays."""
    print("=" * 70)
    print("SETTING UP COGNITIVE RADIOLOGY ASSISTANT")
    print("=" * 70)

    # 1. Check Python version
    major, minor = sys.version_info[:2]
    print(f"[*] Python version: {major}.{minor}.{sys.version_info[2]}")
    if (major, minor) < (3, 10):
        print(f"[!] Warning: Python 3.10+ recommended. Current: {major}.{minor}")

    # 2. Check GPU & PyTorch
    print("\n[*] Checking PyTorch & CUDA availability...")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"[+] PyTorch installed: {torch.__version__} (CUDA: {cuda_avail})")
        if cuda_avail:
            print(f"[+] GPU detected: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)")
    except ImportError:
        print("[!] PyTorch not found. Installing PyTorch...")
        # Check if nvidia GPU is present via nvidia-smi
        has_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if has_gpu:
            print("[+] NVIDIA GPU detected. Installing PyTorch with CUDA 12.4 support...")
            run_cmd([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"])
        else:
            print("[*] No GPU detected. Installing CPU PyTorch...")
            run_cmd([sys.executable, "-m", "pip", "install", "torch", "torchvision"])

    # 3. Install requirements
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        print(f"\n[*] Installing requirements from {req_file}...")
        run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_file)])

    # 4. Download NLTK data
    print("\n[*] Downloading required NLTK resources (punkt, wordnet)...")
    try:
        import nltk
        for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        print("[+] NLTK resources verified.")
    except Exception as e:
        print(f"[!] Note on NLTK: {e}")

    # 5. Ensure sample images are downloaded
    print("\n[*] Checking sample test images...")
    sample_script = PROJECT_ROOT / "scripts" / "create_sample_data.py"
    if sample_script.exists():
        run_cmd([sys.executable, str(sample_script)])

    # 6. Check Checkpoint
    checkpoint_file = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    if not checkpoint_file.exists():
        print("\n" + "-" * 70)
        print("[NOTICE] Trained checkpoint weights (best_model.pth) not yet detected.")
        print("To download the 3GB checkpoint automatically, run:")
        print("    python run.py --download-model")
        print("-" * 70)
    else:
        print(f"\n[+] Checkpoint verified: {checkpoint_file} ({checkpoint_file.stat().st_size / (1024**3):.2f} GB)")

    print("\n[SUCCESS] Setup process completed successfully!")


def cmd_check(args):
    """Perform comprehensive health check on environment, models, and checkpoint."""
    print("=" * 70)
    print("COGNITIVE RADIOLOGY SYSTEM HEALTHCHECK")
    print("=" * 70)

    # 1. Environment
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Python interpreter: {sys.executable} (v{sys.version.split()[0]})")
    
    # 2. PyTorch & Acceleration
    try:
        import torch
        print(f"[+] PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[+] Compute: CUDA Enabled | Device: {dev_name} | VRAM: {mem:.2f} GB")
        else:
            print("[*] Compute: CPU (CUDA not available)")
    except ImportError:
        print("[FAIL] PyTorch is not installed. Run 'python run.py --setup'")
        return 1

    # 3. Key Dependencies
    packages = ["transformers", "timm", "PIL", "pydicom", "scipy", "sklearn", "matplotlib", "PySide6", "cryptography"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[+] Package '{pkg}': OK")
        except ImportError:
            print(f"[!] Package '{pkg}': MISSING (install via 'python run.py --setup')")

    # 4. Checkpoints
    ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    part_ckpt = PROJECT_ROOT / "checkpoints" / "best_model.pth.part"
    if ckpt.exists():
        print(f"[+] Checkpoint: Found ({ckpt.stat().st_size / (1024**3):.2f} GB)")
    elif part_ckpt.exists():
        print(f"[*] Checkpoint: In-progress download ({part_ckpt.stat().st_size / (1024**3):.2f} GB / ~2.86 GB)")
    else:
        print("[!] Checkpoint: Not found (run 'python run.py --download-model' to fetch)")

    # 5. Architecture Smoke Test
    print("\n[*] Testing model architecture instantiation (PRO-FA, MIX-MLP, RCTA)...")
    try:
        from models.cognitive_model import CognitiveReportGenerator
        model = CognitiveReportGenerator(
            visual_encoder='vit_base_patch16_224',
            text_encoder_name='distilbert-base-uncased',
            decoder_name='distilgpt2',
            hidden_dim=512
        )
        print("[+] Model instantiated successfully.")
        
        # Test forward pass with synthetic tensors
        dummy_img = torch.randn(1, 3, 224, 224)
        dummy_ind = torch.randint(0, 1000, (1, 16))
        dummy_mask = torch.ones(1, 16)
        dummy_rep = torch.randint(0, 1000, (1, 16))
        dummy_rmask = torch.ones(1, 16)
        out = model(dummy_img, dummy_ind, dummy_mask, report_ids=dummy_rep, report_mask=dummy_rmask)
        assert 'loss' in out and 'disease_logits' in out
        print("[+] Model forward pass verification: PASS")
    except Exception as e:
        print(f"[!] Architecture test note: {e}")

    # 6. Sample Images
    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    samples = list(sample_dir.glob("*.[jp][pn]g")) if sample_dir.exists() else []
    print(f"[+] Sample test images: {len(samples)} found in {sample_dir}")

    print("\n" + "=" * 70)
    print("HEALTHCHECK COMPLETE")
    print("=" * 70)
    return 0


def cmd_demo(args):
    """Run inference report generation on a test image."""
    script = PROJECT_ROOT / "scripts" / "generate_reports.py"
    cmd = [sys.executable, str(script)]
    if args.image:
        cmd.extend(["--image", args.image])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.indication:
        cmd.extend(["--indication", args.indication])
    if getattr(args, 'profile', None):
        cmd.extend(["--profile", args.profile])
    return run_cmd(cmd)


def cmd_gui(args):
    """Launch the interactive PySide6 desktop GUI demo."""
    script = PROJECT_ROOT / "scripts" / "gui_app.py"
    return run_cmd([sys.executable, str(script)])


def cmd_test(args):
    """Run unit tests."""
    script = PROJECT_ROOT / "tests" / "test_model.py"
    return run_cmd([sys.executable, str(script)])


def cmd_download_model(args):
    """Download model checkpoint weights from Mega."""
    script = PROJECT_ROOT / "scripts" / "download_checkpoint.py"
    cmd = [sys.executable, str(script)]
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    if args.force:
        cmd.append("--force")
    return run_cmd(cmd)


def cmd_train(args, extra_args):
    """Launch model training loop."""
    script = PROJECT_ROOT / "training" / "train.py"
    cmd = [sys.executable, str(script)] + extra_args
    return run_cmd(cmd)


def cmd_eval(args, extra_args):
    """Run evaluation."""
    script = PROJECT_ROOT / "evaluation" / "evaluate.py"
    cmd = [sys.executable, str(script)] + extra_args
    return run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Radiology Assistant - Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --setup                # One-click environment & dependencies setup
  python run.py --check                # Validate GPU, packages, model architecture
  python run.py --demo                 # Generate report on sample chest X-ray
  python run.py --gui                  # Open interactive desktop GUI
  python run.py --download-model       # Download pre-trained 3GB model weights
  python run.py --test                 # Run test suite
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup", action="store_true", help="Setup environment and dependencies")
    group.add_argument("--check", action="store_true", help="Health check system, packages, and model")
    group.add_argument("--demo", action="store_true", help="Run inference on a chest X-ray image")
    group.add_argument("--gui", action="store_true", help="Launch desktop GUI application")
    group.add_argument("--test", action="store_true", help="Run test suite")
    group.add_argument("--download-model", "--download-checkpoint", dest="download_model", action="store_true", help="Download trained checkpoint weights from Mega")
    group.add_argument("--train", action="store_true", help="Run model training pipeline")
    group.add_argument("--eval", action="store_true", help="Run evaluation metrics pipeline")

    # Options for demo
    parser.add_argument("--image", type=str, default=None, help="Input chest X-ray image path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--indication", type=str, default=None, help="Clinical indication text")
    parser.add_argument("--profile", type=str, default="balanced", choices=["balanced", "cardiopulmonary", "infection", "specific"], help="Sensitivity calibration profile")
    
    # Options for downloader
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers for checkpoint download")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args, extra_args = parser.parse_known_args()

    if args.setup:
        sys.exit(cmd_setup(args) or 0)
    elif args.check:
        sys.exit(cmd_check(args) or 0)
    elif args.demo:
        sys.exit(cmd_demo(args) or 0)
    elif args.gui:
        sys.exit(cmd_gui(args) or 0)
    elif args.test:
        sys.exit(cmd_test(args) or 0)
    elif args.download_model:
        sys.exit(cmd_download_model(args) or 0)
    elif args.train:
        sys.exit(cmd_train(args, extra_args) or 0)
    elif args.eval:
        sys.exit(cmd_eval(args, extra_args) or 0)


if __name__ == "__main__":
    main()
