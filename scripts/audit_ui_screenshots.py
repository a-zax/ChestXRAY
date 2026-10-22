"""
UI/UX Screenshot Audit & Validation Harness
===========================================
Programmatically exercises every state of `scripts/gui_app.py` in offscreen mode
and captures high-resolution screenshots to `outputs/ui_screenshots/` for visual audit.
"""

import sys
import os
from pathlib import Path

# Set Qt offscreen mode before importing Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import torch
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.gui_app import MainWindow, BENCHMARK_CASES, ClinicalGrounder

OUTPUT_DIR = ROOT_DIR / "outputs" / "ui_screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_events(app, count=15):
    for _ in range(count):
        app.processEvents()


def capture_screenshot(window, filename, width=1440, height=900):
    window.resize(width, height)
    process_events(QApplication.instance())
    pixmap = window.grab()
    out_path = OUTPUT_DIR / filename
    pixmap.save(str(out_path), "PNG")
    print(f"  [+] Captured: {filename} ({width}x{height}) -> {out_path.stat().st_size // 1024} KB", flush=True)
    return str(out_path)


def run_full_ui_audit():
    print("=" * 70, flush=True)
    print("STARTING CLINICAL UI/UX AUTOMATED SCREENSHOT AUDIT", flush=True)
    print("=" * 70, flush=True)

    app = QApplication(sys.argv)

    # Register Windows standard fonts in offscreen mode
    font_dir = Path("C:/Windows/Fonts")
    for font_file in ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "arial.ttf", "arialbd.ttf", "calibri.ttf"]:
        fpath = font_dir / font_file
        if fpath.exists():
            QFontDatabase.addApplicationFont(str(fpath))

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # 1. State 01: Initial Launch (Empty State, before model is loaded)
    print("[1/16] Testing 01_initial_launch_empty...", flush=True)
    # Temporarily instantiate window without auto-loading to capture empty state
    orig_auto_load = MainWindow.auto_load_checkpoint_if_exists
    MainWindow.auto_load_checkpoint_if_exists = lambda self: None
    window = MainWindow()
    window.show()
    process_events(app)
    capture_screenshot(window, "01_initial_launch_empty.png")

    # 2. State 02: Model Loaded & Active
    print("[2/16] Testing 02_model_ready_no_image...", flush=True)
    MainWindow.auto_load_checkpoint_if_exists = orig_auto_load
    ckpt_path = ROOT_DIR / "checkpoints" / "best_model.pth"
    if ckpt_path.exists():
        # Synchronously load model for audit test
        from models.cognitive_model import CognitiveReportGenerator
        from transformers import AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"      Loading checkpoint on {device}...", flush=True)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CognitiveReportGenerator(
            visual_encoder='vit_base_patch16_224',
            text_encoder_name='distilbert-base-uncased',
            decoder_name='distilgpt2',
            num_diseases=14,
            hidden_dim=512
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        enc_tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        dec_tok = AutoTokenizer.from_pretrained('distilgpt2')
        if dec_tok.pad_token is None:
            dec_tok.pad_token = dec_tok.eos_token
        window.on_model_loaded(model, enc_tok, dec_tok, device)
        process_events(app)
    else:
        # Mock model state for UI test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        window.on_model_loaded(None, None, None, device)
        process_events(app)
    capture_screenshot(window, "02_model_ready_no_image.png")

    # 3. State 03: Sample Case 1 Selected (Normal Adult PA)
    print("[3/16] Testing 03_sample_normal_loaded...", flush=True)
    window.benchmark_combo.setCurrentIndex(1)  # Case 1
    process_events(app)
    capture_screenshot(window, "03_sample_normal_loaded.png")

    # 4. State 04: Analysis In Progress (Simulate rail & busy indicators)
    print("[4/16] Testing 04_analysis_in_progress...", flush=True)
    window.generate_btn.setEnabled(False)
    window.generate_btn.setText("Generating Report...")
    window.result_status.setText("ANALYSIS IN PROGRESS")
    window.result_status.setProperty("state", "neutral")
    window.result_status.style().unpolish(window.result_status)
    window.result_status.style().polish(window.result_status)
    window.loader_progress.setVisible(True)
    window.rail_container.setVisible(True)
    for idx in range(7):
        window.segments[idx].setProperty("active", "true")
        window.segments[idx].style().unpolish(window.segments[idx])
        window.segments[idx].style().polish(window.segments[idx])
    process_events(app)
    capture_screenshot(window, "04_analysis_in_progress.png")

    # 5. State 05: Analysis Complete - Tab 1 (Diagnostic Report)
    print("[5/16] Testing 05_analysis_complete_report...", flush=True)
    probs_normal = {
        "No Finding": 0.94, "Enlarged Cardiomediastinum": 0.05, "Cardiomegaly": 0.04,
        "Lung Opacity": 0.03, "Lung Lesion": 0.01, "Edema": 0.02,
        "Consolidation": 0.02, "Pneumonia": 0.01, "Atelectasis": 0.03,
        "Pneumothorax": 0.01, "Pleural Effusion": 0.02, "Pleural Other": 0.01,
        "Fracture": 0.01, "Support Devices": 0.02
    }
    report_normal = ClinicalGrounder.refine(
        "The lungs are clear. Cardiomediastinal silhouette is normal.",
        probs_normal,
        window.calibrated_thresholds
    )
    window.on_inference_complete(report_normal, probs_normal, report_normal)
    window.tabs.setCurrentIndex(0)
    process_events(app)
    capture_screenshot(window, "05_analysis_complete_report.png")

    # 6. State 06: Tab 2 (14-Disease Pathology Scores with Threshold Markers)
    print("[6/16] Testing 06_tab_pathology_scores...", flush=True)
    window.tabs.setCurrentIndex(1)
    process_events(app)
    capture_screenshot(window, "06_tab_pathology_scores.png")

    # 7. State 07: Tab 3 (Benchmark Reference Ground Truth)
    print("[7/16] Testing 07_tab_reference_ground_truth...", flush=True)
    window.tabs.setCurrentIndex(2)
    process_events(app)
    capture_screenshot(window, "07_tab_reference_ground_truth.png")

    # 8. State 08: Tab 4 (Sensitivity Calibration Default)
    print("[8/16] Testing 08_tab_calibration_default...", flush=True)
    window.tabs.setCurrentIndex(3)
    process_events(app)
    capture_screenshot(window, "08_tab_calibration_default.png")

    # 9. State 09: Calibration with Conservative Preset & Dynamic Re-Grounding
    print("[9/16] Testing 09_calibration_conservative_preset...", flush=True)
    window.preset_combo.setCurrentText("High Specificity (Conservative)")
    process_events(app)
    capture_screenshot(window, "09_calibration_conservative_preset.png")

    # 10. State 10: Sample Case 2 (Cardiomegaly & Heart Failure)
    print("[10/16] Testing 10_sample_chf_cardiomegaly...", flush=True)
    window.benchmark_combo.setCurrentIndex(2)  # Case 2
    process_events(app)
    capture_screenshot(window, "10_sample_chf_cardiomegaly.png")

    # 11. State 11: Case 2 Analysis Results (Showing Detected Pathologies)
    print("[11/16] Testing 11_chf_inference_results...", flush=True)
    probs_chf = {
        "Cardiomegaly": 0.88, "Enlarged Cardiomediastinum": 0.74, "Edema": 0.58,
        "Pleural Effusion": 0.42, "Lung Opacity": 0.49, "Atelectasis": 0.35,
        "Consolidation": 0.22, "Pneumonia": 0.15, "Pneumothorax": 0.02,
        "Pleural Other": 0.08, "Fracture": 0.03, "Support Devices": 0.12,
        "Lung Lesion": 0.04, "No Finding": 0.03
    }
    report_chf = ClinicalGrounder.refine(
        "Moderate to severe cardiomegaly. Interstitial edema.",
        probs_chf,
        window.calibrated_thresholds
    )
    window.on_inference_complete(report_chf, probs_chf, report_chf)
    window.tabs.setCurrentIndex(1)  # View pathology scores with red alerts
    process_events(app)
    capture_screenshot(window, "11_chf_pathology_alerts.png")

    # Also capture report tab for Case 2
    window.tabs.setCurrentIndex(0)
    process_events(app)
    capture_screenshot(window, "11b_chf_diagnostic_report.png")

    # 12. State 12: Sample Case 3 (Pleural Abnormality & Focal Mass)
    print("[12/16] Testing 12_sample_pleural_mass...", flush=True)
    window.benchmark_combo.setCurrentIndex(3)  # Case 3
    probs_mass = {
        "Lung Lesion": 0.86, "Pleural Other": 0.82, "Lung Opacity": 0.61,
        "Pleural Effusion": 0.38, "Consolidation": 0.25, "Atelectasis": 0.30,
        "Cardiomegaly": 0.12, "Enlarged Cardiomediastinum": 0.18,
        "Edema": 0.08, "Pneumonia": 0.14, "Pneumothorax": 0.03,
        "Fracture": 0.02, "Support Devices": 0.05, "No Finding": 0.02
    }
    report_mass = ClinicalGrounder.refine(
        "Well-circumscribed pleural based mass density.",
        probs_mass,
        window.calibrated_thresholds
    )
    window.on_inference_complete(report_mass, probs_mass, report_mass)
    process_events(app)
    capture_screenshot(window, "12_sample_pleural_mass.png")

    # 13. State 13: Curated Benchmark Reference Mode
    print("[13/16] Testing 13_curated_benchmark_reference_mode...", flush=True)
    window.benchmark_mode_checkbox.setChecked(True)
    window.start_generation()
    process_events(app)
    capture_screenshot(window, "13_curated_benchmark_reference_mode.png")
    window.benchmark_mode_checkbox.setChecked(False)

    # 14. State 14: After Workspace Reset
    print("[14/16] Testing 14_workspace_after_reset...", flush=True)
    window.reset_ui()
    process_events(app)
    capture_screenshot(window, "14_workspace_after_reset.png")

    # 15. State 15: Responsive Small Screen (Minimum Supported: 980x680)
    print("[15/16] Testing 15_responsive_min_980x680...", flush=True)
    window.benchmark_combo.setCurrentIndex(1)
    window.on_inference_complete(report_normal, probs_normal, report_normal)
    process_events(app)
    capture_screenshot(window, "15_responsive_min_980x680.png", width=980, height=680)

    # 16. State 16: Standard Laptop Resolution (1366x768)
    print("[16/16] Testing 16_responsive_laptop_1366x768...", flush=True)
    capture_screenshot(window, "16_responsive_laptop_1366x768.png", width=1366, height=768)

    print("=" * 70, flush=True)
    print(f"[SUCCESS] All 16 UI audit screenshots generated in: {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_full_ui_audit()
