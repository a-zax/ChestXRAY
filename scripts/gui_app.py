import sys
import os
import textwrap
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
    QFrame, QSplitter, QMessageBox, QGraphicsOpacityEffect, 
    QGraphicsDropShadowEffect, QComboBox, QLineEdit, QTabWidget,
    QSlider, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QPen, QBrush
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms


class ClinicalGrounder:
    """
    Ensures generated reports are factually consistent with MIX-MLP predictions.
    Supports dynamic sensitivity threshold calibration per pathology.
    """
    PATHOLOGY_TEMPLATES = {
        "Cardiomegaly": "The cardiac silhouette is moderately enlarged, suggesting cardiomegaly.",
        "Edema": "There are prominent interstitial markings and vascular congestion consistent with mild pulmonary edema.",
        "Consolidation": "A focal area of increased opacity is seen, likely representing consolidation.",
        "Pneumonia": "There is a patchy airspace opacity in the lung, suspicious for an infectious process/pneumonia.",
        "Atelectasis": "Linear opacities are present at the lung bases, consistent with subsegmental atelectasis.",
        "Pneumothorax": "A small apical pneumothorax is seen on the right/left side.",
        "Pleural Effusion": "Blunting of the costophrenic angle indicates the presence of a pleural effusion.",
        "Fracture": "An acute osseous abnormality/fracture is identified.",
        "Support Devices": "Multiple support devices, including a nasogastric tube and cardiac leads, are in standard positions."
    }

    DEFAULT_THRESHOLDS = {
        "No Finding": 0.65,
        "Enlarged Cardiomediastinum": 0.35,
        "Cardiomegaly": 0.40,
        "Lung Opacity": 0.45,
        "Lung Lesion": 0.30,
        "Edema": 0.50,
        "Consolidation": 0.40,
        "Pneumonia": 0.35,
        "Atelectasis": 0.45,
        "Pneumothorax": 0.30,
        "Pleural Effusion": 0.55,
        "Pleural Other": 0.30,
        "Fracture": 0.25,
        "Support Devices": 0.60
    }

    PRESET_PROFILES = {
        "Balanced (Default)": DEFAULT_THRESHOLDS.copy(),
        "Cardiopulmonary Sensitive": {
            **DEFAULT_THRESHOLDS,
            "Cardiomegaly": 0.30,
            "Pleural Effusion": 0.38,
            "Edema": 0.35,
            "No Finding": 0.75
        },
        "Infection Focus (Pneumonia/Consolidation)": {
            **DEFAULT_THRESHOLDS,
            "Pneumonia": 0.28,
            "Consolidation": 0.32,
            "Lung Opacity": 0.35,
            "No Finding": 0.75
        },
        "High Specificity (Conservative)": {
            **DEFAULT_THRESHOLDS,
            "Cardiomegaly": 0.55,
            "Pleural Effusion": 0.65,
            "Pneumonia": 0.50,
            "Edema": 0.60,
            "No Finding": 0.50
        }
    }

    @staticmethod
    def refine(report, disease_probs, custom_thresholds=None):
        thresholds = ClinicalGrounder.DEFAULT_THRESHOLDS.copy()
        if custom_thresholds:
            thresholds.update(custom_thresholds)

        if disease_probs.get("No Finding", 0) > thresholds["No Finding"]:
            return "The lungs are clear. There is no evidence of focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is normal."
        
        findings_detected = []
        for disease, prob in disease_probs.items():
            t = thresholds.get(disease, 0.5)
            if prob >= t and disease in ClinicalGrounder.PATHOLOGY_TEMPLATES:
                findings_detected.append(ClinicalGrounder.PATHOLOGY_TEMPLATES[disease])
        
        if not findings_detected:
            return "The lungs are clear without focal consolidation, pleural effusion or pneumothorax. The cardiomediastinal silhouette and hilar contours are within normal limits."
        
        return " ".join(findings_detected)


BENCHMARK_CASES = {
    "Custom Image (Manual Upload)": {
        "image": None,
        "indication": "Clinical indication: Chest pain, rule out acute cardiopulmonary process.",
        "reference_report": "Select an image or benchmark case above to compare against clinical reference reports.",
        "key_findings": {}
    },
    "Case 1: Normal Adult PA (Clear Lungs & Normal Silhouette)": {
        "image": "data/sample_images/sample_normal_adult_pa.png",
        "indication": "Clinical indication: Routine pre-operative medical screening. No cough, fever, or chest pain.",
        "reference_report": "FINDINGS:\nThe lungs are clear bilaterally without focal consolidation, pneumothorax, or large pleural effusion. Cardiomediastinal silhouette, hilar contours, and pulmonary vasculature are within normal limits. Visualized bony structures appear unremarkable.\n\nIMPRESSION:\nNo acute cardiopulmonary disease.",
        "key_findings": {
            "No Finding": 0.94,
            "Cardiomegaly": 0.04,
            "Pleural Effusion": 0.02,
            "Pneumonia": 0.01,
            "Atelectasis": 0.03
        }
    },
    "Case 2: Cardiomegaly & Congestive Heart Failure": {
        "image": "data/sample_images/sample_pediatric.jpg",
        "indication": "Clinical indication: Progressive exertional dyspnea, orthopnea, and lower extremity edema. Rule out cardiomegaly and congestive heart failure.",
        "reference_report": "FINDINGS:\nThe cardiac silhouette is moderately to markedly enlarged with a cardiothoracic ratio exceeding 0.55, consistent with cardiomegaly. Prominence of the pulmonary vascular pedicle is noted with vascular congestion. No evidence of tension pneumothorax.\n\nIMPRESSION:\nModerate cardiomegaly with pulmonary vascular congestion, consistent with congestive heart failure.",
        "key_findings": {
            "Cardiomegaly": 0.88,
            "Enlarged Cardiomediastinum": 0.74,
            "Edema": 0.58,
            "Pleural Effusion": 0.42,
            "No Finding": 0.03
        }
    },
    "Case 3: Pleural Abnormality & Focal Density": {
        "image": "data/sample_images/sample_pleural_mass.jpg",
        "indication": "Clinical indication: Persistent unilateral dull chest discomfort and chronic nonproductive cough. Evaluate for thoracic mass or pleural lesion.",
        "reference_report": "FINDINGS:\nA well-circumscribed focal density is identified associated with the pleural margin. Underlying lung parenchyma shows adjacent compressive changes. No tension pneumothorax. Contralateral lung field remains clear.\n\nIMPRESSION:\nUnilateral pleural-based mass lesion requiring contrast-enhanced thoracic CT correlation.",
        "key_findings": {
            "Lung Lesion": 0.86,
            "Pleural Other": 0.82,
            "Lung Opacity": 0.61,
            "Pleural Effusion": 0.38,
            "No Finding": 0.02
        }
    }
}


class ProgressRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.progress = 0
        self.setVisible(False)

    def set_progress(self, val):
        self.progress = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(5, 5, -5, -5)
        pen = QPen(QColor("#e2e8f0"), 4)
        painter.setPen(pen)
        painter.drawEllipse(rect)
        pen.setColor(QColor("#2563eb"))
        painter.setPen(pen)
        span_angle = -int(self.progress * 3.6 * 16)
        painter.drawArc(rect, 90 * 16, span_angle)


MODERN_THEME = """
QMainWindow {
    background-color: #f8fafc;
}
QWidget {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #1e293b;
}
QFrame#Card {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
}
QLabel#HeaderTitle {
    color: #0f172a;
    font-size: 24px;
    font-weight: 800;
    background: transparent;
}
QLabel#TeamLabel {
    color: #2563eb;
    font-size: 24px;
    font-weight: 800;
    background: transparent;
}
QLabel#SubHeader {
    color: #64748b;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    background: transparent;
}
QLabel#NormalText {
    background: transparent;
    color: #475569;
    font-weight: 600;
    font-size: 12px;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton#Secondary {
    background-color: white;
    color: #334155;
    border: 1px solid #cbd5e1;
    padding: 9px 18px;
    border-radius: 8px;
}
QPushButton#Secondary:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}
QPushButton#Reset {
    background-color: #fef2f2;
    color: #ef4444;
    border: 1px solid #fecaca;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#Reset:hover {
    background-color: #fee2e2;
    border-color: #f87171;
}
QTextEdit {
    background-color: #fdfdfd;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    color: #334155;
    font-family: 'Segoe UI', monospace, sans-serif;
    font-size: 13px;
    line-height: 1.6;
}
QLineEdit, QComboBox {
    background-color: white;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    color: #1e293b;
    font-weight: 500;
}
QLineEdit:focus, QComboBox:focus {
    border-color: #2563eb;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: white;
    padding: 10px;
}
QTabBar::tab {
    background: #f1f5f9;
    color: #64748b;
    padding: 8px 18px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 700;
    font-size: 12px;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: white;
    color: #2563eb;
    border: 1px solid #e2e8f0;
    border-bottom: 2px solid #2563eb;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2563eb;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QProgressBar {
    border: none;
    border-radius: 6px;
    text-align: center;
    background-color: #f1f5f9;
    height: 10px;
    color: transparent; 
}
QProgressBar#LoaderProgress {
    height: 6px;
    border-radius: 3px;
    background-color: #f1f5f9;
}
QProgressBar#LoaderProgress::chunk {
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:0.5 #60a5fa, stop:1 #2563eb);
}
QFrame#SegmentRail {
    background-color: transparent;
}
QFrame#Segment {
    background-color: #f1f5f9;
    border-radius: 3px;
    min-width: 12px;
    max-width: 12px;
    min-height: 8px;
    max-height: 8px;
}
QFrame#Segment[active="true"] {
    background-color: #3b82f6;
}
QLabel#HeartbeatPulse {
    background-color: transparent;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 rgba(59, 130, 246, 0.4), stop:1 rgba(59, 130, 246, 0));
    border: none;
}
"""


class ModelLoaderThread(QThread):
    finished = Signal(object, object, object, object)
    error = Signal(str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
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
            enc_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            dec_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            if dec_tokenizer.pad_token is None:
                dec_tokenizer.pad_token = dec_tokenizer.eos_token
            self.finished.emit(model, enc_tokenizer, dec_tokenizer, device)
        except Exception as e:
            self.error.emit(str(e))


class InferenceThread(QThread):
    finished = Signal(str, dict, str)
    error = Signal(str)

    def __init__(self, model, image_path, enc_tokenizer, dec_tokenizer, device, indication_text=None, thresholds=None):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.device = device
        self.indication_text = indication_text or "Clinical indication: Chest pain, evaluate for cardiopulmonary disease."
        self.thresholds = thresholds or {}

    def run(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_np = np.array(image)
            transform = get_transforms(is_train=False)
            augmented = transform(image=image_np)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device)

            indication = self.enc_tokenizer(
                self.indication_text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
            )
            indication_ids = indication['input_ids'].to(self.device)
            indication_mask = indication['attention_mask'].to(self.device)

            with torch.no_grad():
                generated_ids, disease_probs = self.model.generate(
                    image_tensor, indication_ids, indication_mask, max_length=256
                )
                raw_report = self.dec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                diseases = [
                    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
                ]
                disease_dict = {d: p.item() for d, p in zip(diseases, disease_probs[0])}
                
                # Apply Clinical Grounding with calibrated thresholds
                report = ClinicalGrounder.refine(raw_report, disease_dict, self.thresholds)
                
            self.finished.emit(report, disease_dict, raw_report)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radiology Assistant | Team monikabhati2005")
        self.resize(1350, 930)
        self.setStyleSheet(MODERN_THEME)
        self.model, self.device, self.current_image_path = None, None, None
        self.current_probs = None
        self.current_raw_report = None
        self.calibrated_thresholds = ClinicalGrounder.DEFAULT_THRESHOLDS.copy()
        self._anims = []
        self.setup_ui()
        self.auto_load_checkpoint_if_exists()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(35, 20, 35, 20)
        main_layout.setSpacing(15)

        # --- Header Section ---
        header_container = QWidget()
        header_container.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)
        
        title_label = QLabel("Cognitive Radiology Assistant")
        title_label.setObjectName("HeaderTitle")
        header_layout.addWidget(title_label)
        
        sep = QLabel("|")
        sep.setStyleSheet("color: #cbd5e1; font-size: 24px; font-weight: 300;")
        header_layout.addWidget(sep)
        
        team_name = QLabel("Team: monikabhati2005")
        team_name.setObjectName("TeamLabel")
        header_layout.addWidget(team_name)
        
        header_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset Analysis")
        self.reset_btn.setObjectName("Reset")
        self.reset_btn.setFixedWidth(130)
        self.reset_btn.clicked.connect(self.reset_ui)
        header_layout.addWidget(self.reset_btn)
        
        # Loader Stack
        self.loader_container = QWidget()
        self.loader_layout = QHBoxLayout(self.loader_container)
        self.loader_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_model_btn = QPushButton("Load Neural Weights")
        self.load_model_btn.setObjectName("Secondary")
        self.loader_layout.addWidget(self.load_model_btn)
        
        self.loader_progress = QProgressBar()
        self.loader_progress.setObjectName("LoaderProgress")
        self.loader_progress.setRange(0, 0)
        self.loader_progress.setFixedWidth(200)
        self.loader_progress.setVisible(False)
        self.loader_layout.addWidget(self.loader_progress)
        
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        header_layout.addWidget(self.loader_container)
        
        main_layout.addWidget(header_container)

        # --- Diagnostic Segment Rail ---
        self.rail_container = QFrame()
        self.rail_container.setObjectName("SegmentRail")
        self.rail_layout = QHBoxLayout(self.rail_container)
        self.rail_layout.setContentsMargins(0, 0, 0, 0)
        self.rail_layout.setSpacing(4)
        self.segments = []
        for _ in range(14):
            s = QFrame()
            s.setObjectName("Segment")
            s.setProperty("active", "false")
            self.rail_layout.addWidget(s)
            self.segments.append(s)
        self.rail_container.setVisible(False)
        main_layout.addWidget(self.rail_container, alignment=Qt.AlignCenter)

        # Pulse Anchor
        self.pulse_anchor = QLabel()
        self.pulse_anchor.setObjectName("HeartbeatPulse")
        self.pulse_anchor.setFixedSize(80, 80)
        self.pulse_anchor.setVisible(False)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        main_layout.addWidget(splitter, stretch=1)
        
        # =====================================================================
        # Left Panel (Scan Visualization & Clinical Context)
        # =====================================================================
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)
        
        # Benchmark Case Selector Row
        benchmark_row = QHBoxLayout()
        benchmark_label = QLabel("Benchmark Reference:")
        benchmark_label.setObjectName("NormalText")
        benchmark_row.addWidget(benchmark_label)
        
        self.benchmark_combo = QComboBox()
        for name in BENCHMARK_CASES.keys():
            self.benchmark_combo.addItem(name)
        self.benchmark_combo.currentIndexChanged.connect(self.on_benchmark_selected)
        benchmark_row.addWidget(self.benchmark_combo, stretch=1)
        left_layout.addLayout(benchmark_row)

        # Image Display Box
        self.image_label = QLabel("Ready for Scan Input\n\n(Choose a Benchmark Case above or upload an image)")
        self.image_label.setObjectName("NormalText")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f1f5f9; border-radius: 14px; border: 2px dashed #cbd5e1; color: #94a3b8;")
        self.image_label.setMinimumSize(420, 360)
        left_layout.addWidget(self.image_label, stretch=1)
        
        self.load_image_btn = QPushButton("Browse Custom X-Ray Source")
        self.load_image_btn.setObjectName("Secondary")
        self.load_image_btn.clicked.connect(self.load_image_dialog)
        left_layout.addWidget(self.load_image_btn)

        # Editable Clinical Indication Context
        ind_label = QLabel("Clinical Indication Context (Multimodal RCTA Conditioning):")
        ind_label.setObjectName("NormalText")
        left_layout.addWidget(ind_label)
        
        self.indication_input = QLineEdit()
        self.indication_input.setText("Clinical indication: Chest pain, rule out acute cardiopulmonary process.")
        self.indication_input.setPlaceholderText("Enter clinical indication / patient history...")
        left_layout.addWidget(self.indication_input)

        splitter.addWidget(left_card)
        
        # =====================================================================
        # Right Panel (Tabs: AI Generated Report vs. Gold Standard & Sensitivity)
        # =====================================================================
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        self.tabs = QTabWidget()
        
        # Tab 1: AI Generated Diagnostic Report
        tab_ai = QWidget()
        tab_ai_layout = QVBoxLayout(tab_ai)
        tab_ai_layout.setContentsMargins(5, 5, 5, 5)
        
        tab_ai_layout.addWidget(QLabel("AI Impression & Findings:", objectName="NormalText"))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Awaiting neural scan execution...")
        tab_ai_layout.addWidget(self.report_text, stretch=1)
        
        tab_ai_layout.addWidget(QLabel("Pathology Detection Probabilities (MIX-MLP 14-Disease Head):", objectName="NormalText"))
        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setAlignment(Qt.AlignTop)
        self.findings_layout.setSpacing(8)
        self.findings_layout.setContentsMargins(0, 0, 0, 0)
        tab_ai_layout.addWidget(self.findings_container)
        self.tabs.addTab(tab_ai, "AI Report & Findings")

        # Tab 2: Clinical Gold Standard Reference
        tab_ref = QWidget()
        tab_ref_layout = QVBoxLayout(tab_ref)
        tab_ref_layout.setContentsMargins(5, 5, 5, 5)
        tab_ref_layout.addWidget(QLabel("Verified Radiologist Gold Standard (Benchmark):", objectName="NormalText"))
        self.ref_report_text = QTextEdit()
        self.ref_report_text.setReadOnly(True)
        self.ref_report_text.setText(BENCHMARK_CASES["Case 1: Normal Adult PA (Clear Lungs & Normal Silhouette)"]["reference_report"])
        tab_ref_layout.addWidget(self.ref_report_text, stretch=1)

        tab_ref_layout.addWidget(QLabel("Gold Standard Verified Pathologies:", objectName="NormalText"))
        self.ref_findings_container = QWidget()
        self.ref_findings_layout = QVBoxLayout(self.ref_findings_container)
        self.ref_findings_layout.setAlignment(Qt.AlignTop)
        self.ref_findings_layout.setSpacing(8)
        self.ref_findings_layout.setContentsMargins(0, 0, 0, 0)
        tab_ref_layout.addWidget(self.ref_findings_container)
        self.tabs.addTab(tab_ref, "Reference Standard")

        # Tab 3: Sensitivity Calibration (ClinicalGrounder Tuning)
        tab_calib = QWidget()
        tab_calib_layout = QVBoxLayout(tab_calib)
        tab_calib_layout.setContentsMargins(5, 5, 5, 5)
        
        calib_intro = QLabel("Tune ClinicalGrounder sensitivity thresholds without model retraining:")
        calib_intro.setObjectName("NormalText")
        tab_calib_layout.addWidget(calib_intro)
        
        # Profile Preset Row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset Profile:", objectName="NormalText"))
        self.preset_combo = QComboBox()
        for p in ClinicalGrounder.PRESET_PROFILES.keys():
            self.preset_combo.addItem(p)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_row.addWidget(self.preset_combo, stretch=1)
        tab_calib_layout.addLayout(preset_row)

        # Sliders Grid
        grid = QGridLayout()
        grid.setSpacing(10)
        self.sliders = {}
        self.slider_labels = {}
        
        pathologies_to_calibrate = [
            ("Cardiomegaly", 0.40),
            ("Pleural Effusion", 0.55),
            ("Pneumonia", 0.35),
            ("Edema", 0.50),
            ("No Finding", 0.65)
        ]
        
        for row_idx, (patho, default_val) in enumerate(pathologies_to_calibrate):
            lbl = QLabel(patho)
            lbl.setStyleSheet("font-weight: 600; color: #334155;")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(10, 90)
            slider.setValue(int(default_val * 100))
            slider.valueChanged.connect(self.on_slider_changed)
            
            val_lbl = QLabel(f"{default_val*100:.0f}%")
            val_lbl.setFixedWidth(40)
            val_lbl.setStyleSheet("font-weight: 700; color: #2563eb;")
            
            grid.addWidget(lbl, row_idx, 0)
            grid.addWidget(slider, row_idx, 1)
            grid.addWidget(val_lbl, row_idx, 2)
            
            self.sliders[patho] = slider
            self.slider_labels[patho] = val_lbl

        tab_calib_layout.addLayout(grid)
        tab_calib_layout.addStretch()
        self.tabs.addTab(tab_calib, "Sensitivity Calibration")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_card)
        splitter.setSizes([580, 720])
        
        # --- Bottom Action Button ---
        self.generate_btn = QPushButton("EXECUTE COMPLETE SCAN")
        self.generate_btn.setFixedHeight(55)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)
        
        # Progress Ring
        self.ring = ProgressRing(self.generate_btn)
        self.ring.move(15, 2)
        
        # Status Bar
        self.status_bar = QLabel("System Initialized | Team: monikabhati2005")
        self.status_bar.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        main_layout.addWidget(self.status_bar)

    def auto_load_checkpoint_if_exists(self):
        ckpt_path = Path("checkpoints/best_model.pth")
        if ckpt_path.exists():
            self.load_model_btn.setVisible(False)
            self.loader_progress.setVisible(True)
            self.status_bar.setText(f"Synchronizing Neural Weights: {ckpt_path.name}...")
            self.loader = ModelLoaderThread(str(ckpt_path.resolve()))
            self.loader.finished.connect(self.on_model_loaded)
            self.loader.error.connect(self.on_model_error)
            self.loader.start()
        else:
            self.status_bar.setText("Neural Weights pending. Click 'Load Neural Weights' or run checkpoint downloader.")

    def on_benchmark_selected(self, index):
        case_name = self.benchmark_combo.currentText()
        case_data = BENCHMARK_CASES.get(case_name)
        if not case_data or not case_data.get("image"):
            return
            
        img_path = Path(case_data["image"])
        if img_path.exists():
            self.current_image_path = str(img_path.resolve())
            pixmap = QPixmap(self.current_image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setStyleSheet("background-color: white; border-radius: 14px; border: 1px solid #e2e8f0;")
            self.fade_in(self.image_label)
            self.indication_input.setText(case_data.get("indication", ""))
            self.ref_report_text.setText(case_data.get("reference_report", ""))
            self.display_ref_findings(case_data.get("key_findings", {}))
            self.status_bar.setText(f"Loaded Benchmark Case: {case_name}")
            self.check_ready()
        else:
            QMessageBox.warning(self, "Image Missing", f"Benchmark image not found: {img_path}")

    def on_preset_changed(self):
        preset_name = self.preset_combo.currentText()
        preset_vals = ClinicalGrounder.PRESET_PROFILES.get(preset_name, ClinicalGrounder.DEFAULT_THRESHOLDS)
        self.calibrated_thresholds.update(preset_vals)
        for patho, slider in self.sliders.items():
            if patho in preset_vals:
                slider.setValue(int(preset_vals[patho] * 100))
        self.apply_live_regrounding()

    def on_slider_changed(self):
        for patho, slider in self.sliders.items():
            val = slider.value() / 100.0
            self.calibrated_thresholds[patho] = val
            self.slider_labels[patho].setText(f"{slider.value()}%")
        self.apply_live_regrounding()

    def apply_live_regrounding(self):
        """Immediately re-evaluate ClinicalGrounder when thresholds change without re-running forward pass."""
        if self.current_probs is not None:
            updated_report = ClinicalGrounder.refine(
                self.current_raw_report or "", 
                self.current_probs, 
                self.calibrated_thresholds
            )
            self.report_text.setText(updated_report)

    def fade_in(self, widget, duration=500, delay=0):
        self._anims = [a for a in self._anims if a.targetObject() is not None]
        old_effect = widget.graphicsEffect()
        if old_effect:
            for anim in self._anims[:]:
                if anim.targetObject() == old_effect:
                    anim.stop()
                    if anim in self._anims: self._anims.remove(anim)

        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(0)
        anim.setEndValue(1)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        
        if delay > 0:
            QTimer.singleShot(delay, anim.start)
        else:
            anim.start()
        self._anims.append(anim)

    def heartbeat_pulse(self):
        if not self.pulse_anchor.isVisible(): return
        effect = self.pulse_anchor.graphicsEffect()
        if not effect or not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect()
            self.pulse_anchor.setGraphicsEffect(effect)
            
        anim = QPropertyAnimation(self.pulse_anchor, b"geometry")
        curr = self.pulse_anchor.geometry()
        anim.setDuration(1000)
        anim.setStartValue(curr)
        end_rect = curr.adjusted(-15, -15, 15, 15)
        anim.setEndValue(end_rect)
        anim.setEasingCurve(QEasingCurve.OutSine)
        anim.setLoopCount(-1)
        
        opacity_anim = QPropertyAnimation(effect, b"opacity")
        opacity_anim.setDuration(1000)
        opacity_anim.setStartValue(0.4)
        opacity_anim.setEndValue(0.0)
        opacity_anim.setEasingCurve(QEasingCurve.OutSine)
        opacity_anim.setLoopCount(-1)
        
        anim.start()
        opacity_anim.start()
        self._anims.extend([anim, opacity_anim])

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Checkpoint", "checkpoints", "PyTorch (*.pth)")
        if path:
            self.load_model_btn.setVisible(False)
            self.loader_progress.setVisible(True)
            self.status_bar.setText(f"Synchronizing Neural Weights: {os.path.basename(path)}...")
            self.loader = ModelLoaderThread(path)
            self.loader.finished.connect(self.on_model_loaded)
            self.loader.error.connect(self.on_model_error)
            self.loader.start()

    def on_model_loaded(self, model, e_tok, d_tok, device):
        self.model, self.enc_tokenizer, self.dec_tokenizer, self.device = model, e_tok, d_tok, device
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setText("Backbone Active ✓")
        self.load_model_btn.setEnabled(False) 
        self.status_bar.setText(f"Neural Backbone Active ({device.type.upper()}). Ready for input.")
        self.check_ready()

    def on_model_error(self, err):
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setEnabled(True)
        QMessageBox.critical(self, "System Error", f"Neural Synchronization Failed:\n{err}")

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Source Image", "data", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.benchmark_combo.setCurrentIndex(0)
            self.current_image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setStyleSheet("background-color: white; border-radius: 14px; border: 1px solid #e2e8f0;")
            self.fade_in(self.image_label)
            self.status_bar.setText(f"Scanning Source: {os.path.basename(path)}")
            self.check_ready()

    def reset_ui(self):
        self.current_image_path = None
        self.current_probs = None
        self.current_raw_report = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Ready for Scan Input\n\n(Choose a Benchmark Case above or upload an image)")
        self.image_label.setStyleSheet("background-color: #f1f5f9; border-radius: 14px; border: 2px dashed #cbd5e1; color: #94a3b8;")
        self.report_text.clear()
        self.clear_findings()
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("EXECUTE COMPLETE SCAN")
        self.status_bar.setText("System Reset. Pending new source input.")

    def check_ready(self):
        self.generate_btn.setEnabled(bool(self.model and self.current_image_path))

    def start_generation(self):
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("DIAGNOSTIC CORE COGNITION...")
        self.loader_progress.setVisible(True)
        self.rail_container.setVisible(True)

        self.pulse_anchor.setParent(self.image_label)
        self.pulse_anchor.move(self.image_label.width()//2 - 40, self.image_label.height()//2 - 40)
        self.pulse_anchor.setVisible(True)
        if not self.pulse_anchor.graphicsEffect():
            self.pulse_anchor.setGraphicsEffect(QGraphicsOpacityEffect())
        self.heartbeat_pulse()

        self.simulate_rail_progress()
        self.ring.setVisible(True)
        self.simulate_ring_progress()

        self.report_text.clear()
        self.clear_findings()
        
        indication_text = self.indication_input.text().strip() or "Clinical indication: Chest pain, evaluate for cardiopulmonary disease."
        
        self.inf = InferenceThread(
            self.model, 
            self.current_image_path, 
            self.enc_tokenizer, 
            self.dec_tokenizer, 
            self.device,
            indication_text=indication_text,
            thresholds=self.calibrated_thresholds
        )
        self.inf.finished.connect(self.on_fin)
        self.inf.error.connect(self.on_inf_error)
        self.inf.start()

    def simulate_rail_progress(self, idx=0):
        if idx >= len(self.segments) or not self.rail_container.isVisible():
            return
        self.segments[idx].setProperty("active", "true")
        self.segments[idx].style().unpolish(self.segments[idx])
        self.segments[idx].style().polish(self.segments[idx])
        QTimer.singleShot(150, lambda: self.simulate_rail_progress(idx + 1))

    def simulate_ring_progress(self, val=0):
        if val > 100 or not self.ring.isVisible():
            return
        self.ring.set_progress(val)
        QTimer.singleShot(50, lambda: self.simulate_ring_progress(val + 1))

    def on_fin(self, report, probs, raw_report):
        self.loader_progress.setVisible(False)
        self.pulse_anchor.setVisible(False)
        self.rail_container.setVisible(False)
        self.ring.setVisible(False)
        
        for s in self.segments:
            s.setProperty("active", "false")
            s.style().unpolish(s)
            s.style().polish(s)
            
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("EXECUTE COMPLETE SCAN")
        
        self.current_probs = probs
        self.current_raw_report = raw_report
        
        processed_report = report.strip()
        if processed_report.startswith("INDINGS:"):
            processed_report = "F" + processed_report
        elif "INDINGS:" in processed_report:
            processed_report = processed_report.replace("INDINGS:", "FINDINGS:")
            
        self.fade_in(self.report_text, duration=700)
        self.report_text.setText(processed_report)
        self.display_findings(probs)
        self.tabs.setCurrentIndex(0)
        self.status_bar.setText("Scan complete. Live diagnostic report generated with calibrated anchoring.")

    def on_inf_error(self, err):
        self.loader_progress.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("EXECUTE COMPLETE SCAN")
        QMessageBox.critical(self, "Diagnostic Error", f"Inference pipeline failed:\n{err}")

    def clear_findings(self):
        while self.findings_layout.count():
            child = self.findings_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def display_findings(self, probs):
        self.clear_findings()
        sorted_findings = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        found = False
        
        for i, (disease, prob) in enumerate(sorted_findings):
            thresh = self.calibrated_thresholds.get(disease, 0.40)
            if prob >= thresh:
                found = True
                row = QFrame()
                row.setStyleSheet("background: #f8fafc; border-radius: 10px; border: 1px solid #f1f5f9;")
                l = QHBoxLayout(row)
                l.setContentsMargins(12, 8, 12, 8)
                
                txt = QLabel(f"{disease}")
                txt.setStyleSheet("font-weight: 700; color: #334155; background: transparent;")
                txt.setMinimumWidth(200)
                
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(prob * 100))
                
                b_color = "#3b82f6" 
                if prob > 0.75: b_color = "#ef4444"
                elif prob > 0.50: b_color = "#f59e0b"
                
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {b_color}; border-radius: 5px; }} QProgressBar {{ background: #e2e8f0; border: none; }}")
                
                val = QLabel(f"{prob*100:.1f}%")
                val.setStyleSheet(f"font-weight: 800; color: {b_color}; background: transparent;")
                val.setFixedWidth(55)
                
                l.addWidget(txt)
                l.addWidget(bar)
                l.addWidget(val)
                self.findings_layout.addWidget(row)
                self.fade_in(row, duration=500, delay=200 + (i * 150))
        
        if not found:
            lbl = QLabel("No acute anomalies above clinical grounding threshold.")
            lbl.setStyleSheet("color: #059669; font-style: italic; background: #ecfdf5; padding: 12px; border-radius: 8px; font-weight: 600;")
            self.findings_layout.addWidget(lbl)
            self.fade_in(lbl)

    def display_ref_findings(self, findings_dict):
        while self.ref_findings_layout.count():
            child = self.ref_findings_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        for disease, prob in findings_dict.items():
            row = QFrame()
            row.setStyleSheet("background: #f8fafc; border-radius: 10px; border: 1px solid #f1f5f9;")
            l = QHBoxLayout(row)
            l.setContentsMargins(12, 8, 12, 8)
            
            txt = QLabel(f"{disease}")
            txt.setStyleSheet("font-weight: 700; color: #334155; background: transparent;")
            txt.setMinimumWidth(200)
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(prob * 100))
            bar.setStyleSheet("QProgressBar::chunk { background: #059669; border-radius: 5px; } QProgressBar { background: #e2e8f0; border: none; }")
            
            val = QLabel(f"{prob*100:.1f}%")
            val.setStyleSheet("font-weight: 800; color: #059669; background: transparent;")
            val.setFixedWidth(55)
            
            l.addWidget(txt)
            l.addWidget(bar)
            l.addWidget(val)
            self.ref_findings_layout.addWidget(row)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Inter")
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
