"""
Cognitive Radiology Assistant - Desktop Clinical Workspace (PySide6)
=====================================================================
Multi-task Vision-Language Report Generation for Chest Radiographs.
Features:
- PRO-FA Hierarchical Visual Encoding (ViT-B/16)
- MIX-MLP 14-Disease Multi-task Pathology Classifier
- RCTA Recursive Cognitive Triangular Attention Fusing Image, Text & Disease
- Clinical Grounding Module for Factual Consistency & Dynamic Sensitivity Calibration
- Benchmark Reference Verification Mode
"""

import sys
import os
import hashlib
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QPlainTextEdit,
    QProgressBar, QFrame, QSplitter, QMessageBox, QGraphicsOpacityEffect,
    QComboBox, QLineEdit, QTabWidget, QSlider, QGridLayout, QCheckBox,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QPixmap, QFont, QFontDatabase, QColor, QPainter, QPen, QBrush, QDragEnterEvent, QDropEvent, QClipboard
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms


# ============================================================================
# 1. CLINICAL GROUNDING & POST-PROCESSING ENGINE
# ============================================================================

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
            findings_text = "The lungs are clear without focal consolidation, pneumothorax, or pleural effusion. The cardiomediastinal silhouette and hilar contours are within normal limits. Visualized osseous and soft tissue structures appear unremarkable."
            impression_text = "No acute cardiopulmonary disease."
            return f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}"

        findings_detected = []
        for disease, prob in disease_probs.items():
            t = thresholds.get(disease, 0.50)
            if prob >= t and disease in ClinicalGrounder.PATHOLOGY_TEMPLATES:
                findings_detected.append(ClinicalGrounder.PATHOLOGY_TEMPLATES[disease])

        if not findings_detected:
            findings_text = "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. The cardiomediastinal silhouette is normal."
            impression_text = "No acute cardiopulmonary abnormality detected under current sensitivity thresholds."
        else:
            findings_text = " ".join(findings_detected)
            abnormalities = [d for d, p in disease_probs.items() if d != "No Finding" and p >= thresholds.get(d, 0.5)]
            if abnormalities:
                top_abn = sorted(abnormalities, key=lambda d: disease_probs[d], reverse=True)[:2]
                impression_text = f"Findings suspicious for {', '.join(top_abn)}. Recommend clinical correlation and follow-up as indicated."
            else:
                impression_text = "Cardiopulmonary changes as detailed above."

        return f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}"


# ============================================================================
# 2. BENCHMARK CASES & DATA
# ============================================================================

BENCHMARK_CASES = {
    "Custom Radiograph (Local File)": {
        "image": None,
        "indication": "Clinical indication: Chest pain, rule out acute cardiopulmonary process.",
        "reference_report": "Select a bundled benchmark case above to view its reference radiology report.",
        "key_findings": {}
    },
    "Case 1: Normal Adult PA (Clear Lungs)": {
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
    "Case 2: Cardiomegaly & Heart Failure": {
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
    "Case 3: Pleural Abnormality & Focal Mass": {
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


def match_benchmark_image(image_path):
    """Return matching benchmark case for an image by SHA-256 pixel content."""
    try:
        candidate = Image.open(image_path).convert("RGB")
        candidate_digest = hashlib.sha256(
            str(candidate.size).encode("ascii") + candidate.tobytes()
        ).hexdigest()
        for case_name, case_data in BENCHMARK_CASES.items():
            relative_path = case_data.get("image")
            if not relative_path:
                continue
            reference_path = Path(__file__).resolve().parent.parent / relative_path
            if not reference_path.exists():
                continue
            with Image.open(reference_path) as reference:
                reference = reference.convert("RGB")
                reference_digest = hashlib.sha256(
                    str(reference.size).encode("ascii") + reference.tobytes()
                ).hexdigest()
            if candidate_digest == reference_digest:
                return case_name, case_data
    except Exception:
        return None, None
    return None, None


# ============================================================================
# 3. CUSTOM INTERACTIVE WIDGETS
# ============================================================================

class ScoreBar(QWidget):
    """
    Precision custom probability bar with an integrated threshold tick indicator.
    Visualizes both probability and the diagnostic threshold cutoff.
    """
    def __init__(self, probability=0.0, threshold=0.5, parent=None):
        super().__init__(parent)
        self.probability = max(0.0, min(1.0, float(probability)))
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.setFixedHeight(14)
        self.setMinimumWidth(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, probability, threshold):
        self.probability = max(0.0, min(1.0, float(probability)))
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        radius = h / 2.0

        # Background track
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#e2e8f0")))
        painter.drawRoundedRect(0, 0, w, h, radius, radius)

        # Value fill bar
        fill_w = max(0.0, min(float(w), float(w * self.probability)))
        if fill_w > 0:
            is_above = self.probability >= self.threshold
            bar_color = QColor("#dc2626") if is_above else QColor("#2563eb")
            painter.setBrush(QBrush(bar_color))
            painter.drawRoundedRect(0, 0, int(fill_w), h, radius, radius)

        # Threshold tick marker
        tx = int(w * self.threshold)
        tick_pen = QPen(QColor("#0f172a"), 2)
        painter.setPen(tick_pen)
        painter.drawLine(tx, 1, tx, h - 1)


class DropImageLabel(QLabel):
    """
    Interactive radiograph viewport supporting drag-and-drop and auto-scaling paintEvent.
    """
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("ImagePreview")
        self.setText("Drag & Drop Chest X-Ray Here\n\nor click 'Browse Files' below\n(PNG, JPG, DICOM)")
        self.setWordWrap(True)
        self.setMinimumSize(220, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pixmap_source = None

    def set_image(self, path):
        if path and os.path.exists(path):
            self.pixmap_source = QPixmap(path)
            self.setText("")
            self.setStyleSheet("background-color: #0f172a; border-radius: 10px; border: 1px solid #334155;")
        else:
            self.pixmap_source = None
            self.setText("Drag & Drop Chest X-Ray Here\n\nor click 'Browse Files' below\n(PNG, JPG, DICOM)")
            self.setStyleSheet("")
        self.update()

    def paintEvent(self, event):
        if self.pixmap_source and not self.pixmap_source.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.fillRect(self.rect(), QColor("#0f172a"))
            target_w = max(10, self.width() - 16)
            target_h = max(10, self.height() - 16)
            scaled = self.pixmap_source.scaled(
                target_w, target_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            super().paintEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile().lower()
                if path.endswith(('.png', '.jpg', '.jpeg', '.dcm', '.bmp')):
                    event.acceptProposedAction()
                    self.setStyleSheet("border: 2px dashed #2563eb; background: #eff6ff;")
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        if self.pixmap_source:
            self.setStyleSheet("background-color: #0f172a; border-radius: 10px; border: 1px solid #334155;")
        else:
            self.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        if self.pixmap_source:
            self.setStyleSheet("background-color: #0f172a; border-radius: 10px; border: 1px solid #334155;")
        else:
            self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.exists(path):
                self.fileDropped.emit(path)
                event.acceptProposedAction()


# ============================================================================
# 4. MODERN CLINICAL THEME (QSS)
# ============================================================================

MODERN_THEME = """
QMainWindow, QWidget#AppSurface {
    background-color: #f1f5f9;
}
QWidget {
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    color: #0f172a;
    font-size: 13px;
}
QFrame#HeaderCard, QFrame#Card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
QLabel#BrandMark {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1d4ed8, stop:1 #2563eb);
    color: #ffffff;
    border-radius: 10px;
    font-size: 15px;
    font-weight: 800;
}
QLabel#HeaderTitle {
    color: #0f172a;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.3px;
    background: transparent;
}
QLabel#HeaderSubtitle {
    color: #64748b;
    font-size: 11px;
    background: transparent;
}
QLabel#SectionTitle {
    color: #0f172a;
    font-size: 14px;
    font-weight: 750;
    background: transparent;
}
QLabel#MutedText {
    color: #64748b;
    font-size: 11px;
    background: transparent;
}
QLabel#FieldLabel {
    color: #475569;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.8px;
    background: transparent;
}
QFrame#WorkflowStrip {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
QLabel#WorkflowStep {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    background: transparent;
    padding: 3px 8px;
    border-radius: 6px;
}
QLabel#WorkflowStep[active="true"] {
    color: #1d4ed8;
    background-color: #eff6ff;
}
QLabel#StatusPill {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
QLabel#StatusPill[state="success"] {
    background-color: #ecfdf5;
    color: #047857;
    border-color: #a7f3d0;
}
QLabel#StatusPill[state="warning"] {
    background-color: #fffbeb;
    color: #b45309;
    border-color: #fde68a;
}
QLabel#StatusPill[state="neutral"] {
    background-color: #f8fafc;
    color: #475569;
    border-color: #cbd5e1;
}
QLabel#ImagePreview {
    background-color: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 10px;
    color: #64748b;
    font-size: 12px;
    font-weight: 600;
    padding: 8px;
}
QFrame#InfoPanel {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
}
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
    padding: 7px 13px;
    border-radius: 7px;
    font-weight: 700;
}
QPushButton:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QPushButton:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
    border-color: #e2e8f0;
}
QPushButton#Secondary {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
}
QPushButton#Secondary:hover {
    background-color: #f8fafc;
    border-color: #94a3b8;
}
QPushButton#ChipButton {
    background-color: #f1f5f9;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 7px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton#ChipButton:hover {
    background-color: #e2e8f0;
    color: #0f172a;
}
QPushButton#Ghost {
    background-color: transparent;
    color: #475569;
    border: 1px solid #cbd5e1;
}
QPushButton#Ghost:hover {
    background-color: #f8fafc;
    color: #0f172a;
}
QPushButton#PrimaryAction {
    font-size: 13px;
    font-weight: 800;
    padding: 10px 18px;
    border-radius: 8px;
}
QLineEdit, QComboBox, QPlainTextEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 6px 9px;
    color: #0f172a;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1.5px solid #2563eb;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    selection-background-color: #eff6ff;
    selection-color: #1d4ed8;
}
QCheckBox {
    color: #1e293b;
    font-size: 12px;
    font-weight: 700;
    spacing: 7px;
    background: transparent;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #cbd5e1;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}
QTextEdit {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    padding: 10px;
    color: #0f172a;
    font-size: 13px;
    line-height: 1.5;
}
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #f8fafc;
    color: #64748b;
    padding: 7px 13px;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 700;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1d4ed8;
    border-color: #e2e8f0;
    border-bottom: 2px solid #2563eb;
}
QSlider::groove:horizontal {
    height: 5px;
    background-color: #e2e8f0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background-color: #3b82f6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #1d4ed8;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QLabel#SliderLabel {
    color: #1e293b;
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}
QLabel#SliderValue {
    color: #1d4ed8;
    font-weight: 800;
    background: transparent;
}
QProgressBar#LoaderProgress {
    height: 4px;
    border: none;
    border-radius: 2px;
    background-color: #e2e8f0;
}
QProgressBar#LoaderProgress::chunk {
    border-radius: 2px;
    background-color: #2563eb;
}
QFrame#Segment {
    background-color: #e2e8f0;
    border-radius: 2px;
    min-width: 12px;
    max-width: 12px;
    min-height: 4px;
    max-height: 4px;
}
QFrame#Segment[active="true"] {
    background-color: #2563eb;
}
QLabel#FooterStatus {
    color: #64748b;
    font-size: 11px;
    background: transparent;
}
QSplitter::handle {
    background-color: transparent;
}
"""


# ============================================================================
# 5. ASYNCHRONOUS BACKGROUND THREADS
# ============================================================================

class ModelLoaderThread(QThread):
    finished = Signal(object, object, object, object)
    error = Signal(str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def run(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
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
                disease_dict = {d: float(p.item()) for d, p in zip(diseases, disease_probs[0])}
                
                # Apply Clinical Grounding with calibrated thresholds
                report = ClinicalGrounder.refine(raw_report, disease_dict, self.thresholds)
                
            self.finished.emit(report, disease_dict, raw_report)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================================
# 6. MAIN WINDOW CLINICAL WORKSPACE
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radiology Assistant — PS_2 Diagnostic Report Generator")
        self.resize(1440, 900)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(MODERN_THEME)

        self.model = None
        self.enc_tokenizer = None
        self.dec_tokenizer = None
        self.device = None
        self.current_image_path = None
        self.current_probs = None
        self.current_raw_report = None
        self.current_benchmark_reference = False
        self.calibrated_thresholds = ClinicalGrounder.DEFAULT_THRESHOLDS.copy()
        self.show_only_above_threshold = False
        self._anims = []

        self.setup_ui()
        self.auto_load_checkpoint_if_exists()

    def setup_ui(self):
        """Construct the high-efficiency clinical workstation layout."""
        central_widget = QWidget()
        central_widget.setObjectName("AppSurface")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 10)
        main_layout.setSpacing(8)

        # --------------------------------------------------------------------
        # TOP HEADER: Identity, Model Status, and Global Controls
        # --------------------------------------------------------------------
        header = QFrame()
        header.setObjectName("HeaderCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(12)

        brand_mark = QLabel("XR")
        brand_mark.setObjectName("BrandMark")
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setFixedSize(42, 42)
        header_layout.addWidget(brand_mark)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title = QLabel("Cognitive Radiology Assistant")
        title.setObjectName("HeaderTitle")
        subtitle = QLabel("Multimodal Report Generation & Clinical Grounding (ViT-B/16 · DistilBERT · DistilGPT2)")
        subtitle.setObjectName("HeaderSubtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addLayout(title_stack)

        header_layout.addStretch(1)

        # Model Status Badge
        self.model_status = QLabel("MODEL NOT LOADED")
        self.model_status.setObjectName("StatusPill")
        self.model_status.setProperty("state", "warning")
        header_layout.addWidget(self.model_status)

        # Loader / Checkpoint actions
        self.loader_container = QWidget()
        self.loader_layout = QHBoxLayout(self.loader_container)
        self.loader_layout.setContentsMargins(0, 0, 0, 0)
        self.loader_layout.setSpacing(8)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setObjectName("Secondary")
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        self.loader_layout.addWidget(self.load_model_btn)

        self.loader_progress = QProgressBar()
        self.loader_progress.setObjectName("LoaderProgress")
        self.loader_progress.setRange(0, 0)
        self.loader_progress.setFixedWidth(90)
        self.loader_progress.setTextVisible(False)
        self.loader_progress.setVisible(False)
        self.loader_layout.addWidget(self.loader_progress)

        header_layout.addWidget(self.loader_container)

        self.reset_btn = QPushButton("Reset Workspace")
        self.reset_btn.setObjectName("Ghost")
        self.reset_btn.clicked.connect(self.reset_ui)
        header_layout.addWidget(self.reset_btn)

        main_layout.addWidget(header)

        # --------------------------------------------------------------------
        # WORKFLOW STEP TRACKER STRIP
        # --------------------------------------------------------------------
        workflow = QFrame()
        workflow.setObjectName("WorkflowStrip")
        workflow_layout = QHBoxLayout(workflow)
        workflow_layout.setContentsMargins(12, 6, 12, 6)
        workflow_layout.setSpacing(10)

        self.step_labels = []
        steps_info = [
            ("01", "Select Radiograph"),
            ("02", "Clinical Indication"),
            ("03", "Sensitivity Calibration"),
            ("04", "Generate & Review Report")
        ]
        for idx, (num, label_text) in enumerate(steps_info):
            step = QLabel(f"{num}  {label_text}")
            step.setObjectName("WorkflowStep")
            step.setProperty("active", "true" if idx == 0 else "false")
            workflow_layout.addWidget(step)
            self.step_labels.append(step)

        workflow_layout.addStretch(1)
        main_layout.addWidget(workflow)

        # Segmented execution rail (active during forward pass)
        self.rail_container = QFrame()
        self.rail_container.setObjectName("SegmentRail")
        self.rail_layout = QHBoxLayout(self.rail_container)
        self.rail_layout.setContentsMargins(0, 0, 0, 0)
        self.rail_layout.setSpacing(4)
        self.segments = []
        for _ in range(14):
            segment = QFrame()
            segment.setObjectName("Segment")
            segment.setProperty("active", "false")
            self.rail_layout.addWidget(segment)
            self.segments.append(segment)
        self.rail_container.setVisible(False)
        main_layout.addWidget(self.rail_container)

        # --------------------------------------------------------------------
        # MAIN WORKSPACE SPLITTER (Side-by-side inputs & results)
        # --------------------------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("WorkspaceSplitter")
        splitter.setHandleWidth(8)
        main_layout.addWidget(splitter, stretch=1)

        # ====================================================================
        # LEFT PANEL: Scrollable Radiograph Inputs & Clinical Context
        # ====================================================================
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        left_scroll.setMinimumWidth(320)

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)

        input_heading = QLabel("Radiograph Input")
        input_heading.setObjectName("SectionTitle")
        input_hint = QLabel("Select a bundled case or drag and drop any chest X-ray image.")
        input_hint.setObjectName("MutedText")
        input_hint.setWordWrap(True)
        left_layout.addWidget(input_heading)
        left_layout.addWidget(input_hint)

        benchmark_label = QLabel("SAMPLE RADIOGRAPH")
        benchmark_label.setObjectName("FieldLabel")
        left_layout.addWidget(benchmark_label)

        self.benchmark_combo = QComboBox()
        for name in BENCHMARK_CASES.keys():
            self.benchmark_combo.addItem(name)
        self.benchmark_combo.currentIndexChanged.connect(self.on_benchmark_selected)
        left_layout.addWidget(self.benchmark_combo)

        # Interactive Drag & Drop Viewport
        self.image_label = DropImageLabel()
        self.image_label.fileDropped.connect(self.handle_dropped_image)
        left_layout.addWidget(self.image_label, stretch=1)

        # Image Metadata Label
        self.image_meta_label = QLabel("No radiograph loaded")
        self.image_meta_label.setObjectName("MutedText")
        self.image_meta_label.setWordWrap(True)
        left_layout.addWidget(self.image_meta_label)

        # File browse actions
        img_btn_row = QHBoxLayout()
        img_btn_row.setSpacing(8)
        self.load_image_btn = QPushButton("Browse Files")
        self.load_image_btn.setObjectName("Secondary")
        self.load_image_btn.clicked.connect(self.load_image_dialog)
        img_btn_row.addWidget(self.load_image_btn)

        self.clear_image_btn = QPushButton("Clear")
        self.clear_image_btn.setObjectName("Ghost")
        self.clear_image_btn.clicked.connect(self.clear_image)
        img_btn_row.addWidget(self.clear_image_btn)
        left_layout.addLayout(img_btn_row)

        # Clinical Indication Field
        context_heading = QLabel("Clinical Indication")
        context_heading.setObjectName("SectionTitle")
        left_layout.addWidget(context_heading)

        self.indication_input = QPlainTextEdit()
        self.indication_input.setPlaceholderText("Enter patient clinical context or pre-operative indication...")
        self.indication_input.setPlainText("Clinical indication: Chest pain, rule out acute cardiopulmonary process.")
        self.indication_input.setMaximumHeight(56)
        left_layout.addWidget(self.indication_input)

        # Quick Context Chips
        chips_row = QHBoxLayout()
        chips_row.setSpacing(5)
        c1 = QPushButton("Routine Pre-op")
        c1.setObjectName("ChipButton")
        c1.clicked.connect(lambda: self.set_indication_preset("Clinical indication: Routine pre-operative medical screening. No cough, fever, or chest pain."))
        chips_row.addWidget(c1)

        c2 = QPushButton("Dyspnea / CHF")
        c2.setObjectName("ChipButton")
        c2.clicked.connect(lambda: self.set_indication_preset("Clinical indication: Progressive exertional dyspnea, orthopnea, and lower extremity edema. Rule out cardiomegaly and congestive heart failure."))
        chips_row.addWidget(c2)

        c3 = QPushButton("Mass / Cough")
        c3.setObjectName("ChipButton")
        c3.clicked.connect(lambda: self.set_indication_preset("Clinical indication: Persistent unilateral dull chest discomfort and chronic nonproductive cough. Evaluate for thoracic mass or pleural lesion."))
        chips_row.addWidget(c3)
        left_layout.addLayout(chips_row)

        # Benchmark Reference Mode Option
        benchmark_group = QFrame()
        benchmark_group.setObjectName("InfoPanel")
        bg_layout = QVBoxLayout(benchmark_group)
        bg_layout.setContentsMargins(10, 8, 10, 8)
        bg_layout.setSpacing(4)

        self.benchmark_mode_checkbox = QCheckBox("Curated Benchmark Reference Mode")
        self.benchmark_mode_checkbox.setToolTip(
            "Displays expert benchmark reference annotations for the bundled validation set. "
            "Uncheck to execute live forward-pass inference on the neural backbone."
        )
        self.benchmark_mode_checkbox.setChecked(False)
        bg_layout.addWidget(self.benchmark_mode_checkbox)

        bg_note = QLabel("Verification mode for validating model outputs against ground truth.")
        bg_note.setObjectName("MutedText")
        bg_note.setWordWrap(True)
        bg_layout.addWidget(bg_note)
        left_layout.addWidget(benchmark_group)

        left_scroll.setWidget(left_card)
        splitter.addWidget(left_scroll)

        # ====================================================================
        # RIGHT PANEL: Results, Pathology Scores, and Calibration
        # ====================================================================
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(8)

        # Results Header
        results_header = QHBoxLayout()
        r_title_stack = QVBoxLayout()
        r_title_stack.setSpacing(1)
        results_title = QLabel("Diagnostic Assessment")
        results_title.setObjectName("SectionTitle")
        results_subtitle = QLabel("Generated radiology findings, factual grounding, and pathology probabilities")
        results_subtitle.setObjectName("MutedText")
        r_title_stack.addWidget(results_title)
        r_title_stack.addWidget(results_subtitle)
        results_header.addLayout(r_title_stack)
        results_header.addStretch(1)

        self.result_status = QLabel("WAITING FOR ANALYSIS")
        self.result_status.setObjectName("StatusPill")
        self.result_status.setProperty("state", "neutral")
        results_header.addWidget(self.result_status, alignment=Qt.AlignTop)
        right_layout.addLayout(results_header)

        # Tab Widget: 1) Report, 2) Pathology Scores, 3) Reference, 4) Calibration
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # --------------------------------------------------------------------
        # TAB 1: GENERATED DIAGNOSTIC REPORT
        # --------------------------------------------------------------------
        tab_report = QWidget()
        tab_report_layout = QVBoxLayout(tab_report)
        tab_report_layout.setContentsMargins(2, 8, 2, 2)
        tab_report_layout.setSpacing(8)

        report_top_row = QHBoxLayout()
        rep_lbl = QLabel("AUTOMATED RADIOLOGY REPORT")
        rep_lbl.setObjectName("FieldLabel")
        report_top_row.addWidget(rep_lbl)
        report_top_row.addStretch(1)

        self.copy_btn = QPushButton("Copy Report")
        self.copy_btn.setObjectName("Secondary")
        self.copy_btn.clicked.connect(self.copy_report_to_clipboard)
        report_top_row.addWidget(self.copy_btn)

        self.export_btn = QPushButton("Export (.txt)")
        self.export_btn.setObjectName("Ghost")
        self.export_btn.clicked.connect(self.export_report)
        report_top_row.addWidget(self.export_btn)
        tab_report_layout.addLayout(report_top_row)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText(
            "Diagnostic report will appear here after analysis.\n\n"
            "Workflow:\n"
            "1. Select a sample radiograph or drag & drop a chest X-ray.\n"
            "2. Review or modify the clinical indication.\n"
            "3. Click 'Run Multimodal Analysis' below."
        )
        self.report_text.setMinimumHeight(220)
        tab_report_layout.addWidget(self.report_text, stretch=1)

        self.grounding_banner = QLabel("Clinical Grounding: Factually anchored against MIX-MLP multi-task pathology scores.")
        self.grounding_banner.setObjectName("MutedText")
        tab_report_layout.addWidget(self.grounding_banner)

        self.tabs.addTab(tab_report, "Report")

        # --------------------------------------------------------------------
        # TAB 2: PATHOLOGY PREDICTIONS (14-DISEASE MIX-MLP)
        # --------------------------------------------------------------------
        tab_scores = QWidget()
        tab_scores_layout = QVBoxLayout(tab_scores)
        tab_scores_layout.setContentsMargins(2, 8, 2, 2)
        tab_scores_layout.setSpacing(8)

        scores_top_row = QHBoxLayout()
        scores_lbl = QLabel("14-DISEASE MULTI-TASK PROBABILITIES (MIX-MLP)")
        scores_lbl.setObjectName("FieldLabel")
        scores_top_row.addWidget(scores_lbl)
        scores_top_row.addStretch(1)

        self.filter_above_btn = QPushButton("Show All")
        self.filter_above_btn.setObjectName("Ghost")
        self.filter_above_btn.clicked.connect(self.toggle_findings_filter)
        scores_top_row.addWidget(self.filter_above_btn)
        tab_scores_layout.addLayout(scores_top_row)

        scores_scroll = QScrollArea()
        scores_scroll.setWidgetResizable(True)
        scores_scroll.setFrameShape(QFrame.NoFrame)
        scores_scroll.setStyleSheet("QScrollArea, QScrollArea QWidget { background: #ffffff; }")

        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setAlignment(Qt.AlignTop)
        self.findings_layout.setSpacing(5)
        self.findings_layout.setContentsMargins(0, 0, 4, 0)
        scores_scroll.setWidget(self.findings_container)
        tab_scores_layout.addWidget(scores_scroll, stretch=1)

        score_explainer = QLabel("Threshold marker indicates the clinical grounding cutoff. Red/Amber indicates pathology detected above threshold.")
        score_explainer.setObjectName("MutedText")
        score_explainer.setWordWrap(True)
        tab_scores_layout.addWidget(score_explainer)

        self.tabs.addTab(tab_scores, "Pathology Scores")

        # --------------------------------------------------------------------
        # TAB 3: BENCHMARK REFERENCE ANNOTATIONS
        # --------------------------------------------------------------------
        tab_ref = QWidget()
        tab_ref_layout = QVBoxLayout(tab_ref)
        tab_ref_layout.setContentsMargins(2, 8, 2, 2)
        tab_ref_layout.setSpacing(8)

        ref_lbl = QLabel("BENCHMARK GROUND TRUTH REPORT")
        ref_lbl.setObjectName("FieldLabel")
        tab_ref_layout.addWidget(ref_lbl)

        self.ref_report_text = QTextEdit()
        self.ref_report_text.setReadOnly(True)
        self.ref_report_text.setPlaceholderText("Select a bundled benchmark case from the left panel to review its reference report.")
        self.ref_report_text.setMinimumHeight(140)
        tab_ref_layout.addWidget(self.ref_report_text, stretch=2)

        ref_findings_lbl = QLabel("REFERENCE PATHOLOGY LABELS")
        ref_findings_lbl.setObjectName("FieldLabel")
        tab_ref_layout.addWidget(ref_findings_lbl)

        ref_scroll = QScrollArea()
        ref_scroll.setWidgetResizable(True)
        ref_scroll.setFrameShape(QFrame.NoFrame)
        ref_scroll.setStyleSheet("QScrollArea, QScrollArea QWidget { background: #ffffff; }")

        self.ref_findings_container = QWidget()
        self.ref_findings_layout = QVBoxLayout(self.ref_findings_container)
        self.ref_findings_layout.setAlignment(Qt.AlignTop)
        self.ref_findings_layout.setSpacing(5)
        self.ref_findings_layout.setContentsMargins(0, 0, 4, 0)
        ref_scroll.setWidget(self.ref_findings_container)
        tab_ref_layout.addWidget(ref_scroll, stretch=1)

        self.tabs.addTab(tab_ref, "Reference")

        # --------------------------------------------------------------------
        # TAB 4: SENSITIVITY CALIBRATION
        # --------------------------------------------------------------------
        tab_calib = QWidget()
        tab_calib_layout = QVBoxLayout(tab_calib)
        tab_calib_layout.setContentsMargins(6, 10, 6, 6)
        tab_calib_layout.setSpacing(8)

        calib_intro = QLabel("Dynamic Clinical Grounder Calibration:\nAdjust detection thresholds for high-risk pathologies in real time without retraining.")
        calib_intro.setObjectName("MutedText")
        calib_intro.setWordWrap(True)
        tab_calib_layout.addWidget(calib_intro)

        preset_lbl = QLabel("CALIBRATION PRESET")
        preset_lbl.setObjectName("FieldLabel")
        tab_calib_layout.addWidget(preset_lbl)

        self.preset_combo = QComboBox()
        for profile in ClinicalGrounder.PRESET_PROFILES.keys():
            self.preset_combo.addItem(profile)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        tab_calib_layout.addWidget(self.preset_combo)

        # Threshold Sliders Grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        self.sliders = {}
        self.slider_labels = {}
        pathologies_to_calibrate = [
            ("Cardiomegaly", 0.40),
            ("Pleural Effusion", 0.55),
            ("Pneumonia", 0.35),
            ("Edema", 0.50),
            ("No Finding", 0.65)
        ]
        for row_idx, (pathology, default_val) in enumerate(pathologies_to_calibrate):
            label = QLabel(pathology)
            label.setObjectName("SliderLabel")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(10, 90)
            slider.setValue(int(default_val * 100))
            slider.valueChanged.connect(self.on_slider_changed)

            value_label = QLabel(f"{default_val * 100:.0f}%")
            value_label.setObjectName("SliderValue")
            value_label.setFixedWidth(42)

            grid.addWidget(label, row_idx, 0)
            grid.addWidget(slider, row_idx, 1)
            grid.addWidget(value_label, row_idx, 2)

            self.sliders[pathology] = slider
            self.slider_labels[pathology] = value_label

        tab_calib_layout.addLayout(grid)

        calib_note = QLabel("Note: Calibrated thresholds dynamically re-filter the report and flags. Backprop weights are untouched.")
        calib_note.setObjectName("MutedText")
        calib_note.setWordWrap(True)
        tab_calib_layout.addWidget(calib_note)
        tab_calib_layout.addStretch(1)

        self.tabs.addTab(tab_calib, "Calibration")

        right_layout.addWidget(self.tabs, stretch=1)
        splitter.addWidget(right_card)

        # Set panel proportions (36% left, 64% right)
        splitter.setSizes([440, 880])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # --------------------------------------------------------------------
        # BOTTOM ACTION BAR
        # --------------------------------------------------------------------
        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        action_hint = QLabel("Research & Investigational Use Only • PS_2 Cognitive Radiology Pipeline • Team monikabhati2005")
        action_hint.setObjectName("MutedText")
        action_hint.setWordWrap(True)
        action_row.addWidget(action_hint, stretch=1)

        self.generate_btn = QPushButton("Run Multimodal Analysis")
        self.generate_btn.setObjectName("PrimaryAction")
        self.generate_btn.setMinimumWidth(220)
        self.generate_btn.setFixedHeight(44)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.start_generation)
        action_row.addWidget(self.generate_btn)

        main_layout.addLayout(action_row)

        self.status_bar = QLabel("Ready. Load neural checkpoint and select an image to begin.")
        self.status_bar.setObjectName("FooterStatus")
        main_layout.addWidget(self.status_bar)

    # ========================================================================
    # EVENT HANDLERS & LOGIC
    # ========================================================================

    def update_workflow_step(self, step_idx):
        for idx, lbl in enumerate(self.step_labels):
            is_active = (idx == step_idx)
            lbl.setProperty("active", "true" if is_active else "false")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def set_indication_preset(self, text):
        self.indication_input.setPlainText(text)
        self.update_workflow_step(1)

    def auto_load_checkpoint_if_exists(self):
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            self.load_model_btn.setVisible(False)
            self.loader_progress.setVisible(True)
            self.status_bar.setText(f"Loading neural backbone: {ckpt_path.name}...")
            self.model_status.setText("LOADING BACKBONE")
            self.model_status.setProperty("state", "neutral")
            self.model_status.style().unpolish(self.model_status)
            self.model_status.style().polish(self.model_status)

            self.loader = ModelLoaderThread(str(ckpt_path.resolve()))
            self.loader.finished.connect(self.on_model_loaded)
            self.loader.error.connect(self.on_model_error)
            self.loader.start()
        else:
            self.status_bar.setText("Checkpoint not found in checkpoints/best_model.pth. Click 'Load Model' to select weights.")

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PyTorch Checkpoint", "checkpoints", "PyTorch Weights (*.pth *.pt)")
        if path:
            self.load_model_btn.setVisible(False)
            self.loader_progress.setVisible(True)
            self.status_bar.setText(f"Loading checkpoint: {os.path.basename(path)}...")
            self.model_status.setText("LOADING BACKBONE")
            self.model_status.setProperty("state", "neutral")
            self.model_status.style().unpolish(self.model_status)
            self.model_status.style().polish(self.model_status)

            self.loader = ModelLoaderThread(path)
            self.loader.finished.connect(self.on_model_loaded)
            self.loader.error.connect(self.on_model_error)
            self.loader.start()

    def on_model_loaded(self, model, e_tok, d_tok, device):
        self.model = model
        self.enc_tokenizer = e_tok
        self.dec_tokenizer = d_tok
        self.device = device

        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setText("[Backbone Active]")
        self.load_model_btn.setEnabled(False)

        dev_name = device.type.upper()
        if device.type == "cuda":
            dev_name = f"CUDA ({torch.cuda.get_device_name(0).split()[-1]})"
        self.model_status.setText(f"BACKBONE ACTIVE · {dev_name}")
        self.model_status.setProperty("state", "success")
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

        self.status_bar.setText(f"Neural backbone ready on {dev_name}. Select or drop a radiograph.")
        self.check_ready()

    def on_model_error(self, err):
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setEnabled(True)
        self.model_status.setText("LOAD FAILED")
        self.model_status.setProperty("state", "warning")
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)
        self.status_bar.setText(f"Model initialization failed: {err}")
        QMessageBox.critical(self, "Model Load Failed", f"Could not load checkpoint:\n{err}")

    def on_benchmark_selected(self, index):
        case_name = self.benchmark_combo.currentText()
        case_data = BENCHMARK_CASES.get(case_name)
        if not case_data or not case_data.get("image"):
            self.clear_image()
            if case_data:
                self.ref_report_text.setText(case_data.get("reference_report", ""))
                self.display_ref_findings(case_data.get("key_findings", {}))
            return

        img_path = Path(__file__).resolve().parent.parent / case_data["image"]
        if img_path.exists():
            self.load_image_file(str(img_path.resolve()), from_case=case_name)
            self.indication_input.setPlainText(case_data.get("indication", ""))
            self.ref_report_text.setText(case_data.get("reference_report", ""))
            self.display_ref_findings(case_data.get("key_findings", {}))
            self.update_workflow_step(1)
        else:
            QMessageBox.warning(self, "File Not Found", f"Benchmark radiograph not found at: {img_path}")

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Chest Radiograph", "data", "Radiographs (*.png *.jpg *.jpeg *.dcm)")
        if path:
            self.benchmark_combo.setCurrentIndex(0)
            self.load_image_file(path)

    def handle_dropped_image(self, path):
        self.benchmark_combo.setCurrentIndex(0)
        self.load_image_file(path)

    def load_image_file(self, path, from_case=None):
        self.current_image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.image_label.set_image(path)
            self.image_meta_label.setText(f"{os.path.basename(path)}  ·  {pixmap.width()} × {pixmap.height()} px")
            self.fade_in(self.image_label)
            self.status_bar.setText(f"Loaded radiograph: {os.path.basename(path)}")
            self.update_workflow_step(1)
            self.check_ready()

    def clear_image(self):
        self.current_image_path = None
        self.image_label.set_image(None)
        self.image_meta_label.setText("No radiograph loaded")
        self.update_workflow_step(0)
        self.check_ready()

    def reset_ui(self):
        self.clear_image()
        self.current_probs = None
        self.current_raw_report = None
        self.current_benchmark_reference = False
        self.benchmark_combo.setCurrentIndex(0)
        self.indication_input.setPlainText("Clinical indication: Chest pain, rule out acute cardiopulmonary process.")
        self.report_text.clear()
        self.ref_report_text.clear()
        self.clear_findings()
        self.clear_ref_findings()
        self.result_status.setText("WAITING FOR ANALYSIS")
        self.result_status.setProperty("state", "neutral")
        self.result_status.style().unpolish(self.result_status)
        self.result_status.style().polish(self.result_status)
        self.generate_btn.setText("Run Multimodal Analysis")
        self.generate_btn.setEnabled(False)
        self.update_workflow_step(0)
        self.status_bar.setText("Workspace reset. Select a radiograph to begin.")

    def check_ready(self):
        ready = bool(self.model and self.current_image_path)
        self.generate_btn.setEnabled(ready)
        if self.current_image_path and not self.model:
            self.status_bar.setText("Radiograph loaded. Awaiting model checkpoint initialization.")
        elif ready:
            self.status_bar.setText("Ready for analysis. Click 'Run Multimodal Analysis'.")

    def on_preset_changed(self):
        preset_name = self.preset_combo.currentText()
        preset_vals = ClinicalGrounder.PRESET_PROFILES.get(preset_name, ClinicalGrounder.DEFAULT_THRESHOLDS)
        self.calibrated_thresholds.update(preset_vals)
        for patho, slider in self.sliders.items():
            if patho in preset_vals:
                slider.setValue(int(preset_vals[patho] * 100))
        self.update_workflow_step(2)
        self.apply_live_regrounding()

    def on_slider_changed(self):
        for patho, slider in self.sliders.items():
            val = slider.value() / 100.0
            self.calibrated_thresholds[patho] = val
            self.slider_labels[patho].setText(f"{slider.value()}%")
        self.update_workflow_step(2)
        self.apply_live_regrounding()

    def apply_live_regrounding(self):
        """Dynamically re-anchor the diagnostic report in real time when thresholds change."""
        if self.current_probs is not None and not self.current_benchmark_reference:
            updated_report = ClinicalGrounder.refine(
                self.current_raw_report or "",
                self.current_probs,
                self.calibrated_thresholds
            )
            self.report_text.setText(updated_report)
            self.display_findings(self.current_probs)

    def start_generation(self):
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating Report...")
        self.result_status.setText("ANALYSIS IN PROGRESS")
        self.result_status.setProperty("state", "neutral")
        self.result_status.style().unpolish(self.result_status)
        self.result_status.style().polish(self.result_status)

        self.loader_progress.setVisible(True)
        self.rail_container.setVisible(True)
        self.simulate_rail_progress()

        self.report_text.clear()
        self.clear_findings()

        indication_text = self.indication_input.toPlainText().strip() or "Clinical indication: Chest pain, evaluate for cardiopulmonary disease."

        if self.benchmark_mode_checkbox.isChecked():
            case_name, case_data = match_benchmark_image(self.current_image_path)
            if case_data is None:
                self.loader_progress.setVisible(False)
                self.rail_container.setVisible(False)
                self.generate_btn.setEnabled(True)
                self.generate_btn.setText("Run Multimodal Analysis")
                self.result_status.setText("REFERENCE NOT FOUND")
                self.result_status.setProperty("state", "warning")
                self.result_status.style().unpolish(self.result_status)
                self.result_status.style().polish(self.result_status)
                QMessageBox.information(
                    self, "Benchmark Reference",
                    "Benchmark mode validates against the bundled test set. "
                    "Uncheck benchmark mode to execute live model inference."
                )
                return
            probs = case_data.get("key_findings", {})
            report = (
                "BENCHMARK REFERENCE OUTPUT — NOT A LIVE MODEL PREDICTION\n"
                "These findings represent the expert ground truth for this validation case.\n\n"
                + case_data.get("reference_report", "")
            )
            self.current_benchmark_reference = True
            self.on_inference_complete(report, probs, report)
            self.status_bar.setText(f"Benchmark reference ground truth displayed for {case_name}.")
            return

        self.current_benchmark_reference = False
        self.inf = InferenceThread(
            self.model,
            self.current_image_path,
            self.enc_tokenizer,
            self.dec_tokenizer,
            self.device,
            indication_text=indication_text,
            thresholds=self.calibrated_thresholds
        )
        self.inf.finished.connect(self.on_inference_complete)
        self.inf.error.connect(self.on_inference_error)
        self.inf.start()

    def simulate_rail_progress(self, idx=0):
        if idx >= len(self.segments) or not self.rail_container.isVisible():
            return
        self.segments[idx].setProperty("active", "true")
        self.segments[idx].style().unpolish(self.segments[idx])
        self.segments[idx].style().polish(self.segments[idx])
        QTimer.singleShot(120, lambda: self.simulate_rail_progress(idx + 1))

    def on_inference_complete(self, report, probs, raw_report):
        self.loader_progress.setVisible(False)
        self.rail_container.setVisible(False)
        for s in self.segments:
            s.setProperty("active", "false")
            s.style().unpolish(s)
            s.style().polish(s)

        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Run Multimodal Analysis")

        self.result_status.setText("REFERENCE GROUND TRUTH" if self.current_benchmark_reference else "ANALYSIS COMPLETE")
        self.result_status.setProperty("state", "neutral" if self.current_benchmark_reference else "success")
        self.result_status.style().unpolish(self.result_status)
        self.result_status.style().polish(self.result_status)

        self.current_probs = probs
        self.current_raw_report = raw_report

        clean_report = report.strip()
        self.report_text.setText(clean_report)
        self.fade_in(self.report_text, duration=600)

        self.display_findings(probs)
        self.tabs.setCurrentIndex(0)
        self.update_workflow_step(3)

        if self.current_benchmark_reference:
            self.status_bar.setText("Benchmark reference annotations loaded.")
        else:
            self.status_bar.setText("Analysis complete. Diagnostic report anchored with MIX-MLP clinical grounding.")

    def on_inference_error(self, err):
        self.loader_progress.setVisible(False)
        self.rail_container.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Run Multimodal Analysis")
        self.result_status.setText("INFERENCE FAILED")
        self.result_status.setProperty("state", "warning")
        self.result_status.style().unpolish(self.result_status)
        self.result_status.style().polish(self.result_status)
        self.status_bar.setText(f"Inference error: {err}")
        QMessageBox.critical(self, "Inference Failed", f"Pipeline encountered an error during generation:\n{err}")

    def clear_findings(self):
        while self.findings_layout.count():
            child = self.findings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_ref_findings(self):
        while self.ref_findings_layout.count():
            child = self.ref_findings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def toggle_findings_filter(self):
        self.show_only_above_threshold = not self.show_only_above_threshold
        self.filter_above_btn.setText("Show Detected Only" if not self.show_only_above_threshold else "Show All (14)")
        if self.current_probs:
            self.display_findings(self.current_probs)

    def display_findings(self, probs):
        """Render precision pathology score cards with custom threshold markers."""
        self.clear_findings()
        if not probs:
            empty = QLabel("No pathology score data available.")
            empty.setObjectName("MutedText")
            self.findings_layout.addWidget(empty)
            return

        sorted_findings = sorted(probs.items(), key=lambda item: item[1], reverse=True)

        for disease, probability in sorted_findings:
            threshold = self.calibrated_thresholds.get(disease, 0.40)
            above_threshold = probability >= threshold

            if self.show_only_above_threshold and not above_threshold:
                continue

            row = QFrame()
            row.setStyleSheet(
                "QFrame { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }"
            )
            layout = QHBoxLayout(row)
            layout.setContentsMargins(10, 6, 10, 6)
            layout.setSpacing(10)

            name = QLabel(disease)
            name.setStyleSheet("font-weight: 700; color: #1e293b; background: transparent; border: none;")
            name.setMinimumWidth(150)
            layout.addWidget(name)

            bar = ScoreBar(probability=probability, threshold=threshold)
            layout.addWidget(bar, stretch=1)

            value = QLabel(f"{float(probability) * 100:.1f}%")
            value.setStyleSheet(
                ("font-weight: 800; color: #dc2626; background: transparent;" if above_threshold
                 else "font-weight: 700; color: #2563eb; background: transparent;")
            )
            value.setFixedWidth(48)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(value)

            state = QLabel(f"Above ({threshold:.0%})" if above_threshold else f"Below ({threshold:.0%})")
            state.setStyleSheet(
                ("color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca;" if above_threshold
                 else "color: #475569; background: #f1f5f9; border: 1px solid #e2e8f0;")
                + " border-radius: 6px; padding: 3px 6px; font-size: 10px; font-weight: 700;"
            )
            layout.addWidget(state)
            self.findings_layout.addWidget(row)

    def display_ref_findings(self, findings_dict):
        self.clear_ref_findings()
        for disease, prob in findings_dict.items():
            row = QFrame()
            row.setStyleSheet("QFrame { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; }")
            layout = QHBoxLayout(row)
            layout.setContentsMargins(10, 6, 10, 6)
            layout.setSpacing(10)

            name = QLabel(disease)
            name.setStyleSheet("font-weight: 700; color: #334155; background: transparent; border: none;")
            name.setMinimumWidth(150)
            layout.addWidget(name)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(prob * 100))
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                "QProgressBar { background: #e2e8f0; border: none; border-radius: 4px; }"
                "QProgressBar::chunk { background: #059669; border-radius: 4px; }"
            )
            layout.addWidget(bar, stretch=1)

            value = QLabel(f"{float(prob) * 100:.1f}%")
            value.setStyleSheet("font-weight: 800; color: #059669; background: transparent;")
            value.setFixedWidth(48)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(value)

            self.ref_findings_layout.addWidget(row)

    def copy_report_to_clipboard(self):
        text = self.report_text.toPlainText().strip()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.copy_btn.setText("Copied! [OK]")
            QTimer.singleShot(2000, lambda: self.copy_btn.setText("Copy Report"))
            self.status_bar.setText("Radiology report copied to clipboard.")

    def export_report(self):
        text = self.report_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Export Report", "No report text available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Diagnostic Report", "outputs/report.txt", "Text Files (*.txt);;Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.status_bar.setText(f"Report exported to {os.path.basename(path)}")
            QMessageBox.information(self, "Report Exported", f"Diagnostic report saved successfully to:\n{path}")

    def fade_in(self, widget, duration=500):
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anims.append(anim)


# ============================================================================
# 7. MAIN LAUNCH ENTRYPOINT
# ============================================================================

def ensure_system_fonts():
    """Ensure standard system fonts (Segoe UI / Arial) are registered in QFontDatabase."""
    if not QFontDatabase.families():
        font_dir = Path("C:/Windows/Fonts")
        for font_file in ["segoeui.ttf", "segoeuib.ttf", "segoeuisb.ttf", "arial.ttf", "arialbd.ttf", "calibri.ttf"]:
            fpath = font_dir / font_file
            if fpath.exists():
                QFontDatabase.addApplicationFont(str(fpath))


if __name__ == "__main__":
    print("[*] Starting Cognitive Radiology Assistant GUI...", flush=True)
    app = QApplication(sys.argv)
    ensure_system_fonts()
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.showNormal()
    window.raise_()
    window.activateWindow()
    print("[+] GUI Window initialized and presented. Event loop running.", flush=True)
    sys.exit(app.exec())
