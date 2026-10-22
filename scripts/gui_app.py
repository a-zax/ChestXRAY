import sys
import os
import textwrap

class ClinicalGrounder:
    """
    Ensures generated reports are factually consistent with MIX-MLP predictions.
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

    @staticmethod
    def refine(report, disease_probs):
        thresholds = {
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
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
                               QFrame, QSplitter, QMessageBox, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QPixmap, QFont, QColor, QPainter, QPen, QBrush
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.dataset import get_transforms

from transformers import AutoTokenizer

class ProgressRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
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

# --- Professional Light Theme ---
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
    font-weight: 500;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 14px 28px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton#Secondary {
    background-color: white;
    color: #334155;
    border: 1px solid #e2e8f0;
}
QPushButton#Secondary:hover {
    background-color: #f8fafc;
    border-color: #cbd5e1;
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
    padding: 15px;
    color: #334155;
    font-family: 'Segoe UI', sans-serif;
    line-height: 1.8;
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
    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(self, model, image_path, enc_tokenizer, dec_tokenizer, device):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.device = device

    def run(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            image_np = np.array(image)
            transform = get_transforms(is_train=False)
            augmented = transform(image=image_np)
            image_tensor = augmented['image'].unsqueeze(0).to(self.device)

            indication_text = "Clinical indication: Deep analysis requested."
            indication = self.enc_tokenizer(
                indication_text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
            )
            indication_ids = indication['input_ids'].to(self.device)
            indication_mask = indication['attention_mask'].to(self.device)

            with torch.no_grad():
                generated_ids, disease_probs = self.model.generate(
                    image_tensor, indication_ids, indication_mask, max_length=256
                )
                report = self.dec_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                diseases = [
                    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
                ]
                disease_dict = {d: p.item() for d, p in zip(diseases, disease_probs[0])}
                
                # Apply Clinical Grounding
                report = ClinicalGrounder.refine(report, disease_dict)
                
            self.finished.emit(report, disease_dict)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radiology Assistant | Team monikabhati2005")
        self.resize(1300, 900)
        self.setStyleSheet(MODERN_THEME)
        self.model, self.device, self.current_image_path = None, None, None
        self._anims = []
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(25)

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
        self.reset_btn.setFixedWidth(130) # Fixed width as requested
        self.reset_btn.clicked.connect(self.reset_ui)
        header_layout.addWidget(self.reset_btn)
        
        # Loader Stack
        self.loader_container = QWidget()
        self.loader_layout = QHBoxLayout(self.loader_container)
        self.loader_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_model_btn = QPushButton("Load Neural Weights")
        self.load_model_btn.setObjectName("Secondary")
        self.loader_layout.addWidget(self.load_model_btn)
        
        # The Neural Ribbon (Indeterminate Progress)
        self.loader_progress = QProgressBar()
        self.loader_progress.setObjectName("LoaderProgress")
        self.loader_progress.setRange(0, 0)
        self.loader_progress.setFixedWidth(240)
        self.loader_progress.setVisible(False)
        self.loader_layout.addWidget(self.loader_progress)
        
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        header_layout.addWidget(self.loader_container)
        
        main_layout.addWidget(header_container)

        # --- Segmented Diagnostic Rail (Brainstorm Item 2A) ---
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

        # --- Scan Heartbeat Anchor ---
        self.pulse_anchor = QLabel()
        self.pulse_anchor.setObjectName("HeartbeatPulse")
        self.pulse_anchor.setFixedSize(80, 80)
        self.pulse_anchor.setVisible(False)
        # We will position this dynamically or use it as a subtle overlay
        # WorkArea (Splitter)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        main_layout.addWidget(splitter)
        
        # Left Panel (Scan Visualization)
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(25, 25, 25, 25)
        left_layout.setSpacing(20)
        
        viz_header = QLabel("Image Visualization")
        viz_header.setObjectName("SubHeader")
        left_layout.addWidget(viz_header)
        
        self.image_label = QLabel("Ready for Scan Input")
        self.image_label.setObjectName("NormalText")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f1f5f9; border-radius: 16px; border: 2px dashed #cbd5e1; color: #94a3b8;")
        self.image_label.setMinimumSize(450, 450)
        left_layout.addWidget(self.image_label)
        
        self.load_image_btn = QPushButton("Pick X-Ray Source")
        self.load_image_btn.setObjectName("Secondary")
        self.load_image_btn.clicked.connect(self.load_image_dialog)
        left_layout.addWidget(self.load_image_btn)
        
        splitter.addWidget(left_card)
        
        # Right Panel (Insights)
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(25, 25, 25, 25)
        right_layout.setSpacing(20)
        
        res_header = QLabel("Diagnostic Summary")
        res_header.setObjectName("SubHeader")
        right_layout.addWidget(res_header)
        
        right_layout.addWidget(QLabel("AI Impression:", objectName="NormalText"))
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Waiting for scan triggers...")
        right_layout.addWidget(self.report_text)
        
        right_layout.addWidget(QLabel("Anomaly Analysis Probabilities:", objectName="NormalText"))
        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setAlignment(Qt.AlignTop)
        self.findings_layout.setSpacing(12)
        self.findings_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.findings_container)
        
        splitter.addWidget(right_card)
        splitter.setSizes([600, 680])
        
        # --- Bottom Action ---
        self.generate_btn = QPushButton("EXECUTE COMPLETE SCAN")
        self.generate_btn.setFixedHeight(60)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)
        
        # Progress Ring Overlay
        self.ring = ProgressRing(self.generate_btn)
        self.ring.move(10, 0) # Center vertically in the button
        # Anchor for progress ring to better align
        self.ring.setFixedSize(50, 50)
        self.ring.move(15, 5)
        
        # Status Bar
        self.status_bar = QLabel("System Initialized | Team: monikabhati2005")
        self.status_bar.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        main_layout.addWidget(self.status_bar)

    def fade_in(self, widget, duration=600, delay=0):
        # 1. Clean up stale animations
        self._anims = [a for a in self._anims if a.targetObject() is not None]
        
        # 2. Stop any animation currently targeting this widget's existing effect
        old_effect = widget.graphicsEffect()
        if old_effect:
            for anim in self._anims[:]:
                if anim.targetObject() == old_effect:
                    anim.stop()
                    if anim in self._anims: self._anims.remove(anim)

        # 3. Apply new effect and animation
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
        
        # Ensure effect exists before animating
        effect = self.pulse_anchor.graphicsEffect()
        if not effect or not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect()
            self.pulse_anchor.setGraphicsEffect(effect)
            
        # Scale animation that keeps center
        anim = QPropertyAnimation(self.pulse_anchor, b"geometry")
        curr = self.pulse_anchor.geometry()
        
        anim.setDuration(1000)
        anim.setStartValue(curr)
        # Grow by 15px on all sides
        end_rect = curr.adjusted(-15, -15, 15, 15)
        anim.setEndValue(end_rect)
        anim.setEasingCurve(QEasingCurve.OutSine)
        anim.setLoopCount(-1)
        
        opacity_anim = QPropertyAnimation(effect, b"opacity")
        opacity_anim.setDuration(1000)
        opacity_anim.setStartValue(0.4)
        opacity_anim.setEndValue(0.0) # Fade out as it expands
        opacity_anim.setEasingCurve(QEasingCurve.OutSine)
        opacity_anim.setLoopCount(-1)
        
        anim.start()
        opacity_anim.start()
        self._anims.extend([anim, opacity_anim])

    def set_shadow_depth(self, widget, color="#e2e8f0", radius=10):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(radius)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(color))
        widget.setGraphicsEffect(shadow)
        return shadow

    def animate_shadow(self, shadow, start_radius, end_radius, duration=1000):
        anim = QPropertyAnimation(shadow, b"blurRadius")
        anim.setDuration(duration)
        anim.setStartValue(start_radius)
        anim.setEndValue(end_radius)
        anim.setEasingCurve(QEasingCurve.InOutQuad)
        anim.setLoopCount(-1)
        anim.start()
        self._anims.append(anim)

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Checkpoint", "", "PyTorch (*.pth)")
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
        self.status_bar.setText("Neural Backbone Active. Ready for input.")
        self.check_ready()

    def on_model_error(self, err):
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setEnabled(True)
        QMessageBox.critical(self, "System Error", f"Neural Synchronization Failed:\n{err}")

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Source Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.current_image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setStyleSheet("background-color: white; border-radius: 16px; border: 1px solid #e2e8f0;")
            self.fade_in(self.image_label)
            self.status_bar.setText(f"Scanning Source: {os.path.basename(path)}")
            self.check_ready()

    def reset_ui(self):
        self.current_image_path = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Ready for Scan Input")
        self.image_label.setStyleSheet("background-color: #f1f5f9; border-radius: 16px; border: 2px dashed #cbd5e1; color: #94a3b8;")
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
        
        # Shadow Motion (Brainstorm Item 5 - Shadow Motion)
        for card in [self.image_label.parent(), self.report_text.parent()]:
            if hasattr(card, "graphicsEffect"):
                 s = self.set_shadow_depth(card, color="#3b82f6", radius=20)
                 self.animate_shadow(s, 20, 40)

        # Position Heartbeat in center of image
        self.pulse_anchor.setParent(self.image_label)
        self.pulse_anchor.move(self.image_label.width()//2 - 40, self.image_label.height()//2 - 40)
        self.pulse_anchor.setVisible(True)
        if not self.pulse_anchor.graphicsEffect():
            self.pulse_anchor.setGraphicsEffect(QGraphicsOpacityEffect())
        self.heartbeat_pulse()

        # Step through segments simulation (Brainstorm Item 2A)
        self.simulate_rail_progress()
        
        # Circular Ring simulation (Brainstorm Item 2C)
        self.ring.setVisible(True)
        self.simulate_ring_progress()

        self.report_text.clear()
        self.clear_findings()
        self.inf = InferenceThread(self.model, self.current_image_path, self.enc_tokenizer, self.dec_tokenizer, self.device)
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

    def on_fin(self, report, probs):
        self.loader_progress.setVisible(False)
        self.pulse_anchor.setVisible(False)
        self.rail_container.setVisible(False)
        self.ring.setVisible(False)
        
        # Reset segments
        for s in self.segments:
            s.setProperty("active", "false")
            s.style().unpolish(s)
            s.style().polish(s)
            
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("EXECUTE COMPLETE SCAN")
        
        # Post-process common model artifacts (like 'INDINGS' for 'FINDINGS')
        processed_report = report.strip()
        if processed_report.startswith("INDINGS:"):
            processed_report = "F" + processed_report
        elif "INDINGS:" in processed_report:
            processed_report = processed_report.replace("INDINGS:", "FINDINGS:")
            
        # Staggered entrance for report
        self.fade_in(self.report_text, duration=800)
        self.report_text.setText(processed_report)
        
        # Staggered findings (triggered inside display_findings)
        self.display_findings(probs)
        self.status_bar.setText("Scan complete. Diagnostic insights staggered for review.")

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
            if prob > 0.5:
                found = True
                row = QFrame()
                row.setStyleSheet("background: #f8fafc; border-radius: 12px; border: 1px solid #f1f5f9;")
                l = QHBoxLayout(row)
                l.setContentsMargins(15, 12, 15, 12)
                
                txt = QLabel(f"{disease}")
                txt.setStyleSheet("font-weight: 700; color: #334155; background: transparent;")
                txt.setMinimumWidth(220) # Ensure label is not cut
                
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(int(prob * 100))
                
                # Professional dynamic coloring
                b_color = "#3b82f6" 
                if prob > 0.8: b_color = "#ef4444"
                elif prob > 0.65: b_color = "#f59e0b"
                
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {b_color}; border-radius: 5px; }} QProgressBar {{ background: #e2e8f0; border: none; }}")
                
                val = QLabel(f"{prob*100:.1f}%")
                val.setStyleSheet(f"font-weight: 800; color: {b_color}; background: transparent;")
                val.setFixedWidth(60)
                
                l.addWidget(txt)
                l.addWidget(bar)
                l.addWidget(val)
                
                self.findings_layout.addWidget(row)
                # Significant stagger for "Medical Narrative" reveal
                self.fade_in(row, duration=600, delay=500 + (i * 250))
        
        if not found:
            lbl = QLabel("No clinical anomalies identified within high confidence spectrum.")
            lbl.setStyleSheet("color: #059669; font-style: italic; background: #ecfdf5; padding: 15px; border-radius: 10px; font-weight: 600;")
            self.findings_layout.addWidget(lbl)
            self.fade_in(lbl)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Global Font tweak
    font = QFont("Inter")
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
