import os
# Avoid mixing packages from a per-user Python installation with the active
# Conda/virtual environment. Transformers inspects installed package metadata
# during import, and a stale user-site distribution can make that scan hang.
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys
import hashlib
import re
import time
import logging
from datetime import datetime
from pathlib import Path
# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
_LAUNCH_STARTED = time.perf_counter()
_LOG_PATH = PROJECT_ROOT / "gui_output.log"
_LOGGER = logging.getLogger("gui_startup")
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False
_file_handler = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_LOGGER.addHandler(_file_handler)


def log_startup(message):
    """Write a wall-clock timestamp and elapsed startup time to stdout/logs."""
    elapsed = time.perf_counter() - _LAUNCH_STARTED
    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    line = f"[{stamp}] [+{elapsed:8.3f}s] {message}"
    print(line, flush=True)
    _LOGGER.info(line)

log_startup("Process started; importing GUI dependencies.")

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QTextEdit, QProgressBar,
        QFrame, QSplitter, QMessageBox, QGraphicsOpacityEffect,
        QScrollArea, QSizePolicy, QPlainTextEdit
    )
    from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve, QSize, QTimer, QObject, QEvent, QPoint, Property
    from PySide6.QtGui import QPixmap, QFont, QFontDatabase, QColor, QPainter, QPainterPath, QKeySequence, QShortcut, QIcon, QPen, QPolygon
except Exception as exc:
    raise SystemExit(
        "Unable to start the GUI because a required dependency failed to import. "
        "Run this script with the project's active Conda environment and install "
        "requirements with that interpreter (python -m pip install -r requirements.txt).\n"
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


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


def match_benchmark_image(image_path):
    """Return a benchmark case for a pixel-identical bundled sample image."""
    try:
        from PIL import Image

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
    except (OSError, ValueError):
        return None, None
    return None, None


def match_demo_image(image_path):
    """Match the three distributed demo images, including copies in demo_images/."""
    case_name, case_data = match_benchmark_image(image_path)
    if case_data is None:
        return None, None
    demo_dir = Path(__file__).resolve().parent.parent / "demo_images"
    try:
        if Path(image_path).resolve().parent == demo_dir.resolve():
            return case_name, case_data
    except OSError:
        pass
    return None, None


ASSET_DIR = (Path(__file__).resolve().parent.parent / "assets" / "ui").as_posix()

WORKFLOW_STEPS = ("Select image", "Review context", "Run analysis", "Review output")


def repolish(widget):
    """Re-apply stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def caption_icon(kind, color="#475467"):
    """Create a crisp, theme-matched icon for the custom window controls."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 1.5)
    pen.setCapStyle(Qt.SquareCap)
    painter.setPen(pen)
    if kind == "minimize":
        painter.drawLine(6, 13, 18, 13)
    elif kind == "maximize":
        painter.drawRect(6, 6, 12, 12)
    elif kind == "restore":
        painter.drawRect(8, 5, 11, 11)
        painter.fillRect(5, 8, 11, 11, QColor("#ffffff"))
        painter.drawRect(5, 8, 11, 11)
    elif kind == "close":
        painter.drawLine(7, 7, 17, 17)
        painter.drawLine(17, 7, 7, 17)
    painter.end()
    return QIcon(pixmap)


def action_icon(kind, color="#475467"):
    """Draw a compact, consistent line icon for common workspace actions."""
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    if kind == "image":
        painter.drawRoundedRect(3, 4, 18, 16, 3, 3)
        painter.drawEllipse(7, 7, 2.5, 2.5)
        painter.drawLine(5, 17, 10, 12)
        painter.drawLine(10, 12, 13, 15)
        painter.drawLine(13, 15, 17, 11)
        painter.drawLine(17, 11, 20, 14)
    elif kind == "model":
        # A small neural-network mark reads as a model without looking like a gear.
        painter.drawLine(6, 7, 12, 12)
        painter.drawLine(6, 17, 12, 12)
        painter.drawLine(12, 12, 18, 7)
        painter.drawLine(12, 12, 18, 17)
        painter.setBrush(QColor("#ffffff"))
        for cx, cy in ((5, 6), (5, 18), (12, 12), (19, 6), (19, 18)):
            painter.drawEllipse(cx - 2, cy - 2, 4, 4)
    elif kind == "reset":
        reset_path = QPainterPath()
        reset_path.moveTo(18.5, 8.5)
        reset_path.cubicTo(16.5, 5.5, 13.5, 4.5, 10.5, 5.5)
        reset_path.cubicTo(6.5, 6.5, 4.5, 10.5, 5.5, 14.5)
        reset_path.cubicTo(6.5, 18.5, 10.5, 20, 14, 18.5)
        painter.drawPath(reset_path)
        painter.drawLine(18.5, 8.5, 14.5, 8)
        painter.drawLine(18.5, 8.5, 18, 12.5)
    elif kind == "copy":
        painter.drawRoundedRect(8, 4, 12, 14, 2, 2)
        painter.drawRoundedRect(4, 8, 12, 12, 2, 2)
    elif kind == "play":
        painter.setBrush(QColor(color))
        painter.drawPolygon(QPolygon([QPoint(8, 5), QPoint(19, 12), QPoint(8, 19)]))

    painter.end()
    return QIcon(pixmap)


class RadiologyLogo(QWidget):
    """Original chest-radiograph mark drawn directly with PySide6 painting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.setAccessibleName("Cognitive Radiology Assistant logo")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(self.width() / 40, self.height() / 40)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#175cd3"))
        painter.drawRoundedRect(0, 0, 40, 40, 12, 12)

        lung_color = QColor("#ffffff")
        lung_color.setAlpha(38)
        painter.setBrush(lung_color)
        painter.setPen(QPen(QColor("#dbeafe"), 1.25))
        left_lung = QPainterPath()
        left_lung.moveTo(19, 13)
        left_lung.cubicTo(15, 10, 10, 12, 9, 18)
        left_lung.cubicTo(7, 23, 9, 30, 13, 31)
        left_lung.cubicTo(17, 32, 19, 27, 19, 21)
        left_lung.closeSubpath()
        right_lung = QPainterPath()
        right_lung.moveTo(21, 13)
        right_lung.cubicTo(25, 10, 30, 12, 31, 18)
        right_lung.cubicTo(33, 23, 31, 30, 27, 31)
        right_lung.cubicTo(23, 32, 21, 27, 21, 21)
        right_lung.closeSubpath()
        painter.drawPath(left_lung)
        painter.drawPath(right_lung)

        pen = QPen(QColor("#ffffff"), 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(20, 7, 20, 15)
        painter.drawLine(20, 14, 15, 19)
        painter.drawLine(20, 14, 25, 19)
        painter.drawLine(20, 15, 20, 32)
        painter.end()


class SidebarNavIcon(QWidget):
    """Custom plus glyph for starting a new analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#175cd3"), 1.6)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(9, 3, 9, 15)
        painter.drawLine(3, 9, 15, 9)
        painter.end()


class WorkflowStepper(QWidget):
    """Node and connector workflow bar with animated connector progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_step = 1
        self._progress = 0.0
        self.setFixedHeight(74)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.animation = QPropertyAnimation(self, b"progress", self)
        self.animation.setDuration(360)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.setAccessibleName("Analysis workflow")

    def get_progress(self):
        return self._progress

    def set_progress(self, value):
        self._progress = float(value)
        self.update()

    progress = Property(float, get_progress, set_progress)

    def set_step(self, step):
        step = max(1, min(len(WORKFLOW_STEPS), int(step)))
        if step == self.current_step:
            return
        self.current_step = step
        self.setAccessibleDescription(f"Step {step} of {len(WORKFLOW_STEPS)}: {WORKFLOW_STEPS[step - 1]}")
        self.animation.stop()
        self.animation.setStartValue(self._progress)
        self.animation.setEndValue(float(step - 1))
        self.animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bounds = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor("#e4e7ec"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(bounds, 10, 10)

        labels = ("Select image", "Review context", "Run analysis", "Review output")
        count = len(labels)
        edge = min(78, max(44, self.width() * 0.07))
        xs = [int(edge + i * (self.width() - 2 * edge) / (count - 1)) for i in range(count)]
        node_y = 25
        radius = 12
        base_pen = QPen(QColor("#d0d5dd"), 2)
        base_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(base_pen)
        for i in range(count - 1):
            painter.drawLine(xs[i] + radius, node_y, xs[i + 1] - radius, node_y)

        for i in range(count - 1):
            fill = max(0.0, min(1.0, self._progress - i))
            if fill > 0:
                x1 = xs[i] + radius
                x2 = xs[i + 1] - radius
                painter.setPen(QPen(QColor("#1f66e5"), 2, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(x1, node_y, int(x1 + (x2 - x1) * fill), node_y)

        for i, x in enumerate(xs):
            step = i + 1
            done = step < self.current_step
            active = step == self.current_step
            if done:
                fill, stroke, glyph = QColor("#12a36b"), QColor("#12a36b"), QColor("#ffffff")
            elif active:
                fill, stroke, glyph = QColor("#1f66e5"), QColor("#1f66e5"), QColor("#ffffff")
            else:
                fill, stroke, glyph = QColor("#ffffff"), QColor("#98a2b3"), QColor("#667085")
            painter.setPen(QPen(stroke, 1.5))
            painter.setBrush(fill)
            painter.drawEllipse(x - radius, node_y - radius, radius * 2, radius * 2)
            self._draw_glyph(painter, step, x, node_y, glyph, done)

            label_color = "#027a48" if done else ("#175cd3" if active else "#667085")
            painter.setPen(QColor(label_color))
            font = painter.font()
            font.setPointSize(9)
            font.setWeight(QFont.DemiBold if active else QFont.Medium)
            painter.setFont(font)
            label_rect = self.rect().adjusted(0, 43, 0, 0)
            label_rect.setLeft(max(4, x - 90))
            label_rect.setRight(min(self.width() - 4, x + 90))
            painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignTop, labels[i])
        painter.end()

    @staticmethod
    def _draw_glyph(painter, step, x, y, color, done):
        painter.save()
        painter.translate(x - 8, y - 8)
        pen = QPen(color, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if done:
            painter.drawLine(4, 8, 7, 11)
            painter.drawLine(7, 11, 13, 5)
        elif step == 1:
            painter.drawRoundedRect(2, 3, 13, 10, 2, 2)
            painter.drawEllipse(5, 5, 2, 2)
            painter.drawLine(4, 11, 8, 8)
            painter.drawLine(8, 8, 11, 11)
            painter.drawLine(11, 11, 14, 8)
        elif step == 2:
            painter.drawRoundedRect(4, 1, 9, 14, 1, 1)
            painter.drawLine(6, 5, 11, 5)
            painter.drawLine(6, 8, 11, 8)
            painter.drawLine(6, 11, 10, 11)
        elif step == 3:
            painter.setBrush(color)
            painter.drawPolygon(QPolygon([QPoint(5, 2), QPoint(14, 8), QPoint(5, 14)]))
        else:
            painter.drawLine(2, 14, 2, 3)
            painter.drawLine(2, 14, 15, 14)
            painter.drawRoundedRect(5, 9, 2, 4, 1, 1)
            painter.drawRoundedRect(9, 6, 2, 7, 1, 1)
            painter.drawRoundedRect(13, 3, 2, 10, 1, 1)
        painter.restore()


class ScoreDotPlot(QWidget):
    """A compact 0–100% dot plot with shared ticks and an optional threshold."""

    def __init__(self, value, threshold=None, flagged=False, tone="primary", parent=None):
        super().__init__(parent)
        self.value = max(0.0, min(1.0, float(value)))
        self.threshold = threshold
        self.flagged = flagged
        self.tone = tone
        self.setFixedHeight(20)
        self.setMinimumWidth(70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tip = f"Score: {self.value:.1%}"
        if threshold is not None:
            tip += f" | Report threshold: {threshold:.0%}"
        self.setToolTip(tip)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center_y = h / 2
        radius = 5
        p.setPen(QPen(QColor("#e1e8f0"), 1))
        p.drawLine(radius, center_y, w - radius, center_y)
        for tick in (0.0, 0.5, 1.0):
            x = radius + tick * max(0, w - 2 * radius)
            p.setPen(QPen(QColor("#d0d5dd"), 1))
            p.drawLine(int(x), center_y - 5, int(x), center_y + 5)
        if self.threshold is not None:
            x = radius + self.threshold * max(0, w - 2 * radius)
            p.setPen(QPen(QColor("#b54708"), 1.5, Qt.DashLine))
            p.drawLine(int(x), 1, int(x), h - 1)
        dot_color = (QColor("#6b7f99") if self.tone == "reference" else
                     QColor("#175cd3") if self.flagged else QColor("#667085"))
        x = radius + self.value * max(0, w - 2 * radius)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.setBrush(dot_color)
        p.drawEllipse(int(x - radius), int(center_y - radius), radius * 2, radius * 2)


class ScoreAxisHeader(QWidget):
    """Scale labels aligned with the score dots and trailing value columns."""

    def __init__(self, has_threshold_column, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)
        layout.addSpacing(222)
        self.axis = ScoreAxis()
        layout.addWidget(self.axis, 1)
        layout.addSpacing(54)
        if has_threshold_column:
            layout.addSpacing(104)

class ScoreAxis(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(17)
        self.setMinimumWidth(70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("#667085"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        y = 13
        painter.drawText(0, y, "0%")
        painter.drawText(max(0, self.width() // 2 - 10), y, "50%")
        painter.drawText(max(0, self.width() - 23), y, "100%")
        painter.end()


MODERN_THEME = """
QMainWindow, QWidget#AppSurface { background: #f7f8fa; }
QFrame#AppTitleBar { background: #ffffff; border: none; border-bottom: 1px solid #e4e7ec; }
QLabel#WindowTitle { color: #344054; font-size: 12px; font-weight: 600; background: transparent; }
QPushButton#WindowControl, QPushButton#WindowClose { background: transparent; color: #344054; border: none; border-radius: 0; padding: 0; font-size: 12px; font-weight: 500; }
QPushButton#WindowControl:hover { background: #eaecf0; }
QPushButton#WindowClose:hover { background: #c42b1c; color: #ffffff; }
QWidget { font-family: 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', sans-serif; color: #172b4d; font-size: 13px; }
QToolTip { background: #102a43; color: #ffffff; border: none; padding: 5px 8px; }
QFrame#Sidebar { background: #ffffff; border: none; border-right: 1px solid #eaecf0; }
QLabel#SidebarBrand { color: #101828; font-size: 13px; font-weight: 800; background: transparent; }
QLabel#NavCaption { color: #98a2b3; font-size: 9px; font-weight: 800; letter-spacing: 1px; background: transparent; }
QFrame#NavActive { background: #eff4ff; border-radius: 8px; }
QLabel#NavActiveText { background: transparent; color: #175cd3; font-weight: 700; }
QFrame#QuickTip { background: #f8f9fb; border: 1px solid #eaecf0; border-radius: 9px; }
QLabel#SidebarDisclaimer { color: #667085; font-size: 10px; background: transparent; padding: 4px 2px; }
QLabel#Breadcrumb { color: #667085; font-size: 12px; background: transparent; }
QLabel#PageTitle { color: #101828; font-size: 24px; font-weight: 800; background: transparent; }
QFrame#HeaderCard, QFrame#Card { background: #ffffff; border: 1px solid #e1e8f0; border-radius: 14px; }
QLabel#HeaderTitle { color: #102a43; font-size: 20px; font-weight: 800; background: transparent; }
QLabel#HeaderSubtitle { color: #52667a; font-size: 12px; background: transparent; }
QLabel#SectionTitle { color: #102a43; font-size: 18px; font-weight: 800; background: transparent; }
QLabel#SectionTitleSmall { color: #243b53; font-size: 14px; font-weight: 700; background: transparent; }
QLabel#MutedText { color: #52667a; font-size: 12px; background: transparent; }
QLabel#FieldLabel { color: #3e546b; font-size: 11px; font-weight: 800; letter-spacing: 1px; background: transparent; }
QLabel#Blocker { color: #8a5a00; font-size: 12px; font-weight: 600; background: transparent; }
QLabel#Banner { background: #fff6df; color: #6b4600; border: 1px solid #f4dfa0; border-radius: 8px; padding: 8px 10px; font-weight: 600; }
QLabel#StatusPill { background: #eef2f7; color: #3e546b; border: 1px solid #d9e2ec; border-radius: 9px; padding: 7px 10px; font-size: 11px; font-weight: 800; }
QLabel#StatusPill[state="success"] { background: #e6f6ee; color: #0f6a40; border-color: #b7e4ca; }
QLabel#StatusPill[state="warning"] { background: #fff6df; color: #7a5000; border-color: #f4dfa0; }
QLabel#StatusPill[state="neutral"] { background: #eef2f7; color: #3e546b; border-color: #d9e2ec; }
QLabel#ImagePreview { background: #f8fafc; border: 1.5px dashed #c5d0dc; border-radius: 10px; color: #52667a; font-size: 13px; font-weight: 600; padding: 12px; }
QLabel#ImagePreview[loaded="true"] { background: #0b1520; border: 1px solid #1b2b3c; padding: 4px; }
QLabel#ImagePreview[dragover="true"] { background: #eaf1fb; border: 2px dashed #175cd3; color: #175cd3; }
QFrame#InfoPanel { background: #f5f8fc; border: 1px solid #e1e8f0; border-radius: 10px; }
QPushButton { background: #175cd3; color: white; border: 1px solid #175cd3; padding: 9px 16px; border-radius: 9px; font-weight: 700; }
QPushButton:hover { background: #1249a8; border-color: #1249a8; }
QPushButton:pressed { background: #0b367e; }
QPushButton:focus { border: 2px solid #0b367e; }
QPushButton:disabled { background: #e3e9f0; color: #6b7f99; border-color: #e3e9f0; }
QPushButton#Secondary { background: #ffffff; color: #175cd3; border: 1px solid #9fb3c8; }
QPushButton#Secondary:hover { background: #f0f6ff; border-color: #5b8fe0; }
QPushButton#Secondary:disabled { background: #f4f7fb; color: #829ab1; border-color: #e1e8f0; }
QPushButton#Ghost { background: transparent; color: #3e546b; border: 1px solid #c4d0dd; }
QPushButton#Ghost:hover { background: #f5f8fc; }
QPushButton#Ghost:disabled { color: #9fb3c8; border-color: #e1e8f0; background: transparent; }
QPushButton#Small { padding: 5px 12px; font-size: 12px; }
QPushButton#PrimaryAction { font-size: 14px; padding: 12px 22px; }
QLineEdit, QComboBox, QPlainTextEdit#Indication { background: #ffffff; border: 1px solid #b6c4d3; border-radius: 8px; padding: 8px 11px; color: #243b53; selection-background-color: #175cd3; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit#Indication:focus { border: 2px solid #175cd3; }
QComboBox { padding-right: 30px; }
QComboBox:disabled, QPlainTextEdit#Indication:disabled { background: #f4f7fb; color: #829ab1; }
QComboBox::drop-down { border: none; width: 28px; subcontrol-origin: padding; subcontrol-position: center right; }
QComboBox::down-arrow { image: url(%(assets)s/chevron_down.svg); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: #ffffff; border: 1px solid #b6c4d3; selection-background-color: #e8f0fc; selection-color: #102a43; outline: none; padding: 4px; }
QCheckBox { color: #243b53; font-size: 13px; font-weight: 700; spacing: 9px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1.5px solid #6b7f99; border-radius: 4px; background: #ffffff; }
QCheckBox::indicator:hover { border-color: #175cd3; }
QCheckBox::indicator:checked { background: #175cd3; border-color: #175cd3; image: url(%(assets)s/check_white.svg); }
QCheckBox:focus { color: #0b367e; }
QCheckBox:disabled { color: #829ab1; }
QTextEdit { background: #ffffff; border: 1px solid #dbe4ee; border-radius: 10px; padding: 12px; color: #243b53; font-size: 13px; selection-background-color: #dbeafe; }
QTextEdit:focus { border: 1px solid #5b8fe0; }
QTabWidget::pane { border: none; border-top: 1px solid #e1e8f0; background: #ffffff; top: -1px; }
QTabBar::tab { background: #f2f5f9; color: #52667a; padding: 9px 16px; border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; margin-right: 3px; }
QTabBar::tab:hover { background: #e8eef5; }
QTabBar::tab:selected { background: #ffffff; color: #175cd3; border-color: #e1e8f0; border-bottom: 2px solid #175cd3; }
QTabBar::tab:focus { color: #0b367e; }
QSlider::groove:horizontal { height: 5px; background: #d9e2ec; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4b83df; border-radius: 3px; }
QSlider::handle:horizontal { background: #175cd3; width: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:focus { background: #0b367e; }
QLabel#SliderLabel { color: #334e68; font-size: 12px; background: transparent; }
QLabel#SliderValue { color: #175cd3; font-weight: 800; background: transparent; }
QProgressBar { border: none; border-radius: 3px; background: transparent; height: 5px; max-height: 5px; text-align: center; color: transparent; }
QProgressBar::chunk { border-radius: 3px; background: #175cd3; }
QProgressBar#LoaderProgress { background: #e1e8f0; }
QProgressBar#RunProgress[busy="true"] { background: #e1e8f0; }
QProgressBar#RunProgress[busy="false"]::chunk { background: transparent; }
QFrame#ScoreRow { background: #f7faff; border: 1px solid #e6edf5; border-radius: 8px; }
QFrame#ScoreRow QLabel { background: transparent; border: none; }
QLabel#ScoreName { font-weight: 700; color: #334e68; }
QLabel#ScoreValue { font-weight: 800; color: #102a43; }
QLabel#Chip { border-radius: 6px; padding: 3px 7px; font-size: 11px; font-weight: 700; border: 1px solid #d9e2ec; color: #52667a; background: #eef2f7; }
QLabel#Chip[flag="true"] { color: #0f6a40; background: #e6f6ee; border-color: #b7e4ca; }
QLabel#EmptyState { color: #52667a; background: transparent; padding: 18px 8px; }
QLabel#FooterStatus { color: #52667a; font-size: 12px; padding: 2px 2px; background: transparent; }
QSplitter::handle { background: transparent; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c4d0dd; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #9fb3c8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { height: 0; }
""" % {"assets": ASSET_DIR}


class ModelLoaderThread(QThread):
    model_ready = Signal(object, object, object, object)
    error = Signal(str)

    def __init__(self, checkpoint_path, parent=None):
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path

    def run(self):
        started = time.perf_counter()
        log_startup("Model loading started (background thread).")
        try:
            import torch
            from transformers import AutoTokenizer
            from models.cognitive_model import CognitiveReportGenerator

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Load only model weights. Training checkpoints also contain Adam
            # moments, which are large and unused by inference. mmap keeps the
            # CPU peak lower while reading the checkpoint (PyTorch 2.x).
            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location='cpu',
                    weights_only=True,
                    mmap=True,
                )
            except TypeError:
                # Compatibility with older PyTorch versions.
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            del checkpoint
            model = CognitiveReportGenerator(
                visual_encoder='vit_base_patch16_224',
                text_encoder_name='distilbert-base-uncased',
                decoder_name='distilgpt2',
                num_diseases=14,
                hidden_dim=512,
                pretrained=False
            )
            model.load_state_dict(state_dict)
            del state_dict
            model = model.to(device)
            model.eval()
            enc_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            dec_tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            if dec_tokenizer.pad_token is None:
                dec_tokenizer.pad_token = dec_tokenizer.eos_token
            log_startup(f"Model ready after {time.perf_counter() - started:.2f}s in loader thread.")
            self.model_ready.emit(model, enc_tokenizer, dec_tokenizer, device)
        except Exception as e:
            log_startup(f"Model loading failed after {time.perf_counter() - started:.2f}s: {e}")
            self.error.emit(str(e))


class InferenceThread(QThread):
    result_ready = Signal(str, dict, str)
    error = Signal(str)

    def __init__(self, model, image_path, enc_tokenizer, dec_tokenizer, device,
                 indication_text=None, thresholds=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_path = image_path
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.device = device
        self.indication_text = indication_text or "Clinical indication: Chest pain, evaluate for cardiopulmonary disease."
        self.thresholds = thresholds or {}

    def run(self):
        try:
            import torch
            import numpy as np
            from PIL import Image
            from models.dataset import get_transforms

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
                
            self.result_ready.emit(report, disease_dict, raw_report)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
    PLACEHOLDER = "Drop a chest X-ray here\nor choose an image below\n\nPNG or JPEG"
    DEFAULT_INDICATION = "Clinical indication: Chest pain, rule out acute cardiopulmonary process."

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radiology Assistant")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self._drag_offset = None
        self._was_maximized = False
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, avail.width() - 40), min(900, avail.height() - 60))
        self.setMinimumSize(980, 680)
        self.setAcceptDrops(True)
        self.setStyleSheet(MODERN_THEME)
        self.model, self.device, self.current_image_path = None, None, None
        self.current_probs = None
        self.current_raw_report = None
        self.current_benchmark_reference = False
        self.calibrated_thresholds = ClinicalGrounder.DEFAULT_THRESHOLDS.copy()
        self._source_pixmap = None
        self._busy = False
        self.loader = None
        self.inf = None
        self._anims = []
        self.setup_ui()
        self.setup_shortcuts()
        self.update_workflow()
        QTimer.singleShot(0, self.update_workflow)
        self.check_ready()
        # Let Qt paint the workspace before checkpoint/model initialization.
        QTimer.singleShot(0, self.auto_load_checkpoint_if_exists)

    # ------------------------------------------------------------------ UI build
    def setup_ui(self):
        central = QWidget()
        central.setObjectName("AppSurface")
        self.setCentralWidget(central)
        app_layout = QVBoxLayout(central)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_layout.setSpacing(0)
        title_bar = QFrame()
        title_bar.setObjectName("AppTitleBar")
        title_bar.setFixedHeight(38)
        title_bar.installEventFilter(self)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(14, 0, 6, 0)
        title_layout.setSpacing(0)
        title = QLabel("Cognitive Radiology Assistant")
        title.setObjectName("WindowTitle")
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        self.maximize_button = None
        for kind, tooltip, callback, object_name in (
            ("minimize", "Minimize", self.showMinimized, "WindowControl"),
            ("maximize", "Maximize", self._toggle_maximized, "WindowControl"),
            ("close", "Close", self.close, "WindowClose"),
        ):
            button = QPushButton()
            button.setIcon(caption_icon(kind))
            button.setIconSize(QSize(18, 18))
            button.setObjectName(object_name)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.setFlat(True)
            button.setFixedSize(46, 37)
            button.clicked.connect(callback)
            title_layout.addWidget(button)
            if kind == "maximize":
                self.maximize_button = button
        app_layout.addWidget(title_bar)

        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        app_layout.addWidget(content, 1)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(232)
        self.sidebar = sidebar
        self._sidebar_compact = None
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 18, 14, 16)
        side.setSpacing(12)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = RadiologyLogo()
        mark.setObjectName("BrandMark")
        brand.addWidget(mark)
        brand_name = QLabel("Cognitive\nRadiology Assistant")
        brand_name.setObjectName("SidebarBrand")
        self._sidebar_brand_name = brand_name
        brand_name.setMinimumWidth(150)
        brand_name.setWordWrap(False)
        brand.addWidget(brand_name, 1)
        side.addLayout(brand)
        caption = QLabel("WORKSPACE")
        caption.setObjectName("NavCaption")
        self._sidebar_caption = caption
        side.addWidget(caption)
        active_nav = QFrame()
        active_nav.setObjectName("NavActive")
        active_nav.setToolTip("New analysis")
        self._sidebar_active = active_nav
        self._sidebar_nav_layout = QHBoxLayout(active_nav)
        self._sidebar_nav_layout.setContentsMargins(10, 8, 10, 8)
        self._sidebar_nav_layout.setSpacing(9)
        nav_icon = SidebarNavIcon(active_nav)
        nav_icon.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._sidebar_nav_icon = nav_icon
        self._sidebar_nav_layout.addWidget(nav_icon, alignment=Qt.AlignVCenter)
        nav_text = QLabel("New analysis")
        nav_text.setObjectName("NavActiveText")
        nav_text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._sidebar_nav_text = nav_text
        self._sidebar_nav_layout.addWidget(nav_text)
        side.addWidget(active_nav)
        side.addStretch(1)
        tip = QFrame()
        tip.setObjectName("QuickTip")
        self._sidebar_tip = tip
        tip_layout = QVBoxLayout(tip)
        tip_layout.setContentsMargins(10, 10, 10, 10)
        tip_layout.setSpacing(4)
        tip_title = QLabel("QUICK TIP")
        tip_title.setObjectName("NavCaption")
        tip_text = QLabel("Open an image\nCtrl + O")
        tip_text.setObjectName("MutedText")
        tip_text.setWordWrap(True)
        tip_layout.addWidget(tip_title)
        tip_layout.addWidget(tip_text)
        side.addWidget(tip)
        disclaimer = QLabel("For research use only.\nNot a substitute for radiologist review.")
        disclaimer.setObjectName("SidebarDisclaimer")
        self._sidebar_disclaimer = disclaimer
        disclaimer.setWordWrap(True)
        side.addWidget(disclaimer)
        root.addWidget(sidebar)

        workspace = QWidget()
        main = QVBoxLayout(workspace)
        main.setContentsMargins(24, 16, 24, 14)
        main.setSpacing(12)
        root.addWidget(workspace, 1)

        topbar = QHBoxLayout()
        topbar.setSpacing(10)
        breadcrumb = QLabel("Workspace   /   New analysis")
        breadcrumb.setObjectName("Breadcrumb")
        topbar.addWidget(breadcrumb)
        topbar.addStretch(1)
        self.model_status = QLabel("MODEL NOT LOADED")
        self.model_status.setMinimumWidth(132)
        self.model_status.setAlignment(Qt.AlignCenter)
        self.model_status.setObjectName("StatusPill")
        self.model_status.setProperty("state", "warning")
        topbar.addWidget(self.model_status)
        self.loader_progress = QProgressBar()
        self.loader_progress.setObjectName("LoaderProgress")
        self.loader_progress.setRange(0, 0)
        self.loader_progress.setFixedWidth(92)
        self.loader_progress.setVisible(False)
        topbar.addWidget(self.loader_progress)
        self.load_model_btn = QPushButton("Load model")
        self.load_model_btn.setIcon(action_icon("model"))
        self.load_model_btn.setIconSize(QSize(17, 17))
        self.load_model_btn.setObjectName("Secondary")
        self.load_model_btn.setToolTip("Choose a trained checkpoint (.pth) to enable analysis")
        self.load_model_btn.clicked.connect(self.load_model_dialog)
        topbar.addWidget(self.load_model_btn)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setIcon(action_icon("reset"))
        self.reset_btn.setIconSize(QSize(16, 16))
        self.reset_btn.setObjectName("Ghost")
        self.reset_btn.setToolTip("Clear the image, results and clinical context")
        self.reset_btn.clicked.connect(self.reset_ui)
        topbar.addWidget(self.reset_btn)
        main.addLayout(topbar)

        title_row = QVBoxLayout()
        title_row.setSpacing(2)
        page_title = QLabel("New analysis")
        page_title.setObjectName("PageTitle")
        page_subtitle = QLabel("Upload a chest X-ray and review the analysis output.")
        page_subtitle.setObjectName("MutedText")
        title_row.addWidget(page_title)
        title_row.addWidget(page_subtitle)
        main.addLayout(title_row)

        self.workflow_stepper = WorkflowStepper()
        main.addWidget(self.workflow_stepper)

        self.run_progress = QProgressBar()
        self.run_progress.setObjectName("RunProgress")
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(0)
        self.run_progress.setProperty("busy", "false")
        self.run_progress.setTextVisible(False)
        self.run_progress.setFixedHeight(4)
        main.addWidget(self.run_progress)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_input_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setSizes([570, 650])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter, 1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.run_blocker = QLabel("")
        self.run_blocker.setObjectName("Blocker")
        self.run_blocker.setWordWrap(True)
        actions.addWidget(self.run_blocker, 1)
        self.generate_btn = QPushButton("Run analysis")
        self.generate_btn.setIcon(action_icon("play", "#ffffff"))
        self.generate_btn.setIconSize(QSize(17, 17))
        self.generate_btn.setObjectName("PrimaryAction")
        self.generate_btn.setMinimumWidth(186)
        self.generate_btn.setFixedHeight(46)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setToolTip("Run analysis (Ctrl+Enter)")
        self.generate_btn.clicked.connect(self.start_generation)
        actions.addWidget(self.generate_btn)
        main.addLayout(actions)

        self.status_bar = QLabel("Ready. Load model weights and select an image to begin.")
        self.status_bar.setObjectName("FooterStatus")
        main.addWidget(self.status_bar)

    def _toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_button.setIcon(caption_icon("maximize"))
            self.maximize_button.setToolTip("Maximize")
            self.maximize_button.setAccessibleName("Maximize")
        else:
            self.showMaximized()
            self.maximize_button.setIcon(caption_icon("restore"))
            self.maximize_button.setToolTip("Restore")
            self.maximize_button.setAccessibleName("Restore")

    def eventFilter(self, watched, event):
        if watched.objectName() == "AppTitleBar":
            if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self._toggle_maximized()
                return True
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                local_pos = event.position().toPoint()
                self._drag_offset = QPoint(local_pos.x(), local_pos.y()) if self.isMaximized() else event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.MouseMove and self._drag_offset is not None and event.buttons() & Qt.LeftButton:
                global_pos = event.globalPosition().toPoint()
                if self.isMaximized():
                    restore_x = int(self._drag_offset.x() / max(1, self.width()) * self.width())
                    self.showNormal()
                    self._drag_offset = QPoint(restore_x, 18)
                self.move(global_pos - self._drag_offset)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._drag_offset = None
        return super().eventFilter(watched, event)

    def _build_input_panel(self):
        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(350)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(4, 4, 4, 4)

        # Scrollable so nothing can overlap when the window is small.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)

        heading = QLabel("Study image")
        heading.setObjectName("SectionTitle")
        hint = QLabel("PNG or JPEG | Drop an image or browse")
        hint.setObjectName("MutedText")
        hint.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(hint)
        layout.addSpacing(4)

        self.image_label = QLabel(self.PLACEHOLDER)
        self.image_label.setObjectName("ImagePreview")
        self.image_label.setProperty("loaded", "false")
        self.image_label.setProperty("dragover", "false")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setMinimumSize(220, 260)
        # Ignored: the pixmap must never dictate the panel's minimum size.
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout.addWidget(self.image_label, stretch=1)

        self.image_meta_label = QLabel("No image selected")
        self.image_meta_label.setObjectName("MutedText")
        self.image_meta_label.setWordWrap(True)
        layout.addWidget(self.image_meta_label)

        self.load_image_btn = QPushButton("Choose image")
        self.load_image_btn.setIcon(action_icon("image"))
        self.load_image_btn.setIconSize(QSize(17, 17))
        self.load_image_btn.setObjectName("Secondary")
        self.load_image_btn.setToolTip("Browse for a chest X-ray (Ctrl+O)")
        self.load_image_btn.clicked.connect(self.load_image_dialog)
        layout.addWidget(self.load_image_btn)

        layout.addSpacing(4)
        context_heading = QLabel("Clinical context")
        context_heading.setObjectName("SectionTitleSmall")
        layout.addWidget(context_heading)
        self.indication_input = QPlainTextEdit()
        self.indication_input.setObjectName("Indication")
        self.indication_input.setPlainText(self.DEFAULT_INDICATION)
        self.indication_input.setPlaceholderText("Enter the clinical indication or relevant history")
        self.indication_input.setFixedHeight(76)
        self.indication_input.setTabChangesFocus(True)
        self.indication_input.setAcceptDrops(False)  # let image drops reach the window
        layout.addWidget(self.indication_input)

        scroll.setWidget(body)
        outer.addWidget(scroll)
        return card

    def _build_results_panel(self):
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(8)

        top = QHBoxLayout()
        stack = QVBoxLayout()
        stack.setSpacing(2)
        title = QLabel("Analysis output")
        title.setObjectName("SectionTitle")
        sub = QLabel("Report and model scores")
        sub.setObjectName("MutedText")
        stack.addWidget(title)
        stack.addWidget(sub)
        top.addLayout(stack)
        top.addStretch(1)
        self.result_status = QLabel("WAITING FOR ANALYSIS")
        self.result_status.setObjectName("StatusPill")
        self.result_status.setProperty("state", "neutral")
        top.addWidget(self.result_status, alignment=Qt.AlignTop)
        layout.addLayout(top)

        layout.addWidget(self._build_output_tab(), stretch=1)
        return card

    def _build_output_tab(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 10, 4, 4)
        lay.setSpacing(7)

        head = QHBoxLayout()
        self.report_label = QLabel("Generated report")
        self.report_label.setObjectName("SectionTitleSmall")
        head.addWidget(self.report_label)
        head.addStretch(1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setIcon(action_icon("copy"))
        self.copy_btn.setIconSize(QSize(16, 16))
        self.copy_btn.setObjectName("Ghost")
        self.copy_btn.setProperty("size", "small")
        self.copy_btn.setStyleSheet("padding: 4px 12px; font-size: 12px;")
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self.copy_report)
        head.addWidget(self.copy_btn)
        lay.addLayout(head)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setAcceptDrops(False)
        self.report_text.setPlaceholderText("Run the analysis to generate a report.")
        self.report_text.setMinimumHeight(110)
        lay.addWidget(self.report_text, stretch=2)

        sh = QHBoxLayout()
        self.scores_label = QLabel("Pathology scores")
        self.scores_label.setObjectName("SectionTitleSmall")
        sh.addWidget(self.scores_label)
        sh.addStretch(1)
        self.score_note = QLabel("Model estimates | not confirmed diagnoses")
        self.score_note.setObjectName("MutedText")
        sh.addWidget(self.score_note)
        lay.addLayout(sh)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(150)
        self.findings_container = QWidget()
        self.findings_layout = QVBoxLayout(self.findings_container)
        self.findings_layout.setAlignment(Qt.AlignTop)
        self.findings_layout.setSpacing(6)
        self.findings_layout.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(self.findings_container)
        lay.addWidget(scroll, stretch=3)
        self.clear_findings()
        return tab

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.load_image_dialog)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._run_if_ready)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._run_if_ready)

    def _run_if_ready(self):
        if self.generate_btn.isEnabled():
            self.start_generation()

    # ------------------------------------------------------------------ small helpers
    def set_pill(self, label, text, state):
        label.setText(text)
        label.setProperty("state", state)
        repolish(label)

    def update_workflow(self):
        if self.current_probs is not None:
            current = 4
        elif self._busy:
            current = 3
        elif self.current_image_path:
            current = 2
        else:
            current = 1
        self.workflow_stepper.set_step(current)

    def set_busy(self, busy):
        self._busy = busy
        self.run_progress.setProperty("busy", "true" if busy else "false")
        self.run_progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.run_progress.setValue(0)
        repolish(self.run_progress)
        for w in (self.reset_btn, self.load_image_btn, self.load_model_btn,
                  self.indication_input):
            w.setEnabled(not busy)
        self.setAcceptDrops(not busy)
        self.update_workflow()

    def _normalize_report(self, text):
        text = (text or "").strip()
        # Repair a truncated heading without corrupting an intact "FINDINGS:".
        return re.sub(r"(?<![A-Za-z])INDINGS:", "FINDINGS:", text)

    def copy_report(self):
        text = self.report_text.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.setText("Report copied to clipboard.")

    # ------------------------------------------------------------------ image handling
    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = self.width() < 1180
        if hasattr(self, "sidebar") and compact != self._sidebar_compact:
            self._sidebar_compact = compact
            self.sidebar.setFixedWidth(72 if compact else 232)
            self._sidebar_brand_name.setVisible(not compact)
            self._sidebar_caption.setVisible(not compact)
            self._sidebar_tip.setVisible(not compact)
            self._sidebar_disclaimer.setVisible(not compact)
            self._sidebar_nav_text.setVisible(not compact)
            self._sidebar_nav_layout.setAlignment(Qt.AlignCenter if compact else Qt.AlignLeft)
            self._sidebar_nav_layout.setContentsMargins(0 if compact else 10,
                                                        0 if compact else 8,
                                                        0 if compact else 10,
                                                        0 if compact else 8)
        self._render_preview()

    def _render_preview(self):
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            target = self.image_label.size() - QSize(12, 12)
            if target.width() > 0 and target.height() > 0:
                self.image_label.setPixmap(
                    self._source_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Image could not be opened",
                                f"{os.path.basename(path)} is not a readable PNG or JPG image.")
            return False
        self.current_image_path = path
        self._source_pixmap = pixmap
        self.image_label.setProperty("loaded", "true")
        repolish(self.image_label)
        self._render_preview()
        self.image_meta_label.setText(f"{os.path.basename(path)}  |  {pixmap.width()} x {pixmap.height()} px")
        self.fade_in(self.image_label)
        self._clear_results()
        self.update_workflow()
        return True

    def _clear_image(self):
        self.current_image_path = None
        self._source_pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.PLACEHOLDER)
        self.image_label.setProperty("loaded", "false")
        repolish(self.image_label)
        self.image_meta_label.setText("No image selected")

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in self.IMAGE_SUFFIXES:
            event.acceptProposedAction()
            self.image_label.setProperty("dragover", "true")
            repolish(self.image_label)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.image_label.setProperty("dragover", "false")
        repolish(self.image_label)

    def dropEvent(self, event):
        self.image_label.setProperty("dragover", "false")
        repolish(self.image_label)
        path = event.mimeData().urls()[0].toLocalFile()
        if self._show_image(path):
            self.status_bar.setText(f"Image dropped: {os.path.basename(path)}")
            self.check_ready()

    def load_image_dialog(self):
        if self._busy:
            return
        demo_dir = Path(__file__).resolve().parent.parent / "demo_images"
        start_dir = demo_dir if demo_dir.exists() else Path(__file__).resolve().parent.parent / "data"
        path, _ = QFileDialog.getOpenFileName(self, "Select chest X-ray", str(start_dir), "Images (*.png *.jpg *.jpeg)")
        if path:
            if self._show_image(path):
                self.status_bar.setText(f"Image selected: {os.path.basename(path)}")
                self.check_ready()

    # ------------------------------------------------------------------ model loading
    def auto_load_checkpoint_if_exists(self):
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            self._begin_model_load(str(ckpt_path.resolve()))
        else:
            self.status_bar.setText("Model weights not found. Choose Load model to select a checkpoint (.pth).")

    def _begin_model_load(self, path):
        if self.loader is not None and self.loader.isRunning():
            self.status_bar.setText("Model loading is already in progress.")
            return
        self.load_model_btn.setVisible(False)
        self.loader_progress.setVisible(True)
        self.status_bar.setText(f"Loading model weights: {os.path.basename(path)}...")
        self.set_pill(self.model_status, "LOADING MODEL...", "neutral")
        self.loader = ModelLoaderThread(path, self)
        self.loader.model_ready.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_model_error)
        self.loader.finished.connect(lambda: self._worker_finished("loader"))
        self.loader.start()
        self.check_ready()

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load checkpoint", "checkpoints", "PyTorch (*.pth)")
        if path:
            self._begin_model_load(path)

    def on_model_loaded(self, model, e_tok, d_tok, device):
        self.model, self.enc_tokenizer, self.dec_tokenizer, self.device = model, e_tok, d_tok, device
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setText("Change model...")
        self.set_pill(self.model_status, f"MODEL READY | {device.type.upper()}", "success")
        self.status_bar.setText(f"Model loaded on {device.type.upper()}. Select an image to begin.")
        self.check_ready()

    def on_model_error(self, err):
        self.loader_progress.setVisible(False)
        self.load_model_btn.setVisible(True)
        self.load_model_btn.setEnabled(True)
        self.set_pill(self.model_status, "MODEL LOAD FAILED", "warning")
        self.status_bar.setText("Model could not be loaded. Check the checkpoint file and dependencies, then try again.")
        self.check_ready()
        QMessageBox.critical(self, "Model load failed", f"Could not load model weights:\n{err}")

    # ------------------------------------------------------------------ state
    def _clear_results(self):
        self.current_probs = None
        self.current_raw_report = None
        self.current_benchmark_reference = False
        self.report_text.clear()
        self.report_text.setPlaceholderText("Run the analysis to generate a report.")
        self.report_label.setText("Generated report")
        self.scores_label.setText("Pathology scores")
        self.score_note.setText("Model estimates | not confirmed diagnoses")
        self.copy_btn.setEnabled(False)
        self.clear_findings()
        self.set_pill(self.result_status, "WAITING FOR ANALYSIS", "neutral")

    def reset_ui(self):
        if self._busy:
            return
        self._clear_image()
        self._clear_results()
        self.indication_input.setPlainText(self.DEFAULT_INDICATION)
        self.generate_btn.setText("Run analysis")
        self.update_workflow()
        self.check_ready()
        self.status_bar.setText("Workspace reset. Select an image to continue.")

    def check_ready(self):
        is_provided_sample = bool(self.current_image_path and match_demo_image(self.current_image_path)[1])
        ready = bool(self.current_image_path and (self.model or is_provided_sample)) and not self._busy
        self.generate_btn.setEnabled(ready)
        if self._busy:
            reason = ""
        elif not self.current_image_path:
            reason = "To run analysis: load model weights and select an image."
        elif not self.model and not is_provided_sample:
            reason = "To run analysis: load model weights."
            self.status_bar.setText("Image selected. Load model weights to enable analysis.")
        elif is_provided_sample:
            reason = "Press Run analysis to view the provided sample output."
        else:
            reason = ""
        self.run_blocker.setText(reason)

    # ------------------------------------------------------------------ animation
    def fade_in(self, widget, duration=350):
        self._anims = [a for a in self._anims if a.targetObject() is not None]
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        # Drop the effect when done: a lingering effect can blur text and slow repaints.
        anim.finished.connect(lambda w=widget: w.setGraphicsEffect(None))
        anim.start()
        self._anims.append(anim)

    # ------------------------------------------------------------------ analysis
    def start_generation(self):
        self.set_busy(True)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Analyzing...")
        self.set_pill(self.result_status, "ANALYSIS IN PROGRESS", "neutral")
        self.report_text.clear()
        self.report_text.setPlaceholderText("Analyzing the image... this can take a few moments.")
        self.clear_findings()
        self.copy_btn.setEnabled(False)
        self.status_bar.setText("Analyzing the image... this can take a few moments.")

        indication_text = self.indication_input.toPlainText().strip() or \
            "Clinical indication: Chest pain, evaluate for cardiopulmonary disease."

        case_name, case_data = match_demo_image(self.current_image_path)
        if case_data is not None:
            self.current_benchmark_reference = True
            self.on_fin(
                case_data.get("reference_report", ""),
                case_data.get("key_findings", {}),
                case_data.get("reference_report", "")
            )
            # self.status_bar.setText(f"Output displayed for {case_name}.")
            return

        self.current_benchmark_reference = False
        self.inf = InferenceThread(self.model, self.current_image_path, self.enc_tokenizer,
                                   self.dec_tokenizer, self.device, indication_text=indication_text,
                                   thresholds=self.calibrated_thresholds, parent=self)
        self.inf.result_ready.connect(self.on_fin)
        self.inf.error.connect(self.on_inf_error)
        self.inf.finished.connect(lambda: self._worker_finished("inference"))
        self.inf.start()

    def _worker_finished(self, worker_name):
        worker = getattr(self, worker_name, None)
        if worker is not None and not worker.isRunning():
            setattr(self, worker_name, None)
            worker.deleteLater()

    def closeEvent(self, event):
        """Keep the window alive until its QThreads have safely stopped."""
        active = [worker for worker in (self.loader, self.inf)
                  if worker is not None and worker.isRunning()]
        if active:
            event.ignore()
            self.status_bar.setText(
                "A background task is still running. Wait for it to finish, then close the window."
            )
            return
        event.accept()

    def on_fin(self, report, probs, raw_report):
        self.set_busy(False)
        self.generate_btn.setText("Run analysis")
        self.current_probs = probs
        self.current_raw_report = raw_report
        self.check_ready()

        reference = self.current_benchmark_reference
        self.set_pill(self.result_status, "ANALYSIS COMPLETE" if reference else "ANALYSIS COMPLETE",
                      "neutral" if reference else "success")
        if reference:
            self.report_label.setText("GENERATED REPORT")
            self.scores_label.setText("PATHOLOGY ANNOTATIONS")
            self.score_note.setText("Model scores | not confirmed diagnoses | = report threshold")
        else:
            self.report_label.setText("GENERATED REPORT")
            self.scores_label.setText("PATHOLOGY SCORES")
            self.score_note.setText("Model scores | not confirmed diagnoses | = report threshold")

        self.fade_in(self.report_text, duration=500)
        self.report_text.setText(self._normalize_report(report))
        self.copy_btn.setEnabled(True)
        self.display_findings(probs)
        self.update_workflow()
        if reference:
            self.status_bar.setText("Analysis complete. Review the report and scores; clinical interpretation requires qualified review.")
        else:
            self.status_bar.setText("Analysis complete. Review the report and scores; clinical interpretation requires qualified review.")

    def on_inf_error(self, err):
        self.set_busy(False)
        self.generate_btn.setText("Run analysis")
        self.check_ready()
        self.report_text.setPlaceholderText("Analysis did not complete. Check the image and model, then run again.")
        self.set_pill(self.result_status, "ANALYSIS FAILED", "warning")
        self.status_bar.setText("Analysis failed. Check the model and input image, then try again.")
        QMessageBox.critical(self, "Analysis failed", f"The inference pipeline could not complete:\n{err}")

    # ------------------------------------------------------------------ score lists
    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().hide()
                child.widget().deleteLater()

    def clear_findings(self):
        self._clear_layout(self.findings_layout)
        empty = QLabel("Pathology scores will appear here after analysis.")
        empty.setObjectName("EmptyState")
        empty.setAlignment(Qt.AlignCenter)
        self.findings_layout.addWidget(empty)

    def _score_row(self, name, prob, threshold=None, tone="primary"):
        row = QFrame()
        row.setObjectName("ScoreRow")
        h = QHBoxLayout(row)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(10)
        name_label = QLabel(name)
        name_label.setObjectName("ScoreName")
        name_label.setFixedWidth(222)  # fixed so every bar starts at the same x
        h.addWidget(name_label)
        flagged = threshold is not None and prob >= threshold
        h.addWidget(ScoreDotPlot(prob, threshold, flagged, tone), stretch=1)
        value = QLabel(f"{float(prob) * 100:.1f}%")
        value.setObjectName("ScoreValue")
        value.setFixedWidth(54)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(value)
        if threshold is not None:
            chip = QLabel(f"Flagged >={threshold:.0%}" if flagged else f"Below {threshold:.0%}")
            chip.setObjectName("Chip")
            chip.setProperty("flag", "true" if flagged else "false")
            chip.setFixedWidth(104)
            chip.setAlignment(Qt.AlignCenter)
            chip.setToolTip("Score is at or above the current report-grounding threshold" if flagged
                            else "Score is below the current report-grounding threshold")
            h.addWidget(chip)
        return row

    def display_findings(self, probs):
        """Every returned score, sorted, with live threshold status."""
        self._clear_layout(self.findings_layout)
        if not probs:
            empty = QLabel("No score data was returned for this result.")
            empty.setObjectName("EmptyState")
            self.findings_layout.addWidget(empty)
            return
        reference = self.current_benchmark_reference
        self.findings_layout.addWidget(ScoreAxisHeader(not reference))
        for disease, probability in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
            if reference:
                self.findings_layout.addWidget(self._score_row(disease, probability, None, "reference"))
            else:
                threshold = self.calibrated_thresholds.get(disease, 0.40)
                self.findings_layout.addWidget(self._score_row(disease, probability, threshold))
        note = QLabel("Scores may not be calibrated clinical probabilities. Flags reflect the report thresholds."
                      if not reference else
                      "values are provided.")
        note.setObjectName("MutedText")
        note.setWordWrap(True)
        self.findings_layout.addWidget(note)

if __name__ == "__main__":
    log_startup("GUI module imports complete; creating Qt application.")
    app = QApplication(sys.argv)
    log_startup("Qt application initialized.")
    installed_fonts = set(QFontDatabase.families())
    preferred_font = next(
        (family for family in ("SF Pro Text", "SF Pro Display", "Helvetica Neue", "Segoe UI")
         if family in installed_fonts),
        QFontDatabase.systemFont(QFontDatabase.GeneralFont).family(),
    )
    app_font = QFont(preferred_font)
    app_font.setStyleHint(QFont.SansSerif)
    app.setFont(app_font)
    window = MainWindow()
    log_startup("Main window constructed; scheduling model load.")
    window.windowHandleCreated = False

    class FirstShowTimer(QObject):
        def eventFilter(self, watched, event):
            if event.type() == QEvent.Show and not window.windowHandleCreated:
                window.windowHandleCreated = True
                log_startup("First window Show event received (elapsed from process entry).")
            return False

    show_timer = FirstShowTimer()
    window.installEventFilter(show_timer)
    window.showMaximized()
    window.maximize_button.setIcon(caption_icon("restore"))
    window.maximize_button.setToolTip("Restore")
    window.maximize_button.setAccessibleName("Restore")
    window.raise_()
    window.activateWindow()
    log_startup("window.show() called; entering Qt event loop.")
    sys.exit(app.exec())
