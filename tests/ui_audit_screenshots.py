"""Headless UI audit harness: drives the real MainWindow through each state and grabs screenshots."""
import sys, os, types, importlib.util, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
APP = os.environ.get("APP_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---- stub heavy ML deps (UI only) ----
torch = types.ModuleType("torch")
class _Dev:
    type = "cpu"
torch.device = lambda *a, **k: _Dev()
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
torch.load = lambda *a, **k: {}
torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
sys.modules["torch"] = torch
tf = types.ModuleType("transformers"); tf.AutoTokenizer = object; sys.modules["transformers"] = tf
m = types.ModuleType("models"); sys.modules["models"] = m
mc = types.ModuleType("models.cognitive_model"); mc.CognitiveReportGenerator = object; sys.modules["models.cognitive_model"] = mc
md = types.ModuleType("models.dataset"); md.get_transforms = lambda **k: None; sys.modules["models.dataset"] = md

spec = importlib.util.spec_from_file_location("gui_app", f"{APP}/scripts/gui_app.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

from PySide6.QtWidgets import QApplication, QMessageBox, QComboBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
app = QApplication([])
app.setFont(QFont("Inter"))

# silence modal dialogs but record them
dialogs = []
for n in ("warning", "critical", "information"):
    setattr(QMessageBox, n, staticmethod(lambda *a, _n=n, **k: dialogs.append((_n, a[1:3]))))

def pump(ms=120):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)

win = g.MainWindow()
W, H = int(os.environ.get("W", 1440)), int(os.environ.get("H", 900))
win.resize(W, H); win.show(); pump(300)

def shot(name):
    pump(750)
    win.grab().save(f"{OUT}/{name}.png")
    print("saved", name)

PROBS = {"No Finding": .03, "Enlarged Cardiomediastinum": .74, "Cardiomegaly": .88, "Lung Opacity": .22,
         "Lung Lesion": .05, "Edema": .58, "Consolidation": .12, "Pneumonia": .08, "Atelectasis": .31,
         "Pneumothorax": .02, "Pleural Effusion": .42, "Pleural Other": .04, "Fracture": .03, "Support Devices": .15}

scenario = os.environ.get("SCENARIO", "all")

shot("01_launch_empty")
# model loaded state
win.on_model_loaded(object(), None, None, _Dev())
shot("02_model_ready_no_image")
# image input is exercised via Browse/drag-and-drop in the application.
# running
win.generate_btn.setEnabled(True)
class FakeInf:
    def __init__(self,*a,**k): pass
g.InferenceThread = lambda *a, **k: types.SimpleNamespace(
    result_ready=types.SimpleNamespace(connect=lambda f: None), error=types.SimpleNamespace(connect=lambda f: None), start=lambda: None)
win.start_generation(); pump(700); shot("04_analysis_running")
raw = "INDINGS: heart is enlarged. lungs are clear."
rep = g.ClinicalGrounder.refine(raw, PROBS, win.calibrated_thresholds)
win.on_fin(rep, PROBS, raw); shot("05_results_model_output")
# Results now use a single output panel without separate reference/calibration tabs.
# automatic reference output for a provided sample image
win.reset_ui(); win._show_image(f"{APP}/demo_images/sample_normal_adult_pa.png")
shot("10_sample_reference_output")
# non-sample image continues through model inference
win.reset_ui(); win.current_image_path = f"{APP}/data/sample_images/sample_pediatric.jpg"
win.on_model_loaded(object(), None, None, _Dev())
# error state
win.reset_ui()
win.on_inf_error("CUDA out of memory (simulated)"); shot("11_error_state")
# reset
win.reset_ui(); shot("12_after_reset")
# long text
win.indication_input.setPlainText("Clinical indication: " + "persistent cough and fever for three weeks with weight loss, ")
shot("13_long_indication")
# small window
win.resize(980, 680)
win.on_fin(rep, PROBS, raw); shot("14_min_size_980x680")
win.resize(1366, 768); shot("15_laptop_1366x768")
print("dialogs:", dialogs)
