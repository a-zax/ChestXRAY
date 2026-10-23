"""Render an alternative, image-first radiology workspace concept."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "focused_workspace_prototype_v2.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 1600, 1000
im = Image.new("RGB", (W, H), "#f2f4f7")
d = ImageDraw.Draw(im)
FONT_DIR = Path("C:/Windows/Fonts")

def f(size, bold=False):
    path = FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()

def rr(box, radius, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)

def t(x, y, value, size=15, fill="#1d2b3a", bold=False):
    d.text((x, y), value, font=f(size, bold), fill=fill)

# Dark, restrained application chrome
d.rectangle((0, 0, W, 76), fill="#122235")
rr((26, 15, 72, 61), 12, "#2c75d8")
t(37, 26, "XR", 20, "#ffffff", True)
t(88, 15, "RADIOLOGY WORKSPACE", 12, "#a9bed4", True)
t(88, 34, "Chest X-ray review", 18, "#ffffff", True)
rr((1333, 19, 1483, 57), 9, "#1d344b", "#38516b")
t(1351, 30, "●  MODEL READY", 12, "#9de0bd", True)
rr((1493, 19, 1571, 57), 9, "#1d344b", "#38516b")
t(1510, 30, "Reset", 13, "#e8eff6", True)

# Compact workflow progress line
d.rectangle((0, 76, W, 123), fill="#ffffff")
steps = [(36, "01  IMAGE", True), (250, "02  CONTEXT", True), (480, "03  ANALYZE", False), (700, "04  REVIEW", False)]
for x, label, active in steps:
    t(x, 92, label, 12, "#1768ca" if active else "#8292a3", True)
d.rectangle((35, 119, 162, 122), fill="#2879df")
d.rectangle((162, 119, W, 122), fill="#e7ebef")

# Two balanced work surfaces
rr((24, 145, 948, 930), 16, "#ffffff", "#dce2e8")
rr((968, 145, 1576, 930), 16, "#ffffff", "#dce2e8")

# Image workspace
t(52, 168, "Image review", 21, "#172b40", True)
t(52, 198, "Check positioning and image quality before analysis.", 13, "#728195")
rr((48, 231, 924, 749), 12, "#0d1824")
image_path = ROOT / "demo_images" / "sample_normal_adult_pa.png"
with Image.open(image_path) as src:
    src = ImageOps.contain(src.convert("RGB"), (560, 478))
    im.paste(src, (206 + (560-src.width)//2, 250 + (478-src.height)//2))
d = ImageDraw.Draw(im)
# Viewer controls
rr((70, 694, 902, 730), 8, "#1b2a39")
t(88, 704, "−", 18, "#d7e1ea", True)
t(122, 706, "100%", 12, "#ffffff", True)
t(188, 704, "+", 18, "#d7e1ea", True)
t(802, 706, "FIT", 11, "#b5c6d6", True)
t(52, 769, "sample_normal_adult_pa.png", 13, "#52667a", True)
t(52, 791, "PA view  ·  224 × 224 preview", 12, "#8795a4")
rr((48, 820, 924, 883), 10, "#f7f9fb", "#dfe5eb")
t(68, 831, "CLINICAL CONTEXT", 11, "#62768b", True)
t(68, 852, "Chest pain; evaluate for acute cardiopulmonary disease.", 14, "#34485d")
rr((48, 895, 924, 922), 8, "#ffffff", "#cbd5df")
t(425, 900, "Choose another image", 13, "#256dc1", True)

# Report and findings workspace
t(996, 168, "Analysis", 21, "#172b40", True)
rr((1404, 164, 1548, 197), 9, "#eef2f5", "#dce3e9")
t(1422, 174, "AWAITING RUN", 11, "#62768b", True)
t(996, 204, "Run the analysis to review the report and scores.", 13, "#728195")
d.rectangle((996, 230, 1548, 231), fill="#e5e9ed")
t(996, 251, "REPORT", 11, "#64788c", True)
rr((996, 277, 1548, 510), 11, "#fbfcfd", "#e0e5ea")
t(1018, 301, "The report will appear here after analysis.", 15, "#8795a4")
rr((1436, 242, 1548, 272), 8, "#ffffff", "#d1dbe4")
t(1462, 250, "Copy", 12, "#256dc1", True)

t(996, 538, "PATHOLOGY SCORES", 11, "#64788c", True)
t(996, 561, "Scores are model estimates, not confirmed diagnoses.", 12, "#8795a4")
scores = [("Cardiomegaly", .18), ("Edema", .12), ("Lung opacity", .10), ("Pleural effusion", .06), ("Pneumothorax", .03)]
for idx, (label, value) in enumerate(scores):
    y = 596 + idx*55
    t(998, y, label, 13, "#34485d", True)
    rr((1171, y+5, 1481, y+14), 5, "#e8edf2")
    rr((1171, y+5, 1171+310*value, y+14), 5, "#6692c5")
    t(1492, y-1, f"{value:.0%}", 13, "#273d53", True)
    d.rectangle((996, y+31, 1548, y+32), fill="#edf0f3")

# Persistent action footer
d.rectangle((0, 950, W, H), fill="#ffffff")
d.rectangle((0, 950, W, 951), fill="#dce2e8")
t(28, 968, "For research use only  ·  Not a substitute for radiologist review", 12, "#758597")
rr((1376, 957, 1572, 994), 9, "#1f6fd1")
t(1428, 967, "Run analysis  →", 14, "#ffffff", True)

im.save(OUT, optimize=True)
print(OUT)
