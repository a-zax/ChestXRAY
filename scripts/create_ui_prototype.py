from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "focused_workspace_prototype.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 1600, 1000
im = Image.new("RGB", (W, H), "#f3f6fb")
d = ImageDraw.Draw(im)

FONT_DIR = Path("C:/Windows/Fonts")
def font(size, bold=False):
    path = FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()

def rounded(box, radius, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def txt(x, y, text, size=16, color="#17324f", bold=False):
    d.text((x, y), text, font=font(size, bold), fill=color)

def line(points, fill="#e1e8f0", width=1):
    d.line(points, fill=fill, width=width)

# Header
rounded((24, 22, 1576, 104), 16, "#ffffff", "#dce5ef")
rounded((44, 39, 98, 91), 13, "#174ea6")
txt(57, 54, "XR", 22, "#ffffff", True)
txt(116, 37, "Cognitive Radiology Assistant", 24, "#102a43", True)
txt(116, 69, "Chest X-ray analysis workspace", 14, "#62758a")
rounded((1262, 41, 1430, 83), 10, "#eef5ff", "#d7e5f7")
txt(1280, 53, "MODEL READY", 13, "#175cd3", True)
rounded((1450, 41, 1552, 83), 10, "#ffffff", "#cbd7e4")
txt(1473, 53, "Reset", 15, "#27435f", True)

# Workflow strip
rounded((24, 120, 1576, 166), 12, "#eaf1fb", "#d9e5f4")
steps = [(50, "1  Image", True), (260, "2  Context", True), (475, "3  Analyze", False), (700, "4  Review", False)]
for x, label, active in steps:
    if active and label.startswith("1"):
        rounded((x-12, 129, x+112, 157), 9, "#175cd3")
        txt(x, 135, label, 14, "#ffffff", True)
    else:
        txt(x, 135, label, 14, "#175cd3" if active else "#667b91", active)
for x in (205, 420, 645):
    txt(x, 135, "›", 18, "#9aabbd")

# Main cards
rounded((24, 184, 536, 930), 18, "#ffffff", "#dce5ef")
rounded((552, 184, 1576, 930), 18, "#ffffff", "#dce5ef")

# Left: image and context
txt(50, 207, "Image", 22, "#102a43", True)
txt(50, 239, "Add a chest X-ray to begin.", 14, "#657a90")
rounded((48, 278, 512, 650), 14, "#0d1926")
xray_path = ROOT / "demo_images" / "sample_normal_adult_pa.png"
with Image.open(xray_path) as source:
    source = ImageOps.contain(source.convert("RGB"), (340, 340))
    im.paste(source, (110 + (340-source.width)//2, 294 + (340-source.height)//2))
d = ImageDraw.Draw(im)
txt(50, 666, "sample_normal_adult_pa.png", 13, "#62758a")
rounded((48, 692, 512, 740), 10, "#ffffff", "#b7c8db")
txt(205, 706, "Choose image", 15, "#175cd3", True)
txt(50, 766, "Clinical context", 16, "#102a43", True)
rounded((48, 794, 512, 876), 10, "#ffffff", "#b7c8db")
txt(64, 811, "Clinical indication: chest pain,", 14, "#344e68")
txt(64, 836, "evaluate for acute cardiopulmonary disease.", 14, "#344e68")

# Right: output panel
txt(580, 207, "Analysis results", 22, "#102a43", True)
txt(580, 239, "Report and pathology scores", 14, "#657a90")
rounded((1417, 202, 1548, 237), 9, "#eef2f7", "#d8e1eb")
txt(1432, 212, "READY", 12, "#50677e", True)
line([(576, 262), (1552, 262)], "#e3eaf2")
txt(580, 283, "Report", 16, "#102a43", True)
rounded((1435, 275, 1548, 310), 9, "#ffffff", "#cbd7e4")
txt(1461, 285, "Copy", 13, "#175cd3", True)
rounded((580, 324, 1548, 557), 12, "#fbfcfe", "#d7e1ec")
txt(606, 350, "Your report will appear here after analysis.", 16, "#8193a5")
txt(580, 585, "Pathology scores", 16, "#102a43", True)
txt(1260, 588, "Model scores are not confirmed diagnoses", 12, "#6b7f93")

rows = [
    ("Cardiomegaly", 0.18, "18%"),
    ("Edema", 0.12, "12%"),
    ("Lung opacity", 0.10, "10%"),
    ("Pleural effusion", 0.06, "6%"),
    ("Pneumothorax", 0.03, "3%"),
]
for i, (name, value, label) in enumerate(rows):
    y = 624 + i*55
    rounded((580, y, 1548, y+43), 9, "#f8faff", "#e4ebf3")
    txt(598, y+12, name, 14, "#344e68", True)
    rounded((824, y+17, 1452, y+27), 5, "#e3eaf2")
    rounded((824, y+17, 824+628*value, y+27), 5, "#6d91c4")
    txt(1471, y+10, label, 14, "#243b53", True)

# Bottom status and action
line([(24, 950), (1576, 950)], "#dce5ef")
txt(34, 965, "For research use only · Not a substitute for radiologist review", 12, "#64798f")
rounded((1395, 958, 1552, 994), 10, "#175cd3")
txt(1430, 967, "Run analysis", 14, "#ffffff", True)

im.save(OUT, optimize=True)
print(OUT)
