"""Render a SaaS-style chest X-ray analysis workspace concept."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "saas_workspace_prototype.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 1600, 1000
im = Image.new("RGB", (W, H), "#f7f8fa")
d = ImageDraw.Draw(im)
FONT_DIR = Path("C:/Windows/Fonts")

def font(size, bold=False):
    try:
        return ImageFont.truetype(str(FONT_DIR / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)
    except OSError:
        return ImageFont.load_default()

def text(x, y, s, size=14, color="#344054", bold=False):
    d.text((x, y), s, font=font(size, bold), fill=color)

def card(box, radius=12, fill="#ffffff", stroke="#e4e7ec"):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=1)

def pill(box, label, bg, fg, size=12):
    d.rounded_rectangle(box, radius=8, fill=bg)
    text(box[0]+12, box[1]+7, label, size, fg, True)

# SaaS navigation rail
d.rectangle((0, 0, 236, H), fill="#ffffff")
d.rectangle((235, 0, 236, H), fill="#eaecf0")
d.rounded_rectangle((24, 22, 62, 60), radius=11, fill="#155eef")
text(32, 29, "XR", 18, "#ffffff", True)
text(73, 23, "Cognitive", 17, "#101828", True)
text(73, 45, "RADIOLOGY ASSISTANT", 8, "#667085", True)
text(26, 91, "WORKSPACE", 10, "#98a2b3", True)

nav_items = ["New analysis"]
for idx, label in enumerate(nav_items):
    y = 111 + idx*45
    d.rounded_rectangle((14, y-5, 220, y+34), radius=8, fill="#eff4ff")
    d.rounded_rectangle((29, y+6, 43, y+20), radius=3, outline="#175cd3", width=2)
    d.rectangle((33, y+10, 39, y+16), fill="#175cd3")
    text(58, y+5, label, 14, "#175cd3", True)
text(26, 188, "ONE WORKSPACE · ONE TASK", 9, "#98a2b3", True)
card((16, 224, 220, 318), 10, "#f8f9fb", "#eaecf0")
text(30, 241, "QUICK TIP", 10, "#98a2b3", True)
text(30, 263, "Open an image", 13, "#344054", True)
text(30, 286, "Press Ctrl + O to browse", 11, "#667085")

# Top bar
d.rectangle((236, 0, W, 72), fill="#ffffff")
d.rectangle((236, 71, W, 72), fill="#eaecf0")
text(270, 27, "Workspace", 13, "#667085")
text(352, 27, "/", 13, "#98a2b3")
text(372, 27, "New analysis", 13, "#344054", True)
pill((1369, 19, 1538, 55), "MODEL READY", "#ecfdf3", "#027a48", 11)

# Page heading and main action
text(270, 104, "New analysis", 27, "#101828", True)
text(270, 143, "Upload a chest X-ray and review the generated findings.", 14, "#667085")

# Workflow steps in subtle SaaS stepper
card((270, 174, 1548, 224), 10)
steps = [(294, "1", "Image"), (570, "2", "Context"), (846, "3", "Analysis"), (1122, "4", "Review")]
for i, (x, number, label) in enumerate(steps):
    fill = "#12b76a" if i == 0 else ("#155eef" if i == 1 else "#f2f4f7")
    fg = "#ffffff" if i < 2 else "#667085"
    d.ellipse((x, 185, x+28, 213), fill=fill)
    text(x+9, 189, number, 12, fg, True)
    text(x+39, 190, label, 13, "#344054" if i < 2 else "#667085", i < 2)
    if i < 3:
        d.rectangle((x+125, 198, x+238, 200), fill="#eaecf0")

# Main panels
card((270, 244, 875, 887), 12)
card((895, 244, 1548, 887), 12)

# Study form / image card
text(294, 267, "Study image", 17, "#101828", True)
text(294, 294, "PNG or JPEG · frontal chest view", 12, "#667085")
d.rounded_rectangle((294, 325, 851, 650), radius=10, fill="#101828")
sample_path = ROOT / "demo_images" / "sample_normal_adult_pa.png"
with Image.open(sample_path) as src:
    dimensions = src.size
    src = ImageOps.contain(src.convert("RGB"), (520, 305))
    im.paste(src, (294 + (557-src.width)//2, 332 + (310-src.height)//2))
d = ImageDraw.Draw(im)
text(294, 663, sample_path.name, 12, "#475467", True)
text(294, 684, f"{sample_path.stat().st_size / (1024*1024):.1f} MB  ·  {dimensions[0]} × {dimensions[1]} px", 11, "#98a2b3")
card((294, 711, 851, 767), 9, "#ffffff", "#d0d5dd")
text(502, 729, "Choose image", 13, "#175cd3", True)
text(294, 789, "Clinical context", 14, "#344054", True)
card((294, 817, 851, 867), 8, "#ffffff", "#d0d5dd")
text(309, 834, "Chest pain; evaluate for acute cardiopulmonary disease.", 12, "#475467")

# Output card
text(921, 267, "Analysis output", 17, "#101828", True)
text(921, 294, "Report and model scores", 12, "#667085")
card((1394, 263, 1522, 294), 8, "#eff8ff", "#d1e9ff")
text(1410, 272, "READY TO RUN", 10, "#175cd3", True)
d.rectangle((921, 313, 1522, 314), fill="#eaecf0")
text(921, 333, "Generated report", 13, "#344054", True)
card((921, 360, 1522, 522), 9, "#fcfcfd", "#eaecf0")
text(941, 383, "Run the analysis to generate a report.", 13, "#98a2b3")
card((1412, 327, 1522, 355), 7, "#ffffff", "#d0d5dd")
text(1439, 334, "Copy", 11, "#175cd3", True)
text(921, 546, "Pathology scores", 13, "#344054", True)
text(921, 568, "Scores appear here after analysis.", 12, "#98a2b3")
card((921, 604, 1522, 730), 9, "#fcfcfd", "#eaecf0")
text(1085, 650, "No scores yet", 14, "#667085", True)
text(1010, 676, "Run analysis to populate pathology scores.", 12, "#98a2b3")

# Sticky footer CTA
d.rectangle((236, 912, W, 913), fill="#eaecf0")
d.rectangle((236, 913, W, H), fill="#ffffff")
text(270, 947, "Ready when you are", 12, "#667085")
d.rounded_rectangle((1362, 929, 1548, 979), radius=9, fill="#155eef")
text(1391, 945, "Run analysis   →", 14, "#ffffff", True)

im.save(OUT, optimize=True)
print(OUT)
