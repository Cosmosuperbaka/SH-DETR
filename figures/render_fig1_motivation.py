#!/usr/bin/env python3
"""Fig. 1 (motivation) — cross-modal prediction discrepancy, rebuilt from real outputs.

Data provenance (all reproducible on the experiment server):

    ~/rsc-detr/infer_scene000004_single_modality.py

runs the *same* naive concat-merged RT-DETR baseline (fold03 checkpoint) three
times, zeroing one branch at a time:

    rgb_only : model(rgb, 0)     -> outputs/vedai_scene000004_single_modality/scene000004_rgb_only_predictions.json
    ir_only  : model(0, ir)      -> outputs/vedai_scene000004_single_modality/scene000004_ir_only_predictions.json
    rgb_ir   : model(rgb, ir)    -> outputs/vedai_scene000004_single_modality/scene000004_rgb_ir_predictions.json

The figure reports, for each ground-truth target, the highest-scoring detection
that shares the target's class and reaches IoU >= 0.5. No number is typed by
hand: every score printed on the canvas is read from those JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from rscdetr_paths import DATASETS, ROOT  # noqa: E402

IMAGE_ID = 4
RGB_PATH = DATASETS / "VEDAI/Vehicules1024/00000004_co.png"
IR_PATH = DATASETS / "VEDAI/Vehicules1024/00000004_ir.png"
PRED_DIR = ROOT / "outputs/vedai_scene000004_single_modality"

OUTPUT_PATH = ROOT / "out_qual/fig1_motivation_v2.png"

# GT boxes (xyxy) as recorded in the inference script for VEDAI scene 000004
# fold03. Name -> (category_id, box)
TARGETS = {
    "Camping car": (5, (322, 434, 346, 507)),
    "Car": (1, (272, 615, 312, 632)),
}

MATCH_IOU = 0.50
RED = (230, 32, 32)
GREEN = (0, 158, 71)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

CROP = (140, 340, 560, 760)      # 420x420 window covering both targets
PANEL = 700                       # rendered panel size (square)
TITLE_BAND = 64
MARGIN = 28
COLUMN_GAP = 36
LEGEND_BAND = 86
BOX_WIDTH = 5
DASH = 14
DASH_GAP = 9
FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "NimbusRoman-Bold" if bold else "NimbusRoman-Regular"
    path = FONT_DIR / f"{name}.otf"
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Nimbus Roman not found: {path}")


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def best_match(preds, cat, gbox):
    """Highest-scoring detection of class *cat* with IoU>=MATCH_IOU against gbox."""
    best = None
    for p in preds:
        if int(p["category_id"]) != cat:
            continue
        x, y, w, h = p["bbox"]
        box = (x, y, x + w, y + h)
        if box_iou(gbox, box) < MATCH_IOU:
            continue
        if best is None or p["score"] > best[0]:
            best = (float(p["score"]), box)
    return best


def load_preds(tag: str):
    path = PRED_DIR / f"scene000004_{tag}_predictions.json"
    return json.loads(path.read_text())


def to_panel(box):
    cx0, cy0, cx1, cy1 = CROP
    sx = PANEL / (cx1 - cx0)
    sy = PANEL / (cy1 - cy0)
    x1, y1, x2, y2 = box
    return (
        max(0, round((x1 - cx0) * sx)),
        max(0, round((y1 - cy0) * sy)),
        min(PANEL - 1, round((x2 - cx0) * sx)),
        min(PANEL - 1, round((y2 - cy0) * sy)),
    )


def dashed_rect(draw, box, color, width, dash, gap):
    x1, y1, x2, y2 = box
    if x2 - x1 < 3 or y2 - y1 < 3:
        return
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        x += dash + gap
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap


def main() -> None:
    rgb_only = load_preds("rgb_only")
    ir_only = load_preds("ir_only")

    results = {}
    for name, (cat, gbox) in TARGETS.items():
        rgb_hit = best_match(rgb_only, cat, gbox)
        ir_hit = best_match(ir_only, cat, gbox)
        results[name] = (cat, gbox, rgb_hit, ir_hit)
        print(f"{name:<12} RGB-only={rgb_hit[0]:.3f}  IR-only={ir_hit[0]:.3f}")

    canvas_w = 2 * MARGIN + 2 * PANEL + COLUMN_GAP
    canvas_h = MARGIN + TITLE_BAND + PANEL + LEGEND_BAND + MARGIN
    canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
    font_title = load_font(46)
    font_label = load_font(30)
    font_legend = load_font(32)

    views = (("RGB View", RGB_PATH, "rgb"), ("IR View", IR_PATH, "ir"))
    for col, (title, img_path, which) in enumerate(views):
        left = MARGIN + col * (PANEL + COLUMN_GAP)
        src = Image.open(img_path).convert("RGB")
        panel = src.crop(CROP).resize((PANEL, PANEL), Image.LANCZOS)
        draw = ImageDraw.Draw(panel)

        # ground truth first, predictions on top
        for name, (cat, gbox, rgb_hit, ir_hit) in results.items():
            dashed_rect(draw, to_panel(gbox), GREEN, BOX_WIDTH, DASH, DASH_GAP)
        for name, (cat, gbox, rgb_hit, ir_hit) in results.items():
            hit = rgb_hit if which == "rgb" else ir_hit
            if hit is not None:
                dashed_rect(draw, to_panel(hit[1]), RED, BOX_WIDTH, DASH, DASH_GAP)

        # labels, placed to the right of each target box
        occupied = []
        for name, (cat, gbox, rgb_hit, ir_hit) in results.items():
            hit = rgb_hit if which == "rgb" else ir_hit
            score = hit[0] if hit is not None else float("nan")
            label = f"{name} {score:.2f}"
            tb = draw.textbbox((0, 0), label, font=font_label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            bx1, by1, bx2, by2 = to_panel(gbox)
            candidates = [(bx2 + 12, by1), (bx2 + 12, by2 - th), (bx1, by2 + 8), (bx1, by1 - th - 8)]
            chosen = None
            for x, y in candidates:
                x = min(max(4, x), max(4, PANEL - tw - 5))
                y = min(max(4, y), max(4, PANEL - th - 5))
                rect = (x - 3, y - 3, x + tw + 4, y + th + 4)
                if all(rect[2] <= o[0] or o[2] <= rect[0] or rect[3] <= o[1] or o[3] <= rect[1]
                       for o in occupied):
                    chosen = (x, y, rect)
                    break
            if chosen is None:
                continue
            x, y, rect = chosen
            draw.rectangle(rect, fill=(255, 255, 255), outline=RED, width=3)
            draw.text((x - tb[0], y - tb[1]), label, fill=TEXT, font=font_label)
            occupied.append(rect)

        canvas.paste(panel, (left, MARGIN + TITLE_BAND))
        cd = ImageDraw.Draw(canvas)
        tb = cd.textbbox((0, 0), title, font=font_title)
        cd.text((left + (PANEL - (tb[2] - tb[0])) // 2, MARGIN + TITLE_BAND - 12 - (tb[3] - tb[1]) - tb[1]),
                title, fill=TEXT, font=font_title)

    # legend
    cd = ImageDraw.Draw(canvas)
    entries = ((GREEN, "Ground truth"), (RED, "Prediction"))
    swatch, gap = 40, 60
    widths = []
    for _, label in entries:
        tb = cd.textbbox((0, 0), label, font=font_legend)
        widths.append(swatch + 14 + (tb[2] - tb[0]))
    x = (canvas_w - (sum(widths) + gap * (len(entries) - 1))) // 2
    y = canvas_h - LEGEND_BAND + 20
    for (color, label), wdt in zip(entries, widths):
        if color == GREEN:
            dashed_rect(cd, (x, y, x + swatch, y + 30), color, 5, 10, 7)
        else:
            dashed_rect(cd, (x, y, x + swatch, y + 30), color, 5, 10, 7)
        tb = cd.textbbox((0, 0), label, font=font_legend)
        cd.text((x + swatch + 14, y + (30 - (tb[3] - tb[1])) // 2 - tb[1]), label,
                fill=TEXT, font=font_legend)
        x += wdt + gap

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"wrote={OUTPUT_PATH}  canvas={canvas_w}x{canvas_h}")
    for name, (cat, gbox, rgb_hit, ir_hit) in results.items():
        print(f"  {name}: rgb={rgb_hit[0]:.3f} ir={ir_hit[0]:.3f} "
              f"delta={rgb_hit[0] - ir_hit[0]:.3f}")


if __name__ == "__main__":
    main()
