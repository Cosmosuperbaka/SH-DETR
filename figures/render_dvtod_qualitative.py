#!/usr/bin/env python3
"""Fig.9 (DVTOD) qualitative figure, v3.

Same 4x2 grid / RGB-over-IR layout and the same method order as the v2 DVTOD
renderer, with two changes:
  * all geometry is multiplied by SCALE so the PNG carries roughly twice the
    pixels of the v2 export;
  * ground-truth boxes are drawn as green dashed rectangles on every panel and
    the legend gains a matching "Ground truth" entry.
The scene, thresholds and colour semantics are unchanged.
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
from rscdetr_paths import ROOT  # noqa: E402


B = ROOT / "compare/DVTOD_compare_20260910"

STEM = "1857"
OUT_PATH = ROOT / "out_qual/fig9_dvtod_1857_v3.png"

GT_PATH = ROOT / "datasets/RTDOD_HBB_3class/annotations/instances_val.json"
RGB_DIR = B / "data/DVTOD_3class_yolo/images/val"
IR_DIR = B / "data/DVTOD_3class_yolo/image/val"

PANELS = (
    ("RT-DETR concat", ROOT / "result/RTDOD_HBB_3class/perclass_concat_best/predictions.json"),
    ("C²Former", B / "results/c2former_raw/predictions.json"),
    ("RT-DETR (IR)", ROOT / "result/RTDOD_HBB_3class/perclass_ir_best/predictions.json"),
    ("SFFR", B / "runs/sffr_eval/sffr_final/best_predictions.json"),
    ("C2DFF-Net", B / "runs/c2dff_eval/c2dff_final/predictions.json"),
    ("DARFNet", B / "runs/darfnet_eval/darfnet_final/best_predictions.json"),
    ("CMFADet", B / "runs/cmfadet_eval/cmfadet_100e/predictions.json"),
    ("RSC-DETR", ROOT / "result/RTDOD_HBB_3class/perclass_sh_c2_best/predictions.json"),
)
OURS = "RSC-DETR"

CLASS_NAMES = {0: "Person", 1: "Car", 2: "Bicycle"}

SCORE_THRESHOLD = 0.50
MATCH_IOU = 0.50
RED = (230, 32, 32)
BLUE = (0, 132, 220)
YELLOW = (255, 214, 0)
GREEN = (0, 158, 71)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

SCALE = 2
MODALITY_WIDTH = 400 * SCALE
MODALITY_GAP = 8 * SCALE
TITLE_BAND = 46 * SCALE
TITLE_GAP = 8 * SCALE
CANVAS_MARGIN = 24 * SCALE
COLUMN_GAP = 14 * SCALE
ROW_GAP = 30 * SCALE
LEGEND_BAND = 56 * SCALE
BOX_WIDTH = 4 * SCALE
GT_BOX_WIDTH = 3 * SCALE
DASH = 10 * SCALE
DASH_GAP = 7 * SCALE
LABEL_FONT_SIZE = 20 * SCALE
MAX_LABELS_PER_PANEL = 3

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


def load_gt():
    payload = json.loads(GT_PATH.read_text())
    stem_by_id = {}
    gt = {}
    for im in payload["images"]:
        stem = Path(im["file_name"]).stem
        stem_by_id[int(im["id"])] = stem
        gt[stem] = []
    for a in payload["annotations"]:
        stem = stem_by_id[int(a["image_id"])]
        x, y, w, h = map(float, a["bbox"])
        gt[stem].append(((x, y, x + w, y + h), int(a["category_id"])))
    return gt, stem_by_id


def load_preds(path, stem_by_id, w=1920, h=1080):
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload)
    by_stem = {}
    for r in rows:
        iid = r["image_id"]
        stem = stem_by_id.get(int(iid)) if isinstance(iid, int) else str(iid)
        if stem is None:
            stem = str(iid)
        score = float(r.get("score", 0.0))
        if score < SCORE_THRESHOLD:
            continue
        x, y, bw, bh = map(float, r["bbox"])
        box = (max(0.0, x), max(0.0, y), min(x + bw, w), min(y + bh, h))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        by_stem.setdefault(stem, []).append((box, score, int(r["category_id"])))
    return by_stem


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "NimbusRoman-Bold" if bold else "NimbusRoman-Regular"
    path = FONT_DIR / f"{name}.otf"
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Nimbus Roman not found: {path}")


def draw_dashed_rect(draw, box, color, width, dash, gap) -> None:
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


def draw_gt(draw, gt_one, panel_w, panel_h) -> None:
    sx = panel_w / 1920.0
    sy = panel_h / 1080.0
    for box, _cat in gt_one:
        display_box = (
            min(panel_w - 1, max(0, round(box[0] * sx))),
            min(panel_h - 1, max(0, round(box[1] * sy))),
            min(panel_w - 1, max(0, round(box[2] * sx))),
            min(panel_h - 1, max(0, round(box[3] * sy))),
        )
        draw_dashed_rect(draw, display_box, GREEN, GT_BOX_WIDTH, DASH, DASH_GAP)


def draw_detections(draw, detections, gt, others, label_font, panel_w, panel_h, is_ours) -> None:
    rendered = []
    for box, score, cat in detections:
        sx = panel_w / 1920.0
        sy = panel_h / 1080.0
        display_box = (
            min(panel_w - 1, max(0, round(box[0] * sx))),
            min(panel_h - 1, max(0, round(box[1] * sy))),
            min(panel_w - 1, max(0, round(box[2] * sx))),
            min(panel_h - 1, max(0, round(box[3] * sy))),
        )
        matched = [g for g in gt if g[1] == cat and box_iou(box, g[0]) >= MATCH_IOU]
        if matched:
            target = max(matched, key=lambda g: box_iou(box, g[0]))
            shared = any(o[2] == target[1] and box_iou(o[0], target[0]) >= MATCH_IOU for o in others)
            color = YELLOW if is_ours and not shared else RED
        else:
            color = BLUE
        draw.rectangle(display_box, outline=color, width=BOX_WIDTH)
        rendered.append((display_box, score, cat, color))

    color_priority = {YELLOW: 0, BLUE: 1, RED: 2}
    selected = sorted(rendered, key=lambda item: (color_priority[item[3]], -item[1]))[:MAX_LABELS_PER_PANEL]
    occupied = []
    for display_box, score, cat, color in selected:
        label = f"{CLASS_NAMES.get(cat, f'class-{cat}')} {score:.2f}"
        tb = draw.textbbox((0, 0), label, font=label_font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        x1, y1, x2, y2 = display_box
        candidates = [(x1, y1 - text_h - 5 * SCALE), (x1, y2 + 4 * SCALE),
                      (x2 + 4 * SCALE, y1), (x1 - text_w - 4 * SCALE, y1)]
        chosen = None
        for x, y in candidates:
            x = min(max(2, x), max(2, panel_w - text_w - 3))
            y = min(max(2, y), max(2, panel_h - text_h - 3))
            rect = (x - 1, y - 1, x + text_w + 2, y + text_h + 2)
            overlaps = any(not (rect[2] <= old[0] or old[2] <= rect[0] or
                                rect[3] <= old[1] or old[3] <= rect[1]) for old in occupied)
            if not overlaps:
                chosen = (x, y, rect)
                break
        if chosen is None:
            continue
        label_x, label_y, rect = chosen
        draw.rectangle(rect, fill=color, outline=color, width=1)
        draw.text((label_x - tb[0], label_y - tb[1]), label,
                  fill=(TEXT if color == YELLOW else WHITE), font=label_font)
        occupied.append(rect)


def draw_legend(canvas, top, font) -> None:
    draw = ImageDraw.Draw(canvas)
    entries = ((RED, "Correct detection"), (BLUE, "False detection"),
               (GREEN, "Ground truth"), (YELLOW, "RSC-DETR-only true positive"))
    swatch, gap = 18 * SCALE, 28 * SCALE
    widths = []
    for _, label in entries:
        tb = draw.textbbox((0, 0), label, font=font)
        widths.append(swatch + 8 * SCALE + tb[2] - tb[0])
    left = (canvas.width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for color, label in entries:
        if color == GREEN:
            draw_dashed_rect(draw, (left, top, left + swatch, top + swatch),
                             color, GT_BOX_WIDTH, DASH // 2, DASH_GAP // 2)
        else:
            draw.rectangle((left, top, left + swatch, top + swatch), outline=color, width=3 * SCALE)
        tb = draw.textbbox((0, 0), label, font=font)
        draw.text((left + swatch + 8 * SCALE, top - tb[1] + (swatch - (tb[3] - tb[1])) // 2),
                  label, fill=TEXT, font=font)
        left += swatch + 8 * SCALE + tb[2] - tb[0] + gap


def main() -> None:
    gt, stem_by_id = load_gt()
    methods = {name: load_preds(path, stem_by_id) for name, path in PANELS}

    gt_one = gt[STEM]
    rgb = Image.open(RGB_DIR / f"{STEM}.jpg").convert("RGB")
    ir = Image.open(IR_DIR / f"{STEM}.jpg").convert("RGB")
    assert rgb.size == (1920, 1080), rgb.size
    assert ir.size == (1920, 1080), ir.size

    modality_height = round(MODALITY_WIDTH * 1080 / 1920)
    others = [d for name, _ in PANELS if name != OURS for d in methods[name].get(STEM, [])]

    panel_height = TITLE_BAND + modality_height + MODALITY_GAP + modality_height
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_WIDTH + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    font_title = load_font(28 * SCALE)
    font_label = load_font(LABEL_FONT_SIZE)
    font_legend = load_font(22 * SCALE)

    for index, (name, _path) in enumerate(PANELS):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        is_ours = name == OURS
        dets = methods[name].get(STEM, [])

        for offset, src in ((TITLE_BAND, rgb), (TITLE_BAND + MODALITY_GAP + modality_height, ir)):
            panel = src.resize((MODALITY_WIDTH, modality_height), Image.LANCZOS)
            draw = ImageDraw.Draw(panel)
            draw_gt(draw, gt_one, MODALITY_WIDTH, modality_height)
            draw_detections(draw, dets, gt_one, others, font_label,
                            MODALITY_WIDTH, modality_height, is_ours)
            canvas.paste(panel, (panel_left, image_top + offset))

        canvas_draw = ImageDraw.Draw(canvas)
        tb = canvas_draw.textbbox((0, 0), name, font=font_title)
        title_x = panel_left + (MODALITY_WIDTH - (tb[2] - tb[0])) // 2
        title_y = image_top + TITLE_BAND - TITLE_GAP - (tb[3] - tb[1]) - tb[1]
        canvas_draw.text((title_x, title_y), name, fill=TEXT, font=font_title)

    draw_legend(canvas, canvas_height - CANVAS_MARGIN - 24 * SCALE, font_legend)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, format="PNG", optimize=True)
    print(f"wrote {OUT_PATH}  {canvas_width}x{canvas_height}  stem={STEM}")
    print("counts=" + ", ".join(f"{n}:{len(methods[n].get(STEM, []))}" for n, _ in PANELS))
    print(f"gt={len(gt_one)}")


if __name__ == "__main__":
    main()
