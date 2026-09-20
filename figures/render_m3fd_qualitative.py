#!/usr/bin/env python3
"""Fig.8 (M3FD-LT20) qualitative figure, v3.

Same layout as the v2 renderer (4x2 grid, RGB over IR per method) with two
changes:
  * every geometry constant is multiplied by SCALE, roughly doubling the
    exported pixel count for a crisper print;
  * ground-truth boxes are drawn as green dashed rectangles and the legend
    gains a matching "Ground truth" entry.
Method list, thresholds and colour semantics are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS, CFT, LCAFNET, MSOD  # noqa: E402


IMAGE_ID = 36
STEM = "00400"
VI_PATH = DATASETS / "M3FD/raw/vi/00400.png"
IR_PATH = DATASETS / "M3FD/raw/ir/00400.png"
GT_PATH = Path(
    f"{DATASETS}/M3FD/processed/lt20_seed42/"
    "annotations/instances_test.json"
)
OUTPUT_PATH = ROOT / "out_qual/fig8_m3fd_00400_v3.png"

METHOD_ORDER = (
    "CFT",
    "RT-DETR RGB",
    "RT-DETR concat",
    "C2DFF-Net",
    "YOLOv11-RGBT",
    "LCAFNet",
    "CLDyN+RT-DETR",
    "SH-DETR",
)

SCORE_THRESHOLD = 0.50
MATCH_IOU = 0.50
RED = (230, 32, 32)
BLUE = (0, 132, 220)
YELLOW = (255, 214, 0)
GREEN = (0, 158, 71)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

SCALE = 2
MODALITY_WIDTH = 330 * SCALE
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
LABEL_FONT_SIZE = 22 * SCALE
MAX_LABELS_PER_PANEL = 4

CLASS_NAMES = {
    0: "People",
    1: "Car",
    2: "Bus",
    3: "Lamp",
    4: "Motorcycle",
    5: "Truck",
}

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")

JSON_SOURCES = {
    "RT-DETR RGB": (
        ROOT / "compare/M3FD-LT20/original-size-45e-trial/"
        "rtdetr-rgb-seed42-native-b8-45e/valbest_test_coco/predictions.json",
        0,
    ),
    "RT-DETR concat": (
        ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/baseline/predictions.json",
        0,
    ),
    "CLDyN+RT-DETR": (
        ROOT / "compare/CLDyN_M3FD-lt20/cldyn-1/eval_m3fd_map/"
        "cldyn-vfn-rtdetr-1/val_best_test_full_20260906/predictions.json",
        0,
    ),
    "SH-DETR": (
        ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/predictions.json",
        0,
    ),
}

YOLO_SOURCES = {
    "CFT": Path(
        f"{MSOD}/runs/M3FD-LT20/"
        "cft_x3_s42_300e_b8_1024_valbest_test_20260802/labels/00400.txt"
    ),
    "C2DFF-Net": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/"
    "C2DFF_best_test_s42_img1024_b8/labels/00400.txt",
    "YOLOv11-RGBT": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/"
    "YOLOv11_RGBT_best_test_s42_img1024_b8/labels/00400.txt",
    "LCAFNet": Path(
        f"{LCAFNET}/runs/M3FD-LT20/"
        "lcafnet_s42_b4_1024_plus100_uuid4_valbest_test_20260802/labels/00400.txt"
    ),
}

OURS = "SH-DETR"


def coco_xywh_to_xyxy(box):
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def clip_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    return (
        min(max(x1, 0.0), w - 1.0), min(max(y1, 0.0), h - 1.0),
        min(max(x2, 0.0), w - 1.0), min(max(y2, 0.0), h - 1.0),
    )


def load_coco_json(path, offset, w, h):
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
    out = []
    for row in rows:
        if int(row.get("image_id")) != IMAGE_ID:
            continue
        if float(row.get("score", 0.0)) < SCORE_THRESHOLD:
            continue
        box = clip_box(coco_xywh_to_xyxy(row["bbox"]), w, h)
        if box is not None:
            out.append((box, float(row["score"]), int(row["category_id"]) + offset))
    return out


def load_yolo_labels(path, w, h):
    out = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        category, cx, cy, bw, bh, score = map(float, fields[:6])
        if score < SCORE_THRESHOLD:
            continue
        box = clip_box(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), w, h)
        if box is not None:
            out.append((box, score, int(category)))
    return out


def load_ground_truth(w, h):
    payload = json.loads(GT_PATH.read_text())
    out = []
    for row in payload["annotations"]:
        if int(row["image_id"]) != IMAGE_ID:
            continue
        box = clip_box(coco_xywh_to_xyxy(row["bbox"]), w, h)
        if box is not None:
            out.append((box, int(row["category_id"])))
    return out


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
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


def draw_gt(draw, gt, scale_x, scale_y, panel_w, panel_h) -> None:
    for box, _cat in gt:
        display_box = (
            min(panel_w - 1, max(0, round(box[0] * scale_x))),
            min(panel_h - 1, max(0, round(box[1] * scale_y))),
            min(panel_w - 1, max(0, round(box[2] * scale_x))),
            min(panel_h - 1, max(0, round(box[3] * scale_y))),
        )
        draw_dashed_rect(draw, display_box, GREEN, GT_BOX_WIDTH, DASH, DASH_GAP)


def draw_detections(draw, detections, scale_x, scale_y, panel_w, panel_h, is_ours, gt, others, label_font):
    rendered = []
    for box, score, cat in detections:
        display_box = (
            min(panel_w - 1, max(0, round(box[0] * scale_x))),
            min(panel_h - 1, max(0, round(box[1] * scale_y))),
            min(panel_w - 1, max(0, round(box[2] * scale_x))),
            min(panel_h - 1, max(0, round(box[3] * scale_y))),
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
        text_color = TEXT if color == YELLOW else WHITE
        draw.text((label_x - tb[0], label_y - tb[1]), label, fill=text_color, font=label_font)
        occupied.append(rect)


def draw_legend(canvas, top, font) -> None:
    draw = ImageDraw.Draw(canvas)
    entries = ((RED, "Correct detection"), (BLUE, "False detection"),
               (GREEN, "Ground truth"), (YELLOW, "SH-DETR-only true positive"))
    swatch = 18 * SCALE
    gap = 28 * SCALE
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
    source = Image.open(VI_PATH).convert("RGB")
    ir = Image.open(IR_PATH).convert("RGB")
    assert source.size == (1024, 768) and ir.size == source.size
    modality_height = round(MODALITY_WIDTH * source.height / source.width)

    methods = {name: load_coco_json(path, offset, source.width, source.height)
               for name, (path, offset) in JSON_SOURCES.items()}
    for name, path in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(path, source.width, source.height)
    ground_truth = load_ground_truth(source.width, source.height)
    assert set(methods) == set(METHOD_ORDER)

    others = [det for name in METHOD_ORDER if name != OURS for det in methods[name]]
    scale_x = MODALITY_WIDTH / source.width
    scale_y = modality_height / source.height

    panel_height = TITLE_BAND + modality_height + MODALITY_GAP + modality_height
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_WIDTH + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font_title = load_font(28 * SCALE)
    font_label = load_font(LABEL_FONT_SIZE)
    font_legend = load_font(22 * SCALE)

    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        is_ours = name == OURS

        for offset, src in ((TITLE_BAND, source),
                            (TITLE_BAND + modality_height + MODALITY_GAP, ir)):
            panel = src.resize((MODALITY_WIDTH, modality_height), Image.LANCZOS)
            draw = ImageDraw.Draw(panel)
            draw_gt(draw, ground_truth, scale_x, scale_y, MODALITY_WIDTH, modality_height)
            draw_detections(draw, methods[name], scale_x, scale_y, MODALITY_WIDTH, modality_height,
                            is_ours, ground_truth, others, font_label)
            canvas.paste(panel, (panel_left, image_top + offset))

        canvas_draw = ImageDraw.Draw(canvas)
        title_x = panel_left + MODALITY_WIDTH // 2
        title_baseline = image_top + TITLE_BAND - TITLE_GAP
        canvas_draw.text((title_x, title_baseline), name, fill=TEXT, font=font_title, anchor="ms")

    draw_legend(canvas, canvas_height - CANVAS_MARGIN - 24 * SCALE, font_legend)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"wrote={OUTPUT_PATH}")
    print(f"canvas={canvas_width}x{canvas_height} image={IMAGE_ID} ({STEM}.png)")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))
    print(f"gt={len(ground_truth)}")


if __name__ == "__main__":
    main()
