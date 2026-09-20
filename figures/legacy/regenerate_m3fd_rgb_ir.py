#!/usr/bin/env python3
"""Regenerate the M3FD-LT20 2x4 qualitative comparison with RGB+IR stacked panels.

Mirrors the VEDAI version: RGB (top) and infrared (bottom) views per method,
Nimbus Roman text and compact panel size. No arrow annotation is used.
highlights a small pedestrian target which CMFC-DETR localizes with markedly
higher confidence than the RT-DETR concat baseline (0.86 vs 0.32).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/denglingjun/re-detr-last")
IMAGE_ID = 36
STEM = "00400"
VI_PATH = Path("/home/denglingjun/vedai_data/datasets/M3FD/raw/vi/00400.png")
IR_PATH = Path("/home/denglingjun/vedai_data/datasets/M3FD/raw/ir/00400.png")
GT_PATH = Path(
    "/home/denglingjun/vedai_data/datasets/M3FD/processed/lt20_seed42/"
    "annotations/instances_test.json"
)
OUTPUT_PATH = ROOT / "CMFC_DETR_unpacked/figures/qualitative_m3fd_rgb_ir_2x4.png"

METHOD_ORDER = (
    "CFT",
    "RT-DETR RGB",
    "RT-DETR concat",
    "C2DFF-Net",
    "YOLOv11-RGBT",
    "LCAFNet",
    "CLDyN+RT-DETR",
    "CMFC-DETR",
)

SCORE_THRESHOLD = 0.50
MATCH_IOU = 0.50
RED = (230, 32, 32)
BLUE = (0, 132, 220)
YELLOW = (255, 214, 0)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)
ARROW = (200, 30, 30)

MODALITY_WIDTH = 330
MODALITY_GAP = 8
TITLE_BAND = 46
TITLE_GAP = 8
CANVAS_MARGIN = 24
COLUMN_GAP = 14
ROW_GAP = 30
LEGEND_BAND = 46
BOX_WIDTH = 4
LABEL_FONT_SIZE = 22
MAX_LABELS_PER_PANEL = 4

# M3FD category IDs follow the dataset annotation order.
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
    "CMFC-DETR": (
        ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/predictions.json",
        0,
    ),
}

YOLO_SOURCES = {
    "CFT": Path(
        "/home/denglingjun/multispectral-object-detection/runs/M3FD-LT20/"
        "cft_x3_s42_300e_b8_1024_valbest_test_20260802/labels/00400.txt"
    ),
    "C2DFF-Net": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/"
    "C2DFF_best_test_s42_img1024_b8/labels/00400.txt",
    "YOLOv11-RGBT": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/"
    "YOLOv11_RGBT_best_test_s42_img1024_b8/labels/00400.txt",
    "LCAFNet": Path(
        "/home/denglingjun/LCAFNet/runs/M3FD-LT20/"
        "lcafnet_s42_b4_1024_plus100_uuid4_valbest_test_20260802/labels/00400.txt"
    ),
}

# No arrow annotation is used in the M3FD qualitative figure.
ARROW_TARGET = None
ARROW_NOTE = ""


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def clip_box(box: Sequence[float], w: int, h: int):
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    return (
        min(max(x1, 0.0), w - 1.0), min(max(y1, 0.0), h - 1.0),
        min(max(x2, 0.0), w - 1.0), min(max(y2, 0.0), h - 1.0),
    )


def load_coco_json(path: Path, offset: int, w: int, h: int):
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


def load_yolo_labels(path: Path, w: int, h: int):
    out = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        category, cx, cy, bw, bh, score = map(float, fields[:6])
        if score < SCORE_THRESHOLD:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        box = clip_box((x1, y1, x2, y2), w, h)
        if box is not None:
            out.append((box, score, int(category)))
    return out


def load_ground_truth(w: int, h: int):
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


def draw_detections(draw, detections, scale_x, scale_y, panel_w, panel_h,
                    is_cmfc, gt, others, label_font):
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
            shared = any(
                o[2] == target[1] and box_iou(o[0], target[0]) >= MATCH_IOU
                for o in others
            )
            color = YELLOW if is_cmfc and not shared else RED
        else:
            color = BLUE
        draw.rectangle(display_box, outline=color, width=BOX_WIDTH)
        rendered.append((display_box, score, cat, color))

    # Annotate only the most informative boxes. All boxes remain visible;
    # suppressing low-score duplicate labels keeps crowded scenes readable.
    color_priority = {YELLOW: 0, BLUE: 1, RED: 2}
    selected = sorted(rendered, key=lambda item: (color_priority[item[3]], -item[1]))[:MAX_LABELS_PER_PANEL]
    occupied = []
    for display_box, score, cat, color in selected:
        label = f"{CLASS_NAMES.get(cat, f'class-{cat}')} {score:.2f}"
        tb = draw.textbbox((0, 0), label, font=label_font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        x1, y1, x2, y2 = display_box
        candidates = [
            (x1, y1 - text_h - 5),
            (x1, y2 + 4),
            (x2 + 4, y1),
            (x1 - text_w - 4, y1),
        ]
        chosen = None
        for x, y in candidates:
            x = min(max(2, x), max(2, panel_w - text_w - 3))
            y = min(max(2, y), max(2, panel_h - text_h - 3))
            rect = (x - 1, y - 1, x + text_w + 2, y + text_h + 2)
            overlaps = any(not (rect[2] <= old[0] or old[2] <= rect[0] or
                                rect[3] <= old[1] or old[3] <= rect[1])
                           for old in occupied)
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


def draw_legend(canvas, top, font):
    draw = ImageDraw.Draw(canvas)
    entries = ((RED, "Correct detection"), (BLUE, "False detection"),
               (YELLOW, "SH-DETR-only true positive"))
    swatch = 18
    gap = 28
    widths = []
    for _, label in entries:
        tb = draw.textbbox((0, 0), label, font=font)
        widths.append(swatch + 8 + tb[2] - tb[0])
    left = (canvas.width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for color, label in entries:
        draw.rectangle((left, top, left + swatch, top + swatch), outline=color, width=3)
        tb = draw.textbbox((0, 0), label, font=font)
        draw.text((left + swatch + 8, top - tb[1] + (swatch - (tb[3] - tb[1])) // 2),
                  label, fill=TEXT, font=font)
        left += swatch + 8 + tb[2] - tb[0] + gap


def main() -> None:
    source = Image.open(VI_PATH).convert("RGB")
    ir = Image.open(IR_PATH).convert("RGB")
    assert source.size == (1024, 768) and ir.size == source.size
    modality_height = round(MODALITY_WIDTH * source.height / source.width)

    methods = {
        name: load_coco_json(path, offset, source.width, source.height)
        for name, (path, offset) in JSON_SOURCES.items()
    }
    for name, path in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(path, source.width, source.height)
    ground_truth = load_ground_truth(source.width, source.height)
    assert set(methods) == set(METHOD_ORDER)

    others = [
        det for name in METHOD_ORDER[:-1] for det in methods[name]
    ]
    scale_x = MODALITY_WIDTH / source.width
    scale_y = modality_height / source.height

    panel_height = TITLE_BAND + modality_height + MODALITY_GAP + modality_height
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_WIDTH + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font_title = load_font(28)
    font_label = load_font(LABEL_FONT_SIZE)
    font_legend = load_font(22)

    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        is_cmfc = name == "CMFC-DETR"

        rgb_panel = source.resize((MODALITY_WIDTH, modality_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rgb_panel)
        draw_detections(draw, methods[name], scale_x, scale_y, MODALITY_WIDTH, modality_height,
                        is_cmfc, ground_truth, others, font_label)
        canvas.paste(rgb_panel, (panel_left, image_top + TITLE_BAND))

        ir_panel = ir.resize((MODALITY_WIDTH, modality_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(ir_panel)
        draw_detections(draw, methods[name], scale_x, scale_y, MODALITY_WIDTH, modality_height,
                        is_cmfc, ground_truth, others, font_label)
        canvas.paste(ir_panel, (panel_left, image_top + TITLE_BAND + modality_height + MODALITY_GAP))

        canvas_draw = ImageDraw.Draw(canvas)
        # Baseline-anchored title: all names share one baseline, so glyphs stay
        # at the same height even when a name contains descenders (e.g. the
        # "y" in CLDyN+RT-DETR). The old ink-bbox bottom alignment lifted any
        # descender-bearing name by its descender depth.
        title_x = panel_left + MODALITY_WIDTH // 2
        title_baseline = image_top + TITLE_BAND - TITLE_GAP
        canvas_draw.text((title_x, title_baseline), name, fill=TEXT,
                         font=font_title, anchor="ms")

    draw_legend(canvas, canvas_height - CANVAS_MARGIN - 20, font_legend)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"wrote={OUTPUT_PATH}")
    print(f"canvas={canvas_width}x{canvas_height} image={IMAGE_ID} ({STEM}.png)")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))


if __name__ == "__main__":
    main()
