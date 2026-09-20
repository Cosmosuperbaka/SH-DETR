#!/usr/bin/env python3
"""Fig.7 (VEDAI) qualitative figure, v3.

Changes relative to the v2 crop renderer:
  * all geometry is scaled by SCALE so the exported PNG carries ~2x the pixels
    (crisper when the figure is placed at 0.76\\textwidth in the paper);
  * ground-truth boxes are drawn as green dashed rectangles on every panel;
  * the legend gains a "Ground truth" entry to match.
Everything else (methods, thresholds, crop window, colour semantics) is
unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS, CFT, LCAFNET  # noqa: E402


IMAGE_ID = 1033
IMAGE_KEY = "00001033_co"
IMAGE_PATH = DATASETS / "VEDAI/Vehicules1024/00001033_co.png"
IR_IMAGE_PATH = DATASETS / "VEDAI/Vehicules1024/00001033_ir.png"
GT_PATH = DATASETS / "VEDAI/annotations/vedai_fold01_test_class8.json"
OUTPUT_PATH = ROOT / "out_qual/fig7_vedai_1033_v3.png"
RSVDET_JSON = ROOT / "outputs/compare_vedai_icafusion_rsvdet/rsvdet/eval_best/predictions.json"

CROP = (348, 637, 648, 937)

METHOD_ORDER = (
    "CFT",
    "C2DFF-Net",
    "RT-DETR RGB",
    "RT-DETR concat",
    "RSVDet",
    "YOLOv11-RGBT",
    "LCAFNet",
    "SH-DETR",
)

SCORE_THRESHOLD = 0.70
RSVDET_THRESHOLD = 0.70
MATCH_IOU = 0.50
RED = (230, 32, 32)
YELLOW = (255, 214, 0)
GREEN = (0, 158, 71)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

SCALE = 2
MODALITY_SIZE = 240 * SCALE
MODALITY_GAP = 8 * SCALE
TITLE_BAND = 44 * SCALE
TITLE_GAP = 6 * SCALE
CANVAS_MARGIN = 20 * SCALE
COLUMN_GAP = 12 * SCALE
ROW_GAP = 24 * SCALE
LEGEND_BAND = 64 * SCALE
BOX_WIDTH = 4 * SCALE
GT_BOX_WIDTH = 3 * SCALE
DASH = 10 * SCALE
DASH_GAP = 7 * SCALE

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int


JSON_SOURCES = {
    "CFT": (
        CFT / "runs/VEDAI/CFT-26-fold1_valbest_test_20260802/best_predictions.json",
        1,
    ),
    "C2DFF-Net": (
        ROOT / "compare/C2DFF_VEDAI/runs/c2dff_s3407_uuid2_valbest_test_20260802/predictions.json",
        1,
    ),
    "RT-DETR RGB": (
        ROOT / "outputs/vedai_direct_test_unified/rgb/seed_3407/predictions.json",
        0,
    ),
    "RT-DETR concat": (
        ROOT / "outputs/vedai_direct_test_unified/baseline/seed_3407/predictions.json",
        0,
    ),
    "SH-DETR": (
        ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json",
        0,
    ),
}

YOLO_SOURCES = {
    "YOLOv11-RGBT": ROOT
    / "compare/YOLOv11_RGBT/runs/yolov11_rgbt_vedai_s3407_uuid2_valbest_test_20260802/labels/00001033_co.txt",
    "LCAFNet": Path(
        f"{LCAFNET}/runs/VEDAI/"
        "lcafnet_s3407_uuid_valbest_test_20260802/labels/00001033_co.txt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--rsvdet-json", type=Path)
    return parser.parse_args()


def same_image(value: object) -> bool:
    if isinstance(value, int):
        return value == IMAGE_ID
    text = str(value)
    return text in {str(IMAGE_ID), IMAGE_KEY, f"{IMAGE_KEY}.png"}


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def yolo_xywhn_to_xyxy(box, image_width, image_height):
    center_x, center_y, width, height = map(float, box)
    center_x *= image_width
    center_y *= image_height
    width *= image_width
    height *= image_height
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def clip_box(box, image_width, image_height):
    values = tuple(map(float, box))
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        return None
    x1 = min(max(x1, 0.0), image_width - 1.0)
    y1 = min(max(y1, 0.0), image_height - 1.0)
    x2 = min(max(x2, 0.0), image_width - 1.0)
    y2 = min(max(y2, 0.0), image_height - 1.0)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def load_coco_json(path, category_offset, image_width, image_height):
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
    detections = []
    for row in rows:
        if not same_image(row.get("image_id")) or float(row.get("score", 0.0)) < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is None:
            continue
        detections.append(Detection(xyxy, float(row["score"]), int(row["category_id"]) + category_offset))
    return detections


def load_yolo_labels(path, image_width, image_height):
    detections = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        category, center_x, center_y, width, height, score = map(float, fields[:6])
        if score < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(
            yolo_xywhn_to_xyxy((center_x, center_y, width, height), image_width, image_height),
            image_width,
            image_height,
        )
        if xyxy is not None:
            detections.append(Detection(xyxy, score, int(category) + 1))
    return detections


def load_rsvdet_json(path, image_width, image_height):
    detections = []
    for row in json.loads(path.read_text()):
        if not same_image(row.get("image_id")):
            continue
        if float(row["score"]) < RSVDET_THRESHOLD:
            continue
        xyxy = clip_box(row["xyxy"], image_width, image_height) if "xyxy" in row else clip_box(
            coco_xywh_to_xyxy(row["bbox"]), image_width, image_height
        )
        if xyxy is not None:
            detections.append(Detection(xyxy, float(row["score"]), int(row["category_id"])))
    return detections


def load_ground_truth(image_width, image_height):
    payload = json.loads(GT_PATH.read_text())
    detections = []
    for row in payload["annotations"]:
        if not same_image(row["image_id"]):
            continue
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is not None:
            detections.append(Detection(xyxy, 1.0, int(row["category_id"])))
    return detections


def box_iou(left, right) -> float:
    x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
    x2 = min(left[2], right[2]); y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def cmfc_color(detection, ground_truth, other_methods):
    matching_gt = [
        target for target in ground_truth
        if target.category_id == detection.category_id and box_iou(target.xyxy, detection.xyxy) >= MATCH_IOU
    ]
    if not matching_gt:
        return RED
    target = max(matching_gt, key=lambda item: box_iou(item.xyxy, detection.xyxy))
    shared = any(
        other.category_id == target.category_id and box_iou(other.xyxy, target.xyxy) >= MATCH_IOU
        for other in other_methods
    )
    return RED if shared else YELLOW


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "NimbusRoman-Bold" if bold else "NimbusRoman-Regular"
    path = FONT_DIR / f"{name}.otf"
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Nimbus Roman not found: {path}")


def to_display(box) -> tuple[int, int, int, int]:
    cx0, cy0, cx1, cy1 = CROP
    sx = MODALITY_SIZE / (cx1 - cx0)
    sy = MODALITY_SIZE / (cy1 - cy0)
    x1, y1, x2, y2 = box
    return (
        min(MODALITY_SIZE - 1, max(0, round((x1 - cx0) * sx))),
        min(MODALITY_SIZE - 1, max(0, round((y1 - cy0) * sy))),
        min(MODALITY_SIZE - 1, max(0, round((x2 - cx0) * sx))),
        min(MODALITY_SIZE - 1, max(0, round((y2 - cy0) * sy))),
    )


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box, color, width, dash, gap) -> None:
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


def draw_gt(draw: ImageDraw.ImageDraw, ground_truth: Sequence[Detection]) -> None:
    for target in ground_truth:
        draw_dashed_rect(draw, to_display(target.xyxy), GREEN, GT_BOX_WIDTH, DASH, DASH_GAP)


def draw_detections(draw: ImageDraw.ImageDraw, detections, color_for) -> None:
    for detection in detections:
        display_box = to_display(detection.xyxy)
        if display_box[2] - display_box[0] < 2 or display_box[3] - display_box[1] < 2:
            continue
        draw.rectangle(display_box, outline=color_for(detection), width=BOX_WIDTH)


def main() -> None:
    args = parse_args()
    source = Image.open(IMAGE_PATH).convert("RGB")
    ir = Image.open(IR_IMAGE_PATH).convert("RGB")
    methods = {
        name: load_coco_json(path, offset, source.width, source.height)
        for name, (path, offset) in JSON_SOURCES.items()
    }
    for name, path in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(path, source.width, source.height)
    methods["RSVDet"] = load_rsvdet_json(args.rsvdet_json or RSVDET_JSON, source.width, source.height)
    ground_truth = load_ground_truth(source.width, source.height)
    assert set(methods) == set(METHOD_ORDER)

    other_methods = [d for name in METHOD_ORDER[:-1] for d in methods[name]]

    panel_height = TITLE_BAND + MODALITY_SIZE + MODALITY_GAP + MODALITY_SIZE
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_SIZE + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font_title = load_font(24 * SCALE)
    font_label = load_font(20 * SCALE)

    def color_for_cmfc(detection: Detection):
        return cmfc_color(detection, ground_truth, other_methods)

    def color_red(_: Detection):
        return RED

    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_SIZE + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        color_for = color_for_cmfc if name == "SH-DETR" else color_red

        for offset, src in (
            (TITLE_BAND, source),
            (TITLE_BAND + MODALITY_SIZE + MODALITY_GAP, ir),
        ):
            panel = src.crop(CROP).resize((MODALITY_SIZE, MODALITY_SIZE), Image.LANCZOS)
            draw = ImageDraw.Draw(panel)
            draw_gt(draw, ground_truth)
            draw_detections(draw, methods[name], color_for)
            canvas.paste(panel, (panel_left, image_top + offset))

        canvas_draw = ImageDraw.Draw(canvas)
        text_box = canvas_draw.textbbox((0, 0), name, font=font_title)
        title_x = panel_left + (MODALITY_SIZE - (text_box[2] - text_box[0])) // 2
        title_y = image_top + TITLE_BAND - TITLE_GAP - (text_box[3] - text_box[1]) - text_box[1]
        canvas_draw.text((title_x, title_y), name, fill=TEXT, font=font_title)

    canvas_draw = ImageDraw.Draw(canvas)
    legend_top = canvas_height - LEGEND_BAND + 22 * SCALE
    entries = ((RED, "Detection"), (GREEN, "Ground truth"), (YELLOW, "SH-DETR-only true positive"))
    swatch = 24 * SCALE
    gap = 40 * SCALE
    widths = []
    for _, label in entries:
        tb = canvas_draw.textbbox((0, 0), label, font=font_label)
        widths.append(swatch + 8 * SCALE + (tb[2] - tb[0]))
    legend_x = (canvas_width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for (color, label), width in zip(entries, widths):
        if color == GREEN:
            draw_dashed_rect(canvas_draw, (legend_x, legend_top, legend_x + swatch, legend_top + swatch),
                             color, GT_BOX_WIDTH, DASH // 2, DASH_GAP // 2)
        else:
            canvas_draw.rectangle([legend_x, legend_top, legend_x + swatch, legend_top + swatch],
                                  outline=color, width=3 * SCALE)
        tb = canvas_draw.textbbox((0, 0), label, font=font_label)
        th = tb[3] - tb[1]
        canvas_draw.text((legend_x + swatch + 8 * SCALE, legend_top + (swatch - th) // 2 - tb[1]),
                         label, fill=TEXT, font=font_label)
        legend_x += width + gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(f"wrote={args.output}")
    print(f"canvas={canvas_width}x{canvas_height} crop={CROP} thr={SCORE_THRESHOLD}")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))
    print(f"gt={len(ground_truth)}")
    yellow = [d for d in methods["SH-DETR"] if color_for_cmfc(d) == YELLOW]
    print(f"cmfc_unique(yellow)={len(yellow)}")


if __name__ == "__main__":
    sys.exit(main())
