#!/usr/bin/env python3
"""Render the M3FD-LT20 2x4 qualitative comparison (Fig8) for scene 36 (00400.png)
with RGB+IR stacked panels and a genuine CMFC-only true positive.

* each method panel shows the visible (vi) view (top) and the infrared (ir) view
  (bottom) of M3FD scene 00400, making the multimodal input explicit;
* detection/matching threshold is 0.50; the car target at (284,432,345,479) is a
  NATURAL CMFC-only true positive: CMFC-DETR matches it at score 0.889 while every
  other method has zero matching detection (best_match_score = 0 for all 7 others),
  so it is drawn in yellow without any threshold adjustment;
* red = correct detection, cyan = false positive, yellow = CMFC-only true positive
  (matching the qualitative-figure legend used across the paper);
* text is set in Nimbus Roman (Times-metric compatible).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS, CFT, LCAFNET, MSOD  # noqa: E402


IMAGE_ID = 36
IMAGE_KEY = "00400"
IMAGE_PATH = DATASETS / "M3FD/raw/vi/00400.png"
IR_IMAGE_PATH = DATASETS / "M3FD/raw/ir/00400.png"
GT_PATH = Path(
    f"{DATASETS}/M3FD/processed/lt20_seed42/annotations/instances_test.json"
)
OUTPUT_PATH = ROOT / "CMFC_DETR_unpacked/figures/qualitative_m3fd_rgb_ir_36_2x4.png"

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
YELLOW = (255, 214, 0)
CYAN = (0, 200, 220)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)
ARROW = (200, 30, 30)

MODALITY_WIDTH = 330          # RGB/IR thumbnail width  (1024 -> 330)
MODALITY_HEIGHT = 248         # RGB/IR thumbnail height (768 -> 248)
MODALITY_GAP = 8
TITLE_BAND = 46
TITLE_GAP = 8
CANVAS_MARGIN = 24
COLUMN_GAP = 14
ROW_GAP = 30
NOTE_BAND = 120               # bottom band: legend + arrow note
BOX_WIDTH = 4

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int


JSON_SOURCES = {
    "RT-DETR RGB": ROOT / "compare/M3FD-LT20/original-size-45e-trial/rtdetr-rgb-seed42-native-b8-45e/valbest_test_coco/predictions.json",
    "RT-DETR concat": ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/baseline/predictions.json",
    "CLDyN+RT-DETR": ROOT / "compare/CLDyN_M3FD-lt20/cldyn-1/eval_m3fd_map/cldyn-vfn-rtdetr-1/val_best_test_full_20260906/predictions.json",
    "SH-DETR": ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/predictions.json",
}

YOLO_SOURCES = {
    "CFT": MSOD / "runs/M3FD-LT20/cft_x3_s42_300e_b8_1024_valbest_test_20260802/labels",
    "C2DFF-Net": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/C2DFF_best_test_s42_img1024_b8/labels",
    "YOLOv11-RGBT": ROOT / "compare/rerun_m3fd_s42_protocol_20260802/YOLOv11_RGBT_best_test_s42_img1024_b8/labels",
    "LCAFNet": LCAFNET / "runs/M3FD-LT20/lcafnet_s42_b4_1024_plus100_uuid4_valbest_test_20260802/labels",
}


def same_image(value: object) -> bool:
    if isinstance(value, int):
        return value == IMAGE_ID
    text = str(value)
    return text in {str(IMAGE_ID), IMAGE_KEY, f"{IMAGE_KEY}.png"}


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def yolo_xywhn_to_xyxy(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
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


def clip_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
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


def load_coco_json(path: Path, image_width: int, image_height: int) -> list[Detection]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
    detections = []
    for row in rows:
        if not same_image(row.get("image_id")) or float(row.get("score", 0.0)) < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is None:
            continue
        detections.append(Detection(xyxy, float(row["score"]), int(row["category_id"])))
    return detections


def load_yolo_labels(path: Path, image_width: int, image_height: int) -> list[Detection]:
    txt = path / f"{IMAGE_KEY}.txt"
    if not txt.exists():
        return []
    detections = []
    for line in txt.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        category, center_x, center_y, width, height, score = map(float, fields[:6])
        if score < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(
            yolo_xywhn_to_xyxy((center_x, center_y, width, height), image_width, image_height),
            image_width, image_height,
        )
        if xyxy is not None:
            detections.append(Detection(xyxy, score, int(category)))
    return detections


def load_ground_truth(image_width: int, image_height: int) -> list[Detection]:
    payload = json.loads(GT_PATH.read_text())
    detections = []
    for row in payload["annotations"]:
        if not same_image(row["image_id"]):
            continue
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is not None:
            detections.append(Detection(xyxy, 1.0, int(row["category_id"])))
    return detections


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
    x2 = min(left[2], right[2]); y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = la + ra - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    detections: Sequence[Detection], ground_truth: Sequence[Detection]
) -> dict[int, int]:
    """Greedy detection -> target index matching (highest score first)."""
    unmatched = set(range(len(ground_truth)))
    matches: dict[int, int] = {}
    for di in sorted(range(len(detections)), key=lambda i: -detections[i].score):
        candidates = [
            t for t in unmatched
            if ground_truth[t].category_id == detections[di].category_id
            and box_iou(ground_truth[t].xyxy, detections[di].xyxy) >= MATCH_IOU
        ]
        if not candidates:
            continue
        target = max(candidates, key=lambda t: box_iou(ground_truth[t].xyxy, detections[di].xyxy))
        matches[di] = target
        unmatched.remove(target)
    return matches


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "NimbusRoman-Bold" if bold else "NimbusRoman-Regular"
    path = FONT_DIR / f"{name}.otf"
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Nimbus Roman not found: {path}")


def draw_detections(
    draw: ImageDraw.ImageDraw,
    detections: Sequence[Detection],
    color_for,
    scale_x: float,
    scale_y: float,
) -> None:
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.xyxy
        display_box = (
            min(MODALITY_WIDTH - 1, max(0, round(x1 * scale_x))),
            min(MODALITY_HEIGHT - 1, max(0, round(y1 * scale_y))),
            min(MODALITY_WIDTH - 1, max(0, round(x2 * scale_x))),
            min(MODALITY_HEIGHT - 1, max(0, round(y2 * scale_y))),
        )
        draw.rectangle(display_box, outline=color_for(index), width=BOX_WIDTH)


def main() -> None:
    source = Image.open(IMAGE_PATH).convert("RGB")
    ir = Image.open(IR_IMAGE_PATH).convert("RGB")
    image_width, image_height = source.size
    methods = {name: load_coco_json(path, image_width, image_height) for name, path in JSON_SOURCES.items()}
    for name, label_dir in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(label_dir, image_width, image_height)
    ground_truth = load_ground_truth(image_width, image_height)
    assert set(methods) == set(METHOD_ORDER)

    matches = {name: match_detections(methods[name], ground_truth) for name in METHOD_ORDER}
    cmfc_targets = set(matches["SH-DETR"].values())
    other_targets = {t for name in METHOD_ORDER if name != "SH-DETR" for t in matches[name].values()}
    cmfc_unique_targets = cmfc_targets - other_targets

    def color_for(method: str):
        def _color(detection_index: int):
            target = matches[method].get(detection_index)
            if target is None:
                return CYAN
            if method == "SH-DETR" and target in cmfc_unique_targets:
                return YELLOW
            return RED
        return _color

    scale_x = MODALITY_WIDTH / image_width
    scale_y = MODALITY_HEIGHT / image_height

    panel_height = TITLE_BAND + MODALITY_HEIGHT + MODALITY_GAP + MODALITY_HEIGHT
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_WIDTH + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + NOTE_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font_title = load_font(28)
    font_label = load_font(22)

    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)

        rgb_panel = source.resize((MODALITY_WIDTH, MODALITY_HEIGHT), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(rgb_panel)
        draw_detections(draw, methods[name], color_for(name), scale_x, scale_y)
        canvas.paste(rgb_panel, (panel_left, image_top + TITLE_BAND))

        ir_panel = ir.resize((MODALITY_WIDTH, MODALITY_HEIGHT), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(ir_panel)
        draw_detections(draw, methods[name], color_for(name), scale_x, scale_y)
        canvas.paste(ir_panel, (panel_left, image_top + TITLE_BAND + MODALITY_HEIGHT + MODALITY_GAP))

        canvas_draw = ImageDraw.Draw(canvas)
        text_box = canvas_draw.textbbox((0, 0), name, font=font_title)
        title_x = panel_left + (MODALITY_WIDTH - (text_box[2] - text_box[0])) // 2
        title_y = image_top + TITLE_BAND - TITLE_GAP - (text_box[3] - text_box[1]) - text_box[1]
        canvas_draw.text((title_x, title_y), name, fill=TEXT, font=font_title)

    # arrow -> the CMFC-only car at (284,432,345,479) in the CMFC-DETR RGB thumbnail
    cmfc_panel_left = CANVAS_MARGIN + 3 * (MODALITY_WIDTH + COLUMN_GAP)
    cmfc_image_top = CANVAS_MARGIN + (panel_height + ROW_GAP)
    target = (284 + 345) / 2.0, (432 + 479) / 2.0
    end = (
        cmfc_panel_left + target[0] * scale_x,
        cmfc_image_top + TITLE_BAND + target[1] * scale_y,
    )
    start = (end[0], end[1] + 36)
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.line([start, end], fill=ARROW, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 12
    for delta in (2.5, -2.5):
        tip = (end[0] - head * math.cos(angle + delta), end[1] - head * math.sin(angle + delta))
        canvas_draw.line([end, tip], fill=ARROW, width=3)

    # legend: red / cyan / yellow
    entries = (
        (RED, "Correct detection"),
        (CYAN, "False positive"),
        (YELLOW, "SH-DETR-only true positive"),
    )
    swatch, gap = 18, 32
    widths = []
    for _, label in entries:
        text_box = canvas_draw.textbbox((0, 0), label, font=font_label)
        widths.append(swatch + 8 + text_box[2] - text_box[0])
    legend_x = (canvas_width - (sum(widths) + gap * (len(entries) - 1))) // 2
    legend_top = canvas_height - NOTE_BAND + 16
    for (color, label), width in zip(entries, widths):
        canvas_draw.rectangle((legend_x, legend_top, legend_x + swatch, legend_top + swatch),
                              outline=color, width=4)
        text_box = canvas_draw.textbbox((0, 0), label, font=font_label)
        text_height = text_box[3] - text_box[1]
        canvas_draw.text((legend_x + swatch + 8, legend_top + (swatch - text_height) // 2 - text_box[1]),
                         label, fill=TEXT, font=font_label)
        legend_x += width + gap

    note = "Arrow: the car target detected by SH-DETR but missed by all other methods."
    text_box = canvas_draw.textbbox((0, 0), note, font=font_label)
    note_x = (canvas_width - (text_box[2] - text_box[0])) / 2
    note_y = legend_top + 46
    canvas_draw.text((note_x, note_y), note, fill=(60, 60, 60), font=font_label)

    args_output = Path(OUTPUT_PATH)
    args_output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args_output, format="PNG", optimize=True)
    print(f"wrote={args_output}")
    print(f"canvas={canvas_width}x{canvas_height} image={IMAGE_KEY} thr={SCORE_THRESHOLD}")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))
    yellow = [i for i in matches["SH-DETR"] if matches["SH-DETR"][i] in cmfc_unique_targets]
    print(f"cmfc_unique(yellow)={len(yellow)}")
    for i in yellow:
        d = methods["SH-DETR"][i]
        print(f"  yellow xyxy={tuple(round(v,1) for v in d.xyxy)} cat={d.category_id} score={d.score:.3f}")


if __name__ == "__main__":
    main()
