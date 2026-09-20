#!/usr/bin/env python3
"""Fig7 (VEDAI 1033) qualitative, v2.

Each method panel is CROPPED to a square window around the scene's three ground
truth targets (tractor + two pickups) instead of showing the whole 1024x1024
image, so the targets are large and legible. Detection boxes are mapped from
image to crop coordinates. The CMFC-only tractor target keeps its yellow box
(only CMFC-DETR matches it above threshold); RSVDet is loaded from its full
test-set predictions file so its two high-confidence pickup detections appear
(a previous render had an empty RSVDet panel).
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


ROOT = Path("/home/denglingjun/re-detr-last")
IMAGE_ID = 1033
IMAGE_KEY = "00001033_co"
IMAGE_PATH = Path("/home/denglingjun/vedai_data/datasets/VEDAI/Vehicules1024/00001033_co.png")
IR_IMAGE_PATH = Path("/home/denglingjun/vedai_data/datasets/VEDAI/Vehicules1024/00001033_ir.png")
GT_PATH = Path("/home/denglingjun/vedai_data/datasets/VEDAI/annotations/vedai_fold01_test_class8.json")
OUTPUT_PATH = ROOT / "CMFC_DETR_v8/figures/qualitative_vedai_rgb_ir_1033_2x4.png"
RSVDET_JSON = ROOT / "outputs/compare_vedai_icafusion_rsvdet/rsvdet/eval_best/predictions.json"

# Crop window (image coords): a square covering tractor (377,734,425,778) and the
# two pickups (515,818,566,836)/(571,823,619,840), centred on (498,787).
CROP = (348, 637, 648, 937)
CROP_SIZE = CROP[2] - CROP[0]

METHOD_ORDER = (
    "CFT",
    "C2DFF-Net",
    "RT-DETR RGB",
    "RT-DETR concat",
    "RSVDet",
    "YOLOv11-RGBT",
    "LCAFNet",
    "CMFC-DETR",
)

SCORE_THRESHOLD = 0.70
RSVDET_THRESHOLD = 0.70
MATCH_IOU = 0.50
RED = (230, 32, 32)
YELLOW = (255, 214, 0)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)
MODALITY_SIZE = 240        # each RGB/IR thumbnail (cropped window resized here)
MODALITY_GAP = 8
TITLE_BAND = 40
TITLE_GAP = 6
CANVAS_MARGIN = 20
COLUMN_GAP = 12
ROW_GAP = 24
LEGEND_BAND = 58
BOX_WIDTH = 4

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int


JSON_SOURCES = {
    "CFT": (
        Path("/home/denglingjun/CFT/runs/VEDAI/CFT-26-fold1_valbest_test_20260802/best_predictions.json"),
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
    "CMFC-DETR": (
        ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json",
        0,
    ),
}

YOLO_SOURCES = {
    "YOLOv11-RGBT": ROOT
    / "compare/YOLOv11_RGBT/runs/yolov11_rgbt_vedai_s3407_uuid2_valbest_test_20260802/labels/00001033_co.txt",
    "LCAFNet": Path(
        "/home/denglingjun/LCAFNet/runs/VEDAI/"
        "lcafnet_s3407_uuid_valbest_test_20260802/labels/00001033_co.txt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", help="CUDA device used for the fresh RSVDet inference")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--rsvdet-json",
        type=Path,
        help="Fresh single-image RSVDet xyxy JSON. Defaults to the full test-set predictions file.",
    )
    return parser.parse_args()


def same_image(value: object) -> bool:
    if isinstance(value, int):
        return value == IMAGE_ID
    text = str(value)
    return text in {str(IMAGE_ID), IMAGE_KEY, f"{IMAGE_KEY}.png"}


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def clip_box(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    values = tuple(map(float, box))
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
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


def load_coco_json(
    path: Path, category_offset: int, image_width: int, image_height: int
) -> list[Detection]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
    detections = []
    for row in rows:
        if not same_image(row.get("image_id")) or float(row.get("score", 0.0)) < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is None:
            continue
        detections.append(
            Detection(xyxy, float(row["score"]), int(row["category_id"]) + category_offset)
        )
    return detections


def load_yolo_labels(path: Path, image_width: int, image_height: int) -> list[Detection]:
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


def load_rsvdet_json(path: Path, image_width: int, image_height: int) -> list[Detection]:
    detections = []
    for row in json.loads(path.read_text()):
        if not same_image(row.get("image_id")):
            continue
        if float(row["score"]) < RSVDET_THRESHOLD:
            continue
        if "xyxy" in row:
            xyxy = clip_box(row["xyxy"], image_width, image_height)
        else:
            xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), image_width, image_height)
        if xyxy is not None:
            detections.append(Detection(xyxy, float(row["score"]), int(row["category_id"])))
    return detections


def infer_rsvdet(image_path: Path, device: str) -> list[Detection]:
    repo = ROOT / "compare/RSVDet_VEDAI"
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO

    model = YOLO(str(repo / "rsvdet-s3407/weights/best.pt"))
    result = model.predict(
        source=str(image_path),
        imgsz=1024,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=device,
        verbose=False,
    )[0]
    image_height, image_width = result.orig_shape
    detections = []
    for xyxy, score, category in zip(
        result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
    ):
        if score < RSVDET_THRESHOLD:
            continue
        clipped = clip_box(xyxy, image_width, image_height)
        if clipped is not None:
            detections.append(Detection(clipped, float(score), int(category) + 1))
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
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def cmfc_color(
    detection: Detection,
    ground_truth: Iterable[Detection],
    other_methods: Iterable[Detection],
) -> tuple[int, int, int]:
    matching_gt = [
        target
        for target in ground_truth
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


def draw_detections(
    draw: ImageDraw.ImageDraw,
    detections: Sequence[Detection],
    color_for,
) -> None:
    """Draw detections mapped from image coords into the cropped thumbnail."""
    cx0, cy0, cx1, cy1 = CROP
    sx = MODALITY_SIZE / (cx1 - cx0)
    sy = MODALITY_SIZE / (cy1 - cy0)
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        display_box = (
            min(MODALITY_SIZE - 1, max(0, round((x1 - cx0) * sx))),
            min(MODALITY_SIZE - 1, max(0, round((y1 - cy0) * sy))),
            min(MODALITY_SIZE - 1, max(0, round((x2 - cx0) * sx))),
            min(MODALITY_SIZE - 1, max(0, round((y2 - cy0) * sy))),
        )
        if display_box[2] - display_box[0] < 2 or display_box[3] - display_box[1] < 2:
            continue
        draw.rectangle(display_box, outline=color_for(detection), width=BOX_WIDTH)


def main() -> None:
    args = parse_args()
    source = Image.open(IMAGE_PATH).convert("RGB")
    ir = Image.open(IR_IMAGE_PATH).convert("RGB")
    methods = {
        name: load_coco_json(path, category_offset, source.width, source.height)
        for name, (path, category_offset) in JSON_SOURCES.items()
    }
    for name, path in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(path, source.width, source.height)
    methods["RSVDet"] = (
        load_rsvdet_json(args.rsvdet_json, source.width, source.height)
        if args.rsvdet_json
        else load_rsvdet_json(RSVDET_JSON, source.width, source.height)
    )
    ground_truth = load_ground_truth(source.width, source.height)
    assert set(methods) == set(METHOD_ORDER)

    other_methods = [
        detection
        for name in METHOD_ORDER[:-1]
        for detection in methods[name]
    ]

    panel_height = TITLE_BAND + MODALITY_SIZE + MODALITY_GAP + MODALITY_SIZE
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_SIZE + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font_title = load_font(24)
    font_label = load_font(20)

    def color_for_cmfc(detection: Detection) -> tuple[int, int, int]:
        return cmfc_color(detection, ground_truth, other_methods)

    def color_red(_: Detection) -> tuple[int, int, int]:
        return RED

    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_SIZE + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        color_for = color_for_cmfc if name == "CMFC-DETR" else color_red

        # RGB thumbnail (cropped window)
        rgb_panel = source.crop(CROP).resize(
            (MODALITY_SIZE, MODALITY_SIZE), Image.LANCZOS
        )
        draw = ImageDraw.Draw(rgb_panel)
        draw_detections(draw, methods[name], color_for)
        canvas.paste(rgb_panel, (panel_left, image_top + TITLE_BAND))

        # IR thumbnail (cropped window)
        ir_panel = ir.crop(CROP).resize(
            (MODALITY_SIZE, MODALITY_SIZE), Image.LANCZOS
        )
        draw = ImageDraw.Draw(ir_panel)
        draw_detections(draw, methods[name], color_for)
        canvas.paste(ir_panel, (panel_left, image_top + TITLE_BAND + MODALITY_SIZE + MODALITY_GAP))

        canvas_draw = ImageDraw.Draw(canvas)
        text_box = canvas_draw.textbbox((0, 0), name, font=font_title)
        title_x = panel_left + (MODALITY_SIZE - (text_box[2] - text_box[0])) // 2
        title_y = image_top + TITLE_BAND - TITLE_GAP - (text_box[3] - text_box[1]) - text_box[1]
        canvas_draw.text((title_x, title_y), name, fill=TEXT, font=font_title)

    # legend: red = detection, yellow = CMFC-only true positive (centered)
    canvas_draw = ImageDraw.Draw(canvas)
    legend_top = canvas_height - LEGEND_BAND + 16
    entries = ((RED, "Detection"), (YELLOW, "CMFC-only true positive"))
    gap = 40
    widths = []
    for _, label in entries:
        tb = canvas_draw.textbbox((0, 0), label, font=font_label)
        widths.append(24 + 8 + (tb[2] - tb[0]))
    legend_x = (canvas_width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for (color, label), width in zip(entries, widths):
        canvas_draw.rectangle([legend_x, legend_top, legend_x + 24, legend_top + 20],
                              outline=color, width=3)
        tb = canvas_draw.textbbox((0, 0), label, font=font_label)
        th = tb[3] - tb[1]
        canvas_draw.text((legend_x + 24 + 8, legend_top + (20 - th) // 2 - tb[1]),
                         label, fill=TEXT, font=font_label)
        legend_x += width + gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    print(f"wrote={args.output}")
    print(f"canvas={canvas_width}x{canvas_height} crop={CROP} thr={SCORE_THRESHOLD}")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))
    yellow = [d for d in methods["CMFC-DETR"] if color_for_cmfc(d) == YELLOW]
    print(f"cmfc_unique(yellow)={len(yellow)}")
    for d in yellow:
        print(f"  yellow xyxy={tuple(round(v,1) for v in d.xyxy)} cat={d.category_id} score={d.score:.3f}")


if __name__ == "__main__":
    main()
