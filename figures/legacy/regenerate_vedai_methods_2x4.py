#!/usr/bin/env python3
"""Regenerate the VEDAI 2x4 qualitative comparison from authoritative outputs."""

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
    if (_parent / "rscdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from rscdetr_paths import ROOT, DATASETS, CFT, LCAFNET, PAPER  # noqa: E402
PAPER_ROOT = PAPER
IMAGE_ID = 25
IMAGE_KEY = "00000025_co"
IMAGE_PATH = DATASETS / "VEDAI/Vehicules1024/00000025_co.png"
GT_PATH = DATASETS / "VEDAI/annotations/vedai_fold01_test_class8.json"
OUTPUT_PATH = PAPER_ROOT / "figures/qualitative_vedai_methods_2x4_table1_nodeyolo.png"

METHOD_ORDER = (
    "CFT",
    "C2DFF-Net",
    "RT-DETR RGB",
    "RT-DETR concat",
    "RSVDet",
    "YOLOv11-RGBT",
    "LCAFNet",
    "RSC-DETR",
)

SCORE_THRESHOLD = 0.50
RSVDET_THRESHOLD = 0.10
MATCH_IOU = 0.50
RED = (230, 32, 32)
YELLOW = (255, 214, 0)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

PANEL_SIZE = 440
TITLE_BAND = 54
TITLE_GAP = 10
CANVAS_MARGIN = 22
COLUMN_GAP = 18
ROW_GAP = 24
BOX_WIDTH = 4


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    category_id: int


@dataclass(frozen=True)
class ResizePad:
    input_width: float
    input_height: float
    pad_left: float
    pad_top: float
    scale_x: float
    scale_y: float


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
    "RSC-DETR": (
        ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json",
        0,
    ),
}

YOLO_SOURCES = {
    "YOLOv11-RGBT": ROOT
    / "compare/YOLOv11_RGBT/runs/yolov11_rgbt_vedai_s3407_uuid2_valbest_test_20260802/labels/00000025_co.txt",
    "LCAFNet": Path(
        f"{LCAFNET}/runs/VEDAI/"
        "lcafnet_s3407_uuid_valbest_test_20260802/labels/00000025_co.txt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", help="CUDA device used for the fresh RSVDet inference")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--rsvdet-json",
        type=Path,
        help="Fresh single-image RSVDet xyxy JSON. If omitted, inference runs from best.pt.",
    )
    return parser.parse_args()


def same_image(value: object) -> bool:
    if isinstance(value, int):
        return value == IMAGE_ID
    text = str(value)
    return text in {str(IMAGE_ID), IMAGE_KEY, f"{IMAGE_KEY}.png"}


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    """COCO JSON bbox is absolute top-left xywh, not xyxy."""
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def yolo_xywhn_to_xyxy(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO center-xywh directly into original-image xyxy."""
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


def undo_resize_pad_xyxy(
    box: Sequence[float], meta: ResizePad
) -> tuple[float, float, float, float]:
    """Undo model resize/padding for absolute input-space xyxy coordinates."""
    x1, y1, x2, y2 = map(float, box)
    return (
        (x1 - meta.pad_left) / meta.scale_x,
        (y1 - meta.pad_top) / meta.scale_y,
        (x2 - meta.pad_left) / meta.scale_x,
        (y2 - meta.pad_top) / meta.scale_y,
    )


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
        if float(row["score"]) < RSVDET_THRESHOLD:
            continue
        xyxy = clip_box(row["xyxy"], image_width, image_height)
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


def rscdetr_color(
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


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def title_position(
    draw: ImageDraw.ImageDraw, title: str, panel_left: int, image_top: int, font: ImageFont.ImageFont
) -> tuple[int, int]:
    """Keep method text in a dedicated band with a fixed gap above every image/box.

    Returns baseline-anchored coordinates (anchor "ms" at the call site): the
    y coordinate is the text baseline, so glyph height stays uniform even for
    names containing descenders.
    """
    x = panel_left + PANEL_SIZE // 2
    y = image_top - TITLE_GAP
    return x, y


def draw_panel(
    canvas: Image.Image,
    source: Image.Image,
    title: str,
    detections: Sequence[Detection],
    panel_left: int,
    image_top: int,
    ground_truth: Sequence[Detection],
    other_methods: Sequence[Detection],
    font: ImageFont.ImageFont,
) -> None:
    panel = source.resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    scale_x = PANEL_SIZE / source.width
    scale_y = PANEL_SIZE / source.height
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        display_box = (
            min(PANEL_SIZE - 1, max(0, round(x1 * scale_x))),
            min(PANEL_SIZE - 1, max(0, round(y1 * scale_y))),
            min(PANEL_SIZE - 1, max(0, round(x2 * scale_x))),
            min(PANEL_SIZE - 1, max(0, round(y2 * scale_y))),
        )
        color = (
            rscdetr_color(detection, ground_truth, other_methods)
            if title == "RSC-DETR"
            else RED
        )
        draw.rectangle(display_box, outline=color, width=BOX_WIDTH)
    canvas.paste(panel, (panel_left, image_top))
    canvas_draw = ImageDraw.Draw(canvas)
    tx, ty = title_position(canvas_draw, title, panel_left, image_top, font)
    canvas_draw.text((tx, ty), title, fill=TEXT, font=font, anchor="ms")


def assert_in_bounds(methods: dict[str, list[Detection]], width: int, height: int) -> None:
    for method, detections in methods.items():
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                raise AssertionError(f"{method} has an out-of-bounds box: {detection}")


def main() -> None:
    args = parse_args()
    source = Image.open(IMAGE_PATH).convert("RGB")
    methods = {
        name: load_coco_json(path, category_offset, source.width, source.height)
        for name, (path, category_offset) in JSON_SOURCES.items()
    }
    for name, path in YOLO_SOURCES.items():
        methods[name] = load_yolo_labels(path, source.width, source.height)
    methods["RSVDet"] = (
        load_rsvdet_json(args.rsvdet_json, source.width, source.height)
        if args.rsvdet_json
        else infer_rsvdet(IMAGE_PATH, args.device)
    )
    ground_truth = load_ground_truth(source.width, source.height)
    assert set(methods) == set(METHOD_ORDER)
    assert_in_bounds(methods, source.width, source.height)

    other_methods = [
        detection
        for name in METHOD_ORDER[:-1]
        for detection in methods[name]
    ]
    canvas_width = 2 * CANVAS_MARGIN + 4 * PANEL_SIZE + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * (TITLE_BAND + PANEL_SIZE) + ROW_GAP
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    font = load_font(28)
    for index, name in enumerate(METHOD_ORDER):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (PANEL_SIZE + COLUMN_GAP)
        image_top = CANVAS_MARGIN + TITLE_BAND + row * (TITLE_BAND + PANEL_SIZE + ROW_GAP)
        draw_panel(
            canvas,
            source,
            name,
            methods[name],
            panel_left,
            image_top,
            ground_truth,
            other_methods,
            font,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    yellow_count = sum(
        rscdetr_color(detection, ground_truth, other_methods) == YELLOW
        for detection in methods["RSC-DETR"]
    )
    print(f"wrote={args.output}")
    print(f"image={IMAGE_KEY} size={source.width}x{source.height}")
    print("order=" + " | ".join(METHOD_ORDER))
    print("counts=" + ", ".join(f"{name}:{len(methods[name])}" for name in METHOD_ORDER))
    print(f"rscdetr_yellow={yellow_count} rscdetr_red={len(methods['RSC-DETR']) - yellow_count}")


if __name__ == "__main__":
    main()
