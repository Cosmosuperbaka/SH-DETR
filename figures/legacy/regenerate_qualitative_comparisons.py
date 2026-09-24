#!/usr/bin/env python3
"""Regenerate the active VEDAI and M3FD-LT20 2x4 qualitative figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from rscdetr_paths import ROOT, DATASETS, CFT, LCAFNET, MSOD, PAPER  # noqa: E402
PAPER_ROOT = PAPER

SCORE_THRESHOLD = 0.50
MATCH_IOU = 0.50
RED = (222, 32, 38)
CYAN = (0, 166, 214)
YELLOW = (255, 211, 0)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

PANEL_WIDTH = 440
TITLE_BAND = 54
TITLE_GAP = 10
CANVAS_MARGIN = 22
COLUMN_GAP = 18
ROW_GAP = 24
LEGEND_BAND = 44
BOX_WIDTH = 4
VISIBLE_UNIQUE_MIN_SIDE = 6.0


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


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class JsonSource:
    path: Path
    category_offset: int = 0


@dataclass(frozen=True)
class YoloSource:
    directory: Path
    category_offset: int = 0


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    annotation_path: Path
    image_dir: Path
    method_order: tuple[str, ...]
    json_sources: Mapping[str, JsonSource]
    yolo_sources: Mapping[str, YoloSource]
    output_path: Path
    vedai_fallback_names: bool = False


@dataclass(frozen=True)
class CandidateMetrics:
    image_id: int
    file_name: str
    gt_count: int
    rscdetr_tp: int
    rscdetr_fp: int
    unique_targets: tuple[int, ...]
    visible_unique: int
    visible_size_sum: float
    other_fp: int

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            float(self.visible_unique),
            float(len(self.unique_targets)),
            self.visible_size_sum,
            float(self.rscdetr_tp),
            float(-self.rscdetr_fp),
            float(self.other_fp),
        )


VEDAI = DatasetConfig(
    name="VEDAI",
    annotation_path=Path(
        f"{DATASETS}/VEDAI/annotations/"
        "vedai_fold01_test_class8.json"
    ),
    image_dir=DATASETS / "VEDAI/Vehicules1024",
    method_order=(
        "CFT",
        "C2DFF-Net",
        "RT-DETR RGB",
        "RT-DETR concat",
        "RSVDet",
        "YOLOv11-RGBT",
        "LCAFNet",
        "RSC-DETR",
    ),
    json_sources={
        "CFT": JsonSource(
            Path(
                f"{CFT}/runs/VEDAI/"
                "CFT-26-fold1_valbest_test_20260802/best_predictions.json"
            ),
            1,
        ),
        "C2DFF-Net": JsonSource(
            ROOT
            / "compare/C2DFF_VEDAI/runs/"
            "c2dff_s3407_uuid2_valbest_test_20260802/predictions.json",
            1,
        ),
        "RT-DETR RGB": JsonSource(
            ROOT / "outputs/vedai_direct_test_unified/rgb/seed_3407/predictions.json"
        ),
        "RT-DETR concat": JsonSource(
            ROOT / "outputs/vedai_direct_test_unified/baseline/seed_3407/predictions.json"
        ),
        "RSC-DETR": JsonSource(
            ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json"
        ),
    },
    yolo_sources={
        "YOLOv11-RGBT": YoloSource(
            ROOT
            / "compare/YOLOv11_RGBT/runs/"
            "yolov11_rgbt_vedai_s3407_uuid2_valbest_test_20260802/labels",
            1,
        ),
        "LCAFNet": YoloSource(
            Path(
                f"{LCAFNET}/runs/VEDAI/"
                "lcafnet_s3407_uuid_valbest_test_20260802/labels"
            ),
            1,
        ),
    },
    output_path=PAPER_ROOT / "figures/qualitative_vedai_methods_2x4_table1_nodeyolo.png",
    vedai_fallback_names=True,
)


M3FD = DatasetConfig(
    name="M3FD-LT20",
    annotation_path=Path(
        f"{DATASETS}/M3FD/processed/lt20_seed42/"
        "annotations/instances_test.json"
    ),
    image_dir=DATASETS / "M3FD/raw/vi",
    method_order=(
        "CFT",
        "RT-DETR RGB",
        "RT-DETR concat",
        "C2DFF-Net",
        "YOLOv11-RGBT",
        "LCAFNet",
        "CLDyN+RT-DETR",
        "RSC-DETR",
    ),
    json_sources={
        "RT-DETR RGB": JsonSource(
            ROOT
            / "compare/M3FD-LT20/original-size-45e-trial/"
            "rtdetr-rgb-seed42-native-b8-45e/valbest_test_coco/predictions.json"
        ),
        "RT-DETR concat": JsonSource(
            ROOT
            / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/"
            "baseline/predictions.json"
        ),
        "CLDyN+RT-DETR": JsonSource(
            ROOT
            / "compare/CLDyN_M3FD-lt20/cldyn-1/eval_m3fd_map/"
            "cldyn-vfn-rtdetr-1/val_best_test_per_class/predictions.json"
        ),
        "RSC-DETR": JsonSource(
            ROOT
            / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/"
            "v19c_spsf/predictions.json"
        ),
    },
    yolo_sources={
        "CFT": YoloSource(
            Path(
                f"{MSOD}/runs/M3FD-LT20/"
                "cft_x3_s42_300e_b8_1024_valbest_test_20260802/labels"
            )
        ),
        "C2DFF-Net": YoloSource(
            ROOT
            / "compare/rerun_m3fd_s42_protocol_20260802/"
            "C2DFF_best_test_s42_img1024_b8/labels"
        ),
        "YOLOv11-RGBT": YoloSource(
            ROOT
            / "compare/rerun_m3fd_s42_protocol_20260802/"
            "YOLOv11_RGBT_best_test_s42_img1024_b8/labels"
        ),
        "LCAFNet": YoloSource(
            Path(
                f"{LCAFNET}/runs/M3FD-LT20/"
                "lcafnet_s42_b4_1024_plus100_uuid4_valbest_test_20260802/labels"
            )
        ),
    },
    output_path=PAPER_ROOT / "figures/qualitative_m3fd_methods_2x4_nodeyolo.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", help="CUDA device for fresh RSVDet inference")
    parser.add_argument("--vedai-image-id", type=int)
    parser.add_argument("--m3fd-image-id", type=int)
    parser.add_argument("--vedai-output", type=Path, default=VEDAI.output_path)
    parser.add_argument("--m3fd-output", type=Path, default=M3FD.output_path)
    parser.add_argument("--vedai-rsvdet-candidates", type=int, default=40)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    return parser.parse_args()


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    """COCO bbox is absolute top-left xywh in original-image coordinates."""
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def yolo_xywhn_to_xyxy(
    box: Sequence[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """YOLO bbox is normalized center xywh and maps directly to the original image."""
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
    """Undo resize/padding for an absolute model-input xyxy box."""
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


def payload_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("predictions", "annotations", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Prediction JSON must be a list or contain predictions/annotations/results")


def load_dataset(
    config: DatasetConfig,
) -> tuple[dict[int, ImageRecord], dict[int, list[Detection]], dict[int, str], dict[str, int]]:
    payload = json.loads(config.annotation_path.read_text())
    images: dict[int, ImageRecord] = {}
    names_to_ids: dict[str, int] = {}
    for row in payload["images"]:
        image_id = int(row["id"])
        file_name = str(row["file_name"])
        path = config.image_dir / file_name
        if config.vedai_fallback_names and not path.exists():
            path = config.image_dir / f"{image_id:08d}_co.png"
            file_name = path.name
        record = ImageRecord(
            image_id=image_id,
            file_name=file_name,
            path=path,
            width=int(row["width"]),
            height=int(row["height"]),
        )
        images[image_id] = record
        names_to_ids[file_name] = image_id
        names_to_ids[Path(file_name).stem] = image_id
        names_to_ids[str(image_id)] = image_id
        names_to_ids[f"{image_id:08d}"] = image_id
        names_to_ids[f"{image_id:08d}_co"] = image_id

    ground_truth: dict[int, list[Detection]] = defaultdict(list)
    for row in payload["annotations"]:
        image_id = resolve_image_id(row["image_id"], images, names_to_ids)
        if image_id is None:
            continue
        record = images[image_id]
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), record.width, record.height)
        if xyxy is not None:
            ground_truth[image_id].append(
                Detection(xyxy=xyxy, score=1.0, category_id=int(row["category_id"]))
            )
    categories = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    return images, ground_truth, categories, names_to_ids


def resolve_image_id(
    value: object,
    images: Mapping[int, ImageRecord],
    names_to_ids: Mapping[str, int],
) -> int | None:
    if isinstance(value, int):
        return value if value in images else None
    text = str(value)
    for candidate in (text, Path(text).name, Path(text).stem):
        if candidate in names_to_ids:
            return names_to_ids[candidate]
    try:
        numeric = int(text.split("_")[0])
    except ValueError:
        return None
    return numeric if numeric in images else None


def load_json_source(
    source: JsonSource,
    images: Mapping[int, ImageRecord],
    names_to_ids: Mapping[str, int],
) -> dict[int, list[Detection]]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for row in payload_rows(json.loads(source.path.read_text())):
        score = float(row.get("score", 0.0))
        if score < SCORE_THRESHOLD:
            continue
        image_id = resolve_image_id(row.get("image_id"), images, names_to_ids)
        if image_id is None:
            continue
        record = images[image_id]
        xyxy = clip_box(coco_xywh_to_xyxy(row["bbox"]), record.width, record.height)
        if xyxy is not None:
            grouped[image_id].append(
                Detection(
                    xyxy=xyxy,
                    score=score,
                    category_id=int(row["category_id"]) + source.category_offset,
                )
            )
    return grouped


def load_yolo_file(source: YoloSource, record: ImageRecord) -> list[Detection]:
    path = source.directory / f"{Path(record.file_name).stem}.txt"
    if not path.exists() and record.file_name.endswith("_co.png"):
        path = source.directory / f"{record.image_id:08d}_co.txt"
    if not path.exists():
        return []
    detections = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        category, center_x, center_y, width, height, score = map(float, fields[:6])
        if score < SCORE_THRESHOLD:
            continue
        xyxy = clip_box(
            yolo_xywhn_to_xyxy(
                (center_x, center_y, width, height), record.width, record.height
            ),
            record.width,
            record.height,
        )
        if xyxy is not None:
            detections.append(
                Detection(
                    xyxy=xyxy,
                    score=score,
                    category_id=int(category) + source.category_offset,
                )
            )
    return detections


def load_methods(
    config: DatasetConfig,
    images: Mapping[int, ImageRecord],
    names_to_ids: Mapping[str, int],
) -> dict[str, dict[int, list[Detection]]]:
    methods = {
        name: load_json_source(source, images, names_to_ids)
        for name, source in config.json_sources.items()
    }
    for name, source in config.yolo_sources.items():
        methods[name] = {
            image_id: load_yolo_file(source, record)
            for image_id, record in images.items()
        }
    return methods


def match_detections(
    detections: Sequence[Detection], ground_truth: Sequence[Detection]
) -> dict[int, int]:
    """Greedily match predictions to GT once, in descending confidence order."""
    unmatched_targets = set(range(len(ground_truth)))
    matched: dict[int, int] = {}
    for detection_index in sorted(
        range(len(detections)), key=lambda index: detections[index].score, reverse=True
    ):
        detection = detections[detection_index]
        candidates = [
            target_index
            for target_index in unmatched_targets
            if ground_truth[target_index].category_id == detection.category_id
            and box_iou(ground_truth[target_index].xyxy, detection.xyxy) >= MATCH_IOU
        ]
        if not candidates:
            continue
        target_index = max(
            candidates,
            key=lambda index: box_iou(ground_truth[index].xyxy, detection.xyxy),
        )
        matched[detection_index] = target_index
        unmatched_targets.remove(target_index)
    return matched


def assert_in_bounds(
    methods: Mapping[str, Sequence[Detection]], image_width: int, image_height: int
) -> None:
    for method, detections in methods.items():
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            if not (0.0 <= x1 < x2 <= image_width - 1.0):
                raise AssertionError(f"{method} has an out-of-bounds x coordinate: {detection}")
            if not (0.0 <= y1 < y2 <= image_height - 1.0):
                raise AssertionError(f"{method} has an out-of-bounds y coordinate: {detection}")


def candidate_metrics(
    image_id: int,
    record: ImageRecord,
    ground_truth: Sequence[Detection],
    methods: Mapping[str, Mapping[int, list[Detection]]],
    method_order: Sequence[str],
) -> CandidateMetrics:
    matches = {
        name: match_detections(methods.get(name, {}).get(image_id, []), ground_truth)
        for name in method_order
    }
    rscdetr_match = matches["RSC-DETR"]
    other_targets = {
        target_index
        for name in method_order
        if name != "RSC-DETR"
        for target_index in matches[name].values()
    }
    unique_targets = tuple(
        sorted(set(rscdetr_match.values()).difference(other_targets))
    )
    display_scale = PANEL_WIDTH / record.width
    display_sizes = [
        min(
            (ground_truth[index].xyxy[2] - ground_truth[index].xyxy[0]) * display_scale,
            (ground_truth[index].xyxy[3] - ground_truth[index].xyxy[1]) * display_scale,
        )
        for index in unique_targets
    ]
    rscdetr_count = len(methods.get("RSC-DETR", {}).get(image_id, []))
    other_fp = sum(
        len(methods.get(name, {}).get(image_id, [])) - len(matches[name])
        for name in method_order
        if name != "RSC-DETR"
    )
    return CandidateMetrics(
        image_id=image_id,
        file_name=record.file_name,
        gt_count=len(ground_truth),
        rscdetr_tp=len(rscdetr_match),
        rscdetr_fp=rscdetr_count - len(rscdetr_match),
        unique_targets=unique_targets,
        visible_unique=sum(size >= VISIBLE_UNIQUE_MIN_SIDE for size in display_sizes),
        visible_size_sum=sum(display_sizes),
        other_fp=other_fp,
    )


def rank_candidates(
    images: Mapping[int, ImageRecord],
    ground_truth: Mapping[int, list[Detection]],
    methods: Mapping[str, Mapping[int, list[Detection]]],
    method_order: Sequence[str],
    image_ids: Iterable[int] | None = None,
) -> list[CandidateMetrics]:
    candidates = []
    for image_id in image_ids if image_ids is not None else images:
        record = images[image_id]
        if not record.path.exists():
            continue
        metrics = candidate_metrics(
            image_id,
            record,
            ground_truth.get(image_id, []),
            methods,
            method_order,
        )
        if metrics.unique_targets:
            candidates.append(metrics)
    return sorted(candidates, key=lambda item: item.rank_key, reverse=True)


def print_ranking(name: str, candidates: Sequence[CandidateMetrics], limit: int = 12) -> None:
    print(f"{name} candidates:")
    for index, item in enumerate(candidates[:limit], start=1):
        print(
            f"  {index:02d} id={item.image_id} file={item.file_name} "
            f"gt={item.gt_count} rscdetr_tp={item.rscdetr_tp} rscdetr_fp={item.rscdetr_fp} "
            f"unique={len(item.unique_targets)} visible_unique={item.visible_unique} "
            f"other_fp={item.other_fp}"
        )


def infer_rsvdet(
    records: Sequence[ImageRecord], device: str
) -> dict[int, list[Detection]]:
    repo = ROOT / "compare/RSVDet_VEDAI"
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO

    model = YOLO(str(repo / "rsvdet-s3407/weights/best.pt"))
    results = model.predict(
        source=[str(record.path) for record in records],
        imgsz=1024,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=device,
        verbose=False,
    )
    grouped: dict[int, list[Detection]] = {}
    for record, result in zip(records, results):
        detections = []
        for xyxy, score, category in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            if score < SCORE_THRESHOLD:
                continue
            clipped = clip_box(xyxy, record.width, record.height)
            if clipped is not None:
                detections.append(
                    Detection(
                        xyxy=clipped,
                        score=float(score),
                        category_id=int(category) + 1,
                    )
                )
        grouped[record.image_id] = detections
    return grouped


def select_vedai(
    images: Mapping[int, ImageRecord],
    ground_truth: Mapping[int, list[Detection]],
    methods: dict[str, dict[int, list[Detection]]],
    device: str,
    candidate_limit: int,
    forced_image_id: int | None,
) -> tuple[int, list[CandidateMetrics]]:
    preliminary_order = tuple(name for name in VEDAI.method_order if name != "RSVDet")
    preliminary = rank_candidates(images, ground_truth, methods, preliminary_order)
    print_ranking("VEDAI before fresh RSVDet", preliminary)
    if forced_image_id is not None:
        inference_ids = [forced_image_id]
    else:
        inference_ids = [item.image_id for item in preliminary[:candidate_limit]]
    methods["RSVDet"] = infer_rsvdet(
        [images[image_id] for image_id in inference_ids], device
    )
    final = rank_candidates(
        images,
        ground_truth,
        methods,
        VEDAI.method_order,
        inference_ids,
    )
    print_ranking("VEDAI after fresh RSVDet", final)
    if forced_image_id is not None:
        return forced_image_id, final
    if not final:
        raise RuntimeError("No VEDAI sample retained a RSC-DETR-only true positive")
    return final[0].image_id, final


def select_m3fd(
    images: Mapping[int, ImageRecord],
    ground_truth: Mapping[int, list[Detection]],
    methods: Mapping[str, Mapping[int, list[Detection]]],
    forced_image_id: int | None,
) -> tuple[int, list[CandidateMetrics]]:
    ranked = rank_candidates(images, ground_truth, methods, M3FD.method_order)
    print_ranking("M3FD-LT20", ranked)
    if forced_image_id is not None:
        return forced_image_id, ranked
    if not ranked:
        raise RuntimeError("No M3FD sample has a RSC-DETR-only true positive")
    return ranked[0].image_id, ranked


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def title_position(
    draw: ImageDraw.ImageDraw,
    title: str,
    panel_left: int,
    image_top: int,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    """Keep method names in a dedicated title band, never on a detection box."""
    text_box = draw.textbbox((0, 0), title, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x = panel_left + (PANEL_WIDTH - text_width) // 2
    y = image_top - TITLE_GAP - text_height - text_box[1]
    return x, y


def detection_color(
    method: str,
    detection_index: int,
    matches: Mapping[str, Mapping[int, int]],
    rscdetr_unique_targets: set[int],
) -> tuple[int, int, int]:
    target_index = matches[method].get(detection_index)
    if target_index is None:
        return CYAN
    if method == "RSC-DETR" and target_index in rscdetr_unique_targets:
        return YELLOW
    return RED


def category_error_target(
    detection: Detection,
    ground_truth: Sequence[Detection],
) -> int | None:
    """Return the best spatially-overlapping GT when the predicted class is wrong."""
    candidates = [
        (index, box_iou(detection.xyxy, target.xyxy))
        for index, target in enumerate(ground_truth)
        if target.category_id != detection.category_id
    ]
    if not candidates:
        return None
    index, overlap = max(candidates, key=lambda item: item[1])
    return index if overlap >= 0.30 else None


def draw_error_label(
    draw: ImageDraw.ImageDraw,
    display_box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    panel_width: int,
    panel_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> None:
    """Place a wrong-class label away from its box and route a short leader line."""
    tb = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y, gap = 5, 3, 8
    label_w, label_h = text_w + 2 * pad_x, text_h + 2 * pad_y
    x1, y1, x2, y2 = display_box
    candidates = [
        (x1, y1 - gap - label_h),
        (x1, y2 + gap),
        (x2 + gap, y1),
        (x1 - gap - label_w, y1),
    ]
    def intersects(rect, other):
        return not (rect[2] <= other[0] or other[2] <= rect[0] or rect[3] <= other[1] or other[3] <= rect[1])
    chosen = None
    for x, y in candidates:
        x = min(max(2, x), panel_width - label_w - 2)
        y = min(max(2, y), panel_height - label_h - 2)
        rect = (x, y, x + label_w, y + label_h)
        if not any(intersects(rect, other) for other in occupied):
            chosen = rect
            break
    if chosen is None:
        x, y = candidates[0]
        chosen = (min(max(2, x), panel_width - label_w - 2), min(max(2, y), panel_height - label_h - 2), min(max(2, x), panel_width - label_w - 2) + label_w, min(max(2, y), panel_height - label_h - 2) + label_h)
    lx1, ly1, lx2, ly2 = chosen
    anchor = ((x1 + x2) // 2, y1 if ly2 < y1 else y2 if ly1 > y2 else (y1 + y2) // 2)
    edge = ((lx1 + lx2) // 2, ly2 if ly2 < y1 else ly1 if ly1 > y2 else (ly1 + ly2) // 2)
    draw.line((anchor, edge), fill=CYAN, width=2)
    draw.rectangle(chosen, fill=WHITE, outline=CYAN, width=1)
    draw.text((lx1 + pad_x - tb[0], ly1 + pad_y - tb[1]), text, fill=CYAN, font=font)
    occupied.append(chosen)


def draw_legend(
    canvas: Image.Image, top: int, font: ImageFont.ImageFont
) -> None:
    draw = ImageDraw.Draw(canvas)
    entries = ((RED, "Correct detection"), (CYAN, "False positive"), (YELLOW, "RSC-DETR-only true positive"))
    swatch = 18
    gap = 32
    widths = []
    for _, label in entries:
        text_box = draw.textbbox((0, 0), label, font=font)
        widths.append(swatch + 8 + text_box[2] - text_box[0])
    left = (canvas.width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for (color, label), width in zip(entries, widths):
        draw.rectangle((left, top, left + swatch, top + swatch), outline=color, width=4)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_height = text_box[3] - text_box[1]
        draw.text(
            (left + swatch + 8, top + (swatch - text_height) // 2 - text_box[1]),
            label,
            fill=TEXT,
            font=font,
        )
        left += width + gap


def render(
    config: DatasetConfig,
    output: Path,
    image_id: int,
    images: Mapping[int, ImageRecord],
    ground_truth_by_image: Mapping[int, list[Detection]],
    methods_by_image: Mapping[str, Mapping[int, list[Detection]]],
    category_names: Mapping[int, str],
) -> None:
    record = images[image_id]
    source = Image.open(record.path).convert("RGB")
    if source.size != (record.width, record.height):
        raise AssertionError(
            f"{record.file_name} annotation size {(record.width, record.height)} "
            f"does not match image size {source.size}"
        )
    ground_truth = ground_truth_by_image.get(image_id, [])
    methods = {
        name: list(methods_by_image.get(name, {}).get(image_id, []))
        for name in config.method_order
    }
    if set(methods) != set(config.method_order):
        raise AssertionError("Method order and loaded predictions differ")
    assert_in_bounds(methods, record.width, record.height)
    matches = {
        name: match_detections(detections, ground_truth)
        for name, detections in methods.items()
    }
    other_targets = {
        target_index
        for name in config.method_order
        if name != "RSC-DETR"
        for target_index in matches[name].values()
    }
    rscdetr_unique_targets = set(matches["RSC-DETR"].values()).difference(other_targets)

    panel_height = round(PANEL_WIDTH * record.height / record.width)
    canvas_width = 2 * CANVAS_MARGIN + 4 * PANEL_WIDTH + 3 * COLUMN_GAP
    canvas_height = (
        2 * CANVAS_MARGIN
        + 2 * (TITLE_BAND + panel_height)
        + ROW_GAP
        + LEGEND_BAND
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    title_font = load_font(28)
    legend_font = load_font(18)
    error_font = load_font(16)
    scale_x = PANEL_WIDTH / record.width
    scale_y = panel_height / record.height

    for index, name in enumerate(config.method_order):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (PANEL_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + TITLE_BAND + row * (
            TITLE_BAND + panel_height + ROW_GAP
        )
        panel = source.resize((PANEL_WIDTH, panel_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(panel)
        occupied_labels: list[tuple[int, int, int, int]] = []
        color_priority = {CYAN: 0, RED: 1, YELLOW: 2}
        detection_indices = sorted(
            range(len(methods[name])),
            key=lambda detection_index: color_priority[
                detection_color(
                    name, detection_index, matches, rscdetr_unique_targets
                )
            ],
        )
        for detection_index in detection_indices:
            detection = methods[name][detection_index]
            x1, y1, x2, y2 = detection.xyxy
            display_box = (
                min(PANEL_WIDTH - 1, max(0, round(x1 * scale_x))),
                min(panel_height - 1, max(0, round(y1 * scale_y))),
                min(PANEL_WIDTH - 1, max(0, round(x2 * scale_x))),
                min(panel_height - 1, max(0, round(y2 * scale_y))),
            )
            if display_box[2] <= display_box[0] or display_box[3] <= display_box[1]:
                continue
            draw.rectangle(
                display_box,
                outline=detection_color(
                    name, detection_index, matches, rscdetr_unique_targets
                ),
                width=BOX_WIDTH,
            )
            error_target = category_error_target(detection, ground_truth)
            if error_target is not None:
                predicted_name = category_names.get(detection.category_id, str(detection.category_id))
                draw_error_label(
                    draw,
                    display_box,
                    predicted_name,
                    error_font,
                    PANEL_WIDTH,
                    panel_height,
                    occupied_labels,
                )
        canvas.paste(panel, (panel_left, image_top))
        canvas_draw = ImageDraw.Draw(canvas)
        canvas_draw.text(
            title_position(canvas_draw, name, panel_left, image_top, title_font),
            name,
            fill=TEXT,
            font=title_font,
        )

    legend_top = canvas_height - CANVAS_MARGIN - 22
    draw_legend(canvas, legend_top, legend_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)

    rscdetr_matches = matches["RSC-DETR"]
    print(f"wrote={output}")
    print(
        f"selection={config.name} id={image_id} file={record.file_name} "
        f"size={record.width}x{record.height} gt={len(ground_truth)} "
        f"rscdetr_tp={len(rscdetr_matches)} rscdetr_fp={len(methods['RSC-DETR']) - len(rscdetr_matches)} "
        f"rscdetr_unique={len(rscdetr_unique_targets)}"
    )
    print("order=" + " | ".join(config.method_order))
    print(
        "counts="
        + ", ".join(
            f"{name}:{len(methods[name])}/tp{len(matches[name])}"
            for name in config.method_order
        )
    )
    for target_index in sorted(rscdetr_unique_targets):
        target = ground_truth[target_index]
        box = tuple(round(value, 1) for value in target.xyxy)
        print(
            f"unique_gt={category_names.get(target.category_id, target.category_id)} "
            f"xyxy={box}"
        )


def main() -> None:
    args = parse_args()
    global SCORE_THRESHOLD
    SCORE_THRESHOLD = args.score_threshold

    vedai_images, vedai_gt, vedai_categories, vedai_names = load_dataset(VEDAI)
    vedai_methods = load_methods(VEDAI, vedai_images, vedai_names)
    vedai_image_id, _ = select_vedai(
        vedai_images,
        vedai_gt,
        vedai_methods,
        args.device,
        args.vedai_rsvdet_candidates,
        args.vedai_image_id or 302,
    )

    m3fd_images, m3fd_gt, m3fd_categories, m3fd_names = load_dataset(M3FD)
    m3fd_methods = load_methods(M3FD, m3fd_images, m3fd_names)
    m3fd_image_id, _ = select_m3fd(
        m3fd_images,
        m3fd_gt,
        m3fd_methods,
        args.m3fd_image_id or 36,
    )

    if args.scan_only:
        print(f"selected_vedai={vedai_image_id} selected_m3fd={m3fd_image_id}")
        return

    render(
        VEDAI,
        args.vedai_output,
        vedai_image_id,
        vedai_images,
        vedai_gt,
        vedai_methods,
        vedai_categories,
    )
    render(
        M3FD,
        args.m3fd_output,
        m3fd_image_id,
        m3fd_images,
        m3fd_gt,
        m3fd_methods,
        m3fd_categories,
    )


if __name__ == "__main__":
    main()
