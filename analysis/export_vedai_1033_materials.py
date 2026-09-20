#!/usr/bin/env python3
"""Export VEDAI scene 00001033 as figure-ready RGB/IR image materials."""

import json
import shutil
import zipfile
from pathlib import Path


from PIL import Image, ImageDraw

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS  # noqa: E402


DATA = DATASETS / "VEDAI"
IMAGE_DIR = DATA / "Vehicules1024"
ANN = DATA / "annotations/vedai_fold01_test_class8.json"
OUT = ROOT / "流程图总图素材"
ZIP = ROOT / "流程图总图素材.zip"
FOCUS_OUT = OUT / "1比1清晰截图"
STEM = "00001033"
RED = (230, 32, 32)
BOX_WIDTH = 5


def get_boxes() -> list[list[float]]:
    payload = json.loads(ANN.read_text())
    boxes = []
    for row in payload["annotations"]:
        image_id = row["image_id"]
        if image_id not in (1033, "1033", f"{STEM}_co", f"{STEM}_co.png"):
            continue
        x, y, w, h = map(float, row["bbox"])
        boxes.append([x, y, x + w, y + h])
    if not boxes:
        raise RuntimeError("No annotations found for 00001033")
    return boxes


def draw_boxes(src: Path, dst: Path, boxes: list[list[float]]) -> None:
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=RED, width=BOX_WIDTH)
    image.save(dst, format="PNG", optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FOCUS_OUT.mkdir(parents=True, exist_ok=True)
    rgb = IMAGE_DIR / f"{STEM}_co.png"
    ir = IMAGE_DIR / f"{STEM}_ir.png"
    boxes = get_boxes()

    shutil.copy2(rgb, OUT / "00001033_visible_original.png")
    shutil.copy2(ir, OUT / "00001033_infrared_original.png")
    draw_boxes(rgb, OUT / "00001033_visible_red_boxes.png", boxes)
    draw_boxes(ir, OUT / "00001033_infrared_red_boxes.png", boxes)

    # Square focus crop around all three targets.  The same crop is applied to
    # RGB, IR, and their red-box versions so the panels remain geometrically
    # aligned.  Upscaling makes the small targets readable in a paper figure.
    crop = (300, 624, 700, 1024)  # 400x400, square, includes all GT boxes
    crop_size = (800, 800)
    for name in (
        "00001033_visible_original.png",
        "00001033_infrared_original.png",
        "00001033_visible_red_boxes.png",
        "00001033_infrared_red_boxes.png",
    ):
        image = Image.open(OUT / name).convert("RGB")
        resample = getattr(Image, "Resampling", Image).LANCZOS
        focused = image.crop(crop).resize(crop_size, resample)
        focused.save(FOCUS_OUT / name, format="PNG", optimize=True)

    readme = OUT / "README.txt"
    readme.write_text(
        "VEDAI 00001033 figure materials\n"
        "- visible_original: original RGB/visible image\n"
        "- infrared_original: original infrared image\n"
        "- visible_red_boxes: visible image with ground-truth HBBs in red\n"
        "- infrared_red_boxes: infrared image with the same ground-truth HBBs in red\n"
        "- confidence scores are not displayed\n"
        f"- image size: 1024x1024; boxes: {len(boxes)}\n"
        f"- annotation source: {ANN}\n",
        encoding="utf-8",
    )
    (FOCUS_OUT / "README.txt").write_text(
        "1:1 square focus crops (800x800) for clear inspection of all red boxes.\n"
        "The same crop is used for visible and infrared images.\n"
        "Original-image crop coordinates: left=300, top=624, right=700, bottom=1024.\n",
        encoding="utf-8",
    )

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"流程图总图素材/{path.relative_to(OUT)}")
    print(f"output={OUT}")
    print(f"zip={ZIP}")
    print(f"boxes={len(boxes)}")


if __name__ == "__main__":
    main()
