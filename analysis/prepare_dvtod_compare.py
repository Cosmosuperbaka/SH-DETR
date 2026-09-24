#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the unified DVTOD 3-class RGB/IR data roots used by the four
comparison baselines (C2DFF-Net, CMFADet, SFFR, C2Former).

Protocol (mirrors the "common-size" step of the RSC-DETR C2 dual-stream setup):
  * RGB  : datasets/DVTOD dataset/vis1/images/{train,val}   (1920x1080, untouched)
  * IR   : datasets/DVTOD dataset/ir1/images/{train,val}    (640x512)
           resized to 1920x1080 so that both streams share one spatial size.
  * labels: vis1/labels/{train,val} (3 classes: 0 Person, 1 Car, 2 Bicycle),
            shared by both modalities (they are byte-identical in DVTOD).

Two layouts are produced because the upstream repos disagree on naming:
  A) DVTOD_3class_yolo/  images/  + image/   + labels/
     C2DFF-Net derives IR as f.replace("images","image");
     CMFADet reads the explicit `train_ir` / `val_ir` keys.
  B) DVTOD_3class_sffr/  visible/ + infrared/ + labels/
     SFFR's img2label_paths maps 'visible'/'infrared' -> 'labels'.

NOTE: every modality directory is a *real* directory holding per-file symlinks.
Directory-level symlinks must be avoided: ultralytics calls Path(...).resolve()
on the yaml paths, which would collapse a directory symlink back to the raw
dataset and break the `images` -> `image` substitution.
"""
from __future__ import annotations

import os
from pathlib import Path


import cv2

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from rscdetr_paths import ROOT  # noqa: E402


DATASET = ROOT / "datasets/DVTOD dataset"
BASE = ROOT / "compare/DVTOD_compare_20260910"
DATA = BASE / "data"
RGB_IR_SIZE = (1920, 1080)  # (w, h)
SPLITS = ("train", "val")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if os.readlink(dst) == str(src):
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(str(src), str(dst))


def link_tree(src_dir: Path, dst_dir: Path) -> int:
    """Create a real dst_dir filled with per-file symlinks to src_dir."""
    if dst_dir.is_symlink():  # replace a legacy directory-level symlink
        dst_dir.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.is_file():
            link_file(p, dst_dir / p.name)
            n += 1
    return n


def resize_ir(dst_root: Path) -> None:
    """Resize every ir1 image to 1920x1080 into dst_root/<split>."""
    for split in SPLITS:
        src_dir = DATASET / "ir1" / "images" / split
        dst_dir = dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        done = 0
        for p in files:
            out = dst_dir / p.name
            if out.exists() and not out.is_symlink() and out.stat().st_size > 0:
                done += 1
                continue
            if out.is_symlink():
                out.unlink()
            im = cv2.imread(str(p))
            if im is None:
                raise SystemExit(f"[resize_ir] cannot read {p}")
            im = cv2.resize(im, RGB_IR_SIZE, interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(str(out), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise SystemExit(f"[resize_ir] cannot write {out}")
            done += 1
        print(f"[resize_ir] {split}: {done}/{len(files)} -> {dst_dir}")


def build_yolo_layout() -> None:
    root = DATA / "DVTOD_3class_yolo"
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        n_img = link_tree(DATASET / "vis1" / "images" / split, root / "images" / split)
        n_lbl = link_tree(DATASET / "vis1" / "labels" / split, root / "labels" / split)
        print(f"[yolo layout] {split}: images={n_img} labels={n_lbl}")
    resize_ir(root / "image")
    print(f"[yolo layout] ready: {root}")


def build_sffr_layout() -> None:
    root = DATA / "DVTOD_3class_sffr"
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        link_tree(DATASET / "vis1" / "images" / split, root / "visible" / split)
        link_tree(DATASET / "vis1" / "labels" / split, root / "labels" / split)
        # point `infrared` at the already-resized IR produced by the yolo layout
        src_ir = DATA / "DVTOD_3class_yolo" / "image" / split
        n = link_tree(src_ir, root / "infrared" / split)
        print(f"[sffr layout] {split}: infrared={n}")
    print(f"[sffr layout] ready: {root}")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    build_yolo_layout()
    build_sffr_layout()


if __name__ == "__main__":
    main()
