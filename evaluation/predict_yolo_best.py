#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ultralytics validation (with COCO json export) on a trained checkpoint.

    python predict_yolo_best.py --repo <ultralytics fork> --weights best.pt \
        --data <data.yaml> --imgsz 1920 --batch 4 --device 0 \
        --project <dir> --name <run name>
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    os.chdir(str(a.repo))

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS["wandb"] = False

    Path(a.project).mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    metrics = model.val(
        data=a.data,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        save_json=True,
        plots=False,
        project=a.project,
        name=a.name,
        exist_ok=True,
    )
    print("mAP50-95:", metrics.box.map, "mAP50:", metrics.box.map50, "mAP75:", metrics.box.map75)


if __name__ == "__main__":
    main()
