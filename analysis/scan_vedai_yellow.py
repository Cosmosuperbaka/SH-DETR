#!/usr/bin/env python3
"""Scan the full VEDAI fold01 test set (121 scenes) under the CURRENT prediction
files of 7 methods (CFT, C2DFF-Net, RT-DETR RGB, RT-DETR concat, YOLOv11-RGBT,
LCAFNet, SH-DETR) and report every scene where SH-DETR still owns a
SH-DETR-only true positive (the yellow box condition).

RSVDet is deliberately excluded here: adding a method can only shrink the
unique set, never grow it. Scenes that survive this scan are then re-checked
with fresh RSVDet inference before being used for Fig7.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS, CFT, LCAFNET  # noqa: E402


GT_PATH = DATASETS / "VEDAI/annotations/vedai_fold01_test_class8.json"
IMAGE_DIR = DATASETS / "VEDAI/Vehicules1024"
CLASS8 = {1: "car", 2: "truck", 3: "pickup", 4: "tractor", 5: "camping_car", 6: "boat", 7: "plane", 8: "van"}

MATCH_IOU = 0.50
SCORE_THRESHOLD = 0.50

JSON_SOURCES = {
    "CFT": (CFT / "runs/VEDAI/CFT-26-fold1_valbest_test_20260802/best_predictions.json", 1),
    "C2DFF-Net": (ROOT / "compare/C2DFF_VEDAI/runs/c2dff_s3407_uuid2_valbest_test_20260802/predictions.json", 1),
    "RT-DETR RGB": (ROOT / "outputs/vedai_direct_test_unified/rgb/seed_3407/predictions.json", 0),
    "RT-DETR concat": (ROOT / "outputs/vedai_direct_test_unified/baseline/seed_3407/predictions.json", 0),
    "SH-DETR": (ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json", 0),
}
YOLO_SOURCES = {
    "YOLOv11-RGBT": ROOT / "compare/YOLOv11_RGBT/runs/yolov11_rgbt_vedai_s3407_uuid2_valbest_test_20260802/labels",
    "LCAFNet": LCAFNET / "runs/VEDAI/lcafnet_s3407_uuid_valbest_test_20260802/labels",
}

METHOD_ORDER = list(JSON_SOURCES) + list(YOLO_SOURCES)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def norm_image_id(v) -> int | None:
    """Accept int 25, str '25', str '00000025_co'. Return None if not parseable."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    text = str(v)
    if text.endswith("_co"):
        text = text[:-3]
    if text.endswith(".txt"):
        text = text[:-4]
    if text.endswith(".png"):
        text = text[:-4]
    if not text.isdigit():
        return None
    return int(text)


def clip(box, W, H):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None
    return (max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2))


def load_gt():
    payload = json.load(open(GT_PATH))
    gt_by_img = {}
    cats = {a["id"]: c["name"] for c in payload["categories"] for a in [{"id": c["id"]}]}
    for ann in payload["annotations"]:
        img_id = norm_image_id(ann["image_id"])
        if img_id is None:
            continue
        x, y, w, h = ann["bbox"]
        box = clip((x, y, x + w, y + h), 1024, 1024)
        if box is None:
            continue
        gt_by_img.setdefault(img_id, []).append({"cat": ann["category_id"], "xyxy": box})
    return gt_by_img


def load_all_methods():
    methods = {name: {} for name in METHOD_ORDER}
    # JSON sources
    for name, (path, offset) in JSON_SOURCES.items():
        payload = json.load(open(path))
        rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
        for r in rows:
            img_id = norm_image_id(r.get("image_id"))
            score = float(r.get("score", 0.0))
            if img_id is None or score < SCORE_THRESHOLD:
                continue
            bb = [float(v) for v in r["bbox"]]
            box = clip((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]), 1024, 1024)
            if box is None:
                continue
            methods[name].setdefault(img_id, []).append({"cat": int(r.get("category_id", 0)) + offset, "score": score, "xyxy": box})
    # YOLO sources
    for name, d in YOLO_SOURCES.items():
        for txt in sorted(d.glob("*_co.txt")):
            img_id = norm_image_id(txt.stem)
            for line in txt.read_text().splitlines():
                f = line.split()
                if len(f) < 6:
                    continue
                cat, cx, cy, w, h, sc = map(float, f[:6])
                if sc < SCORE_THRESHOLD:
                    continue
                box = clip(((cx - w / 2) * 1024, (cy - h / 2) * 1024, (cx + w / 2) * 1024, (cy + h / 2) * 1024), 1024, 1024)
                if box is None:
                    continue
                methods[name].setdefault(img_id, []).append({"cat": int(cat) + 1, "score": sc, "xyxy": box})
    return methods


def match(dets, gts):
    unmatched = set(range(len(gts)))
    m = {}
    for d in sorted(dets, key=lambda x: -x["score"]):
        cand = [t for t in unmatched if gts[t]["cat"] == d["cat"] and iou(gts[t]["xyxy"], d["xyxy"]) >= MATCH_IOU]
        if cand:
            t = max(cand, key=lambda i: iou(gts[i]["xyxy"], d["xyxy"]))
            m[t] = 1
            unmatched.remove(t)
    return m


def main():
    gt_by_img = load_gt()
    methods = load_all_methods()
    print(f"gt scenes={len(gt_by_img)} method_order={METHOD_ORDER}", flush=True)

    tp_total = {n: 0 for n in METHOD_ORDER}
    det_scenes = {n: 0 for n in METHOD_ORDER}
    shdetr_tp_scenes = 0
    rows = []
    for img_id, gts in sorted(gt_by_img.items()):
        ms = {name: match(methods[name].get(img_id, []), gts) for name in METHOD_ORDER}
        for n in METHOD_ORDER:
            tp_total[n] += len(ms[n])
            if methods[n].get(img_id):
                det_scenes[n] += 1
        other = {t for name in METHOD_ORDER if name != "SH-DETR" for t in ms[name]}
        shdetr = ms["SH-DETR"]
        unique = sorted(set(shdetr).difference(other))
        if shdetr:
            shdetr_tp_scenes += 1
        if not unique:
            continue
        unique_cats = [CLASS8.get(gts[t]["cat"], str(gts[t]["cat"])) for t in unique]
        gt_cats = sorted(set(CLASS8.get(g["cat"], str(g["cat"])) for g in gts))
        rows.append((img_id, len(gts), len(unique), unique_cats, gt_cats,
                     {n: len(methods[n].get(img_id, [])) for n in METHOD_ORDER}))

    print(f"\nper-method total TP over fold01 test: {tp_total}")
    print(f"scenes with >=1 det per method: {det_scenes}")
    print(f"scenes where SH-DETR has >=1 TP: {shdetr_tp_scenes}")
    rows.sort(key=lambda r: (-r[2], -r[1]))
    print(f"\n=== scenes with SH-DETR-only unique TP (no RSVDet yet): {len(rows)} ===")
    for img_id, ng, nu, ucats, gcats, counts in rows:
        print(f"id={img_id:>6} file={img_id:08d}_co  gt={ng} unique={nu} unique_cats={ucats} gt_cats={gcats}")
        print(f"       counts={counts}")


if __name__ == "__main__":
    main()
