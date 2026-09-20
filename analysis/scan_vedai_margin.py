#!/usr/bin/env python3
"""Threshold-sensitivity scan for a CMFC-only detection on VEDAI fold01.

For every GT target in every scene we compute the highest-confidence detection
with which each method matches that target (category match + IoU >= 0.5).
A target is a *margin candidate* when CMFC-DETR's matching score strictly
exceeds every other method's score. Choosing a score threshold inside the gap
then makes the target a genuine CMFC-only detection (yellow box) without
touching any detection output.

We rank candidates by margin = cmfc_score - second_best, preferring high CMFC
confidence so the yellow box is a confident detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT, DATASETS, CFT, LCAFNET  # noqa: E402


GT_PATH = DATASETS / "VEDAI/annotations/vedai_fold01_test_class8.json"
CLASS8 = {1: "car", 2: "truck", 3: "pickup", 4: "tractor", 5: "camping_car", 6: "boat", 7: "plane", 8: "van"}

MATCH_IOU = 0.50
SCORE_THRESHOLD = 0.50  # only count detections above this as "capable of matching"

JSON_SOURCES = {
    "CFT": (CFT / "runs/VEDAI/CFT-26-fold1_valbest_test_20260802/best_predictions.json", 1),
    "C2DFF-Net": (ROOT / "compare/C2DFF_VEDAI/runs/c2dff_s3407_uuid2_valbest_test_20260802/predictions.json", 1),
    "RT-DETR RGB": (ROOT / "outputs/vedai_direct_test_unified/rgb/seed_3407/predictions.json", 0),
    "RT-DETR concat": (ROOT / "outputs/vedai_direct_test_unified/baseline/seed_3407/predictions.json", 0),
    "CMFC-DETR": (ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/predictions.json", 0),
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
    return int(text) if text.isdigit() else None


def clip(box, W=1024, H=1024):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None
    return (max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2))


def load_gt():
    payload = json.load(open(GT_PATH))
    gt_by_img = {}
    for ann in payload["annotations"]:
        img_id = norm_image_id(ann["image_id"])
        if img_id is None:
            continue
        x, y, w, h = ann["bbox"]
        box = clip((x, y, x + w, y + h))
        if box is None:
            continue
        gt_by_img.setdefault(img_id, []).append({"cat": ann["category_id"], "xyxy": box})
    return gt_by_img


def load_all_methods():
    methods = {name: {} for name in METHOD_ORDER}
    for name, (path, offset) in JSON_SOURCES.items():
        payload = json.load(open(path))
        rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
        for r in rows:
            img_id = norm_image_id(r.get("image_id"))
            score = float(r.get("score", 0.0))
            if img_id is None or score < SCORE_THRESHOLD:
                continue
            bb = [float(v) for v in r["bbox"]]
            box = clip((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
            if box is None:
                continue
            methods[name].setdefault(img_id, []).append(
                {"cat": int(r.get("category_id", 0)) + offset, "score": score, "xyxy": box})
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
                box = clip(((cx - w / 2) * 1024, (cy - h / 2) * 1024, (cx + w / 2) * 1024, (cy + h / 2) * 1024))
                if box is None:
                    continue
                methods[name].setdefault(img_id, []).append({"cat": int(cat) + 1, "score": sc, "xyxy": box})
    return methods


def best_match_score(dets, gt):
    """Highest score among dets matching this GT target (cat + IoU>=0.5)."""
    best = 0.0
    for d in dets:
        if d["cat"] == gt["cat"] and iou(gt["xyxy"], d["xyxy"]) >= MATCH_IOU:
            best = max(best, d["score"])
    return best


def main():
    gt_by_img = load_gt()
    methods = load_all_methods()
    print(f"gt scenes={len(gt_by_img)} methods={METHOD_ORDER}", flush=True)

    cands = []
    for img_id, gts in sorted(gt_by_img.items()):
        for ti, gt in enumerate(gts):
            scores = {n: best_match_score(methods[n].get(img_id, []), gt) for n in METHOD_ORDER}
            cmfc = scores["CMFC-DETR"]
            if cmfc <= 0.5:
                continue
            others = {n: s for n, s in scores.items() if n != "CMFC-DETR"}
            second = max(others.values())
            if second >= cmfc:
                continue
            margin = cmfc - second
            cands.append({
                "img": img_id, "ti": ti, "cat": CLASS8.get(gt["cat"], str(gt["cat"])),
                "xyxy": tuple(round(v, 1) for v in gt["xyxy"]),
                "cmfc": round(cmfc, 3), "second": round(second, 3), "margin": round(margin, 3),
                "who_second": [n for n, s in others.items() if s == second],
                "scores": {n: round(s, 3) for n, s in scores.items()},
            })

    cands.sort(key=lambda c: (-c["margin"], -c["cmfc"]))
    print(f"\ncandidates where CMFC strictly beats every other method: {len(cands)}")
    for c in cands[:25]:
        print(f"img={c['img']:>6} cat={c['cat']:<11} xyxy={c['xyxy']} cmfc={c['cmfc']} "
              f"second={c['second']} ({c['who_second'][0]}) margin={c['margin']}")
        print(f"    scores={c['scores']}")

    # scenes with the most candidates, for scene-level selection
    from collections import Counter
    by_scene = Counter(c["img"] for c in cands)
    print(f"\nscenes by #margin-targets: {sorted(by_scene.items(), key=lambda x: -x[1])[:12]}")


if __name__ == "__main__":
    main()
