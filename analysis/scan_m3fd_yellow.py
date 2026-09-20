#!/usr/bin/env python3
"""Scan M3FD-LT20 test set for CMFC-only detections.

Reports (a) targets that are NATURALLY CMFC-only at SCORE_THRESHOLD=0.5
(yellow-box candidates with no threshold trickery) and (b) margin candidates
where CMFC strictly beats every other method's confidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

ROOT = Path("/home/denglingjun/re-detr-last")
GT_PATH = Path("/home/denglingjun/vedai_data/datasets/M3FD/processed/lt20_seed42/annotations/instances_test.json")
W, H = 1024, 768
NAMES = {0: "People", 1: "Car", 2: "Bus", 3: "Lamp", 4: "Motorcycle", 5: "Truck"}

MATCH_IOU = 0.50
SCORE_THRESHOLD = 0.50

JSON_SOURCES = {
    "RT-DETR RGB": (ROOT / "compare/M3FD-LT20/original-size-45e-trial/rtdetr-rgb-seed42-native-b8-45e/valbest_test_coco/predictions.json", 0),
    "RT-DETR concat": (ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/baseline/predictions.json", 0),
    "CLDyN+RT-DETR": (ROOT / "compare/CLDyN_M3FD-lt20/cldyn-1/eval_m3fd_map/cldyn-vfn-rtdetr-1/val_best_test_per_class/predictions.json", 0),
    "CMFC-DETR": (ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/predictions.json", 0),
}
YOLO_SOURCES = {
    "CFT": (Path("/home/denglingjun/multispectral-object-detection/runs/M3FD-LT20/cft_x3_s42_300e_b8_1024_valbest_test_20260802/labels"), 0),
    "C2DFF-Net": (ROOT / "compare/rerun_m3fd_s42_protocol_20260802/C2DFF_best_test_s42_img1024_b8/labels", 0),
    "YOLOv11-RGBT": (ROOT / "compare/rerun_m3fd_s42_protocol_20260802/YOLOv11_RGBT_best_test_s42_img1024_b8/labels", 0),
    "LCAFNet": (Path("/home/denglingjun/LCAFNet/runs/M3FD-LT20/lcafnet_s42_b4_1024_plus100_uuid4_valbest_test_20260802/labels"), 0),
}
METHOD_ORDER = ["CFT", "RT-DETR RGB", "RT-DETR concat", "C2DFF-Net", "YOLOv11-RGBT", "LCAFNet", "CLDyN+RT-DETR", "CMFC-DETR"]


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def norm_image_id(v, names_to_ids=None) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v in (names_to_ids or {}) else v
    text = str(v)
    for suf in ("_co", ".png", ".jpg", ".txt"):
        if text.endswith(suf):
            text = text[:-len(suf)]
    if names_to_ids is not None:
        if text in names_to_ids:
            return names_to_ids[text]
    return int(text) if text.isdigit() else None


def clip(box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None
    return (max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2))


def load_gt():
    payload = json.load(open(GT_PATH))
    names_to_ids = {}
    gt_by_img = {}
    for row in payload["images"]:
        img_id = int(row["id"])
        fn = str(row["file_name"])
        names_to_ids[fn] = img_id
        names_to_ids[Path(fn).stem] = img_id
        names_to_ids[str(img_id)] = img_id
        names_to_ids[f"{img_id:08d}"] = img_id
    for ann in payload["annotations"]:
        img_id = norm_image_id(ann["image_id"], names_to_ids)
        if img_id is None:
            continue
        x, y, w, h = ann["bbox"]
        box = clip((x, y, x + w, y + h))
        if box is None:
            continue
        gt_by_img.setdefault(img_id, []).append({"cat": ann["category_id"], "xyxy": box})
    return gt_by_img, names_to_ids


def load_all_methods(names_to_ids):
    methods = {name: {} for name in METHOD_ORDER}
    for name, (path, offset) in JSON_SOURCES.items():
        payload = json.load(open(path))
        rows = payload if isinstance(payload, list) else payload.get("predictions", payload.get("annotations", []))
        for r in rows:
            img_id = norm_image_id(r.get("image_id"), names_to_ids)
            score = float(r.get("score", 0.0))
            if img_id is None or score < SCORE_THRESHOLD:
                continue
            bb = [float(v) for v in r["bbox"]]
            box = clip((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
            if box is None:
                continue
            methods[name].setdefault(img_id, []).append(
                {"cat": int(r.get("category_id", 0)) + offset, "score": score, "xyxy": box})
    for name, (d, offset) in YOLO_SOURCES.items():
        for txt in sorted(d.glob("*.txt")):
            img_id = norm_image_id(txt.stem, names_to_ids)
            if img_id is None:
                continue
            for line in txt.read_text().splitlines():
                f = line.split()
                if len(f) < 6:
                    continue
                cat, cx, cy, w, h, sc = map(float, f[:6])
                if sc < SCORE_THRESHOLD:
                    continue
                box = clip(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
                if box is None:
                    continue
                methods[name].setdefault(img_id, []).append({"cat": int(cat) + offset, "score": sc, "xyxy": box})
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


def best_match_score(dets, gt):
    best = 0.0
    for d in dets:
        if d["cat"] == gt["cat"] and iou(gt["xyxy"], d["xyxy"]) >= MATCH_IOU:
            best = max(best, d["score"])
    return best


def main():
    gt_by_img, names_to_ids = load_gt()
    methods = load_all_methods(names_to_ids)
    print(f"gt scenes={len(gt_by_img)}", flush=True)

    natural = []      # unique at 0.5 threshold
    margins = []      # cmfc strictly > everyone, cmfc>0.5
    tp_total = {n: 0 for n in METHOD_ORDER}
    for img_id, gts in sorted(gt_by_img.items()):
        ms = {name: match(methods[name].get(img_id, []), gts) for name in METHOD_ORDER}
        for n in METHOD_ORDER:
            tp_total[n] += len(ms[n])
        other = {t for name in METHOD_ORDER if name != "CMFC-DETR" for t in ms[name]}
        for ti, gt in enumerate(gts):
            scores = {n: best_match_score(methods[n].get(img_id, []), gt) for n in METHOD_ORDER}
            cmfc = scores["CMFC-DETR"]
            others = {n: s for n, s in scores.items() if n != "CMFC-DETR"}
            second = max(others.values())
            if cmfc <= 0.5:
                continue
            if ti in ms["CMFC-DETR"] and ti not in other:
                natural.append({"img": img_id, "cat": NAMES.get(gt["cat"], str(gt["cat"])),
                                "xyxy": tuple(round(v, 1) for v in gt["xyxy"]), "cmfc": round(cmfc, 3)})
            if second < cmfc:
                margins.append({"img": img_id, "cat": NAMES.get(gt["cat"], str(gt["cat"])),
                                "xyxy": tuple(round(v, 1) for v in gt["xyxy"]),
                                "cmfc": round(cmfc, 3), "second": round(second, 3),
                                "margin": round(cmfc - second, 3),
                                "who": [n for n, s in others.items() if s == second]})

    print(f"\nper-method total TP (M3FD-LT20 test): {tp_total}")
    print(f"\nNATURAL CMFC-only TP at thr=0.5: {len(natural)}")
    for c in sorted(natural, key=lambda c: -c["cmfc"])[:30]:
        print(f"  img={c['img']:>4} cat={c['cat']:<10} xyxy={c['xyxy']} cmfc={c['cmfc']}")
    print(f"\nMARGIN candidates (cmfc strictly best): {len(margins)}")
    for c in sorted(margins, key=lambda c: -c["margin"])[:30]:
        print(f"  img={c['img']:>4} cat={c['cat']:<10} xyxy={c['xyxy']} cmfc={c['cmfc']} "
              f"second={c['second']} ({c['who'][0]}) margin={c['margin']}")
    by_scene = Counter(c["img"] for c in natural)
    print(f"\nscenes with natural unique: {dict(sorted(by_scene.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
