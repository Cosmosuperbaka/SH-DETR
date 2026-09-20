#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


GROUPS = {
    "vedai": {
        "head": ["car", "pickup"],
        "medium": ["camping_car", "truck", "other", "tractor", "boat"],
        "tail": ["van"],
    },
    "m3fd": {
        "head": ["People", "Car"],
        "medium": ["Bus", "Lamp", "Truck"],
        "tail": ["Motorcycle"],
    },
}


def build_gt(images_dir, labels_dir, names, out):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    cats = [{"id": i, "name": n} for i, n in enumerate(names)]
    images, anns = [], []
    ann_id = 1
    for img_id, img_path in enumerate(sorted(images_dir.iterdir()), 1):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        image_id = int(img_path.stem) if img_path.stem.isdigit() else img_path.stem
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})
        lab = labels_dir / f"{img_path.stem}.txt"
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
            })
            ann_id += 1
    gt = {"images": images, "annotations": anns, "categories": cats}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, ensure_ascii=False) + "\n")
    return out


def normalize_preds(pred_path, gt_path, out_path):
    gt = json.loads(Path(gt_path).read_text())
    valid_images = {im["id"] for im in gt["images"]}
    valid_cats = {c["id"] for c in gt["categories"]}
    preds = json.loads(Path(pred_path).read_text())
    fixed = []
    for p in preds:
        q = dict(p)
        if q["image_id"] not in valid_images and isinstance(q["image_id"], str) and q["image_id"].isdigit():
            q["image_id"] = int(q["image_id"])
        if q["category_id"] not in valid_cats and q["category_id"] - 1 in valid_cats:
            q["category_id"] -= 1
        if q["image_id"] in valid_images and q["category_id"] in valid_cats:
            fixed.append(q)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixed, ensure_ascii=False) + "\n")
    return out_path


def summarize(gt_path, pred_path, dataset, out_path):
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    precisions = ev.eval["precision"]
    recalls = ev.eval["recall"]
    per_class = []
    for idx, cid in enumerate(ev.params.catIds):
        p = precisions[:, :, idx, 0, 2]
        r = recalls[:, idx, 0, 2]
        per_class.append({
            "id": int(cid),
            "name": names[cid],
            "ap": float(p[p > -1].mean()) if (p > -1).any() else math.nan,
            "ap50": float(precisions[0, :, idx, 0, 2][precisions[0, :, idx, 0, 2] > -1].mean()) if (precisions[0, :, idx, 0, 2] > -1).any() else math.nan,
            "ap75": float(precisions[5, :, idx, 0, 2][precisions[5, :, idx, 0, 2] > -1].mean()) if (precisions[5, :, idx, 0, 2] > -1).any() else math.nan,
            "ar100": float(r[r > -1].mean()) if (r > -1).any() else math.nan,
        })
    by_name = {r["name"]: r for r in per_class}
    group_summary = {}
    for group, cls_names in GROUPS[dataset].items():
        rows = [by_name[n] for n in cls_names if n in by_name]
        group_summary[group] = {
            "ap": sum(r["ap"] for r in rows) / len(rows),
            "ar100": sum(r["ar100"] for r in rows) / len(rows),
        }
    summary = {
        "stats": [float(x) for x in ev.stats],
        "per_class": per_class,
        "groups": group_summary,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--gt")
    ap.add_argument("--pred")
    ap.add_argument("--dataset", choices=["vedai", "m3fd"])
    ap.add_argument("--out")
    args = ap.parse_args()
    gt = Path(args.gt)
    if not gt.exists():
        build_gt(args.images, args.labels, args.names, gt)
    norm_pred = Path(args.out).with_name(Path(args.out).stem + "_preds_norm.json")
    normalize_preds(args.pred, gt, norm_pred)
    summarize(gt, norm_pred, args.dataset, args.out)


if __name__ == "__main__":
    main()
