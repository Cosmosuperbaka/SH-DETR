#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified COCO evaluation for the DVTOD comparison baselines.

Mirrors ``rtdetrv2_pytorch/tools/eval_per_class_test.py`` exactly (same
``faster_coco_eval`` calls and the same precision/recall slicing), so RT-DETR
predictions and ultralytics predictions are scored identically.

Ultralytics writes ``image_id`` as the numeric file stem of the image, while the
DVTOD COCO annotations use sequential ids.  ``--remap-stem`` converts stems to
annotation ids through ``file_name``; ``--remap-order`` falls back to the sorted
order of the annotation ids (used for SFFR's YOLOv5 output).

Usage:
    python eval_coco_preds.py --gt instances_val.json \
        --pred predictions.json --out per_class.json [--remap-stem]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster


def safe_mean(arr):
    valid = arr[arr > -1]
    return float(valid.mean()) if valid.size else float("nan")


def remap_by_stem(gt_file: str, preds: list) -> list:
    gt = json.load(open(gt_file, "r", encoding="utf-8"))
    stem_to_id = {}
    for im in gt["images"]:
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        try:
            stem_to_id[int(stem)] = int(im["id"])
        except ValueError:
            stem_to_id[stem] = int(im["id"])
    out, missing = [], 0
    for p in preds:
        iid = p["image_id"]
        if iid in stem_to_id:
            q = dict(p)
            q["image_id"] = stem_to_id[iid]
            out.append(q)
        else:
            missing += 1
    if missing:
        print(f"[remap] WARNING {missing}/{len(preds)} predictions could not be mapped")
    return out


def remap_by_order(gt_file: str, preds: list) -> list:
    gt = json.load(open(gt_file, "r", encoding="utf-8"))
    gt_ids = sorted(int(im["id"]) for im in gt["images"])
    pred_ids = sorted({p["image_id"] for p in preds})
    if len(gt_ids) != len(pred_ids):
        raise SystemExit(
            f"[remap-order] id count mismatch: gt={len(gt_ids)} pred={len(pred_ids)}")
    mapping = dict(zip(pred_ids, gt_ids))
    out = []
    for p in preds:
        q = dict(p)
        q["image_id"] = mapping[p["image_id"]]
        out.append(q)
    return out


def summarize(gt_file: str, pred_list: list):
    tmp = Path(gt_file).parent / "_tmp_pred_for_eval.json"
    tmp.write_text(json.dumps(pred_list))
    try:
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(str(tmp))
        ev = COCOeval_faster(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

        precision = ev.eval["precision"]
        recall = ev.eval["recall"]
        stats = [float(x) for x in ev.stats]
        rows = []
        for k, cat in enumerate(coco_gt.loadCats(coco_gt.getCatIds())):
            rows.append({
                "id": cat["id"],
                "name": cat["name"],
                "ap": safe_mean(precision[:, :, k, 0, 2]),
                "ap50": safe_mean(precision[0, :, k, 0, 2]),
                "ap75": safe_mean(precision[5, :, k, 0, 2]),
                "ap90": safe_mean(precision[8, :, k, 0, 2]),
                "aps": safe_mean(precision[:, :, k, 1, 2]),
                "apm": safe_mean(precision[:, :, k, 2, 2]),
                "apl": safe_mean(precision[:, :, k, 3, 2]),
                "ar100": safe_mean(recall[:, k, 0, 2]),
            })
    finally:
        if tmp.exists():
            tmp.unlink()
    return stats, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--remap-stem", action="store_true")
    ap.add_argument("--remap-order", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.0)
    args = ap.parse_args()

    preds = json.load(open(args.pred, "r", encoding="utf-8"))
    if isinstance(preds, dict):
        preds = preds.get("predictions", preds.get("annotations", []))
    print(f"[eval] loaded {len(preds)} predictions from {args.pred}")
    if args.score_thr > 0:
        preds = [p for p in preds if p.get("score", 1.0) >= args.score_thr]
        print(f"[eval] kept {len(preds)} predictions with score >= {args.score_thr}")

    if args.remap_stem:
        preds = remap_by_stem(args.gt, preds)
    elif args.remap_order:
        preds = remap_by_order(args.gt, preds)
    else:
        # auto-detect: already annotation ids -> keep; stems -> map; else by order
        gt = json.load(open(args.gt, "r", encoding="utf-8"))
        gt_ids = {int(im["id"]) for im in gt["images"]}
        pred_ids = {p["image_id"] for p in preds}
        if pred_ids <= gt_ids:
            print("[eval] prediction ids already match the annotation ids; no remap")
        else:
            stems = set()
            for im in gt["images"]:
                stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
                try:
                    stems.add(int(stem))
                except ValueError:
                    stems.add(stem)
            if pred_ids & stems:
                print("[eval] remapping by file stem")
                preds = remap_by_stem(args.gt, preds)
            else:
                print("[eval] remapping by sorted image order")
                preds = remap_by_order(args.gt, preds)

    stats, rows = summarize(args.gt, preds)
    out = {
        "gt": args.gt,
        "pred": args.pred,
        "coco_stats": stats,
        "overall": {
            "AP": stats[0], "AP50": stats[1], "AP75": stats[2],
            "APs": stats[3], "APm": stats[4], "APl": stats[5],
            "AR100": stats[8],
        },
        "per_class": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")

    print("\nClass             AP     AP50    AP75    AP90    APs     APm     APl     AR100")
    print("-" * 86)
    for r in rows:
        print(f"{r['name']:<14s} "
              f"{r['ap']:.4f}  {r['ap50']:.4f}  {r['ap75']:.4f}  {r['ap90']:.4f}  "
              f"{r['aps']:.4f}  {r['apm']:.4f}  {r['apl']:.4f}  {r['ar100']:.4f}")
    print(f"\nOVERALL  AP={stats[0]:.4f}  AP50={stats[1]:.4f}  AP75={stats[2]:.4f}  "
          f"APs={stats[3]:.4f}  APm={stats[4]:.4f}  APl={stats[5]:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
