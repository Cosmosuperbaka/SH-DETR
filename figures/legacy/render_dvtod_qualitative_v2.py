#!/usr/bin/env python3
"""DVTOD qualitative comparison (2x4 grid, RGB over IR per method).

Same visual language as the VEDAI / M3FD-LT20 qualitative figures already in
the paper: one panel per method, visible (RGB) on top and infrared (IR) below,
red boxes for correct detections, blue for false detections, and a yellow box
for a true positive that only SH-DETR recovers.  Nimbus Roman text and the same
panel geometry.

Two id conventions have to be reconciled: the RT-DETR family writes the
sequential COCO id (1..573), while the YOLO-family runs write the file stem
(e.g. 1005).  Everything is normalised onto the file stem here.

Usage:
    python dvtod_qualitative.py --scan                 # rank candidate scenes
    python dvtod_qualitative.py --stem 2016 --out out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/denglingjun/re-detr-last")
B = ROOT / "compare/DVTOD_compare_20260910"

GT_PATH = ROOT / "datasets/RTDOD_HBB_3class/annotations/instances_val.json"
RGB_DIR = B / "data/DVTOD_3class_yolo/images/val"       # 1920x1080 visible
IR_DIR = B / "data/DVTOD_3class_yolo/image/val"         # thermal, resized to 1920x1080

# Panel order mirrors Table IV (Ours last).
PANELS = (
    ("RT-DETR concat", ROOT / "result/RTDOD_HBB_3class/perclass_concat_best/predictions.json"),
    ("C²Former",       B / "results/c2former_raw/predictions.json"),
    ("RT-DETR (IR)",   ROOT / "result/RTDOD_HBB_3class/perclass_ir_best/predictions.json"),
    ("SFFR",           B / "runs/sffr_eval/sffr_final/best_predictions.json"),
    ("C2DFF-Net",      B / "runs/c2dff_eval/c2dff_final/predictions.json"),
    ("DARFNet",        B / "runs/darfnet_eval/darfnet_final/best_predictions.json"),
    ("CMFADet",        B / "runs/cmfadet_eval/cmfadet_100e/predictions.json"),
    ("SH-DETR",        ROOT / "result/RTDOD_HBB_3class/perclass_sh_c2_best/predictions.json"),
)
OURS = "SH-DETR"

CLASS_NAMES = {0: "Person", 1: "Car", 2: "Bicycle"}

SCORE_THRESHOLD = 0.50
MATCH_IOU = 0.50
RED = (230, 32, 32)
BLUE = (0, 132, 220)
YELLOW = (255, 214, 0)
WHITE = (255, 255, 255)
TEXT = (17, 17, 17)

MODALITY_WIDTH = 400
MODALITY_GAP = 8
TITLE_BAND = 46
TITLE_GAP = 8
CANVAS_MARGIN = 24
COLUMN_GAP = 14
ROW_GAP = 30
LEGEND_BAND = 46
BOX_WIDTH = 4
LABEL_FONT_SIZE = 20
MAX_LABELS_PER_PANEL = 3

FONT_DIR = Path("/usr/share/fonts/opentype/urw-base35")


# --------------------------------------------------------------------- data
def load_gt():
    payload = json.loads(GT_PATH.read_text())
    stem_by_id = {}
    gt = {}
    for im in payload["images"]:
        stem = Path(im["file_name"]).stem
        stem_by_id[int(im["id"])] = stem
        gt[stem] = []
    for a in payload["annotations"]:
        stem = stem_by_id[int(a["image_id"])]
        x, y, w, h = map(float, a["bbox"])
        gt[stem].append(((x, y, x + w, y + h), int(a["category_id"])))
    return gt, stem_by_id


def load_preds(path: Path, stem_by_id, w=1920, h=1080):
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("predictions", payload)
    by_stem: dict[str, list] = {}
    for r in rows:
        iid = r["image_id"]
        stem = stem_by_id.get(int(iid)) if isinstance(iid, int) else str(iid)
        if stem is None:
            stem = str(iid)
        score = float(r.get("score", 0.0))
        if score < SCORE_THRESHOLD:
            continue
        x, y, bw, bh = map(float, r["bbox"])
        box = (max(0.0, x), max(0.0, y), min(x + bw, w), min(y + bh, h))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        by_stem.setdefault(stem, []).append((box, score, int(r["category_id"])))
    return by_stem


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def matched_gt(dets, gt):
    """Return the set of GT indices matched by this method's detections."""
    hit = set()
    for box, _s, cat in dets:
        best, best_iou = None, MATCH_IOU
        for gi, (gbox, gcat) in enumerate(gt):
            if gcat != cat or gi in hit:
                continue
            iou = box_iou(box, gbox)
            if iou >= best_iou:
                best, best_iou = gi, iou
        if best is not None:
            hit.add(best)
    return hit


# ------------------------------------------------------------------ ranking
def scan(gt, methods, top=12):
    rows = []
    for stem, g in gt.items():
        if not g:
            continue
        found = {n: matched_gt(methods[n].get(stem, []), g) for n, _ in PANELS}
        others = set().union(*[v for n, v in found.items() if n != OURS])
        ours = found[OURS]
        unique = ours - others
        rows.append((len(unique), len(ours), len(g), stem,
                     len(others), len(ours) - len(others)))
    rows.sort(key=lambda r: (-r[0], -r[1], abs(r[2] - 6), -r[2]))
    print(f"{'uniq':>5} {'ours':>5} {'gt':>4} {'stem':>6} {'others':>7} {'diff':>5}")
    for r in rows[:top]:
        print(f"{r[0]:>5} {r[1]:>5} {r[2]:>4} {r[3]:>6} {r[4]:>7} {r[5]:>5}")
    return rows


# ----------------------------------------------------------------- drawing
def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "NimbusRoman-Bold" if bold else "NimbusRoman-Regular"
    path = FONT_DIR / f"{name}.otf"
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Nimbus Roman not found: {path}")


def draw_detections(draw, detections, gt, others, label_font,
                    panel_w, panel_h, is_ours):
    rendered = []
    for box, score, cat in detections:
        sx = panel_w / 1920.0
        sy = panel_h / 1080.0
        display_box = (
            min(panel_w - 1, max(0, round(box[0] * sx))),
            min(panel_h - 1, max(0, round(box[1] * sy))),
            min(panel_w - 1, max(0, round(box[2] * sx))),
            min(panel_h - 1, max(0, round(box[3] * sy))),
        )
        matched = [g for g in gt if g[1] == cat and box_iou(box, g[0]) >= MATCH_IOU]
        if matched:
            target = max(matched, key=lambda g: box_iou(box, g[0]))
            shared = any(o[2] == target[1] and box_iou(o[0], target[0]) >= MATCH_IOU
                         for o in others)
            color = YELLOW if is_ours and not shared else RED
        else:
            color = BLUE
        draw.rectangle(display_box, outline=color, width=BOX_WIDTH)
        rendered.append((display_box, score, cat, color))

    color_priority = {YELLOW: 0, BLUE: 1, RED: 2}
    selected = sorted(rendered, key=lambda item: (color_priority[item[3]], -item[1]))[:MAX_LABELS_PER_PANEL]
    occupied = []
    for display_box, score, cat, color in selected:
        label = f"{CLASS_NAMES.get(cat, f'class-{cat}')} {score:.2f}"
        tb = draw.textbbox((0, 0), label, font=label_font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        x1, y1, x2, y2 = display_box
        candidates = [(x1, y1 - text_h - 5), (x1, y2 + 4),
                      (x2 + 4, y1), (x1 - text_w - 4, y1)]
        chosen = None
        for x, y in candidates:
            x = min(max(2, x), max(2, panel_w - text_w - 3))
            y = min(max(2, y), max(2, panel_h - text_h - 3))
            rect = (x - 1, y - 1, x + text_w + 2, y + text_h + 2)
            overlaps = any(not (rect[2] <= old[0] or old[2] <= rect[0] or
                                rect[3] <= old[1] or old[3] <= rect[1])
                           for old in occupied)
            if not overlaps:
                chosen = (x, y, rect)
                break
        if chosen is None:
            continue
        label_x, label_y, rect = chosen
        draw.rectangle(rect, fill=color, outline=color, width=1)
        draw.text((label_x - tb[0], label_y - tb[1]), label,
                  fill=(TEXT if color == YELLOW else WHITE), font=label_font)
        occupied.append(rect)


def draw_legend(canvas, top, font):
    draw = ImageDraw.Draw(canvas)
    entries = ((RED, "Correct detection"), (BLUE, "False detection"),
               (YELLOW, "SH-DETR-only true positive"))
    swatch, gap = 18, 28
    widths = []
    for _, label in entries:
        tb = draw.textbbox((0, 0), label, font=font)
        widths.append(swatch + 8 + tb[2] - tb[0])
    left = (canvas.width - (sum(widths) + gap * (len(entries) - 1))) // 2
    for color, label in entries:
        draw.rectangle((left, top, left + swatch, top + swatch), outline=color, width=3)
        tb = draw.textbbox((0, 0), label, font=font)
        draw.text((left + swatch + 8, top - tb[1] + (swatch - (tb[3] - tb[1])) // 2),
                  label, fill=TEXT, font=font)
        left += swatch + 8 + tb[2] - tb[0] + gap


def render(stem, gt_one, methods, out_path, title_suffix=""):
    rgb_path = RGB_DIR / f"{stem}.jpg"
    ir_path = IR_DIR / f"{stem}.jpg"
    rgb = Image.open(rgb_path).convert("RGB")
    ir = Image.open(ir_path).convert("RGB")
    assert rgb.size == (1920, 1080), rgb.size
    assert ir.size == (1920, 1080), ir.size

    modality_height = round(MODALITY_WIDTH * 1080 / 1920)
    others = [d for name, _ in PANELS if name != OURS for d in methods[name].get(stem, [])]

    panel_height = TITLE_BAND + modality_height + MODALITY_GAP + modality_height
    canvas_width = 2 * CANVAS_MARGIN + 4 * MODALITY_WIDTH + 3 * COLUMN_GAP
    canvas_height = 2 * CANVAS_MARGIN + 2 * panel_height + ROW_GAP + LEGEND_BAND
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    font_title = load_font(28)
    font_label = load_font(LABEL_FONT_SIZE)
    font_legend = load_font(22)

    for index, (name, _path) in enumerate(PANELS):
        row, column = divmod(index, 4)
        panel_left = CANVAS_MARGIN + column * (MODALITY_WIDTH + COLUMN_GAP)
        image_top = CANVAS_MARGIN + row * (panel_height + ROW_GAP)
        is_ours = name == OURS
        dets = methods[name].get(stem, [])

        for offset, src in ((0, rgb), (MODALITY_GAP + modality_height, ir)):
            panel = src.resize((MODALITY_WIDTH, modality_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(panel)
            draw_detections(draw, dets, gt_one, others, font_label,
                            MODALITY_WIDTH, modality_height, is_ours)
            canvas.paste(panel, (panel_left, image_top + TITLE_BAND + offset))

        canvas_draw = ImageDraw.Draw(canvas)
        label = name + title_suffix
        tb = canvas_draw.textbbox((0, 0), label, font=font_title)
        title_x = panel_left + (MODALITY_WIDTH - (tb[2] - tb[0])) // 2
        title_y = image_top + TITLE_BAND - TITLE_GAP - (tb[3] - tb[1]) - tb[1]
        canvas_draw.text((title_x, title_y), label, fill=TEXT, font=font_title)

    draw_legend(canvas, canvas_height - CANVAS_MARGIN - 20, font_legend)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"wrote {out_path}  {canvas_width}x{canvas_height}  stem={stem}")
    print("counts=" + ", ".join(f"{n}:{len(methods[n].get(stem, []))}" for n, _ in PANELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--stem", default="")
    ap.add_argument("--out", default="/home/denglingjun/re-detr-last/out_qual/dvtod_qual.png")
    args = ap.parse_args()

    gt, stem_by_id = load_gt()
    methods = {name: load_preds(path, stem_by_id) for name, path in PANELS}

    if args.scan:
        scan(gt, methods)
        return
    if not args.stem:
        raise SystemExit("need --stem or --scan")
    render(args.stem, gt[args.stem], methods, args.out)


if __name__ == "__main__":
    main()
