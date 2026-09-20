#!/usr/bin/env python3
"""Infer VEDAI scene 000004 (fold03 test) with the naive concat-merged RT-DETR baseline
fold03 checkpoint and write COCO-format predictions JSON.

Scene 000004 lives in fold03 test, so the fold03 baseline checkpoint must be used
(baseline-10fold/baseline-fold3/best.pth). The result feeds Fig.1, which should
show the RGB+IR views of this scene with real baseline detection boxes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


import torch

CODE = ROOT / "rtdetrv2_pytorch"
sys.path.insert(0, str(CODE))

from src.core import YAMLConfig, yaml_utils  # noqa: E402


for _parent in Path(__file__).resolve().parents:
    if (_parent / "shdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from shdetr_paths import ROOT  # noqa: E402


CFG = CODE / "configs/rtdetrv2/rtdetrv2_r50vd_vedai_1024_concat_baseline_seed3407_fold03_30e.yml"
CKPT = ROOT / "results/VEDAI/baseline-10fold/baseline-fold3/best.pth"
TARGET_ID = 4
OUT = ROOT / "outputs/vedai_scene000004_baseline_fold03_predictions.json"


def main() -> None:
    device = "cuda:0"
    update_dict = yaml_utils.parse_cli([])
    update_dict.update({"resume": str(CKPT), "device": device})
    cfg = YAMLConfig(str(CFG), **update_dict)
    model = cfg.model
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = state.get("ema", state).get("module", state.get("model", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device)

    preds: list[dict] = []
    val_loader = cfg.val_dataloader
    with torch.no_grad():
        for samples, targets in val_loader:
            if isinstance(samples, dict):
                samples = {k: v.to(device) for k, v in samples.items()}
            else:
                samples = samples.to(device)
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postprocessor(outputs, orig_sizes)
            for tgt, res in zip(targets, results):
                img_id = int(tgt["image_id"].item())
                if img_id != TARGET_ID:
                    continue
                for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
                    x1, y1, x2, y2 = box.tolist()
                    preds.append(
                        {
                            "image_id": img_id,
                            "category_id": int(label.item()),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score.item()),
                        }
                    )
            if any(int(t["image_id"].item()) == TARGET_ID for t in targets):
                break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(preds, indent=2))
    print(f"wrote={OUT} count={len(preds)}")


if __name__ == "__main__":
    main()
