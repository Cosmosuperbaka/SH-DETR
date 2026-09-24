# SH-DETR

Reference code for the paper

> **Symmetric Fusion and Reliability-Aware Supervision for Long-Tailed RGB–IR
> Remote-Sensing Detection**

SH-DETR is a dual-stream RGB–IR detector for aerial remote sensing. It keeps the
RT-DETR inference pathway unchanged and adds three training-side components plus
a harmonizer that coordinates them.

| Component | Role | Stage |
|---|---|---|
| **SPSF** — Shared–Private Symmetric Fusion | Models a shared semantic centre and symmetric modality-specific deviations at the `S5` feature level, reducing redundant cross-modal responses before the encoder. | Feature fusion |
| **PFHM** — Prototype-guided Fine-localization Harmonization | Transfers the localization quality of matched queries into prototype-based classification supervision through IoU-aware soft targets. | Training supervision |
| **ACRHM** — Annotation-guided Class-reliability Harmonization | Combines training-annotation priors with detached online matching states to estimate class reliability and regulate class-wise supervision under long-tailed distributions. | Training supervision |
| **Harmonizer** | Derives gates and a bounded class-wise supervision budget from training progress, prototype maturity, localization quality, assignment stability, and annotation evidence, and routes the total loss between the native and budget forms. | Training control |

Inference uses only the two-stream backbones, SPSF, the encoder, the decoder and
the main detection heads — no auxiliary branch is added at test time.

Reported results (AP, %):

| Dataset | Backbone-scale setting | AP | AP₅₀ | AP₇₅ | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| VEDAI | 1024 | **59.31** | — | — | 89.6 | 748 |
| M3FD-LT20 | 1024 | **55.90** | — | — | — | — |
| DVTOD | 1920 | **56.66** | 89.49 | 61.21 | 90.10 | 1485.5 |

The RT-DETR concat baseline is improved by +1.78 / +0.88 / +0.40 AP points on the
three datasets respectively.

---

## Repository layout

This repository contains the **figure-rendering and evaluation code** used to
produce the paper. It intentionally excludes datasets, training runs and model
weights — those live on the experiment server (see
[docs/PATHS.md](docs/PATHS.md)).

```
SH-DETR/
├── figures/
│   ├── render_vedai_qualitative.py     # Fig. 7  (VEDAI qualitative comparison)
│   ├── render_m3fd_qualitative.py      # Fig. 8  (M3FD-LT20 qualitative comparison)
│   ├── render_dvtod_qualitative.py     # Fig. 9  (DVTOD qualitative comparison)
│   └── legacy/                         # earlier renderers, kept for provenance
├── evaluation/
│   ├── eval_coco_preds.py              # COCO AP / AP50 / AP75 and per-class AP
│   ├── build_dvtod_table.py            # assembles the DVTOD comparison table
│   ├── measure_complexity.py           # Params / GFLOPs / latency
│   ├── summarize_yolo_compare_coco.py  # aggregates YOLO-family comparison runs
│   └── predict_yolo_best.py            # runs inference from a YOLO best.pt
├── analysis/
│   ├── plot_error_composition.py       # error-composition breakdown
│   ├── plot_paper_analysis_candidates.py
│   ├── generate_combined_distribution.py  # class-distribution figure
│   ├── infer_scene000004_baseline.py
│   ├── prepare_dvtod_compare.py
│   ├── scan_vedai_yellow.py            # scene ranking for qualitative figures
│   ├── scan_m3fd_yellow.py
│   ├── scan_vedai_margin.py
│   └── export_vedai_1033_materials.py
└── docs/
    └── PATHS.md                        # dataset / checkpoint / prediction layout
```

### About the three qualitative figures

All three figures share one visual language:

* one panel per method, **visible (RGB) on top and infrared (IR) below**;
* **red** = correct detection, **blue** = false detection,
  **green dashed** = ground truth,
  **yellow** = a true positive recovered only by SH-DETR;
* every geometry constant is multiplied by `SCALE = 2`, so the exported PNG
  carries roughly twice the pixels of the earlier 240 px / 330 px / 400 px
  thumbnails and stays crisp at `\textwidth` in print.

Each renderer is standalone and only needs `Pillow` plus the paths defined at the
top of the file.

```bash
# on the experiment server, from any directory
python3 render_vedai_qualitative.py     # -> out_qual/fig7_vedai_1033_v3.png
python3 render_m3fd_qualitative.py      # -> out_qual/fig8_m3fd_00400_v3.png
python3 render_dvtod_qualitative.py     # -> out_qual/fig9_dvtod_1857_v3.png
```

The three scripts were validated against Pillow 9.0.1 (Python 3.10); they use
`Image.LANCZOS`, which works on both the 9.x and 10.x series.

---

## Environment

```bash
python3 -m pip install -r requirements.txt
```

`Pillow` is the only hard dependency of the figure renderers. The evaluation and
analysis scripts additionally expect a working PyTorch / Ultralytics environment
when they have to run inference instead of reading cached predictions.

## Data and checkpoints

Nothing in this repository downloads data automatically, and **no experiment
path is hard-coded**. Every location is resolved at import time by
[`shdetr_paths.py`](shdetr_paths.py), which reads environment variables and
falls back to defaults derived from the current user's home directory:

| Variable | Meaning | Default |
|---|---|---|
| `SHDETR_WORKSPACE` | Root holding all experiment trees | `~` |
| `SHDETR_ROOT` | Main SH-DETR working repository | `$SHDETR_WORKSPACE/sh-detr` |
| `SHDETR_DATASETS` | Dataset root containing `VEDAI/`, `M3FD/` | `$SHDETR_WORKSPACE/vedai_data/datasets` |
| `SHDETR_CFT` | CFT comparison checkout | `$SHDETR_WORKSPACE/CFT` |
| `SHDETR_LCAFNET` | LCAFNet comparison checkout | `$SHDETR_WORKSPACE/LCAFNet` |
| `SHDETR_MSOD` | Multispectral-object-detection checkout | `$SHDETR_WORKSPACE/multispectral-object-detection` |
| `SHDETR_PAPER` | Unpacked paper sources (figure target) | `$SHDETR_ROOT/SH_DETR` |
| `SHDETR_OUT_QUAL` | Qualitative-figure output root | `$SHDETR_ROOT/out_qual` |

Print the resolved values before running anything, and export the ones your
machine needs:

```bash
python3 shdetr_paths.py                 # show every resolved path
export SHDETR_ROOT=/data/shd/detr       # override a single location
export SHDETR_WORKSPACE=/mnt/experiments
```

Inside the repository the experiment trees are referenced as
`ROOT/datasets/` (DVTOD, RTDOD), `ROOT/compare/` (comparison methods),
`ROOT/outputs/` (SH-DETR / RT-DETR predictions) and `ROOT/result/`
(RTDOD predictions). Scripts are portable as long as these trees exist
somewhere and the variables point at them.

See [docs/PATHS.md](docs/PATHS.md) for the full list, including which
`predictions.json` file feeds which panel of each figure.

## Reproducing the evaluation tables

```bash
# COCO-style metrics from a predictions.json
python3 evaluation/eval_coco_preds.py --help

# complexity (params / GFLOPs / latency)
python3 evaluation/measure_complexity.py --help

# assemble the DVTOD comparison table
python3 evaluation/build_dvtod_table.py
```

Thresholds used by the qualitative renderers are a **confidence threshold of
0.50** (0.70 for the VEDAI panel, matching the released VEDAI comparison
predictions) and a **same-class IoU threshold of 0.50**.

## Notes on reproducibility

* The qualitative panels are selected illustrative scenes, not dataset-level
  recall measurements. The scripts import cached `predictions.json` /
  YOLO `labels/*.txt` outputs; re-running inference may change individual scores.
* Panels that draw a yellow box re-derive "recovered only by SH-DETR" by matching
  detections to ground truth at IoU ≥ 0.50 and checking whether any other method
  matches the same target.
* Class-index conventions differ between the RT-DETR family (1-based COCO ids)
  and the YOLO-family runs (0-based). Each renderer normalises them locally.

## Citation

```bibtex
@article{chen2026shdetr,
  title   = {Symmetric Fusion and Reliability-Aware Supervision
             for Long-Tailed {RGB--IR} Remote-Sensing Detection},
  author  = {Chen, Yi and Deng, Lingjun and Zhong, Chuen-Ho and Liu, Chang and Dong, Yanni},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2026}
}
```

## Acknowledgements

The numerical calculations in this work were carried out on the supercomputing
system of the Supercomputing Center of Wuhan University. This study was supported
by the National Natural Science Foundation of China under Grant U2541203.
