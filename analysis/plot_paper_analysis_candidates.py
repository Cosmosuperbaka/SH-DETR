#!/usr/bin/env python3
"""Generate candidate paper-analysis plots from existing evaluation artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper_analysis_candidates"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

def load_per_class(path: Path):
    return {str(x["name"]): x for x in json.loads(path.read_text())}

def make_gain_plot():
    datasets = {
        "VEDAI": (ROOT / "outputs/vedai_perclass_all/RT-DETR_concat/per_class.json", ROOT / "outputs/vedai_s3407_requested_perclass/v19c_spsf/per_class.json"),
        "M3FD-LT20": (ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/baseline/per_class.json", ROOT / "outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/per_class.json"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, (name, (base_p, sh_p)) in zip(axes, datasets.items()):
        base, sh = load_per_class(base_p), load_per_class(sh_p)
        labels = [k for k in base if k in sh]
        gains = np.array([(sh[k]["ap"] - base[k]["ap"]) * 100 for k in labels])
        order = np.argsort(gains); labels = [labels[i] for i in order]; gains = gains[order]
        y = np.arange(len(labels)); colors = ["#d95f02" if x < 0 else "#1b9e77" for x in gains]
        ax.barh(y, gains, color=colors, alpha=.9); ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(y, labels); ax.set_xlabel("RSC-DETR − concat AP (percentage points)"); ax.set_title(name); ax.grid(axis="x", alpha=.25)
        for yi, value in zip(y, gains):
            ax.text(value + (0.15 if value >= 0 else -0.15), yi, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin - 0.45, xmax + 0.45)
    fig.savefig(OUT / "fig_class_ap_gain_vs_concat.png", dpi=300); fig.savefig(OUT / "fig_class_ap_gain_vs_concat.pdf"); plt.close(fig)

def read_curve(path: Path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try: item = json.loads(line)
        except Exception: continue
        val = item.get("val_coco_eval_bbox")
        if isinstance(val, list) and len(val) >= 3 and item.get("epoch") is not None:
            rows.append((int(item["epoch"])+1, float(val[0])*100, float(val[2])*100))
    return rows

def make_curve_plot():
    datasets = {
        "VEDAI": (ROOT / "result/VEDAI/test-x1x3-v19c_seed3407/log.txt", ROOT / "result/VEDAI/test-x1x3-v19c-spsf_seed3407/log.txt"),
        "M3FD-LT20": (ROOT / "ablations/rtdetr-concat-native-res-b8-45e_M3FD-LT20_b8-45e/log.txt", ROOT / "ablations/rtdetr-x1x3-v19c-spsf-native-res-b8-45e_M3FD-LT20_b8-45e/log.txt"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True); plotted = []
    for ax, (name, (base_p, sh_p)) in zip(axes, datasets.items()):
        for path, label, color in ((base_p, "RT-DETR concat", "#7570b3"), (sh_p, "RSC-DETR", "#1b9e77")):
            if not path.exists(): continue
            rows = read_curve(path)
            if not rows: continue
            ep, ap, ap75 = zip(*rows); ax.plot(ep, ap, marker="o", ms=2.5, label=f"{label} AP", color=color); ax.plot(ep, ap75, ls="--", marker=".", ms=2, label=f"{label} AP75", color=color, alpha=.65); plotted.append((name, label, len(rows)))
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation metric (%)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(OUT / "fig_training_curve_ap_ap75_candidate.png", dpi=300); fig.savefig(OUT / "fig_training_curve_ap_ap75_candidate.pdf"); plt.close(fig)
    (OUT / "README.md").write_text("# Paper analysis candidates\n\n- `fig_class_ap_gain_vs_concat`: existing per-class AP, RSC-DETR minus concat.\n- `fig_training_curve_ap_ap75_candidate`: descriptive validation curves only; a matched no-dynamic-coordination ablation is not available.\n\nPlotted curves: " + repr(plotted) + "\n", encoding="utf-8")

if __name__ == "__main__":
    make_gain_plot(); make_curve_plot(); print(OUT)
