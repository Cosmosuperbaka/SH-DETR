#!/usr/bin/env python3
"""Generate the combined VEDAI + M3FD-LT20 class-distribution figure.

Replaces the two separate single-column distribution figures (Fig.6/Fig.7) with
one double-column, two-panel figure. Uses Nimbus Roman (Times-metric compatible)
so the type matches the paper body set in newtxtext.

Head/tail partition follows the paper's cumulative-instance threshold tau=0.6:
  VEDAI   head = {car, pickup} (cum 62.9%),  tail = other 6 classes
  M3FD    head = {Car, People} (cum 80.1%), tail = other 4 classes
Verified against Table II: SH-DETR VEDAI AP-h = mean(car 62.99, pickup 62.75)
= 62.87, AP-t = mean of the six tail classes = 58.12.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- register Nimbus Roman so chart text matches the Times-style body ---
_FONT_DIR = "/usr/share/fonts/opentype/urw-base35"
for _name in ("NimbusRoman-Regular", "NimbusRoman-Bold", "NimbusRoman-Italic"):
    _p = Path(_FONT_DIR) / f"{_name}.otf"
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))
matplotlib.rcParams["font.family"] = "Nimbus Roman"
matplotlib.rcParams["axes.unicode_minus"] = False

# --- data (instances per class, descending frequency) ---
VEDAI = [
    ("car", 1215), ("pickup", 846), ("camping car", 351), ("truck", 270),
    ("other", 180), ("tractor", 171), ("boat", 153), ("van", 90),
]
M3FD = [
    ("Car", 8379), ("People", 4767), ("Lamp", 1657), ("Truck", 697),
    ("Bus", 499), ("Motorcycle", 407),
]

# cumulative-instance proportion beyond which classes become "tail"
TAU = 0.6


def partition(data: list[tuple[str, int]]) -> list[tuple[str, int, str]]:
    total = sum(n for _, n in data)
    prev_cum = 0
    out = []
    for name, n in data:
        head = prev_cum <= TAU * total  # head until the accumulated ratio first passes TAU
        prev_cum += n
        out.append((name, n, "head" if head else "tail"))
    return out


HEAD_COLOR = "#1f5b8c"   # deep blue
TAIL_COLOR = "#d1652e"   # warm orange
GRID_COLOR = "#b9b9b9"


def draw_panel(ax, data, title):
    rows = partition(data)
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    colors = [HEAD_COLOR if r[2] == "head" else TAIL_COLOR for r in rows]

    y = list(range(len(rows)))[::-1]
    bars = ax.barh(y, counts, height=0.62, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of training instances")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_xlim(0, max(counts) * 1.06)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, n in zip(bars, counts):
        ax.text(
            n + max(counts) * 0.015, b.get_y() + b.get_height() / 2,
            f"{n:,}", va="center", ha="left", fontsize=8.5, color="#222222",
        )


def main() -> None:
    out_pdf = Path("figures/distribution_vedai_m3fd.pdf")
    out_png = Path("figures/distribution_vedai_m3fd.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.4), gridspec_kw={"wspace": 0.42}
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.33)
    draw_panel(ax1, VEDAI, "(a) VEDAI")
    draw_panel(ax2, M3FD, "(b) M3FD-LT20")
    handles = [
        Patch(facecolor=HEAD_COLOR, label="head"),
        Patch(facecolor=TAIL_COLOR, label="tail"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote={out_pdf.resolve()}")
    print(f"wrote={out_png.resolve()}")
    print("VEDAI head:", [r[0] for r in partition(VEDAI) if r[2] == "head"])
    print("M3FD  head:", [r[0] for r in partition(M3FD) if r[2] == "head"])


if __name__ == "__main__":
    main()
