#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the DVTOD comparison table from per-class + complexity JSONs.

Inputs are collected in ``--results-dir`` with the naming convention
``<key>_per_class.json`` and ``<key>_complexity.json``.  ``rows.json`` (optional)
declares the display order and the venue/year metadata of every method.

Usage:
    python build_dvtod_table.py --results-dir results --meta meta.json \
        --out-md table.md --out-tex table.tex
"""

import argparse
import json
from pathlib import Path


def fmt(v, nd=2, scale=100.0):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "-"
    return f"{f * scale:.{nd}f}"


def load(results_dir: Path, key: str):
    pc = results_dir / f"{key}_per_class.json"
    cx = results_dir / f"{key}_complexity.json"
    out = {"per_class": None, "complexity": None}
    if pc.exists():
        out["per_class"] = json.loads(pc.read_text())
    if cx.exists():
        out["complexity"] = json.loads(cx.read_text())
    return out


def per_class_lookup(pc):
    m = {}
    for r in pc.get("per_class", []):
        m[str(r["name"]).lower()] = r
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-tex", default="")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    meta = json.loads(Path(args.meta).read_text())

    header = ["Method", "Venue", "Year", "AP", "AP50", "AP75",
              "Person", "Car", "Bicycle", "AP-s", "AP-m", "AP-l",
              "Params(M)", "GFLOPs", "Latency(ms)"]
    rows = []
    for item in meta["methods"]:
        key = item["key"]
        data = load(rd, key)
        pc, cx = data["per_class"], data["complexity"]
        row = {
            "Method": item.get("label", key),
            "Venue": item.get("venue", "-"),
            "Year": item.get("year", "-"),
            "bold": bool(item.get("bold", False)),
        }
        if pc:
            ov = pc["overall"]
            cl = per_class_lookup(pc)
            row.update({
                "AP": fmt(ov.get("AP")), "AP50": fmt(ov.get("AP50")),
                "AP75": fmt(ov.get("AP75")),
                "Person": fmt(cl.get("person", {}).get("ap")),
                "Car": fmt(cl.get("car", {}).get("ap")),
                "Bicycle": fmt(cl.get("bicycle", {}).get("ap")),
                "AP-s": fmt(ov.get("APs")), "AP-m": fmt(ov.get("APm")),
                "AP-l": fmt(ov.get("APl")),
            })
        else:
            row.update({k: "-" for k in
                        ("AP", "AP50", "AP75", "Person", "Car", "Bicycle",
                         "AP-s", "AP-m", "AP-l")})
        if cx:
            row.update({
                "Params(M)": f"{cx['params_M']:.2f}",
                "GFLOPs": f"{cx['gflops']:.1f}",
                "Latency(ms)": f"{cx['latency_ms']:.1f}",
            })
        else:
            row.update({"Params(M)": "-", "GFLOPs": "-", "Latency(ms)": "-"})
        rows.append(row)

    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = []
        for h in header:
            v = str(r.get(h, "-"))
            cells.append(f"**{v}**" if r.get("bold") and h not in ("Method", "Venue", "Year") else v)
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"

    tex_cols = "llc" + "c" * (len(header) - 3)
    caption = meta.get("caption") or (
        "Comparison on DVTOD (3 classes: Person/Car/Bicycle). All methods use the "
        "same data split, 1920 input and seed 42, with the validation split used "
        "as test."
    )
    tex = [r"\begin{table*}[!t]", r"\centering",
           r"\caption{" + caption + "}",
           r"\label{tab:dvtod_compare}", r"\scriptsize",
           r"\begin{tabular}{" + tex_cols + "}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for h in header:
            v = str(r.get(h, "-"))
            if r.get("bold") and h not in ("Method", "Venue", "Year"):
                v = r"\textbf{" + v + "}"
            cells.append(v)
        tex.append(" & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex = "\n".join(tex) + "\n"

    if args.out_md:
        Path(args.out_md).write_text(md)
    if args.out_tex:
        Path(args.out_tex).write_text(tex)
    print(md)


if __name__ == "__main__":
    main()
