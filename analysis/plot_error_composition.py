#!/usr/bin/env python3
"""Plot matched detection error composition from saved COCO predictions."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_analysis_candidates"
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    "VEDAI": (ROOT/"datasets/VEDAI/annotations/vedai_fold01_test_class8.json",
              ROOT/"outputs/vedai_direct_test_unified/baseline/seed_3407/predictions.json",
              ROOT/"outputs/vedai_direct_test_unified/rtdetr-spsf-alpha050/seed_3407/predictions.json"),
    "M3FD-LT20": (ROOT/"datasets/M3FD lt20/annotations/instances_test.json",
                  ROOT/"outputs/m3fd_lt20_s42_b8_valbest_test_requested/baseline/predictions.json",
                  ROOT/"outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/predictions.json"),
}

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def load_gt(path):
    d=json.loads(path.read_text()); by={}
    for a in d["annotations"]:
        iid=a["image_id"]
        if isinstance(iid, str):
            # VEDAI COCO IDs may be strings such as 00000010_co while
            # detector predictions use the numeric stem 10.
            digits="".join(ch for ch in iid.split("_")[0] if ch.isdigit())
            iid=int(digits) if digits else iid
        by.setdefault(iid, []).append(a)
    return by

def load_pred(path):
    by={}
    for p in json.loads(path.read_text()):
        by.setdefault(p["image_id"], []).append(p)
    for v in by.values(): v.sort(key=lambda x: x.get("score",0), reverse=True)
    return by

def classify(gt_by, pred_by):
    counts={k:0 for k in ("Correct", "Localization error", "Classification error", "Missed")}
    fp=0; total=0
    for iid, gts in gt_by.items():
        preds=pred_by.get(iid, [])
        used=set()
        for gt in gts:
            total += 1
            same=[]; other=[]; all_i=[]
            for j,p in enumerate(preds):
                if j in used: continue
                v=iou(gt["bbox"],p["bbox"]); all_i.append((v,j,p))
                (same if p["category_id"]==gt["category_id"] else other).append((v,j,p))
            best_same=max(same, default=(0,-1,None), key=lambda x:x[0])
            best_other=max(other, default=(0,-1,None), key=lambda x:x[0])
            if best_same[0] >= .5:
                counts["Correct"] += 1; used.add(best_same[1])
            elif best_other[0] >= .5:
                counts["Classification error"] += 1; used.add(best_other[1])
            elif best_same[0] >= .1:
                counts["Localization error"] += 1; used.add(best_same[1])
            else:
                counts["Missed"] += 1
        fp += sum(1 for j,p in enumerate(preds) if j not in used and p.get("score",0)>=.25)
    counts["False positive"] = fp
    return counts, total

def main():
    all_rows=[]
    for dataset,(gtp,bp,sp) in CASES.items():
        gt=load_gt(gtp)
        for name,p in (("RT-DETR concat",bp),("SH-DETR",sp)):
            c,total=classify(gt,load_pred(p)); c["Correct"] = c["Correct"]
            all_rows.append((dataset,name,c,total))
    cats=["Correct","Localization error","Classification error","Missed","False positive"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors=["#1b9e77","#e6ab02","#d95f02","#7570b3","#666666"]
    for ax,dataset in zip(axes,CASES):
        rows=[r for r in all_rows if r[0]==dataset]; x=np.arange(2); bottom=np.zeros(2,float)
        for cat,color in zip(cats,colors):
            vals=np.array([r[2][cat] for r in rows]); ax.bar(x,vals,bottom=bottom,label=cat,color=color); bottom+=vals
        ax.set_xticks(x,[r[1] for r in rows]); ax.set_ylabel("Number of instances / predictions"); ax.set_title(dataset); ax.grid(axis="y",alpha=.25)
    axes[1].legend(fontsize=8,loc="upper right")
    fig.savefig(OUT/"fig_detection_error_composition.png",dpi=300); fig.savefig(OUT/"fig_detection_error_composition.pdf"); plt.close(fig)
    # Publication-oriented version: GT outcomes and false positives use separate denominators.
    gt_cats=["Correct","Localization error","Classification error","Missed"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for col,dataset in enumerate(CASES):
        rows=[r for r in all_rows if r[0]==dataset]; x=np.arange(2); bottom=np.zeros(2,float)
        for cat,color in zip(gt_cats,colors[:4]):
            vals=np.array([r[2][cat]/r[3]*100 for r in rows]); axes[0,col].bar(x,vals,bottom=bottom,label=cat,color=color); bottom+=vals
        axes[0,col].set_title(dataset); axes[0,col].set_xticks(x,[r[1] for r in rows]); axes[0,col].set_ylabel("GT outcome (%)"); axes[0,col].set_ylim(0,100); axes[0,col].grid(axis="y",alpha=.25)
        vals=np.array([r[2]["False positive"] for r in rows]) / np.array([len(load_pred(CASES[dataset][1 if r[1].startswith('RT') else 2])) for r in rows])
        axes[1,col].bar(x,vals,color=["#7570b3","#1b9e77"]); axes[1,col].set_xticks(x,[r[1] for r in rows]); axes[1,col].set_ylabel("False positives / image"); axes[1,col].grid(axis="y",alpha=.25)
    axes[0,1].legend(fontsize=8,loc="upper right")
    fig.savefig(OUT/"fig_detection_error_composition_publication.png",dpi=300); fig.savefig(OUT/"fig_detection_error_composition_publication.pdf"); plt.close(fig)
    with (OUT/"error_composition_counts.json").open("w") as f: json.dump([{"dataset":d,"method":m,"counts":c,"gt_instances":t} for d,m,c,t in all_rows],f,indent=2)
    print(json.dumps(all_rows,indent=2))

if __name__=="__main__": main()
