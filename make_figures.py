#!/usr/bin/env python3
"""Recreate repository figures from aggregate CSV and JSON outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

COLORS = {"confidence": "#1f6686", "focus": "#bd641e", "joint": "#6650a4"}
NAMES = {
    "temperature_scaled_confidence": ("Temperature-scaled confidence", COLORS["confidence"], "-"),
    "laplacian_variance": ("Laplacian variance", COLORS["focus"], "--"),
    "confidence_plus_laplacian": ("Confidence + Laplacian variance", COLORS["joint"], "-."),
}


def render_original_curve(original: Path, output: Path):
    with (original/"risk_coverage.csv").open(encoding="utf-8",newline="") as file:
        rows = list(csv.DictReader(file))
    result=json.loads((original/"results.json").read_text(encoding="utf-8"))
    coverage=result["primary"]["temperature_scaled_confidence"]["actual_coverage"]
    fig,ax=plt.subplots(figsize=(9,4.9),constrained_layout=True)
    for policy,(label,color,style) in NAMES.items():
        records=[r for r in rows if r["selector"]==policy and float(r["coverage"])>=.5]
        if not records:
            raise ValueError(f"No curve data for {policy}")
        ax.plot([float(r["coverage"]) for r in records], [float(r["risk"]) for r in records],
                linestyle=style,color=color,lw=2,label=label)
        ax.scatter([coverage],[result["primary"][policy]["risk"]],color=color,s=30,zorder=5)
    ax.axvline(coverage,color="#777777",ls=":",lw=1.2)
    ax.set(xlim=(.5,1),ylim=(0,.65),xlabel="Coverage: fraction of images retained",
           ylabel="Error rate among retained predictions",
           title="Original pilot: selective risk on 184 test images")
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis="y",alpha=.2)
    ax.legend(loc="lower right",fontsize=9,frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"original_risk_coverage.png",dpi=180,facecolor="white")
    plt.close(fig)


def render_classwise(original: Path, output: Path):
    result=json.loads((original/"results.json").read_text(encoding="utf-8"))
    confidence=result["primary"]["temperature_scaled_confidence"]["classes"]
    joint=result["primary"]["confidence_plus_laplacian"]["classes"]
    classes=["Diabetic","Pressure","Surgical","Venous"]
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    x=np.arange(len(classes));width=.35
    a=ax.bar(x-width/2,[confidence[c]["risk"] for c in classes],width,
             label="Confidence",color=COLORS["confidence"])
    b=ax.bar(x+width/2,[joint[c]["risk"] for c in classes],width,
             label="Confidence + focus",color=COLORS["joint"])
    for bars,record in [(a,confidence),(b,joint)]:
        for bar,cls in zip(bars,classes):
            ax.annotate(f"n={record[cls]['accepted']}",(bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
    ax.set(xticks=x,xticklabels=classes,ylim=(0,1),
           ylabel="Error rate among retained images",
           title="Original pilot: retained error by true wound class")
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.legend(frameon=False,fontsize=9,loc="upper right")
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"original_classwise_risk.png",dpi=180,facecolor="white")
    plt.close(fig)


def render_fold_differences(train_cv: Path, output: Path):
    result=json.loads((train_cv/"summary.json").read_text(encoding="utf-8"))
    folds=result["folds"]
    difference=[100*(f["policies"]["confidence_plus_focus"]["risk"] -
                     f["policies"]["learned_confidence"]["risk"]) for f in folds]
    colors=[COLORS["focus"] if d>0 else COLORS["confidence"] for d in difference]
    fig,ax=plt.subplots(figsize=(7.8,4.5),constrained_layout=True)
    x=np.arange(len(folds))
    bars=ax.bar(x,difference,color=colors,width=.6)
    ax.axhline(0,color="#555555",lw=1)
    for bar,val in zip(bars,difference):
        ax.annotate(f"{val:+.2f}",(bar.get_x()+bar.get_width()/2,val),
                    xytext=(0,5 if val>=0 else -12),textcoords="offset points",
                    ha="center",fontsize=9)
    ax.set(xticks=x,xticklabels=[f"Fold {f['fold']}" for f in folds],
           ylim=(-4.5,4.5),ylabel="Joint minus learned confidence error (percentage points)",
           title="Training-pool cross-validation: difference at matched coverage")
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(output/"train_cv_fold_differences.png",dpi=180,facecolor="white")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,default=Path("results/original"))
    p.add_argument("--train-cv",type=Path,default=Path("results/train_cv"))
    p.add_argument("--output",type=Path,default=Path("figures"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    render_original_curve(args.original,args.output)
    render_classwise(args.original,args.output)
    render_fold_differences(args.train_cv,args.output)
    print(f"Figures written to {args.output}")


if __name__=="__main__":
    main()
