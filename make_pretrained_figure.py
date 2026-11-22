#!/usr/bin/env python3
"""Plot aggregate matched-fold comparisons without displaying clinical images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def make_figure(baseline: Path, pretrained: Path, output: Path):
    base_result = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    new_result = json.loads((pretrained / "summary.json").read_text(encoding="utf-8"))
    for key in ("repository", "commit", "training_images"):
        if base_result["dataset"][key] != new_result["dataset"][key]:
            raise ValueError(f"Dataset metadata differ: {key}")
    if base_result["protocol"]["outer_seed"] != new_result["protocol"]["outer_seed"]:
        raise ValueError("Outer-fold seeds differ")
    original, new = base_result["summary"], new_result["summary"]
    radii = [row["radius_px"] for row in original]
    if radii != [row["radius_px"] for row in new] or any(a["images"] != b["images"] for a, b in zip(original, new)):
        raise ValueError("The two result files must describe the same radii and image counts")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
    pipelines = [(original, "Handcrafted features", "#b65a22"), (new, "Frozen ResNet-18", "#246682")]
    for rows, name, color in pipelines:
        axes[0].plot(radii, [1 - row["unfiltered_error_rate"] for row in rows],
                     marker="o", lw=2, color=color, label=name)
        for policy, style, suffix in (("learned_confidence", "-", "confidence"),
                                      ("confidence_plus_focus", "--", "confidence + focus")):
            axes[1].plot(radii, [row["matched_coverage"][policy]["risk"] for row in rows],
                         marker="o", lw=1.8, color=color, linestyle=style, label=f"{name}: {suffix}")
    axes[0].set(title="Before deferral", ylabel="Classification accuracy")
    axes[1].set(title="Matched coverage (80.7%)", ylabel="Error among retained predictions")
    for ax in axes:
        ax.set(xlabel="Synthetic blur radius at 128 px", xticks=radii, ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Pretrained and handcrafted pipelines: exploratory comparison", fontsize=13, fontweight="bold")
    fig.text(.5, -.02, "Same 545 held-out images across five image-level folds; patient independence unknown.",
             ha="center", va="top", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "pretrained_comparison.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/blur_sensitivity"))
    parser.add_argument("--pretrained", type=Path, default=Path("results/pretrained_comparison"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    make_figure(args.baseline, args.pretrained, args.output)
    print(f"Figure written to {args.output / 'pretrained_comparison.png'}")


if __name__ == "__main__":
    main()
