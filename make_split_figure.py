#!/usr/bin/env python3
"""Plot descriptive paired differences across dependent image-level splits."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(results: Path, output: Path):
    data = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    seeds = data["protocol"]["outer_seeds"]
    rows = {(r["seed"], r["model"]): r for r in data["rows"]}
    if len(seeds) != 5 or any((s, m) not in rows for s in seeds for m in ("handcrafted", "frozen_resnet18")):
        raise ValueError("Expected both pipelines for five recorded split seeds")
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
    for model, label, color, offset, marker in (("handcrafted", "Handcrafted", "#b65a22", -.1, "o"),
                                                 ("frozen_resnet18", "Frozen ResNet features", "#246682", .1, "s")):
        values = [100 * rows[(seed, model)]["joint_minus_confidence"] for seed in seeds]
        ax.scatter(x + offset, values, marker=marker, s=82, color=color, label=label, zorder=3)
    ax.set_xticks(x, labels=[str(seed) for seed in seeds])
    ax.set(xlabel="Outer split seed", ylabel="Joint minus confidence-only error (percentage points)")
    ax.set_title("Sensitivity of the matched-coverage difference to image splits", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.text(.5, -.035, "Negative: fewer joint-rule errors. Each run retains 440/545 images.\n"
             "The same images recur across seeds; points are dependent and are not confidence intervals.",
             ha="center", va="top", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "split_sensitivity.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/split_sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    make_figure(args.results, args.output)
    print(f"Figure written to {args.output / 'split_sensitivity.png'}")


if __name__ == "__main__":
    main()
