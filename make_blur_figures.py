#!/usr/bin/env python3
"""Plot aggregate synthetic-blur results without displaying patient images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def make_figure(results: Path, output: Path):
    data = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    rows = data["summary"]
    radii = [item["radius_px"] for item in rows]
    colors = {"learned_confidence": "#1f6686", "confidence_plus_focus": "#6650a4"}
    labels = {"learned_confidence": "Learned confidence", "confidence_plus_focus": "Confidence + focus"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    fig.suptitle("Synthetic blur sensitivity: training-pool exploration", fontsize=14, fontweight="bold")

    axes[0, 0].plot(radii, [row["median_laplacian_variance"] for row in rows],
                    marker="o", color="#a65d20", lw=2)
    axes[0, 0].set(title="Focus proxy", ylabel="Median Laplacian variance")
    for policy in colors:
        axes[0, 1].plot(radii, [row["matched_coverage"][policy]["risk"] for row in rows],
                        marker="o", color=colors[policy], label=labels[policy], lw=2)
        axes[1, 0].plot(radii, [row["fixed_threshold"][policy]["coverage"] for row in rows],
                        marker="o", color=colors[policy], label=labels[policy], lw=2)
        axes[1, 1].plot(radii, [row["fixed_threshold"][policy]["risk"] for row in rows],
                        marker="o", color=colors[policy], label=labels[policy], lw=2)
    axes[0, 1].set(title="Matched coverage", ylabel="Retained classification error")
    axes[1, 0].set(title="Frozen thresholds", ylabel="Coverage retained")
    axes[1, 1].set(title="Frozen thresholds", ylabel="Retained classification error")
    for ax in axes.flat:
        ax.set_xlabel("Gaussian blur radius after resizing (pixels)")
        ax.set_xticks(radii)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(0, 1)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(.5, -.01,
             "Exploratory image-level CV; 545 images; patient independence unknown. "
             "Matched coverage is about 80.7%.",
             ha="center", va="top", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "blur_sensitivity.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/blur_sensitivity"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    make_figure(args.results, args.output)
    print(f"Figure written to {args.output / 'blur_sensitivity.png'}")


if __name__ == "__main__":
    main()
