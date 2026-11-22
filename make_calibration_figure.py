#!/usr/bin/env python3
"""Draw held-out top-label reliability diagrams from aggregate audit bins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_figure(results: Path, output: Path):
    data = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    rows = {row["radius_px"]: row for row in data["summary"] if row["model"] == "frozen_resnet18"}
    if 0.0 not in rows or 3.0 not in rows:
        raise ValueError("Expected frozen ResNet results for both clean and radius 3.0")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, radius, title in zip(axes, (0.0, 3.0), ("Clean images", "Strong synthetic blur")):
        ax.plot([.2, 1], [.2, 1], color="#8a8a8a", linestyle="--", lw=1.3)
        for state, label, color, marker in (("raw", "Before scaling", "#345e80", "o"),
                                            ("temperature_scaled", "After scaling", "#c46530", "s")):
            m = rows[radius][state]
            populated = [b for b in m["bins"] if b["count"]]
            x = [b["mean_confidence"] for b in populated]
            y = [b["accuracy"] for b in populated]
            sizes = [26 + 160 * b["count"] / m["n"] for b in populated]
            ax.plot(x, y, color=color, alpha=.7, linewidth=1.3)
            ax.scatter(x, y, s=sizes, color=color, marker=marker, edgecolor="white", linewidth=.5,
                       label=f"{label} (ECE {m['ece_top_label_10_equal_width_bins']:.3f})", zorder=3)
        ax.set(title=title, xlabel="Mean predicted confidence", xlim=(.2, 1.02), ylim=(-.04, 1.04))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, framealpha=.88, edgecolor="none", fontsize=8, loc="lower right")
    axes[0].set_ylabel("Observed accuracy within bin")
    fig.suptitle("ResNet-feature confidence: held-out calibration audit", fontsize=13, fontweight="bold")
    fig.text(.5, -.025, "Pooled image-level folds (n=545); circle/square size reflects bin count. Dashed line: ideal agreement.\n"
             "Synthetic blur; bins may be sparse; patient independence is unknown.",
             ha="center", va="top", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "calibration_audit.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/calibration_audit"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    make_figure(args.results, args.output)
    print(f"Figure written to {args.output / 'calibration_audit.png'}")


if __name__ == "__main__":
    main()
