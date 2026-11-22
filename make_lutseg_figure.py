#!/usr/bin/env python3
"""Plot aggregate results of the separate LUTSeg patient-grouped pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(results: Path, output: Path):
    data = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    models = data["models"]
    names = ("handcrafted", "handcrafted_balanced", "frozen_resnet18", "frozen_resnet18_balanced")
    labels = ("Handcrafted", "Handcrafted\nbalanced", "ResNet", "ResNet\nbalanced")
    if data["dataset"]["images"] != 141 or any(name not in models for name in names):
        raise ValueError("Expected all four variants of the pinned LUTSeg experiment")
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)
    for policy, title, shift, color in (("learned_confidence", "Learned confidence", -.18, "#246682"),
                                        ("confidence_plus_focus", "Confidence + focus", .18, "#be642b")):
        metrics = [models[name]["selection"]["matched_coverage"][policy] for name in names]
        if any(m["accepted"] != 115 for m in metrics):
            raise ValueError("Expected matched retention of 115 images")
        bars = axes[0].bar(x + shift, [m["errors"] for m in metrics], width=.34, color=color, label=title)
        axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set(title="Errors among 115 retained images", ylabel="Incorrect predictions", ylim=(0, 125))
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    correct_positive = []
    for name in names:
        m = models[name]["unfiltered"]["classes"]["present"]
        if m["images"] != 32:
            raise ValueError("Expected 32 positive images")
        correct_positive.append(m["images"] - m["errors"])
    bars = axes[1].bar(x, correct_positive, color=["#a9b7c1", "#6e90a7", "#a9b7c1", "#6e90a7"], width=.58)
    axes[1].bar_label(bars, labels=[f"{n}/32" for n in correct_positive], padding=3, fontsize=9)
    axes[1].set(title="Detected positive images before deferral", ylabel="Correct positive predictions", ylim=(0, 35))
    for ax in axes:
        ax.set_xticks(x, labels=labels)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LUTSeg: exploratory patient-disjoint tissue-presence pilot", fontsize=13, fontweight="bold")
    fig.text(.5, -.025, "141 images, 39 patient groups; target is any necrotic-tissue pixel in source masks.\n"
             "Different from AZH wound-type classification. Balanced variants were added after initial results.",
             ha="center", va="top", fontsize=8, color="#555555")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "lutseg_patient_pilot.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/lutseg_patient_pilot"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    make_figure(args.results, args.output)
    print(f"Figure written to {args.output / 'lutseg_patient_pilot.png'}")


if __name__ == "__main__":
    main()
