#!/usr/bin/env python3
"""Exploratory repeated image-level splits for the two AZH pipelines.

These are dependent analyses of the same images, not patient-level replication.
No clinical images, embeddings, or individual predictions are saved.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
from PIL import __version__ as pillow_version
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from run_azh_pilot import COMMIT, REPOSITORY, load_data
from run_pretrained_comparison import WEIGHT_SHA256, WEIGHT_URL, ensure_weights, extract_embeddings
from run_train_cv import COVERAGE, INNER_SEED, OUTER_SEED, aggregate, evaluate_fold
from study_utils import outer_folds

# Fixed for this retrospective follow-up, after earlier results were visible.
SEEDS = (OUTER_SEED, 20261110, 20261111, 20261112, 20261113)


def describe_differences(rows):
    """Describe paired risk differences; do not estimate inferential uncertainty."""
    if not rows:
        raise ValueError("at least one split result is required")
    differences = np.asarray([row["joint_minus_confidence"] for row in rows], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("risk differences must be finite")
    return {"mean_difference": float(differences.mean()), "minimum_difference": float(differences.min()),
            "maximum_difference": float(differences.max()),
            "joint_fewer_errors": int((differences < 0).sum()),
            "equal_errors": int((differences == 0).sum()),
            "joint_more_errors": int((differences > 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ensure_weights(args.weights, args.download_weights)
    samples, audit = load_data(args.data_dir, args.download_data)
    train = [s for s in samples if s.split == "Train"]
    if len(train) != 545:
        raise ValueError(f"Expected 545 deduplicated training wound images, got {len(train)}")
    y = np.asarray([s.label for s in train])
    focus = np.asarray([s.focus for s in train])
    baseline = np.stack([s.features for s in train])
    all_embeddings, torch_version, vision_version = extract_embeddings(train, args.data_dir, args.weights, args.batch_size)
    features = {"handcrafted": baseline, "frozen_resnet18": all_embeddings[0.0]}
    rows = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            # Create the same outer splits once for both model pipelines.
            splits = list(outer_folds(y, seed=seed))
            if sorted(np.concatenate([test for _, test in splits]).tolist()) != list(range(len(train))):
                raise AssertionError("Every image must be held out once per split seed")
            for name, x in features.items():
                folds = [evaluate_fold(x, focus, y, tr, te, number)
                         for number, (tr, te) in enumerate(splits, 1)]
                pooled = aggregate(folds)
                confidence = pooled["learned_confidence"]
                joint = pooled["confidence_plus_focus"]
                if confidence["accepted"] != joint["accepted"] or confidence["accepted"] != 440:
                    raise AssertionError("Policies must retain 440 of 545 images per seed")
                rows.append({
                    "seed": seed, "model": name, "images": len(train),
                    "unfiltered_errors": sum(fold["unfiltered_errors"] for fold in folds),
                    "accepted": confidence["accepted"],
                    "confidence_errors": confidence["errors"], "confidence_risk": confidence["risk"],
                    "joint_errors": joint["errors"], "joint_risk": joint["risk"],
                    "joint_minus_confidence": joint["risk"] - confidence["risk"],
                    "fold_counts": [{"fold": f["fold"], "held_out": f["sizes"]["outer_test"],
                                     "confidence_errors": f["policies"]["learned_confidence"]["errors"],
                                     "joint_errors": f["policies"]["confidence_plus_focus"]["errors"]}
                                    for f in folds],
                })

    root = Path(__file__).resolve().parent
    original = json.loads((root / "results/train_cv/summary.json").read_text(encoding="utf-8"))["summary"]
    pretrained = json.loads((root / "results/pretrained_comparison/summary.json").read_text(encoding="utf-8"))["summary"][0]["matched_coverage"]
    for name, expected in (("handcrafted", original), ("frozen_resnet18", pretrained)):
        row = next(r for r in rows if r["seed"] == OUTER_SEED and r["model"] == name)
        if (row["confidence_errors"] != expected["learned_confidence"]["errors"] or
                row["joint_errors"] != expected["confidence_plus_focus"]["errors"] or
                row["accepted"] != expected["learned_confidence"]["accepted"]):
            raise AssertionError(f"Previously recorded clean result differs for {name}")

    result = {
        "analysis": "Exploratory sensitivity to five image-level outer split seeds",
        "retrospective_status": "Designed after earlier results were visible; not preregistered.",
        "dataset": {"repository": REPOSITORY, "commit": COMMIT, "training_images": len(train), "audit": audit},
        "protocol": {
            "outer_seeds": list(SEEDS), "outer_folds": 5, "inner_seed": INNER_SEED,
            "requested_coverage": COVERAGE, "retained_per_seed": 440,
            "comparison": "Like-for-like learned confidence versus learned confidence plus focus; ranking within each held-out fold.",
            "test_archive": "Read for exact deduplication only; no original-test predictions are scored.",
            "patient_or_source_grouping": False,
            "dependence": "The same images and overlapping training pools recur across seeds; these are not independent replications or confidence intervals.",
            "weights_url": WEIGHT_URL, "weights_sha256": WEIGHT_SHA256,
        },
        "rows": rows,
        "descriptive_ranges": {name: describe_differences([row for row in rows if row["model"] == name])
                               for name in features},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__, "pillow": pillow_version,
                        "torch": torch_version, "torchvision": vision_version},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        fields = ["seed", "model", "images", "unfiltered_errors", "accepted", "confidence_errors", "confidence_risk",
                  "joint_errors", "joint_risk", "joint_minus_confidence"]
        writer.writerow(fields)
        writer.writerows([[row[field] for field in fields] for row in rows])
    print((args.output / "metrics.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
