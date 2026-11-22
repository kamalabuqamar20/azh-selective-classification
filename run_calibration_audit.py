#!/usr/bin/env python3
"""Exploratory held-out calibration audit for the two AZH image pipelines.

No patient images, individual predictions or embeddings are written to disk.
The splits are image-level; this is not patient-independent evaluation.
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
from scipy.special import softmax
import sklearn
from threadpoolctl import threadpool_limits

from run_azh_pilot import CLASSES, COMMIT, REPOSITORY, load_data
from run_blur_sensitivity import RADII, build_variants
from run_pretrained_comparison import WEIGHT_SHA256, WEIGHT_URL, ensure_weights, extract_embeddings
from run_train_cv import OUTER_SEED, fit_fold
from study_utils import outer_folds


def probability_metrics(probabilities, labels, classes, n_bins=10):
    """Return pooled top-label ECE and multiclass NLL/Brier without records."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.asarray(classes)
    if (p.ndim != 2 or y.ndim != 1 or classes.ndim != 1 or not len(y) or
            p.shape != (len(y), len(classes)) or len(classes) < 2 or
            len(np.unique(classes)) != len(classes)):
        raise ValueError("probabilities, labels and classes have incompatible shapes")
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or
            not np.allclose(p.sum(axis=1), 1.0, atol=1e-7, rtol=0)):
        raise ValueError("each probability vector must be finite and sum to one")
    if not set(y.tolist()).issubset(set(classes.tolist())):
        raise ValueError("labels must be included in classes")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    one_hot = (y[:, None] == classes[None, :]).astype(np.float64)
    true_probability = (p * one_hot).sum(axis=1)
    confidence = p.max(axis=1)
    correct = (classes[p.argmax(axis=1)] == y)
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for index in range(n_bins):
        mask = bin_index == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_confidence - accuracy)
        bins.append({"lower": index / n_bins, "upper": (index + 1) / n_bins,
                     "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return {
        "n": len(y), "errors": int((~correct).sum()), "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "nll": float(-np.log(np.clip(true_probability, 1e-15, 1.0)).mean()),
        "brier_multiclass": float(np.square(p - one_hot).sum(axis=1).mean()),
        "ece_top_label_10_equal_width_bins": float(ece), "bins": bins,
    }


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
    train = [sample for sample in samples if sample.split == "Train"]
    if len(train) != 545:
        raise ValueError(f"Expected 545 deduplicated training wound images, got {len(train)}")
    y = np.asarray([sample.label for sample in train])
    classes = np.asarray(sorted(CLASSES))
    baseline, focus = build_variants(train, args.data_dir)
    pretrained, torch_version, vision_version = extract_embeddings(train, args.data_dir, args.weights, args.batch_size)
    pipelines = {"handcrafted": baseline, "frozen_resnet18": pretrained}
    pooled = {name: {radius: {state: np.empty((len(y), len(classes)), dtype=np.float64)
                            for state in ("raw", "temperature_scaled")}
                     for radius in RADII} for name in pipelines}
    seen, fold_details = [], []
    with threadpool_limits(limits=1):
        for number, (outer_train, outer_test) in enumerate(outer_folds(y, seed=OUTER_SEED), 1):
            fold = {"fold": number, "held_out_images": len(outer_test), "models": {}}
            for name, features in pipelines.items():
                fitted = fit_fold(features[0.0], focus[0.0], y, outer_train, number)
                if not np.array_equal(fitted.model.classes_, classes):
                    raise AssertionError("Class order changed across folds")
                if any(set(idx.tolist()) & set(outer_test.tolist()) for idx in
                       (fitted.model_idx, fitted.calibration_idx, fitted.selector_idx)):
                    raise AssertionError("Held-out images entered a fitted partition")
                fold["models"][name] = {"temperature": fitted.temperature,
                    "model_fit_images": len(fitted.model_idx), "calibration_images": len(fitted.calibration_idx),
                    "selector_fit_images": len(fitted.selector_idx)}
                for radius in RADII:
                    logits = fitted.model.decision_function(features[radius][outer_test])
                    raw = softmax(logits, axis=1)
                    scaled = softmax(logits / fitted.temperature, axis=1)
                    if not np.array_equal(raw.argmax(axis=1), scaled.argmax(axis=1)):
                        raise AssertionError("Positive temperature changed predicted classes")
                    pooled[name][radius]["raw"][outer_test] = raw
                    pooled[name][radius]["temperature_scaled"][outer_test] = scaled
            seen.extend(outer_test.tolist())
            fold_details.append(fold)
    if sorted(seen) != list(range(len(y))):
        raise AssertionError("Every image must be held out exactly once")

    root = Path(__file__).resolve().parent
    references = {
        "handcrafted": json.loads((root / "results/blur_sensitivity/summary.json").read_text(encoding="utf-8")),
        "frozen_resnet18": json.loads((root / "results/pretrained_comparison/summary.json").read_text(encoding="utf-8")),
    }
    rows = []
    for name in pipelines:
        for index, radius in enumerate(RADII):
            metrics = {state: probability_metrics(pooled[name][radius][state], y, classes)
                       for state in ("raw", "temperature_scaled")}
            expected = references[name]["summary"][index]
            if expected["radius_px"] != radius or metrics["raw"]["errors"] != expected["unfiltered_errors"]:
                raise AssertionError(f"Unfiltered held-out counts differ from recorded {name} results at radius {radius}")
            rows.append({"model": name, "radius_px": radius, "raw": metrics["raw"],
                         "temperature_scaled": metrics["temperature_scaled"]})
    result = {
        "analysis": "Exploratory pooled out-of-fold calibration on AZH training-pool images",
        "retrospective_status": "Designed after previous results were visible; not preregistered.",
        "dataset": {"repository": REPOSITORY, "commit": COMMIT, "training_images": len(train), "audit": audit},
        "protocol": {
            "outer_seed": OUTER_SEED, "outer_folds": 5, "patient_or_source_grouping": False,
            "test_archive": "Read for exact deduplication only; original-test predictions are not scored.",
            "temperature": "Fitted on separate clean inner calibration images per fold and frozen across blur levels.",
            "pooling": "Metrics and equal-width top-label ECE bins are calculated on all 545 held-out predictions per condition.",
            "brier": "Mean across images of the sum across four classes of squared probability minus one-hot label.",
            "nll": "Mean negative log probability of the true class, clipped below at 1e-15.",
            "ece": "10 equal-width bins; top-confidence 1.0 is in the final bin; empty bins contribute zero.",
            "limitations": "Image-level folds, uncertain patient overlap, synthetic blur, no clinical validation or patient-level uncertainty.",
            "weights_url": WEIGHT_URL, "weights_sha256": WEIGHT_SHA256,
        },
        "summary": rows, "folds": fold_details,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__, "pillow": pillow_version,
                        "torch": torch_version, "torchvision": vision_version},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["model", "radius_px", "state", "n", "errors", "accuracy", "mean_confidence", "nll", "brier_multiclass", "ece_top_label_10_equal_width_bins"])
        for row in rows:
            for state in ("raw", "temperature_scaled"):
                m = row[state]
                writer.writerow([row["model"], row["radius_px"], state, m["n"], m["errors"], m["accuracy"],
                                 m["mean_confidence"], m["nll"], m["brier_multiclass"], m["ece_top_label_10_equal_width_bins"]])
    print((args.output / "metrics.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
