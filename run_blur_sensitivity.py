#!/usr/bin/env python3
"""Exploratory paired synthetic-blur stress test on the AZH training pool.

The original test archive is read for exact deduplication, but is never scored.
Outputs contain aggregate statistics only, not patient images or predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps, __version__ as pillow_version
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from run_azh_pilot import COMMIT, REPOSITORY, describe_image, load_data, verified_archive
from run_train_cv import COVERAGE, OUTER_SEED, POLICIES, aggregate, fit_fold
from study_utils import outer_folds, validated_risk

RADII = (0.0, 0.75, 1.5, 3.0)


def blur_features(image: Image.Image, radius: float) -> tuple[np.ndarray, float]:
    """Resize first; blur in 128-pixel coordinates; extract the fixed features."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("blur radius must be finite and non-negative")
    resized = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    if radius:
        resized = resized.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return describe_image(resized)


def build_variants(samples, directory: Path):
    """Return paired arrays for deduplicated training samples, never image files."""
    indexes = {sample.digest: i for i, sample in enumerate(samples)}
    if len(indexes) != len(samples):
        raise ValueError("training sample byte digests must be unique")
    x = {radius: np.empty((len(samples), 704), dtype=float) for radius in RADII}
    focus = {radius: np.empty(len(samples), dtype=float) for radius in RADII}
    found = set()
    with zipfile.ZipFile(verified_archive(directory, "Train", False)) as archive:
        for name in sorted(archive.namelist()):
            suffix = Path(name).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                continue
            blob = archive.read(name)
            digest = hashlib.sha256(blob).hexdigest()
            if digest not in indexes or digest in found:
                continue
            idx = indexes[digest]
            with Image.open(io.BytesIO(blob)) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            for radius in RADII:
                x[radius][idx], focus[radius][idx] = blur_features(image, radius)
            # Ensure the new clean path is identical to the existing benchmark.
            if not np.array_equal(x[0.0][idx], samples[idx].features) or not np.isclose(
                focus[0.0][idx], samples[idx].focus, rtol=0, atol=1e-12
            ):
                raise AssertionError("clean features differ from the existing pipeline")
            found.add(digest)
    if len(found) != len(samples):
        raise ValueError(f"Found {len(found)} of {len(samples)} deduplicated training images")
    return x, focus


def threshold_for_coverage(scores, coverage=COVERAGE):
    """Select a cutoff on development scores; ties may retain extra samples."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("threshold scores must be a nonempty finite vector")
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    keep = math.ceil(coverage * len(scores))
    return float(np.sort(scores)[-keep])


def fixed_summary(errors, scores, threshold, clean_keep):
    """Evaluate a frozen threshold and paired changes from clean decisions."""
    errors, scores, clean_keep = np.asarray(errors), np.asarray(scores), np.asarray(clean_keep)
    if (errors.ndim != 1 or scores.ndim != 1 or clean_keep.ndim != 1 or
            not len(errors) or len(errors) != len(scores) or len(scores) != len(clean_keep)):
        raise ValueError("errors, scores and clean decisions must be equal nonempty vectors")
    if (not np.isin(errors, [0, 1]).all() or not np.isfinite(scores).all() or
            not np.isfinite(threshold) or clean_keep.dtype != np.dtype(bool)):
        raise ValueError("invalid errors, scores, threshold or clean decisions")
    keep = scores >= threshold
    accepted = int(keep.sum())
    mistakes = int(errors[keep].sum())
    return {
        "accepted": accepted,
        "errors": mistakes,
        "coverage": accepted / len(errors),
        "risk": mistakes / accepted if accepted else None,
        "accepted_clean_deferred_here": int((clean_keep & ~keep).sum()),
        "deferred_clean_accepted_here": int((~clean_keep & keep).sum()),
    }


def evaluate_conditions(x, focus, labels, outer_train, outer_test, fold_number):
    """Fit on clean inner data once; score the same held-out images at each radius."""
    if set(x) != set(RADII) or set(focus) != set(RADII):
        raise ValueError("feature and focus arrays must contain the configured radii")
    fitted = fit_fold(x[0.0], focus[0.0], labels, outer_train, fold_number)
    inner = (fitted.model_idx, fitted.calibration_idx, fitted.selector_idx)
    if any(set(part.tolist()) & set(outer_test.tolist()) for part in inner):
        raise AssertionError("outer test images overlap inner partitions")

    # Calibration-set scores choose thresholds without any held-out scores.
    _, cal_scores = fitted.predict(x[0.0][fitted.calibration_idx], focus[0.0][fitted.calibration_idx])
    thresholds = {name: threshold_for_coverage(cal_scores[name]) for name in POLICIES}
    clean_predictions, clean_scores = fitted.predict(x[0.0][outer_test], focus[0.0][outer_test])
    clean_errors = (clean_predictions != labels[outer_test]).astype(int)
    clean_keeps = {name: clean_scores[name] >= thresholds[name] for name in POLICIES}
    conditions = []
    for radius in RADII:
        predictions, scores = fitted.predict(x[radius][outer_test], focus[radius][outer_test])
        errors = (predictions != labels[outer_test]).astype(int)
        conditions.append({
            "radius_px": radius,
            "unfiltered_errors": int(errors.sum()),
            "prediction_changed": int((predictions != clean_predictions).sum()),
            "correct_clean_wrong_here": int(((clean_errors == 0) & (errors == 1)).sum()),
            "wrong_clean_correct_here": int(((clean_errors == 1) & (errors == 0)).sum()),
            "matched": {name: validated_risk(errors, labels[outer_test], scores[name], COVERAGE)
                        for name in POLICIES},
            "fixed": {name: fixed_summary(errors, scores[name], thresholds[name], clean_keeps[name])
                      for name in POLICIES},
        })
    return {
        "fold": fold_number,
        "sizes": {"model_fit": len(fitted.model_idx), "calibration": len(fitted.calibration_idx),
                  "selector_fit": len(fitted.selector_idx), "outer_test": len(outer_test)},
        "fixed_thresholds_from_clean_calibration": thresholds,
        "conditions": conditions,
    }


def summarise(folds, focus):
    """Pool counts after ranking separately within each fold and condition."""
    n = sum(fold["sizes"]["outer_test"] for fold in folds)
    output = []
    for i, radius in enumerate(RADII):
        parts = [fold["conditions"][i] for fold in folds]
        matched_folds = [{"sizes": fold["sizes"], "policies": part["matched"]}
                         for fold, part in zip(folds, parts)]
        fixed = {}
        for name in POLICIES:
            records = [part["fixed"][name] for part in parts]
            accepted = sum(item["accepted"] for item in records)
            errors = sum(item["errors"] for item in records)
            fixed[name] = {
                "accepted": accepted, "errors": errors, "coverage": accepted / n,
                "risk": errors / accepted if accepted else None,
                "accepted_clean_deferred_here": sum(item["accepted_clean_deferred_here"] for item in records),
                "deferred_clean_accepted_here": sum(item["deferred_clean_accepted_here"] for item in records),
            }
        matched = aggregate(matched_folds)
        output.append({
            "radius_px": radius,
            "images": n,
            "median_laplacian_variance": float(np.median(focus[radius])),
            "images_with_lower_focus_than_clean": int((focus[radius] < focus[0.0]).sum()),
            "unfiltered_errors": sum(part["unfiltered_errors"] for part in parts),
            "unfiltered_error_rate": sum(part["unfiltered_errors"] for part in parts) / n,
            "prediction_changed": sum(part["prediction_changed"] for part in parts),
            "correct_clean_wrong_here": sum(part["correct_clean_wrong_here"] for part in parts),
            "wrong_clean_correct_here": sum(part["wrong_clean_correct_here"] for part in parts),
            "matched_coverage": matched,
            "matched_joint_minus_learned_confidence": (
                matched["confidence_plus_focus"]["risk"] - matched["learned_confidence"]["risk"]
            ),
            "fixed_threshold": fixed,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download the two pinned AZH archives if absent.")
    args = parser.parse_args()

    samples, audit = load_data(args.data_dir, args.download)
    training = [sample for sample in samples if sample.split == "Train"]
    if len(training) != 545:
        raise ValueError(f"Expected 545 deduplicated training wound images, got {len(training)}")
    labels = np.asarray([sample.label for sample in training])
    x, focus = build_variants(training, args.data_dir)
    folds, seen = [], []
    with threadpool_limits(limits=1):
        for number, (outer_train, outer_test) in enumerate(outer_folds(labels, seed=OUTER_SEED), 1):
            folds.append(evaluate_conditions(x, focus, labels, outer_train, outer_test, number))
            seen.extend(outer_test.tolist())
    if sorted(seen) != list(range(len(training))):
        raise AssertionError("Each image must occur in exactly one outer test fold")
    summary = summarise(folds, focus)
    out = {
        "analysis": "Exploratory paired synthetic-blur sensitivity on the AZH training pool",
        "retrospective_status": "Designed after the original pilot and training-pool results were visible; not preregistered.",
        "test_archive_used": "Read for exact deduplication only; no original-test predictions scored.",
        "patient_or_source_grouping": False,
        "dataset": {"repository": REPOSITORY, "commit": COMMIT, "training_images": len(training), "audit": audit},
        "protocol": {
            "outer_folds": 5, "outer_seed": OUTER_SEED, "requested_coverage": COVERAGE,
            "radius_pixels_after_resize": list(RADII), "resize": [128, 128],
            "blur": "Pillow ImageFilter.GaussianBlur (extended-box approximation)",
            "training": "All classifiers, temperature and selectors fitted on clean inner data once per fold.",
            "matched_coverage": "Top ceil(coverage * n) scores re-ranked separately in each fold and condition.",
            "fixed_threshold": "Cutoff set from clean calibration-set scores and frozen across held-out conditions; ties may increase retention.",
            "limitations": "Synthetic blur; image-level CV; unknown patient overlap; no clinical validation.",
        },
        "summary": summary,
        "folds": folds,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pillow": pillow_version},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["radius_px", "policy", "evaluation", "images", "accepted", "errors", "coverage", "risk"])
        for condition in summary:
            for name in POLICIES:
                for kind in ("matched_coverage", "fixed_threshold"):
                    result = condition[kind][name]
                    writer.writerow([condition["radius_px"], name, kind, condition["images"], result["accepted"],
                                     result["errors"], result["coverage"], result["risk"]])
    print(json.dumps([{
        "radius_px": s["radius_px"], "median_focus": s["median_laplacian_variance"],
        "unfiltered_error_rate": s["unfiltered_error_rate"],
        "matched_confidence_error": s["matched_coverage"]["learned_confidence"]["risk"],
        "matched_joint_error": s["matched_coverage"]["confidence_plus_focus"]["risk"],
        "fixed_confidence_coverage": s["fixed_threshold"]["learned_confidence"]["coverage"],
        "fixed_joint_coverage": s["fixed_threshold"]["confidence_plus_focus"]["coverage"],
    } for s in summary], indent=2))


if __name__ == "__main__":
    main()
