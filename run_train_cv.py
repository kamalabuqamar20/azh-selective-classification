#!/usr/bin/env python3
"""Exploratory image-level five-fold sensitivity analysis on the AZH train pool.

This was designed after the original test-pilot results were visible.
The official test archive is read for exact deduplication but is not scored in
this analysis. Patient and acquisition-source IDs are unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_azh_pilot import CLASSES, COMMIT, REPOSITORY, load_data
from study_utils import (fit_correctness_selectors, inner_partitions, outer_folds,
                         scores_for_policies, validated_risk)

OUTER_SEED = 20261109
INNER_SEED = 42
COVERAGE = .80
POLICIES = ("confidence", "focus", "learned_confidence", "confidence_plus_focus")


def evaluate_fold(x, focus, labels, outer_train, outer_test, fold_number):
    """Fit each component inside an outer fold, then evaluate its held-out fold."""
    model_idx, calibration_idx, selector_idx = inner_partitions(labels, outer_train, seed=INNER_SEED)
    partitions = [set(x.tolist()) for x in (model_idx, calibration_idx, selector_idx, outer_test)]
    assert sum(map(len, partitions)) == len(set.union(*partitions))

    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, solver="lbfgs"))
    model.fit(x[model_idx], labels[model_idx])
    classes = model.classes_
    cal_logits = model.decision_function(x[calibration_idx])

    def loss(log_temperature):
        p = softmax(cal_logits / np.exp(log_temperature), axis=1)
        return float(log_loss(labels[calibration_idx], p, labels=classes))

    fit = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    if not fit.success:
        raise RuntimeError(f"Temperature fit failed in fold {fold_number}")
    temperature = float(np.exp(fit.x))
    selector_logits = model.decision_function(x[selector_idx])
    selector_probs = softmax(selector_logits / temperature, axis=1)
    selector_correct = (classes[selector_logits.argmax(axis=1)] == labels[selector_idx]).astype(int)
    one, joint = fit_correctness_selectors(selector_probs, focus[selector_idx], selector_correct)

    # No outer-fold labels are accessed for fitting the model, calibration or selectors.
    test_logits = model.decision_function(x[outer_test])
    test_probs = softmax(test_logits / temperature, axis=1)
    scores = scores_for_policies(test_probs, focus[outer_test], one, joint)
    errors = (classes[test_logits.argmax(axis=1)] != labels[outer_test]).astype(int)
    return {
        "fold": fold_number,
        "sizes": {"model_fit": len(model_idx), "calibration": len(calibration_idx),
                  "selector_fit": len(selector_idx), "outer_test": len(outer_test)},
        "temperature": temperature,
        "unfiltered_errors": int(errors.sum()),
        "unfiltered_accuracy": float(1-errors.mean()),
        "confidence_selector_slope": float(one[-1].coef_[0, 0]),
        "joint_selector_coefficients": [float(v) for v in joint[-1].coef_[0]],
        "policies": {name: validated_risk(errors, labels[outer_test], scores[name], COVERAGE)
                     for name in POLICIES},
    }


def aggregate(folds):
    """Pool retained counts; per-fold rankings are kept separate."""
    summary = {}
    total_test = sum(fold["sizes"]["outer_test"] for fold in folds)
    for name in POLICIES:
        accepted = sum(fold["policies"][name]["accepted"] for fold in folds)
        errors = sum(fold["policies"][name]["errors"] for fold in folds)
        by_class = {}
        for class_name in CLASSES.values():
            parts = [fold["policies"][name]["classes"][class_name] for fold in folds]
            n = sum(p["n"] for p in parts)
            kept = sum(p["accepted"] for p in parts)
            mistakes = sum(p["errors"] for p in parts)
            by_class[class_name] = {
                "n": n, "accepted": kept, "errors": mistakes,
                "coverage": kept / n, "risk": mistakes / kept if kept else None,
            }
        if all(v["accepted"] > 0 for v in by_class.values()):
            standardised = sum(v["n"]/total_test * v["risk"] for v in by_class.values())
        else:
            standardised = None
        summary[name] = {
            "test_images": total_test, "accepted": accepted, "errors": errors,
            "coverage": accepted/total_test, "risk": errors/accepted,
            "class_standardised_risk": standardised, "classes": by_class,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true",
                        help="Download the original pinned archives if absent.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    samples, audit = load_data(args.data_dir, args.download)
    train_samples = [s for s in samples if s.split == "Train"]
    if len(train_samples) != 545:
        raise ValueError(f"Expected 545 deduplicated wound images in train; got {len(train_samples)}")
    x = np.stack([sample.features for sample in train_samples])
    focus = np.asarray([sample.focus for sample in train_samples])
    labels = np.asarray([sample.label for sample in train_samples])
    folds, seen = [], []
    with threadpool_limits(limits=1):
        for n, (training, testing) in enumerate(outer_folds(labels, seed=OUTER_SEED), 1):
            folds.append(evaluate_fold(x, focus, labels, training, testing, n))
            seen.extend(testing.tolist())
    if sorted(seen) != list(range(len(train_samples))):
        raise AssertionError("Each image must be evaluated in exactly one outer fold.")
    overall = aggregate(folds)
    out = {
        "analysis": "Five-fold image-level exploration within the original training pool",
        "analysis_date": "2026-11-09",
        "research_question": "Does the focus proxy add to a matched learned confidence selector?",
        "retrospective_status": "Designed after the original test pilot was visible; not preregistered.",
        "test_archive_used": "Read for exact deduplication only; no original-test predictions scored.",
        "patient_or_source_grouping": False,
        "dataset": {"repository": REPOSITORY, "commit": COMMIT, "training_images": len(train_samples),
                    "audit": audit},
        "protocol": {"outer_folds": 5, "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED,
                     "requested_coverage": COVERAGE,
                     "model": "same gradient/RGB histogram multinomial logistic regression as original pilot",
                     "learned_selector_comparison": "one- and two-feature logistic selectors fitted on the same data"},
        "folds": folds,
        "summary": overall,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (args.output/"summary.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with (args.output/"fold_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fold","policy","n_test","accepted","errors","coverage","risk",
                         "class_standardised_risk","within_class_matched_risk"])
        for fold in folds:
            for name in POLICIES:
                m=fold["policies"][name]
                writer.writerow([fold["fold"],name,fold["sizes"]["outer_test"],m["accepted"],m["errors"],
                                 m["actual_coverage"],m["risk"],m["standardised_risk"],
                                 m["within_class_matched_risk"]])
    with (args.output/"class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["policy","class","n","accepted","errors","coverage","risk"])
        for name, result in overall.items():
            for cls, stats in result["classes"].items():
                writer.writerow([name,cls,stats["n"],stats["accepted"],stats["errors"],
                                 stats["coverage"],stats["risk"]])
    print(json.dumps({"sample_count":len(train_samples),
                      "summary":{k:{a:b for a,b in v.items() if a!="classes"} for k,v in overall.items()},
                      "folds":[{"fold":f["fold"],"errors":{k:f["policies"][k]["errors"] for k in POLICIES}}
                               for f in folds]},indent=2))


if __name__ == "__main__":
    main()
