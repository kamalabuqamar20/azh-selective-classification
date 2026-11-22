#!/usr/bin/env python3
"""Exploratory patient-grouped necrotic-tissue presence pilot on LUTSeg.

This is a new binary image-level target derived from segmentation metadata,
not the four-class AZH wound-type task or clinical validation. Images, patient
IDs, individual predictions and intermediate embeddings are never exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from lutseg_data import MANIFEST_SHA256, METADATA_SHA256, REVISION, SOURCE, load_verified_data, safe_image_path
from patient_splits import patient_disjoint_folds, patient_disjoint_inner
from run_azh_pilot import describe_image
from run_blur_sensitivity import threshold_for_coverage
from run_pretrained_comparison import WEIGHT_SHA256, WEIGHT_URL, ensure_weights, prepare_image
from study_utils import fit_correctness_selectors, scores_for_policies

OUTER_SEED = 20261121
INNER_SEED = 20261122
COVERAGE = .8
POLICIES = ("confidence", "learned_confidence", "confidence_plus_focus")
LABELS = ("absent", "present")


def register_decoded_image(seen, image, patient_group, label):
    """Reject identical decoded images across groups or with conflicting labels."""
    rgb = image.convert("RGB")
    key = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    if key in seen:
        previous_group, previous_label = seen[key]
        if previous_group != patient_group:
            raise ValueError("an identical decoded image appears in different patient groups")
        if previous_label != label:
            raise ValueError("an identical decoded image has conflicting target labels")
        return True
    seen[key] = (patient_group, label)
    return False


def extract_features(rows, directory: Path, weight_path: Path, batch_size: int):
    """Apply the same fixed image views as the AZH pretrained comparison."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError("Install the optional CPU torch dependencies documented for this pilot") from exc
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    baseline = np.empty((len(rows), 704), dtype=np.float64)
    focus = np.empty(len(rows), dtype=np.float64)
    pretrained = np.empty((len(rows), 512), dtype=np.float32)
    batch, indexes, seen = [], [], {}
    within_patient_duplicates = 0

    def flush():
        if not batch:
            return
        with torch.inference_mode():
            out = encoder(torch.from_numpy(np.stack(batch))).cpu().numpy()
        pretrained[indexes] = out
        batch.clear()
        indexes.clear()

    for index, row in enumerate(rows):
        path = safe_image_path(directory, row["image_file_name"])
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        within_patient_duplicates += register_decoded_image(
            seen, image, row["patient_id"].strip(), 4 in row["classes_present"])
        baseline[index], focus[index] = describe_image(image)
        batch.append(prepare_image(image, 0.0))
        indexes.append(index)
        if len(batch) >= batch_size:
            flush()
    flush()
    audit = {"unique_decoded_images": len(seen),
             "within_patient_duplicate_images": within_patient_duplicates,
             "cross_patient_decoded_duplicates": 0,
             "conflicting_target_labels": 0}
    return {"handcrafted": baseline, "frozen_resnet18": pretrained}, focus, torch.__version__, torchvision.__version__, audit


def binary_logits(model, features):
    """Represent the fitted binary log odds as two equivalent softmax logits."""
    z = np.asarray(model.decision_function(features))
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("expected finite binary decision scores")
    return np.column_stack([np.zeros_like(z), z])


def fit_and_score(features, focus, labels, groups, outer_train, outer_test, class_weight=None):
    model_idx, cal_idx, selector_idx = patient_disjoint_inner(
        labels, groups, outer_train, n_splits=5, seed=INNER_SEED)
    pieces = (model_idx, cal_idx, selector_idx, outer_test)
    sets = [set(groups[p]) for p in pieces]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("patient groups crossed a fitting or held-out partition")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or balanced")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, solver="lbfgs",
                                                               class_weight=class_weight))
    model.fit(features[model_idx], labels[model_idx])
    if tuple(model.classes_) != LABELS:
        raise ValueError("binary classifier must include both target classes")
    cal_logits = binary_logits(model, features[cal_idx])

    def loss(log_temperature):
        p = softmax(cal_logits / np.exp(log_temperature), axis=1)
        return float(log_loss(labels[cal_idx], p, labels=model.classes_))

    optimum = minimize_scalar(loss, bounds=(-3, 3), method="bounded")
    if not optimum.success:
        raise RuntimeError("temperature fit failed")
    temperature = float(np.exp(optimum.x))

    def probabilities(idx):
        return softmax(binary_logits(model, features[idx]) / temperature, axis=1)

    p_sel = probabilities(selector_idx)
    correct = (model.classes_[p_sel.argmax(axis=1)] == labels[selector_idx]).astype(int)
    if set(correct.tolist()) != {0, 1}:
        raise ValueError("selector-fit patients lack both correct and incorrect predictions; comparison not estimable")
    one, joint = fit_correctness_selectors(p_sel, focus[selector_idx], correct)
    cal_scores = scores_for_policies(probabilities(cal_idx), focus[cal_idx], one, joint)
    test_probs = probabilities(outer_test)
    test_scores = scores_for_policies(test_probs, focus[outer_test], one, joint)
    predictions = model.classes_[test_probs.argmax(axis=1)]
    thresholds = {name: threshold_for_coverage(cal_scores[name], COVERAGE) for name in POLICIES}
    sizes = {name: {"images": len(idx), "patient_groups": len(set(groups[idx])),
                    "positive_images": int((labels[idx] == "present").sum())}
             for name, idx in zip(("model_fit", "calibration", "selector_fit", "test"), pieces)}
    return predictions, test_scores, thresholds, sizes, temperature


def selection_mask(scores, threshold, kind):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if kind == "matched_coverage":
        keep = np.zeros(len(scores), dtype=bool)
        keep[np.argsort(-scores, kind="stable")[:math.ceil(COVERAGE * len(scores))]] = True
        return keep
    if kind == "fixed_threshold":
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        return scores >= threshold
    raise ValueError("unknown selection evaluation")


def selection_metrics(errors, labels, groups, keep):
    """Aggregate image risk and macro patient risk without exporting IDs."""
    errors, labels, groups, keep = map(np.asarray, (errors, labels, groups, keep))
    if (any(x.ndim != 1 for x in (errors, labels, groups, keep)) or not len(errors) or
            len({len(errors), len(labels), len(groups), len(keep)}) != 1 or
            keep.dtype != np.dtype(bool) or not np.isin(errors, (0, 1)).all()):
        raise ValueError("invalid held-out selection vectors")
    accepted = int(keep.sum())
    patients = np.unique(groups)
    patient_risks = [float(errors[keep & (groups == g)].mean()) for g in patients if (keep & (groups == g)).any()]
    classes = {}
    for label in LABELS:
        cls = labels == label
        kept = keep & cls
        classes[label] = {"images": int(cls.sum()), "accepted": int(kept.sum()),
                          "errors": int(errors[kept].sum()),
                          "risk": float(errors[kept].mean()) if kept.any() else None}
    return {"images": len(errors), "patient_groups": len(patients), "accepted": accepted,
            "errors": int(errors[keep].sum()), "coverage": accepted / len(errors),
            "risk": float(errors[keep].mean()) if accepted else None,
            "patients_with_retained_images": len(patient_risks),
            "patients_all_images_deferred": len(patients) - len(patient_risks),
            "mean_patient_error_rate": float(np.mean(patient_risks)) if patient_risks else None,
            "classes": classes}


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
    rows = load_verified_data(args.data_dir, args.download_data)
    labels = np.asarray(["present" if 4 in row["classes_present"] else "absent" for row in rows])
    groups = np.asarray([row["patient_id"].strip() for row in rows])
    if len(set(groups)) != 39 or int((labels == "present").sum()) != 32:
        raise ValueError("pinned dataset group or target counts differ from the audited release")
    features, focus, torch_version, vision_version, pixel_audit = extract_features(rows, args.data_dir, args.weights, args.batch_size)
    splits = patient_disjoint_folds(labels, groups, n_splits=5, seed=OUTER_SEED)
    if sorted(np.concatenate([test for _, test in splits]).tolist()) != list(range(len(rows))):
        raise AssertionError("each source image must appear once in held-out folds")
    output = {}
    with threadpool_limits(limits=1):
        variants = [(name, x, weight) for name, x in features.items() for weight in (None, "balanced")]
        for base_name, x, class_weight in variants:
            name = base_name if class_weight is None else base_name + "_balanced"
            errors = np.zeros(len(rows), dtype=int)
            selected = {kind: {policy: np.zeros(len(rows), dtype=bool) for policy in POLICIES}
                        for kind in ("matched_coverage", "fixed_threshold")}
            folds = []
            for number, (tr, te) in enumerate(splits, 1):
                predictions, scores, thresholds, sizes, temperature = fit_and_score(x, focus, labels, groups, tr, te, class_weight)
                errors[te] = (predictions != labels[te]).astype(int)
                fold = {"fold": number, "sizes": sizes, "temperature": temperature,
                        "unfiltered_errors": int(errors[te].sum()), "policies": {}}
                for policy in POLICIES:
                    fold["policies"][policy] = {}
                    for kind in selected:
                        keep = selection_mask(scores[policy], thresholds[policy], kind)
                        selected[kind][policy][te] = keep
                        fold["policies"][policy][kind] = {"accepted": int(keep.sum()),
                                                           "errors": int(errors[te][keep].sum())}
                folds.append(fold)
            summaries = {kind: {policy: selection_metrics(errors, labels, groups, mask)
                                for policy, mask in policies.items()} for kind, policies in selected.items()}
            output[name] = {"class_weight": class_weight,
                            "unfiltered": selection_metrics(errors, labels, groups, np.ones(len(rows), dtype=bool)),
                            "unfiltered_errors": int(errors.sum()), "unfiltered_accuracy": float(1 - errors.mean()),
                            "selection": summaries, "folds": folds}
    result = {
        "analysis": "Exploratory patient-grouped LUTSeg necrotic-tissue-presence classification",
        "retrospective_status": "Designed after AZH findings and LUTSeg metadata were inspected; class-balanced variants were added after the unweighted pilot was observed; not preregistered.",
        "scope": "Different task from AZH four-class wound type; not an external validation of the AZH classifier.",
        "dataset": {"source": SOURCE, "revision": REVISION, "metadata_sha256": METADATA_SHA256,
                    "manifest_sha256": MANIFEST_SHA256, "images": len(rows), "patient_groups": len(set(groups)),
                    "positive_images": int((labels == "present").sum()),
                    "positive_patient_groups": len(set(groups[labels == "present"])),
                    "negative_patient_groups": len(set(groups[labels == "absent"])), "decoded_pixel_audit": pixel_audit},
        "always_absent_reference": {"images": len(rows), "errors": int((labels == "present").sum()),
                                    "accuracy": float((labels == "absent").mean()),
                                    "positive_recall": 0.0,
                                    "description": "Fixed constant comparator; not a trained or selective model."},
        "protocol": {"target": "Binary image-level presence of at least one class-4 necrotic-tissue pixel in source reference mask metadata.",
                     "caveat": "Derived from segmentation metadata, not a separately clinically adjudicated binary diagnosis; threshold is any pixel.",
                     "grouping": "Five outer patient-disjoint folds; within each outer training pool, five group-disjoint inner folds separate classifier, temperature, and selector fitting.",
                     "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED, "requested_coverage": COVERAGE,
                     "selection": "Matched: per-fold test-score ranking. Fixed: threshold from calibration scores, frozen before held-out evaluation.",
                     "posthoc_class_weight": "Balanced loss variants were added after inspecting unweighted pilot results and remain exploratory.",
                     "data_usage": "Authors' published train/validation are pooled for this new exploratory grouped CV; no LUTSeg-trained weights are used.",
                     "limitations": "39 patients with leprosy-related ulcers; repeated observations, small inner subsets, synthetic binary target, no independent external cohort or clinical validation.",
                     "weights_url": WEIGHT_URL, "weights_sha256": WEIGHT_SHA256},
        "models": output,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__, "pillow": pillow_version,
                        "torch": torch_version, "torchvision": vision_version},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["model", "evaluation", "policy", "images", "patient_groups", "accepted", "errors", "coverage", "risk", "patients_with_retained_images", "patients_all_images_deferred", "mean_patient_error_rate"])
        for name, model in output.items():
            for kind, policies in model["selection"].items():
                for policy, m in policies.items():
                    writer.writerow([name, kind, policy, m["images"], m["patient_groups"], m["accepted"], m["errors"], m["coverage"], m["risk"], m["patients_with_retained_images"], m["patients_all_images_deferred"], m["mean_patient_error_rate"]])
    with (args.output / "class_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["model", "evaluation", "policy", "class", "images", "accepted", "errors", "risk"])
        for name, model in output.items():
            comparisons = [("unfiltered", "all", model["unfiltered"])]
            comparisons.extend((kind, policy, m) for kind, policies in model["selection"].items()
                               for policy, m in policies.items())
            for kind, policy, m in comparisons:
                for label, values in m["classes"].items():
                    writer.writerow([name, kind, policy, label, values["images"], values["accepted"],
                                     values["errors"], values["risk"]])
    print((args.output / "metrics.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
