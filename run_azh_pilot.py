#!/usr/bin/env python3
"""Exploratory, image-only selective classification on the public AZH release.

Runs on CPU using the publisher's fixed train/test archives. No patient images
are written to the output folder. This is a methodological pilot, not clinical
validation or a replication of DFUC2021.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logit
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

REPOSITORY = "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations"
COMMIT = "e7dd90de113dcbd39665f2bd0e5613adeffff5fa"
RAW = "https://raw.githubusercontent.com/uwm-bigdata/wound-classification-using-images-and-locations"
ARCHIVES = {
    "Train": ("Train.zip", "ebfff40b57e626ae38e7d27b163cebd86c8ddf793b4f64b3fd8a9bef0d5eeddc"),
    "Test": ("Test.zip", "55ef08b374b80bb3231d6fef106362f77c2becec1b79fb3dd2ce69a6d54c6760"),
}
CLASSES = {"D": "Diabetic", "P": "Pressure", "S": "Surgical", "V": "Venous"}
PRIMARY_COVERAGE = 0.80
SPLIT_SEED = 42
BOOT_SEED = 20261108
BOOT_DRAWS = 2000


@dataclass
class Sample:
    digest: str
    pixel_digest: str
    label: str
    split: str
    features: np.ndarray
    focus: float


def verified_archive(directory: Path, split: str, download: bool) -> Path:
    name, expected_sha = ARCHIVES[split]
    path = directory / name
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {name}. Supply the authors' archive or use --download.")
        directory.mkdir(parents=True, exist_ok=True)
        url = f"{RAW}/{COMMIT}/dataset/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "AZH-selective-pilot"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(40_000_001)
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {name}")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {name}")
    return path


def describe_image(img: Image.Image) -> tuple[np.ndarray, float]:
    # One fixed resize for both baseline features and the focus proxy.
    im = img.resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(im)
    gray = np.asarray(im.convert("L"), dtype=np.float64)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    magnitude = np.hypot(gx, gy)
    angle_bin = np.floor(np.mod(np.arctan2(gy, gx), np.pi) * 8 / np.pi).astype(int) % 8
    features = []
    for y in range(0, 128, 16):
        for x in range(0, 128, 16):
            b = np.bincount(angle_bin[y:y+16, x:x+16].ravel(),
                            weights=magnitude[y:y+16, x:x+16].ravel(), minlength=8)
            features.extend(b / max(np.linalg.norm(b), 1e-12))
    for y in (0, 64):
        for x in (0, 64):
            for c in range(3):
                h = np.bincount((rgb[y:y+64, x:x+64, c] // 16).ravel(), minlength=16)
                features.extend(h / h.sum())
    # Four-neighbour Laplacian, interior pixels only. Variance in squared
    # 8-bit intensity units. This responds to edges/texture as well as blur.
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return np.asarray(features, dtype=np.float64), float(np.var(lap))


def load_data(directory: Path, download: bool) -> tuple[list[Sample], dict]:
    samples: list[Sample] = []
    raw_counts: Counter = Counter()
    size_counts: Counter = Counter()
    seen_bytes: dict = {}
    seen_pixels: dict = {}
    duplicate_counts: Counter = Counter()
    dedup_counts: Counter = Counter()
    # Keep test copies when the same image appears in both official splits.
    for split in ("Test", "Train"):
        with zipfile.ZipFile(verified_archive(directory, split, download)) as archive:
            for name in sorted(archive.namelist()):
                p = PurePosixPath(name)
                if (len(p.parts) != 3 or p.parts[0] != split or
                        p.suffix.lower() not in {".jpg", ".jpeg", ".png"} or p.name.startswith(".")):
                    continue
                label = p.parts[1]
                if label not in (*CLASSES, "N", "BG"):
                    raise ValueError(f"Unexpected class {label}")
                raw_counts[(split, label)] += 1
                blob = archive.read(name)
                digest = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as original:
                    img = ImageOps.exif_transpose(original).convert("RGB")
                if max(img.size) > 700 or min(img.size) > 525:
                    size_counts[label] += 1
                pixels = hashlib.sha256(f"{img.width}x{img.height}:".encode() + img.tobytes()).hexdigest()
                previous = seen_bytes.get(digest) or seen_pixels.get(pixels)
                if previous:
                    p_split, p_label = previous
                    if p_label != label:
                        raise ValueError("Identical image has conflicting source labels")
                    method = "byte_identical" if digest in seen_bytes else "pixel_identical"
                    duplicate_counts[(method, "across_split" if p_split != split else "within_split", label)] += 1
                    continue
                seen_bytes[digest] = (split, label)
                seen_pixels[pixels] = (split, label)
                dedup_counts[(split, label)] += 1
                if label in CLASSES:
                    features, focus = describe_image(img)
                    samples.append(Sample(digest, pixels, label, split, features, focus))
    audit = {
        "raw_counts": {split: {k: raw_counts[(split, k)] for k in (*CLASSES, "N", "BG")}
                       for split in ("Train", "Test")},
        "raw_images": sum(raw_counts.values()),
        "unique_images_after_exact_byte_and_pixel_dedup": sum(dedup_counts.values()),
        "removed_duplicates": [{"match": key[0], "scope": key[1], "class": key[2], "count": value}
                               for key, value in sorted(duplicate_counts.items())],
        "dimension_flag_counts_all_classes": {k: size_counts[k] for k in (*CLASSES, "N", "BG")},
        "dimension_flag": "long side >700 px or short side >525 px, before fixed resize",
    }
    assert audit["raw_images"] == 930, audit
    assert len({x.digest for x in samples}) == len(samples)
    assert len({x.pixel_digest for x in samples}) == len(samples)
    return samples, audit


def risk_summary(errors: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                 coverage: float) -> dict:
    n = len(errors)
    k = math.ceil(coverage * n)
    rank = np.argsort(-scores, kind="stable")
    keep = rank[:k]
    class_stats = {}
    standardised = 0.0
    within_class_errors, within_class_accepted = 0, 0
    for c, label in CLASSES.items():
        n_c = int(np.sum(labels == c))
        accepted = keep[labels[keep] == c]
        err_c = int(errors[accepted].sum())
        m_c = len(accepted)
        # Evaluation-only diagnostic with the same retention fraction in each
        # true class. This cannot be used as an inference-time selector.
        oracle_keep = rank[labels[rank] == c][:math.ceil(coverage * n_c)]
        within_class_errors += int(errors[oracle_keep].sum())
        within_class_accepted += len(oracle_keep)
        if m_c:
            standardised += (n_c / n) * (err_c / m_c)
        else:
            standardised = float("nan")
        class_stats[label] = {
            "n": n_c, "accepted": m_c, "errors": err_c, "coverage": m_c / n_c,
            "risk": err_c / m_c if m_c else None,
        }
    return {"accepted": k, "errors": int(errors[keep].sum()),
            "actual_coverage": k / n, "risk": float(errors[keep].mean()),
            "standardised_risk": float(standardised) if np.isfinite(standardised) else None,
            "within_class_matched_risk": within_class_errors / within_class_accepted,
            "within_class_matched_accepted": within_class_accepted, "classes": class_stats}


def confidence_ece(probs: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> float:
    conf = probs.max(axis=1)
    acc = (pred == labels).astype(float)
    b = np.minimum((10 * conf).astype(int), 9)
    return sum(float(np.mean(b == j) * abs(acc[b == j].mean() - conf[b == j].mean()))
               for j in range(10) if np.any(b == j))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--download", action="store_true", help="Retrieve only the two pinned public AZH archives.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(args.data_dir, args.download)
    x = np.stack([s.features for s in data])
    f = np.asarray([s.focus for s in data])
    y = np.asarray([s.label for s in data])
    idx_train = np.asarray([i for i, s in enumerate(data) if s.split == "Train"])
    test = np.asarray([i for i, s in enumerate(data) if s.split == "Test"])
    train, val = train_test_split(idx_train, test_size=.30, stratify=y[idx_train], random_state=SPLIT_SEED)
    calib, selector = train_test_split(val, test_size=.50, stratify=y[val], random_state=SPLIT_SEED)
    split_sets = [set(p) for p in (train, calib, selector, test)]
    assert sum(map(len, split_sets)) == len(set.union(*split_sets)) == len(data)

    with threadpool_limits(limits=1):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, solver="lbfgs"))
        model.fit(x[train], y[train])
        class_order = model.classes_
        z_cal = model.decision_function(x[calib])
        def nll(log_temperature: float) -> float:
            return float(log_loss(y[calib], softmax(z_cal / np.exp(log_temperature), axis=1), labels=class_order))
        optimum = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        if not optimum.success:
            raise RuntimeError("Temperature fit did not converge")
        temp = float(np.exp(optimum.x))
        z_sel = model.decision_function(x[selector])
        p_sel = softmax(z_sel / temp, axis=1)
        ys = (class_order[z_sel.argmax(axis=1)] == y[selector]).astype(int)
        def selector_inputs(probabilities: np.ndarray, variance: np.ndarray) -> np.ndarray:
            return np.column_stack([logit(np.clip(probabilities.max(axis=1), 1e-8, 1-1e-8)),
                                    np.log1p(variance)])
        joint = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, solver="lbfgs"))
        joint.fit(selector_inputs(p_sel, f[selector]), ys)

        # The fixed model and both validation fits are complete before scoring test.
        z = model.decision_function(x[test])
        raw = softmax(z, axis=1)
        calibrated = softmax(z / temp, axis=1)
        labels = y[test]
        pred = class_order[z.argmax(axis=1)]
        errors = (pred != labels).astype(int)
        scores = {
            "temperature_scaled_confidence": calibrated.max(axis=1),
            "laplacian_variance": f[test],
            "confidence_plus_laplacian": joint.predict_proba(selector_inputs(calibrated, f[test]))[:, 1],
        }

    primary = {name: risk_summary(errors, labels, value, PRIMARY_COVERAGE)
               for name, value in scores.items()}
    # Paired, class-stratified image bootstrap; re-rank each resample at the same
    # coverage. Intervals condition on the fitted model and the test class mix.
    rng = np.random.default_rng(BOOT_SEED)
    groups = [np.flatnonzero(labels == c) for c in CLASSES]
    base_name, _, joint_name = scores
    delta = {"risk": [], "standardised_risk": [], "within_class_matched_risk": []}
    for _ in range(BOOT_DRAWS):
        b = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        base_b = risk_summary(errors[b], labels[b], scores[base_name][b], PRIMARY_COVERAGE)
        joint_b = risk_summary(errors[b], labels[b], scores[joint_name][b], PRIMARY_COVERAGE)
        for metric in delta:
            if base_b[metric] is not None and joint_b[metric] is not None:
                delta[metric].append(joint_b[metric] - base_b[metric])
    deltas = {metric: {
        "joint_minus_confidence": primary[joint_name][metric] - primary[base_name][metric],
        "percentile_95_interval": [float(q) for q in np.quantile(vals, [.025, .975])],
        "valid_bootstrap_draws": len(vals),
    } for metric, vals in delta.items()}

    with (args.output / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["selector", "accepted", "total", "coverage", "risk"])
        secondary = {}
        for name, value in scores.items():
            order = np.argsort(-value, kind="stable")
            k = np.arange(1, len(test) + 1)
            risks = np.cumsum(errors[order]) / k
            for m, risk in zip(k, risks):
                w.writerow([name, int(m), len(test), float(m/len(test)), float(risk)])
            secondary[name] = {
                "aurc": float(risks.mean()),
                "correctness_auroc": float(roc_auc_score(1-errors, value)),
            }
    all_splits = {"model_fit": train, "temperature_fit": calib, "joint_selector_fit": selector, "test": test}
    out = {
        "run_label": "AI-assisted exploratory four-class AZH image benchmark",
        "source": {"repository": REPOSITORY, "commit": COMMIT,
                   "archive_sha256": {v[0]: v[1] for v in ARCHIVES.values()}},
        "dataset_audit": audit,
        "class_map": CLASSES,
        "split_sizes": {name: int(len(idx)) for name, idx in all_splits.items()},
        "split_classes": {name: {CLASSES[c]: int(np.sum(y[idx] == c)) for c in CLASSES}
                          for name, idx in all_splits.items()},
        "model": "L2 multinomial logistic regression on 512 local gradient-histogram and 192 local RGB-histogram features",
        "temperature": temp,
        "test_unfiltered_accuracy": float(1-errors.mean()),
        "test_unfiltered_errors": int(errors.sum()),
        "test_nll_before": float(log_loss(labels, raw, labels=class_order)),
        "test_nll_after": float(log_loss(labels, calibrated, labels=class_order)),
        "test_ece_10bins_before": confidence_ece(raw, pred, labels),
        "test_ece_10bins_after": confidence_ece(calibrated, pred, labels),
        "joint_selector_standardised_coefficients": [float(v) for v in joint[-1].coef_[0]],
        "selector_fit_accuracy_of_fixed_classifier": float(ys.mean()),
        "primary_requested_coverage": PRIMARY_COVERAGE,
        "primary": primary,
        "paired_differences": deltas,
        "secondary": secondary,
        "test_focus_by_true_class": {CLASSES[c]: {
            "median": float(np.median(f[test][labels == c])),
            "q25": float(np.quantile(f[test][labels == c], .25)),
            "q75": float(np.quantile(f[test][labels == c], .75)),
        } for c in CLASSES},
        "settings": {"split_seed": SPLIT_SEED, "bootstrap_seed": BOOT_SEED,
                     "bootstrap_draws": BOOT_DRAWS, "resize": [128, 128]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "pillow": pillow_version},
    }
    (args.output / "results.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": audit, "splits": out["split_sizes"], "temperature": temp,
        "unfiltered_accuracy": out["test_unfiltered_accuracy"],
        "nll": [out["test_nll_before"], out["test_nll_after"]],
        "primary": {n: {k: v for k, v in m.items() if k != "classes"} for n, m in primary.items()},
        "joint_differences": deltas, "secondary": secondary,
    }, indent=2))


if __name__ == "__main__":
    main()

