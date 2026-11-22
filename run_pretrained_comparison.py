#!/usr/bin/env python3
"""Exploratory frozen ResNet-18 comparison on AZH training-pool folds only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps, __version__ as pillow_version
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from run_azh_pilot import COMMIT, REPOSITORY, load_data, verified_archive
from run_blur_sensitivity import RADII, build_variants, evaluate_conditions, summarise
from run_train_cv import OUTER_SEED
from study_utils import outer_folds

WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
MEAN = np.asarray([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.asarray([.229, .224, .225], dtype=np.float32)[:, None, None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path: Path, download: bool) -> None:
    """Use the pinned official file only; never silently download weights."""
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {path}; supply the pinned weights or use --download-weights")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        try:
            request = urllib.request.Request(WEIGHT_URL, headers={"User-Agent": "AZH-research-comparison"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
            if sha256_file(temporary) != WEIGHT_SHA256:
                raise ValueError("Pretrained checkpoint SHA-256 mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256_file(path) != WEIGHT_SHA256:
        raise ValueError("Pretrained checkpoint SHA-256 mismatch")


def prepare_image(image: Image.Image, radius: float) -> np.ndarray:
    """Apply the baseline's 128-pixel blur, then ImageNet normalisation."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("blur radius must be finite and non-negative")
    resized = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    if radius:
        resized = resized.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    resized = resized.resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return (pixels - MEAN) / STD


def extract_embeddings(samples, directory: Path, weight_path: Path, batch_size: int):
    """Extract frozen embeddings for train images only; persist no features."""
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError("Install optional CPU torch and torchvision as described in docs/PRETRAINED_COMPARISON.md") from exc
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.eval()

    indexes = {sample.digest: i for i, sample in enumerate(samples)}
    if len(indexes) != len(samples):
        raise ValueError("training sample byte digests must be unique")
    embeddings = {radius: np.empty((len(samples), 512), dtype=np.float32) for radius in RADII}
    found = set()
    pending, positions = [], []

    def flush():
        if not pending:
            return
        inputs = torch.from_numpy(np.stack(pending))
        with torch.inference_mode():
            values = encoder(inputs).cpu().numpy()
        for value, (idx, radius) in zip(values, positions):
            embeddings[radius][idx] = value
        pending.clear()
        positions.clear()

    with zipfile.ZipFile(verified_archive(directory, "Train", False)) as archive:
        for name in sorted(archive.namelist()):
            if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            blob = archive.read(name)
            digest = hashlib.sha256(blob).hexdigest()
            if digest not in indexes or digest in found:
                continue
            idx = indexes[digest]
            with Image.open(io.BytesIO(blob)) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            for radius in RADII:
                pending.append(prepare_image(image, radius))
                positions.append((idx, radius))
                if len(pending) >= batch_size:
                    flush()
            found.add(digest)
    flush()
    if len(found) != len(samples):
        raise ValueError(f"Found {len(found)} of {len(samples)} deduplicated training images")
    return embeddings, torch.__version__, torchvision.__version__


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
    training = [sample for sample in samples if sample.split == "Train"]
    if len(training) != 545:
        raise ValueError(f"Expected 545 deduplicated training wound images, got {len(training)}")
    labels = np.asarray([sample.label for sample in training])
    _, focus = build_variants(training, args.data_dir)
    embeddings, torch_version, torchvision_version = extract_embeddings(training, args.data_dir, args.weights, args.batch_size)
    folds, seen = [], []
    with threadpool_limits(limits=1):
        for number, (outer_train, outer_test) in enumerate(outer_folds(labels, seed=OUTER_SEED), 1):
            folds.append(evaluate_conditions(embeddings, focus, labels, outer_train, outer_test, number))
            seen.extend(outer_test.tolist())
    if sorted(seen) != list(range(len(training))):
        raise AssertionError("Each image must occur in exactly one outer test fold")
    result = {
        "analysis": "Exploratory frozen ImageNet ResNet-18 on the same AZH training-pool blur folds",
        "retrospective_status": "Designed after original and blur results were visible; not preregistered.",
        "test_archive_used": "Read for exact deduplication only; no original-test predictions scored.",
        "patient_or_source_grouping": False,
        "dataset": {"repository": REPOSITORY, "commit": COMMIT, "training_images": len(training), "audit": audit},
        "protocol": {
            "weights": "ResNet18_Weights.IMAGENET1K_V1", "weight_source": WEIGHT_URL,
            "weight_sha256": WEIGHT_SHA256, "encoder": "Frozen ResNet-18 with identity classification layer",
            "features": 512, "training": "Clean inner data only; logistic head, temperature and selectors refit per fold.",
            "preprocessing": "EXIF transpose, RGB, 128-square bilinear resize, synthetic blur, 224-square bilinear resize, ImageNet normalisation.",
            "coverage": "Matched rankings are per fold and condition; thresholds are fixed from clean calibration scores.",
            "outer_seed": OUTER_SEED, "outer_folds": 5, "blur_radius_px_at_128": list(RADII),
            "device": "cpu", "torch_threads": 4, "batch_size": args.batch_size,
            "limitations": "Image-level folds, unknown patient overlap, synthetic blur, no clinical validation.",
        },
        "summary": summarise(folds, focus), "folds": folds,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "pillow": pillow_version, "torch": torch_version, "torchvision": torchvision_version},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["radius_px", "unfiltered_errors", "policy", "evaluation", "images", "accepted", "errors", "coverage", "risk"])
        for row in result["summary"]:
            for policy in ("learned_confidence", "confidence_plus_focus"):
                for kind in ("matched_coverage", "fixed_threshold"):
                    s = row[kind][policy]
                    writer.writerow([row["radius_px"], row["unfiltered_errors"], policy, kind,
                                     row["images"], s["accepted"], s["errors"], s["coverage"], s["risk"]])
    print(json.dumps([{
        "radius_px": row["radius_px"], "unfiltered_errors": row["unfiltered_errors"],
        "matched_confidence_errors": row["matched_coverage"]["learned_confidence"]["errors"],
        "matched_joint_errors": row["matched_coverage"]["confidence_plus_focus"]["errors"],
        "fixed_confidence_coverage": row["fixed_threshold"]["learned_confidence"]["coverage"],
        "fixed_joint_coverage": row["fixed_threshold"]["confidence_plus_focus"]["coverage"],
    } for row in result["summary"]], indent=2))


if __name__ == "__main__":
    main()
