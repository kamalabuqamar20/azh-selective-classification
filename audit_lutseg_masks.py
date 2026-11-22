#!/usr/bin/env python3
"""Check LUTSeg metadata against source masks and summarise label size."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.parse

import numpy as np
from PIL import Image

from lutseg_data import (MANIFEST_SHA256, METADATA_SHA256, RAW, REVISION, SOURCE,
                         checked_download, load_verified_data, safe_image_path, safe_source_path)


def summarise_areas(fractions, groups):
    """Aggregate necrotic fraction of annotated wound pixels; never output IDs."""
    fractions, groups = np.asarray(fractions, dtype=float), np.asarray(groups)
    if (fractions.ndim != 1 or groups.ndim != 1 or not len(fractions) or len(fractions) != len(groups) or
            not np.isfinite(fractions).all() or np.any(fractions < 0) or np.any(fractions > 1)):
        raise ValueError("fractions and groups must be equal finite vectors with fractions in [0, 1]")
    positive = fractions > 0
    values = fractions[positive]
    thresholds = []
    for threshold in (0.001, 0.01, 0.05, 0.10):
        selected = fractions >= threshold
        thresholds.append({"threshold_fraction": threshold, "images": int(selected.sum()),
                           "patient_groups": len(set(groups[selected]))})
    return {"images": len(fractions), "patient_groups": len(set(groups)),
            "positive_images": int(positive.sum()), "positive_patient_groups": len(set(groups[positive])),
            "positive_fraction_min": float(values.min()) if len(values) else None,
            "positive_fraction_median": float(np.median(values)) if len(values) else None,
            "positive_fraction_max": float(values.max()) if len(values) else None,
            "minimum_area_thresholds": thresholds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-data", action="store_true")
    args = parser.parse_args()
    rows = load_verified_data(args.data_dir, args.download_data)
    manifest = {}
    for line in (args.data_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        manifest[name.strip().lstrip("*")] = digest

    def prepare(row):
        name = row["mask_file_name"]
        if name not in manifest:
            raise ValueError("source mask missing from checksum manifest")
        target = safe_source_path(args.data_dir, name, "Masks")
        checked_download(f"{RAW}/{urllib.parse.quote(name, safe='/')}", target, manifest[name], args.download_data)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(prepare, rows))
    fractions = []
    for row in rows:
        with Image.open(safe_source_path(args.data_dir, row["mask_file_name"], "Masks")) as source_mask:
            mask = np.asarray(source_mask)
            mask_size = source_mask.size
        with Image.open(safe_image_path(args.data_dir, row["image_file_name"])) as source_image:
            image_size = source_image.size
        if mask.ndim != 2 or mask_size != image_size or set(np.unique(mask).tolist()) != set(row["classes_present"]):
            raise ValueError("source mask dimensions or classes do not match corresponding image metadata")
        wound_pixels = int((mask > 0).sum())
        if not wound_pixels:
            raise ValueError("reference mask has no annotated wound tissue")
        fractions.append(int((mask == 4).sum()) / wound_pixels)
    groups = [row["patient_id"].strip() for row in rows]
    result = {"analysis": "Aggregate verification and necrotic-area audit of source LUTSeg masks",
              "dataset": {"source": SOURCE, "revision": REVISION, "metadata_sha256": METADATA_SHA256,
                          "manifest_sha256": MANIFEST_SHA256},
              "protocol": {"mask_class": 4, "denominator": "number of pixels assigned any wound tissue class 1-5",
                           "checks": "source metadata classes and RGB/mask dimensions checked for every image",
                           "limitations": "reference annotation uncertainty, wound-area thresholds are descriptive and not clinical definitions"},
              "summary": summarise_areas(fractions, groups)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
