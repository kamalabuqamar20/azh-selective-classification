#!/usr/bin/env python3
"""Aggregate-only feasibility audit of the WoundsDB case-description table."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.request

DESCRIPTION_URL = "https://dataverse.harvard.edu/api/access/datafile/14118414"
DESCRIPTION_SHA256 = "ed5f27bcd21400168e45af85a018559ef75e51cf55cacb6700a74bfce480c31a"
CASE = re.compile(r"case[_-]?(\d+)", re.IGNORECASE)
DAY = re.compile(r"day[_-]?(\d+)", re.IGNORECASE)


def audit_rows(rows):
    """Summarise supplied photo records; never return patient identifiers."""
    images = Counter()
    groups = defaultdict(set)
    labels_by_group = defaultdict(set)
    case_days, seen_paths = set(), set()
    count = 0
    for row in rows:
        if str(row.get("Photo", "")).strip() not in {"1", "1.0"}:
            continue
        path = str(row.get("Path to files", "")).strip()
        label = str(row.get("Wound aetiology", "")).strip()
        case, day = CASE.search(path), DAY.search(path)
        if not path or not label or not case or not day:
            raise ValueError("Each photo row needs a path with case/day and an aetiology label")
        if path in seen_paths:
            raise ValueError("Duplicate photo path in metadata")
        seen_paths.add(path)
        case_id, day_id = case.group(1), day.group(1)
        images[label] += 1
        groups[label].add(case_id)
        labels_by_group[case_id].add(label)
        case_days.add((case_id, day_id))
        count += 1
    if not count:
        raise ValueError("No eligible photo rows")
    labels = [{"source_aetiology": label, "photo_records": images[label],
               "case_groups": len(groups[label])} for label in sorted(images)]
    return {
        "photo_records": count,
        "distinct_case_identifiers": len(labels_by_group),
        "distinct_case_day_pairs": len(case_days),
        "case_groups_with_multiple_source_labels": sum(len(labels) > 1 for labels in labels_by_group.values()),
        "source_aetiology_counts": labels,
        "labels_with_fewer_than_five_case_groups": [item["source_aetiology"] for item in labels if item["case_groups"] < 5],
        "interpretation": "Counts reflect the supplied table, not clinical validation; case groups may contribute to multiple labels.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.description.is_file():
        if not args.download:
            raise FileNotFoundError("Provide the Dataverse description table or use --download")
        args.description.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DESCRIPTION_URL, headers={"User-Agent": "AZH-feasibility-audit/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(1_000_001)
        if hashlib.sha256(data).hexdigest() != DESCRIPTION_SHA256:
            raise ValueError("Description table SHA-256 mismatch")
        args.description.write_bytes(data)
    if hashlib.sha256(args.description.read_bytes()).hexdigest() != DESCRIPTION_SHA256:
        raise ValueError("Description table SHA-256 mismatch")
    with args.description.open(encoding="utf-8-sig", newline="") as stream:
        result = audit_rows(csv.DictReader(stream, delimiter="\t"))
    out = {
        "analysis": "Aggregate audit of WoundsDB case descriptions; no model was trained or evaluated",
        "dataset_doi": "10.7910/DVN/6A1HAG", "metadata_file": "description.tab",
        "metadata_source": DESCRIPTION_URL, "metadata_sha256": DESCRIPTION_SHA256,
        "source_interpretation": "The provider describes case_X as a patient identifier and day_Y as a visit day.",
        "results": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "aetiology_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_aetiology", "photo_records", "case_groups"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["source_aetiology_counts"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
