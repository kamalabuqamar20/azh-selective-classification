"""Pinned, verified LUTSeg source handling; raw patient images stay local."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

REVISION = "e5115d4b2e5ecfc727f530deca9f3b0d032fb4bb"
SOURCE = "https://huggingface.co/datasets/ksanchez84/LUTSeg"
RAW = f"{SOURCE}/resolve/{REVISION}"
METADATA_SHA256 = "b5441614ed425eeb463769e0283843829fdb15e3124216c804fc95fe9846c596"
MANIFEST_SHA256 = "e95c821acaae2779787ee4641896ef0571189e1096e16d86245056673540b29a"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_download(url: str, target: Path, expected: str, download: bool):
    """Use a pinned hash and require opt-in for every network download."""
    if target.is_file():
        if digest(target) != expected:
            raise ValueError(f"SHA-256 mismatch for {target}")
        return
    if not download:
        raise FileNotFoundError(f"Missing {target}; obtain verified files or supply --download-data")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "patient-grouped-wound-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
        if digest(temp) != expected:
            raise ValueError(f"SHA-256 mismatch for {target}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def safe_source_path(root: Path, name: str, category: str) -> Path:
    if not isinstance(name, str) or category not in ("Images", "Masks"):
        raise ValueError("source path or category is invalid")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or len(relative.parts) < 2 or relative.parts[0] != category or
            any(part in ("", ".", "..") for part in relative.parts) or "\\" in name):
        raise ValueError("invalid source path")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("source path escapes dataset directory")
    return target


def safe_image_path(root: Path, name: str) -> Path:
    return safe_source_path(root, name, "Images")


def validate_rows(data: bytes, expected_count: int | None = None):
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if not rows or (expected_count is not None and len(rows) != expected_count):
        raise ValueError("unexpected number of metadata rows")
    seen = set()
    for row in rows:
        group = row.get("patient_id")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("each image requires a nonempty patient group")
        name = row.get("image_file_name")
        safe_image_path(Path("/tmp/lutseg-path-validation"), name)
        safe_source_path(Path("/tmp/lutseg-path-validation"), row.get("mask_file_name"), "Masks")
        if name in seen:
            raise ValueError("duplicate image path")
        seen.add(name)
        present = row.get("classes_present")
        if (not isinstance(present, list) or any(not isinstance(v, int) or v not in range(6) for v in present)):
            raise ValueError("invalid tissue classes")
    return rows


def load_verified_data(directory: Path, download: bool):
    """Return in-memory metadata only after all 141 RGB image hashes verify."""
    metadata, manifest = directory / "metadata.jsonl", directory / "checksums.sha256"
    checked_download(f"{RAW}/metadata.jsonl", metadata, METADATA_SHA256, download)
    checked_download(f"{RAW}/checksums.sha256", manifest, MANIFEST_SHA256, download)
    rows = validate_rows(metadata.read_bytes(), expected_count=141)
    checks = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        if name in checks or len(checksum) != 64:
            raise ValueError("invalid checksum manifest")
        checks[name] = checksum

    def prepare(row):
        name = row["image_file_name"]
        target = safe_image_path(directory, name)
        if name not in checks:
            raise ValueError("image missing from checksum manifest")
        url = f"{RAW}/{urllib.parse.quote(name, safe='/')}"
        checked_download(url, target, checks[name], download)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(prepare, rows))
    return rows
