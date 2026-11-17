"""Versioned, hashed model artifact manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


ARTIFACT_SCHEMA_VERSION = "video_suitability_model_artifact/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_artifact_directory(root: Path, model_type: str, model_version: str) -> Path:
    safe_type = model_type.replace("/", "_")
    safe_version = model_version.replace("/", "_")
    destination = Path(root) / f"{safe_type}__{safe_version}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_manifest(
    artifact_dir: Path,
    *,
    model_type: str,
    model_version: str,
    training_scope: str,
    files: Iterable[str],
    metadata: Dict[str, Any],
) -> Path:
    directory = Path(artifact_dir)
    hashes = {}
    for filename in files:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[filename] = sha256_file(path)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": model_type,
        "model_version": model_version,
        "training_scope": training_scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": hashes,
        "metadata": metadata,
    }
    path = directory / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_and_validate_manifest(artifact_dir: Path) -> Dict[str, Any]:
    directory = Path(artifact_dir)
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported model artifact schema")
    for filename, expected in manifest.get("files_sha256", {}).items():
        actual = sha256_file(directory / filename)
        if actual != expected:
            raise ValueError(f"Model artifact checksum mismatch: {filename}")
    return manifest

