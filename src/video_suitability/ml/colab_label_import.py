"""Read-only compatibility import for historical Colab COLMAP label batches."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..datasets.co3d import CO3DSequence, CO3DSequenceLoader
from ..video import uniform_sample_indices
from .artifacts import sha256_file
from .registration_training import (
    LABEL_SCHEMA_VERSION,
    REQUIRED_LABEL_COLUMNS,
    registration_sampling_configuration,
)


IMPORT_SCHEMA_VERSION = "colab_registration_import/1.0"
EXPECTED_INPUT_COUNT = 60
SUCCESS_STATUSES = {"success", "partial", "succeeded", "measured_success"}
FAILURE_STATUSES = {"failed", "reconstruction_failed"}
UNAVAILABLE_STATUSES = {
    "unavailable",
    "infrastructure_unavailable",
    "pipeline_exception",
    "not_run",
    "missing",
}
VERIFIABLE_FAILURE_STAGES = {
    "mapper_sequential",
    "mapper_exhaustive",
    "sparse_mapping",
}
FEATURE_COLUMN_NAMES = {
    "mean_brightness",
    "mean_contrast",
    "mean_sharpness",
    "mean_orb_keypoints",
    "mean_frame_difference",
    "mean_orb_good_matches",
    "mean_orb_match_ratio",
    "mean_orb_ransac_inlier_ratio",
}


class ColabImportError(ValueError):
    """Raised when one historical record cannot be imported safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not result:
        raise ColabImportError("record identity cannot be converted to a safe name")
    return result


def _first(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _required(row: Mapping[str, Any], names: Sequence[str], description: str) -> str:
    value = _first(row, names)
    if not value:
        raise ColabImportError(f"missing {description}; checked columns {list(names)}")
    return value


def _integer(row: Mapping[str, Any], names: Sequence[str], description: str) -> int:
    try:
        return int(float(_required(row, names, description)))
    except ValueError as exc:
        raise ColabImportError(f"{description} is not an integer") from exc


def _number(row: Mapping[str, Any], names: Sequence[str], description: str) -> float:
    try:
        value = float(_required(row, names, description))
    except ValueError as exc:
        raise ColabImportError(f"{description} is not numeric") from exc
    if not np.isfinite(value):
        raise ColabImportError(f"{description} is not finite")
    return value


def detect_colab_schema(columns: Sequence[str]) -> str:
    names = set(columns)
    if {"category", "sequence_name", "n_sampled_frames", "registration_ratio"} <= names:
        return "legacy_co3d_sequence_labels"
    if {"sequence", "status", "total_input_image_count", "registered_image_ratio"} <= names:
        return "persistent_reconstruction_summary"
    if {"sequence_id", "reconstruction_status", "registered_image_ratio"} <= names:
        return "generic_registration_metrics"
    return "unknown"


def _load_group_map(path: Optional[Path]) -> Dict[Tuple[str, str], Dict[str, str]]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "category",
            "sequence_name",
            "source_group",
            "source_group_provenance",
            "variant",
            "evaluation_role",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Group map is missing columns: {sorted(missing)}")
        output: Dict[Tuple[str, str], Dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            key = ((row.get("category") or "").strip(), (row.get("sequence_name") or "").strip())
            if not key[1] or key in output:
                raise ValueError(f"Missing/duplicate group-map identity at line {line_number}")
            output[key] = {name: (row.get(name) or "").strip() for name in required}
    return output


def _parse_path_list(value: str) -> List[str]:
    if not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except Exception:
            continue
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    return []


def _read_source_result(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabImportError(f"invalid source reconstruction JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ColabImportError(f"source reconstruction JSON is not an object: {path}")
    return value


def _find_source_result(
    row: Mapping[str, str],
    csv_path: Path,
    roots: Sequence[Path],
    category: str,
    sequence: str,
) -> Optional[Path]:
    explicit = _first(
        row,
        ["label_artifact", "reconstruction_result", "reconstruction_result_json"],
    )
    candidates: List[Path] = []
    if explicit:
        path = Path(explicit).expanduser()
        candidates.append((csv_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
    candidates.extend(
        [
            csv_path.parent / f"{category}__{sequence}.json",
            csv_path.parent / f"{sequence}.json",
        ]
    )
    for root in roots:
        for pattern in (
            f"**/{category}__{sequence}.json",
            f"**/{sequence}/**/reconstruction_result_persistent.json",
            f"**/{sequence}/**/reconstruction_result.json",
        ):
            candidates.extend(root.glob(pattern))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _metrics_from_source_result(source: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = source.get("objective_metrics")
    if isinstance(metrics, dict):
        return {
            "status": str(source.get("status", "")),
            "failure_stage": source.get("failure_stage"),
            "failure_reason": source.get("failure_reason"),
            "commands": source.get("commands") or [],
            "total": metrics.get("total_input_image_count", source.get("total_input_image_count")),
            "registered": metrics.get("registered_image_count"),
            "ratio": metrics.get("registered_image_ratio"),
            "points": metrics.get("reconstructed_point_count"),
            "reprojection": metrics.get("mean_reprojection_error_pixels"),
            "track_length": metrics.get("mean_track_length_images"),
        }
    return {
        "status": source.get("reconstruction_status", source.get("status", "")),
        "failure_stage": source.get("failure_stage"),
        "failure_reason": source.get("error", source.get("failure_reason")),
        "commands": source.get("commands") or [],
        "total": source.get("n_sampled_frames", source.get("total_input_image_count")),
        "registered": source.get("n_registered_cameras", source.get("registered_image_count")),
        "ratio": source.get("registration_ratio", source.get("registered_image_ratio")),
        "points": source.get("n_points_3d", source.get("reconstructed_point_count")),
        "reprojection": source.get("mean_reprojection_error", source.get("mean_reprojection_error_pixels")),
        "track_length": source.get("mean_track_length", source.get("mean_track_length_images")),
    }


def _validated_metrics(
    row: Mapping[str, str], source: Mapping[str, Any]
) -> Dict[str, Any]:
    total = _integer(row, ["n_sampled_frames", "total_input_image_count"], "input image count")
    registered = _integer(row, ["n_registered_cameras", "registered_image_count"], "registered image count")
    ratio = _number(row, ["registration_ratio", "registered_image_ratio"], "registration ratio")
    status = _required(row, ["reconstruction_status", "status"], "reconstruction status").lower()
    if total != EXPECTED_INPUT_COUNT:
        raise ColabImportError(
            f"label used {total} input images, but the final protocol requires exactly 60"
        )
    if not 0 <= registered <= total or not 0 <= ratio <= 1:
        raise ColabImportError("registration count/ratio is outside valid bounds")
    if not np.isclose(ratio, registered / total, atol=1e-9, rtol=1e-9):
        raise ColabImportError("registration ratio is inconsistent with counts")
    source_metrics = _metrics_from_source_result(source)
    for name, expected, actual in (
        ("input image count", total, source_metrics["total"]),
        ("registered image count", registered, source_metrics["registered"]),
        ("registration ratio", ratio, source_metrics["ratio"]),
    ):
        if actual is None:
            raise ColabImportError(f"source reconstruction artifact has no {name}")
        if not np.isclose(float(expected), float(actual), atol=1e-9, rtol=1e-9):
            raise ColabImportError(f"CSV {name} differs from source reconstruction artifact")
    source_status = str(source_metrics["status"] or "").lower()
    if status in SUCCESS_STATUSES:
        if source_status not in SUCCESS_STATUSES:
            raise ColabImportError("successful CSV row does not match source result status")
        normalized_status = "succeeded"
        failure_stage = None
        failure_verified = False
    elif status in FAILURE_STATUSES:
        failure_stage = str(source_metrics["failure_stage"] or "").lower()
        reason = str(source_metrics["failure_reason"] or "").lower()
        if source_status not in FAILURE_STATUSES:
            raise ColabImportError("failed CSV row does not match source result status")
        if registered != 0 or ratio != 0.0:
            raise ColabImportError("failed reconstruction target must explicitly record 0/60")
        if failure_stage not in VERIFIABLE_FAILURE_STAGES:
            if "no sparse model" in reason:
                failure_stage = "sparse_mapping"
            else:
                raise ColabImportError(
                    "failure is not proven to be a mapper/no-sparse-model failure"
                )
        if not source_metrics["commands"]:
            raise ColabImportError("failure artifact has no recorded COLMAP command evidence")
        normalized_status = "reconstruction_failed"
        failure_verified = True
    elif status in UNAVAILABLE_STATUSES:
        raise ColabImportError("infrastructure/unavailable result has no training target")
    else:
        raise ColabImportError(f"unsupported reconstruction status: {status}")
    return {
        "status": normalized_status,
        "failure_stage": failure_stage,
        "failure_verified": failure_verified,
        "total": total,
        "registered": registered,
        "ratio": ratio,
        "points": source_metrics["points"],
        "reprojection": source_metrics["reprojection"],
        "track_length": source_metrics["track_length"],
        "commands": source_metrics["commands"],
        "failure_reason": source_metrics["failure_reason"],
    }


def _directory_fingerprint(paths: Sequence[Path]) -> str:
    value = [{"name": path.name, "sha256": sha256_file(path)} for path in paths]
    return _sha256_json(value)


def _manifest_hashes(path: Path) -> Optional[Dict[str, str]]:
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        result = {str(row["filename"]): str(row["sha256"]).lower() for row in rows}
    except (KeyError, OSError):
        return None
    return result or None


def _input_integrity_hashes(path: Path) -> Optional[Dict[str, str]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hashes = payload.get("frame_sha256") if isinstance(payload, dict) else None
    if not isinstance(hashes, dict):
        return None
    return {str(name): str(value).lower() for name, value in hashes.items()}


def _find_exact_input_directory(
    row: Mapping[str, str], roots: Sequence[Path], category: str, sequence: str
) -> Optional[Path]:
    parsed_paths = _parse_path_list(_first(row, ["sampled_frame_paths"]))
    if parsed_paths:
        parents = {str(Path(value).parent) for value in parsed_paths}
        for parent in parents:
            candidate = Path(parent)
            if candidate.is_dir():
                return candidate.resolve()
    candidates: List[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / category / sequence / "images",
                root / "colmap" / category / sequence / "images",
                root / sequence / "images",
                root / "evaluation" / "reconstruction" / sequence / "images",
            ]
        )
        candidates.extend(
            path for path in root.glob(f"**/{sequence}/images") if path.is_dir()
        )
    for candidate in candidates:
        images = sorted(path for path in candidate.glob("*") if path.is_file())
        if len(images) == EXPECTED_INPUT_COUNT:
            return candidate.resolve()
    return None


def _write_collected_input(
    source_images: Sequence[Path],
    destination: Path,
    *,
    fidelity: str,
    hash_provenance: str,
    sampling_strategy: str,
    regeneration_provenance: Optional[Mapping[str, Any]] = None,
    source_indices: Optional[Sequence[int]] = None,
    timestamps: Optional[Sequence[Optional[float]]] = None,
) -> Dict[str, Any]:
    import cv2

    if len(source_images) != EXPECTED_INPUT_COUNT:
        raise ColabImportError(
            f"input collection contains {len(source_images)} files instead of 60"
        )
    input_root = destination
    images_dir = input_root / "images"
    images_dir.mkdir(parents=True)
    source_before = _directory_fingerprint(source_images)
    rows: List[Dict[str, Any]] = []
    dimensions = []
    for position, source in enumerate(source_images):
        suffix = source.suffix.lower() or ".jpg"
        filename = f"frame_{position:06d}{suffix}"
        target = images_dir / filename
        shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise ColabImportError(f"input copy checksum mismatch: {source}")
        image = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if image is None:
            raise ColabImportError(f"input image cannot be decoded: {source}")
        height, width = image.shape[:2]
        dimensions.append((width, height))
        rows.append(
            {
                "sequence_number": position,
                "filename": filename,
                "requested_source_frame_index": (
                    source_indices[position] if source_indices is not None else position
                ),
                "actual_source_frame_index": (
                    source_indices[position] if source_indices is not None else position
                ),
                "timestamp_seconds": (
                    timestamps[position]
                    if timestamps is not None and timestamps[position] is not None
                    else ""
                ),
                "width": width,
                "height": height,
                "sha256": sha256_file(target),
            }
        )
    source_after = _directory_fingerprint(source_images)
    if source_before != source_after:
        raise ColabImportError("source input files changed during import")
    manifest = input_root / "frames_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "source_kind": "co3d_sequence" if fidelity == "faithfully_regenerated" else "colab_colmap_input_directory",
        "source_integrity_before_sha256": source_before,
        "source_integrity_after_sha256": source_after,
        "input_collection_unchanged_during_import": True,
        "requested_sample_fps": None,
        "sampling_strategy": sampling_strategy,
        "requested_frame_count": EXPECTED_INPUT_COUNT,
        "extracted_frame_count": EXPECTED_INPUT_COUNT,
        "failed_source_frame_indices": [],
        "first_timestamp_seconds": rows[0]["timestamp_seconds"] or None,
        "last_timestamp_seconds": rows[-1]["timestamp_seconds"] or None,
        "temporal_coverage_fraction": 1.0,
        "output_width": dimensions[0][0],
        "output_height": dimensions[0][1],
        "output_dimensions_unique": sorted({f"{width}x{height}" for width, height in dimensions}),
        "linear_resize_scale": 1.0,
        "jpeg_quality": None,
        "copy_mode": "byte_for_byte_no_resize_no_reencode",
        "input_fidelity": fidelity,
        "input_hash_provenance": hash_provenance,
        "regeneration_provenance": regeneration_provenance,
    }
    metadata_path = input_root / "extraction_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    sampling = registration_sampling_configuration(metadata)
    return {
        "manifest": manifest,
        "images_dir": images_dir,
        "metadata": metadata_path,
        "sampling_configuration_sha256": _sha256_json(sampling),
        "fidelity": fidelity,
        "hash_provenance": hash_provenance,
        "regeneration_provenance": (
            _canonical_json(regeneration_provenance) if regeneration_provenance else ""
        ),
    }


def _collect_exact_input(
    source_dir: Path, destination: Path, category: str, sequence: str
) -> Dict[str, Any]:
    images = sorted(path for path in source_dir.iterdir() if path.is_file())
    if len(images) != EXPECTED_INPUT_COUNT:
        raise ColabImportError("saved COLMAP input directory does not contain exactly 60 files")
    actual_hashes = {path.name: sha256_file(path) for path in images}
    parent = source_dir.parent
    manifest_hashes = _manifest_hashes(parent / "frames_manifest.csv")
    integrity_hashes = _input_integrity_hashes(parent / "input_integrity.json")
    original_verified = actual_hashes in (manifest_hashes, integrity_hashes)
    provenance = (
        "original_notebook_verified"
        if original_verified
        else "import_time_hash_only"
    )
    return _write_collected_input(
        images,
        destination,
        fidelity="exact_labeled_frames",
        hash_provenance=provenance,
        sampling_strategy="saved_colmap_input_filename_order",
    )


def _co3d_sequence(
    loader: CO3DSequenceLoader, category: str, sequence_name: str
) -> CO3DSequence:
    sequence = next(
        (item for item in loader.load_category(category) if item.sequence_name == sequence_name),
        None,
    )
    if sequence is None:
        raise ColabImportError(f"CO3D sequence is missing: {category}/{sequence_name}")
    return sequence


def _regenerate_co3d_input(
    loader: CO3DSequenceLoader,
    category: str,
    sequence_name: str,
    destination: Path,
    strategy: str,
    evidence_path: Path,
) -> Tuple[Dict[str, Any], CO3DSequence]:
    sequence = _co3d_sequence(loader, category, sequence_name)
    count = len(sequence.frames)
    if count < EXPECTED_INPUT_COUNT:
        raise ColabImportError(f"CO3D sequence has only {count} frames; cannot regenerate 60")
    if strategy == "legacy_floor_linspace":
        indices = sorted(
            set(np.linspace(0, count - 1, EXPECTED_INPUT_COUNT).astype(int).tolist())
        )
    elif strategy == "production_round_linspace":
        indices = uniform_sample_indices(count, EXPECTED_INPUT_COUNT)
    else:
        raise ColabImportError(
            "original inputs are missing and no supported regeneration sampling strategy was declared"
        )
    if len(indices) != EXPECTED_INPUT_COUNT:
        raise ColabImportError("declared regeneration strategy did not produce exactly 60 unique frames")
    selected = [sequence.frames[index] for index in indices]
    category_dir = loader.root / category
    provenance = {
        "regeneration_schema_version": "co3d_byte_copy/1.0",
        "category": category,
        "sequence_name": sequence_name,
        "source_sequence_frame_count": count,
        "ordering": "CO3D frame_annotations frame_number ascending",
        "sampling_strategy": strategy,
        "sampling_declaration_evidence": str(evidence_path),
        "sampling_declaration_evidence_sha256": sha256_file(evidence_path),
        "selected_sequence_positions": indices,
        "selected_frame_numbers": [frame.frame_number for frame in selected],
        "transformation": "byte_for_byte_copy",
        "resize": None,
        "reencode": False,
        "frame_annotations_sha256": sha256_file(category_dir / "frame_annotations.jgz"),
        "sequence_annotations_sha256": sha256_file(category_dir / "sequence_annotations.jgz"),
    }
    collected = _write_collected_input(
        [frame.image_path for frame in selected],
        destination,
        fidelity="faithfully_regenerated",
        hash_provenance="faithfully_regenerated_from_co3d",
        sampling_strategy=strategy,
        regeneration_provenance=provenance,
        source_indices=indices,
        timestamps=[frame.timestamp_seconds for frame in selected],
    )
    return collected, sequence


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_colab_registration_labels(
    label_csvs: Sequence[Path],
    output_dir: Path,
    *,
    batch_roots: Sequence[Path] = (),
    co3d_root: Optional[Path] = None,
    group_map_csv: Optional[Path] = None,
    regeneration_sampling_strategy: Optional[str] = None,
    regeneration_evidence_path: Optional[Path] = None,
    batch_id_prefix: str = "colab-import",
) -> Dict[str, Any]:
    """Import compatible historical rows without mutating or rerunning their evidence."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    roots = [Path(path).expanduser().resolve() for path in batch_roots]
    group_map = _load_group_map(group_map_csv)
    loader = CO3DSequenceLoader(co3d_root) if co3d_root is not None else None
    evidence_path = (
        Path(regeneration_evidence_path).expanduser().resolve()
        if regeneration_evidence_path is not None
        else None
    )
    if evidence_path is not None and not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    source_csvs: List[Dict[str, Any]] = []
    schemas: List[Dict[str, Any]] = []
    seen_records = set()
    for raw_csv in label_csvs:
        csv_path = Path(raw_csv).expanduser().resolve()
        before_hash = sha256_file(csv_path)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        schema = detect_colab_schema(columns)
        schemas.append(
            {
                "path": str(csv_path),
                "schema": schema,
                "columns": columns,
                "existing_production_feature_columns": sorted(
                    FEATURE_COLUMN_NAMES & set(columns)
                ),
            }
        )
        csv_roots = [csv_path.parent, *roots]
        for line_number, row in enumerate(rows, start=2):
            category = _first(row, ["category"])
            sequence = _first(row, ["sequence_name", "sequence", "sequence_id"])
            attempt = _first(row, ["attempt"])
            batch_id = _first(row, ["batch_id"]) or f"{batch_id_prefix}:{csv_path.stem}"
            identity = {
                "source_csv": str(csv_path),
                "source_csv_sha256": before_hash,
                "line_number": line_number,
                "schema": schema,
                "category": category,
                "sequence_name": sequence,
                "attempt": attempt,
                "batch_id": batch_id,
            }
            try:
                if schema == "unknown":
                    raise ColabImportError("unrecognized Colab CSV schema")
                if not sequence:
                    raise ColabImportError("sequence identity is missing")
                record_id = _first(row, ["record_id"]) or ":".join(
                    part for part in (batch_id, category, sequence, attempt) if part
                )
                if record_id in seen_records:
                    raise ColabImportError("duplicate record identity across imported batches")
                seen_records.add(record_id)
                mapping = group_map.get((category, sequence)) or group_map.get(("", sequence))
                if mapping is None:
                    inline = {
                        "source_group": _first(row, ["source_group", "split_group_id"]),
                        "source_group_provenance": _first(row, ["source_group_provenance", "split_group_provenance"]),
                        "variant": _first(row, ["variant"]),
                        "evaluation_role": _first(row, ["evaluation_role"]),
                    }
                    if all(inline.values()):
                        mapping = inline
                source_result_path = _find_source_result(
                    row, csv_path, csv_roots, category, sequence
                )
                if source_result_path is None:
                    raise ColabImportError(
                        "no per-record reconstruction JSON was found; CSV-only metrics are not marked verified"
                    )
                source_result = _read_source_result(source_result_path)
                metrics = _validated_metrics(row, source_result)
                if mapping is None:
                    raise ColabImportError(
                        "source_group/variant/evaluation_role provenance is missing"
                    )
                collected_root = destination / "collected_inputs" / _safe_name(record_id)
                original_dir = _find_exact_input_directory(
                    row, csv_roots, category, sequence
                )
                co3d_sequence = None
                if original_dir is not None:
                    collected = _collect_exact_input(
                        original_dir, collected_root, category, sequence
                    )
                else:
                    if loader is None:
                        raise ColabImportError(
                            "exact labeled frames are missing and no CO3D root was supplied for regeneration"
                        )
                    if regeneration_sampling_strategy is None:
                        raise ColabImportError(
                            "exact labeled frames are missing and regeneration sampling strategy is undeclared"
                        )
                    if evidence_path is None:
                        raise ColabImportError(
                            "regeneration requires a notebook/config file proving the declared sampling strategy"
                        )
                    collected, co3d_sequence = _regenerate_co3d_input(
                        loader,
                        category,
                        sequence,
                        collected_root,
                        regeneration_sampling_strategy,
                        evidence_path,
                    )
                source_group = mapping["source_group"]
                source_group_provenance = mapping["source_group_provenance"]
                if co3d_sequence is not None and not source_group:
                    source_group = co3d_sequence.split_group_id
                    source_group_provenance = co3d_sequence.split_group_provenance
                if not all(
                    [
                        source_group,
                        source_group_provenance,
                        mapping["variant"],
                        mapping["evaluation_role"],
                    ]
                ):
                    raise ColabImportError("group-map provenance contains blank fields")
                normalized_root = destination / "normalized_label_artifacts"
                normalized_root.mkdir(exist_ok=True)
                normalized_path = normalized_root / f"{_safe_name(record_id)}.json"
                normalized = {
                    "schema_version": "imported_colmap_result/1.0",
                    "status": metrics["status"],
                    "failure_stage": metrics["failure_stage"],
                    "failure_reason": metrics["failure_reason"],
                    "commands": metrics["commands"],
                    "total_input_image_count": metrics["total"],
                    "objective_metrics": (
                        {
                            "registered_image_count": metrics["registered"],
                            "total_input_image_count": metrics["total"],
                            "registered_image_ratio": metrics["ratio"],
                            "reconstructed_point_count": metrics["points"],
                            "mean_reprojection_error_pixels": metrics["reprojection"],
                            "mean_track_length_images": metrics["track_length"],
                        }
                        if metrics["status"] == "succeeded"
                        else None
                    ),
                    "import_provenance": {
                        "source_csv": str(csv_path),
                        "source_csv_sha256": before_hash,
                        "source_csv_line": line_number,
                        "source_result": str(source_result_path),
                        "source_result_sha256": sha256_file(source_result_path),
                        "metric_verification": "csv_matches_per_record_result_json",
                        "colmap_rerun": False,
                    },
                }
                normalized_path.write_text(
                    json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                canonical = {name: "" for name in REQUIRED_LABEL_COLUMNS}
                canonical.update(
                    {
                        "label_schema_version": LABEL_SCHEMA_VERSION,
                        "batch_id": batch_id,
                        "record_id": record_id,
                        "sequence_id": sequence,
                        "source_group": source_group,
                        "source_group_provenance": source_group_provenance,
                        "variant": mapping["variant"],
                        "evaluation_role": mapping["evaluation_role"],
                        "reconstruction_status": metrics["status"],
                        "failure_stage": metrics["failure_stage"] or "",
                        "failure_verified": str(metrics["failure_verified"]).lower(),
                        "registered_image_count": str(metrics["registered"]),
                        "total_input_image_count": str(metrics["total"]),
                        "registration_percentage": str(metrics["ratio"] * 100.0),
                        "label_provenance_kind": "colmap_sparse_reconstruction",
                        "label_protocol_version": f"imported-{schema}/1.0",
                        "label_artifact": str(normalized_path),
                        "label_artifact_sha256": sha256_file(normalized_path),
                        "label_generated_at_utc": _first(
                            row,
                            ["label_generated_at_utc", "generated_at_utc", "completed_at_utc"],
                        )
                        or "original_label_generation_time_not_recorded",
                        "frames_manifest": str(collected["manifest"]),
                        "frames_manifest_sha256": sha256_file(collected["manifest"]),
                        "images_directory": str(collected["images_dir"]),
                        "extraction_metadata": str(collected["metadata"]),
                        "extraction_metadata_sha256": sha256_file(collected["metadata"]),
                        "sampling_protocol_id": f"imported-{schema}/1.0",
                        "sampling_configuration_sha256": collected[
                            "sampling_configuration_sha256"
                        ],
                        "input_fidelity": collected["fidelity"],
                        "input_hash_provenance": collected["hash_provenance"],
                        "regeneration_provenance": collected[
                            "regeneration_provenance"
                        ],
                    }
                )
                accepted.append(
                    {
                        **identity,
                        "record_id": record_id,
                        "source_group": source_group,
                        "source_group_provenance": source_group_provenance,
                        "variant": mapping["variant"],
                        "evaluation_role": mapping["evaluation_role"],
                        "input_fidelity": collected["fidelity"],
                        "input_hash_provenance": collected["hash_provenance"],
                        "target_percentage": metrics["ratio"] * 100.0,
                        "canonical": canonical,
                    }
                )
            except Exception as exc:
                reason = str(exc)
                status = _first(row, ["reconstruction_status", "status"]).lower()
                classification = (
                    "excluded_unavailable"
                    if status in UNAVAILABLE_STATUSES or "infrastructure/unavailable" in reason
                    else "rejected_incompatible"
                )
                rejected.append(
                    {
                        **identity,
                        "classification": classification,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        after_hash = sha256_file(csv_path)
        source_csvs.append(
            {
                "path": str(csv_path),
                "sha256_before": before_hash,
                "sha256_after": after_hash,
                "unchanged": before_hash == after_hash,
                "row_count": len(rows),
                "schema": schema,
            }
        )
    canonical_rows = [item["canonical"] for item in accepted]
    _write_csv(
        destination / "imported_registration_labels.csv",
        canonical_rows,
        sorted(REQUIRED_LABEL_COLUMNS),
    )
    audit_rows = [
        {
            "classification": "accepted",
            "record_id": item["record_id"],
            "category": item["category"],
            "sequence_name": item["sequence_name"],
            "source_group": item["source_group"],
            "variant": item["variant"],
            "input_fidelity": item["input_fidelity"],
            "input_hash_provenance": item["input_hash_provenance"],
            "reason": "",
            "source_csv": item["source_csv"],
            "line_number": item["line_number"],
        }
        for item in accepted
    ] + [
        {
            "classification": item["classification"],
            "record_id": "",
            "category": item["category"],
            "sequence_name": item["sequence_name"],
            "source_group": "",
            "variant": "",
            "input_fidelity": "",
            "input_hash_provenance": "",
            "reason": item["reason"],
            "source_csv": item["source_csv"],
            "line_number": item["line_number"],
        }
        for item in rejected
    ]
    audit_fields = [
        "classification",
        "record_id",
        "category",
        "sequence_name",
        "source_group",
        "variant",
        "input_fidelity",
        "input_hash_provenance",
        "reason",
        "source_csv",
        "line_number",
    ]
    _write_csv(destination / "import_audit.csv", audit_rows, audit_fields)
    report = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_trained": False,
        "colmap_run": False,
        "original_inputs_modified": False,
        "source_csvs": source_csvs,
        "detected_schemas": schemas,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": [
            {key: value for key, value in item.items() if key != "canonical"}
            for item in accepted
        ],
        "rejected": rejected,
        "canonical_labels_csv": str(destination / "imported_registration_labels.csv"),
    }
    (destination / "import_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report

