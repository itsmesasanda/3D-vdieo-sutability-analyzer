"""Compatibility-gated preparation of COLMAP-registration training data.

This module extracts features and packages immutable evidence only.  It deliberately
contains no estimator fitting or model activation path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import sha256_file
from .registration_features import (
    FEATURE_EXTRACTOR_IMPLEMENTATION,
    FEATURE_EXTRACTOR_SCHEMA_VERSION,
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features_from_frame_manifest,
    load_feature_spec,
)


LABEL_SCHEMA_VERSION = "registration_colmap_labels/1.0"
TRAINING_RECORD_SCHEMA_VERSION = "registration_training_record/1.0"
TRAINING_BUNDLE_SCHEMA_VERSION = "registration_training_bundle/1.0"
GROUPED_CV_SCHEMA_VERSION = "registration_grouped_cv/1.0"
EXPECTED_COLMAP_INPUT_FRAME_COUNT = 60
DEVELOPMENT_ROLE = "development"
HOLDOUT_ROLE = "independent_holdout"
SUCCESS_STATUSES = {"succeeded", "measured_success"}
VERIFIED_FAILURE_STATUSES = {"reconstruction_failed", "verified_reconstruction_failure"}
UNUSABLE_STATUSES = {
    "infrastructure_unavailable",
    "unavailable_infrastructure",
    "unavailable",
    "not_run",
    "missing",
    "cancelled",
    "interrupted",
    "not_run_pilot_did_not_succeed",
}
VERIFIABLE_FAILURE_STAGES = {
    "mapper_sequential",
    "mapper_exhaustive",
    "sparse_mapping",
}
REQUIRED_LABEL_COLUMNS = {
    "label_schema_version",
    "batch_id",
    "record_id",
    "sequence_id",
    "source_group",
    "source_group_provenance",
    "variant",
    "evaluation_role",
    "reconstruction_status",
    "failure_stage",
    "failure_verified",
    "registered_image_count",
    "total_input_image_count",
    "registration_percentage",
    "label_provenance_kind",
    "label_protocol_version",
    "label_artifact",
    "label_artifact_sha256",
    "label_generated_at_utc",
    "frames_manifest",
    "frames_manifest_sha256",
    "images_directory",
    "extraction_metadata",
    "extraction_metadata_sha256",
    "sampling_protocol_id",
    "sampling_configuration_sha256",
    "input_fidelity",
    "input_hash_provenance",
    "regeneration_provenance",
}


class RegistrationTrainingCompatibilityError(ValueError):
    """Raised for an individual label/input compatibility failure."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_relative_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned:
        raise RegistrationTrainingCompatibilityError("record_id has no safe filename form")
    return cleaned


def _required_text(row: Mapping[str, str], key: str) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise RegistrationTrainingCompatibilityError(f"{key} is required")
    return value


def _parse_bool(value: str, key: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise RegistrationTrainingCompatibilityError(f"{key} must be true or false")


def _parse_int(row: Mapping[str, str], key: str) -> int:
    try:
        return int(_required_text(row, key))
    except ValueError as exc:
        raise RegistrationTrainingCompatibilityError(f"{key} must be an integer") from exc


def _parse_float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(_required_text(row, key))
    except ValueError as exc:
        raise RegistrationTrainingCompatibilityError(f"{key} must be numeric") from exc
    if not np.isfinite(value):
        raise RegistrationTrainingCompatibilityError(f"{key} must be finite")
    return value


def _resolve_input_path(csv_path: Path, row: Mapping[str, str], key: str) -> Path:
    value = Path(_required_text(row, key)).expanduser()
    return (csv_path.parent / value).resolve() if not value.is_absolute() else value.resolve()


def _verify_hashed_file(path: Path, expected_hash: str, description: str) -> None:
    if not path.is_file():
        raise RegistrationTrainingCompatibilityError(f"{description} is missing: {path}")
    if len(expected_hash) != 64 or sha256_file(path) != expected_hash.lower():
        raise RegistrationTrainingCompatibilityError(
            f"{description} checksum does not match the label CSV: {path}"
        )


def _load_json(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationTrainingCompatibilityError(
            f"{description} is not valid JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RegistrationTrainingCompatibilityError(f"{description} must be a JSON object")
    return value


def registration_sampling_configuration(
    metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the canonical label-input preprocessing configuration."""

    keys = (
        "schema_version",
        "source_codec_fourcc",
        "source_fps",
        "source_frame_count",
        "source_duration_seconds",
        "orientation_metadata_degrees",
        "opencv_orientation_auto_enabled",
        "requested_sample_fps",
        "sampling_strategy",
        "requested_frame_count",
        "extracted_frame_count",
        "failed_source_frame_indices",
        "first_timestamp_seconds",
        "last_timestamp_seconds",
        "temporal_coverage_fraction",
        "output_width",
        "output_height",
        "linear_resize_scale",
        "jpeg_quality",
    )
    return {key: metadata.get(key) for key in keys}


def _validate_label_and_provenance(
    row: Mapping[str, str], csv_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    status = _required_text(row, "reconstruction_status").lower()
    registered = _parse_int(row, "registered_image_count")
    total = _parse_int(row, "total_input_image_count")
    percentage = _parse_float(row, "registration_percentage")
    if total != EXPECTED_COLMAP_INPUT_FRAME_COUNT:
        raise RegistrationTrainingCompatibilityError(
            f"total_input_image_count must be {EXPECTED_COLMAP_INPUT_FRAME_COUNT}"
        )
    if not 0 <= registered <= total or not 0 <= percentage <= 100:
        raise RegistrationTrainingCompatibilityError(
            "registration counts/percentage are outside valid bounds"
        )
    measured_percentage = 100.0 * registered / total
    if not np.isclose(percentage, measured_percentage, atol=1e-6, rtol=1e-9):
        raise RegistrationTrainingCompatibilityError(
            "registration_percentage does not equal registered_image_count / "
            "total_input_image_count"
        )
    if _required_text(row, "label_provenance_kind") != "colmap_sparse_reconstruction":
        raise RegistrationTrainingCompatibilityError(
            "label_provenance_kind must be colmap_sparse_reconstruction"
        )
    label_path = _resolve_input_path(csv_path, row, "label_artifact")
    label_hash = _required_text(row, "label_artifact_sha256")
    _verify_hashed_file(label_path, label_hash, "COLMAP label artifact")
    label_artifact = _load_json(label_path, "COLMAP label artifact")
    artifact_status = str(label_artifact.get("status", "")).strip().lower()
    if status in SUCCESS_STATUSES:
        if artifact_status != "succeeded":
            raise RegistrationTrainingCompatibilityError(
                "successful label row does not reference a succeeded COLMAP result"
            )
        metrics = label_artifact.get("objective_metrics")
        if not isinstance(metrics, dict):
            raise RegistrationTrainingCompatibilityError(
                "successful COLMAP result has no measured objective_metrics"
            )
        if int(metrics.get("registered_image_count", -1)) != registered:
            raise RegistrationTrainingCompatibilityError(
                "registered_image_count differs from the COLMAP result"
            )
        if int(metrics.get("total_input_image_count", -1)) != total:
            raise RegistrationTrainingCompatibilityError(
                "total_input_image_count differs from the COLMAP result"
            )
        artifact_ratio = float(metrics.get("registered_image_ratio", -1.0))
        if not np.isclose(artifact_ratio * 100.0, percentage, atol=1e-6, rtol=1e-9):
            raise RegistrationTrainingCompatibilityError(
                "registration percentage differs from COLMAP objective_metrics"
            )
    elif status in VERIFIED_FAILURE_STATUSES:
        failure_stage = _required_text(row, "failure_stage").lower()
        if not _parse_bool(row.get("failure_verified", ""), "failure_verified"):
            raise RegistrationTrainingCompatibilityError(
                "reconstruction failure must be explicitly verified"
            )
        if artifact_status != "reconstruction_failed":
            raise RegistrationTrainingCompatibilityError(
                "verified failure does not reference a reconstruction_failed artifact"
            )
        if failure_stage not in VERIFIABLE_FAILURE_STAGES:
            raise RegistrationTrainingCompatibilityError(
                f"failure stage is not accepted as reconstruction evidence: {failure_stage}"
            )
        if str(label_artifact.get("failure_stage", "")).lower() != failure_stage:
            raise RegistrationTrainingCompatibilityError(
                "failure_stage differs from the COLMAP result"
            )
        if not label_artifact.get("commands"):
            raise RegistrationTrainingCompatibilityError(
                "verified failure has no recorded COLMAP command evidence"
            )
        if registered != 0 or percentage != 0.0:
            raise RegistrationTrainingCompatibilityError(
                "verified no-model reconstruction failure must explicitly record 0/60"
            )
        if int(label_artifact.get("total_input_image_count", -1)) != total:
            raise RegistrationTrainingCompatibilityError(
                "failure artifact input count differs from the label row"
            )
    else:
        raise RegistrationTrainingCompatibilityError(
            f"status is not a measured success or verified reconstruction failure: {status}"
        )
    provenance = {
        "kind": _required_text(row, "label_provenance_kind"),
        "protocol_version": _required_text(row, "label_protocol_version"),
        "artifact": str(label_path),
        "artifact_sha256": label_hash.lower(),
        "generated_at_utc": _required_text(row, "label_generated_at_utc"),
        "source_label_csv": str(csv_path),
        "source_label_csv_sha256": sha256_file(csv_path),
    }
    target = {
        "name": "colmap_registered_image_percentage",
        "units": "percent_0_to_100",
        "value": percentage,
        "registered_image_count": registered,
        "total_input_image_count": total,
        "reconstruction_status": status,
        "failure_stage": (row.get("failure_stage") or "").strip() or None,
        "failure_verified": _parse_bool(row.get("failure_verified", ""), "failure_verified"),
    }
    return target, provenance


def _extract_compatible_record(
    row: Mapping[str, str],
    csv_path: Path,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    if _required_text(row, "label_schema_version") != LABEL_SCHEMA_VERSION:
        raise RegistrationTrainingCompatibilityError("unsupported label_schema_version")
    role = _required_text(row, "evaluation_role").lower()
    if role not in {DEVELOPMENT_ROLE, HOLDOUT_ROLE}:
        raise RegistrationTrainingCompatibilityError(
            "evaluation_role must be development or independent_holdout"
        )
    fidelity = _required_text(row, "input_fidelity").lower()
    if fidelity not in {"exact_labeled_frames", "faithfully_regenerated"}:
        raise RegistrationTrainingCompatibilityError(
            "input_fidelity must be exact_labeled_frames or faithfully_regenerated"
        )
    regeneration = (row.get("regeneration_provenance") or "").strip()
    hash_provenance = _required_text(row, "input_hash_provenance")
    if fidelity == "faithfully_regenerated" and not regeneration:
        raise RegistrationTrainingCompatibilityError(
            "faithfully_regenerated inputs require regeneration_provenance"
        )
    target, label_provenance = _validate_label_and_provenance(row, csv_path)
    manifest_path = _resolve_input_path(csv_path, row, "frames_manifest")
    manifest_hash = _required_text(row, "frames_manifest_sha256")
    _verify_hashed_file(manifest_path, manifest_hash, "60-frame input manifest")
    images_dir = _resolve_input_path(csv_path, row, "images_directory")
    extraction_path = _resolve_input_path(csv_path, row, "extraction_metadata")
    extraction_hash = _required_text(row, "extraction_metadata_sha256")
    _verify_hashed_file(extraction_path, extraction_hash, "frame extraction metadata")
    extraction = _load_json(extraction_path, "frame extraction metadata")
    if int(extraction.get("requested_frame_count", -1)) != EXPECTED_COLMAP_INPUT_FRAME_COUNT:
        raise RegistrationTrainingCompatibilityError(
            "extraction metadata does not declare exactly 60 requested frames"
        )
    if int(extraction.get("extracted_frame_count", -1)) != EXPECTED_COLMAP_INPUT_FRAME_COUNT:
        raise RegistrationTrainingCompatibilityError(
            "extraction metadata does not declare exactly 60 extracted frames"
        )
    if extraction.get("failed_source_frame_indices"):
        raise RegistrationTrainingCompatibilityError(
            "extraction metadata records missing source frames"
        )
    integrity_unchanged = (
        extraction.get("source_unchanged_after_extraction") is True
        or extraction.get("input_collection_unchanged_during_import") is True
    )
    if not integrity_unchanged:
        raise RegistrationTrainingCompatibilityError(
            "input/source immutability was not verified during extraction or import"
        )
    integrity_before = extraction.get(
        "source_sha256_before", extraction.get("source_integrity_before_sha256")
    )
    integrity_after = extraction.get(
        "source_sha256_after", extraction.get("source_integrity_after_sha256")
    )
    if not integrity_before or integrity_before != integrity_after:
        raise RegistrationTrainingCompatibilityError(
            "input/source integrity hashes differ before and after collection"
        )
    if float(extraction.get("temporal_coverage_fraction", 0.0)) < 0.999:
        raise RegistrationTrainingCompatibilityError(
            "60-frame extraction does not cover the complete source timeline"
        )
    features, diagnostics = extract_registration_features_from_frame_manifest(
        manifest_path,
        spec,
        images_dir=images_dir,
        expected_input_frame_count=EXPECTED_COLMAP_INPUT_FRAME_COUNT,
    )
    sampling = registration_sampling_configuration(extraction)
    sampling_hash = _sha256_json(sampling)
    declared_sampling_hash = _required_text(
        row, "sampling_configuration_sha256"
    ).lower()
    if sampling_hash != declared_sampling_hash:
        raise RegistrationTrainingCompatibilityError(
            "sampling configuration hash differs from the label batch declaration"
        )
    return {
        "training_record_schema_version": TRAINING_RECORD_SCHEMA_VERSION,
        "record_id": _required_text(row, "record_id"),
        "sequence_id": _required_text(row, "sequence_id"),
        "source_group": _required_text(row, "source_group"),
        "source_group_provenance": _required_text(row, "source_group_provenance"),
        "variant": _required_text(row, "variant"),
        "batch_id": _required_text(row, "batch_id"),
        "evaluation_role": role,
        "target": target,
        "label_provenance": label_provenance,
        "input_provenance": {
            "input_fidelity": fidelity,
            "input_hash_provenance": hash_provenance,
            "regeneration_provenance": regeneration or None,
            "frames_manifest": str(manifest_path),
            "frames_manifest_sha256": manifest_hash.lower(),
            "images_directory": str(images_dir),
            "extraction_metadata": str(extraction_path),
            "extraction_metadata_sha256": extraction_hash.lower(),
            "source_integrity_sha256": integrity_before,
            "sampling_protocol_id": _required_text(row, "sampling_protocol_id"),
            "sampling_configuration": sampling,
            "sampling_configuration_sha256": sampling_hash,
        },
        "feature_contract": {
            "schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
            "implementation": FEATURE_EXTRACTOR_IMPLEMENTATION,
            "feature_names_in_order": list(REGISTRATION_FEATURE_NAMES),
            "input_adapter": "colmap_frame_manifest",
            "expected_input_frame_count": EXPECTED_COLMAP_INPUT_FRAME_COUNT,
            "sampled_manifest_positions": diagnostics["requested_frame_indices"],
            "all_input_hashes_verified": diagnostics["all_input_hashes_verified"],
        },
        "features": features,
    }


def _read_label_rows(paths: Sequence[Path]) -> Tuple[List[Tuple[Dict[str, str], Path, int]], List[Dict[str, Any]]]:
    rows: List[Tuple[Dict[str, str], Path, int]] = []
    sources: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_LABEL_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Label CSV {path} is missing columns: {sorted(missing)}")
            file_rows = [dict(row) for row in reader]
        rows.extend((row, path, index) for index, row in enumerate(file_rows, start=2))
        sources.append(
            {"path": str(path), "sha256": sha256_file(path), "row_count": len(file_rows)}
        )
    if not rows:
        raise ValueError("Merged label CSV inputs contain no rows")
    return rows, sources


def _grouped_cv_manifest(
    records: Sequence[Mapping[str, Any]], *, fold_count: int, seed: int
) -> Dict[str, Any]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    group_roles: Dict[str, set] = {}
    for record in records:
        group_roles.setdefault(str(record["source_group"]), set()).add(
            str(record["evaluation_role"])
        )
    leaked = {group: sorted(roles) for group, roles in group_roles.items() if len(roles) > 1}
    if leaked:
        raise ValueError(f"Physical source groups cross development/holdout: {leaked}")
    development_groups: Dict[str, List[str]] = {}
    holdout_groups: Dict[str, List[str]] = {}
    for record in records:
        destination = (
            holdout_groups
            if record["evaluation_role"] == HOLDOUT_ROLE
            else development_groups
        )
        destination.setdefault(str(record["source_group"]), []).append(
            str(record["record_id"])
        )
    if not holdout_groups:
        raise ValueError("At least one predeclared independent holdout group is required")
    if len(development_groups) < fold_count:
        raise ValueError(
            f"Need at least {fold_count} development source groups for {fold_count}-fold CV"
        )
    rng = random.Random(seed)
    groups = list(development_groups)
    rng.shuffle(groups)
    groups.sort(key=lambda group: len(development_groups[group]), reverse=True)
    fold_sizes = [0] * fold_count
    assignments: Dict[str, int] = {}
    for group in groups:
        fold = min(range(fold_count), key=lambda value: (fold_sizes[value], value))
        assignments[group] = fold
        fold_sizes[fold] += len(development_groups[group])
    fingerprint_value = [
        {
            "record_id": record["record_id"],
            "source_group": record["source_group"],
            "variant": record["variant"],
            "evaluation_role": record["evaluation_role"],
            "target": record["target"],
            "features": record["features"],
        }
        for record in sorted(records, key=lambda item: str(item["record_id"]))
    ]
    return {
        "schema_version": GROUPED_CV_SCHEMA_VERSION,
        "seed": seed,
        "group_key": "source_group",
        "variant_key": "variant",
        "fold_count": fold_count,
        "development_group_fold_assignments": dict(sorted(assignments.items())),
        "development_record_ids": sorted(
            record_id for values in development_groups.values() for record_id in values
        ),
        "independent_holdout_groups": sorted(holdout_groups),
        "independent_holdout_record_ids": sorted(
            record_id for values in holdout_groups.values() for record_id in values
        ),
        "independent_holdout_excluded_from_model_selection": True,
        "dataset_fingerprint": _sha256_json(fingerprint_value),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")


def _flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    flattened = {
        "training_record_schema_version": record["training_record_schema_version"],
        "record_id": record["record_id"],
        "sequence_id": record["sequence_id"],
        "source_group": record["source_group"],
        "source_group_provenance": record["source_group_provenance"],
        "variant": record["variant"],
        "batch_id": record["batch_id"],
        "evaluation_role": record["evaluation_role"],
        "registration_percentage": record["target"]["value"],
        "registered_image_count": record["target"]["registered_image_count"],
        "total_input_image_count": record["target"]["total_input_image_count"],
        "reconstruction_status": record["target"]["reconstruction_status"],
        "failure_stage": record["target"]["failure_stage"] or "",
        "failure_verified": record["target"]["failure_verified"],
        "label_provenance_json": _canonical_json(record["label_provenance"]),
        "input_provenance_json": _canonical_json(record["input_provenance"]),
        "feature_contract_json": _canonical_json(record["feature_contract"]),
    }
    flattened.update(record["features"])
    return flattened


def _write_records_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [_flatten_record(record) for record in records]
    fieldnames = [
        "training_record_schema_version",
        "record_id",
        "sequence_id",
        "source_group",
        "source_group_provenance",
        "variant",
        "batch_id",
        "evaluation_role",
        "registration_percentage",
        "registered_image_count",
        "total_input_image_count",
        "reconstruction_status",
        "failure_stage",
        "failure_verified",
        "label_provenance_json",
        "input_provenance_json",
        "feature_contract_json",
        *REGISTRATION_FEATURE_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_parity_case(
    record: Mapping[str, Any], destination: Path
) -> Tuple[Dict[str, Any], List[str]]:
    safe_id = _safe_relative_name(str(record["record_id"]))
    case_root = destination / "parity" / safe_id
    images_destination = case_root / "images"
    images_destination.mkdir(parents=True)
    input_provenance = record["input_provenance"]
    source_manifest = Path(input_provenance["frames_manifest"])
    source_images = Path(input_provenance["images_directory"])
    manifest_destination = case_root / "frames_manifest.csv"
    shutil.copy2(source_manifest, manifest_destination)
    with source_manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    files = [str(manifest_destination.relative_to(destination))]
    for row in rows:
        filename = str(row["filename"])
        source = (source_images / filename).resolve()
        target = images_destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != str(row["sha256"]).lower():
            raise RegistrationTrainingCompatibilityError(
                f"Parity copy checksum mismatch: {filename}"
            )
        files.append(str(target.relative_to(destination)))
    relative_manifest = str(manifest_destination.relative_to(destination))
    relative_images = str(images_destination.relative_to(destination))
    case = {
        "case_id": str(record["record_id"]),
        "input_type": "colmap_frame_manifest",
        "frames_manifest": relative_manifest,
        "frames_manifest_sha256": sha256_file(manifest_destination),
        "images_directory": relative_images,
        "expected_input_frame_count": EXPECTED_COLMAP_INPUT_FRAME_COUNT,
        "source_group": record["source_group"],
        "variant": record["variant"],
        "expected_features": record["features"],
    }
    return case, files


def prepare_registration_training_bundle(
    label_csvs: Sequence[Path],
    output_dir: Path,
    feature_spec_path: Path,
    *,
    bundle_version: str,
    fold_count: int = 5,
    seed: int = 20261118,
    parity_case_count: int = 1,
) -> Dict[str, Any]:
    """Extract compatible records and package evidence without fitting a model."""

    if not bundle_version.strip():
        raise ValueError("bundle_version is required")
    if parity_case_count < 1:
        raise ValueError("parity_case_count must be positive")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    spec = load_feature_spec(feature_spec_path)
    label_rows, label_sources = _read_label_rows(label_csvs)
    records: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    incompatibilities: List[Dict[str, Any]] = []
    seen_record_ids = set()
    for row, csv_path, line_number in label_rows:
        identity = {
            "record_id": (row.get("record_id") or "").strip(),
            "sequence_id": (row.get("sequence_id") or "").strip(),
            "source_group": (row.get("source_group") or "").strip(),
            "variant": (row.get("variant") or "").strip(),
            "batch_id": (row.get("batch_id") or "").strip(),
            "source_csv": str(csv_path),
            "line_number": line_number,
        }
        status = (row.get("reconstruction_status") or "").strip().lower()
        if status in UNUSABLE_STATUSES:
            exclusions.append(
                {
                    **identity,
                    "reconstruction_status": status,
                    "reason_code": "unavailable_or_infrastructure_status",
                    "reason": (
                        "No numerical target was created; infrastructure/not-run "
                        "statuses are not reconstruction failures."
                    ),
                }
            )
            continue
        record_id = identity["record_id"]
        if record_id in seen_record_ids:
            incompatibilities.append(
                {**identity, "reason_code": "duplicate_record_id", "reason": "record_id appears in more than one merged label row"}
            )
            continue
        seen_record_ids.add(record_id)
        try:
            records.append(_extract_compatible_record(row, csv_path, spec))
        except Exception as exc:
            incompatibilities.append(
                {
                    **identity,
                    "reconstruction_status": status,
                    "reason_code": "compatibility_validation_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    records.sort(key=lambda item: str(item["record_id"]))
    split_manifest: Optional[Dict[str, Any]] = None
    split_error: Optional[str] = None
    try:
        split_manifest = _grouped_cv_manifest(
            records, fold_count=fold_count, seed=seed
        )
    except ValueError as exc:
        split_error = str(exc)
    training_ready = bool(records) and not incompatibilities and split_manifest is not None
    feature_spec_destination = destination / "feature_extractor.json"
    feature_spec_destination.write_text(
        json.dumps(spec, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    records_jsonl = destination / "training_records.jsonl"
    _write_jsonl(records_jsonl, records)
    _write_records_csv(destination / "training_records.csv", records)
    development = [record for record in records if record["evaluation_role"] == DEVELOPMENT_ROLE]
    holdout = [record for record in records if record["evaluation_role"] == HOLDOUT_ROLE]
    _write_records_csv(destination / "development_records.csv", development)
    _write_records_csv(destination / "independent_holdout_records.csv", holdout)
    if split_manifest is not None:
        _write_json(destination / "grouped_cv_manifest.json", split_manifest)
    parity_cases: List[Dict[str, Any]] = []
    parity_files: List[str] = []
    if records:
        parity_candidates = development or records
        chosen_groups = set()
        for record in parity_candidates:
            if record["source_group"] in chosen_groups:
                continue
            case, files = _copy_parity_case(record, destination)
            parity_cases.append(case)
            parity_files.extend(files)
            chosen_groups.add(record["source_group"])
            if len(parity_cases) >= parity_case_count:
                break
    parity = {
        "schema_version": "registration_feature_parity/1.0",
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
        "cases": parity_cases,
    }
    (destination / "parity_examples.json").write_text(
        json.dumps(parity, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    compatibility = {
        "schema_version": TRAINING_BUNDLE_SCHEMA_VERSION,
        "bundle_version": bundle_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_ready": training_ready,
        "model_trained": False,
        "model_activated": False,
        "colmap_run_during_build": False,
        "feature_contract": {
            "schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
            "implementation": FEATURE_EXTRACTOR_IMPLEMENTATION,
            "feature_names_in_order": list(REGISTRATION_FEATURE_NAMES),
            "feature_spec_sha256": sha256_file(feature_spec_destination),
            "expected_colmap_input_frame_count": EXPECTED_COLMAP_INPUT_FRAME_COUNT,
        },
        "runtime_compatibility": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
        },
        "label_sources": label_sources,
        "input_row_count": len(label_rows),
        "compatible_record_count": len(records),
        "development_record_count": len(development),
        "independent_holdout_record_count": len(holdout),
        "excluded_record_count": len(exclusions),
        "incompatible_record_count": len(incompatibilities),
        "exclusions": exclusions,
        "incompatibilities": incompatibilities,
        "grouped_cv_error": split_error,
        "parity_case_count": len(parity_cases),
    }
    _write_json(destination / "compatibility_report.json", compatibility)
    files = [
        "feature_extractor.json",
        "training_records.jsonl",
        "training_records.csv",
        "development_records.csv",
        "independent_holdout_records.csv",
        "parity_examples.json",
        "compatibility_report.json",
        *parity_files,
    ]
    if split_manifest is not None:
        files.append("grouped_cv_manifest.json")
    bundle_manifest = {
        "bundle_schema_version": TRAINING_BUNDLE_SCHEMA_VERSION,
        "bundle_version": bundle_version,
        "created_at_utc": compatibility["created_at_utc"],
        "training_ready": training_ready,
        "contains_trained_model": False,
        "activates_production_inference": False,
        "target_name": "colmap_registered_image_percentage",
        "target_units": "percent_0_to_100",
        "training_scope": "genuine_colmap_registration_labels",
        "record_count": len(records),
        "files_sha256": {
            relative: sha256_file(destination / relative) for relative in sorted(set(files))
        },
    }
    _write_json(destination / "training_bundle_manifest.json", bundle_manifest)
    return compatibility

