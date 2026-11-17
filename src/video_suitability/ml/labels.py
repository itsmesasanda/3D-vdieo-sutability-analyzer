"""Strict import and merge helpers for externally produced reconstruction labels."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .schema import (
    DatasetRecord,
    ReconstructionTarget,
    TargetProvenance,
)


REQUIRED_LABEL_COLUMNS = {
    "sequence_id",
    "target_score",
    "binary_suitable",
    "reconstruction_status",
    "target_provenance_kind",
    "target_protocol_version",
    "source_artifact",
    "source_sha256",
}


def _optional_float(value: Optional[str]) -> Optional[float]:
    return None if value is None or not value.strip() else float(value)


def _optional_int(value: Optional[str]) -> Optional[int]:
    return None if value is None or not value.strip() else int(value)


def _optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid optional boolean: {value!r}")


def load_reconstruction_targets_csv(
    path: Path, *, allow_synthetic_targets: bool = False
) -> Dict[str, ReconstructionTarget]:
    """Load labels without filling blanks or converting unavailable runs to zero."""

    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_LABEL_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label CSV is missing columns: {sorted(missing)}")
        targets: Dict[str, ReconstructionTarget] = {}
        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            if not sequence_id:
                raise ValueError(f"Missing sequence_id at label CSV line {line_number}")
            if sequence_id in targets:
                raise ValueError(f"Duplicate label sequence_id: {sequence_id}")
            synthetic = _optional_bool(row.get("is_synthetic")) or False
            provenance = TargetProvenance(
                kind=(row.get("target_provenance_kind") or "").strip(),
                protocol_version=(row.get("target_protocol_version") or "").strip(),
                source_artifact=(row.get("source_artifact") or "").strip() or None,
                source_sha256=(row.get("source_sha256") or "").strip() or None,
                generated_at_utc=(row.get("generated_at_utc") or "").strip() or None,
                is_synthetic=synthetic,
                notes=(row.get("provenance_notes") or "").strip() or None,
            )
            target = ReconstructionTarget(
                target_score=_optional_float(row.get("target_score")),
                binary_suitable=_optional_bool(row.get("binary_suitable")),
                reconstruction_status=(row.get("reconstruction_status") or "").strip(),
                registered_image_ratio=_optional_float(
                    row.get("registered_image_ratio")
                ),
                reconstructed_point_count=_optional_int(
                    row.get("reconstructed_point_count")
                ),
                mean_reprojection_error_pixels=_optional_float(
                    row.get("mean_reprojection_error_pixels")
                ),
                mean_track_length_images=_optional_float(
                    row.get("mean_track_length_images")
                ),
                human_completeness_score=_optional_float(
                    row.get("human_completeness_score")
                ),
                human_geometric_plausibility_score=_optional_float(
                    row.get("human_geometric_plausibility_score")
                ),
                human_actual_usability=_optional_bool(
                    row.get("human_actual_usability")
                ),
                provenance=provenance,
            )
            # Reuse the record-level safety rules without fabricating a sequence.
            probe = DatasetRecord(
                record_id=sequence_id,
                sequence_id=sequence_id,
                category="label_validation",
                sequence_name=sequence_id,
                split_group_id=sequence_id,
                split_group_provenance="label_validation_only",
                frame_count=1,
                sampled_frame_indices=[0],
                sampled_frame_paths=["not_loaded"],
                numeric_features={},
                target=target,
            )
            probe.validate(allow_synthetic_target=allow_synthetic_targets)
            targets[sequence_id] = target
    if not targets:
        raise ValueError(f"Label CSV contains no rows: {source}")
    return targets


def merge_targets(
    records: Sequence[DatasetRecord],
    targets: Dict[str, ReconstructionTarget],
    *,
    require_every_record: bool = False,
) -> List[DatasetRecord]:
    """Attach exact sequence-ID matches and reject labels for unknown sequences."""

    by_id = {record.sequence_id: record for record in records}
    unknown = sorted(set(targets) - set(by_id))
    if unknown:
        raise ValueError(f"Labels reference unknown sequence IDs: {unknown[:10]}")
    if require_every_record:
        unlabeled = sorted(set(by_id) - set(targets))
        if unlabeled:
            raise ValueError(f"Sequences are missing labels: {unlabeled[:10]}")
    for sequence_id, target in targets.items():
        by_id[sequence_id].target = target
    return list(records)


def load_heuristic_v21_baselines_csv(path: Path) -> Dict[str, float]:
    """Load optional comparison predictions; these values never become targets."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sequence_id",
            "heuristic_v21_score",
            "scoring_version",
            "scoring_revision",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Heuristic baseline CSV is missing columns: {sorted(missing)}")
        scores: Dict[str, float] = {}
        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            if not sequence_id or sequence_id in scores:
                raise ValueError(
                    f"Missing/duplicate sequence_id at heuristic CSV line {line_number}"
                )
            if (row.get("scoring_version") or "").strip() != "heuristic_v2":
                raise ValueError("Baseline scoring_version must be heuristic_v2")
            if (row.get("scoring_revision") or "").strip() != "2.1":
                raise ValueError("Baseline scoring_revision must be 2.1")
            score = float(row["heuristic_v21_score"])
            if not 0 <= score <= 100:
                raise ValueError("heuristic_v21_score must be between 0 and 100")
            scores[sequence_id] = score
    if not scores:
        raise ValueError(f"Heuristic baseline CSV contains no rows: {path}")
    return scores


def merge_heuristic_v21_baselines(
    records: Sequence[DatasetRecord], scores: Dict[str, float]
) -> List[DatasetRecord]:
    by_id = {record.sequence_id: record for record in records}
    unknown = sorted(set(scores) - set(by_id))
    if unknown:
        raise ValueError(f"Heuristic baselines reference unknown sequences: {unknown[:10]}")
    for sequence_id, score in scores.items():
        by_id[sequence_id].heuristic_v21_score = score
    return list(records)


def load_split_group_map_csv(path: Path) -> Dict[str, tuple[str, str]]:
    """Load a human-verified object/scene grouping override for leakage control."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "split_group_id", "split_group_provenance"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Group map CSV is missing columns: {sorted(missing)}")
        groups: Dict[str, tuple[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            sequence_id = (row.get("sequence_id") or "").strip()
            group_id = (row.get("split_group_id") or "").strip()
            provenance = (row.get("split_group_provenance") or "").strip()
            if not sequence_id or not group_id or not provenance:
                raise ValueError(f"Blank group-map field at line {line_number}")
            if sequence_id in groups:
                raise ValueError(f"Duplicate group-map sequence_id: {sequence_id}")
            groups[sequence_id] = (group_id, provenance)
    if not groups:
        raise ValueError(f"Group map CSV contains no rows: {path}")
    return groups


def apply_split_group_map(
    records: Sequence[DatasetRecord], groups: Dict[str, tuple[str, str]]
) -> List[DatasetRecord]:
    by_id = {record.sequence_id: record for record in records}
    unknown = sorted(set(groups) - set(by_id))
    if unknown:
        raise ValueError(f"Group map references unknown sequences: {unknown[:10]}")
    for sequence_id, (group_id, provenance) in groups.items():
        by_id[sequence_id].split_group_id = group_id
        by_id[sequence_id].split_group_provenance = provenance
    return list(records)

