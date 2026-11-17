"""Versioned, nullable dataset records for reconstruction-grounded ML."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


DATASET_SCHEMA_VERSION = "video_suitability_ml_dataset/1.0"
FEATURE_SCHEMA_VERSION = "cv_numeric_features/2.1"
GENUINE_TARGET_KINDS = {
    "colmap_sparse_reconstruction",
    "human_reconstruction_review",
    "downstream_reconstruction_evaluation",
}
FORBIDDEN_TARGET_KINDS = {"heuristic_v1", "heuristic_v2", "heuristic_v2.1"}
UNAVAILABLE_RECONSTRUCTION_STATUSES = {
    "unavailable",
    "unavailable_infrastructure",
    "not_run",
    "missing",
}


class DatasetSchemaError(ValueError):
    """Raised when an ML dataset record is invalid or unsafe for training."""


@dataclass
class TargetProvenance:
    kind: str
    protocol_version: str
    source_artifact: Optional[str]
    source_sha256: Optional[str]
    generated_at_utc: Optional[str]
    is_synthetic: bool = False
    notes: Optional[str] = None


@dataclass
class ReconstructionTarget:
    target_score: Optional[float]
    binary_suitable: Optional[bool]
    reconstruction_status: str
    registered_image_ratio: Optional[float] = None
    reconstructed_point_count: Optional[int] = None
    mean_reprojection_error_pixels: Optional[float] = None
    mean_track_length_images: Optional[float] = None
    human_completeness_score: Optional[float] = None
    human_geometric_plausibility_score: Optional[float] = None
    human_actual_usability: Optional[bool] = None
    provenance: Optional[TargetProvenance] = None


@dataclass
class DatasetRecord:
    record_id: str
    sequence_id: str
    category: str
    sequence_name: str
    split_group_id: str
    split_group_provenance: str
    frame_count: int
    sampled_frame_indices: List[int]
    sampled_frame_paths: List[str]
    numeric_features: Dict[str, Optional[float]]
    object_id: Optional[str] = None
    scene_id: Optional[str] = None
    source_dataset: str = "co3d"
    dataset_schema_version: str = DATASET_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    embedding_cache_key: Optional[str] = None
    embedding_dimension: Optional[int] = None
    heuristic_v21_score: Optional[float] = None
    target: Optional[ReconstructionTarget] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        require_target: bool = False,
        allow_synthetic_target: bool = False,
    ) -> None:
        if self.dataset_schema_version != DATASET_SCHEMA_VERSION:
            raise DatasetSchemaError(
                f"Unsupported dataset schema: {self.dataset_schema_version}"
            )
        if not self.record_id or not self.sequence_id or not self.split_group_id:
            raise DatasetSchemaError("record_id, sequence_id and split_group_id are required")
        if self.frame_count <= 0:
            raise DatasetSchemaError("frame_count must be positive")
        if not self.sampled_frame_indices:
            raise DatasetSchemaError("at least one sampled frame is required")
        if len(self.sampled_frame_indices) != len(self.sampled_frame_paths):
            raise DatasetSchemaError("sampled frame indices and paths must have equal length")
        if self.sampled_frame_indices != sorted(set(self.sampled_frame_indices)):
            raise DatasetSchemaError("sampled frame indices must be unique and ordered")
        for name, value in self.numeric_features.items():
            if not name:
                raise DatasetSchemaError("numeric feature names cannot be empty")
            if value is not None and not math.isfinite(float(value)):
                raise DatasetSchemaError(f"feature {name} is non-finite")
        if self.embedding_cache_key is None and self.embedding_dimension is not None:
            raise DatasetSchemaError("embedding_dimension requires embedding_cache_key")
        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            raise DatasetSchemaError("embedding_dimension must be positive")
        if self.heuristic_v21_score is not None and not 0 <= self.heuristic_v21_score <= 100:
            raise DatasetSchemaError("heuristic_v21_score must be between 0 and 100")
        if require_target and self.target is None:
            raise DatasetSchemaError(f"Missing reconstruction target for {self.sequence_id}")
        if self.target is None:
            return
        provenance = self.target.provenance
        if provenance is None:
            raise DatasetSchemaError(f"Target provenance is required for {self.sequence_id}")
        if not provenance.kind or not provenance.protocol_version:
            raise DatasetSchemaError("Target provenance kind and protocol_version are required")
        if provenance.kind in FORBIDDEN_TARGET_KINDS:
            raise DatasetSchemaError("Heuristic outputs must never be used as ML targets")
        if provenance.is_synthetic and not allow_synthetic_target:
            raise DatasetSchemaError(
                "Synthetic targets are allowed only in explicitly enabled smoke tests"
            )
        if not provenance.is_synthetic and provenance.kind not in GENUINE_TARGET_KINDS:
            raise DatasetSchemaError(f"Unrecognized genuine target kind: {provenance.kind}")
        if require_target and self.target.target_score is None:
            raise DatasetSchemaError(f"target_score is missing for {self.sequence_id}")
        if not self.target.reconstruction_status:
            raise DatasetSchemaError("reconstruction_status is required when a target exists")
        if (
            self.target.target_score is not None
            and self.target.reconstruction_status.lower()
            in UNAVAILABLE_RECONSTRUCTION_STATUSES
        ):
            raise DatasetSchemaError(
                "Unavailable/not-run reconstruction cannot carry a numerical target score"
            )
        bounded_scores = {
            "target_score": self.target.target_score,
            "human_completeness_score": self.target.human_completeness_score,
            "human_geometric_plausibility_score": self.target.human_geometric_plausibility_score,
        }
        for name, value in bounded_scores.items():
            if value is not None and (
                not math.isfinite(float(value)) or not 0 <= float(value) <= 100
            ):
                raise DatasetSchemaError(f"{name} must be finite and between 0 and 100")
        if (
            self.target.registered_image_ratio is not None
            and (
                not math.isfinite(float(self.target.registered_image_ratio))
                or not 0 <= float(self.target.registered_image_ratio) <= 1
            )
        ):
            raise DatasetSchemaError("registered_image_ratio must be between 0 and 1")
        nonnegative_values = {
            "reconstructed_point_count": self.target.reconstructed_point_count,
            "mean_reprojection_error_pixels": self.target.mean_reprojection_error_pixels,
            "mean_track_length_images": self.target.mean_track_length_images,
        }
        for name, value in nonnegative_values.items():
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise DatasetSchemaError(f"{name} must be finite and nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "DatasetRecord":
        data = dict(value)
        target = data.get("target")
        if target is not None:
            target_data = dict(target)
            provenance = target_data.get("provenance")
            target_data["provenance"] = (
                TargetProvenance(**provenance) if provenance is not None else None
            )
            data["target"] = ReconstructionTarget(**target_data)
        return cls(**data)


def write_jsonl(
    records: Iterable[DatasetRecord],
    path: Path,
    *,
    allow_synthetic_targets: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate(allow_synthetic_target=allow_synthetic_targets)
            handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(
    path: Path,
    *,
    require_targets: bool = False,
    allow_synthetic_targets: bool = False,
) -> List[DatasetRecord]:
    records: List[DatasetRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = DatasetRecord.from_dict(json.loads(line))
                record.validate(
                    require_target=require_targets,
                    allow_synthetic_target=allow_synthetic_targets,
                )
            except (json.JSONDecodeError, TypeError, DatasetSchemaError) as exc:
                raise DatasetSchemaError(f"Invalid record at line {line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise DatasetSchemaError(f"Dataset contains no records: {path}")
    if len({record.record_id for record in records}) != len(records):
        raise DatasetSchemaError("Duplicate record_id values")
    if len({record.sequence_id for record in records}) != len(records):
        raise DatasetSchemaError("Each sequence_id must appear in exactly one video-level record")
    return records

