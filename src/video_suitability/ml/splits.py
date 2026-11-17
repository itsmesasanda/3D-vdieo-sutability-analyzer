"""Deterministic leakage-safe group splitting for video-level records."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .schema import DatasetRecord


SPLIT_SCHEMA_VERSION = "video_suitability_group_split/1.0"


@dataclass
class SplitManifest:
    schema_version: str
    seed: int
    group_key: str
    train_sequence_ids: List[str]
    validation_sequence_ids: List[str]
    test_sequence_ids: List[str]
    group_assignments: Dict[str, str]
    dataset_fingerprint: str

    def validate(self) -> None:
        sets = [
            set(self.train_sequence_ids),
            set(self.validation_sequence_ids),
            set(self.test_sequence_ids),
        ]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("A sequence appears in more than one split")
        if set(self.group_assignments.values()) - {"train", "validation", "test"}:
            raise ValueError("Unknown split assignment")

    def split_for(self, sequence_id: str) -> str:
        if sequence_id in self.train_sequence_ids:
            return "train"
        if sequence_id in self.validation_sequence_ids:
            return "validation"
        if sequence_id in self.test_sequence_ids:
            return "test"
        raise KeyError(sequence_id)

    def write(self, path: Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def read(cls, path: Path) -> "SplitManifest":
        manifest = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        manifest.validate()
        return manifest


def dataset_fingerprint(records: Sequence[DatasetRecord]) -> str:
    value = [
        record.to_dict() for record in sorted(records, key=lambda item: item.sequence_id)
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def grouped_split(
    records: Sequence[DatasetRecord],
    *,
    seed: int = 20261117,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SplitManifest:
    """Assign entire physical-object/scene groups to one split.

    Records without verified object/scene identity use their sequence ID as a
    conservative group. Frames never appear independently in this video-level split.
    """

    if len(records) < 3:
        raise ValueError("At least three sequence records are needed for three splits")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("split fractions must be between zero and one")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0:
        raise ValueError("train + validation fractions must be below one")
    groups: Dict[str, List[DatasetRecord]] = {}
    seen_sequences = set()
    for record in records:
        if record.sequence_id in seen_sequences:
            raise ValueError(f"Duplicate sequence record: {record.sequence_id}")
        seen_sequences.add(record.sequence_id)
        groups.setdefault(record.split_group_id, []).append(record)
    if len(groups) < 3:
        raise ValueError("At least three independent split groups are required")

    rng = random.Random(seed)
    ordered_groups = list(groups)
    rng.shuffle(ordered_groups)
    # Largest groups first after seeded tie randomization keeps ratios reasonably close.
    ordered_groups.sort(key=lambda group: len(groups[group]), reverse=True)
    total = len(records)
    targets = {
        "train": total * train_fraction,
        "validation": total * validation_fraction,
        "test": total * test_fraction,
    }
    counts = {name: 0 for name in targets}
    assignments: Dict[str, str] = {}
    split_order = ["train", "validation", "test"]
    for index, group in enumerate(ordered_groups):
        remaining = len(ordered_groups) - index
        empty = [name for name in split_order if counts[name] == 0]
        if empty and remaining <= len(empty):
            chosen = empty[0]
        else:
            chosen = min(
                split_order,
                key=lambda name: counts[name] / max(targets[name], 1e-9),
            )
        assignments[group] = chosen
        counts[chosen] += len(groups[group])

    sequences = {name: [] for name in split_order}
    for group, split in assignments.items():
        sequences[split].extend(record.sequence_id for record in groups[group])
    manifest = SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        seed=seed,
        group_key="DatasetRecord.split_group_id",
        train_sequence_ids=sorted(sequences["train"]),
        validation_sequence_ids=sorted(sequences["validation"]),
        test_sequence_ids=sorted(sequences["test"]),
        group_assignments=dict(sorted(assignments.items())),
        dataset_fingerprint=dataset_fingerprint(records),
    )
    manifest.validate()
    return manifest


def assert_no_group_leakage(
    records: Sequence[DatasetRecord], manifest: SplitManifest
) -> None:
    if manifest.schema_version != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema: {manifest.schema_version}")
    expected_sequences = {record.sequence_id for record in records}
    manifest_sequences = set(manifest.train_sequence_ids) | set(
        manifest.validation_sequence_ids
    ) | set(manifest.test_sequence_ids)
    if expected_sequences != manifest_sequences:
        missing = sorted(expected_sequences - manifest_sequences)
        extra = sorted(manifest_sequences - expected_sequences)
        raise ValueError(
            f"Split manifest/dataset sequence mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_fingerprint = dataset_fingerprint(records)
    if manifest.dataset_fingerprint != expected_fingerprint:
        raise ValueError("Split manifest dataset fingerprint does not match current records")
    group_splits: Dict[str, set] = {}
    for record in records:
        group_splits.setdefault(record.split_group_id, set()).add(
            manifest.split_for(record.sequence_id)
        )
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Split-group leakage detected: {leaked}")

