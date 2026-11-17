#!/usr/bin/env python3
"""Build a versioned CO3D video-level manifest; never creates target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from video_suitability.datasets.co3d import CO3DSequenceLoader
from video_suitability.ml.features import numeric_features_from_co3d_sequence
from video_suitability.ml.labels import (
    apply_split_group_map,
    load_heuristic_v21_baselines_csv,
    load_reconstruction_targets_csv,
    load_split_group_map_csv,
    merge_heuristic_v21_baselines,
    merge_targets,
)
from video_suitability.ml.schema import DatasetRecord, write_jsonl


def build_records(
    co3d_root: Path,
    *,
    categories: Optional[List[str]],
    sample_count: int,
    metadata_only: bool,
) -> tuple[List[DatasetRecord], dict]:
    loader = CO3DSequenceLoader(co3d_root, verify_images=True)
    selected = categories or loader.categories()
    if not selected:
        raise ValueError("No valid CO3D categories were found")
    records: List[DatasetRecord] = []
    timestamp_feature_sequences = 0
    for sequence in loader.iter_sequences(selected):
        sampled = sequence.uniform_sample(sample_count)
        position_by_identity = {id(frame): index for index, frame in enumerate(sequence.frames)}
        positions = [position_by_identity[id(frame)] for frame in sampled]
        features = {}
        feature_diagnostics = {"extraction_status": "metadata_only"}
        if not metadata_only:
            features, feature_diagnostics = numeric_features_from_co3d_sequence(
                sequence, sample_count=sample_count
            )
            feature_diagnostics["extraction_status"] = "complete"
            timestamp_feature_sequences += int(
                feature_diagnostics["timestamp_dependent_features_available"]
            )
        relative_paths = [
            str(frame.image_path.relative_to(loader.root)) for frame in sampled
        ]
        records.append(
            DatasetRecord(
                record_id=sequence.sequence_id,
                sequence_id=sequence.sequence_id,
                category=sequence.category,
                sequence_name=sequence.sequence_name,
                object_id=sequence.object_id,
                scene_id=sequence.scene_id,
                split_group_id=sequence.split_group_id,
                split_group_provenance=sequence.split_group_provenance,
                frame_count=len(sequence.frames),
                sampled_frame_indices=positions,
                sampled_frame_paths=relative_paths,
                numeric_features=features,
                metadata={
                    "co3d": sequence.metadata(),
                    "feature_extraction": feature_diagnostics,
                    "sampled_frame_numbers": [frame.frame_number for frame in sampled],
                    "sampled_frame_paths_relative_to": "configured_co3d_root",
                },
            )
        )
    summary = {
        "co3d_root": str(loader.root),
        "categories": selected,
        "sequence_count": len(records),
        "sample_count_limit": sample_count,
        "numeric_features_extracted": not metadata_only,
        "timestamp_aware_feature_sequence_count": timestamp_feature_sequences,
        "target_scores_created": 0,
        "layout_audit": loader.audit_layout(selected),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate/sample paths but defer numerical CV feature extraction.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional externally created reconstruction/human target CSV.",
    )
    parser.add_argument(
        "--require-all-labels",
        action="store_true",
        help="Fail unless every sequence has an exact label row.",
    )
    parser.add_argument(
        "--heuristic-baselines-csv",
        type=Path,
        help="Optional frozen v2.1 predictions for baseline comparison only.",
    )
    parser.add_argument(
        "--group-map-csv",
        type=Path,
        help="Optional verified physical-object/scene grouping overrides.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    records, summary = build_records(
        args.co3d_root,
        categories=args.categories,
        sample_count=args.sample_count,
        metadata_only=args.metadata_only,
    )
    if args.group_map_csv:
        groups = load_split_group_map_csv(args.group_map_csv)
        apply_split_group_map(records, groups)
        summary["verified_group_override_count"] = len(groups)
    if args.labels_csv:
        targets = load_reconstruction_targets_csv(args.labels_csv)
        merge_targets(records, targets, require_every_record=args.require_all_labels)
        summary["label_row_count"] = len(targets)
        summary["records_with_target_score"] = sum(
            record.target is not None and record.target.target_score is not None
            for record in records
        )
    if args.heuristic_baselines_csv:
        baselines = load_heuristic_v21_baselines_csv(args.heuristic_baselines_csv)
        merge_heuristic_v21_baselines(records, baselines)
        summary["heuristic_v21_baseline_count"] = len(baselines)
    summary["sequence_fallback_group_count"] = sum(
        record.split_group_provenance == "sequence_fallback_object_identity_unavailable"
        for record in records
    )
    write_jsonl(records, args.out)
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dataset": str(args.out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

