import csv
from pathlib import Path
from typing import Optional

import pytest
import numpy as np

from video_suitability.ml.features import EmbeddingCache
from video_suitability.ml.labels import (
    apply_split_group_map,
    load_heuristic_v21_baselines_csv,
    load_reconstruction_targets_csv,
    load_split_group_map_csv,
)
from video_suitability.ml.metrics import evaluate_predictions
from video_suitability.ml.schema import (
    DatasetRecord,
    DatasetSchemaError,
    ReconstructionTarget,
    TargetProvenance,
    read_jsonl,
    write_jsonl,
)
from video_suitability.ml.splits import assert_no_group_leakage, grouped_split


def _record(index: int, group: Optional[str] = None) -> DatasetRecord:
    return DatasetRecord(
        record_id=f"record-{index}",
        sequence_id=f"sequence-{index}",
        category="synthetic_fixture",
        sequence_name=f"sequence-{index}",
        split_group_id=group or f"object-{index}",
        split_group_provenance="test_fixture",
        frame_count=10,
        sampled_frame_indices=[0, 9],
        sampled_frame_paths=["a.png", "b.png"],
        numeric_features={"feature.present": float(index), "feature.missing": None},
    )


def test_nullable_schema_round_trip_and_group_split(tmp_path):
    records = [_record(index, group=f"object-{index // 2}") for index in range(12)]
    path = tmp_path / "dataset.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded[0].numeric_features["feature.missing"] is None
    manifest = grouped_split(loaded, seed=7)
    assert_no_group_leakage(loaded, manifest)
    for group in {record.split_group_id for record in loaded}:
        assert len(
            {
                manifest.split_for(record.sequence_id)
                for record in loaded
                if record.split_group_id == group
            }
        ) == 1


def test_heuristic_and_unavailable_targets_cannot_be_training_labels():
    record = _record(1)
    record.target = ReconstructionTarget(
        target_score=80.0,
        binary_suitable=True,
        reconstruction_status="succeeded",
        provenance=TargetProvenance(
            kind="heuristic_v2.1",
            protocol_version="invalid",
            source_artifact=None,
            source_sha256=None,
            generated_at_utc=None,
        ),
    )
    with pytest.raises(DatasetSchemaError, match="Heuristic outputs"):
        record.validate(require_target=True)

    record.target.provenance.kind = "colmap_sparse_reconstruction"
    record.target.reconstruction_status = "unavailable_infrastructure"
    with pytest.raises(DatasetSchemaError, match="cannot carry"):
        record.validate(require_target=True)


def test_synthetic_target_requires_explicit_opt_in(tmp_path):
    record = _record(1)
    record.target = ReconstructionTarget(
        target_score=50.0,
        binary_suitable=False,
        reconstruction_status="synthetic_fixture",
        provenance=TargetProvenance(
            kind="synthetic_smoke",
            protocol_version="test/1",
            source_artifact="unit_test",
            source_sha256=None,
            generated_at_utc=None,
            is_synthetic=True,
        ),
    )
    path = tmp_path / "synthetic.jsonl"
    # A deliberately explicit fixture write is accepted only by opt-in reads.
    record.validate(allow_synthetic_target=True)
    write_jsonl([record], path, allow_synthetic_targets=True)
    with pytest.raises(DatasetSchemaError, match="Synthetic targets"):
        read_jsonl(path, require_targets=True)
    assert read_jsonl(
        path, require_targets=True, allow_synthetic_targets=True
    )[0].target.target_score == 50.0


def test_label_csv_preserves_blank_numerical_target(tmp_path):
    path = tmp_path / "labels.csv"
    headers = [
        "sequence_id",
        "target_score",
        "binary_suitable",
        "reconstruction_status",
        "target_provenance_kind",
        "target_protocol_version",
        "source_artifact",
        "source_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "sequence_id": "co3d:toy:a",
                "target_score": "",
                "binary_suitable": "",
                "reconstruction_status": "unavailable_infrastructure",
                "target_provenance_kind": "colmap_sparse_reconstruction",
                "target_protocol_version": "colmap_eval/1",
                "source_artifact": "run/status.json",
                "source_sha256": "",
            }
        )
    target = load_reconstruction_targets_csv(path)["co3d:toy:a"]
    assert target.target_score is None
    assert target.binary_suitable is None


def test_metrics_report_counts_ranking_and_missing_class_limitation():
    metrics = evaluate_predictions([10, 20, 30], [12, 18, 29], classification_threshold=25)
    assert metrics["sample_count"] == 3
    assert metrics["mae"] == pytest.approx(5 / 3)
    assert metrics["spearman_rank_correlation"] == pytest.approx(1.0)
    assert metrics["classification"]["both_classes_present"] is True
    one_class = evaluate_predictions([10, 20], [11, 19], classification_threshold=70)
    assert one_class["classification"]["balanced_accuracy"] is None
    assert one_class["classification"]["roc_auc"] is None


def test_embedding_cache_key_tracks_files_and_configuration(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame-bytes")
    cache = EmbeddingCache(tmp_path / "cache")
    key = cache.key("sequence-a", [frame], {"encoder": "synthetic-test"})
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.save(key, values)
    np.testing.assert_array_equal(cache.load(key), values)
    changed = cache.key("sequence-a", [frame], {"encoder": "different"})
    assert changed != key


def test_heuristic_baseline_import_requires_frozen_v21_identity(tmp_path):
    path = tmp_path / "baseline.csv"
    path.write_text(
        "sequence_id,heuristic_v21_score,scoring_version,scoring_revision\n"
        "co3d:toy:a,74.5,heuristic_v2,2.1\n",
        encoding="utf-8",
    )
    assert load_heuristic_v21_baselines_csv(path) == {"co3d:toy:a": 74.5}
    path.write_text(
        "sequence_id,heuristic_v21_score,scoring_version,scoring_revision\n"
        "co3d:toy:a,74.5,heuristic_v2,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision"):
        load_heuristic_v21_baselines_csv(path)


def test_verified_group_map_can_join_multiple_sequences(tmp_path):
    path = tmp_path / "groups.csv"
    path.write_text(
        "sequence_id,split_group_id,split_group_provenance\n"
        "sequence-0,object:toy:shared,manual_review_2026_11\n"
        "sequence-1,object:toy:shared,manual_review_2026_11\n",
        encoding="utf-8",
    )
    records = [_record(0), _record(1), _record(2), _record(3)]
    groups = load_split_group_map_csv(path)
    apply_split_group_map(records, groups)
    assert records[0].split_group_id == records[1].split_group_id
    manifest = grouped_split(records, seed=4)
    assert manifest.split_for("sequence-0") == manifest.split_for("sequence-1")

