import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_suitability.config import AnalysisConfig
from video_suitability.datasets.co3d import (
    CO3DSequenceLoader,
    CO3DValidationError,
)
from video_suitability.ml.features import numeric_features_from_co3d_sequence
from scripts.ml.build_dataset import build_records


def _write_jgz(path: Path, value) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle)


def _co3d_tree(tmp_path: Path, *, timestamps: bool = True) -> Path:
    root = tmp_path / "co3d"
    category = root / "toy"
    images = category / "sequence_a" / "images"
    images.mkdir(parents=True)
    annotations = []
    for frame_number in range(5):
        image = np.full((96, 128, 3), 40 + frame_number * 10, np.uint8)
        cv2.circle(image, (30 + frame_number * 5, 48), 18, (220, 220, 220), 2)
        path = images / f"frame{frame_number:06d}.jpg"
        assert cv2.imwrite(str(path), image)
        annotation = {
            "sequence_name": "sequence_a",
            "frame_number": frame_number,
            "image": {
                "path": f"toy/sequence_a/images/{path.name}",
                "size": [96, 128],
            },
            "mask": {"path": "unused"},
            "viewpoint": {"R": []},
        }
        if timestamps:
            annotation["frame_timestamp"] = frame_number * 0.1
        annotations.append(annotation)
    # Deliberately store out of order; the loader must restore frame order.
    _write_jgz(category / "frame_annotations.jgz", list(reversed(annotations)))
    _write_jgz(
        category / "sequence_annotations.jgz",
        [
            {
                "sequence_name": "sequence_a",
                "meta": {"object_id": "physical_toy_01", "scene_id": "scene_01"},
            }
        ],
    )
    return root


def test_co3d_loader_validates_orders_samples_and_groups(tmp_path):
    root = _co3d_tree(tmp_path)
    loader = CO3DSequenceLoader(root)
    assert loader.categories() == ["toy"]
    sequence = loader.load_category("toy")[0]
    assert [frame.frame_number for frame in sequence.frames] == [0, 1, 2, 3, 4]
    assert [frame.frame_number for frame in sequence.uniform_sample(3)] == [0, 2, 4]
    assert sequence.sequence_id == "co3d:toy:sequence_a"
    assert sequence.object_id == "physical_toy_01"
    assert sequence.split_group_id == "object:toy:physical_toy_01"
    assert sequence.split_group_provenance == "verified_annotation_object_id"
    assert sequence.metadata()["mask_frame_count"] == 5


def test_co3d_numeric_features_do_not_invent_missing_timestamps(tmp_path):
    sequence = CO3DSequenceLoader(_co3d_tree(tmp_path, timestamps=False)).load_category(
        "toy"
    )[0]
    features, diagnostics = numeric_features_from_co3d_sequence(
        sequence,
        sample_count=5,
        config=AnalysisConfig(max_samples=5, max_analysis_side=160),
    )
    assert features["quality.median_luminance"] is not None
    assert features["geometry.median_keypoints_per_frame"] is not None
    assert features["geometry.median_flow_speed_normalized_per_second"] is None
    assert diagnostics["annotated_timestamps_reliable"] is False


def test_co3d_numeric_features_extract_annotated_timing(tmp_path):
    sequence = CO3DSequenceLoader(_co3d_tree(tmp_path, timestamps=True)).load_category(
        "toy"
    )[0]
    features, diagnostics = numeric_features_from_co3d_sequence(
        sequence,
        sample_count=5,
        config=AnalysisConfig(max_samples=5, max_analysis_side=160),
    )
    assert features["video.fps"] == pytest.approx(10.0)
    assert features["video.duration_seconds"] == pytest.approx(0.4)
    assert diagnostics["timestamp_dependent_features_available"] is True


def test_co3d_loader_rejects_duplicate_frame_identity(tmp_path):
    root = _co3d_tree(tmp_path)
    path = root / "toy" / "frame_annotations.jgz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        annotations = json.load(handle)
    annotations.append(dict(annotations[0]))
    _write_jgz(path, annotations)
    with pytest.raises(CO3DValidationError, match="Duplicate frame identity"):
        CO3DSequenceLoader(root).load_category("toy")


def test_dataset_builder_writes_root_relative_sample_paths(tmp_path):
    root = _co3d_tree(tmp_path)
    records, summary = build_records(
        root, categories=["toy"], sample_count=3, metadata_only=True
    )
    assert summary["sequence_count"] == 1
    assert summary["target_scores_created"] == 0
    assert records[0].sampled_frame_indices == [0, 2, 4]
    assert all(not Path(path).is_absolute() for path in records[0].sampled_frame_paths)

