import json
from pathlib import Path

import numpy as np
import pytest

from video_suitability.ml.inference import (
    attach_optional_ml_prediction,
    temporal_prediction_from_embeddings,
)
from video_suitability.ml.schema import (
    DatasetRecord,
    ReconstructionTarget,
    TargetProvenance,
)
from video_suitability.ml.splits import grouped_split


def _synthetic_records(count: int = 12):
    records = []
    for index in range(count):
        target = float(8 + index * 7)
        records.append(
            DatasetRecord(
                record_id=f"synthetic-{index}",
                sequence_id=f"synthetic-{index}",
                category="explicit_synthetic_smoke",
                sequence_name=f"synthetic-{index}",
                split_group_id=f"synthetic-object-{index}",
                split_group_provenance="explicit_synthetic_smoke",
                frame_count=8,
                sampled_frame_indices=[0, 3, 7],
                sampled_frame_paths=["0.png", "3.png", "7.png"],
                numeric_features={
                    "feature.linear": float(index),
                    "feature.with_missing": None if index % 3 == 0 else index / 10,
                },
                heuristic_v21_score=min(100.0, target + 5.0),
                target=ReconstructionTarget(
                    target_score=target,
                    binary_suitable=target >= 70,
                    reconstruction_status="explicit_synthetic_fixture",
                    provenance=TargetProvenance(
                        kind="synthetic_smoke",
                        protocol_version="unit_test/1",
                        source_artifact="generated_in_test",
                        source_sha256=None,
                        generated_at_utc=None,
                        is_synthetic=True,
                    ),
                ),
            )
        )
    return records


def test_random_forest_synthetic_smoke_artifact_is_quarantined(tmp_path):
    pytest.importorskip("sklearn")
    from video_suitability.ml.tabular import train_random_forest

    records = _synthetic_records()
    split = grouped_split(records, seed=12)
    with pytest.raises(Exception, match="Synthetic"):
        train_random_forest(records, split, tmp_path / "rejected", n_estimators=5)
    result = train_random_forest(
        records,
        split,
        tmp_path / "accepted",
        allow_synthetic_targets=True,
        n_estimators=12,
        min_samples_leaf=1,
        random_state=12,
    )
    artifact = Path(result["artifact_dir"])
    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    assert result["training_scope"] == "synthetic_smoke_test_only"
    assert manifest["metadata"]["synthetic_metrics_are_not_real_accuracy"] is True
    assert result["metrics"]["test"]["random_forest"]["model"]["sample_count"] > 0

    heuristic_report = {"overall_score": 77.0}
    attached = attach_optional_ml_prediction(heuristic_report, artifact)
    assert attached["overall_score"] == 77.0
    assert attached["optional_ml_prediction"]["trained_model_used"] is False
    assert "Synthetic smoke-test" in attached["optional_ml_prediction"]["reason"]


def test_temporal_head_is_mask_aware_and_bounded():
    torch = pytest.importorskip("torch")
    from video_suitability.ml.temporal import TemporalLSTMRegressor

    model = TemporalLSTMRegressor(input_dim=8, hidden_dim=6, dropout=0.0)
    values = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    scores = model(values, mask)
    assert tuple(scores.shape) == (2,)
    assert torch.all(scores >= 0) and torch.all(scores <= 100)
    bad_mask = torch.tensor([[True, False, True, False], [True, True, False, False]])
    with pytest.raises(ValueError, match="contiguous prefixes"):
        model(values, bad_mask)


def test_temporal_training_synthetic_smoke_exports_quarantined_artifact(tmp_path):
    pytest.importorskip("torch")
    from video_suitability.ml.temporal import train_temporal_lstm

    records = _synthetic_records()
    split = grouped_split(records, seed=3)
    rng = np.random.default_rng(3)
    embeddings = {
        record.sequence_id: rng.normal(size=(3 + index % 3, 8)).astype(np.float32)
        for index, record in enumerate(records)
    }
    result = train_temporal_lstm(
        records,
        split,
        embeddings,
        tmp_path,
        allow_synthetic_targets=True,
        hidden_dim=6,
        batch_size=4,
        max_epochs=2,
        patience=2,
        random_state=3,
        device="cpu",
    )
    artifact = Path(result["artifact_dir"])
    manifest = json.loads((artifact / "artifact_manifest.json").read_text())
    assert result["training_scope"] == "synthetic_smoke_test_only"
    assert len(result["history"]) == 2
    assert manifest["metadata"]["synthetic_metrics_are_not_real_accuracy"] is True
    assert (artifact / "model.pt").is_file()
    assert (artifact / "checkpoints" / "epoch_001.pt").is_file()
    refused = temporal_prediction_from_embeddings(embeddings["synthetic-0"], artifact)
    assert refused["trained_model_used"] is False
    assert "Synthetic smoke-test" in refused["reason"]

