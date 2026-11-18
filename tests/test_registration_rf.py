import json
import shutil
from pathlib import Path

import numpy as np

from video_suitability.ml.artifacts import sha256_file, write_manifest
from video_suitability.ml.registration_features import (
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features,
    load_feature_spec,
)
from video_suitability.ml.registration_rf import (
    MODEL_TYPE,
    TARGET_NAME,
    artifact_cache_identity,
    attach_registration_prediction,
    predict_colmap_registration_percentage,
)
from video_suitability.presentation import registration_prediction_view


def _template_spec():
    root = Path(__file__).resolve().parents[1]
    return load_feature_spec(
        root
        / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
    )


def _artifact(tmp_path, parity_video, *, mismatch=False):
    directory = tmp_path / "artifact"
    parity_dir = directory / "parity"
    parity_dir.mkdir(parents=True)
    copied = parity_dir / "training_example.mp4"
    shutil.copyfile(parity_video, copied)
    spec = _template_spec()
    features, _ = extract_registration_features(copied, spec)
    if mismatch:
        features = dict(features)
        features[REGISTRATION_FEATURE_NAMES[0]] += 1.0
    (directory / "feature_extractor.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "parity_examples.json").write_text(
        json.dumps(
            {
                "schema_version": "registration_feature_parity/1.0",
                "absolute_tolerance": 1e-6,
                "relative_tolerance": 1e-6,
                "cases": [
                    {
                        "case_id": "training-example-001",
                        "video": "parity/training_example.mp4",
                        "video_sha256": sha256_file(copied),
                        "expected_features": features,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "model.joblib").write_bytes(b"unit-test-placeholder")
    write_manifest(
        directory,
        model_type=MODEL_TYPE,
        model_version="final-labels-test-v1",
        training_scope="genuine_colmap_registration_labels",
        files=[
            "model.joblib",
            "feature_extractor.json",
            "parity_examples.json",
            "parity/training_example.mp4",
        ],
        metadata={
            "target_name": TARGET_NAME,
            "target_units": "percent_0_to_100",
            "feature_count": 8,
            "record_count": 90,
        },
    )
    return directory


def test_eight_feature_extraction_is_ordered_and_deterministic(synthetic_videos):
    spec = _template_spec()
    first, diagnostics = extract_registration_features(
        synthetic_videos["quarter_orbit"], spec
    )
    second, _ = extract_registration_features(
        synthetic_videos["quarter_orbit"], spec
    )
    assert tuple(first) == REGISTRATION_FEATURE_NAMES
    assert len(first) == 8
    np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=0, atol=0)
    assert diagnostics["sample_count"] == 16
    assert diagnostics["pair_count"] == 15
    assert diagnostics["requested_frame_indices"][0] == 0
    assert diagnostics["requested_frame_indices"][-1] == 23


def test_missing_artifact_falls_back_without_prediction(synthetic_videos, tmp_path):
    result = predict_colmap_registration_percentage(
        synthetic_videos["front_only"], tmp_path / "missing"
    )
    assert result["status"] == "unavailable"
    assert result["trained_model_used"] is False
    assert result["affects_suitability_score"] is False
    assert result["affects_viewpoint_diversity"] is False
    assert result["colmap_run"] is False


def test_artifact_replacement_changes_streamlit_cache_identity(tmp_path):
    directory = tmp_path / "replaceable-artifact"
    missing = artifact_cache_identity(directory)
    directory.mkdir()
    manifest = directory / "artifact_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    installed = artifact_cache_identity(directory)
    manifest.write_text('{"model_version":"v2"}\n', encoding="utf-8")
    replaced = artifact_cache_identity(directory)
    assert missing != installed
    assert installed != replaced


def test_parity_verified_artifact_predicts_separately(
    synthetic_videos, tmp_path, monkeypatch
):
    from video_suitability.ml import registration_rf

    artifact = _artifact(tmp_path, synthetic_videos["quarter_orbit"])

    class FakeRandomForest:
        n_features_in_ = 8

        def predict(self, values):
            assert values.shape == (1, 8)
            return np.asarray([73.25])

    monkeypatch.setattr(
        registration_rf, "load_registration_model", lambda _path: FakeRandomForest()
    )
    result = predict_colmap_registration_percentage(
        synthetic_videos["front_only"], artifact
    )
    assert result["status"] == "available"
    assert result["predicted_registration_percentage"] == 73.25
    assert result["feature_parity"]["status"] == "verified"
    assert result["feature_parity"]["case_count"] == 1
    assert result["colmap_run"] is False

    heuristic = {
        "overall_score": 61.5,
        "subscores": {"geometric_consistency": 55.0},
        "viewpoint_diversity": {"label": "Unknown"},
    }
    attached = attach_registration_prediction(
        heuristic, synthetic_videos["front_only"], artifact
    )
    assert attached["overall_score"] == heuristic["overall_score"]
    assert attached["subscores"] == heuristic["subscores"]
    assert attached["viewpoint_diversity"] == heuristic["viewpoint_diversity"]
    assert attached["registration_prediction"][
        "predicted_registration_percentage"
    ] == 73.25


def test_parity_mismatch_refuses_prediction(
    synthetic_videos, tmp_path, monkeypatch
):
    from video_suitability.ml import registration_rf

    artifact = _artifact(
        tmp_path, synthetic_videos["quarter_orbit"], mismatch=True
    )
    model_loaded = {"value": False}

    def unexpected_load(_path):
        model_loaded["value"] = True
        raise AssertionError("model must not load before feature parity succeeds")

    monkeypatch.setattr(registration_rf, "load_registration_model", unexpected_load)
    result = predict_colmap_registration_percentage(
        synthetic_videos["front_only"], artifact
    )
    assert result["status"] == "incompatible"
    assert "parity failed" in result["reason"]
    assert result["trained_model_used"] is False
    assert model_loaded["value"] is False


def test_model_declared_feature_order_must_match(
    synthetic_videos, tmp_path, monkeypatch
):
    from video_suitability.ml import registration_rf

    artifact = _artifact(tmp_path, synthetic_videos["quarter_orbit"])

    class WrongFeatureOrderModel:
        n_features_in_ = 8
        feature_names_in_ = np.asarray(list(reversed(REGISTRATION_FEATURE_NAMES)))

        def predict(self, _values):
            raise AssertionError("prediction must not run with the wrong feature order")

    monkeypatch.setattr(
        registration_rf,
        "load_registration_model",
        lambda _path: WrongFeatureOrderModel(),
    )
    result = predict_colmap_registration_percentage(
        synthetic_videos["front_only"], artifact
    )
    assert result["status"] == "incompatible"
    assert "feature names/order" in result["reason"]


def test_registration_ui_view_stays_separate():
    view = registration_prediction_view(
        {
            "registration_prediction": {
                "status": "available",
                "predicted_registration_percentage": 82.5,
                "model_version": "final-v2",
                "feature_parity": {"status": "verified", "case_count": 2},
                "affects_suitability_score": False,
                "affects_viewpoint_diversity": False,
                "colmap_run": False,
            }
        }
    )
    assert view["percentage"] == 82.5
    assert view["parity_verified"] is True
    assert view["affects_suitability_score"] is False
    assert view["affects_viewpoint_diversity"] is False
    assert view["colmap_run"] is False


def test_streamlit_upload_path_gracefully_attaches_unavailable_model(
    synthetic_videos, tmp_path
):
    import app

    video = synthetic_videos["front_only"]
    result = app._analyze_upload_payload(
        video.read_bytes(),
        video.name,
        registration_artifact_dir=tmp_path / "missing-artifact",
    )
    prediction = result["report"]["registration_prediction"]
    assert prediction["status"] == "unavailable"
    assert prediction["trained_model_used"] is False
    assert result["report"]["viewpoint_diversity"]["label"] == "Low"
    assert result["upload_performance"]["diagnostic_rendering_included"] is False
