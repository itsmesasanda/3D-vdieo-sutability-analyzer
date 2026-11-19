import base64
from pathlib import Path

import app
from video_suitability.ml.experimental_ridge import (
    MODEL_SHA256,
    load_experimental_ridge,
    predict_experimental_ridge_registration,
)
from video_suitability.ml.historical_registration_features import (
    HISTORICAL_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_trusted_ridge_artifact_integrity_order_and_saved_parity():
    model, integrity = load_experimental_ridge()
    assert tuple(model.feature_names_in_) == HISTORICAL_FEATURE_NAMES
    assert model.n_features_in_ == 8
    assert integrity["status"] == "verified"
    assert integrity["model_sha256"] == MODEL_SHA256
    assert integrity["prediction_parity_case_count"] == 3
    assert integrity["training_rows"] == 647
    assert integrity["source_groups"] == 49
    assert integrity["independent_holdout_evaluated"] is False


def test_streamlit_secret_bytes_use_the_same_integrity_gated_loader():
    model_path = (
        ROOT
        / "evaluation/ml/historical_registration_experiment/historical_opencv8_v2"
        / "experimental_model.joblib"
    )
    payload = model_path.read_bytes()

    class FakeStreamlit:
        secrets = {
            "EXPERIMENTAL_RIDGE_ARTIFACT_B64": base64.b64encode(payload).decode("ascii")
        }

    decoded = app._streamlit_ridge_model_bytes(FakeStreamlit())
    assert decoded == payload
    model, integrity = load_experimental_ridge(model_bytes=decoded)
    assert model.__class__.__name__ == "Pipeline"
    assert integrity["model_sha256"] == MODEL_SHA256


def test_untrusted_model_bytes_are_rejected_before_deserialization(monkeypatch):
    import joblib

    called = {"value": False}

    def unsafe_load(_payload):
        called["value"] = True
        raise AssertionError("untrusted bytes must not be deserialized")

    monkeypatch.setattr(joblib, "load", unsafe_load)
    result = predict_experimental_ridge_registration(
        ROOT / "data/samples/good_orbit.mp4", model_bytes=b"not-the-model"
    )
    assert result["status"] == "unavailable"
    assert "integrity check failed" in result["reason"]
    assert called["value"] is False
    assert result["trained_model_used"] is False


def test_live_prediction_is_bounded_separate_and_unvalidated():
    result = predict_experimental_ridge_registration(
        ROOT / "data/samples/good_orbit.mp4"
    )
    assert result["status"] == "available"
    assert 0.0 <= result["predicted_registration_percentage"] <= 100.0
    assert tuple(result["features"]) == HISTORICAL_FEATURE_NAMES
    assert result["independently_validated"] is False
    assert result["affects_suitability_score"] is False
    assert result["affects_viewpoint_diversity"] is False
    assert result["colmap_run"] is False


def test_regression_output_is_clipped_without_changing_raw_value(monkeypatch):
    from video_suitability.ml import experimental_ridge

    class ExtremeModel:
        def predict(self, values):
            assert values.shape == (1, 8)
            return [126.5]

    monkeypatch.setattr(
        experimental_ridge,
        "load_experimental_ridge",
        lambda **_kwargs: (ExtremeModel(), {"status": "verified"}),
    )
    monkeypatch.setattr(
        experimental_ridge,
        "extract_historical_features_from_uploaded_video",
        lambda _path: (
            {name: float(index) for index, name in enumerate(HISTORICAL_FEATURE_NAMES)},
            {"adapter_status": "test"},
        ),
    )
    result = experimental_ridge.predict_experimental_ridge_registration(
        Path("unused-by-test.mp4")
    )
    assert result["raw_prediction_percentage"] == 126.5
    assert result["predicted_registration_percentage"] == 100.0
    assert result["prediction_was_clipped"] is True


def test_missing_artifact_fails_gracefully(tmp_path):
    result = predict_experimental_ridge_registration(
        ROOT / "data/samples/good_orbit.mp4", artifact_directory=tmp_path
    )
    assert result["status"] == "unavailable"
    assert "manifest is missing" in result["reason"]
    assert result["trained_model_used"] is False


def test_optional_inference_failure_cannot_change_main_report(
    synthetic_videos, monkeypatch
):
    video = synthetic_videos["front_only"]
    baseline = app._analyze_upload_payload(
        video.read_bytes(), video.name, include_viewpoint_diversity=False
    )

    def fail_optional(*_args, **_kwargs):
        raise RuntimeError("isolated Ridge failure")

    monkeypatch.setattr(app, "attach_experimental_ridge_prediction", fail_optional)
    failed = app._analyze_upload_payload(
        video.read_bytes(),
        video.name,
        include_viewpoint_diversity=False,
        include_experimental_ridge=True,
    )
    assert failed["report"]["overall_score"] == baseline["report"]["overall_score"]
    assert failed["report"]["subscores"] == baseline["report"]["subscores"]
    assert failed["report"]["viewpoint_diversity"] == baseline["report"][
        "viewpoint_diversity"
    ]
    assert failed["report"]["experimental_ridge_prediction"]["status"] == "unavailable"
    assert failed["report"]["registration_prediction"]["status"] == "unavailable"
