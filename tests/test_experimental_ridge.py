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


def test_opt_in_live_prediction_is_bounded_separate_and_unreliable():
    result = predict_experimental_ridge_registration(
        ROOT / "data/samples/good_orbit.mp4"
    )
    assert result["status"] == "available"
    assert result["live_numeric_prediction_enabled"] is True
    assert 0 <= result["predicted_registration_percentage"] <= 100
    assert tuple(result["features"]) == HISTORICAL_FEATURE_NAMES
    assert result["independently_validated"] is False
    assert result["affects_suitability_score"] is False
    assert result["affects_viewpoint_diversity"] is False
    assert result["colmap_run"] is False


def test_regression_output_is_clipped_without_changing_raw_value():
    from video_suitability.ml import experimental_ridge

    class ExtremeModel:
        def predict(self, values):
            assert values.shape == (1, 8)
            return [126.5]

    result = {}
    experimental_ridge._attach_available_prediction(
        result,
        ExtremeModel(),
        {"status": "verified"},
        {name: float(index) for index, name in enumerate(HISTORICAL_FEATURE_NAMES)},
        {"adapter_status": "test"},
    )
    assert result["raw_prediction_percentage"] == 126.5
    assert result["predicted_registration_percentage"] == 100.0
    assert result["prediction_was_clipped"] is True
    assert "clipped regression ceiling" in result["display_ceiling_warning"]
    diagnostic = result["independent_live_video_diagnostic"]
    assert diagnostic["video_count"] == 4
    assert diagnostic["mae_percentage_points"] == 58.743399
    assert diagnostic["statistically_sufficient"] is False
    assert diagnostic["object_geometry_verified"] is False


def test_streamlit_ridge_panel_displays_unreliable_opt_in_prediction_and_clipping():
    class FakeStreamlit:
        def __init__(self):
            self.markdown_blocks = []

        def markdown(self, value, **_kwargs):
            self.markdown_blocks.append(value)

        def expander(self, *_args, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def dataframe(self, *_args, **_kwargs):
            return None

        def caption(self, value):
            self.markdown_blocks.append(value)

    st = FakeStreamlit()
    app._experimental_ridge_prediction(
        st,
        {
            "experimental_ridge_prediction": {
                "status": "available",
                "predicted_registration_percentage": 100.0,
                "raw_prediction_percentage": 130.0,
                "prediction_was_clipped": True,
            }
        },
    )

    rendered = " ".join(st.markdown_blocks)
    assert "Experimental ML Prediction" in rendered
    assert "Unreliable" in rendered
    assert "100.0%" in rendered
    assert "raw Ridge output (130.0%)" in rendered
    assert "clipped to 100%" in rendered
    assert "not reliable enough to determine" in rendered
    assert "Mean absolute error: 58.74 percentage points across four videos." in rendered


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
