from pathlib import Path

import pytest

import app
from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.video import uniform_sample_indices


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_dependency_and_upload_configuration():
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    config = (REPOSITORY_ROOT / ".streamlit/config.toml").read_text(
        encoding="utf-8"
    )
    packages = (REPOSITORY_ROOT / "packages.txt").read_text(encoding="utf-8")

    assert "-e .[app-ml]" in requirements
    assert "scikit-learn==1.6.1" in (
        REPOSITORY_ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "opencv-python-headless==4.14.0.94" in (
        REPOSITORY_ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "pytest" not in requirements
    assert "maxUploadSize = 150" in config
    assert "headless = true" in config
    assert "ffmpeg" in packages.splitlines()
    assert 'base = "light"' in config
    assert 'textColor = "#18212F"' in config


def test_streamlit_styles_include_desktop_and_mobile_layouts():
    assert "@media (max-width: 780px)" in app.APP_CSS
    assert "@media (max-width: 460px)" in app.APP_CSS
    assert "--coral:#e85f57" in app.APP_CSS
    assert '[data-testid="stCheckbox"] label p' in app.APP_CSS
    assert '[data-testid="stFileUploaderDropzone"]' in app.APP_CSS


def test_streamlit_policy_keeps_random_forest_disabled_by_default(
    synthetic_videos, tmp_path
):
    video = synthetic_videos["front_only"]
    result = app._analyze_upload_payload(
        video.read_bytes(),
        video.name,
        registration_artifact_dir=tmp_path / "even-if-present",
        include_viewpoint_diversity=False,
    )

    prediction = result["report"]["registration_prediction"]
    assert prediction["status"] == "unavailable"
    assert prediction["trained_model_used"] is False
    assert prediction["colmap_run"] is False
    assert result["report"]["experimental_ridge_prediction"]["status"] == "not_requested"
    assert result["report"]["viewpoint_diversity"]["label"] == "Unknown"


def test_upload_limit_is_checked_before_temp_file_write():
    class OversizedBuffer:
        def __len__(self):
            return app.MAX_UPLOAD_BYTES + 1

    with pytest.raises(ValueError, match="150 MB deployment limit"):
        app._analyze_upload_payload(OversizedBuffer(), "large.mp4")


def test_object_selection_preview_matches_tracker_initialization_sample(
    synthetic_videos,
):
    class Upload:
        def __init__(self, path):
            self.name = path.name
            self._data = path.read_bytes()

        def getbuffer(self):
            return self._data

    class StreamlitState:
        session_state = {}

    import cv2

    video = synthetic_videos["quarter_orbit"]
    upload = Upload(video)
    st = StreamlitState()
    preview = app._representative_upload_frame(st, upload)
    metadata = st.session_state["object_selection_preview"]
    capture = cv2.VideoCapture(str(video))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    requested = uniform_sample_indices(frame_count, AnalysisConfig().max_samples)

    assert metadata["source_frame_index"] == requested[len(requested) // 2]
    assert metadata["preview_policy"] == app.OBJECT_SELECTION_PREVIEW_POLICY
    assert list(preview.shape[:2][::-1]) == [
        metadata["analysis_width"],
        metadata["analysis_height"],
    ]
    assert max(preview.shape[:2]) <= AnalysisConfig().max_analysis_side


def test_upload_report_records_tracker_input_and_runtime_provenance(
    synthetic_videos,
):
    video = synthetic_videos["quarter_orbit"]
    bbox = [0.20, 0.10, 0.60, 0.80]
    result = app._analyze_upload_payload(
        video.read_bytes(),
        video.name,
        object_bbox=bbox,
        include_viewpoint_diversity=True,
    )
    report = result["report"]
    tracking = report["viewpoint_diversity"]["object_tracking"]
    upload = report["streamlit_input_provenance"]

    assert upload["selected_object_bbox_normalized"] == bbox
    assert upload["selection_source_frame_index"] == tracking[
        "representative_source_frame_index"
    ]
    assert upload["analysis_frame_width"] == tracking["analysis_frame_width"]
    assert upload["analysis_frame_height"] == tracking["analysis_frame_height"]
    assert upload["tracker_version"] == tracking["tracker_version"]
    assert len(upload["video_sha256"]) == 64
    assert len(report["runtime_provenance"]["object_tracker_source_sha256"]) == 64
    assert len(tracking["sample_statuses"]) == tracking["sample_count"]
    assert tracking["tracker_settings"]["cycle_iou_minimum"] == 0.40


def test_experimental_viewpoint_failure_does_not_break_heuristic(
    monkeypatch, synthetic_videos
):
    from video_suitability import infer

    config = AnalysisConfig(max_samples=8, max_analysis_side=240)
    video = str(synthetic_videos["quarter_orbit"])
    without_viewpoint = analyze_video(
        video, config=config, include_viewpoint_diversity=False
    )

    def fail_viewpoint(*_args, **_kwargs):
        raise RuntimeError("isolated test failure")

    monkeypatch.setattr(infer, "analyze_viewpoint_diversity", fail_viewpoint)
    failed_viewpoint = analyze_video(
        video, config=config, include_viewpoint_diversity=True
    )

    assert failed_viewpoint["overall_score"] == without_viewpoint["overall_score"]
    assert failed_viewpoint["subscores"] == without_viewpoint["subscores"]
    assert failed_viewpoint["viewpoint_diversity"]["label"] == "Unknown"
    assert "isolated test failure" in failed_viewpoint["viewpoint_diversity"]["summary"]
