from pathlib import Path

import pytest

import app
from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_cloud_dependency_and_upload_configuration():
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    config = (REPOSITORY_ROOT / ".streamlit/config.toml").read_text(
        encoding="utf-8"
    )
    packages = (REPOSITORY_ROOT / "packages.txt").read_text(encoding="utf-8")

    assert "-e .[ui]" in requirements
    assert "pytest" not in requirements
    assert "maxUploadSize = 150" in config
    assert "headless = true" in config
    assert "ffmpeg" in packages.splitlines()


def test_streamlit_policy_keeps_live_ml_disabled(synthetic_videos, tmp_path):
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
    assert result["report"]["viewpoint_diversity"]["label"] == "Unknown"


def test_upload_limit_is_checked_before_temp_file_write():
    class OversizedBuffer:
        def __len__(self):
            return app.MAX_UPLOAD_BYTES + 1

    with pytest.raises(ValueError, match="150 MB deployment limit"):
        app._analyze_upload_payload(OversizedBuffer(), "large.mp4")


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
