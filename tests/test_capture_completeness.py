import copy

import pytest

import app
from video_suitability.capture_completeness import analyze_capture_completeness
from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.presentation import capture_completeness_view


OBJECT_BOX = [0.25, 0.10, 0.50, 0.80]


@pytest.fixture(scope="module")
def completeness_reports(synthetic_videos):
    config = AnalysisConfig(max_samples=16, max_analysis_side=400)
    return {
        name: analyze_video(
            str(synthetic_videos[name]),
            config=config,
            object_bbox=OBJECT_BOX,
            include_performance_profile=True,
        )
        for name in (
            "front_only",
            "quarter_orbit",
            "full_orbit",
            "pure_rotation",
            "animated_object",
            "blurry",
        )
    }


def test_capture_guidance_is_conservative_across_synthetic_cases(
    completeness_reports,
):
    observed = {
        name: report["capture_completeness"]["status"]
        for name, report in completeness_reports.items()
    }
    assert observed == {
        "front_only": "limited_view_evidence",
        "quarter_orbit": "not_verified",
        "full_orbit": "not_verified",
        "pure_rotation": "not_verified",
        "animated_object": "not_verified",
        "blurry": "not_verified",
    }
    for report in completeness_reports.values():
        capture = report["capture_completeness"]
        assert capture["elevation_variation"]["status"] == "not_measured"
        assert capture["method"]["affects_suitability_score"] is False
        assert capture["method"]["affects_ridge_prediction"] is False
        assert capture["method"]["affects_viewpoint_classification"] is False
        assert not any("360" in capture.get(key, "") for key in ("label", "summary"))


def test_capture_guidance_does_not_change_existing_results(
    analysis_reports, completeness_reports
):
    for name, report in completeness_reports.items():
        baseline = analysis_reports[name]
        assert report["overall_score"] == baseline["overall_score"]
        assert report["subscores"] == baseline["subscores"]
        assert report["baseline_v1"] == baseline["baseline_v1"]


def test_animated_object_remains_not_verified_after_tracking_improves(
    completeness_reports,
):
    capture = completeness_reports["animated_object"]["capture_completeness"]
    assert capture["label"] == "Capture coverage not verified"
    assert capture["view_variation"]["label"] == "Not verified"
    assert completeness_reports["animated_object"]["viewpoint_diversity"][
        "label"
    ] == "Unknown"


def test_unreliable_tracking_uses_required_not_verified_message():
    result = analyze_capture_completeness(
        {
            "label": "Unknown",
            "evidence_status": "insufficient",
            "method": {"analysis_mode": "object_focused"},
            "decision_explanation": {
                "primary_reason": "The selected object region could not be tracked reliably."
            },
            "object_tracking": {
                "tracked_sample_fraction": 0.50,
                "forward_reached_end": False,
                "backward_reached_start": False,
            },
        }
    )
    assert result["label"] == "Capture coverage not verified"
    assert result["tracking_percent"] == 50


def test_unreliable_foreground_mask_blocks_capture_claim_even_with_full_tracking():
    result = analyze_capture_completeness(
        {
            "label": "High",
            "evidence_status": "sufficient",
            "method": {"analysis_mode": "object_focused"},
            "decision_explanation": {"primary_reason": "Synthetic evidence."},
            "object_tracking": {
                "tracked_sample_fraction": 1.0,
                "forward_reached_end": True,
                "backward_reached_start": True,
                "foreground_separation": {
                    "reliable_for_object_evidence": False,
                    "mask_quality_status": "rejected",
                },
            },
        }
    )
    assert result["status"] == "not_verified"
    assert result["label"] == "Capture coverage not verified"
    assert "foreground mask" in result["summary"]


def test_capture_mode_can_flag_static_camera_object_motion_without_claiming_turntable():
    viewpoint = {
        "label": "Moderate",
        "evidence_status": "sufficient",
        "method": {"analysis_mode": "object_focused"},
        "decision_explanation": {"primary_reason": "Usable evidence."},
        "object_tracking": {
            "tracked_sample_fraction": 1.0,
            "forward_reached_end": True,
            "backward_reached_start": True,
        },
        "indicators": {
            "temporally_separated_substantial_change_fraction": 0.5,
            "temporally_separated_parallax_proxy_pair_fraction": 0.3,
        },
        "background_diagnostics": {
            "evaluated_pair_count": 4,
            "moving_object_with_static_background_pair_fraction": 0.5,
            "pairs": [],
        },
    }
    result = analyze_capture_completeness(viewpoint)
    assert result["capture_mode"]["status"] == "object_motion_pattern"
    assert "consistent with a turntable" in result["capture_mode"]["reason"]
    assert "animation cannot be excluded" in result["capture_mode"]["reason"]

    viewpoint["label"] = "Unknown"
    viewpoint["evidence_status"] = "insufficient"
    viewpoint["decision_explanation"]["primary_reason"] = (
        "Object motion cannot be separated from camera motion."
    )
    conservative = analyze_capture_completeness(viewpoint)
    assert conservative["status"] == "not_verified"
    assert conservative["capture_mode"]["status"] == "object_motion_pattern"


def test_shared_scene_motion_does_not_claim_verified_camera_orbit():
    viewpoint = {
        "label": "Moderate",
        "evidence_status": "sufficient",
        "method": {"analysis_mode": "object_focused"},
        "decision_explanation": {"primary_reason": "Usable image evidence."},
        "object_tracking": {
            "tracked_sample_fraction": 1.0,
            "forward_reached_end": True,
            "backward_reached_start": True,
        },
        "indicators": {
            "temporally_separated_substantial_change_fraction": 0.5,
            "temporally_separated_parallax_proxy_pair_fraction": 0.3,
        },
        "background_diagnostics": {
            "evaluated_pair_count": 3,
            "moving_object_with_static_background_pair_fraction": 0.0,
            "pairs": [
                {
                    "evaluated": True,
                    "object_motion_normalized": 0.02,
                    "background_motion_normalized": 0.015,
                    "object_verified_matches": 20,
                    "background_verified_matches": 30,
                }
                for _ in range(3)
            ],
        },
    }
    result = analyze_capture_completeness(viewpoint)
    mode = result["capture_mode"]
    assert mode["status"] == "shared_scene_motion_pattern"
    assert mode["label"] == "Shared scene-motion pattern"
    assert "one possible explanation" in mode["reason"]
    assert "not verified" in mode["reason"]


def test_whole_frame_result_never_claims_object_capture_coverage(analysis_reports):
    capture = analysis_reports["full_orbit"]["capture_completeness"]
    assert capture["status"] == "not_verified"
    assert "object region was not tracked" in capture["summary"]


def test_optional_capture_failure_is_isolated(monkeypatch, synthetic_videos):
    from video_suitability import infer

    config = AnalysisConfig(max_samples=8, max_analysis_side=240)
    baseline = analyze_video(
        str(synthetic_videos["quarter_orbit"]), config=config
    )

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("isolated completeness failure")

    monkeypatch.setattr(infer, "analyze_capture_completeness", fail_capture)
    failed = analyze_video(str(synthetic_videos["quarter_orbit"]), config=config)
    assert failed["overall_score"] == baseline["overall_score"]
    assert failed["subscores"] == baseline["subscores"]
    assert failed["viewpoint_diversity"] == baseline["viewpoint_diversity"]
    assert failed["capture_completeness"]["status"] == "not_verified"


def test_capture_presentation_and_streamlit_panel_handle_missing_evidence(
    completeness_reports,
):
    report = completeness_reports["front_only"]
    view = capture_completeness_view(report)
    assert view["label"] == "Limited viewpoint variation"
    assert view["elevation_label"] == "Not verified"

    class FakeStreamlit:
        def __init__(self):
            self.markup = ""
            self.messages = []

        def markdown(self, value, **_kwargs):
            self.markup += str(value)

        def expander(self, *_args, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, value):
            self.messages.append(str(value))

        def caption(self, value):
            self.messages.append(str(value))

        def warning(self, value):
            self.messages.append(str(value))

    fake = FakeStreamlit()
    app._capture_completeness(fake, report)
    assert "Object Capture Completeness" in fake.markup
    assert "Limited viewpoint variation" in fake.markup
    assert "Not verified" in fake.markup

    malformed = copy.deepcopy(report)
    malformed["capture_completeness"] = None
    missing = capture_completeness_view(malformed)
    assert missing["status"] == "not_verified"

