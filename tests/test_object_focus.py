import pytest

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.object_focus import normalize_object_bbox


OBJECT_BOX = [0.25, 0.10, 0.50, 0.80]


@pytest.fixture(scope="module")
def object_focused_reports(synthetic_videos):
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


def test_normalized_object_bbox_validation():
    assert normalize_object_bbox(OBJECT_BOX) == tuple(OBJECT_BOX)
    assert normalize_object_bbox(
        {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.8}
    ) == (0.2, 0.1, 0.6, 0.8)
    for invalid in (
        [0.1, 0.2, 0.3],
        [-0.1, 0.1, 0.5, 0.5],
        [0.9, 0.1, 0.2, 0.5],
        [0.1, 0.1, 0.04, 0.5],
    ):
        with pytest.raises(ValueError):
            normalize_object_bbox(invalid)


def test_object_focused_scenarios_are_conservative(object_focused_reports):
    observed = {
        name: report["viewpoint_diversity"]["label"]
        for name, report in object_focused_reports.items()
    }
    assert observed == {
        "front_only": "Low",
        "quarter_orbit": "Moderate",
        # The object-only fixture does not meet the broader whole-frame High rule.
        "full_orbit": "Moderate",
        "pure_rotation": "Unknown",
        "animated_object": "Unknown",
        "blurry": "Unknown",
    }
    full = object_focused_reports["full_orbit"]["viewpoint_diversity"]
    assert full["object_tracking"]["tracked_sample_fraction"] >= 0.90
    assert full["method"]["analysis_mode"] == "object_focused"
    assert full["method"]["affects_overall_score"] is False


def test_object_box_never_changes_existing_score_paths(
    synthetic_videos, analysis_reports, object_focused_reports
):
    for name, object_report in object_focused_reports.items():
        whole_report = analysis_reports[name]
        assert object_report["overall_score"] == whole_report["overall_score"]
        assert object_report["subscores"] == whole_report["subscores"]
        assert object_report["baseline_v1"] == whole_report["baseline_v1"]


def test_object_and_background_evidence_are_reported_separately(
    object_focused_reports,
):
    result = object_focused_reports["quarter_orbit"]["viewpoint_diversity"]
    background = result["background_diagnostics"]
    assert background["pair_count"] > 0
    assert background["evaluated_pair_count"] > 0
    assert background["pairs"]
    assert all(
        "object_verified_matches" in pair
        and "background_verified_matches" in pair
        and "background_dominated" in pair
        and "moving_object_with_static_background" in pair
        and "evaluated" in pair
        for pair in background["pairs"]
    )


def test_untrackable_object_region_returns_unknown(synthetic_videos):
    report = analyze_video(
        str(synthetic_videos["low_texture"]),
        config=AnalysisConfig(max_samples=16, max_analysis_side=400),
        object_bbox=OBJECT_BOX,
    )
    result = report["viewpoint_diversity"]
    assert result["label"] == "Unknown"
    assert result["evidence_status"] == "insufficient"
    assert result["object_tracking"]["tracked_sample_fraction"] < 0.70
    reasons = result["decision_explanation"]["insufficient_evidence_reasons"]
    assert any("could not be tracked reliably" in reason for reason in reasons)


def test_decision_explanations_separate_failure_modes(object_focused_reports):
    front = object_focused_reports["front_only"]["viewpoint_diversity"]
    rotation = object_focused_reports["pure_rotation"]["viewpoint_diversity"]
    blurry = object_focused_reports["blurry"]["viewpoint_diversity"]
    animated = object_focused_reports["animated_object"]["viewpoint_diversity"]

    assert front["decision_explanation"]["classification_rule"] == (
        "stationary_or_repeated_views"
    )
    assert "repeats nearly the same view" in front["decision_explanation"][
        "primary_reason"
    ]
    assert "global homography" in rotation["decision_explanation"][
        "primary_reason"
    ]
    assert "blurry" in blurry["decision_explanation"]["primary_reason"]
    assert any(
        "could not be tracked reliably" in reason or "object animation" in reason
        for reason in animated["decision_explanation"][
            "insufficient_evidence_reasons"
        ]
    )


def test_background_dominance_forces_unknown(
    synthetic_videos, monkeypatch
):
    from video_suitability import object_focus

    def dominated_background(*_args, **_kwargs):
        return {
            "pair_count": 5,
            "evaluated_pair_count": 5,
            "unevaluated_pair_count": 0,
            "background_dominated_pair_fraction": 0.8,
            "median_object_verified_matches": 12.0,
            "median_background_verified_matches": 48.0,
            "pairs": [],
        }

    monkeypatch.setattr(
        object_focus, "_background_comparison", dominated_background
    )
    report = analyze_video(
        str(synthetic_videos["quarter_orbit"]),
        config=AnalysisConfig(max_samples=16, max_analysis_side=400),
        object_bbox=OBJECT_BOX,
    )
    result = report["viewpoint_diversity"]
    assert result["label"] == "Unknown"
    assert "Background-only correspondences dominate" in result[
        "decision_explanation"
    ]["primary_reason"]


def test_production_profile_reuses_features_and_excludes_rendering(
    object_focused_reports,
):
    profile = object_focused_reports["quarter_orbit"]["performance_profile"]
    assert set(profile["stages"]) == {
        "frame_decoding_and_sampling",
        "frame_quality",
        "orb_feature_extraction",
        "adjacent_geometry",
        "object_tracking_and_viewpoint",
        "scoring_and_problem_segments",
    }
    assert "diagnostic" not in " ".join(profile["stages"])
    assert profile["total_wall_seconds"] >= 0
    assert profile["total_cpu_seconds"] >= 0
