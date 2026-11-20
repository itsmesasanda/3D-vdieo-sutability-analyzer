import pytest
import numpy as np

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
        # Once foreground and background features are separated, these 2D proxy
        # fixtures expose only planar foreground warps. Background-layer parallax
        # must not be promoted to object-view evidence.
        "quarter_orbit": "Unknown",
        "full_orbit": "Unknown",
        "pure_rotation": "Unknown",
        "animated_object": "Unknown",
        "blurry": "Unknown",
    }
    full = object_focused_reports["full_orbit"]["viewpoint_diversity"]
    assert full["object_tracking"]["tracked_sample_fraction"] >= 0.90
    assert full["method"]["analysis_mode"] == "object_focused"
    assert full["method"]["affects_overall_score"] is False
    assert "global homography" in full["decision_explanation"]["primary_reason"]


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


def test_hybrid_tracker_handles_moderate_scale_and_partial_occlusion():
    import cv2

    from video_suitability.geometry import detect_orb_features
    from video_suitability.object_focus import track_object_regions

    height, width = 300, 400
    rng = np.random.default_rng(42)
    texture = rng.integers(25, 235, size=(130, 130, 3), dtype=np.uint8)
    cv2.rectangle(texture, (4, 4), (125, 125), (245, 245, 245), 4)
    frames = []
    for index, scale in enumerate(np.linspace(1.0, 1.18, 10)):
        frame = np.full((height, width, 3), (55, 65, 75), dtype=np.uint8)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (70, 80, 90), 1)
        resized = cv2.resize(texture, None, fx=float(scale), fy=float(scale))
        center_x = 200 + index * 2
        center_y = 150
        x0 = center_x - resized.shape[1] // 2
        y0 = center_y - resized.shape[0] // 2
        frame[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        if index in {4, 5}:
            cv2.rectangle(
                frame,
                (center_x - 8, y0),
                (center_x + 18, y0 + resized.shape[0]),
                (55, 65, 75),
                -1,
            )
        frames.append(frame)

    config = AnalysisConfig(max_samples=16, max_analysis_side=400)
    features = detect_orb_features(frames, config)
    tracking = track_object_regions(
        frames,
        features,
        [0.31, 0.27, 0.38, 0.48],
        representative_position=0,
        config=config,
    )
    assert tracking["tracker_version"] == "hybrid_klt_masked_orb_fb_v1"
    assert tracking["tracked_sample_fraction"] >= 0.70
    assert tracking["cycle_verified_step_fraction"] >= 0.70
    assert len(tracking["sample_statuses"]) == tracking["sample_count"]
    assert tracking["tracker_settings"]["maximum_consecutive_failures"] == 2
    assert any(
        "klt" in str(step.get("tracker_method"))
        for step in tracking["steps"]
    )
    assert any(
        "orb" in str(step.get("tracker_method"))
        for step in tracking["steps"]
    )


def test_object_feature_redetection_is_masked_and_background_stays_separate(
    object_focused_reports,
):
    result = object_focused_reports["front_only"]["viewpoint_diversity"]
    tracking = result["object_tracking"]
    assert tracking["foreground_separation"]["available"] is True
    assert tracking["feature_separation"]["method"] == (
        "local_masked_orb_redetection"
    )
    assert tracking["feature_separation"][
        "untracked_frames_have_object_features"
    ] is False
    background = result["background_diagnostics"]
    assert background["evaluated_pair_count"] >= 3
    assert all(
        pair["object_verified_matches"] is not None
        and pair["background_verified_matches"] is not None
        for pair in background["pairs"]
        if pair["evaluated"]
    )


def test_foreground_separation_fallback_cannot_claim_object_coverage(
    synthetic_videos, monkeypatch
):
    from video_suitability import object_focus

    def no_foreground(frame, box):
        x, y, width, height = object_focus._pixel_box(box, frame)
        return np.full(
            (max(1, int(round(height))), max(1, int(round(width)))),
            255,
            dtype=np.uint8,
        ), {
            "method": "bbox_only_fallback",
            "available": False,
            "foreground_fraction_within_box": 1.0,
            "reason": "test_fallback",
        }

    monkeypatch.setattr(
        object_focus, "_reference_foreground_template", no_foreground
    )
    report = analyze_video(
        str(synthetic_videos["quarter_orbit"]),
        config=AnalysisConfig(max_samples=16, max_analysis_side=400),
        object_bbox=OBJECT_BOX,
    )
    result = report["viewpoint_diversity"]
    assert result["label"] == "Unknown"
    assert "could not be separated reliably" in result[
        "decision_explanation"
    ]["primary_reason"]
