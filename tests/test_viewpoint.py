import json

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.presentation import viewpoint_diversity_view


def test_video_scenarios_match_expected_viewpoint_bands(analysis_reports):
    expected = {
        "front_only": "Low",
        "quarter_orbit": "Moderate",
        "full_orbit": "High",
        "pure_rotation": "Unknown",
        "blurry": "Unknown",
    }
    observed = {
        name: analysis_reports[name]["viewpoint_diversity"]["label"]
        for name in expected
    }
    assert observed == expected
    assert "global homography" in " ".join(
        analysis_reports["pure_rotation"]["viewpoint_diversity"]["warnings"]
    )
    assert "blurry" in " ".join(
        analysis_reports["blurry"]["viewpoint_diversity"]["warnings"]
    )


def test_multigap_evidence_detects_repeated_views_without_claiming_coverage(
    analysis_reports,
):
    stationary = analysis_reports["front_only"]["viewpoint_diversity"]
    full = analysis_reports["full_orbit"]["viewpoint_diversity"]

    assert stationary["indicators"][
        "temporally_separated_repeated_view_fraction"
    ] >= 0.8
    assert stationary["repeated_view_intervals"]
    assert full["indicators"]["endpoint_repeated_view_detected"] is True
    assert any("not proof of a complete orbit" in item for item in full["warnings"])
    serialized = json.dumps(full).lower()
    assert "surface captured" not in serialized
    assert "surface coverage" in serialized  # only an explicit non-claim warning


def test_low_texture_and_repetitive_symmetric_evidence_is_unknown(analysis_reports):
    low_texture = analysis_reports["low_texture"]["viewpoint_diversity"]
    repetitive = analysis_reports["repetitive_ambiguous"]["viewpoint_diversity"]

    assert low_texture["label"] == "Unknown"
    assert "too little trackable texture" in " ".join(low_texture["warnings"])
    assert repetitive["label"] == "Unknown"
    assert repetitive["indicators"][
        "median_within_frame_duplicate_descriptor_fraction"
    ] >= 0.10
    assert "repetitive or symmetric" in " ".join(repetitive["warnings"])


def test_intrinsics_enable_scale_free_relative_motion_indicators(synthetic_videos):
    report = analyze_video(
        str(synthetic_videos["full_orbit"]),
        config=AnalysisConfig(max_samples=16, max_analysis_side=320),
        camera_intrinsics={
            "fx": 420.0,
            "fy": 420.0,
            "cx": 200.0,
            "cy": 150.0,
            "image_width": 400.0,
            "image_height": 300.0,
        },
    )
    result = report["viewpoint_diversity"]
    assert result["method"]["intrinsics_used"] is True
    assert result["camera_intrinsics_at_analysis_resolution"]["fx"] == 336.0
    available = [
        item
        for item in result["relative_motion_indicators"]
        if item["essential_matrix_available"]
    ]
    assert available
    assert any(item["recover_pose_inliers"] >= 8 for item in available)
    assert all(item["translation_scale_available"] is False for item in available)
    assert result["indicators"][
        "median_rotation_compensated_parallax_degrees"
    ] is not None


def test_viewpoint_module_is_bounded_and_does_not_affect_scores(
    synthetic_videos, monkeypatch
):
    from video_suitability import infer

    config = AnalysisConfig(max_samples=8, max_analysis_side=240)
    original = analyze_video(str(synthetic_videos["quarter_orbit"]), config=config)

    def replacement(*_args, **_kwargs):
        return {
            "label": "Unknown",
            "evidence_status": "insufficient",
            "summary": "test replacement",
            "method": {"affects_overall_score": False},
            "indicators": {},
            "recommendations": [],
            "warnings": [],
            "repeated_view_intervals": [],
        }

    monkeypatch.setattr(infer, "analyze_viewpoint_diversity", replacement)
    replaced = infer.analyze_video(
        str(synthetic_videos["quarter_orbit"]), config=config
    )
    assert replaced["overall_score"] == original["overall_score"]
    assert replaced["subscores"] == original["subscores"]
    assert replaced["baseline_v1"] == original["baseline_v1"]

    evidence = original["viewpoint_diversity"]
    assert evidence["method"]["affects_overall_score"] is False
    assert evidence["method"]["colmap_run"] is False
    assert evidence["method"]["trained_model_used"] is False
    assert evidence["indicators"]["temporally_separated_pair_count"] <= 15


def test_viewpoint_ui_model_is_separate_and_actionable(analysis_reports):
    view = viewpoint_diversity_view(analysis_reports["quarter_orbit"])
    assert view["label"] == "Moderate"
    assert view["recommendations"]
    assert view["affects_overall_score"] is False
    assert 0 <= view["parallax_proxy_percent"] <= 100
    assert 0 <= view["substantial_change_percent"] <= 100

