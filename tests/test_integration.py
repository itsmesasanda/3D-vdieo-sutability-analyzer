import json
import os
import subprocess
import sys
from pathlib import Path

from video_suitability.infer import main


REQUIRED_TOP_LEVEL_KEYS = {
    "analysis_method",
    "overall_score",
    "confidence_or_uncertainty",
    "subscores",
    "raw_metrics",
    "problem_segments",
    "recommendations",
    "warnings",
    "viewpoint_diversity",
    "capture_completeness",
    "result_interpretation",
    "diagnostic_confidence",
}


def _problem_types(report):
    return {segment["problem_type"] for segment in report["problem_segments"]}


def test_reports_have_required_schema_and_bounded_scores(analysis_reports):
    for report in analysis_reports.values():
        assert REQUIRED_TOP_LEVEL_KEYS <= set(report)
        assert 0 <= report["overall_score"] <= 100
        assert report["analysis_method"]["trained_model_used"] is False
        assert report["analysis_method"]["reconstruction_run"] is False
        assert report["scoring_version"] == "heuristic_v2"
        assert report["scoring_revision"] == "2.1"
        assert report["baseline_v1"]["scoring_version"] == "heuristic_v1"
        assert report["result_interpretation"]["geometric_readiness"] == "unverified"
        assert report["diagnostic_confidence"]["affects_overall_score"] is False
        assert report["score_comparison"]["baseline_v1_score"] == report[
            "baseline_v1"
        ]["overall_score"]
        assert all(0 <= value <= 100 for value in report["subscores"].values())
        json.dumps(report, allow_nan=False)


def test_known_synthetic_failures_reduce_score_and_are_timestamped(analysis_reports):
    good = analysis_reports["smooth_orbit"]
    stationary = analysis_reports["stationary"]
    blurry = analysis_reports["blurry"]
    dark = analysis_reports["dark"]

    assert good["overall_score"] > stationary["overall_score"]
    assert good["overall_score"] > blurry["overall_score"]
    assert good["overall_score"] > dark["overall_score"]
    assert "little_viewpoint_change" in _problem_types(stationary)
    assert "blur" in _problem_types(blurry)
    assert "underexposure" in _problem_types(dark)
    for report in (stationary, blurry, dark):
        assert all(
            segment["end_time_seconds"] >= segment["start_time_seconds"]
            for segment in report["problem_segments"]
        )


def test_diagnostic_confidence_distinguishes_proxy_and_unavailable_measurement(
    analysis_reports,
):
    blurry = analysis_reports["blurry"]
    blur_segments = [
        segment
        for segment in blurry["problem_segments"]
        if segment["problem_type"] == "blur"
    ]
    assert blur_segments
    assert {segment["diagnostic_status"] for segment in blur_segments} == {
        "suspected"
    }
    assert any(
        recommendation.startswith("Inspect visible object edges first")
        for recommendation in blurry["recommendations"]
    )

    from video_suitability.presentation import recommendation_cards, timeline_rows

    report = {
        "raw_metrics": {"video": {"duration_seconds": 2.0}},
        "problem_segments": [
            {
                "problem_type": "motion_sampling_gap_too_wide",
                "source": "orb_ransac_frame_pairs",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "start_frame_index": 0,
                "end_frame_index": 10,
                "sample_pair_count": 1,
                "severity": "high",
                "diagnostic_status": "not_measurable",
                "measurement_scope": "sampled_frame_pair",
                "confidence_reason": "Sampling gap is too wide.",
            }
        ],
    }
    rows = timeline_rows(report)
    assert rows[0]["diagnostic_status"] == "not_measurable"
    assert rows[0]["label"] == "High-frequency shake not measurable"
    assert recommendation_cards(rows) == []


def test_high_score_is_not_interpreted_as_verified_geometry(analysis_reports):
    report = analysis_reports["smooth_orbit"]
    assert report["overall_score"] >= 70
    interpretation = report["result_interpretation"]
    assert interpretation["technical_capture_level"] == "strong"
    assert "geometric readiness unverified" in interpretation["headline"]
    assert interpretation["affects_overall_score"] is False


def test_exposure_warning_states_whole_frame_scope(analysis_reports):
    dark = analysis_reports["dark"]
    assert any(
        "Exposure flags are whole-frame measurements" in warning
        for warning in dark["warnings"]
    )
    underexposure = [
        segment
        for segment in dark["problem_segments"]
        if segment["problem_type"] == "underexposure"
    ]
    assert underexposure
    assert all(
        segment["measurement_scope"] == "whole_frame"
        and segment["diagnostic_status"] == "detected"
        for segment in underexposure
    )


def test_repeated_views_are_explained_without_claiming_coverage(analysis_reports):
    stationary = analysis_reports["stationary"]
    intervals = stationary["viewpoint_diversity"].get("repeated_view_intervals", [])
    if intervals:
        assert any(
            "not evidence of new object viewpoints" in warning
            for warning in stationary["warnings"]
        )


def _triggered_gates(report):
    return {
        gate["gate_id"]
        for gate in report["critical_failure_gate_evaluations"]
        if gate["triggered"]
    }


def test_v2_exposes_controlled_motion_failures(analysis_reports):
    smooth = analysis_reports["smooth_orbit"]
    stationary = analysis_reports["stationary"]
    pure_rotation = analysis_reports["pure_rotation"]
    shaky = analysis_reports["shaky_orbit"]
    fast = analysis_reports["fast_motion"]
    insufficient = analysis_reports["insufficient_coverage"]

    assert stationary["overall_score"] <= 35
    assert "stationary_view" in _triggered_gates(stationary)
    assert pure_rotation["overall_score"] <= 44
    assert "rotation_or_planar_coverage_uncertain" in _triggered_gates(
        pure_rotation
    )
    assert shaky["overall_score"] < smooth["overall_score"]
    assert "irregular_camera_motion" in _problem_types(shaky)
    assert fast["overall_score"] < smooth["overall_score"]
    assert "excessive_camera_speed" in _problem_types(fast)
    assert insufficient["overall_score"] <= 25
    assert "insufficient_temporal_evidence" in _triggered_gates(insufficient)


def test_low_resolution_is_diagnostic_not_automatic_failure(analysis_reports):
    low_resolution = analysis_reports["low_resolution"]
    diagnostics = low_resolution["raw_metrics"]["source_media_diagnostics"]
    assert diagnostics["source_width"] == 160
    assert diagnostics["source_height"] == 120
    assert diagnostics["resolution_alone_triggers_failure"] is False
    assert not any(
        gate["triggered"] and "resolution" in gate["gate_id"]
        for gate in low_resolution["critical_failure_gate_evaluations"]
    )
    assert "rotation_or_planar_coverage_uncertain" not in _triggered_gates(
        low_resolution
    )
    assert low_resolution["overall_score"] >= 45


def test_persistent_severe_blur_cannot_be_hidden_by_coarse_geometry(analysis_reports):
    blurry = analysis_reports["blurry"]
    smooth = analysis_reports["smooth_orbit"]
    assert "persistent_severe_blur" in _triggered_gates(blurry)
    assert blurry["overall_score"] <= 55
    assert blurry["overall_score"] < blurry["baseline_v1"]["overall_score"]
    assert "persistent_severe_blur" not in _triggered_gates(smooth)
    assert smooth["overall_score"] > blurry["overall_score"]


def test_default_dense_sampling_does_not_misclassify_multilayer_parallax(
    synthetic_videos,
):
    from video_suitability.infer import analyze_video

    smooth = analyze_video(str(synthetic_videos["smooth_orbit"]))
    pure_rotation = analyze_video(str(synthetic_videos["pure_rotation"]))
    assert "rotation_or_planar_coverage_uncertain" not in _triggered_gates(smooth)
    assert smooth["overall_score"] > 70
    assert pure_rotation["overall_score"] <= 44


def test_documented_cli_writes_json(synthetic_videos, tmp_path):
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_suitability.infer",
            "--video",
            str(synthetic_videos["orbit_good"]),
            "--out",
            str(output),
            "--max-samples",
            "12",
            "--max-side",
            "320",
        ],
        cwd=repository,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert REQUIRED_TOP_LEVEL_KEYS <= set(report)
    assert "heuristic score" in completed.stderr


def test_cli_refuses_to_overwrite_input_video(synthetic_videos, capsys):
    video = synthetic_videos["stationary"]
    original_size = video.stat().st_size
    result = main(["--video", str(video), "--out", str(video)])
    assert result == 2
    assert video.stat().st_size == original_size
    assert "must not overwrite" in capsys.readouterr().err
