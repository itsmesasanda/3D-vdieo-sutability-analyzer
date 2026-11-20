"""Explainable heuristic scoring, problem segments, and capture guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig


SUBSCORE_WEIGHTS_V1 = {
    "visual_quality": 0.20,
    "exposure_and_lighting": 0.15,
    "feature_trackability": 0.20,
    "geometric_consistency": 0.30,
    "viewpoint_motion": 0.15,
}
SUBSCORE_WEIGHTS = SUBSCORE_WEIGHTS_V1

SUBSCORE_WEIGHTS_V2 = {
    "visual_quality": 0.12,
    "exposure_and_lighting": 0.08,
    "feature_trackability": 0.15,
    "geometric_consistency": 0.35,
    "viewpoint_motion": 0.30,
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _ramp(value: float, bad: float, good: float) -> float:
    if good == bad:
        return float(value >= good)
    return _clip01((value - bad) / (good - bad))


def _inverse_ramp(value: float, good: float, bad: float) -> float:
    return 1.0 - _ramp(value, good, bad)


def _band_score(
    value: float, low_bad: float, low_good: float, high_good: float, high_bad: float
) -> float:
    if value < low_good:
        return _ramp(value, low_bad, low_good)
    if value <= high_good:
        return 1.0
    return _inverse_ramp(value, high_good, high_bad)


def _score(value: float) -> float:
    return round(100.0 * _clip01(value), 2)


def compute_subscores_v1(
    quality_summary: Dict[str, Any], geometry_summary: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Map raw, interpretable metrics to five declared 0-100 subscores."""

    laplacian = float(quality_summary["median_laplacian_variance"])
    sharpness = _ramp(np.log1p(laplacian), np.log1p(25.0), np.log1p(250.0))
    blur_consistency = 1.0 - float(quality_summary["blurry_frame_fraction"])
    contrast = _ramp(float(quality_summary["median_contrast_std"]), 18.0, 55.0)
    visual = 0.50 * sharpness + 0.25 * blur_consistency + 0.25 * contrast

    luminance = float(quality_summary["median_luminance"])
    exposure_band = _band_score(luminance, 25.0, 70.0, 185.0, 235.0)
    clipped_fraction = float(quality_summary["median_dark_pixel_fraction"]) + float(
        quality_summary["median_bright_pixel_fraction"]
    )
    clipping = _inverse_ramp(clipped_fraction, 0.05, 0.45)
    stability = _inverse_ramp(
        float(quality_summary["luminance_std_across_frames"]), 8.0, 38.0
    )
    lighting = 0.60 * exposure_band + 0.20 * clipping + 0.20 * stability

    keypoints = _ramp(
        float(geometry_summary["median_keypoints_per_frame"]), 30.0, 500.0
    )
    matches = _ramp(
        float(geometry_summary["median_ratio_test_matches"]), 8.0, 100.0
    )
    feature_presence = 1.0 - float(geometry_summary["low_feature_frame_fraction"])
    trackability = 0.45 * keypoints + 0.35 * matches + 0.20 * feature_presence

    verified_count = _ramp(
        float(geometry_summary["median_verified_matches"]), 8.0, 70.0
    )
    inlier_ratio = _ramp(float(geometry_summary["median_inlier_ratio"]), 0.20, 0.70)
    verified_ratio = _ramp(
        float(geometry_summary["median_verified_match_ratio"]), 0.01, 0.18
    )
    spatial_coverage = _ramp(
        float(geometry_summary["median_spatial_coverage"]), 0.08, 0.60
    )
    valid_pairs = _clip01(float(geometry_summary["valid_geometry_pair_fraction"]))
    geometry = (
        0.25 * verified_count
        + 0.25 * inlier_ratio
        + 0.20 * verified_ratio
        + 0.15 * spatial_coverage
        + 0.15 * valid_pairs
    )

    displacement_raw = geometry_summary.get("median_displacement_normalized")
    displacement = float(displacement_raw) if displacement_raw is not None else 0.0
    motion_band = _band_score(displacement, 0.001, 0.015, 0.14, 0.35)
    moderate_motion = _clip01(float(geometry_summary["moderate_motion_pair_fraction"]))
    non_stationary = 1.0 - float(geometry_summary["near_stationary_pair_fraction"])
    residual_raw = geometry_summary.get("median_homography_residual_p75_normalized")
    residual = float(residual_raw) if residual_raw is not None else 0.0
    parallax_evidence = _ramp(residual, 0.001, 0.012)
    motion = (
        0.45 * motion_band
        + 0.25 * moderate_motion
        + 0.20 * non_stationary
        + 0.10 * parallax_evidence
    )

    subscores = {
        "visual_quality": _score(visual),
        "exposure_and_lighting": _score(lighting),
        "feature_trackability": _score(trackability),
        "geometric_consistency": _score(geometry),
        "viewpoint_motion": _score(motion),
    }
    details = {
        "visual_quality": {
            "weight": SUBSCORE_WEIGHTS["visual_quality"],
            "evidence": {
                "median_laplacian_variance": laplacian,
                "blurry_frame_fraction": quality_summary["blurry_frame_fraction"],
                "median_contrast_std": quality_summary["median_contrast_std"],
            },
        },
        "exposure_and_lighting": {
            "weight": SUBSCORE_WEIGHTS["exposure_and_lighting"],
            "evidence": {
                "median_luminance": luminance,
                "luminance_std_across_frames": quality_summary[
                    "luminance_std_across_frames"
                ],
                "median_clipped_pixel_fraction": round(clipped_fraction, 6),
            },
        },
        "feature_trackability": {
            "weight": SUBSCORE_WEIGHTS["feature_trackability"],
            "evidence": {
                "median_keypoints_per_frame": geometry_summary[
                    "median_keypoints_per_frame"
                ],
                "median_ratio_test_matches": geometry_summary[
                    "median_ratio_test_matches"
                ],
                "low_feature_frame_fraction": geometry_summary[
                    "low_feature_frame_fraction"
                ],
            },
        },
        "geometric_consistency": {
            "weight": SUBSCORE_WEIGHTS["geometric_consistency"],
            "evidence": {
                "median_verified_matches": geometry_summary["median_verified_matches"],
                "median_inlier_ratio": geometry_summary["median_inlier_ratio"],
                "median_spatial_coverage": geometry_summary[
                    "median_spatial_coverage"
                ],
                "valid_geometry_pair_fraction": geometry_summary[
                    "valid_geometry_pair_fraction"
                ],
            },
        },
        "viewpoint_motion": {
            "weight": SUBSCORE_WEIGHTS["viewpoint_motion"],
            "evidence": {
                "median_displacement_normalized": displacement_raw,
                "moderate_motion_pair_fraction": geometry_summary[
                    "moderate_motion_pair_fraction"
                ],
                "near_stationary_pair_fraction": geometry_summary[
                    "near_stationary_pair_fraction"
                ],
                "homography_residual_p75_normalized": residual_raw,
            },
        },
    }
    for name, value in subscores.items():
        details[name]["score"] = value
    return subscores, details


def _group_frame_issues(frame_metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issue_positions: Dict[str, List[int]] = {}
    for position, metric in enumerate(frame_metrics):
        for issue in metric.get("issues", []):
            issue_positions.setdefault(issue, []).append(position)

    segments: List[Dict[str, Any]] = []
    total = max(1, len(frame_metrics))
    for issue, positions in issue_positions.items():
        groups: List[List[int]] = []
        for position in positions:
            if not groups or position != groups[-1][-1] + 1:
                groups.append([position])
            else:
                groups[-1].append(position)
        for group in groups:
            first = frame_metrics[group[0]]
            last = frame_metrics[group[-1]]
            fraction = len(group) / total
            diagnostic_status = "suspected" if issue == "blur" else "detected"
            measurement_scope = (
                "whole_frame" if issue in {"underexposure", "overexposure"} else "sampled_frame"
            )
            confidence_reason = "Thresholded image measurement was observed."
            low_texture_overlap_fraction = None
            if issue == "blur":
                low_texture_overlap_fraction = float(
                    np.mean(
                        [
                            "low_texture" in frame_metrics[position].get("issues", [])
                            for position in group
                        ]
                    )
                )
                confidence_reason = (
                    "The Laplacian edge-detail proxy is low, but it cannot distinguish "
                    "optical blur from a naturally smooth or low-texture surface."
                )
            elif issue in {"underexposure", "overexposure"}:
                confidence_reason = (
                    "The threshold was measured over the whole frame; object-only "
                    "exposure is not available without a validated foreground mask."
                )
            segments.append(
                {
                    "problem_type": issue,
                    "source": "sampled_frame_quality",
                    "start_time_seconds": first["timestamp_seconds"],
                    "end_time_seconds": last["timestamp_seconds"],
                    "start_frame_index": first["frame_index"],
                    "end_frame_index": last["frame_index"],
                    "sample_count": len(group),
                    "severity": "high" if fraction >= 0.50 else "medium",
                    "diagnostic_status": diagnostic_status,
                    "measurement_scope": measurement_scope,
                    "confidence_reason": confidence_reason,
                    **(
                        {
                            "low_texture_overlap_fraction": round(
                                low_texture_overlap_fraction, 6
                            )
                        }
                        if low_texture_overlap_fraction is not None
                        else {}
                    ),
                }
            )
    return segments


def _group_pair_issues(pair_metrics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issue_positions: Dict[str, List[int]] = {}
    for position, metric in enumerate(pair_metrics):
        for issue in metric.get("issues", []):
            issue_positions.setdefault(issue, []).append(position)

    segments: List[Dict[str, Any]] = []
    total = max(1, len(pair_metrics))
    for issue, positions in issue_positions.items():
        groups: List[List[int]] = []
        for position in positions:
            if not groups or position != groups[-1][-1] + 1:
                groups.append([position])
            else:
                groups[-1].append(position)
        for group in groups:
            first = pair_metrics[group[0]]
            last = pair_metrics[group[-1]]
            fraction = len(group) / total
            if issue == "motion_sampling_gap_too_wide":
                diagnostic_status = "not_measurable"
                confidence_reason = (
                    "The sample interval is too wide to measure high-frequency shake; "
                    "this is an unavailable measurement, not a detected defect."
                )
            elif issue in {
                "irregular_camera_motion",
                "excessive_camera_speed",
                "excessive_viewpoint_jump",
                "little_viewpoint_change",
                "homography_dominant_motion",
            }:
                diagnostic_status = "suspected"
                confidence_reason = (
                    "This is image-space motion evidence and cannot by itself identify "
                    "camera translation, object rotation, or object animation."
                )
            else:
                diagnostic_status = "detected"
                confidence_reason = "Thresholded frame-pair evidence was observed."
            segments.append(
                {
                    "problem_type": issue,
                    "source": "orb_ransac_frame_pairs",
                    "start_time_seconds": first["start_time_seconds"],
                    "end_time_seconds": last["end_time_seconds"],
                    "start_frame_index": first["start_frame_index"],
                    "end_frame_index": last["end_frame_index"],
                    "sample_pair_count": len(group),
                    "severity": "high" if fraction >= 0.50 else "medium",
                    "diagnostic_status": diagnostic_status,
                    "measurement_scope": "sampled_frame_pair",
                    "confidence_reason": confidence_reason,
                }
            )
    return segments


def build_problem_segments(
    frame_metrics: Sequence[Dict[str, Any]], pair_metrics: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Collapse consecutive sampled evidence into timestamped problem intervals."""

    segments = _group_frame_issues(frame_metrics) + _group_pair_issues(pair_metrics)
    return sorted(
        segments,
        key=lambda item: (item["start_time_seconds"], item["problem_type"]),
    )


RECOMMENDATIONS = {
    "underexposure": "Add broad, diffuse light or raise exposure; keep the full object away from crushed shadows.",
    "overexposure": "Reduce exposure or soften direct light to preserve surface detail in highlights.",
    "low_contrast": "Use softer directional lighting and a contrasting background so surface features remain visible.",
    "blur": "Inspect visible object edges first: low edge detail can mean optical blur or a smooth surface. Retake more slowly with stabilization and added light only when edge smear is visible.",
    "low_texture": "Add removable visual markers nearby when acceptable, or improve side lighting to reveal surface texture; reflective/transparent objects remain difficult.",
    "insufficient_correspondence": "Keep the object larger in frame and capture textured, sharp views that feature matching can track.",
    "low_overlap": "Use smaller camera steps and keep roughly 60–80% of the previous view visible between frames.",
    "little_viewpoint_change": "The images change very little. If the object is static, add a slow lateral orbit; image-space evidence alone does not verify the physical motion source.",
    "excessive_viewpoint_jump": "Slow the orbit and avoid abrupt jumps so adjacent views retain trackable overlap.",
    "homography_dominant_motion": "Include lateral translation and views around depth-bearing parts of the object; rotation-only or nearly planar motion provides weak parallax evidence.",
    "low_track_retention": "Reduce camera speed and keep more of the previous view visible so feature tracks survive between frames.",
    "excessive_camera_speed": "Review whether high image velocity came from the camera or the object. For a static object, move the camera more slowly and steadily.",
    "irregular_camera_motion": "Review whether the camera or object caused the irregular image motion. For a static object, use a steadier continuous orbit or stabilization.",
    "ambiguous_repetitive_features": "Include distinctive non-repeating surface context or temporary removable markers when appropriate; repetitive patterns can create ambiguous matches.",
}


def recommendations_for(segments: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    recommendations: List[str] = []
    for segment in segments:
        problem = segment["problem_type"]
        recommendation = RECOMMENDATIONS.get(problem)
        if recommendation and recommendation not in seen:
            recommendations.append(recommendation)
            seen.add(recommendation)
    if not recommendations:
        recommendations.append(
            "Maintain a slow, steady orbit, consistent diffuse lighting, sharp focus, and strong overlap throughout capture."
        )
    return recommendations


def score_analysis_v1(
    quality: Dict[str, Any],
    geometry: Dict[str, Any],
    decoded_sample_count: int,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """Compute the bounded score, declared caps, label, and evidence uncertainty."""

    q = quality["summary"]
    g = geometry["summary"]
    subscores, details = compute_subscores_v1(q, g)
    weighted = sum(SUBSCORE_WEIGHTS[name] * subscores[name] for name in SUBSCORE_WEIGHTS)
    caps: List[Dict[str, Any]] = []

    def apply_cap(limit: float, reason: str) -> None:
        nonlocal weighted
        if weighted > limit:
            weighted = limit
            caps.append({"maximum_score": limit, "reason": reason})

    if decoded_sample_count < 4:
        apply_cap(25.0, "Fewer than four frames were decoded; multi-view evidence is inadequate.")
    elif int(g["pair_count"]) < 3:
        apply_cap(35.0, "Fewer than three frame pairs were available for geometric checks.")
    if float(g["valid_geometry_pair_fraction"]) < 0.10:
        apply_cap(44.0, "Robust geometry was unavailable for at least 90% of sampled pairs.")
    if float(g["near_stationary_pair_fraction"]) >= 0.80:
        apply_cap(44.0, "At least 80% of sampled pairs showed almost no viewpoint change.")
    if max(
        float(q["underexposed_frame_fraction"]), float(q["overexposed_frame_fraction"])
    ) >= 0.90:
        apply_cap(55.0, "At least 90% of sampled frames had severe exposure flags.")

    overall = round(float(np.clip(weighted, 0.0, 100.0)), 2)
    if overall >= config.good_score_threshold:
        label = "likely_suitable"
    elif overall >= config.review_score_threshold:
        label = "review_recommended"
    else:
        label = "unlikely_suitable"

    valid_fraction = float(g["valid_geometry_pair_fraction"])
    if decoded_sample_count >= 24 and valid_fraction >= 0.75:
        level = "higher_for_this_heuristic"
    elif decoded_sample_count >= 8 and valid_fraction >= 0.35:
        level = "moderate"
    else:
        level = "low"
    uncertainty_reasons = [
        "This is an engineering heuristic, not a calibrated probability or an independently validated reconstruction predictor.",
        "Image correspondences cannot prove camera translation and may include background or moving-object features.",
    ]
    if valid_fraction < 0.35:
        uncertainty_reasons.append(
            "Few sampled frame pairs produced enough robust correspondences for geometric evidence."
        )
    if float(g["homography_dominant_pair_fraction"]) >= 0.50:
        uncertainty_reasons.append(
            "Much of the observed motion was explained by a homography, consistent with planar or rotation-dominant capture."
        )

    return {
        "overall_score": overall,
        "suitability_label": label,
        "subscores": subscores,
        "subscore_details": details,
        "score_caps_applied": caps,
        "confidence_or_uncertainty": {
            "evidence_level": level,
            "decoded_sample_count": int(decoded_sample_count),
            "valid_geometry_pair_fraction": round(valid_fraction, 6),
            "notes": uncertainty_reasons,
        },
    }


def _value(mapping: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key)
    return float(value) if value is not None and np.isfinite(value) else default


def compute_subscores_v2(
    quality_summary: Dict[str, Any], geometry_summary: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """V2 subscores: make track retention and viewpoint reliability first-class evidence."""

    legacy, _ = compute_subscores_v1(quality_summary, geometry_summary)
    low_overlap = _clip01(_value(geometry_summary, "low_overlap_pair_fraction"))
    short_gap_retention = geometry_summary.get(
        "median_flow_track_retention_short_gap"
    )
    flow_retention_evaluable = short_gap_retention is not None
    orb_continuity_proxy = _ramp(
        _value(geometry_summary, "median_verified_match_ratio"), 0.01, 0.18
    )
    flow_retention_raw = (
        float(short_gap_retention)
        if flow_retention_evaluable
        else orb_continuity_proxy
    )
    flow_retention_source = (
        "lk_forward_backward_short_gap"
        if flow_retention_evaluable
        else "orb_verified_continuity_fallback"
    )
    flow_retention = _ramp(flow_retention_raw, 0.15, 0.65)
    ambiguity_raw = geometry_summary.get("median_descriptor_ambiguity_fraction")
    ambiguity = _value(geometry_summary, "median_descriptor_ambiguity_fraction", 0.5)
    ambiguity_reliability = _inverse_ramp(ambiguity, 0.25, 0.75)
    overlap_reliability = 1.0 - low_overlap

    trackability = (
        0.35 * legacy["feature_trackability"] / 100.0
        + 0.30 * flow_retention
        + 0.20 * overlap_reliability
        + 0.15 * ambiguity_reliability
    )

    valid_geometry = _clip01(
        _value(geometry_summary, "valid_geometry_pair_fraction")
    )
    verified_matches = _ramp(
        _value(geometry_summary, "median_verified_matches"), 8.0, 70.0
    )
    inlier_ratio = _ramp(
        _value(geometry_summary, "median_inlier_ratio"), 0.20, 0.70
    )
    spatial_coverage = _ramp(
        _value(geometry_summary, "median_spatial_coverage"), 0.08, 0.60
    )
    geometry = (
        0.20 * verified_matches
        + 0.20 * inlier_ratio
        + 0.15 * spatial_coverage
        + 0.20 * valid_geometry
        + 0.15 * flow_retention
        + 0.10 * overlap_reliability
    )
    # When correspondences repeatedly lose overlap, abundant texture must not hide it.
    geometry *= 0.65 + 0.35 * overlap_reliability

    speed_raw = _value(geometry_summary, "median_flow_speed_normalized_per_second")
    speed_suitability = _band_score(speed_raw, 0.005, 0.035, 0.22, 0.55)
    irregular_fraction = _clip01(
        _value(
            geometry_summary,
            "irregular_camera_motion_evaluable_pair_fraction",
        )
    )
    speed_cv = _value(geometry_summary, "flow_speed_robust_cv")
    motion_regularity = 0.60 * (1.0 - irregular_fraction) + 0.40 * _inverse_ramp(
        speed_cv, 0.25, 1.25
    )
    homography_fraction = _clip01(
        _value(geometry_summary, "rotation_or_planar_pair_fraction")
    )
    residual_raw = geometry_summary.get("median_homography_residual_p75_normalized")
    parallax_proxy = _ramp(
        _value(geometry_summary, "median_homography_residual_p75_normalized"),
        0.001,
        0.012,
    )
    translation_evidence = 0.55 * parallax_proxy + 0.45 * (
        1.0 - homography_fraction
    )
    cumulative_motion = _value(
        geometry_summary, "cumulative_flow_motion_normalized"
    )
    coverage_amount = _band_score(cumulative_motion, 0.02, 0.10, 1.20, 3.0)
    motion = (
        0.20 * speed_suitability
        + 0.20 * motion_regularity
        + 0.20 * flow_retention
        + 0.25 * translation_evidence
        + 0.15 * coverage_amount
    )

    subscores = {
        "visual_quality": legacy["visual_quality"],
        "exposure_and_lighting": legacy["exposure_and_lighting"],
        "feature_trackability": _score(trackability),
        "geometric_consistency": _score(geometry),
        "viewpoint_motion": _score(motion),
    }
    details = {
        "visual_quality": {
            "score": subscores["visual_quality"],
            "weight": SUBSCORE_WEIGHTS_V2["visual_quality"],
            "evidence": {
                "median_laplacian_variance": quality_summary[
                    "median_laplacian_variance"
                ],
                "blurry_frame_fraction": quality_summary["blurry_frame_fraction"],
                "note": "Measured at the bounded analysis resolution; source pixel count alone is not scored as failure.",
            },
        },
        "exposure_and_lighting": {
            "score": subscores["exposure_and_lighting"],
            "weight": SUBSCORE_WEIGHTS_V2["exposure_and_lighting"],
            "evidence": {
                "median_luminance": quality_summary["median_luminance"],
                "luminance_std_across_frames": quality_summary[
                    "luminance_std_across_frames"
                ],
            },
        },
        "feature_trackability": {
            "score": subscores["feature_trackability"],
            "weight": SUBSCORE_WEIGHTS_V2["feature_trackability"],
            "evidence": {
                "legacy_feature_score": legacy["feature_trackability"],
                "median_flow_track_retention": flow_retention_raw,
                "track_retention_source": flow_retention_source,
                "optical_flow_short_gap_pair_fraction": geometry_summary.get(
                    "optical_flow_short_gap_pair_fraction"
                ),
                "low_overlap_pair_fraction": low_overlap,
                "median_descriptor_ambiguity_fraction": ambiguity_raw,
            },
        },
        "geometric_consistency": {
            "score": subscores["geometric_consistency"],
            "weight": SUBSCORE_WEIGHTS_V2["geometric_consistency"],
            "evidence": {
                "valid_geometry_pair_fraction": valid_geometry,
                "median_verified_matches": geometry_summary[
                    "median_verified_matches"
                ],
                "median_inlier_ratio": geometry_summary["median_inlier_ratio"],
                "median_spatial_coverage": geometry_summary[
                    "median_spatial_coverage"
                ],
                "low_overlap_pair_fraction": low_overlap,
            },
        },
        "viewpoint_motion": {
            "score": subscores["viewpoint_motion"],
            "weight": SUBSCORE_WEIGHTS_V2["viewpoint_motion"],
            "evidence": {
                "median_flow_speed_normalized_per_second": speed_raw,
                "irregular_camera_motion_evaluable_pair_fraction": irregular_fraction,
                "flow_speed_robust_cv": speed_cv,
                "rotation_or_planar_pair_fraction": homography_fraction,
                "homography_residual_p75_normalized": residual_raw,
                "cumulative_flow_motion_normalized": cumulative_motion,
            },
        },
    }
    return subscores, details


def _label_for_score(score: float, config: AnalysisConfig) -> str:
    if score >= config.good_score_threshold:
        return "likely_suitable"
    if score >= config.review_score_threshold:
        return "review_recommended"
    return "unlikely_suitable"


def score_analysis_v2(
    quality: Dict[str, Any],
    geometry: Dict[str, Any],
    video_metadata: Dict[str, Any],
    decoded_sample_count: int,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """V2 score using conjunctive evidence and documented physical failure gates.

    Thresholds remain the fixed MVP thresholds. Gates are triggered only by measured
    failure conditions; low source resolution alone never triggers a cap.
    """

    q = quality["summary"]
    g = geometry["summary"]
    subscores, details = compute_subscores_v2(q, g)
    weighted_log = sum(
        weight * np.log(max(5.0, float(subscores[name])) / 100.0)
        for name, weight in SUBSCORE_WEIGHTS_V2.items()
    )
    overall = 100.0 * float(np.exp(weighted_log))
    gate_evaluations: List[Dict[str, Any]] = []
    applied_caps: List[Dict[str, Any]] = []

    def gate(
        gate_id: str,
        triggered: bool,
        maximum_score: float,
        physical_basis: str,
        evidence: Dict[str, Any],
    ) -> None:
        nonlocal overall
        evaluation = {
            "gate_id": gate_id,
            "triggered": bool(triggered),
            "maximum_score": maximum_score,
            "physical_basis": physical_basis,
            "evidence": evidence,
        }
        gate_evaluations.append(evaluation)
        if triggered and overall > maximum_score:
            overall = maximum_score
            applied_caps.append(evaluation)

    pair_count = int(_value(g, "pair_count"))
    valid_fraction = _clip01(_value(g, "valid_geometry_pair_fraction"))
    stationary_fraction = _clip01(_value(g, "near_stationary_pair_fraction"))
    low_overlap = _clip01(_value(g, "low_overlap_pair_fraction"))
    homography_fraction = _clip01(
        _value(g, "rotation_or_planar_pair_fraction")
    )
    residual = _value(g, "median_homography_residual_p75_normalized")
    short_gap_retention = g.get("median_flow_track_retention_short_gap")
    flow_retention_evaluable = short_gap_retention is not None
    flow_retention = (
        float(short_gap_retention)
        if flow_retention_evaluable
        else _ramp(_value(g, "median_verified_match_ratio"), 0.01, 0.18)
    )
    excessive_speed = _clip01(
        _value(g, "excessive_camera_speed_pair_fraction")
    )
    irregular_motion = _clip01(
        _value(g, "irregular_camera_motion_evaluable_pair_fraction")
    )
    duration = video_metadata.get("duration_seconds")
    duration_value = float(duration) if duration is not None else None

    gate(
        "insufficient_temporal_evidence",
        decoded_sample_count < 6 or pair_count < 3 or (
            duration_value is not None and duration_value < 0.75
        ),
        25.0,
        "A multi-view reconstruction needs several temporally distinct observations.",
        {
            "decoded_sample_count": decoded_sample_count,
            "pair_count": pair_count,
            "duration_seconds": duration_value,
        },
    )
    gate(
        "geometry_unavailable",
        valid_fraction < 0.10,
        35.0,
        "Almost no sampled pair produced enough robust geometry to support reconstruction.",
        {"valid_geometry_pair_fraction": valid_fraction},
    )
    gate(
        "stationary_view",
        stationary_fraction >= 0.80,
        35.0,
        "Repeated nearly stationary views do not provide multi-view baseline.",
        {"near_stationary_pair_fraction": stationary_fraction},
    )
    rotation_cue_reliable = (
        _value(g, "median_keypoints_per_frame") >= 500.0
        and _value(g, "median_spatial_coverage") >= 0.30
        and flow_retention >= 0.50
        and _value(g, "median_homography_inlier_ratio") >= 0.90
    )
    rotation_planar_trigger = (
        homography_fraction >= 0.80
        and residual < 0.006
        and rotation_cue_reliable
        and (
            (duration_value is not None and duration_value < 3.0)
            or subscores["viewpoint_motion"] < 45.0
        )
    )
    gate(
        "rotation_or_planar_coverage_uncertain",
        rotation_planar_trigger,
        44.0,
        "Motion explained almost entirely by a homography does not demonstrate depth-bearing camera translation; short observation time increases coverage uncertainty.",
        {
            "rotation_or_planar_pair_fraction": homography_fraction,
            "homography_residual_p75_normalized": residual,
            "duration_seconds": duration_value,
            "viewpoint_motion_subscore": subscores["viewpoint_motion"],
            "rotation_cue_reliable": rotation_cue_reliable,
            "median_keypoints_per_frame": g["median_keypoints_per_frame"],
            "median_spatial_coverage": g["median_spatial_coverage"],
            "median_homography_inlier_ratio": g[
                "median_homography_inlier_ratio"
            ],
        },
    )
    gate(
        "persistent_low_overlap",
        low_overlap >= 0.50
        and (
            subscores["geometric_consistency"] < 55.0
            or (flow_retention_evaluable and flow_retention < 0.35)
        ),
        60.0,
        "When at least half of sampled transitions lose overlap, feature-rich frames cannot guarantee stable tracks across the capture.",
        {
            "low_overlap_pair_fraction": low_overlap,
            "geometric_consistency_subscore": subscores[
                "geometric_consistency"
            ],
            "median_flow_track_retention": flow_retention,
            "flow_retention_evaluable": flow_retention_evaluable,
        },
    )
    gate(
        "fast_motion_with_track_loss",
        excessive_speed >= 0.50
        and (
            low_overlap >= 0.30
            or _value(q, "blurry_frame_fraction") >= 0.10
            or (flow_retention_evaluable and flow_retention < 0.35)
        ),
        60.0,
        "Sustained high image velocity combined with blur, low overlap, or track loss undermines correspondence continuity.",
        {
            "excessive_camera_speed_pair_fraction": excessive_speed,
            "low_overlap_pair_fraction": low_overlap,
            "blurry_frame_fraction": q["blurry_frame_fraction"],
            "median_flow_track_retention": flow_retention,
            "flow_retention_evaluable": flow_retention_evaluable,
        },
    )
    gate(
        "irregular_motion_with_overlap_loss",
        irregular_motion >= 0.40 and low_overlap >= 0.30,
        60.0,
        "Frequent direction/speed discontinuities combined with overlap loss are consistent with unstable capture rather than a smooth orbit.",
        {
            "irregular_camera_motion_pair_fraction": irregular_motion,
            "low_overlap_pair_fraction": low_overlap,
        },
    )
    gate(
        "severe_irregular_motion",
        irregular_motion >= 0.70
        and _value(g, "median_flow_speed_normalized_per_second") >= 0.15,
        60.0,
        "Frequent large direction or speed changes are inconsistent with a controlled smooth orbit even when local tracks survive.",
        {
            "irregular_camera_motion_pair_fraction": irregular_motion,
            "median_flow_speed_normalized_per_second": g[
                "median_flow_speed_normalized_per_second"
            ],
        },
    )
    severe_exposure_fraction = max(
        _value(q, "underexposed_frame_fraction"),
        _value(q, "overexposed_frame_fraction"),
    )
    gate(
        "severe_exposure_failure",
        severe_exposure_fraction >= 0.90,
        55.0,
        "If at least 90% of frames are severely under- or overexposed, recoverable surface appearance is persistently limited.",
        {
            "underexposed_frame_fraction": q["underexposed_frame_fraction"],
            "overexposed_frame_fraction": q["overexposed_frame_fraction"],
        },
    )
    persistent_severe_blur = (
        _value(q, "blurry_frame_fraction") >= 0.80
        and _value(q, "median_laplacian_variance") < 15.0
    )
    gate(
        "persistent_severe_blur",
        persistent_severe_blur,
        55.0,
        "When severe blur affects nearly the entire capture, repeatable local detail is unavailable even if coarse correspondences survive.",
        {
            "blurry_frame_fraction": q["blurry_frame_fraction"],
            "median_laplacian_variance_at_analysis_resolution": q[
                "median_laplacian_variance"
            ],
            "resolution_alone_triggers_failure": False,
        },
    )

    reliability_reasons: List[str] = []
    if low_overlap >= 0.50:
        reliability_reasons.append("At least half of sampled pairs have low overlap.")
    if homography_fraction >= 0.70:
        reliability_reasons.append(
            "Most motion is homography-dominant, so translation/parallax is unverified."
        )
    if flow_retention_evaluable and flow_retention < 0.30:
        reliability_reasons.append("Median optical-flow track retention is below 30%.")
    if irregular_motion >= 0.40:
        reliability_reasons.append("Motion is irregular on at least 40% of sampled pairs.")
    if duration_value is not None and duration_value < 3.0:
        reliability_reasons.append("The source is shorter than three seconds.")
    if not flow_retention_evaluable:
        reliability_reasons.append(
            "Uniform sample gaps exceed 0.30 seconds, so shake and LK retention are unresolved; ORB continuity is used instead."
        )
    if valid_fraction < 0.35 or len(reliability_reasons) >= 2:
        reliability = "low"
    elif reliability_reasons:
        reliability = "moderate"
    else:
        reliability = "higher_for_this_heuristic"

    overall = round(float(np.clip(overall, 0.0, 100.0)), 2)
    uncertainty = [
        "V2 remains an uncalibrated engineering heuristic, not a probability or validated accuracy estimate.",
        "Optical flow and homography cues are image-space evidence and cannot prove physical camera translation.",
    ]
    uncertainty.extend(reliability_reasons)
    if video_metadata.get("source_width") and video_metadata.get("source_height"):
        uncertainty.append(
            "Source resolution and compression are reported as diagnostics; low pixel resolution alone is not treated as reconstruction failure."
        )

    return {
        "scoring_version": "heuristic_v2",
        "scoring_revision": "2.1",
        "overall_score": overall,
        "suitability_label": _label_for_score(overall, config),
        "subscores": subscores,
        "subscore_details": details,
        "score_caps_applied": applied_caps,
        "critical_failure_gate_evaluations": gate_evaluations,
        "geometry_reliability": {
            "level": reliability,
            "reasons": reliability_reasons,
            "valid_geometry_pair_fraction": round(valid_fraction, 6),
            "median_flow_track_retention": round(flow_retention, 6),
            "flow_track_retention_source": "lk_forward_backward_short_gap"
            if flow_retention_evaluable
            else "orb_verified_continuity_fallback",
            "low_overlap_pair_fraction": round(low_overlap, 6),
            "rotation_or_planar_pair_fraction": round(
                homography_fraction, 6
            ),
        },
        "confidence_or_uncertainty": {
            "evidence_level": reliability,
            "decoded_sample_count": int(decoded_sample_count),
            "valid_geometry_pair_fraction": round(valid_fraction, 6),
            "notes": uncertainty,
        },
    }


# Backward-compatible helper names. The report explicitly chooses a version.
compute_subscores = compute_subscores_v1
score_analysis = score_analysis_v1
