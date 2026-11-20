"""Pure presentation helpers for the Streamlit competition MVP.

These functions consume the stable JSON report and never alter analysis metrics or
scores. Keeping view-model construction outside Streamlit makes UI behavior testable
without installing the optional UI dependency.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SUBSCORE_PRESENTATION = {
    "visual_quality": (
        "Image detail proxy",
        "Measured edge detail and contrast; smooth surfaces can resemble optical blur.",
    ),
    "exposure_and_lighting": (
        "Whole-frame exposure",
        "Whole-frame brightness and clipping; object-only exposure is not segmented.",
    ),
    "feature_trackability": (
        "Feature tracking",
        "Whether distinctive points can be followed reliably between adjacent views.",
    ),
    "geometric_consistency": (
        "View overlap",
        "Whether matched features agree geometrically; high overlap can still be repeated views.",
    ),
    "viewpoint_motion": (
        "Image-motion evidence",
        "Image-space regularity and motion proxies; camera translation is not identified.",
    ),
}


PROBLEM_GUIDANCE: Dict[str, Dict[str, str]] = {
    "blur": {
        "label": "Possible blur / low edge detail",
        "what": "The sharpness proxy is low; this may be blur or a naturally smooth surface.",
        "why": "Reconstruction needs repeatable local details to match the same surface across views.",
        "action": "Inspect object edges at full size first. If they are visibly smeared, move more slowly, stabilize the phone, add light, and lock focus.",
    },
    "underexposure": {
        "label": "Whole frame appears dark",
        "what": "The whole-frame exposure proxy found extensive dark pixels; object-only exposure was not measured.",
        "why": "Feature detectors cannot recover texture that was clipped to near-black.",
        "action": "Add broad diffuse light or raise exposure while keeping highlights controlled.",
    },
    "overexposure": {
        "label": "Whole frame appears bright",
        "what": "The whole-frame exposure proxy found extensive bright pixels; object-only exposure was not measured.",
        "why": "Uniform white regions provide few stable correspondences between views.",
        "action": "Reduce exposure and soften direct reflections with larger diffuse lights.",
    },
    "low_contrast": {
        "label": "Low contrast",
        "what": "Object edges and texture have weak tonal separation.",
        "why": "Weak gradients make repeatable feature localization less reliable.",
        "action": "Use a contrasting background and soft directional light to reveal shape.",
    },
    "low_texture": {
        "label": "Low surface texture",
        "what": "The sampled view contains few distinctive local patterns.",
        "why": "Smooth, reflective, or transparent areas are difficult to match consistently.",
        "action": "Improve side lighting or add removable non-repeating markers nearby when appropriate.",
    },
    "low_overlap": {
        "label": "Low view overlap",
        "what": "Too little of the previous view remains visible in the next sampled view.",
        "why": "Feature tracks break before they can triangulate stable 3D points.",
        "action": "Take smaller camera steps and keep roughly 60–80% of the previous view visible.",
    },
    "low_track_retention": {
        "label": "Tracks are being lost",
        "what": "Optical-flow points do not survive between nearby frames.",
        "why": "Short tracks weaken camera-pose estimation and multi-view triangulation.",
        "action": "Slow down, keep the object centered, and preserve more overlap.",
    },
    "irregular_camera_motion": {
        "label": "Irregular image motion",
        "what": "Image-space direction or speed changes abruptly between nearby views.",
        "why": "Shake increases blur and makes correspondence continuity less predictable.",
        "action": "Review whether the camera or object moved. For a static object, use stabilization and follow one smooth path.",
    },
    "excessive_camera_speed": {
        "label": "High image velocity",
        "what": "Timestamp-normalized image motion is unusually high.",
        "why": "Fast motion causes blur and moves features too far for reliable tracking.",
        "action": "Slow the orbit and pause briefly if autofocus or exposure needs to settle.",
    },
    "excessive_viewpoint_jump": {
        "label": "Abrupt image-space jump",
        "what": "Adjacent sampled views are separated by a large image-space jump.",
        "why": "Large jumps reduce shared content and can break geometric initialization.",
        "action": "Record a continuous path with smaller angular steps around the object.",
    },
    "insufficient_correspondence": {
        "label": "Too few reliable matches",
        "what": "The pair has too few geometrically verified feature correspondences.",
        "why": "Camera poses cannot be estimated reliably without enough consistent matches.",
        "action": "Keep the object larger, sharper, and well lit; include distinctive context.",
    },
    "little_viewpoint_change": {
        "label": "Little image-space change",
        "what": "Nearby sampled images change very little; physical camera motion is not directly measured.",
        "why": "Triangulation needs a physical baseline, not repeated images from one position.",
        "action": "Move laterally around the static object instead of rotating in place.",
    },
    "homography_dominant_motion": {
        "label": "Depth coverage uncertain",
        "what": "Most motion can be explained by a single planar transform.",
        "why": "This can indicate rotation-only capture or a mostly planar view with weak parallax.",
        "action": "Include lateral translation and views around depth-bearing sides of the object.",
    },
    "ambiguous_repetitive_features": {
        "label": "Repetitive texture",
        "what": "Many local descriptors look similar to one another.",
        "why": "Repeated patterns can match the wrong surface locations and corrupt geometry.",
        "action": "Include distinctive context or temporary non-repeating markers when acceptable.",
    },
    "motion_sampling_gap_too_wide": {
        "label": "High-frequency shake not measurable",
        "what": "Uniform analysis samples are too far apart for short-gap shake measurement.",
        "why": "The score can assess ORB continuity, but cannot confirm frame-to-frame steadiness here.",
        "action": "Review the original interval manually; record a steady path at a consistent speed.",
    },
}


PRIORITY = {
    "blur": 0,
    "underexposure": 1,
    "overexposure": 1,
    "low_overlap": 2,
    "low_track_retention": 3,
    "irregular_camera_motion": 4,
    "excessive_camera_speed": 5,
    "excessive_viewpoint_jump": 5,
}


def score_style(score: float) -> Dict[str, str]:
    if score >= 70:
        return {"label": "Likely suitable", "color": "#16a674", "tone": "positive"}
    if score >= 45:
        return {"label": "Review recommended", "color": "#e49a22", "tone": "warning"}
    return {"label": "Unlikely suitable", "color": "#df5b5b", "tone": "negative"}


def component_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = report.get("subscore_details", {})
    rows = []
    for key, score in report["subscores"].items():
        label, explanation = SUBSCORE_PRESENTATION.get(
            key, (key.replace("_", " ").title(), "Measured heuristic component.")
        )
        weight = details.get(key, {}).get("weight")
        rows.append(
            {
                "key": key,
                "label": label,
                "score": float(score),
                "weight_percent": round(float(weight) * 100.0) if weight is not None else None,
                "explanation": explanation,
                "style": score_style(float(score)),
            }
        )
    return rows


def _optional_percentage(value: Any) -> Optional[int]:
    """Convert an analyzer fraction to a display percentage without inventing zero."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(100.0 * numeric)


def viewpoint_diversity_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the standalone viewpoint result used by the UI."""

    raw_result = report.get("viewpoint_diversity", {})
    result: Mapping[str, Any] = (
        raw_result if isinstance(raw_result, Mapping) else {}
    )
    label = result.get("label", "Unknown")
    styles = {
        "High": {"color": "#16a674", "tone": "positive"},
        "Moderate": {"color": "#e49a22", "tone": "warning"},
        "Low": {"color": "#df5b5b", "tone": "negative"},
        "Unknown": {"color": "#6f7f7c", "tone": "neutral"},
    }


    raw_indicators = result.get("indicators", {})
    raw_tracking = result.get("object_tracking", {})
    raw_background = result.get("background_diagnostics", {})
    raw_method = result.get("method", {})
    raw_decision = result.get("decision_explanation", {})
    raw_interpretation = result.get("geometric_evidence_interpretation", {})
    indicators = raw_indicators if isinstance(raw_indicators, Mapping) else {}
    tracking = raw_tracking if isinstance(raw_tracking, Mapping) else {}
    raw_foreground = tracking.get("foreground_separation", {})
    foreground = raw_foreground if isinstance(raw_foreground, Mapping) else {}
    background = raw_background if isinstance(raw_background, Mapping) else {}
    method = raw_method if isinstance(raw_method, Mapping) else {}
    decision = raw_decision if isinstance(raw_decision, Mapping) else {}
    interpretation = (
        raw_interpretation if isinstance(raw_interpretation, Mapping) else {}
    )
    continuity_raw = interpretation.get("feature_continuity", {})
    change_raw = interpretation.get("viewpoint_change", {})
    scope_raw = interpretation.get("evidence_scope", {})
    physical_raw = interpretation.get("physical_motion", {})
    continuity = continuity_raw if isinstance(continuity_raw, Mapping) else {}
    change = change_raw if isinstance(change_raw, Mapping) else {}
    scope = scope_raw if isinstance(scope_raw, Mapping) else {}
    physical = physical_raw if isinstance(physical_raw, Mapping) else {}
    recommendations = result.get("recommendations", [])
    warnings = result.get("warnings", [])
    repeated_views = result.get("repeated_view_intervals", [])
    evidence_available = bool(indicators)
    parallax_fraction = indicators.get(
        "temporally_separated_parallax_proxy_pair_fraction"
    )
    if parallax_fraction is None:
        parallax_fraction = indicators.get("parallax_proxy_pair_fraction")
    return {
        "label": label,
        "summary": result.get("summary", "Viewpoint evidence is unavailable."),
        "style": styles.get(label, styles["Unknown"]),
        "recommendations": list(recommendations)
        if isinstance(recommendations, (list, tuple))
        else [],
        "warnings": list(warnings) if isinstance(warnings, (list, tuple)) else [],
        "repeated_view_count": (
            len(repeated_views)
            if evidence_available
            and "repeated_view_intervals" in result
            and isinstance(repeated_views, (list, tuple))
            else None
        ),
        "intrinsics_used": bool(method.get("intrinsics_used", False)),
        "affects_overall_score": bool(
            method.get("affects_overall_score", False)
        ),
        "analysis_mode": method.get("analysis_mode", "whole_frame"),
        "primary_reason": decision.get("primary_reason", ""),
        "feature_continuity_label": continuity.get(
            "label", "Not available"
        ),
        "feature_continuity_meaning": continuity.get("meaning", ""),
        "viewpoint_change_label": change.get("label", "Not available"),
        "viewpoint_change_meaning": change.get("meaning", ""),
        "evidence_scope_label": scope.get("label", "Not available"),
        "object_region_isolation_status": scope.get(
            "object_region_isolation_status"
        ),
        "physical_motion_label": (
            "Not verified"
            if physical.get("status") in {None, "not_verified"}
            else physical.get("status", "Not available")
        ),
        "physical_motion_reason": physical.get("reason", ""),
        "confounding_risks": list(interpretation.get("confounding_risks", []))
        if isinstance(interpretation.get("confounding_risks"), (list, tuple))
        else [],
        "tracking_percent": _optional_percentage(
            tracking.get("tracked_sample_fraction")
        ),
        "tracking_cycle_verified_percent": _optional_percentage(
            tracking.get("cycle_verified_step_fraction")
        ),
        "foreground_mask_status": foreground.get(
            "mask_quality_status", "not_available"
        ),
        "foreground_mask_reliable": foreground.get(
            "reliable_for_object_evidence"
        ),
        "foreground_fraction_percent": _optional_percentage(
            foreground.get("foreground_fraction_within_box")
        ),
        "foreground_roi_stability_percent": _optional_percentage(
            (
                foreground.get("roi_sensitivity", {}).get("minimum_mask_iou")
                if isinstance(foreground.get("roi_sensitivity"), Mapping)
                else None
            )
        ),
        "foreground_mask_flags": list(foreground.get("quality_flags", []))
        if isinstance(foreground.get("quality_flags"), (list, tuple))
        else [],
        "background_dominated_percent": _optional_percentage(
            background.get("background_dominated_pair_fraction")
        ),
        "parallax_proxy_percent": _optional_percentage(parallax_fraction),
        "substantial_change_percent": _optional_percentage(
            indicators.get("temporally_separated_substantial_change_fraction")
        ),
        "mixed_motion_percent": _optional_percentage(
            indicators.get("mixed_static_dynamic_pair_fraction")
        ),
    }


def capture_completeness_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the isolated experimental capture-guidance view model."""

    raw = report.get("capture_completeness", {})
    result: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    variation_raw = result.get("view_variation", {})
    elevation_raw = result.get("elevation_variation", {})
    mode_raw = result.get("capture_mode", {})
    method_raw = result.get("method", {})
    variation = variation_raw if isinstance(variation_raw, Mapping) else {}
    elevation = elevation_raw if isinstance(elevation_raw, Mapping) else {}
    mode = mode_raw if isinstance(mode_raw, Mapping) else {}
    method = method_raw if isinstance(method_raw, Mapping) else {}
    suggestions = result.get("suggestions", [])
    limitations = result.get("limitations", [])
    status = str(result.get("status", "not_verified"))
    styles = {
        "varied_view_evidence": {"color": "#237a62", "tone": "positive"},
        "limited_view_evidence": {"color": "#a86816", "tone": "warning"},
        "not_verified": {"color": "#6f7f7c", "tone": "neutral"},
    }
    return {
        "status": status,
        "label": result.get("label", "Capture coverage not verified"),
        "summary": result.get(
            "summary", "Experimental capture guidance is unavailable."
        ),
        "style": styles.get(status, styles["not_verified"]),
        "tracking_percent": result.get("tracking_percent"),
        "view_variation_label": variation.get("label", "Not verified"),
        "substantial_change_percent": variation.get(
            "substantial_change_percent"
        ),
        "parallax_proxy_percent": variation.get("parallax_proxy_percent"),
        "elevation_label": elevation.get("label", "Not verified"),
        "elevation_reason": elevation.get("reason", "Not available"),
        "capture_mode_label": mode.get(
            "label", "Camera orbit versus object rotation: uncertain"
        ),
        "capture_mode_reason": mode.get("reason", "Not available"),
        "suggestions": list(suggestions)
        if isinstance(suggestions, (list, tuple))
        else [],
        "limitations": list(limitations)
        if isinstance(limitations, (list, tuple))
        else [],
        "affects_suitability_score": bool(
            method.get("affects_suitability_score", False)
        ),
        "affects_ridge_prediction": bool(
            method.get("affects_ridge_prediction", False)
        ),
        "affects_viewpoint_classification": bool(
            method.get("affects_viewpoint_classification", False)
        ),
    }


def registration_prediction_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the separate learned COLMAP-registration result for Streamlit."""

    result = report.get("registration_prediction", {})
    status = result.get("status", "unavailable")
    available = status == "available"
    return {
        "status": status,
        "available": available,
        "percentage": result.get("predicted_registration_percentage"),
        "reason": result.get(
            "reason", "No compatible Random Forest prediction is attached."
        ),
        "interpretation": result.get(
            "interpretation",
            "This learned estimate is unavailable; the heuristic remains authoritative.",
        ),
        "model_version": result.get("model_version"),
        "parity_verified": (
            result.get("feature_parity", {}).get("status") == "verified"
        ),
        "parity_case_count": result.get("feature_parity", {}).get("case_count", 0),
        "affects_suitability_score": bool(
            result.get("affects_suitability_score", False)
        ),
        "affects_viewpoint_diversity": bool(
            result.get("affects_viewpoint_diversity", False)
        ),
        "colmap_run": bool(result.get("colmap_run", False)),
    }


def _time_range(start: float, end: float) -> str:
    if abs(end - start) < 0.005:
        return f"{start:.2f}s"
    return f"{start:.2f}–{end:.2f}s"


def timeline_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    duration = report.get("raw_metrics", {}).get("video", {}).get("duration_seconds")
    duration_value = float(duration) if duration else 0.0
    rows: List[Dict[str, Any]] = []
    for segment in report.get("problem_segments", []):
        problem = segment["problem_type"]
        guidance = PROBLEM_GUIDANCE.get(
            problem,
            {
                "label": problem.replace("_", " ").title(),
                "what": "The heuristic detected an unusual condition.",
                "why": "This evidence may reduce reconstruction reliability.",
                "action": "Inspect this interval and repeat it with a slow, stable, well-lit orbit.",
            },
        )
        start = float(segment["start_time_seconds"])
        end = float(segment["end_time_seconds"])
        start_index = int(segment["start_frame_index"])
        end_index = int(segment["end_frame_index"])
        representative_index = int(round((start_index + end_index) / 2.0))
        representative_time = (start + end) / 2.0
        rows.append(
            {
                **segment,
                **guidance,
                "diagnostic_status": segment.get("diagnostic_status", "detected"),
                "confidence_reason": segment.get(
                    "confidence_reason",
                    "This legacy report did not record diagnostic confidence.",
                ),
                "time_range": _time_range(start, end),
                "representative_frame_index": representative_index,
                "representative_time_seconds": round(representative_time, 6),
                "start_percent": round(100.0 * start / duration_value, 3)
                if duration_value > 0
                else 0.0,
                "end_percent": round(100.0 * end / duration_value, 3)
                if duration_value > 0
                else 0.0,
            }
        )
    return rows


def representative_problem_rows(
    rows: Sequence[Dict[str, Any]], limit: int = 8
) -> List[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            PRIORITY.get(row["problem_type"], 20),
            0 if row.get("severity") == "high" else 1,
            -(float(row["end_time_seconds"]) - float(row["start_time_seconds"])),
            float(row["start_time_seconds"]),
        ),
    )
    selected: List[Dict[str, Any]] = []
    used_indices = set()
    used_types = set()
    for row in ordered:
        index = row["representative_frame_index"]
        problem = row["problem_type"]
        if index in used_indices or problem in used_types:
            continue
        selected.append(row)
        used_indices.add(index)
        used_types.add(problem)
        if len(selected) >= limit:
            break
    return selected


def extract_problem_frames(
    video_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    max_frames: int = 8,
    max_side: int = 720,
) -> List[Dict[str, Any]]:
    """Read representative source frames for UI inspection without modifying input."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for representative-frame inspection") from exc

    selected = representative_problem_rows(rows, limit=max_frames)
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video for frame inspection: {video_path}")
    output: List[Dict[str, Any]] = []
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(row["representative_frame_index"]))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        if max(height, width) > max_side:
            scale = max_side / float(max(height, width))
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        output.append(
            {
                "problem_type": row["problem_type"],
                "label": row["label"],
                "time_range": row["time_range"],
                "timestamp_seconds": row["representative_time_seconds"],
                "frame_index": row["representative_frame_index"],
                "why": row["why"],
                "action": row["action"],
                "image_rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            }
        )
    cap.release()
    return output


def recommendation_cards(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("diagnostic_status") == "not_measurable":
            continue
        problem = row["problem_type"]
        if problem not in grouped:
            grouped[problem] = {
                "problem_type": problem,
                "label": row["label"],
                "what": row["what"],
                "why": row["why"],
                "action": row["action"],
                "severity": row["severity"],
                "diagnostic_status": row.get("diagnostic_status", "detected"),
                "confidence_reason": row.get("confidence_reason", ""),
                "time_ranges": [],
            }
        grouped[problem]["time_ranges"].append(row["time_range"])
        if row["severity"] == "high":
            grouped[problem]["severity"] = "high"
        if row.get("diagnostic_status") == "suspected":
            grouped[problem]["diagnostic_status"] = "suspected"
    cards = list(grouped.values())
    for card in cards:
        card["where"] = ", ".join(card.pop("time_ranges")[:5])
    return sorted(
        cards,
        key=lambda card: (
            PRIORITY.get(card["problem_type"], 20),
            0 if card["severity"] == "high" else 1,
        ),
    )


def load_evaluation_rows(
    human_review_path: Path, results_path: Path
) -> List[Dict[str, Any]]:
    with Path(human_review_path).open(newline="", encoding="utf-8") as handle:
        reviews = list(csv.DictReader(handle))
    with Path(results_path).open(newline="", encoding="utf-8") as handle:
        results = {row["filename"]: row for row in csv.DictReader(handle)}
    rows: List[Dict[str, Any]] = []
    for review in reviews:
        result = results.get(review["filename"], {})
        raw_human_score = review.get("human_score_numeric", "").strip()
        human_score: Optional[float] = float(raw_human_score) if raw_human_score else None
        rows.append(
            {
                "filename": review["filename"],
                "baseline_v1_score": float(review["baseline_score"]),
                "v2_score": float(result["v2_score"]) if result.get("v2_score") else None,
                "human_score": human_score,
                "human_score_status": review["human_score_numeric_status"],
                "human_assessment": review["human_qualitative_assessment"].replace("_", " "),
                "camera_speed_feedback": review["camera_speed_feedback"].replace("_", " "),
                "camera_shake_feedback": review["camera_shake_feedback"].replace("_", " "),
                "notes": review["review_notes"],
            }
        )
    return rows

