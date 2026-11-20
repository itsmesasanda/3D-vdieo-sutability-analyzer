"""Conservative capture-completeness guidance from existing viewpoint evidence.

This module does not estimate surface coverage, orbit angle, or named object sides.
It only summarizes evidence already produced by the experimental object-focused
viewpoint path.  It is deliberately isolated from every score and prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _number(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _unverified(reason: str, *, analysis_mode: str) -> Dict[str, Any]:
    suggestions = [
        "Use object-focused analysis with a box that can be tracked across the full recording.",
        "Record a slow path with overlapping lateral views and keep the object rigid where possible.",
        "If upper surfaces matter, add a deliberate elevated pass; this analysis cannot verify that view from the current evidence.",
    ]
    return {
        "schema_version": "experimental_capture_completeness/1.0",
        "status": "not_verified",
        "label": "Capture coverage not verified",
        "summary": reason,
        "view_variation": {
            "status": "not_verified",
            "label": "Not verified",
        },
        "elevation_variation": {
            "status": "not_measured",
            "label": "Not verified",
            "reason": (
                "The existing image-space measurements do not provide a reliable "
                "object-relative higher/lower camera trajectory."
            ),
        },
        "capture_mode": {
            "status": "uncertain",
            "label": "Camera orbit versus object rotation: uncertain",
            "reason": "Reliable object/background motion evidence is unavailable.",
        },
        "suggestions": suggestions,
        "limitations": [
            "No exact orbit angle, 360-degree completion, named side, or surface-coverage percentage is inferred.",
            "A rotating or animated object can mimic camera viewpoint change.",
        ],
        "method": {
            "name": "existing_object_viewpoint_evidence_summary",
            "analysis_mode": analysis_mode,
            "new_feature_matching_run": False,
            "affects_suitability_score": False,
            "affects_ridge_prediction": False,
            "affects_viewpoint_classification": False,
        },
    }


def _capture_mode(viewpoint: Mapping[str, Any]) -> Dict[str, str]:
    background_raw = viewpoint.get("background_diagnostics", {})
    background = background_raw if isinstance(background_raw, Mapping) else {}
    evaluated = int(_number(background.get("evaluated_pair_count")) or 0)
    object_static_background = _number(
        background.get("moving_object_with_static_background_pair_fraction")
    )

    # Reuse the existing object-animation rejection evidence.  The same pattern is
    # also consistent with a rigid turntable capture, so it is never called proof.
    if (
        evaluated >= 3
        and object_static_background is not None
        and object_static_background >= 0.30
    ):
        return {
            "status": "object_motion_pattern",
            "label": "Object-motion / static-background pattern",
            "reason": (
                "Across several long-gap pairs, the selected object moves while "
                "the surrounding background remains nearly static. This is "
                "consistent with a turntable capture, but animation cannot be excluded."
            ),
        }

    rows = background.get("pairs", [])
    usable: List[Mapping[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("evaluated"):
                continue
            object_motion = _number(row.get("object_motion_normalized"))
            background_motion = _number(row.get("background_motion_normalized"))
            if object_motion is None or background_motion is None:
                continue
            if int(_number(row.get("object_verified_matches")) or 0) < 8:
                continue
            if int(_number(row.get("background_verified_matches")) or 0) < 12:
                continue
            usable.append(row)

    co_moving = 0
    for row in usable:
        object_motion = float(row["object_motion_normalized"])
        background_motion = float(row["background_motion_normalized"])
        if object_motion > 0.004 and background_motion > 0.004:
            co_moving += 1
    co_moving_fraction = co_moving / len(usable) if usable else 0.0
    indicators_raw = viewpoint.get("indicators", {})
    indicators = indicators_raw if isinstance(indicators_raw, Mapping) else {}
    parallax = _number(
        indicators.get("temporally_separated_parallax_proxy_pair_fraction")
    )
    if (
        len(usable) >= 3
        and co_moving_fraction >= 0.60
        and parallax is not None
        and parallax >= 0.20
        and viewpoint.get("label") in {"Moderate", "High"}
    ):
        return {
            "status": "shared_scene_motion_pattern",
            "label": "Shared scene-motion pattern",
            "reason": (
                "The object and background both move across several verified "
                "long-gap pairs, with non-homography image support. Camera motion is "
                "one possible explanation, but physical translation and a complete orbit are not verified."
            ),
        }

    return {
        "status": "uncertain",
        "label": "Camera orbit versus object rotation: uncertain",
        "reason": (
            "The available object/background motion does not reliably distinguish "
            "camera motion, rigid object rotation, and object animation."
        ),
    }


def analyze_capture_completeness(
    viewpoint_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Summarize conservative capture guidance without running new analysis."""

    method_raw = viewpoint_result.get("method", {})
    method = method_raw if isinstance(method_raw, Mapping) else {}
    analysis_mode = str(method.get("analysis_mode", "whole_frame"))
    decision_raw = viewpoint_result.get("decision_explanation", {})
    decision = decision_raw if isinstance(decision_raw, Mapping) else {}
    primary_reason = str(
        decision.get("primary_reason")
        or viewpoint_result.get("summary")
        or "Experimental viewpoint evidence is unavailable."
    )

    if analysis_mode != "object_focused":
        return _unverified(
            "Capture coverage not verified because an object region was not tracked; whole-frame motion cannot establish which object views were captured.",
            analysis_mode=analysis_mode,
        )

    tracking_raw = viewpoint_result.get("object_tracking", {})
    tracking = tracking_raw if isinstance(tracking_raw, Mapping) else {}
    tracked_fraction = _number(tracking.get("tracked_sample_fraction"))
    reached_timeline_ends = bool(
        tracking.get("forward_reached_end") and tracking.get("backward_reached_start")
    )
    foreground_raw = tracking.get("foreground_separation", {})
    foreground = foreground_raw if isinstance(foreground_raw, Mapping) else {}
    if foreground and not bool(
        foreground.get("reliable_for_object_evidence", False)
    ):
        result = _unverified(
            "Capture coverage not verified because the selected foreground mask is unstable, incomplete, or background-contaminated.",
            analysis_mode=analysis_mode,
        )
        result["tracking_percent"] = (
            round(100.0 * tracked_fraction) if tracked_fraction is not None else None
        )
        return result
    if tracked_fraction is None or tracked_fraction < 0.70:
        result = _unverified(
            primary_reason,
            analysis_mode=analysis_mode,
        )
        result["tracking_percent"] = (
            round(100.0 * tracked_fraction) if tracked_fraction is not None else None
        )
        return result
    mode = _capture_mode(viewpoint_result)
    label = str(viewpoint_result.get("label", "Unknown"))
    if viewpoint_result.get("evidence_status") != "sufficient" or label == "Unknown":
        result = _unverified(primary_reason, analysis_mode=analysis_mode)
        result["tracking_percent"] = round(100.0 * tracked_fraction)
        if "Background-only" in primary_reason:
            result["suggestions"] = [
                "Object tracking was adequate, but background correspondences dominated; keep the object larger in frame and reduce distinctive background clutter.",
                "Use a plain contrasting background while preserving texture on the object itself.",
                "If upper surfaces matter, add a deliberate elevated pass; current evidence still cannot verify one.",
            ]
        elif "Descriptors are frequently ambiguous" in primary_reason:
            result["suggestions"] = [
                "Tracking was adequate, but repeated or symmetric appearance made object correspondences ambiguous.",
                "Improve diffuse lighting or add removable non-repeating reference detail when appropriate.",
                "Record overlapping lateral and elevated views without assuming that a repeated front view completes the capture.",
            ]
        elif "blurry" in primary_reason.lower():
            result["suggestions"] = [
                "Object tracking alone is not enough: record more slowly with sharper focus and brighter diffuse light.",
                "Repeat the lateral path after sharp object detail is visible in each view.",
            ]
        elif "global homography" in primary_reason.lower():
            result["suggestions"] = [
                "Image change is dominated by a planar/rotation model; it does not verify depth-bearing camera translation.",
                "With a rigid object, add slow lateral camera displacement while preserving overlap and sharp object detail.",
                "Do not interpret a rotating object or in-place camera pan as verified capture completeness.",
            ]
        elif "object animation" in primary_reason.lower() or "mixed" in primary_reason.lower():
            result["suggestions"] = [
                "Keep the object rigid and the background stable so independently moving regions do not mimic viewpoint change.",
                "Repeat a slow overlapping lateral path; current image motion cannot identify the physical motion source.",
            ]
        # A static-background/object-motion pattern can explain why coverage is not
        # verified while still being useful capture-mode guidance. It remains
        # explicitly compatible with either a turntable or animation.
        result["capture_mode"] = mode
        return result

    indicators_raw = viewpoint_result.get("indicators", {})
    indicators = indicators_raw if isinstance(indicators_raw, Mapping) else {}
    change = _number(
        indicators.get("temporally_separated_substantial_change_fraction")
    )
    parallax = _number(
        indicators.get("temporally_separated_parallax_proxy_pair_fraction")
    )
    if label == "Low":
        status = "limited_view_evidence"
        result_label = "Limited viewpoint variation"
        summary = (
            "The tracked object mostly repeats similar views or lacks sufficient "
            "long-gap parallax-like evidence."
        )
        variation = {"status": "limited", "label": "Limited"}
        suggestions = [
            "Add overlapping lateral views around the object instead of repeating the same frontal view.",
            "Keep the object large and sharp while moving in smaller, steady steps.",
            "If upper surfaces matter, add a deliberate elevated pass; the current evidence does not verify one.",
        ]
    else:
        status = "varied_view_evidence"
        result_label = "Multiple object viewpoints detected"
        breadth = "broader" if label == "High" else "partial"
        summary = (
            f"Reliable object tracking supports {breadth} viewpoint variation, "
            "but capture completeness and named object sides are not verified."
        )
        variation = {
            "status": "broader" if label == "High" else "partial",
            "label": "Broader variation" if label == "High" else "Partial variation",
        }
        suggestions = [
            "Preserve the overlapping path and add another deliberate elevation only if upper surfaces are required.",
            "Review the recording manually for task-critical surfaces; this evidence does not identify front, back, top, or bottom.",
        ]

    limitations = [
        "View variation is not exact orbit angle, 360-degree completion, or object-surface coverage.",
        "No specific front, back, top, bottom, or missing surface is identified.",
        "Reflective, symmetric, deforming, or weakly textured objects can make the evidence ambiguous.",
    ]
    if not reached_timeline_ends:
        limitations.append(
            "Object tracking did not reach both ends of the sampled timeline; view guidance uses only the reliably tracked interval."
        )

    return {
        "schema_version": "experimental_capture_completeness/1.0",
        "status": status,
        "label": result_label,
        "summary": summary,
        "tracking_percent": round(100.0 * tracked_fraction),
        "view_variation": {
            **variation,
            "substantial_change_percent": (
                round(100.0 * change) if change is not None else None
            ),
            "parallax_proxy_percent": (
                round(100.0 * parallax) if parallax is not None else None
            ),
        },
        "elevation_variation": {
            "status": "not_measured",
            "label": "Not verified",
            "reason": (
                "Existing matches do not provide a reliable object-relative "
                "higher/lower viewing-angle trajectory."
            ),
        },
        "capture_mode": mode,
        "suggestions": suggestions,
        "limitations": limitations,
        "method": {
            "name": "existing_object_viewpoint_evidence_summary",
            "analysis_mode": analysis_mode,
            "new_feature_matching_run": False,
            "affects_suitability_score": False,
            "affects_ridge_prediction": False,
            "affects_viewpoint_classification": False,
        },
    }

