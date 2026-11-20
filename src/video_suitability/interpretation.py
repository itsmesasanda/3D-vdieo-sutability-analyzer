"""Conservative interpretation of heuristic and experimental evidence.

This module deliberately does not calculate or modify the heuristic score.  It
only states what the already-computed evidence can and cannot support.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_result_interpretation(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Separate technical capture quality from geometric readiness.

    The current pipeline never runs a reconstruction, so even sufficient
    experimental viewpoint evidence cannot verify reconstruction readiness.
    """

    score = float(report.get("overall_score", 0.0))
    if score >= 70.0:
        technical_level = "strong"
        technical_label = "Strong technical capture quality"
    elif score >= 45.0:
        technical_level = "mixed"
        technical_label = "Mixed technical capture quality"
    else:
        technical_level = "limited"
        technical_label = "Limited technical capture quality"

    viewpoint = _mapping(report.get("viewpoint_diversity"))
    capture = _mapping(report.get("capture_completeness"))
    viewpoint_label = str(viewpoint.get("label", "Unknown"))
    viewpoint_status = str(viewpoint.get("evidence_status", "insufficient"))
    capture_status = str(capture.get("status", "not_verified"))

    if viewpoint_status == "sufficient" and viewpoint_label in {
        "Moderate",
        "High",
    }:
        geometry_evidence = "experimental_support"
        geometry_explanation = (
            "Experimental image geometry supports some viewpoint variation, but "
            "no reconstruction was run and object-surface coverage is not verified."
        )
    elif viewpoint_label == "Low" and viewpoint_status == "sufficient":
        geometry_evidence = "limited"
        geometry_explanation = (
            "Experimental image geometry found limited viewpoint variation. It does "
            "not establish a useful camera baseline or object coverage."
        )
    else:
        geometry_evidence = "insufficient"
        geometry_explanation = (
            "The experimental geometry is unavailable, ambiguous, or insufficient; "
            "reconstruction readiness cannot be inferred from the technical score."
        )

    return {
        "technical_capture_level": technical_level,
        "technical_capture_label": technical_label,
        "headline": f"{technical_label} — geometric readiness unverified",
        "geometric_readiness": "unverified",
        "geometry_evidence": geometry_evidence,
        "geometry_explanation": geometry_explanation,
        "viewpoint_category": viewpoint_label,
        "viewpoint_evidence_status": viewpoint_status,
        "capture_completeness_status": capture_status,
        "score_measures": (
            "The numerical heuristic summarizes sampled image quality, whole-frame "
            "exposure, feature trackability, RANSAC consistency, overlap and "
            "image-space motion proxies."
        ),
        "score_does_not_measure": (
            "It does not verify camera translation, object rigidity, named-side or "
            "surface coverage, COLMAP registration, or successful 3D reconstruction."
        ),
        "affects_overall_score": False,
    }

