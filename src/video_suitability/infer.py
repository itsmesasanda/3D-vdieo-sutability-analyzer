"""End-to-end heuristic inference API and command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TypeVar

from . import __version__
from .capture_completeness import analyze_capture_completeness
from .config import AnalysisConfig
from .geometry import analyze_geometry, detect_orb_features
from .interpretation import build_result_interpretation
from .ml.registration_rf import attach_registration_prediction
from .object_focus import analyze_object_focused_viewpoint, normalize_object_bbox
from .quality import analyze_quality
from .runtime_provenance import runtime_provenance
from .scoring import (
    build_problem_segments,
    recommendations_for,
    score_analysis_v1,
    score_analysis_v2,
)
from .video import VideoError, sample_video
from .viewpoint import analyze_viewpoint_diversity


T = TypeVar("T")


ANALYSIS_METHOD = {
    "name": "explainable_cpu_heuristic",
    "version": "2.1",
    "trained_model_used": False,
    "reconstruction_run": False,
    "components": [
        "uniform whole-video frame sampling",
        "exposure/contrast/blur/texture measurements",
        "ORB correspondence with RANSAC geometric verification",
        "timestamp-aware optical flow and track retention",
        "overlap, motion regularity, and cautious rotation/parallax proxies",
        "standalone multi-gap estimated viewpoint-diversity classification",
    ],
}


def analyze_video(
    video_path: str,
    config: Optional[AnalysisConfig] = None,
    camera_intrinsics: Optional[Mapping[str, float]] = None,
    object_bbox: Optional[Sequence[float]] = None,
    include_performance_profile: bool = False,
    include_viewpoint_diversity: bool = True,
) -> Dict[str, Any]:
    """Analyze a local video and return a JSON-serializable heuristic report."""

    cfg = config or AnalysisConfig()
    cfg.validate()
    normalized_object_bbox = (
        normalize_object_bbox(object_bbox) if object_bbox is not None else None
    )
    profile: Dict[str, Any] = {"stages": {}}
    total_wall_started = time.perf_counter()
    total_cpu_started = time.process_time()

    def measured(name: str, operation: Callable[[], T]) -> T:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        value = operation()
        profile["stages"][name] = {
            "wall_seconds": round(time.perf_counter() - wall_started, 6),
            "cpu_seconds": round(time.process_time() - cpu_started, 6),
        }
        return value

    sampled = measured("frame_decoding_and_sampling", lambda: sample_video(Path(video_path), cfg))
    quality = measured(
        "frame_quality",
        lambda: analyze_quality(
            sampled.frames, sampled.frame_indices, sampled.timestamps_seconds
        ),
    )
    detected_features = measured(
        "orb_feature_extraction",
        lambda: detect_orb_features(sampled.frames, cfg),
    )
    geometry = measured(
        "adjacent_geometry",
        lambda: analyze_geometry(
            sampled.frames,
            sampled.frame_indices,
            sampled.timestamps_seconds,
            cfg,
            detected_features=detected_features,
        ),
    )
    object_geometry: Optional[Dict[str, Any]] = None
    viewpoint_failure_reason: Optional[str] = None

    def unavailable_viewpoint(reason: str) -> Dict[str, Any]:
        return {
            "label": "Unknown",
            "evidence_status": "insufficient",
            "summary": reason,
            "method": {
                "analysis_mode": (
                    "object_focused"
                    if normalized_object_bbox is not None
                    else "whole_frame"
                ),
                "affects_overall_score": False,
                "trained_model_used": False,
                "colmap_run": False,
                "intrinsics_used": False,
            },
            "indicators": {},
            "decision_explanation": {
                "primary_reason": reason,
                "insufficient_evidence_reasons": [reason],
            },
            "recommendations": [
                "Enable the experimental viewpoint analysis only when its additional CPU cost is acceptable."
            ],
            "warnings": [reason],
            "repeated_view_intervals": [],
        }

    if not include_viewpoint_diversity:
        viewpoint_failure_reason = (
            "Experimental viewpoint-diversity analysis was not requested for this run."
        )
        viewpoint_diversity = unavailable_viewpoint(viewpoint_failure_reason)
    else:
        try:
            if normalized_object_bbox is None:
                viewpoint_diversity = measured(
                    "whole_frame_viewpoint",
                    lambda: analyze_viewpoint_diversity(
                        sampled.frames,
                        sampled.frame_indices,
                        sampled.timestamps_seconds,
                        quality["summary"],
                        geometry,
                        cfg,
                        detected_features=detected_features,
                        camera_intrinsics=camera_intrinsics,
                    ),
                )
            else:
                viewpoint_diversity, object_geometry = measured(
                    "object_tracking_and_viewpoint",
                    lambda: analyze_object_focused_viewpoint(
                        sampled.frames,
                        sampled.frame_indices,
                        sampled.timestamps_seconds,
                        quality["summary"],
                        geometry,
                        detected_features,
                        normalized_object_bbox,
                        cfg,
                        camera_intrinsics=camera_intrinsics,
                    ),
                )
        except Exception as exc:
            viewpoint_failure_reason = (
                "Experimental viewpoint analysis failed without affecting the "
                f"suitability score: {type(exc).__name__}: {exc}"
            )
            viewpoint_diversity = unavailable_viewpoint(viewpoint_failure_reason)

    def build_scores_and_segments() -> Any:
        baseline = score_analysis_v1(quality, geometry, len(sampled.frames), cfg)
        current = score_analysis_v2(
            quality, geometry, sampled.metadata, len(sampled.frames), cfg
        )
        segments = build_problem_segments(quality["frames"], geometry["pairs"])
        return baseline, current, segments, recommendations_for(segments)

    baseline_v1, score, problem_segments, recommendations = measured(
        "scoring_and_problem_segments", build_scores_and_segments
    )

    warnings = [
        "The overall score is a declared heuristic, not an ML prediction or calibrated probability.",
        "No 3D reconstruction was run; this tool estimates input suitability only.",
        "Motion and parallax measurements are image-space proxies and cannot prove camera translation or a static subject.",
        "Any separately attached ML prediction is not part of this heuristic score or its subscores.",
    ]
    warnings.extend(sampled.warnings)
    if viewpoint_failure_reason:
        warnings.append(viewpoint_failure_reason)
    geometry_summary = geometry["summary"]
    if float(geometry_summary["valid_geometry_pair_fraction"]) < 0.35:
        warnings.append(
            "Geometric evidence is sparse; the score relies more heavily on image-quality measurements."
        )
    if float(geometry_summary["rotation_or_planar_pair_fraction"]) >= 0.50:
        warnings.append(
            "Homography-dominant motion can indicate a planar subject or rotation-dominant capture; capture lateral viewpoints for stronger depth evidence."
        )
    if float(geometry_summary["low_overlap_pair_fraction"]) >= 0.50:
        warnings.append(
            "At least half of sampled transitions have low overlap; feature-rich individual frames may still fail to maintain reconstruction tracks."
        )
    if float(geometry_summary["irregular_camera_motion_pair_fraction"]) >= 0.40:
        warnings.append(
            "Timestamp-normalized optical flow shows frequent image-space direction or speed discontinuities. The physical source (camera motion, object rotation, or animation) is unresolved."
        )
    if float(
        geometry_summary["motion_sampling_gap_too_wide_pair_fraction"]
    ) > 0:
        warnings.append(
            "Some uniform sample gaps exceed 0.30 seconds; high-frequency shake and LK track retention are unresolved there, so ORB continuity is used as a fallback."
        )
    quality_summary = quality["summary"]
    if float(quality_summary.get("blur_and_low_texture_frame_fraction", 0.0)) > 0:
        warnings.append(
            "Some low-sharpness flags coincide with low texture. The Laplacian proxy cannot distinguish optical blur from a smooth object surface; inspect visible edges before retaking."
        )
    if max(
        float(quality_summary.get("underexposed_frame_fraction", 0.0)),
        float(quality_summary.get("overexposed_frame_fraction", 0.0)),
    ) > 0:
        warnings.append(
            "Exposure flags are whole-frame measurements. Object-only exposure is unavailable because no validated foreground mask was used."
        )
    duration = sampled.metadata.get("duration_seconds")
    if duration is not None and duration > 30 and len(sampled.frames) <= cfg.max_samples:
        warnings.append(
            f"A {duration:.1f}s video was represented by {len(sampled.frames)} samples; short defects between samples may be missed."
        )
    if duration is not None and duration < 3.0:
        warnings.append(
            f"The accessible file is only {duration:.2f}s long; whole-object viewpoint coverage is uncertain even when frame correspondences are strong."
        )
    source_width = sampled.metadata.get("source_width")
    source_height = sampled.metadata.get("source_height")
    if source_width and source_height and min(source_width, source_height) < 480:
        warnings.append(
            "The source has a short side below 480 pixels. Fine surface detail may be limited, but resolution alone did not trigger a score cap."
        )
    if normalized_object_bbox is not None:
        warnings.append(
            "The viewpoint category uses the optional selected object region; the suitability score still uses unchanged whole-frame metrics."
        )
        for reason in viewpoint_diversity.get("decision_explanation", {}).get(
            "insufficient_evidence_reasons", []
        ):
            warnings.append(f"Object-focused viewpoint evidence: {reason}")
    repeated_intervals = viewpoint_diversity.get("repeated_view_intervals", [])
    if isinstance(repeated_intervals, list) and repeated_intervals:
        warnings.append(
            "Repeated or near-identical long-gap views were detected. High adjacent overlap can support tracking, but it is not evidence of new object viewpoints or 3D coverage."
        )

    try:
        capture_completeness = analyze_capture_completeness(viewpoint_diversity)
    except Exception as exc:
        # This optional presentation-oriented experiment must never invalidate the
        # existing heuristic or viewpoint report.
        capture_completeness = {
            "schema_version": "experimental_capture_completeness/1.0",
            "status": "not_verified",
            "label": "Capture coverage not verified",
            "summary": (
                "Experimental capture guidance was unavailable without affecting "
                f"the existing results: {type(exc).__name__}."
            ),
            "view_variation": {"status": "not_verified", "label": "Not verified"},
            "elevation_variation": {
                "status": "not_measured",
                "label": "Not verified",
                "reason": "Directional coverage evidence is unavailable.",
            },
            "capture_mode": {
                "status": "uncertain",
                "label": "Camera orbit versus object rotation: uncertain",
                "reason": "Capture-mode evidence is unavailable.",
            },
            "suggestions": [],
            "limitations": [
                "No exact orbit angle, named side, or surface coverage is inferred."
            ],
            "method": {
                "name": "existing_object_viewpoint_evidence_summary",
                "affects_suitability_score": False,
                "affects_ridge_prediction": False,
                "affects_viewpoint_classification": False,
            },
        }

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "analysis_method": ANALYSIS_METHOD,
        "runtime_provenance": runtime_provenance(),
        "scoring_version": score["scoring_version"],
        "scoring_revision": score["scoring_revision"],
        "overall_score": score["overall_score"],
        "suitability_label": score["suitability_label"],
        "score_interpretation": {
            "likely_suitable_at_or_above": cfg.good_score_threshold,
            "review_recommended_at_or_above": cfg.review_score_threshold,
            "thresholds_are_calibrated": False,
        },
        "confidence_or_uncertainty": score["confidence_or_uncertainty"],
        "subscores": score["subscores"],
        "subscore_details": score["subscore_details"],
        "score_caps_applied": score["score_caps_applied"],
        "critical_failure_gate_evaluations": score[
            "critical_failure_gate_evaluations"
        ],
        "geometry_reliability": score["geometry_reliability"],
        "baseline_v1": {
            "scoring_version": "heuristic_v1",
            "overall_score": baseline_v1["overall_score"],
            "suitability_label": baseline_v1["suitability_label"],
            "subscores": baseline_v1["subscores"],
            "score_caps_applied": baseline_v1["score_caps_applied"],
        },
        "score_comparison": {
            "baseline_v1_score": baseline_v1["overall_score"],
            "heuristic_v2_score": score["overall_score"],
            "change_v2_minus_v1": round(
                score["overall_score"] - baseline_v1["overall_score"], 2
            ),
        },
        "raw_metrics": {
            "video": sampled.metadata,
            "source_media_diagnostics": {
                "source_width": source_width,
                "source_height": source_height,
                "source_megapixels": sampled.metadata.get("source_megapixels"),
                "codec": sampled.metadata.get("codec"),
                "estimated_bitrate_mbps": sampled.metadata.get(
                    "estimated_bitrate_mbps"
                ),
                "compressed_bits_per_pixel_per_frame": sampled.metadata.get(
                    "compressed_bits_per_pixel_per_frame"
                ),
                "analysis_linear_scale_from_source": sampled.metadata.get(
                    "analysis_linear_scale_from_source"
                ),
                "median_laplacian_variance_at_analysis_resolution": quality[
                    "summary"
                ]["median_laplacian_variance"],
                "blurry_frame_fraction": quality["summary"][
                    "blurry_frame_fraction"
                ],
                "resolution_alone_triggers_failure": False,
            },
            "quality_summary": quality["summary"],
            "geometry_summary": geometry["summary"],
            "sampled_frame_quality": quality["frames"],
            "sampled_frame_features": geometry["frames"],
            "sampled_pair_geometry": geometry["pairs"],
            "object_geometry_summary": (
                object_geometry["summary"] if object_geometry is not None else None
            ),
        },
        "problem_segments": problem_segments,
        "diagnostic_confidence": {
            "status_definitions": {
                "detected": "A declared image or frame-pair threshold was observed.",
                "suspected": "A proxy was observed, but the physical cause is ambiguous.",
                "not_measurable": "Sampling or evidence was insufficient; this is not a confirmed defect.",
            },
            "segment_counts": {
                status: sum(
                    segment.get("diagnostic_status") == status
                    for segment in problem_segments
                )
                for status in ("detected", "suspected", "not_measurable")
            },
            "sharpness_scope": "whole_frame_edge_detail_proxy",
            "exposure_scope": "whole_frame",
            "foreground_exposure_status": "not_available",
            "physical_motion_source_status": "not_identified",
            "affects_overall_score": False,
        },
        "recommendations": recommendations,
        "warnings": list(dict.fromkeys(warnings)),
        "viewpoint_diversity": viewpoint_diversity,
        "capture_completeness": capture_completeness,
    }
    report["result_interpretation"] = build_result_interpretation(report)
    if include_performance_profile:
        profile["total_wall_seconds"] = round(
            time.perf_counter() - total_wall_started, 6
        )
        profile["total_cpu_seconds"] = round(
            time.process_time() - total_cpu_started, 6
        )
        profile["scope"] = (
            "Production analysis only; upload I/O and diagnostic rendering are measured separately."
        )
        report["performance_profile"] = profile
    return report


def write_report(
    report: Dict[str, Any], output_path: Path, indent: Optional[int] = 2
) -> None:
    """Write a report as UTF-8 JSON, creating only the requested parent directory."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=indent, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-suitability",
        description=(
            "Score a mostly static, object-centric video for likely 3D reconstruction suitability "
            "using an explainable CPU heuristic."
        ),
    )
    parser.add_argument("--video", required=True, help="Path to a local MP4/MOV/AVI/etc. video")
    parser.add_argument("--out", help="JSON output path; omit to print JSON to stdout")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=AnalysisConfig.max_samples,
        help="Uniformly sampled frames to analyze (2-120; default: 32)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=AnalysisConfig.max_analysis_side,
        help="Maximum analysis-frame side in pixels (160-1920; default: 640)",
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    parser.add_argument("--camera-fx", type=float, help="Camera focal length fx in pixels")
    parser.add_argument("--camera-fy", type=float, help="Camera focal length fy in pixels")
    parser.add_argument("--camera-cx", type=float, help="Camera principal point cx in pixels")
    parser.add_argument("--camera-cy", type=float, help="Camera principal point cy in pixels")
    parser.add_argument(
        "--intrinsics-width",
        type=float,
        help="Image width for the supplied intrinsics; omit only if they already match analysis frames",
    )
    parser.add_argument(
        "--intrinsics-height",
        type=float,
        help="Image height for the supplied intrinsics; omit only if they already match analysis frames",
    )
    parser.add_argument(
        "--object-bbox",
        help="Optional normalized object box as x,y,width,height on the middle sampled frame",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Include per-stage wall and process CPU timings in the JSON report",
    )
    parser.add_argument(
        "--registration-artifact",
        type=Path,
        help=(
            "Optional parity-gated Random Forest artifact for a separate predicted "
            "COLMAP image-registration percentage"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = Path(args.out).expanduser().resolve() if args.out else None
        input_path = Path(args.video).expanduser().resolve()
        if output_path is not None and output_path == input_path:
            raise ValueError("--out must not overwrite the input video")
        config = AnalysisConfig(
            max_samples=args.max_samples,
            max_analysis_side=args.max_side,
        )
        intrinsic_values = [args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy]
        if any(value is not None for value in intrinsic_values) and not all(
            value is not None for value in intrinsic_values
        ):
            raise ValueError(
                "--camera-fx, --camera-fy, --camera-cx, and --camera-cy must be supplied together"
            )
        if (args.intrinsics_width is None) != (args.intrinsics_height is None):
            raise ValueError("--intrinsics-width and --intrinsics-height must be supplied together")
        if args.intrinsics_width is not None and not all(
            value is not None for value in intrinsic_values
        ):
            raise ValueError("intrinsic image dimensions require camera intrinsic values")
        camera_intrinsics = None
        if all(value is not None for value in intrinsic_values):
            camera_intrinsics = {
                "fx": args.camera_fx,
                "fy": args.camera_fy,
                "cx": args.camera_cx,
                "cy": args.camera_cy,
            }
            if args.intrinsics_width is not None:
                camera_intrinsics.update(
                    {
                        "image_width": args.intrinsics_width,
                        "image_height": args.intrinsics_height,
                    }
                )
        object_bbox = None
        if args.object_bbox:
            try:
                object_bbox = [
                    float(value.strip()) for value in args.object_bbox.split(",")
                ]
            except ValueError as exc:
                raise ValueError(
                    "--object-bbox must be four comma-separated numbers"
                ) from exc
            object_bbox = list(normalize_object_bbox(object_bbox))
        report = analyze_video(
            args.video,
            config=config,
            camera_intrinsics=camera_intrinsics,
            object_bbox=object_bbox,
            include_performance_profile=args.profile,
        )
        if args.registration_artifact is not None:
            report = attach_registration_prediction(
                report,
                Path(args.video),
                artifact_dir=args.registration_artifact,
            )
        indent = None if args.compact else 2
        if output_path is not None:
            write_report(report, output_path, indent=indent)
            print(
                f"Wrote {args.out} | heuristic score {report['overall_score']:.2f}/100 "
                f"| {report['suitability_label']}",
                file=sys.stderr,
            )
        else:
            print(json.dumps(report, indent=indent, allow_nan=False))
        return 0
    except (VideoError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

