#!/usr/bin/env python3
"""Validate Estimated Viewpoint Diversity on a folder of local real videos."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from scripts.validate_real_videos import find_videos, sha256_file
except ModuleNotFoundError:  # direct `python scripts/...py` execution
    from validate_real_videos import find_videos, sha256_file
from video_suitability.config import AnalysisConfig
from video_suitability.geometry import detect_orb_features
from video_suitability.infer import analyze_video, write_report
from video_suitability.video import sample_video
from video_suitability.viewpoint_diagnostics import (
    render_pair_diagnostic,
    select_diagnostic_pairs,
)


CSV_COLUMNS = [
    "filename",
    "processing_status",
    "predicted_viewpoint_category",
    "confidence_status",
    "confidence_note",
    "overall_suitability_score_unchanged",
    "suitability_label_unchanged",
    "analysis_wall_seconds",
    "analysis_cpu_seconds",
    "diagnostics_wall_seconds",
    "diagnostics_cpu_seconds",
    "total_processing_wall_seconds",
    "total_processing_cpu_seconds",
    "source_sha256",
    "source_unchanged_after_processing",
    "duration_seconds",
    "decoded_sample_count",
    "nearby_pair_count",
    "temporally_separated_pair_count",
    "valid_geometry_pair_count",
    "valid_geometry_pair_fraction",
    "median_nearby_ratio_test_matches",
    "median_nearby_verified_matches",
    "median_separated_verified_matches",
    "all_pair_parallax_proxy_fraction",
    "separated_parallax_proxy_pair_count",
    "separated_parallax_proxy_pair_fraction",
    "intrinsics_used",
    "median_rotation_compensated_parallax_degrees",
    "repeated_view_pair_count",
    "temporally_separated_repeated_view_fraction",
    "endpoint_repeated_view_detected",
    "strong_planar_or_rotation_fraction_among_changed_pairs",
    "median_within_frame_duplicate_descriptor_fraction",
    "median_separated_strict_descriptor_ambiguity_fraction",
    "diagnostic_image_count",
    "diagnostic_images_json",
    "misleading_risk_flags_json",
    "warnings_json",
    "human_review_context",
    "error",
]


RISK_EXPLANATIONS = {
    "background_feature_dominance_unresolved": (
        "ORB matches are not segmented to the object, so a textured static background "
        "can dominate geometry and inflate apparent evidence."
    ),
    "object_motion_unresolved": (
        "The method assumes a static object and cannot reliably separate object motion "
        "from camera motion."
    ),
    "symmetry_or_repetitive_appearance": (
        "Repeated or symmetric descriptors can support geometrically plausible but "
        "incorrect correspondences."
    ),
    "rotation_or_planar_motion": (
        "A global homography explains much of the changed footage; camera rotation or "
        "a planar scene can mimic viewpoint change without useful translation."
    ),
    "intrinsics_unavailable_for_pose_disambiguation": (
        "No calibrated intrinsics were supplied, so essential-matrix pose and "
        "rotation-compensated parallax are unavailable."
    ),
    "intrinsics_accuracy_unverified": (
        "Supplied intrinsics are trusted as metadata; an incorrect focal length, crop, "
        "stabilization transform, or principal point can bias pose/parallax indicators."
    ),
    "sparse_or_unstable_geometry": (
        "Too few pairs have stable verified geometry for a confident diversity result."
    ),
}


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def load_intrinsics_manifest(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """Load optional filename-keyed pinhole intrinsics without guessing values."""

    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intrinsics manifest must be a JSON object keyed by filename")
    output: Dict[str, Dict[str, float]] = {}
    for filename, values in payload.items():
        if not isinstance(filename, str) or not isinstance(values, dict):
            raise ValueError("each intrinsics manifest entry must map a filename to an object")
        output[filename] = {str(key): float(value) for key, value in values.items()}
    return output


def load_review_context(path: Optional[Path]) -> Dict[str, str]:
    """Load qualitative review text only; it is not treated as viewpoint truth."""

    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        filename = row.get("filename", "").strip()
        if not filename:
            continue
        parts = [
            row.get("human_qualitative_assessment", "").replace("_", " ").strip(),
            row.get("camera_speed_feedback", "").replace("_", " ").strip(),
            row.get("camera_shake_feedback", "").replace("_", " ").strip(),
            row.get("review_notes", "").strip(),
        ]
        output[filename] = " | ".join(part for part in parts if part)
    return output


def misleading_risk_flags(viewpoint: Mapping[str, Any]) -> List[str]:
    """Return explicit ways this category could be misleading for manual review."""

    indicators = viewpoint.get("indicators", {})
    flags = ["object_motion_unresolved"]
    if float(indicators.get("valid_geometry_pair_fraction") or 0.0) >= 0.35:
        flags.append("background_feature_dominance_unresolved")
    else:
        flags.append("sparse_or_unstable_geometry")
    if (
        float(
            indicators.get("median_within_frame_duplicate_descriptor_fraction")
            or 0.0
        )
        >= 0.10
        or float(
            indicators.get(
                "median_separated_strict_descriptor_ambiguity_fraction"
            )
            or 0.0
        )
        >= 0.65
    ):
        flags.append("symmetry_or_repetitive_appearance")
    if (
        float(
            indicators.get(
                "strong_planar_or_rotation_fraction_among_changed_pairs"
            )
            or 0.0
        )
        >= 0.50
    ):
        flags.append("rotation_or_planar_motion")
    if viewpoint.get("method", {}).get("intrinsics_used"):
        flags.append("intrinsics_accuracy_unverified")
    else:
        flags.append("intrinsics_unavailable_for_pose_disambiguation")
    return list(dict.fromkeys(flags))


def _safe_stem(path: Path) -> str:
    return "".join(character if character.isalnum() else "_" for character in path.name)


def write_diagnostics(
    video_path: Path,
    report: Mapping[str, Any],
    output_dir: Path,
    config: AnalysisConfig,
    maximum_pairs: int,
) -> List[Dict[str, Any]]:
    """Re-sample the same bounded timeline and write selected pair overlays."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for diagnostic images") from exc
    sampled = sample_video(video_path, config)
    keypoints, descriptors = detect_orb_features(sampled.frames, config)
    selected = select_diagnostic_pairs(
        report["viewpoint_diversity"], maximum_pairs=maximum_pairs
    )
    video_dir = output_dir / _safe_stem(video_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Dict[str, Any]] = []
    for order, item in enumerate(selected, start=1):
        pair = item["pair"]
        start = int(pair["sample_position_start"])
        end = int(pair["sample_position_end"])
        image, metadata = render_pair_diagnostic(
            sampled.frames[start],
            sampled.frames[end],
            keypoints[start],
            keypoints[end],
            descriptors[start],
            descriptors[end],
            pair,
            item["reason"],
            config,
        )
        filename = (
            f"{order:02d}_{item['reason']}_"
            f"{int(pair['start_frame_index']):06d}_{int(pair['end_frame_index']):06d}.jpg"
        )
        target = video_dir / filename
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"OpenCV could not write diagnostic image: {target}")
        artifacts.append(
            {
                **metadata,
                "path": str(target),
                "start_frame_index": int(pair["start_frame_index"]),
                "end_frame_index": int(pair["end_frame_index"]),
                "start_time_seconds": pair["start_time_seconds"],
                "end_time_seconds": pair["end_time_seconds"],
            }
        )
    return artifacts


def flatten_success(
    path: Path,
    report: Mapping[str, Any],
    context: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    risk_flags: Sequence[str],
    human_context: str,
) -> Dict[str, Any]:
    viewpoint = report["viewpoint_diversity"]
    indicators = viewpoint["indicators"]
    geometry = report["raw_metrics"]["geometry_summary"]
    video = report["raw_metrics"]["video"]
    separated_pairs = viewpoint["temporally_separated_pair_evidence"]
    return {
        "filename": path.name,
        "processing_status": "success",
        "predicted_viewpoint_category": viewpoint["label"],
        "confidence_status": viewpoint["evidence_status"],
        "confidence_note": (
            "heuristic evidence only; not calibrated confidence or viewpoint ground truth"
        ),
        "overall_suitability_score_unchanged": report["overall_score"],
        "suitability_label_unchanged": report["suitability_label"],
        "analysis_wall_seconds": context["analysis_wall_seconds"],
        "analysis_cpu_seconds": context["analysis_cpu_seconds"],
        "diagnostics_wall_seconds": context["diagnostics_wall_seconds"],
        "diagnostics_cpu_seconds": context["diagnostics_cpu_seconds"],
        "total_processing_wall_seconds": context["total_processing_wall_seconds"],
        "total_processing_cpu_seconds": context["total_processing_cpu_seconds"],
        "source_sha256": context["source_sha256"],
        "source_unchanged_after_processing": context[
            "source_unchanged_after_processing"
        ],
        "duration_seconds": video.get("duration_seconds"),
        "decoded_sample_count": video.get("decoded_sample_count"),
        "nearby_pair_count": indicators["nearby_pair_count"],
        "temporally_separated_pair_count": indicators[
            "temporally_separated_pair_count"
        ],
        "valid_geometry_pair_count": round(
            float(indicators["valid_geometry_pair_fraction"])
            * (
                int(indicators["nearby_pair_count"])
                + int(indicators["temporally_separated_pair_count"])
            )
        ),
        "valid_geometry_pair_fraction": indicators["valid_geometry_pair_fraction"],
        "median_nearby_ratio_test_matches": geometry["median_ratio_test_matches"],
        "median_nearby_verified_matches": geometry["median_verified_matches"],
        "median_separated_verified_matches": indicators[
            "median_separated_verified_matches"
        ],
        "all_pair_parallax_proxy_fraction": indicators[
            "parallax_proxy_pair_fraction"
        ],
        "separated_parallax_proxy_pair_count": sum(
            bool(pair.get("parallax_proxy_detected")) for pair in separated_pairs
        ),
        "separated_parallax_proxy_pair_fraction": indicators[
            "temporally_separated_parallax_proxy_pair_fraction"
        ],
        "intrinsics_used": viewpoint["method"]["intrinsics_used"],
        "median_rotation_compensated_parallax_degrees": indicators[
            "median_rotation_compensated_parallax_degrees"
        ],
        "repeated_view_pair_count": len(viewpoint["repeated_view_intervals"]),
        "temporally_separated_repeated_view_fraction": indicators[
            "temporally_separated_repeated_view_fraction"
        ],
        "endpoint_repeated_view_detected": indicators[
            "endpoint_repeated_view_detected"
        ],
        "strong_planar_or_rotation_fraction_among_changed_pairs": indicators[
            "strong_planar_or_rotation_fraction_among_changed_pairs"
        ],
        "median_within_frame_duplicate_descriptor_fraction": indicators[
            "median_within_frame_duplicate_descriptor_fraction"
        ],
        "median_separated_strict_descriptor_ambiguity_fraction": indicators[
            "median_separated_strict_descriptor_ambiguity_fraction"
        ],
        "diagnostic_image_count": len(diagnostics),
        "diagnostic_images_json": _json_cell(list(diagnostics)),
        "misleading_risk_flags_json": _json_cell(list(risk_flags)),
        "warnings_json": _json_cell(viewpoint["warnings"]),
        "human_review_context": human_context,
        "error": "",
    }


def _failure_row(path: Path, error: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "filename": path.name,
            "processing_status": "failed",
            "total_processing_wall_seconds": context.get(
                "total_processing_wall_seconds", ""
            ),
            "total_processing_cpu_seconds": context.get(
                "total_processing_cpu_seconds", ""
            ),
            "source_sha256": context.get("source_sha256", ""),
            "source_unchanged_after_processing": context.get(
                "source_unchanged_after_processing", ""
            ),
            "error": error,
        }
    )
    return row


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_validation_report(
    rows: Sequence[Mapping[str, Any]], output_path: Path, input_dir: Path
) -> None:
    successes = [row for row in rows if row["processing_status"] == "success"]
    failures = [row for row in rows if row["processing_status"] != "success"]
    lines = [
        "# Real-world Estimated Viewpoint Diversity validation",
        "",
        "This is a small, non-held-out diagnostic audit. The videos have independent "
        "capture-quality comments, but no verified Low/Moderate/High viewpoint-diversity "
        "ground truth. The results therefore do not establish real-world accuracy.",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "## Results",
        "",
        "| Video | Viewpoint | Evidence | Valid geometry | Median verified nearby / separated | Parallax proxy | Repeated pairs | Planar/rotation | CPU seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in successes:
        lines.append(
            "| {name} | {category} | {evidence} | {valid} | {nearby} / {separated} | {parallax} | {repeated} | {planar} | {cpu} |".format(
                name=row["filename"],
                category=row["predicted_viewpoint_category"],
                evidence=row["confidence_status"],
                valid=_fmt(row["valid_geometry_pair_fraction"]),
                nearby=_fmt(row["median_nearby_verified_matches"], 1),
                separated=_fmt(row["median_separated_verified_matches"], 1),
                parallax=_fmt(row["separated_parallax_proxy_pair_fraction"]),
                repeated=row["repeated_view_pair_count"],
                planar=_fmt(
                    row[
                        "strong_planar_or_rotation_fraction_among_changed_pairs"
                    ]
                ),
                cpu=_fmt(row["total_processing_cpu_seconds"]),
            )
        )
    if not successes:
        lines.append("| _No successful analyses_ | | | | | | | | |")
    lines.extend(
        [
            "",
            "## Potentially misleading cases and misclassification review",
            "",
            "A viewpoint category is not the overall suitability score. Fast motion, "
            "shake, blur, or low overlap can coexist with broad viewpoint change. Such a "
            "case is not automatically a diversity-category error, but it is a likely "
            "user-interpretation error if `High` is read as 'good capture'.",
            "",
        ]
    )
    for row in successes:
        flags = json.loads(str(row["misleading_risk_flags_json"]))
        lines.extend([f"### {row['filename']}", ""])
        lines.append(
            f"- Predicted `{row['predicted_viewpoint_category']}` with `{row['confidence_status']}` heuristic evidence."
        )
        if row.get("human_review_context"):
            lines.append(
                "- Independent capture-quality context (not viewpoint ground truth): "
                + str(row["human_review_context"])
            )
        for flag in flags:
            lines.append(f"- `{flag}`: {RISK_EXPLANATIONS[flag]}")
        diagnostics = json.loads(str(row["diagnostic_images_json"]))
        if diagnostics:
            lines.append("- Diagnostic images:")
            for diagnostic in diagnostics:
                relative = Path(diagnostic["path"]).relative_to(output_path.parent)
                lines.append(
                    f"  - [{diagnostic['selection_reason']}]({relative.as_posix()}) — "
                    f"{diagnostic['verified_matches']} verified matches."
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- ORB/RANSAC can lock onto the background rather than the object; the current "
            "pipeline has no object mask.",
            "- Symmetric and repeated surfaces can create internally consistent false matches.",
            "- Rotation-only and planar motion may fit a homography and do not prove camera translation.",
            "- Intrinsics are optional and never guessed. If provided, crop, stabilization, "
            "digital zoom, or inaccurate calibration can bias essential-matrix results.",
            "- Repeated endpoints may be loop closure or a repeated pass; they do not prove a full orbit.",
            "- No surface-coverage percentage or exact orbit angle is inferred.",
            "",
            f"Failures: {len(failures)}. Successful videos: {len(successes)}.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(
    input_dir: Path,
    output_dir: Path,
    config: AnalysisConfig,
    *,
    maximum_diagnostic_pairs: int = 3,
    intrinsics_manifest: Optional[Path] = None,
    review_csv: Optional[Path] = None,
) -> int:
    videos = find_videos(input_dir)
    if not videos:
        raise RuntimeError(f"No supported video files found under: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    diagnostic_dir = output_dir / "diagnostics"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = load_intrinsics_manifest(intrinsics_manifest)
    reviews = load_review_context(review_csv)
    rows: List[Dict[str, Any]] = []
    for path in videos:
        total_wall_started = time.perf_counter()
        total_cpu_started = time.process_time()
        context: Dict[str, Any] = {}
        try:
            before_hash = sha256_file(path)
            context["source_sha256"] = before_hash
            analysis_wall_started = time.perf_counter()
            analysis_cpu_started = time.process_time()
            report = analyze_video(
                str(path),
                config=config,
                camera_intrinsics=intrinsics.get(path.name),
            )
            context["analysis_wall_seconds"] = round(
                time.perf_counter() - analysis_wall_started, 6
            )
            context["analysis_cpu_seconds"] = round(
                time.process_time() - analysis_cpu_started, 6
            )
            diagnostics_wall_started = time.perf_counter()
            diagnostics_cpu_started = time.process_time()
            diagnostics = write_diagnostics(
                path,
                report,
                diagnostic_dir,
                config,
                maximum_pairs=maximum_diagnostic_pairs,
            )
            context["diagnostics_wall_seconds"] = round(
                time.perf_counter() - diagnostics_wall_started, 6
            )
            context["diagnostics_cpu_seconds"] = round(
                time.process_time() - diagnostics_cpu_started, 6
            )
            after_hash = sha256_file(path)
            context["source_unchanged_after_processing"] = before_hash == after_hash
            if not context["source_unchanged_after_processing"]:
                raise RuntimeError("source hash changed during read-only validation")
            context["total_processing_wall_seconds"] = round(
                time.perf_counter() - total_wall_started, 6
            )
            context["total_processing_cpu_seconds"] = round(
                time.process_time() - total_cpu_started, 6
            )
            report["viewpoint_validation_context"] = {
                **context,
                "filename": path.name,
                "diagnostics": diagnostics,
                "config": {
                    "max_samples": config.max_samples,
                    "max_analysis_side": config.max_analysis_side,
                    "maximum_diagnostic_pairs": maximum_diagnostic_pairs,
                },
            }
            write_report(report, report_dir / f"{path.name}.json")
            flags = misleading_risk_flags(report["viewpoint_diversity"])
            row = flatten_success(
                path,
                report,
                context,
                diagnostics,
                flags,
                reviews.get(path.name, ""),
            )
            rows.append(row)
            print(
                f"OK {path.name}: viewpoint={row['predicted_viewpoint_category']} "
                f"evidence={row['confidence_status']} "
                f"cpu={row['total_processing_cpu_seconds']:.3f}s "
                f"wall={row['total_processing_wall_seconds']:.3f}s"
            )
        except Exception as exc:  # continue to audit every accessible file
            context["total_processing_wall_seconds"] = round(
                time.perf_counter() - total_wall_started, 6
            )
            context["total_processing_cpu_seconds"] = round(
                time.process_time() - total_cpu_started, 6
            )
            if path.exists() and "source_sha256" in context:
                context["source_unchanged_after_processing"] = (
                    sha256_file(path) == context["source_sha256"]
                )
            error = f"{type(exc).__name__}: {exc}"
            rows.append(_failure_row(path, error, context))
            print(f"FAILED {path.name}: {error}")
    csv_path = output_dir / "validation.csv"
    _write_csv(rows, csv_path)
    write_validation_report(rows, output_dir / "REPORT.md", input_dir)
    failures = sum(row["processing_status"] != "success" for row in rows)
    print(
        f"Processed {len(rows)} videos: {len(rows) - failures} succeeded, "
        f"{failures} failed. CSV: {csv_path}"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-diagnostic-pairs", type=int, default=3)
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        help="Optional JSON object keyed by filename; intrinsics are never guessed",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Optional qualitative review CSV; never treated as viewpoint ground truth",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_diagnostic_pairs < 1 or args.max_diagnostic_pairs > 8:
        print("error: --max-diagnostic-pairs must be between 1 and 8")
        return 2
    config = AnalysisConfig(
        max_samples=args.max_samples,
        max_analysis_side=args.max_side,
    )
    try:
        config.validate()
        return run_validation(
            args.input_dir,
            args.output_dir,
            config,
            maximum_diagnostic_pairs=args.max_diagnostic_pairs,
            intrinsics_manifest=args.intrinsics_json,
            review_csv=args.review_csv,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

