#!/usr/bin/env python3
"""Batch real-video validation without modifying source media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video, write_report


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}

SUBSCORE_NAMES = [
    "visual_quality",
    "exposure_and_lighting",
    "feature_trackability",
    "geometric_consistency",
    "viewpoint_motion",
]

SUMMARY_COLUMNS = [
    "filename",
    "processing_status",
    "analysis_seconds",
    "file_size_bytes",
    "source_sha256",
    "source_unchanged_after_processing",
    "codec",
    "fps",
    "frame_count",
    "duration_seconds",
    "source_width",
    "source_height",
    "analysis_width",
    "analysis_height",
    "decoded_sample_count",
    "full_temporal_coverage_confirmed",
    "scoring_version",
    "baseline_v1_score",
    "overall_score",
    "change_v2_minus_v1",
    "suitability_label",
    "geometry_reliability_level",
    *[f"subscore_{name}" for name in SUBSCORE_NAMES],
    "median_luminance",
    "median_contrast_std",
    "median_laplacian_variance",
    "underexposed_frame_fraction",
    "overexposed_frame_fraction",
    "blurry_frame_fraction",
    "low_texture_frame_fraction",
    "median_keypoints_per_frame",
    "median_ratio_test_matches",
    "median_verified_matches",
    "median_inlier_ratio",
    "median_spatial_coverage",
    "valid_geometry_pair_fraction",
    "median_displacement_normalized",
    "near_stationary_pair_fraction",
    "low_overlap_pair_fraction",
    "homography_dominant_pair_fraction",
    "rotation_or_planar_pair_fraction",
    "median_flow_track_retention",
    "median_flow_track_retention_short_gap",
    "optical_flow_short_gap_pair_fraction",
    "motion_sampling_gap_too_wide_pair_fraction",
    "median_flow_speed_normalized_per_second",
    "excessive_camera_speed_pair_fraction",
    "irregular_camera_motion_pair_fraction",
    "flow_speed_robust_cv",
    "median_descriptor_ambiguity_fraction",
    "critical_failure_gates_json",
    "detected_problem_types_json",
    "detected_problems_json",
    "warnings_json",
    "raw_metrics_json",
    "error",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_videos(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def flatten_report(report: Dict[str, Any]) -> Dict[str, Any]:
    context = report["validation_context"]
    video = report["raw_metrics"]["video"]
    quality = report["raw_metrics"]["quality_summary"]
    geometry = report["raw_metrics"]["geometry_summary"]
    problem_types = sorted(
        {segment["problem_type"] for segment in report["problem_segments"]}
    )
    row: Dict[str, Any] = {
        "filename": context["filename"],
        "processing_status": "success",
        "analysis_seconds": context["analysis_seconds"],
        "file_size_bytes": video["file_size_bytes"],
        "source_sha256": context["source_sha256"],
        "source_unchanged_after_processing": context[
            "source_unchanged_after_processing"
        ],
        "codec": video["codec"],
        "fps": video["fps"],
        "frame_count": video["frame_count"],
        "duration_seconds": video["duration_seconds"],
        "source_width": video["source_width"],
        "source_height": video["source_height"],
        "analysis_width": video["analysis_width"],
        "analysis_height": video["analysis_height"],
        "decoded_sample_count": video["decoded_sample_count"],
        "full_temporal_coverage_confirmed": video[
            "full_temporal_coverage_confirmed"
        ],
        "scoring_version": report.get("scoring_version", "heuristic_v1"),
        "baseline_v1_score": report.get("baseline_v1", {}).get(
            "overall_score", report["overall_score"]
        ),
        "overall_score": report["overall_score"],
        "change_v2_minus_v1": report.get("score_comparison", {}).get(
            "change_v2_minus_v1", 0.0
        ),
        "suitability_label": report["suitability_label"],
        "geometry_reliability_level": report.get("geometry_reliability", {}).get(
            "level", report["confidence_or_uncertainty"]["evidence_level"]
        ),
        "critical_failure_gates_json": _json_cell(
            report.get("critical_failure_gate_evaluations", [])
        ),
        "detected_problem_types_json": _json_cell(problem_types),
        "detected_problems_json": _json_cell(report["problem_segments"]),
        "warnings_json": _json_cell(report["warnings"]),
        "raw_metrics_json": _json_cell(report["raw_metrics"]),
        "error": "",
    }
    for name in SUBSCORE_NAMES:
        row[f"subscore_{name}"] = report["subscores"][name]
    for name in [
        "median_luminance",
        "median_contrast_std",
        "median_laplacian_variance",
        "underexposed_frame_fraction",
        "overexposed_frame_fraction",
        "blurry_frame_fraction",
        "low_texture_frame_fraction",
    ]:
        row[name] = quality[name]
    for name in [
        "median_keypoints_per_frame",
        "median_ratio_test_matches",
        "median_verified_matches",
        "median_inlier_ratio",
        "median_spatial_coverage",
        "valid_geometry_pair_fraction",
        "median_displacement_normalized",
        "near_stationary_pair_fraction",
        "low_overlap_pair_fraction",
        "homography_dominant_pair_fraction",
        "rotation_or_planar_pair_fraction",
        "median_flow_track_retention",
        "median_flow_track_retention_short_gap",
        "optical_flow_short_gap_pair_fraction",
        "motion_sampling_gap_too_wide_pair_fraction",
        "median_flow_speed_normalized_per_second",
        "excessive_camera_speed_pair_fraction",
        "irregular_camera_motion_pair_fraction",
        "flow_speed_robust_cv",
        "median_descriptor_ambiguity_fraction",
    ]:
        row[name] = geometry[name]
    return row


def failed_row(
    path: Path,
    elapsed: float,
    source_hash: str,
    unchanged: bool,
    error: str,
) -> Dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row.update(
        {
            "filename": path.name,
            "processing_status": "failed",
            "analysis_seconds": round(elapsed, 3),
            "file_size_bytes": path.stat().st_size,
            "source_sha256": source_hash,
            "source_unchanged_after_processing": unchanged,
            "error": error,
        }
    )
    return row


def risk_notes(report: Dict[str, Any]) -> List[str]:
    """Identify review risks from metrics, never from the filename."""

    score = float(report["overall_score"])
    subscores = report["subscores"]
    geometry = report["raw_metrics"]["geometry_summary"]
    notes: List[str] = []
    triggered_gates = [
        gate["gate_id"]
        for gate in report.get("critical_failure_gate_evaluations", [])
        if gate["triggered"]
    ]
    if triggered_gates:
        notes.append(
            "Triggered v2 reliability gates: " + ", ".join(triggered_gates) + "."
        )
    if score >= 70 and float(geometry["rotation_or_planar_pair_fraction"]) >= 0.5:
        notes.append(
            "Potential false-positive risk: the high score coexists with mostly homography-dominant motion, which may be planar or rotation-heavy rather than useful translation."
        )
    if score >= 70 and float(subscores["viewpoint_motion"]) < 50:
        notes.append(
            "Potential false-positive risk: overall evidence is strong, but the viewpoint-motion subscore is below 50."
        )
    low_overlap_fraction = float(geometry["low_overlap_pair_fraction"])
    if score >= 70 and (
        float(subscores["geometric_consistency"]) < 50
        or low_overlap_fraction >= 0.50
    ):
        notes.append(
            "Potential false-positive risk: the likely-suitable label coexists with weak geometric consistency or low overlap on at least half of sampled pairs."
        )
    elif score >= 70 and low_overlap_fraction >= 0.30:
        notes.append(
            "Review caution: low overlap was detected on at least 30% of sampled pairs despite the likely-suitable label."
        )
    appearance_strength = min(
        float(subscores["visual_quality"]),
        float(subscores["exposure_and_lighting"]),
        float(subscores["feature_trackability"]),
    )
    geometric_weakness = min(
        float(subscores["geometric_consistency"]),
        float(subscores["viewpoint_motion"]),
    )
    if score < 70 and appearance_strength >= 70 and geometric_weakness < 45:
        notes.append(
            "Potential false-negative risk: appearance/trackability are strong, while the result is driven down by image-space geometry or motion proxies that need human verification."
        )
    if not notes:
        notes.append(
            "No specific false-positive/false-negative signature was triggered, but reconstruction outcome remains unverified."
        )
    return notes


def write_summary_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_human_review_template(paths: Sequence[Path], path: Path) -> None:
    columns = [
        "filename",
        "object",
        "camera_motion",
        "lighting",
        "blur",
        "expected_suitability",
        "notes",
        "reviewer",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for source in paths:
            writer.writerow({"filename": source.name})


def write_before_after_components(
    reports: Sequence[Dict[str, Any]], path: Path
) -> None:
    columns = [
        "filename",
        "baseline_v1_score",
        "heuristic_v2_score",
        "change_v2_minus_v1",
        "baseline_v1_label",
        "heuristic_v2_label",
    ]
    for name in SUBSCORE_NAMES:
        columns.extend(
            [
                f"baseline_v1_{name}",
                f"heuristic_v2_{name}",
                f"change_{name}",
            ]
        )
    columns.extend(
        [
            "geometry_reliability_level",
            "triggered_gate_ids_json",
            "v2_warnings_json",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            baseline = report.get("baseline_v1", {})
            row: Dict[str, Any] = {
                "filename": report["validation_context"]["filename"],
                "baseline_v1_score": baseline.get(
                    "overall_score", report["overall_score"]
                ),
                "heuristic_v2_score": report["overall_score"],
                "change_v2_minus_v1": report.get("score_comparison", {}).get(
                    "change_v2_minus_v1", 0.0
                ),
                "baseline_v1_label": baseline.get(
                    "suitability_label", report["suitability_label"]
                ),
                "heuristic_v2_label": report["suitability_label"],
                "geometry_reliability_level": report.get(
                    "geometry_reliability", {}
                ).get("level", ""),
                "triggered_gate_ids_json": _json_cell(
                    [
                        gate["gate_id"]
                        for gate in report.get(
                            "critical_failure_gate_evaluations", []
                        )
                        if gate["triggered"]
                    ]
                ),
                "v2_warnings_json": _json_cell(report["warnings"]),
            }
            for name in SUBSCORE_NAMES:
                old = float(baseline.get("subscores", report["subscores"])[name])
                new = float(report["subscores"][name])
                row[f"baseline_v1_{name}"] = old
                row[f"heuristic_v2_{name}"] = new
                row[f"change_{name}"] = round(new - old, 2)
            writer.writerow(row)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_comparison_report(
    successful: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
    input_dir: Path,
    output_path: Path,
) -> None:
    lines = [
        "# Real-video validation comparison",
        "",
        "This report compares measured heuristic outputs only. Filenames are identifiers, not verified ground truth, and no scoring thresholds were recalibrated from them.",
        "",
        f"Input directory used: `{input_dir}`",
        "",
        "## Results",
        "",
        "| Filename | Duration (s) | FPS | Resolution | Codec | Baseline v1 | V2 | Change | V2 label | Analysis (s) |",
        "|---|---:|---:|---|---|---:|---:|---:|---|---:|",
    ]
    for report in successful:
        video = report["raw_metrics"]["video"]
        context = report["validation_context"]
        lines.append(
            "| {filename} | {duration} | {fps} | {width}×{height} | {codec} | {baseline} | {score} | {change} | {label} | {elapsed} |".format(
                filename=context["filename"],
                duration=_fmt(video["duration_seconds"]),
                fps=_fmt(video["fps"]),
                width=video["source_width"],
                height=video["source_height"],
                codec=video["codec"],
                baseline=_fmt(
                    report.get("baseline_v1", {}).get(
                        "overall_score", report["overall_score"]
                    )
                ),
                score=_fmt(report["overall_score"]),
                change=_fmt(
                    report.get("score_comparison", {}).get(
                        "change_v2_minus_v1", 0.0
                    )
                ),
                label=report["suitability_label"],
                elapsed=_fmt(context["analysis_seconds"], 3),
            )
        )
    if not successful:
        lines.append("| _No successful analyses_ | | | | | | | | | |")

    lines.extend(["", "## Cohort-level observations", ""])
    likely_count = sum(
        report["suitability_label"] == "likely_suitable" for report in successful
    )
    lines.append(
        f"- `{likely_count}/{len(successful)}` successful analyses were labeled `likely_suitable`."
    )
    if successful and likely_count == len(successful):
        lines.append(
            "- There are no low-scoring predictions in this cohort, so observed false-negative behavior cannot be assessed from these files."
        )
    if successful:
        lowest = min(successful, key=lambda report: report["overall_score"])
        highest = max(successful, key=lambda report: report["overall_score"])
        lines.append(
            f"- Scores ranged from {lowest['overall_score']:.2f} ({lowest['validation_context']['filename']}) to {highest['overall_score']:.2f} ({highest['validation_context']['filename']})."
        )

    lines.extend(["", "## Per-video evidence and review risks", ""])
    for report in successful:
        context = report["validation_context"]
        problem_types = sorted(
            {segment["problem_type"] for segment in report["problem_segments"]}
        )
        lines.extend(
            [
                f"### {context['filename']}",
                "",
                f"- Score change: baseline v1 {report.get('baseline_v1', {}).get('overall_score', report['overall_score']):.2f} → v2 {report['overall_score']:.2f} ({report.get('score_comparison', {}).get('change_v2_minus_v1', 0.0):+.2f}).",
                f"- Subscores: `{json.dumps(report['subscores'], sort_keys=True)}`",
                f"- Detected problem types: {', '.join(problem_types) if problem_types else 'none'}.",
                f"- Evidence level: {report['confidence_or_uncertainty']['evidence_level']}.",
            ]
        )
        for note in risk_notes(report):
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(["## Processing failures", ""])
    if failures:
        for failure in failures:
            lines.append(f"- `{failure['filename']}`: {failure['error']}")
    else:
        lines.append("No supported video failed processing.")

    total_seconds = sum(
        float(report["validation_context"]["analysis_seconds"])
        for report in successful
    ) + sum(float(failure["analysis_seconds"]) for failure in failures)
    lines.extend(
        [
            "",
            "## Interpretation and next steps",
            "",
            f"- Total measured analysis time: {total_seconds:.3f} seconds for {len(successful) + len(failures)} videos.",
            "- A reviewer should watch each source and complete `human_review_template.csv` without using the heuristic score as the answer key.",
            "- Compare human observations and, later, genuine downstream reconstruction outcomes against the unchanged heuristic outputs.",
            "- Review every triggered gate and every low geometry-reliability result against independent observation; a cap is a documented heuristic safeguard, not ground truth.",
            "- Do not tune weights or thresholds until the review labels and independent reconstruction outcomes are defined and sufficiently numerous.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(input_dir: Path, output_dir: Path, config: AnalysisConfig) -> int:
    videos = find_videos(input_dir)
    if not videos:
        raise RuntimeError(f"No supported video files found under: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    successful: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for path in videos:
        before_hash = sha256_file(path)
        start = time.perf_counter()
        try:
            report = analyze_video(str(path), config=config)
            elapsed = time.perf_counter() - start
            after_hash = sha256_file(path)
            unchanged = before_hash == after_hash
            if not unchanged:
                raise RuntimeError("Source hash changed during read-only analysis")
            report["validation_context"] = {
                "filename": path.name,
                "source_path": str(path.resolve()),
                "source_sha256": before_hash,
                "source_unchanged_after_processing": True,
                "analysis_seconds": round(elapsed, 3),
                "config": {
                    "max_samples": config.max_samples,
                    "max_analysis_side": config.max_analysis_side,
                },
            }
            write_report(report, output_dir / f"{path.name}.json")
            successful.append(report)
            summary_rows.append(flatten_report(report))
            print(
                f"OK {path.name}: {report['overall_score']:.2f}/100 "
                f"{report['suitability_label']} in {elapsed:.3f}s"
            )
        except Exception as exc:  # continue so every accessible video is attempted
            elapsed = time.perf_counter() - start
            after_hash = sha256_file(path)
            unchanged = before_hash == after_hash
            error = f"{type(exc).__name__}: {exc}"
            failure = failed_row(path, elapsed, before_hash, unchanged, error)
            failures.append(failure)
            summary_rows.append(failure)
            failure_report = {
                "schema_version": "1.0",
                "processing_status": "failed",
                "validation_context": {
                    "filename": path.name,
                    "source_path": str(path.resolve()),
                    "source_sha256": before_hash,
                    "source_unchanged_after_processing": unchanged,
                    "analysis_seconds": round(elapsed, 3),
                },
                "error": error,
            }
            write_report(failure_report, output_dir / f"{path.name}.json")
            print(f"FAILED {path.name}: {error} in {elapsed:.3f}s")

    write_summary_csv(summary_rows, output_dir / "summary.csv")
    write_human_review_template(videos, output_dir / "human_review_template.csv")
    write_before_after_components(
        successful, output_dir / "before_after_components.csv"
    )
    write_comparison_report(
        successful,
        failures,
        input_dir,
        output_dir / "comparison_report.md",
    )
    print(
        f"Processed {len(videos)} videos: {len(successful)} succeeded, "
        f"{len(failures)} failed. Artifacts: {output_dir}"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=640)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AnalysisConfig(
        max_samples=args.max_samples,
        max_analysis_side=args.max_side,
    )
    config.validate()
    try:
        return run_validation(args.input_dir, args.output_dir, config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
