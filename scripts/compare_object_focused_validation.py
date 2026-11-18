#!/usr/bin/env python3
"""Compare whole-frame and object-focused viewpoint evidence from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video, write_report
from video_suitability.object_focus import normalize_object_bbox
from video_suitability.video import VideoError


REQUIRED_SCENARIOS = {
    "front_only",
    "quarter_orbit",
    "full_orbit",
    "rotation_only",
    "blurred",
}

CSV_COLUMNS = [
    "filename",
    "scenario",
    "object_id",
    "bbox_json",
    "whole_category",
    "object_category",
    "classification_changed",
    "whole_evidence_status",
    "object_evidence_status",
    "whole_primary_reason",
    "object_primary_reason",
    "whole_cpu_seconds",
    "object_cpu_seconds",
    "whole_wall_seconds",
    "object_wall_seconds",
    "whole_decode_retries",
    "object_decode_retries",
    "whole_median_separated_verified_matches",
    "object_median_separated_verified_matches",
    "whole_long_gap_parallax_fraction",
    "object_long_gap_parallax_fraction",
    "whole_repeated_view_pairs",
    "object_repeated_view_pairs",
    "object_tracked_sample_fraction",
    "background_dominated_pair_fraction",
    "median_object_verified_matches",
    "median_background_verified_matches",
    "whole_score",
    "object_score",
    "score_paths_unchanged",
    "object_warnings_json",
    "notes",
    "error",
]


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "filename",
        "scenario",
        "object_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def _bbox(row: Mapping[str, str]) -> List[float]:
    return list(
        normalize_object_bbox(
            [
                float(row["bbox_x"]),
                float(row["bbox_y"]),
                float(row["bbox_width"]),
                float(row["bbox_height"]),
            ]
        )
    )


def _timed_analysis(
    path: Path, config: AnalysisConfig, object_bbox: Optional[Sequence[float]]
) -> tuple:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    retries = 0
    while True:
        try:
            report = analyze_video(
                str(path),
                config=config,
                object_bbox=object_bbox,
                include_performance_profile=True,
            )
            break
        except VideoError:
            if retries >= 1:
                raise
            retries += 1
    return (
        report,
        round(time.perf_counter() - wall_started, 6),
        round(time.process_time() - cpu_started, 6),
        retries,
    )


def _flatten(
    manifest_row: Mapping[str, str],
    bbox: Sequence[float],
    whole: Mapping[str, Any],
    object_report: Mapping[str, Any],
    timings: Mapping[str, float],
) -> Dict[str, Any]:
    whole_view = whole["viewpoint_diversity"]
    object_view = object_report["viewpoint_diversity"]
    whole_indicators = whole_view["indicators"]
    object_indicators = object_view["indicators"]
    background = object_view["background_diagnostics"]
    score_unchanged = bool(
        whole["overall_score"] == object_report["overall_score"]
        and whole["subscores"] == object_report["subscores"]
        and whole["baseline_v1"] == object_report["baseline_v1"]
    )
    return {
        "filename": manifest_row["filename"],
        "scenario": manifest_row["scenario"],
        "object_id": manifest_row["object_id"],
        "bbox_json": json.dumps(list(bbox), separators=(",", ":")),
        "whole_category": whole_view["label"],
        "object_category": object_view["label"],
        "classification_changed": whole_view["label"] != object_view["label"],
        "whole_evidence_status": whole_view["evidence_status"],
        "object_evidence_status": object_view["evidence_status"],
        "whole_primary_reason": whole_view["decision_explanation"]["primary_reason"],
        "object_primary_reason": object_view["decision_explanation"]["primary_reason"],
        "whole_cpu_seconds": timings["whole_cpu_seconds"],
        "object_cpu_seconds": timings["object_cpu_seconds"],
        "whole_wall_seconds": timings["whole_wall_seconds"],
        "object_wall_seconds": timings["object_wall_seconds"],
        "whole_decode_retries": timings["whole_decode_retries"],
        "object_decode_retries": timings["object_decode_retries"],
        "whole_median_separated_verified_matches": whole_indicators[
            "median_separated_verified_matches"
        ],
        "object_median_separated_verified_matches": object_indicators[
            "median_separated_verified_matches"
        ],
        "whole_long_gap_parallax_fraction": whole_indicators[
            "temporally_separated_parallax_proxy_pair_fraction"
        ],
        "object_long_gap_parallax_fraction": object_indicators[
            "temporally_separated_parallax_proxy_pair_fraction"
        ],
        "whole_repeated_view_pairs": len(whole_view["repeated_view_intervals"]),
        "object_repeated_view_pairs": len(object_view["repeated_view_intervals"]),
        "object_tracked_sample_fraction": object_view["object_tracking"][
            "tracked_sample_fraction"
        ],
        "background_dominated_pair_fraction": background[
            "background_dominated_pair_fraction"
        ],
        "median_object_verified_matches": background[
            "median_object_verified_matches"
        ],
        "median_background_verified_matches": background[
            "median_background_verified_matches"
        ],
        "whole_score": whole["overall_score"],
        "object_score": object_report["overall_score"],
        "score_paths_unchanged": score_unchanged,
        "object_warnings_json": json.dumps(
            object_view["warnings"], ensure_ascii=False, separators=(",", ":")
        ),
        "notes": manifest_row.get("notes", ""),
        "error": "",
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _display_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_report(
    rows: Sequence[Mapping[str, Any]], path: Path, config: AnalysisConfig
) -> None:
    lines = [
        "# Object-focused real-video comparison",
        "",
        "This comparison is diagnostic, not a real-world accuracy estimate. Scenario labels "
        "are capture-protocol metadata only when a human has supplied them; synthetic "
        "expectations are not ground truth for these recordings.",
        "",
        f"Configuration: `{config.max_samples}` samples, `{config.max_analysis_side}px` maximum analysis side.",
        "",
        "| Video | Scenario | Whole | Object | Tracking | Background-dominated | Verified matches whole/object | CPU whole/object (s) | Decode retries whole/object | Score unchanged |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(
                f"| {row['filename']} | {row['scenario']} | error | error | | | | | | |"
            )
            continue
        lines.append(
            f"| {row['filename']} | {row['scenario']} | {row['whole_category']} | "
            f"{row['object_category']} | "
            f"{_display_number(row['object_tracked_sample_fraction'])} | "
            f"{_display_number(row['background_dominated_pair_fraction'])} | "
            f"{_display_number(row['whole_median_separated_verified_matches'], 1)} / "
            f"{_display_number(row['object_median_separated_verified_matches'], 1)} | "
            f"{_display_number(row['whole_cpu_seconds'])} / "
            f"{_display_number(row['object_cpu_seconds'])} | "
            f"{row.get('whole_decode_retries', '') or 0} / "
            f"{row.get('object_decode_retries', '') or 0} | "
            f"{row['score_paths_unchanged']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Unknown is intentional when tracking, verified object-only matches, descriptor uniqueness, or non-planar parallax evidence is inadequate.",
            "- Background-only verified matches are retained as a contamination diagnostic and do not increase the object-focused category.",
            "- Appearance change is not interpreted as exact camera travel, complete orbit, or object-surface coverage.",
            "- Pure rotation, object symmetry, repeated texture, inaccurate intrinsics, crop/stabilization, and motion blur remain important failure modes.",
            "- The suitability score and optional Random Forest path are not inputs to this comparison and are not modified.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    input_dir: Path,
    manifest: Path,
    output_dir: Path,
    config: AnalysisConfig,
    require_complete_scenarios: bool,
    only_filenames: Optional[Sequence[str]] = None,
) -> int:
    manifest_rows = _load_manifest(manifest)
    scenarios = {row["scenario"] for row in manifest_rows}
    if require_complete_scenarios:
        missing = REQUIRED_SCENARIOS - scenarios
        if missing:
            raise ValueError(
                "manifest is missing required same-object scenarios: "
                + ", ".join(sorted(missing))
            )
        object_ids = {row["object_id"] for row in manifest_rows}
        if len(object_ids) != 1:
            raise ValueError(
                "a complete physical-object validation manifest must use one object_id"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    existing_by_filename: Dict[str, Dict[str, Any]] = {}
    existing_csv = output_dir / "comparison.csv"
    if only_filenames and existing_csv.exists():
        with existing_csv.open(newline="", encoding="utf-8") as handle:
            existing_by_filename = {
                row["filename"]: dict(row) for row in csv.DictReader(handle)
            }
    selected_names = set(only_filenames or [])
    rows_to_process = [
        row
        for row in manifest_rows
        if not selected_names or row["filename"] in selected_names
    ]
    if selected_names - {row["filename"] for row in manifest_rows}:
        missing = selected_names - {row["filename"] for row in manifest_rows}
        raise ValueError(
            "--only-filename is not present in the manifest: "
            + ", ".join(sorted(missing))
        )
    updated_by_filename: Dict[str, Dict[str, Any]] = {}
    failures = 0
    for manifest_row in rows_to_process:
        video_path = input_dir / manifest_row["filename"]
        try:
            bbox = _bbox(manifest_row)
            whole, whole_wall, whole_cpu, whole_retries = _timed_analysis(
                video_path, config, None
            )
            object_report, object_wall, object_cpu, object_retries = _timed_analysis(
                video_path, config, bbox
            )
            row = _flatten(
                manifest_row,
                bbox,
                whole,
                object_report,
                {
                    "whole_wall_seconds": whole_wall,
                    "whole_cpu_seconds": whole_cpu,
                    "object_wall_seconds": object_wall,
                    "object_cpu_seconds": object_cpu,
                    "whole_decode_retries": whole_retries,
                    "object_decode_retries": object_retries,
                },
            )
            write_report(whole, reports_dir / f"{video_path.name}.whole.json")
            write_report(object_report, reports_dir / f"{video_path.name}.object.json")
            print(
                f"OK {video_path.name}: {row['whole_category']} -> "
                f"{row['object_category']}, tracking="
                f"{row['object_tracked_sample_fraction']:.3f}"
            )
        except Exception as exc:
            failures += 1
            row = {column: "" for column in CSV_COLUMNS}
            row.update(
                {
                    "filename": manifest_row.get("filename", ""),
                    "scenario": manifest_row.get("scenario", ""),
                    "object_id": manifest_row.get("object_id", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAILED {video_path.name}: {row['error']}")
        updated_by_filename[manifest_row["filename"]] = row
    results: List[Dict[str, Any]] = []
    for manifest_row in manifest_rows:
        filename = manifest_row["filename"]
        if filename in updated_by_filename:
            results.append(updated_by_filename[filename])
        elif filename in existing_by_filename:
            results.append(existing_by_filename[filename])
        else:
            row = {column: "" for column in CSV_COLUMNS}
            row.update(
                {
                    "filename": filename,
                    "scenario": manifest_row.get("scenario", ""),
                    "object_id": manifest_row.get("object_id", ""),
                    "error": "not processed in this partial run",
                }
            )
            results.append(row)
    _write_csv(results, output_dir / "comparison.csv")
    _write_report(results, output_dir / "REPORT.md", config)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=480)
    parser.add_argument("--require-complete-scenario-set", action="store_true")
    parser.add_argument(
        "--only-filename",
        action="append",
        help="Process only this manifest filename and preserve other existing CSV rows",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = AnalysisConfig(
        max_samples=args.max_samples,
        max_analysis_side=args.max_side,
    )
    try:
        config.validate()
        return run(
            args.input_dir,
            args.manifest,
            args.output_dir,
            config,
            args.require_complete_scenario_set,
            args.only_filename,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
