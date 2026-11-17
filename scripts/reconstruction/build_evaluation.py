#!/usr/bin/env python3
"""Build Milestone 3A result and human-review CSVs without changing score reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from video_suitability.reconstruction import sha256_file


VIDEOS = ("IMG_3913.MOV", "IMG_3914.MOV", "IMG_3915.MOV", "orbit_good.mp4")
EXPECTED_V2 = {
    "IMG_3913.MOV": 78.54,
    "IMG_3914.MOV": 60.00,
    "IMG_3915.MOV": 48.51,
    "orbit_good.mp4": 44.00,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nested(metrics: Optional[Dict[str, Any]], key: str) -> Any:
    return metrics.get(key, "") if metrics else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/samples/real_tests"))
    parser.add_argument("--v1-dir", type=Path, default=Path("reports/real_tests"))
    parser.add_argument("--v2-dir", type=Path, default=Path("reports/real_tests/v2"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("evaluation/reconstruction"))
    args = parser.parse_args()
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename in VIDEOS:
        source = args.source_dir / filename
        v1_path = args.v1_dir / f"{filename}.json"
        v2_path = args.v2_dir / f"{filename}.json"
        if not source.is_file() or not v1_path.is_file() or not v2_path.is_file():
            raise SystemExit(f"Required source or preserved score report is missing for {filename}")
        v1 = load_json(v1_path)
        v2 = load_json(v2_path)
        v2_score = float(v2["overall_score"])
        if round(v2_score, 2) != EXPECTED_V2[filename]:
            raise SystemExit(
                f"Refusing to build comparison: {filename} v2 changed from preserved "
                f"{EXPECTED_V2[filename]:.2f} to {v2_score:.2f}"
            )

        sequence = args.evaluation_dir / Path(filename).stem
        extraction_path = sequence / "extraction_metadata.json"
        reconstruction_path = sequence / "colmap" / "reconstruction_result.json"
        extraction = load_json(extraction_path) if extraction_path.is_file() else None
        reconstruction = load_json(reconstruction_path) if reconstruction_path.is_file() else None
        if reconstruction:
            reconstruction_status = reconstruction["status"]
            failure_stage = reconstruction.get("failure_stage") or ""
            failure_reason = reconstruction.get("failure_reason") or ""
            metrics = reconstruction.get("objective_metrics")
        else:
            reconstruction_status = "not_run_pilot_did_not_succeed"
            failure_stage = "pilot_gate"
            failure_reason = (
                "Not run because the IMG_3913 pilot could not execute without local COLMAP; "
                "this is not a reconstruction-quality result."
            )
            metrics = None

        source_hash = sha256_file(source)
        rows.append(
            {
                "filename": filename,
                "sequence_role": "pilot" if filename == "IMG_3913.MOV" else "post_pilot",
                "source_size_bytes": source.stat().st_size,
                "source_sha256": source_hash,
                "source_unchanged_after_extraction": (
                    extraction.get("source_unchanged_after_extraction", "") if extraction else ""
                ),
                "baseline_v1_score": v1["overall_score"],
                "baseline_v1_label": v1["suitability_label"],
                "baseline_v1_report_sha256": sha256_file(v1_path),
                "v2_score": v2_score,
                "v2_label": v2["suitability_label"],
                "v2_report_sha256": sha256_file(v2_path),
                "extraction_status": "succeeded" if extraction else "not_run",
                "requested_sample_fps": extraction.get("requested_sample_fps", "") if extraction else "",
                "extracted_frame_count": extraction.get("extracted_frame_count", "") if extraction else "",
                "first_timestamp_seconds": extraction.get("first_timestamp_seconds", "") if extraction else "",
                "last_timestamp_seconds": extraction.get("last_timestamp_seconds", "") if extraction else "",
                "temporal_coverage_fraction": extraction.get("temporal_coverage_fraction", "") if extraction else "",
                "extraction_runtime_seconds": extraction.get("extraction_runtime_seconds", "") if extraction else "",
                "reconstruction_status": reconstruction_status,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "registered_image_count": nested(metrics, "registered_image_count"),
                "total_input_image_count": (
                    nested(metrics, "total_input_image_count")
                    if metrics
                    else reconstruction.get("total_input_image_count", "") if reconstruction else ""
                ),
                "registered_image_ratio": nested(metrics, "registered_image_ratio"),
                "reconstructed_point_count": nested(metrics, "reconstructed_point_count"),
                "mean_reprojection_error_pixels": nested(metrics, "mean_reprojection_error_pixels"),
                "mean_track_length_images": nested(metrics, "mean_track_length_images"),
                "camera_center_trajectory_length_arbitrary_units": nested(
                    metrics, "camera_center_trajectory_length_arbitrary_units"
                ),
                "camera_center_max_baseline_arbitrary_units": nested(
                    metrics, "camera_center_max_baseline_arbitrary_units"
                ),
                "mapper_runtime_seconds": (
                    reconstruction.get("mapper_runtime_seconds", "") if reconstruction else ""
                ),
                "colmap_total_runtime_seconds": (
                    reconstruction.get("total_runtime_seconds", "") if reconstruction else ""
                ),
                "dense_reconstruction_run": (
                    reconstruction.get("dense_reconstruction_run", False)
                    if reconstruction
                    else False
                ),
                "objective_metrics_available": bool(metrics),
                "coverage_interpretation": nested(metrics, "coverage_interpretation"),
                "sparse_ply": reconstruction.get("sparse_ply", "") if reconstruction else "",
                "sparse_preview": reconstruction.get("sparse_preview", "") if reconstruction else "",
            }
        )

    results_path = args.evaluation_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    review_path = args.evaluation_dir / "human_review_form.csv"
    if not review_path.exists():
        review_fields = [
            "filename",
            "reconstruction_status",
            "reviewer",
            "review_date",
            "object_completeness_0_to_5",
            "geometric_plausibility_0_to_5",
            "missing_regions",
            "artifacts",
            "actual_usability_yes_no_uncertain",
            "notes",
        ]
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=review_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "filename": row["filename"],
                        "reconstruction_status": row["reconstruction_status"],
                    }
                )
    print(f"wrote {results_path} ({len(rows)} rows)")
    print(f"human review form: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

