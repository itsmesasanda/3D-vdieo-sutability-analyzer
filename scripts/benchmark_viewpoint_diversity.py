#!/usr/bin/env python3
"""Run the deterministic viewpoint-diversity scenario and CPU timing audit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2

from video_suitability.config import AnalysisConfig
from video_suitability.geometry import analyze_geometry, detect_orb_features
from video_suitability.infer import analyze_video
from video_suitability.quality import analyze_quality
from video_suitability.synthetic import generate_test_videos
from video_suitability.video import sample_video
from video_suitability.viewpoint import analyze_viewpoint_diversity


EXPECTED = {
    "front_only": "Low",
    "quarter_orbit": "Moderate",
    "full_orbit": "High",
    "pure_rotation": "Unknown",
    "blurry": "Unknown",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=400)
    args = parser.parse_args()
    config = AnalysisConfig(
        max_samples=args.max_samples,
        max_analysis_side=args.max_side,
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="viewpoint-benchmark-") as directory:
        videos = generate_test_videos(
            Path(directory), width=400, height=300, frame_count=24, fps=12.0
        )
        for scenario, expected in EXPECTED.items():
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            report = analyze_video(str(videos[scenario]), config=config)
            cpu_seconds = time.process_time() - cpu_started
            wall_seconds = time.perf_counter() - wall_started
            result = report["viewpoint_diversity"]

            sampled = sample_video(videos[scenario], config)
            quality = analyze_quality(
                sampled.frames,
                sampled.frame_indices,
                sampled.timestamps_seconds,
            )
            features = detect_orb_features(sampled.frames, config)
            geometry = analyze_geometry(
                sampled.frames,
                sampled.frame_indices,
                sampled.timestamps_seconds,
                config,
                detected_features=features,
            )
            module_wall_started = time.perf_counter()
            module_cpu_started = time.process_time()
            direct_result = analyze_viewpoint_diversity(
                sampled.frames,
                sampled.frame_indices,
                sampled.timestamps_seconds,
                quality["summary"],
                geometry,
                config,
                detected_features=features,
            )
            module_cpu_seconds = time.process_time() - module_cpu_started
            module_wall_seconds = time.perf_counter() - module_wall_started
            row = {
                "scenario": scenario,
                "expected": expected,
                "observed": result["label"],
                "matched_expectation": result["label"] == expected,
                "wall_seconds": round(wall_seconds, 6),
                "cpu_seconds": round(cpu_seconds, 6),
                "viewpoint_module_wall_seconds": round(module_wall_seconds, 6),
                "viewpoint_module_cpu_seconds": round(module_cpu_seconds, 6),
                "direct_module_label_matched": direct_result["label"]
                == result["label"],
                "sampled_frames": result["indicators"]["sampled_frame_count"],
                "nearby_pairs": result["indicators"]["nearby_pair_count"],
                "separated_pairs": result["indicators"][
                    "temporally_separated_pair_count"
                ],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    payload = {
        "benchmark_scope": "synthetic regression and local CPU timing; not real-world accuracy",
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported by platform",
        "config": {
            "max_samples": config.max_samples,
            "max_analysis_side": config.max_analysis_side,
            "orb_features": config.orb_features,
        },
        "all_expectations_matched": all(
            row["matched_expectation"] and row["direct_module_label_matched"]
            for row in rows
        ),
        "total_wall_seconds": round(sum(row["wall_seconds"] for row in rows), 6),
        "total_cpu_seconds": round(sum(row["cpu_seconds"] for row in rows), 6),
        "total_viewpoint_module_wall_seconds": round(
            sum(row["viewpoint_module_wall_seconds"] for row in rows), 6
        ),
        "total_viewpoint_module_cpu_seconds": round(
            sum(row["viewpoint_module_cpu_seconds"] for row in rows), 6
        ),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["all_expectations_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

