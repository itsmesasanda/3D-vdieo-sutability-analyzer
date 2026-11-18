#!/usr/bin/env python3
"""Compare whole-frame and object-focused analysis on controlled video fixtures."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.synthetic import generate_test_videos


SCENARIOS = {
    "front_only": {"whole": "Low", "object": "Low"},
    "quarter_orbit": {"whole": "Moderate", "object": "Moderate"},
    "full_orbit": {"whole": "High", "object": "Moderate"},
    "pure_rotation": {"whole": "Unknown", "object": "Unknown"},
    "blurry": {"whole": "Unknown", "object": "Unknown"},
}
OBJECT_BBOX = [0.25, 0.10, 0.50, 0.80]


def _timed(path: Path, config: AnalysisConfig, bbox: Optional[Sequence[float]]):
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    report = analyze_video(str(path), config=config, object_bbox=bbox)
    return (
        report,
        round(time.perf_counter() - wall_started, 6),
        round(time.process_time() - cpu_started, 6),
    )


def _evidence(report: Mapping[str, Any]) -> Dict[str, Any]:
    viewpoint = report["viewpoint_diversity"]
    indicators = viewpoint["indicators"]
    return {
        "category": viewpoint["label"],
        "evidence_status": viewpoint["evidence_status"],
        "primary_reason": viewpoint["decision_explanation"]["primary_reason"],
        "median_separated_verified_matches": indicators[
            "median_separated_verified_matches"
        ],
        "long_gap_parallax_proxy_fraction": indicators[
            "temporally_separated_parallax_proxy_pair_fraction"
        ],
        "long_gap_repeated_view_fraction": indicators[
            "temporally_separated_repeated_view_fraction"
        ],
        "strict_descriptor_ambiguity_fraction": indicators[
            "median_separated_strict_descriptor_ambiguity_fraction"
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=400)
    args = parser.parse_args(argv)
    config = AnalysisConfig(
        max_samples=args.max_samples,
        max_analysis_side=args.max_side,
    )
    config.validate()
    results = []
    all_expectations_matched = True
    all_scores_unchanged = True
    with tempfile.TemporaryDirectory(prefix="object_focus_benchmark_") as directory:
        videos = generate_test_videos(
            Path(directory), width=400, height=300, frame_count=24, fps=12.0
        )
        for scenario, expected in SCENARIOS.items():
            whole, whole_wall, whole_cpu = _timed(videos[scenario], config, None)
            focused, focused_wall, focused_cpu = _timed(
                videos[scenario], config, OBJECT_BBOX
            )
            whole_view = _evidence(whole)
            object_view = _evidence(focused)
            object_result = focused["viewpoint_diversity"]
            matched = bool(
                whole_view["category"] == expected["whole"]
                and object_view["category"] == expected["object"]
            )
            unchanged = bool(
                whole["overall_score"] == focused["overall_score"]
                and whole["subscores"] == focused["subscores"]
                and whole["baseline_v1"] == focused["baseline_v1"]
            )
            all_expectations_matched &= matched
            all_scores_unchanged &= unchanged
            results.append(
                {
                    "scenario": scenario,
                    "expected": expected,
                    "matched_expectation": matched,
                    "classification_changed": (
                        whole_view["category"] != object_view["category"]
                    ),
                    "whole": {
                        **whole_view,
                        "wall_seconds": whole_wall,
                        "cpu_seconds": whole_cpu,
                    },
                    "object_focused": {
                        **object_view,
                        "wall_seconds": focused_wall,
                        "cpu_seconds": focused_cpu,
                        "tracked_sample_fraction": object_result[
                            "object_tracking"
                        ]["tracked_sample_fraction"],
                        "background_dominated_pair_fraction": object_result[
                            "background_diagnostics"
                        ]["background_dominated_pair_fraction"],
                    },
                    "score_paths_unchanged": unchanged,
                }
            )
    payload = {
        "scope": (
            "Controlled layered 2D regression fixtures; not physical-object or "
            "real-world accuracy validation."
        ),
        "platform": platform.platform(),
        "config": {
            "max_samples": config.max_samples,
            "max_analysis_side": config.max_analysis_side,
            "object_bbox": OBJECT_BBOX,
        },
        "all_expectations_matched": all_expectations_matched,
        "all_score_paths_unchanged": all_scores_unchanged,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all_expectations_matched and all_scores_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
