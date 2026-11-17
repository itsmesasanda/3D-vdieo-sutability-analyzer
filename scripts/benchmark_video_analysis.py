#!/usr/bin/env python3
"""Benchmark repeatable end-to-end analysis without modifying the source video."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--reference-report", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    source_hash = sha256_file(args.video)
    reference = (
        json.loads(args.reference_report.read_text(encoding="utf-8"))
        if args.reference_report
        else None
    )
    config = AnalysisConfig(
        max_samples=args.max_samples, max_analysis_side=args.max_side
    )
    rows = []
    for index in range(args.runs):
        started = time.perf_counter()
        report = analyze_video(str(args.video), config=config)
        elapsed = time.perf_counter() - started
        comparison = None
        if reference:
            comparison = {
                "score_equal": report["overall_score"] == reference["overall_score"],
                "subscores_equal": report["subscores"] == reference["subscores"],
                "quality_metrics_equal": report["raw_metrics"]["quality_summary"]
                == reference["raw_metrics"]["quality_summary"],
                "geometry_metrics_equal": report["raw_metrics"]["geometry_summary"]
                == reference["raw_metrics"]["geometry_summary"],
                "sampled_frame_metrics_equal": report["raw_metrics"][
                    "sampled_frame_quality"
                ]
                == reference["raw_metrics"]["sampled_frame_quality"],
                "sampled_pair_metrics_equal": report["raw_metrics"][
                    "sampled_pair_geometry"
                ]
                == reference["raw_metrics"]["sampled_pair_geometry"],
                "problem_segments_equal": report["problem_segments"]
                == reference["problem_segments"],
            }
        row = {
            "run": index + 1,
            "analysis_seconds": round(elapsed, 6),
            "score": report["overall_score"],
            "scoring_revision": report.get("scoring_revision"),
            "decode_strategy": report["raw_metrics"]["video"].get(
                "decode_strategy"
            ),
            "decoded_sample_count": report["raw_metrics"]["video"][
                "decoded_sample_count"
            ],
            "temporal_coverage_fraction": report["raw_metrics"]["video"][
                "temporal_coverage_fraction"
            ],
            "reference_semantics": comparison,
        }
        rows.append(row)
        print(json.dumps(row))
    source_hash_after = sha256_file(args.video)
    payload = {
        "video": str(args.video.resolve()),
        "source_sha256_before": source_hash,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash == source_hash_after,
        "platform": platform.platform(),
        "runs": rows,
        "mean_analysis_seconds": round(
            statistics.mean(row["analysis_seconds"] for row in rows), 6
        ),
        "minimum_analysis_seconds": min(row["analysis_seconds"] for row in rows),
        "maximum_analysis_seconds": max(row["analysis_seconds"] for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["source_unchanged"]:
        raise SystemExit("source checksum changed during benchmark")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

