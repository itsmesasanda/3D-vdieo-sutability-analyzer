#!/usr/bin/env python3
"""Benchmark exact eight-feature extraction and optional RF prediction on CPU."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence

from video_suitability.ml.artifacts import sha256_file
from video_suitability.ml.registration_features import (
    extract_registration_features,
    load_feature_spec,
)
from video_suitability.ml.registration_rf import (
    predict_colmap_registration_percentage,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--feature-spec",
        type=Path,
        help="Benchmark extraction only when a trained artifact is not yet available",
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.artifact_dir is None and args.feature_spec is None:
        raise SystemExit("provide --artifact-dir or --feature-spec")
    before_hash = sha256_file(args.video)
    rows = []
    if args.artifact_dir is not None:
        for run in range(1, args.runs + 1):
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            result = predict_colmap_registration_percentage(
                args.video, args.artifact_dir
            )
            rows.append(
                {
                    "run": run,
                    "mode": "artifact_prediction",
                    "wall_seconds": round(time.perf_counter() - wall_started, 6),
                    "cpu_seconds": round(time.process_time() - cpu_started, 6),
                    "status": result["status"],
                    "predicted_registration_percentage": result.get(
                        "predicted_registration_percentage"
                    ),
                    "performance_profile": result.get("performance_profile"),
                    "reason": result.get("reason"),
                }
            )
    else:
        spec = load_feature_spec(args.feature_spec)
        for run in range(1, args.runs + 1):
            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            features, diagnostics = extract_registration_features(args.video, spec)
            rows.append(
                {
                    "run": run,
                    "mode": "feature_extraction_only_no_trained_model",
                    "wall_seconds": round(time.perf_counter() - wall_started, 6),
                    "cpu_seconds": round(time.process_time() - cpu_started, 6),
                    "status": "features_only",
                    "features": features,
                    "sample_count": diagnostics["sample_count"],
                    "pair_count": diagnostics["pair_count"],
                }
            )
    after_hash = sha256_file(args.video)
    payload = {
        "scope": (
            "Local CPU timing. Feature-only mode does not imply a trained prediction "
            "or real-world model accuracy."
        ),
        "platform": platform.platform(),
        "video": str(args.video),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_unchanged": before_hash == after_hash,
        "run_count": len(rows),
        "mean_wall_seconds": round(
            statistics.mean(row["wall_seconds"] for row in rows), 6
        ),
        "mean_cpu_seconds": round(
            statistics.mean(row["cpu_seconds"] for row in rows), 6
        ),
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
