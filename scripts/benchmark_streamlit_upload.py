#!/usr/bin/env python3
"""Benchmark the exact Streamlit upload-to-result path and deferred diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app import _analyze_upload_payload
from video_suitability.presentation import extract_problem_frames


def _parse_bbox(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    values = [float(value.strip()) for value in raw.split(",")]
    if len(values) != 4:
        raise ValueError("--object-bbox requires x,y,width,height")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--object-bbox")
    parser.add_argument("--diagnostic-frames", type=int, default=6)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    bbox = _parse_bbox(args.object_bbox)
    source_bytes = args.video.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rows = []
    for index in range(args.runs):
        result = _analyze_upload_payload(
            source_bytes,
            args.video.name,
            object_bbox=bbox,
        )
        diagnostics_wall_started = time.perf_counter()
        diagnostics_cpu_started = time.process_time()
        diagnostics = extract_problem_frames(
            args.video,
            result["timeline"],
            max_frames=args.diagnostic_frames,
            max_side=720,
        )
        diagnostics_wall = time.perf_counter() - diagnostics_wall_started
        diagnostics_cpu = time.process_time() - diagnostics_cpu_started
        row = {
            "run": index + 1,
            "filename": args.video.name,
            "object_bbox": bbox,
            "viewpoint_label": result["report"]["viewpoint_diversity"]["label"],
            "suitability_score": result["report"]["overall_score"],
            "upload_path": result["upload_performance"],
            "pipeline": result["report"]["performance_profile"],
            "deferred_diagnostic_rendering": {
                "frame_count": len(diagnostics),
                "wall_seconds": round(diagnostics_wall, 6),
                "cpu_seconds": round(diagnostics_cpu, 6),
            },
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    current_hash = hashlib.sha256(args.video.read_bytes()).hexdigest()
    payload = {
        "scope": (
            "Actual app upload payload path; diagnostic rendering measured separately. "
            "Single-host timings are not a cross-platform guarantee."
        ),
        "platform": platform.platform(),
        "source_sha256_before": source_hash,
        "source_sha256_after": current_hash,
        "source_unchanged": current_hash == source_hash,
        "runs": rows,
        "mean_upload_to_result_wall_seconds": round(
            statistics.mean(
                row["upload_path"]["upload_to_result_wall_seconds"] for row in rows
            ),
            6,
        ),
        "mean_deferred_diagnostic_wall_seconds": round(
            statistics.mean(
                row["deferred_diagnostic_rendering"]["wall_seconds"] for row in rows
            ),
            6,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

