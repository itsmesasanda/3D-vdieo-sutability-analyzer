#!/usr/bin/env python3
"""Run a CPU-compatible sparse COLMAP experiment on one isolated frame sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_suitability.reconstruction import ReconstructionError, run_sparse_colmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--workspace-dir", required=True, type=Path)
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable or command name")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--sequential-overlap", type=int, default=10)
    parser.add_argument("--retry-exhaustive", action="store_true")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU SIFT only when the installed COLMAP exposes compatible flags",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_sparse_colmap(
            args.images_dir,
            args.workspace_dir,
            colmap_executable=args.colmap,
            camera_model=args.camera_model,
            sequential_overlap=args.sequential_overlap,
            retry_exhaustive=args.retry_exhaustive,
            use_gpu=args.use_gpu,
        )
    except ReconstructionError as exc:
        print(f"COLMAP experiment failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "succeeded" else 3


if __name__ == "__main__":
    raise SystemExit(main())

