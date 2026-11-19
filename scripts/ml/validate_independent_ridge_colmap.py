#!/usr/bin/env python3
"""Run a small independent Ridge-versus-COLMAP validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_suitability.ml.independent_validation import (
    IndependentValidationError,
    load_validation_manifest,
    run_independent_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with video_id, role and video_path columns",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        help=(
            "Trusted experimental_model.joblib. If omitted, the loader checks its "
            "usual local ignored-artifact path."
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help="Directory containing the tracked historical metadata/parity files",
    )
    parser.add_argument("--colmap", default="colmap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = load_validation_manifest(args.manifest)
        result = run_independent_validation(
            records,
            args.output_dir,
            model_artifact=args.model_artifact,
            artifact_directory=args.artifact_directory,
            colmap_executable=args.colmap,
        )
    except IndependentValidationError as exc:
        print(f"Validation setup failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"CSV: {result['csv']}")
    print(f"Report: {result['report']}")
    return 0 if result["summary"]["valid_ground_truth_count"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
