#!/usr/bin/env python3
"""Analyze an already exported COLMAP text model and make a sparse preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_suitability.reconstruction import (
    ReconstructionError,
    analyze_colmap_text_model,
    render_sparse_preview,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--total-input-images", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    try:
        metrics = analyze_colmap_text_model(args.model_dir, args.total_input_images)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.preview:
            render_sparse_preview(args.model_dir, args.preview)
    except ReconstructionError as exc:
        print(f"model analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

