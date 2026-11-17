#!/usr/bin/env python3
"""Extract immutable, timestamped COLMAP input frames from one video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_suitability.reconstruction import ReconstructionError, extract_reconstruction_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metadata = extract_reconstruction_frames(
            args.video,
            args.output_dir,
            sample_fps=args.sample_fps,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
        )
    except ReconstructionError as exc:
        print(f"frame extraction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

