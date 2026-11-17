#!/usr/bin/env python3
"""Generate the deterministic local MP4 validation suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from video_suitability.synthetic import generate_test_videos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/samples")
    args = parser.parse_args()
    outputs = generate_test_videos(Path(args.out_dir))
    for name, path in outputs.items():
        print(f"{name}: {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
