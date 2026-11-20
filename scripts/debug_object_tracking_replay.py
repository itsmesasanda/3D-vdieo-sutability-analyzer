"""Replay the exact Streamlit upload analysis and save object-tracker diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app


def _bbox(value: str) -> Sequence[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numbers") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,width,height")
    return values


def replay(video: Path, bbox: Sequence[float]) -> Dict[str, Any]:
    payload = video.read_bytes()
    result = app._analyze_upload_payload(
        payload,
        video.name,
        object_bbox=bbox,
        include_viewpoint_diversity=True,
    )
    report = result["report"]
    viewpoint = report["viewpoint_diversity"]
    tracking = viewpoint.get("object_tracking", {})
    return {
        "input": report.get("streamlit_input_provenance"),
        "runtime": report.get("runtime_provenance"),
        "heuristic_score": report.get("overall_score"),
        "viewpoint_label": viewpoint.get("label"),
        "tracked_sample_count": tracking.get("tracked_sample_count"),
        "sample_count": tracking.get("sample_count"),
        "tracked_sample_fraction": tracking.get("tracked_sample_fraction"),
        "cycle_verified_step_fraction": tracking.get(
            "cycle_verified_step_fraction"
        ),
        "foreground_separation": tracking.get("foreground_separation"),
        "tracker_settings": tracking.get("tracker_settings"),
        "sample_statuses": tracking.get("sample_statuses", []),
        "transition_attempts": tracking.get("steps", []),
        "background_diagnostics": viewpoint.get("background_diagnostics"),
        "performance": result.get("upload_performance"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the exact Streamlit object-focused analysis call."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--bbox", required=True, type=_bbox)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = replay(args.video, args.bbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"{args.video.name}: {output['tracked_sample_count']}/"
        f"{output['sample_count']} samples tracked; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
