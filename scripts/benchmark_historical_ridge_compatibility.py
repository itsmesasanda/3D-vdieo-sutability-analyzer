#!/usr/bin/env python3
"""Compare the recreated legacy extractor with the labeling-notebook function."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from video_suitability.ml.historical_registration_features import (
    HISTORICAL_EXTRACTOR_SOURCE_SHA256,
    HISTORICAL_FEATURE_NAMES,
    extract_historical_features_from_image_directory,
    prepare_video_for_historical_probe,
)


def _reference_function(notebook_path: Path):
    import cv2

    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    matches = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def extract_features(image_dir):" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "extract_features":
                function_source = ast.get_source_segment(source, node)
                matches.append(function_source)
    if len(matches) != 1:
        raise ValueError("Expected exactly one extract_features function in notebook")
    source = matches[0]
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != HISTORICAL_EXTRACTOR_SOURCE_SHA256:
        raise ValueError(f"Historical notebook function hash mismatch: {digest}")
    namespace = {"np": np, "cv2": cv2}
    exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace["extract_features"], digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference, source_hash = _reference_function(args.notebook)
    import cv2
    import PIL

    rows = []
    for video in args.video:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        with tempfile.TemporaryDirectory(prefix="historical-reference-") as temporary:
            image_root = Path(temporary) / "images"
            preparation = prepare_video_for_historical_probe(video, image_root)
            recreated, diagnostics = extract_historical_features_from_image_directory(
                image_root
            )
            reference_values = reference(image_root)
            reference_order = tuple(reference_values)
            recreated_order = tuple(recreated)
            differences = {
                name: abs(float(recreated[name]) - float(reference_values[name]))
                for name in HISTORICAL_FEATURE_NAMES
            }
            rows.append(
                {
                    "video": str(video.resolve()),
                    "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "reference_feature_order": list(reference_order),
                    "recreated_feature_order": list(recreated_order),
                    "feature_order_exact": reference_order
                    == recreated_order
                    == HISTORICAL_FEATURE_NAMES,
                    "features": recreated,
                    "maximum_absolute_feature_difference": max(differences.values()),
                    "numerical_parity": all(value == 0.0 for value in differences.values()),
                    "preprocessing": preparation,
                    "diagnostics": diagnostics,
                    "wall_seconds": round(time.perf_counter() - wall_started, 6),
                    "cpu_seconds": round(time.process_time() - cpu_started, 6),
                }
            )
    report = {
        "schema_version": "historical_ridge_video_compatibility/1.0",
        "historical_function_source_sha256": source_hash,
        "runtime": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "videos": rows,
        "prepared_input_function_parity": all(
            row["feature_order_exact"] and row["numerical_parity"] for row in rows
        ),
        "live_ridge_inference_authorized": False,
        "blockers": [
            "Historical training began with prepared CO3D JPEG directories; no executed upload-video adapter exists.",
            "Training-time prepared frames/hashes are unavailable, so video-adapter parity cannot be checked against saved training examples.",
            "Historical ORB and resize defaults were not fully version-pinned across labeling batches.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
