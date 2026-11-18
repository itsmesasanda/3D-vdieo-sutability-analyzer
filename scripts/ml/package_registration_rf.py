#!/usr/bin/env python3
"""Validate and hash a final COLMAP-registration Random Forest artifact."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

from video_suitability.ml.artifacts import write_manifest
from video_suitability.ml.registration_features import load_feature_spec
from video_suitability.ml.registration_rf import (
    MODEL_TYPE,
    TARGET_NAME,
    load_registration_model,
    verify_training_feature_parity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--record-count", type=int, required=True)
    parser.add_argument("--training-run-id", required=True)
    args = parser.parse_args()
    directory = args.artifact_dir
    if args.record_count <= 0:
        raise ValueError("--record-count must be positive")
    manifest_path = directory / "artifact_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "artifact_manifest.json already exists; preserve it and package a new versioned directory"
        )
    for filename in ("model.joblib", "feature_extractor.json", "parity_examples.json"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    spec = load_feature_spec(directory / "feature_extractor.json")
    parity = json.loads(
        (directory / "parity_examples.json").read_text(encoding="utf-8")
    )
    if parity.get("schema_version") != "registration_feature_parity/1.0":
        raise ValueError("unsupported parity_examples.json schema_version")
    if not isinstance(parity.get("cases"), list) or not parity["cases"]:
        raise ValueError("at least one saved training parity case is required")
    parity_result = verify_training_feature_parity(directory, spec, parity)
    model = load_registration_model(directory / "model.joblib")
    if getattr(model, "n_features_in_", None) != 8:
        raise ValueError("model.joblib must declare n_features_in_=8")
    declared_names = getattr(model, "feature_names_in_", None)
    if declared_names is not None and tuple(
        str(name) for name in declared_names
    ) != tuple(spec["feature_names_in_order"]):
        raise ValueError("model.joblib feature names/order do not match the feature spec")
    parity_files = []
    for case in parity["cases"]:
        input_type = case.get("input_type", "video")
        if input_type == "video":
            relative_files = [str(case["video"])]
        elif input_type == "colmap_frame_manifest":
            manifest_relative = str(case["frames_manifest"])
            manifest_path = (directory / manifest_relative).resolve()
            manifest_path.relative_to(directory.resolve())
            images_relative = Path(str(case["images_directory"]))
            with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
                image_rows = list(csv.DictReader(handle))
            relative_files = [manifest_relative] + [
                str(images_relative / str(row["filename"])) for row in image_rows
            ]
        else:
            raise ValueError(f"unsupported parity input_type: {input_type}")
        for relative in relative_files:
            path = (directory / relative).resolve()
            path.relative_to(directory.resolve())
            if not path.is_file():
                raise FileNotFoundError(path)
            parity_files.append(relative)
    files = [
        "model.joblib",
        "feature_extractor.json",
        "parity_examples.json",
        *parity_files,
    ]
    written = write_manifest(
        directory,
        model_type=MODEL_TYPE,
        model_version=args.model_version,
        training_scope="genuine_colmap_registration_labels",
        files=files,
        metadata={
            "target_name": TARGET_NAME,
            "target_units": "percent_0_to_100",
            "feature_count": 8,
            "record_count": args.record_count,
            "training_run_id": args.training_run_id,
            "python": platform.python_version(),
            "feature_parity": parity_result,
        },
    )
    print(
        json.dumps(
            {
                "artifact_manifest": str(written),
                "model_version": args.model_version,
                "parity": parity_result,
                "files": files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
