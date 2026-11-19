"""Minimal independent Ridge-versus-COLMAP diagnostic validation workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..reconstruction import run_sparse_colmap
from .experimental_ridge import predict_experimental_ridge_from_prepared_images
from .historical_registration_features import (
    HISTORICAL_INPUT_FRAME_COUNT,
    LIVE_VIDEO_ADAPTER_CONTRACT,
    LIVE_VIDEO_ADAPTER_CONTRACT_SHA256,
    prepare_video_to_historical_60_jpegs,
)


CSV_FIELDS = (
    "video_id",
    "role",
    "filename",
    "video_sha256",
    "predicted_registration_percent",
    "actual_registered_images",
    "total_input_images",
    "actual_registration_percent",
    "absolute_error_percentage_points",
    "ground_truth_valid",
    "reconstruction_status",
    "raw_colmap_status",
    "failure_stage",
    "failure_reason",
    "preparation_seconds",
    "ridge_prediction_seconds",
    "colmap_seconds",
    "total_execution_seconds",
)

FROZEN_COLMAP_SETTINGS = {
    "camera_model": "SIMPLE_RADIAL",
    "single_camera": True,
    "matcher": "sequential",
    "sequential_overlap": 6,
    "retry_exhaustive": False,
    "use_gpu": False,
    "max_image_size": 800,
    "max_num_features": 2048,
    "num_threads": 2,
}

HISTORICAL_COMPARISON = {
    "matched": [
        "60 ordered inputs",
        "CPU SIFT",
        "maximum feature-extraction image size 800",
        "maximum 2048 SIFT features when supported by installed COLMAP",
        "two extraction/matching/mapper threads when supported",
        "single shared camera",
        "sequential matcher overlap 6",
        "no automatic exhaustive-matcher retry",
    ],
    "differences": [
        "Historical labels used prepared CO3D images; this pilot uses codec-decoded uploaded videos and the frozen JPEG adapter.",
        "The historical notebooks relied on the installed ImageReader camera-model default; this pilot explicitly freezes SIMPLE_RADIAL.",
        "COLMAP build/version may differ from the historical Colab runtime and is recorded for each pilot run.",
        "The pilot exports and analyzes the selected sparse model using repository utilities; that export does not change registration count.",
    ],
}


class IndependentValidationError(RuntimeError):
    """Raised for invalid manifests or unsafe output destinations."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_validation_manifest(path: Path) -> List[Dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise IndependentValidationError(f"Validation manifest is missing: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"video_id", "role", "video_path"}
    if not rows or not required.issubset(rows[0]):
        raise IndependentValidationError(
            "Manifest must contain video_id, role and video_path columns"
        )
    records: List[Dict[str, Any]] = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        video_id = (row.get("video_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", video_id):
            raise IndependentValidationError(
                f"Row {row_number} has an unsafe or empty video_id"
            )
        if video_id in seen:
            raise IndependentValidationError(f"Duplicate video_id: {video_id}")
        seen.add(video_id)
        raw_path = Path((row.get("video_path") or "").strip()).expanduser()
        video_path = raw_path if raw_path.is_absolute() else source.parent / raw_path
        video_path = video_path.resolve()
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise IndependentValidationError(
                f"Video for {video_id} is missing or empty: {video_path}"
            )
        records.append(
            {
                "video_id": video_id,
                "role": (row.get("role") or "unspecified").strip(),
                "video_path": video_path,
            }
        )
    return records


def classify_colmap_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Interpret only completed, objective COLMAP outcomes as ground truth."""

    raw_status = result.get("status")
    stage = result.get("failure_stage")
    if raw_status == "succeeded":
        metrics = result.get("objective_metrics")
        try:
            registered = int(metrics["registered_image_count"])
        except (KeyError, TypeError, ValueError):
            return {
                "reconstruction_status": "pipeline_failure",
                "ground_truth_valid": False,
                "actual_registered_images": None,
                "failure_reason": "Succeeded COLMAP result has no valid registered-image count",
            }
        if not 0 <= registered <= HISTORICAL_INPUT_FRAME_COUNT:
            return {
                "reconstruction_status": "pipeline_failure",
                "ground_truth_valid": False,
                "actual_registered_images": None,
                "failure_reason": "Registered-image count is outside 0..60",
            }
        return {
            "reconstruction_status": "succeeded",
            "ground_truth_valid": True,
            "actual_registered_images": registered,
            "failure_reason": None,
        }
    if raw_status == "reconstruction_failed" and stage == "sparse_mapping":
        commands = result.get("commands") or []
        if commands and all(int(command.get("returncode", -1)) == 0 for command in commands):
            return {
                "reconstruction_status": "completed_mapper_failure",
                "ground_truth_valid": True,
                "actual_registered_images": 0,
                "failure_reason": result.get("failure_reason"),
            }
    if raw_status == "infrastructure_unavailable":
        category = "infrastructure_failure"
    elif raw_status == "reconstruction_failed":
        category = "pipeline_failure"
    else:
        category = "pipeline_failure"
    return {
        "reconstruction_status": category,
        "ground_truth_valid": False,
        "actual_registered_images": None,
        "failure_reason": result.get("failure_reason") or "No valid completed COLMAP result",
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_validation_report(path: Path, rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    valid_errors = [
        float(row["absolute_error_percentage_points"])
        for row in rows
        if row.get("ground_truth_valid")
        and row.get("absolute_error_percentage_points") not in (None, "")
    ]
    mae = float(np.mean(valid_errors)) if valid_errors else None
    lines = [
        "# Independent Ridge/COLMAP diagnostic validation",
        "",
        "This is an initial diagnostic study, not a statistically sufficient accuracy evaluation.",
        "The Ridge model was not retrained and the three videos must be independent of its historical training groups.",
        "",
        "| Video | Role | Ridge prediction | Actual registration | Absolute error | Status | Total time |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        prediction = (
            f"{_format_value(row.get('predicted_registration_percent'))}%"
            if row.get("predicted_registration_percent") not in (None, "")
            else "Not available"
        )
        actual = (
            f"{_format_value(row.get('actual_registered_images'))}/60 "
            f"({_format_value(row.get('actual_registration_percent'))}%)"
            if row.get("ground_truth_valid")
            else "Not available"
        )
        error = (
            f"{_format_value(row.get('absolute_error_percentage_points'))} pp"
            if row.get("absolute_error_percentage_points") not in (None, "")
            else "Not available"
        )
        lines.append(
            "| {video_id} | {role} | {prediction} | {actual} | {error} | {status} | {seconds} s |".format(
                video_id=row["video_id"],
                role=row["role"],
                prediction=prediction,
                actual=actual,
                error=error,
                status=row["reconstruction_status"],
                seconds=_format_value(row.get("total_execution_seconds")),
            )
        )
    lines.extend(
        [
            "",
            f"Valid completed ground-truth evaluations: **{sum(bool(row.get('ground_truth_valid')) for row in rows)} / {len(rows)}**.",
            (
                f"Diagnostic MAE over {len(valid_errors)} valid prediction/ground-truth pairs: **{mae:.2f} percentage points**."
                if mae is not None
                else "Diagnostic MAE: **Not calculated** because no valid completed prediction/ground-truth pair exists."
            ),
            "",
            "A completed mapper run that produces no sparse model is recorded separately as `completed_mapper_failure` and contributes a verified 0/60. Missing COLMAP, command errors, interruptions, and export failures never become zero-registration labels and never enter MAE.",
            "",
            "## Historical comparability",
            "",
            "Matched settings: " + "; ".join(HISTORICAL_COMPARISON["matched"]) + ".",
            "",
            "Known differences:",
        ]
    )
    lines.extend(f"- {item}" for item in HISTORICAL_COMPARISON["differences"])
    lines.extend(
        [
            "",
            "Registration percentage is not proof of reconstruction quality, camera-orbit coverage, or captured surface percentage.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "video_count": len(rows),
        "valid_ground_truth_count": sum(
            bool(row.get("ground_truth_valid")) for row in rows
        ),
        "valid_error_count": len(valid_errors),
        "mae_percentage_points": round(mae, 6) if mae is not None else None,
    }


def _colmap_version(executable: str) -> Optional[str]:
    resolved = shutil.which(executable)
    if resolved is None:
        return None
    completed = subprocess.run(
        [resolved, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    first_line = next(
        (line.strip() for line in (completed.stdout or "").splitlines() if line.strip()),
        None,
    )
    return first_line


def run_independent_validation(
    records: List[Mapping[str, Any]],
    output_directory: Path,
    *,
    model_artifact: Optional[Path] = None,
    artifact_directory: Optional[Path] = None,
    colmap_executable: str = "colmap",
) -> Dict[str, Any]:
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise IndependentValidationError(
            f"Output directory is not empty; refusing to overwrite evidence: {destination}"
        )
    if model_artifact:
        model_path = Path(model_artifact).expanduser().resolve()
        if not model_path.is_file():
            raise IndependentValidationError(
                f"Ridge model artifact is missing: {model_path}"
            )
        model_bytes = model_path.read_bytes()
    else:
        model_bytes = None
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "independent_ridge_colmap_validation/1.0",
        "adapter_contract": dict(LIVE_VIDEO_ADAPTER_CONTRACT),
        "adapter_contract_sha256": LIVE_VIDEO_ADAPTER_CONTRACT_SHA256,
        "colmap_settings": dict(FROZEN_COLMAP_SETTINGS),
        "historical_comparison": HISTORICAL_COMPARISON,
        "colmap_executable": colmap_executable,
        "colmap_version": _colmap_version(colmap_executable),
        "model_retrained": False,
    }
    _write_json(destination / "validation_config.json", config)
    output_rows: List[Dict[str, Any]] = []
    for record in records:
        started = time.perf_counter()
        video_id = str(record["video_id"])
        video_path = Path(record["video_path"]).resolve()
        run_dir = destination / "runs" / video_id
        images_dir = run_dir / "images"
        colmap_dir = run_dir / "colmap"
        run_dir.mkdir(parents=True)
        row: Dict[str, Any] = {
            "video_id": video_id,
            "role": str(record.get("role") or "unspecified"),
            "filename": video_path.name,
            "video_sha256": _sha256_file(video_path),
            "predicted_registration_percent": None,
            "actual_registered_images": None,
            "total_input_images": HISTORICAL_INPUT_FRAME_COUNT,
            "actual_registration_percent": None,
            "absolute_error_percentage_points": None,
            "ground_truth_valid": False,
            "reconstruction_status": "preparation_failure",
            "raw_colmap_status": None,
            "failure_stage": "preparation",
            "failure_reason": None,
            "preparation_seconds": None,
            "ridge_prediction_seconds": None,
            "colmap_seconds": None,
            "total_execution_seconds": None,
        }
        try:
            source_hash_before = row["video_sha256"]
            stage_started = time.perf_counter()
            preparation = prepare_video_to_historical_60_jpegs(
                video_path, images_dir
            )
            row["preparation_seconds"] = round(
                time.perf_counter() - stage_started, 6
            )
            source_hash_after = _sha256_file(video_path)
            preparation["source_video_sha256_before"] = source_hash_before
            preparation["source_video_sha256_after"] = source_hash_after
            preparation["source_video_unchanged"] = source_hash_before == source_hash_after
            if not preparation["source_video_unchanged"]:
                raise IndependentValidationError("Source video changed during preparation")
            if len(list(images_dir.glob("*.jpg"))) != HISTORICAL_INPUT_FRAME_COUNT:
                raise IndependentValidationError("Adapter did not produce exactly 60 JPEGs")
            _write_json(run_dir / "preparation_manifest.json", preparation)

            stage_started = time.perf_counter()
            prediction = predict_experimental_ridge_from_prepared_images(
                images_dir,
                artifact_directory=artifact_directory,
                model_bytes=model_bytes,
            )
            row["ridge_prediction_seconds"] = round(
                time.perf_counter() - stage_started, 6
            )
            _write_json(run_dir / "ridge_prediction.json", prediction)
            if prediction.get("status") == "available":
                row["predicted_registration_percent"] = prediction[
                    "predicted_registration_percentage"
                ]

            row["reconstruction_status"] = "pipeline_failure"
            row["failure_stage"] = "colmap_execution"
            stage_started = time.perf_counter()
            colmap_result = run_sparse_colmap(
                images_dir,
                colmap_dir,
                colmap_executable=colmap_executable,
                camera_model=FROZEN_COLMAP_SETTINGS["camera_model"],
                sequential_overlap=FROZEN_COLMAP_SETTINGS["sequential_overlap"],
                retry_exhaustive=FROZEN_COLMAP_SETTINGS["retry_exhaustive"],
                use_gpu=FROZEN_COLMAP_SETTINGS["use_gpu"],
                max_image_size=FROZEN_COLMAP_SETTINGS["max_image_size"],
                max_num_features=FROZEN_COLMAP_SETTINGS["max_num_features"],
                num_threads=FROZEN_COLMAP_SETTINGS["num_threads"],
            )
            row["colmap_seconds"] = round(time.perf_counter() - stage_started, 6)
            interpreted = classify_colmap_result(colmap_result)
            row.update(interpreted)
            row["raw_colmap_status"] = colmap_result.get("status")
            row["failure_stage"] = colmap_result.get("failure_stage")
            if row["ground_truth_valid"]:
                registered = int(row["actual_registered_images"])
                actual_percent = 100.0 * registered / HISTORICAL_INPUT_FRAME_COUNT
                row["actual_registration_percent"] = round(actual_percent, 6)
                if row["predicted_registration_percent"] is not None:
                    row["absolute_error_percentage_points"] = round(
                        abs(float(row["predicted_registration_percent"]) - actual_percent),
                        6,
                    )
        except Exception as exc:
            row["failure_reason"] = str(exc) or exc.__class__.__name__
        row["total_execution_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(run_dir / "validation_record.json", row)
        output_rows.append(row)
        _write_csv(destination / "validation_results.csv", output_rows)
    summary = write_validation_report(
        destination / "VALIDATION_REPORT.md", output_rows
    )
    _write_json(destination / "validation_summary.json", summary)
    return {
        "rows": output_rows,
        "summary": summary,
        "csv": str(destination / "validation_results.csv"),
        "report": str(destination / "VALIDATION_REPORT.md"),
    }
