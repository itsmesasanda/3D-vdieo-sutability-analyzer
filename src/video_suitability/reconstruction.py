"""Reproducible sparse-COLMAP experiment helpers.

This module is intentionally separate from heuristic inference.  It never changes
heuristic scores and it never treats unavailable COLMAP infrastructure as a failed
reconstruction.  Sparse reconstruction metrics are evidence, not an object-
completeness rating.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class ReconstructionError(RuntimeError):
    """Raised for invalid experiment inputs or failed reconstruction stages."""


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ReconstructionError("OpenCV is required for frame extraction") from exc
    return cv2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _codec_name(raw_fourcc: float) -> str:
    value = int(raw_fourcc)
    chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(char for char in chars if char.isprintable()).strip("\x00 ") or "unknown"


def extraction_indices(frame_count: int, fps: float, sample_fps: float) -> List[int]:
    """Return ordered indices spanning the source timeline at the target cadence."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("source fps must be positive")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if sample_fps >= fps:
        return list(range(frame_count))
    step = fps / sample_fps
    indices = np.rint(np.arange(0.0, frame_count, step)).astype(int)
    result = sorted(set(int(np.clip(index, 0, frame_count - 1)) for index in indices))
    if result[-1] != frame_count - 1:
        result.append(frame_count - 1)
    return result


def _resize(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return frame, 1.0
    scale = max_side / float(largest)
    resized = cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def extract_reconstruction_frames(
    video_path: Path,
    output_dir: Path,
    *,
    sample_fps: float = 2.0,
    max_side: int = 1600,
    jpeg_quality: int = 95,
) -> Dict[str, Any]:
    """Extract timestamped, oriented frames without changing the source video.

    OpenCV's orientation auto-rotation is explicitly enabled.  The source orientation
    tag and both encoded/decoded dimensions are preserved in the metadata so COLMAP's
    image geometry is auditable.
    """

    cv2 = _cv2()
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ReconstructionError(f"Source video is missing or empty: {source}")
    if max_side < 320:
        raise ReconstructionError("max_side must be at least 320 pixels")
    if not 1 <= jpeg_quality <= 100:
        raise ReconstructionError("jpeg_quality must be between 1 and 100")
    if destination.exists() and any(destination.iterdir()):
        raise ReconstructionError(
            f"Output directory is not empty; refusing to overwrite evidence: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    images_dir = destination / "images"
    images_dir.mkdir()

    source_hash_before = sha256_file(source)
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise ReconstructionError(f"OpenCV could not open source video: {source}")

    orientation_auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    orientation_meta_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_auto_property is not None:
        cap.set(orientation_auto_property, 1.0)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    encoded_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    encoded_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = _codec_name(cap.get(cv2.CAP_PROP_FOURCC))
    orientation_degrees = (
        float(cap.get(orientation_meta_property))
        if orientation_meta_property is not None
        else None
    )
    orientation_auto = (
        bool(round(cap.get(orientation_auto_property)))
        if orientation_auto_property is not None
        else None
    )
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ReconstructionError(
            "The codec did not provide a usable frame count and FPS; deterministic "
            "timestamped extraction cannot proceed."
        )

    requested_indices = extraction_indices(frame_count, fps, sample_fps)
    rows: List[Dict[str, Any]] = []
    failed_indices: List[int] = []
    decoded_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    output_scale: Optional[float] = None
    for sequence_number, requested_index in enumerate(requested_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(requested_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            failed_indices.append(requested_index)
            continue
        reported_next = cap.get(cv2.CAP_PROP_POS_FRAMES)
        actual_index = (
            max(0, int(round(reported_next)) - 1) if reported_next > 0 else requested_index
        )
        timestamp = actual_index / fps
        decoded_height, decoded_width = frame.shape[:2]
        resized, scale = _resize(frame, max_side)
        output_height, output_width = resized.shape[:2]
        filename = f"frame_{sequence_number:06d}_t{timestamp:010.3f}.jpg"
        output_path = images_dir / filename
        if not cv2.imwrite(
            str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        ):
            cap.release()
            raise ReconstructionError(f"OpenCV failed to write extracted frame: {output_path}")
        decoded_shape = (decoded_width, decoded_height)
        output_shape = (output_width, output_height)
        output_scale = scale
        rows.append(
            {
                "sequence_number": len(rows),
                "filename": filename,
                "requested_source_frame_index": requested_index,
                "actual_source_frame_index": actual_index,
                "timestamp_seconds": round(timestamp, 9),
                "width": output_width,
                "height": output_height,
                "sha256": sha256_file(output_path),
            }
        )
    cap.release()

    if len(rows) < 2:
        raise ReconstructionError("Fewer than two frames could be extracted")
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before:
        raise ReconstructionError("Source checksum changed during extraction")

    manifest_path = destination / "frames_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    duration = frame_count / fps
    metadata: Dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_stage": "milestone_3a_sparse_reconstruction",
        "source_video": str(source),
        "source_filename": source.name,
        "source_size_bytes": source.stat().st_size,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_unchanged_after_extraction": True,
        "source_codec_fourcc": codec,
        "source_fps": round(fps, 9),
        "source_frame_count": frame_count,
        "source_duration_seconds": round(duration, 9),
        "encoded_width": encoded_width,
        "encoded_height": encoded_height,
        "orientation_metadata_degrees": orientation_degrees,
        "opencv_orientation_auto_enabled": orientation_auto,
        "decoded_width_after_orientation": decoded_shape[0] if decoded_shape else None,
        "decoded_height_after_orientation": decoded_shape[1] if decoded_shape else None,
        "requested_sample_fps": sample_fps,
        "sampling_strategy": "source-frame indices at a fixed cadence across the full timeline, including the final frame",
        "requested_frame_count": len(requested_indices),
        "extracted_frame_count": len(rows),
        "failed_source_frame_indices": failed_indices,
        "first_timestamp_seconds": rows[0]["timestamp_seconds"],
        "last_timestamp_seconds": rows[-1]["timestamp_seconds"],
        "temporal_coverage_fraction": round(
            rows[-1]["actual_source_frame_index"] / max(frame_count - 1, 1), 9
        ),
        "output_width": output_shape[0] if output_shape else None,
        "output_height": output_shape[1] if output_shape else None,
        "linear_resize_scale": round(float(output_scale), 9)
        if output_scale is not None
        else None,
        "jpeg_quality": jpeg_quality,
        "images_directory": str(images_dir),
        "frames_manifest": str(manifest_path),
        "extraction_runtime_seconds": round(time.perf_counter() - started, 6),
    }
    _json_dump(destination / "extraction_metadata.json", metadata)
    return metadata


def _quaternion_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ReconstructionError("Invalid zero-length COLMAP image quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _parse_colmap_images(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: List[Dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        parts = line.split()
        # Pose records have exactly ten fixed fields plus a possibly spaced image name.
        if len(parts) < 10:
            raise ReconstructionError(f"Malformed COLMAP images.txt pose line: {line}")
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = (float(value) for value in parts[1:5])
            tx, ty, tz = (float(value) for value in parts[5:8])
            camera_id = int(parts[8])
        except ValueError as exc:
            raise ReconstructionError(f"Malformed COLMAP images.txt pose line: {line}") from exc
        name = " ".join(parts[9:])
        rotation = _quaternion_rotation_matrix(qw, qx, qy, qz)
        translation = np.array([tx, ty, tz], dtype=np.float64)
        center = -(rotation.T @ translation)
        records.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "qvec_world_to_camera": np.array([qw, qx, qy, qz], dtype=np.float64),
                "tvec_world_to_camera": translation,
                "camera_center": center,
            }
        )
        # The following POINTS2D line is part of this image even when blank.
        index += 2
    return records


def _parse_colmap_points(path: Path) -> Tuple[np.ndarray, List[float], List[int]]:
    xyz: List[List[float]] = []
    errors: List[float] = []
    track_lengths: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                raise ReconstructionError(f"Malformed COLMAP points3D.txt line: {line}")
            try:
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
                errors.append(float(parts[7]))
            except ValueError as exc:
                raise ReconstructionError(f"Malformed COLMAP points3D.txt line: {line}") from exc
            track_lengths.append(max(0, (len(parts) - 8) // 2))
    array = np.asarray(xyz, dtype=np.float64)
    if not xyz:
        array = np.empty((0, 3), dtype=np.float64)
    return array, errors, track_lengths


def analyze_colmap_text_model(model_dir: Path, total_input_images: int) -> Dict[str, Any]:
    """Measure an exported sparse model without inferring subjective completeness."""

    directory = Path(model_dir)
    images_path = directory / "images.txt"
    points_path = directory / "points3D.txt"
    if not images_path.is_file() or not points_path.is_file():
        raise ReconstructionError(f"COLMAP text model is incomplete: {directory}")
    images = _parse_colmap_images(images_path)
    points, errors, track_lengths = _parse_colmap_points(points_path)
    centers = (
        np.stack([record["camera_center"] for record in images])
        if images
        else np.empty((0, 3), dtype=np.float64)
    )
    trajectory_length = None
    max_baseline = None
    camera_center_extent = None
    if len(centers) >= 2:
        ordered = sorted(
            zip(images, centers), key=lambda item: (item[0]["name"], item[0]["image_id"])
        )
        ordered_centers = np.stack([item[1] for item in ordered])
        trajectory_length = float(np.linalg.norm(np.diff(ordered_centers, axis=0), axis=1).sum())
        differences = centers[:, None, :] - centers[None, :, :]
        max_baseline = float(np.linalg.norm(differences, axis=2).max())
        camera_center_extent = [float(value) for value in np.ptp(centers, axis=0)]

    registered = len(images)
    return {
        "registered_image_count": registered,
        "total_input_image_count": int(total_input_images),
        "registered_image_ratio": round(registered / total_input_images, 9)
        if total_input_images > 0
        else None,
        "reconstructed_point_count": int(len(points)),
        "mean_reprojection_error_pixels": round(float(np.mean(errors)), 9)
        if errors
        else None,
        "median_reprojection_error_pixels": round(float(np.median(errors)), 9)
        if errors
        else None,
        "mean_track_length_images": round(float(np.mean(track_lengths)), 9)
        if track_lengths
        else None,
        "median_track_length_images": round(float(np.median(track_lengths)), 9)
        if track_lengths
        else None,
        "camera_center_trajectory_length_arbitrary_units": round(trajectory_length, 9)
        if trajectory_length is not None
        else None,
        "camera_center_max_baseline_arbitrary_units": round(max_baseline, 9)
        if max_baseline is not None
        else None,
        "camera_center_extent_xyz_arbitrary_units": camera_center_extent,
        "coverage_interpretation": (
            "Registration ratio and camera-center extent are geometry support indicators only. "
            "They do not measure object surface completeness, and COLMAP scale is arbitrary."
        ),
        "registered_images": [record["name"] for record in images],
        "camera_centers": [
            {
                "image_id": record["image_id"],
                "camera_id": record["camera_id"],
                "name": record["name"],
                "qvec_world_to_camera_qw_qx_qy_qz": [
                    round(float(v), 12) for v in record["qvec_world_to_camera"]
                ],
                "tvec_world_to_camera_xyz": [
                    round(float(v), 12) for v in record["tvec_world_to_camera"]
                ],
                "xyz_arbitrary_units": [round(float(v), 9) for v in record["camera_center"]],
            }
            for record in images
        ],
    }


def render_sparse_preview(model_dir: Path, output_path: Path, size: int = 1200) -> Path:
    """Render XY/XZ/YZ sparse-point and camera-center projections with OpenCV."""

    cv2 = _cv2()
    directory = Path(model_dir)
    points, _, _ = _parse_colmap_points(directory / "points3D.txt")
    images = _parse_colmap_images(directory / "images.txt")
    centers = (
        np.stack([record["camera_center"] for record in images])
        if images
        else np.empty((0, 3), dtype=np.float64)
    )
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    panel_width = size // 3
    projections = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    for panel, (axis_a, axis_b, label) in enumerate(projections):
        left = panel * panel_width
        cv2.rectangle(canvas, (left, 0), (left + panel_width - 1, size - 1), (200, 200, 200), 1)
        cv2.putText(canvas, label, (left + 18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        combined = np.concatenate([points, centers], axis=0) if len(centers) else points
        if len(combined) == 0:
            cv2.putText(canvas, "no geometry", (left + 18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 180), 2)
            continue
        xy = combined[:, [axis_a, axis_b]]
        low = np.percentile(xy, 1, axis=0)
        high = np.percentile(xy, 99, axis=0)
        span = np.maximum(high - low, 1e-9)

        def project(values: np.ndarray) -> np.ndarray:
            normalized = np.clip((values - low) / span, 0.0, 1.0)
            result = np.empty_like(normalized)
            result[:, 0] = left + 24 + normalized[:, 0] * (panel_width - 48)
            result[:, 1] = size - 30 - normalized[:, 1] * (size - 90)
            return np.rint(result).astype(int)

        if len(points):
            for x, y in project(points[:, [axis_a, axis_b]]):
                cv2.circle(canvas, (int(x), int(y)), 1, (90, 90, 90), -1)
        if len(centers):
            projected_centers = project(centers[:, [axis_a, axis_b]])
            for first, second in zip(projected_centers, projected_centers[1:]):
                cv2.line(canvas, tuple(first), tuple(second), (210, 120, 20), 2, cv2.LINE_AA)
            for x, y in projected_centers:
                cv2.circle(canvas, (int(x), int(y)), 5, (20, 80, 220), -1, cv2.LINE_AA)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ReconstructionError(f"Could not write sparse preview: {output}")
    return output


@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    runtime_seconds: float
    log_path: str


def _run_logged(command: Sequence[str], log_path: Path) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    runtime = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.write(completed.stdout or "")
    return CommandResult(list(command), completed.returncode, runtime, str(log_path))


def _help_text(executable: str, subcommand: str) -> str:
    completed = subprocess.run(
        [executable, subcommand, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout or ""


def _supported_flag(help_text: str, candidates: Sequence[str]) -> Optional[str]:
    return next((candidate for candidate in candidates if candidate in help_text), None)


def run_sparse_colmap(
    images_dir: Path,
    workspace_dir: Path,
    *,
    colmap_executable: str = "colmap",
    camera_model: str = "OPENCV",
    sequential_overlap: int = 10,
    retry_exhaustive: bool = False,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """Run sparse COLMAP and return explicit status plus objective metrics.

    CPU is the deterministic default. ``use_gpu`` only enables acceleration when
    the installed COLMAP help exposes GPU switches for both feature extraction and
    matching; otherwise the runner records and uses the CPU path.
    """

    images = Path(images_dir).resolve()
    workspace = Path(workspace_dir).resolve()
    image_files = sorted(images.glob("*.jpg"))
    base: Dict[str, Any] = {
        "schema_version": "1.0",
        "analysis_type": "sparse_colmap_reconstruction",
        "dense_reconstruction_run": False,
        "images_directory": str(images),
        "total_input_image_count": len(image_files),
        "camera_model": camera_model,
        "sequential_overlap": sequential_overlap,
        "retry_exhaustive": retry_exhaustive,
        "gpu_requested": bool(use_gpu),
        "selected_device": "unresolved",
        "colmap_executable_requested": colmap_executable,
        "commands": [],
        "objective_metrics": None,
    }
    executable = shutil.which(colmap_executable)
    if executable is None:
        base.update(
            {
                "status": "infrastructure_unavailable",
                "failure_stage": "environment",
                "failure_reason": f"COLMAP executable was not found: {colmap_executable}",
                "mapper_runtime_seconds": None,
                "total_runtime_seconds": 0.0,
            }
        )
        _json_dump(workspace / "reconstruction_result.json", base)
        return base
    if len(image_files) < 2:
        raise ReconstructionError(f"Need at least two JPEG frames in {images}")
    if workspace.exists() and any(workspace.iterdir()):
        raise ReconstructionError(
            f"COLMAP workspace is not empty; refusing to overwrite evidence: {workspace}"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    logs = workspace / "logs"
    database = workspace / "database.db"
    sparse_sequential = workspace / "sparse_sequential"
    sparse_sequential.mkdir()

    feature_help = _help_text(executable, "feature_extractor")
    matching_help = _help_text(executable, "sequential_matcher")
    feature_gpu_flag = _supported_flag(
        feature_help, ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"]
    )
    matching_gpu_flag = _supported_flag(
        matching_help, ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"]
    )
    gpu_flags_supported = bool(feature_gpu_flag and matching_gpu_flag)
    gpu_enabled = bool(use_gpu and gpu_flags_supported)
    base["gpu_flags_supported"] = gpu_flags_supported
    base["selected_device"] = "gpu" if gpu_enabled else "cpu"
    feature_command = [
        executable,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
    ]
    if feature_gpu_flag:
        feature_command.extend([feature_gpu_flag, "1" if gpu_enabled else "0"])
    sequential_command = [
        executable,
        "sequential_matcher",
        "--database_path",
        str(database),
        "--SequentialMatching.overlap",
        str(sequential_overlap),
    ]
    if matching_gpu_flag:
        sequential_command.extend([matching_gpu_flag, "1" if gpu_enabled else "0"])
    mapper_command = [
        executable,
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_sequential),
    ]

    started = time.perf_counter()
    results: List[CommandResult] = []
    for stage, command in [
        ("feature_extractor", feature_command),
        ("sequential_matcher", sequential_command),
        ("mapper_sequential", mapper_command),
    ]:
        result = _run_logged(command, logs / f"{stage}.log")
        results.append(result)
        base["commands"].append(result.__dict__)
        if result.returncode != 0:
            base.update(
                {
                    "status": "reconstruction_failed",
                    "failure_stage": stage,
                    "failure_reason": f"COLMAP {stage} exited with code {result.returncode}",
                    "mapper_runtime_seconds": next(
                        (item.runtime_seconds for item in results if "mapper" in item.log_path),
                        None,
                    ),
                    "total_runtime_seconds": round(time.perf_counter() - started, 6),
                }
            )
            _json_dump(workspace / "reconstruction_result.json", base)
            return base

    model_dirs = sorted(path for path in sparse_sequential.iterdir() if path.is_dir())
    if not model_dirs and retry_exhaustive:
        exhaustive_help = _help_text(executable, "exhaustive_matcher")
        exhaustive_gpu_flag = _supported_flag(
            exhaustive_help, ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"]
        )
        exhaustive_command = [executable, "exhaustive_matcher", "--database_path", str(database)]
        if exhaustive_gpu_flag:
            exhaustive_command.extend([exhaustive_gpu_flag, "1" if gpu_enabled else "0"])
        sparse_exhaustive = workspace / "sparse_exhaustive"
        sparse_exhaustive.mkdir()
        exhaustive_mapper_command = [
            executable,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(sparse_exhaustive),
        ]
        for stage, command in [
            ("exhaustive_matcher", exhaustive_command),
            ("mapper_exhaustive", exhaustive_mapper_command),
        ]:
            result = _run_logged(command, logs / f"{stage}.log")
            results.append(result)
            base["commands"].append(result.__dict__)
            if result.returncode != 0:
                break
        model_dirs = sorted(path for path in sparse_exhaustive.iterdir() if path.is_dir())

    mapper_runtime = sum(
        result.runtime_seconds
        for result in results
        if Path(result.log_path).stem.startswith("mapper")
    )
    if not model_dirs:
        base.update(
            {
                "status": "reconstruction_failed",
                "failure_stage": "sparse_mapping",
                "failure_reason": "COLMAP mapper produced no sparse model",
                "mapper_runtime_seconds": round(mapper_runtime, 6),
                "total_runtime_seconds": round(time.perf_counter() - started, 6),
            }
        )
        _json_dump(workspace / "reconstruction_result.json", base)
        return base

    converted_root = workspace / "model_text_candidates"
    converted_root.mkdir()
    candidate_metrics: List[Tuple[Dict[str, Any], Path, Path]] = []
    for candidate_index, model_dir in enumerate(model_dirs):
        converted = converted_root / str(candidate_index)
        converted.mkdir()
        command = [
            executable,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(converted),
            "--output_type",
            "TXT",
        ]
        result = _run_logged(command, logs / f"model_converter_{candidate_index}.log")
        base["commands"].append(result.__dict__)
        if result.returncode == 0:
            candidate_metrics.append(
                (analyze_colmap_text_model(converted, len(image_files)), model_dir, converted)
            )
    if not candidate_metrics:
        base.update(
            {
                "status": "reconstruction_failed",
                "failure_stage": "model_export",
                "failure_reason": "No mapper model could be converted to COLMAP text format",
                "mapper_runtime_seconds": round(mapper_runtime, 6),
                "total_runtime_seconds": round(time.perf_counter() - started, 6),
            }
        )
        _json_dump(workspace / "reconstruction_result.json", base)
        return base
    metrics, selected_binary, selected_text = max(
        candidate_metrics,
        key=lambda item: (
            item[0]["registered_image_count"], item[0]["reconstructed_point_count"]
        ),
    )
    analyzer = _run_logged(
        [executable, "model_analyzer", "--path", str(selected_binary)],
        logs / "model_analyzer.log",
    )
    base["commands"].append(analyzer.__dict__)
    ply_path = workspace / "sparse_points_and_cameras.ply"
    converter = _run_logged(
        [
            executable,
            "model_converter",
            "--input_path",
            str(selected_binary),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ],
        logs / "model_converter_ply.log",
    )
    base["commands"].append(converter.__dict__)
    preview_path = workspace / "sparse_preview.png"
    render_sparse_preview(selected_text, preview_path)
    base.update(
        {
            "status": "succeeded",
            "failure_stage": None,
            "failure_reason": None,
            "selected_sparse_model": str(selected_binary),
            "selected_text_model": str(selected_text),
            "sparse_ply": str(ply_path) if converter.returncode == 0 else None,
            "sparse_preview": str(preview_path),
            "objective_metrics": metrics,
            "mapper_runtime_seconds": round(mapper_runtime, 6),
            "total_runtime_seconds": round(time.perf_counter() - started, 6),
        }
    )
    _json_dump(workspace / "reconstruction_result.json", base)
    return base

