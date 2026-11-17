"""Numerical CV features plus optional frozen ResNet-18 embedding caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import AnalysisConfig
from ..datasets.co3d import CO3DSequence
from ..geometry import analyze_geometry
from ..quality import analyze_quality


NUMERIC_FEATURE_VERSION = "cv_numeric_features/2.1"
QUALITY_FEATURES = (
    "median_luminance",
    "luminance_std_across_frames",
    "median_contrast_std",
    "median_laplacian_variance",
    "median_gradient_magnitude",
    "median_entropy_bits",
    "median_dark_pixel_fraction",
    "median_bright_pixel_fraction",
    "underexposed_frame_fraction",
    "overexposed_frame_fraction",
    "blurry_frame_fraction",
    "low_contrast_frame_fraction",
    "low_texture_frame_fraction",
)
GEOMETRY_FEATURES = (
    "median_keypoints_per_frame",
    "minimum_keypoints_per_frame",
    "low_feature_frame_fraction",
    "pair_count",
    "valid_geometry_pair_count",
    "valid_geometry_pair_fraction",
    "median_ratio_test_matches",
    "median_verified_matches",
    "median_match_ratio",
    "median_verified_match_ratio",
    "median_inlier_ratio",
    "median_spatial_coverage",
    "median_displacement_normalized",
    "median_displacement_normalized_per_second",
    "median_homography_residual_p75_normalized",
    "moderate_motion_pair_fraction",
    "near_stationary_pair_fraction",
    "excessive_motion_pair_fraction",
    "low_overlap_pair_fraction",
    "rotation_or_planar_pair_fraction",
    "homography_dominant_pair_fraction",
    "median_homography_inlier_ratio",
    "median_descriptor_ambiguity_fraction",
    "high_descriptor_ambiguity_pair_fraction",
    "median_flow_track_retention",
    "median_flow_track_retention_short_gap",
    "optical_flow_short_gap_pair_fraction",
    "motion_sampling_gap_too_wide_pair_fraction",
    "low_track_retention_pair_fraction",
    "median_flow_speed_normalized_per_second",
    "p90_flow_speed_normalized_per_second",
    "flow_speed_robust_cv",
    "excessive_camera_speed_pair_fraction",
    "irregular_camera_motion_pair_fraction",
    "irregular_camera_motion_evaluable_pair_fraction",
    "p90_motion_direction_change_degrees",
    "median_abs_rotation_rate_degrees_per_second",
    "rotation_rate_mad_degrees_per_second",
    "median_affine_residual_normalized",
    "cumulative_flow_motion_normalized",
)
VIDEO_FEATURES = (
    "duration_seconds",
    "fps",
    "frame_count",
    "source_megapixels",
    "estimated_bitrate_mbps",
    "compressed_bits_per_pixel_per_frame",
    "decoded_sample_count",
    "temporal_coverage_fraction",
    "full_temporal_coverage_confirmed",
)

TIMESTAMP_DEPENDENT_GEOMETRY_FEATURES = {
    "median_displacement_normalized_per_second",
    "median_flow_speed_normalized_per_second",
    "p90_flow_speed_normalized_per_second",
    "flow_speed_robust_cv",
    "excessive_camera_speed_pair_fraction",
    "irregular_camera_motion_pair_fraction",
    "irregular_camera_motion_evaluable_pair_fraction",
    "p90_motion_direction_change_degrees",
    "median_abs_rotation_rate_degrees_per_second",
    "rotation_rate_mad_degrees_per_second",
}


def _nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def numeric_features_from_report(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    raw = report["raw_metrics"]
    quality = raw["quality_summary"]
    geometry = raw["geometry_summary"]
    video = raw["video"]
    output: Dict[str, Optional[float]] = {}
    for name in QUALITY_FEATURES:
        output[f"quality.{name}"] = _nullable_float(quality.get(name))
    for name in GEOMETRY_FEATURES:
        output[f"geometry.{name}"] = _nullable_float(geometry.get(name))
    for name in VIDEO_FEATURES:
        output[f"video.{name}"] = _nullable_float(video.get(name))
    return output


def _resize_for_features(frame: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def numeric_features_from_co3d_sequence(
    sequence: CO3DSequence,
    *,
    sample_count: int = 32,
    config: Optional[AnalysisConfig] = None,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    """Extract the existing v2.1 CV measurements without creating a heuristic label.

    CO3D annotations do not universally expose reliable capture timestamps. When
    timestamps are absent or non-monotonic, non-temporal geometry remains available
    but every speed/irregularity feature is explicitly missing rather than computed
    from an invented FPS.
    """

    import cv2

    cfg = config or AnalysisConfig(max_samples=sample_count)
    cfg.validate()
    sampled = sequence.uniform_sample(sample_count)
    frames = []
    for frame in sampled:
        decoded = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Could not decode CO3D frame: {frame.image_path}")
        frames.append(_resize_for_features(decoded, cfg.max_analysis_side))
    position_by_identity = {id(frame): index for index, frame in enumerate(sequence.frames)}
    sequence_positions = [position_by_identity[id(frame)] for frame in sampled]
    annotated_timestamps = [frame.timestamp_seconds for frame in sampled]
    has_reliable_timestamps = len(annotated_timestamps) >= 2 and all(
        value is not None for value in annotated_timestamps
    )
    if has_reliable_timestamps:
        timestamps = [float(value) for value in annotated_timestamps]
        has_reliable_timestamps = all(
            second > first for first, second in zip(timestamps, timestamps[1:])
        )
    if not has_reliable_timestamps:
        # Ordinal positions let RANSAC/overlap metrics run. All features whose
        # semantics depend on physical time are nulled below.
        timestamps = [float(index) for index in range(len(sampled))]

    quality = analyze_quality(frames, sequence_positions, timestamps)
    geometry = analyze_geometry(frames, sequence_positions, timestamps, cfg)
    sizes = [frame.image_size_hw for frame in sequence.frames if frame.image_size_hw]
    megapixels = (
        float(np.median([height * width for height, width in sizes])) / 1_000_000.0
        if sizes
        else None
    )
    full_timestamps = [frame.timestamp_seconds for frame in sequence.frames]
    full_timing_reliable = len(full_timestamps) >= 2 and all(
        value is not None for value in full_timestamps
    )
    if full_timing_reliable:
        full_values = np.asarray(full_timestamps, dtype=float)
        full_deltas = np.diff(full_values)
        full_timing_reliable = bool(
            np.isfinite(full_values).all() and np.all(full_deltas > 0)
        )
    duration = (
        float(full_values[-1] - full_values[0]) if full_timing_reliable else None
    )
    fps = (
        float(1.0 / np.median(full_deltas)) if full_timing_reliable else None
    )
    video_summary = {
        "duration_seconds": duration,
        "fps": fps,
        "frame_count": len(sequence.frames),
        "source_megapixels": megapixels,
        "estimated_bitrate_mbps": None,
        "compressed_bits_per_pixel_per_frame": None,
        "decoded_sample_count": len(frames),
        "temporal_coverage_fraction": 1.0 if len(sequence.frames) > 1 else 0.0,
        "full_temporal_coverage_confirmed": True,
    }
    report_shape = {
        "raw_metrics": {
            "quality_summary": quality["summary"],
            "geometry_summary": geometry["summary"],
            "video": video_summary,
        }
    }
    features = numeric_features_from_report(report_shape)
    if not has_reliable_timestamps:
        for name in TIMESTAMP_DEPENDENT_GEOMETRY_FEATURES:
            features[f"geometry.{name}"] = None
    diagnostics = {
        "sampled_frame_count": len(sampled),
        "sampled_sequence_positions": sequence_positions,
        "annotated_timestamps_reliable": has_reliable_timestamps,
        "timestamp_dependent_features_available": has_reliable_timestamps,
    }
    return features, diagnostics


class EmbeddingCache:
    """Atomic NPZ cache keyed by sequence, files, and extractor configuration."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        sequence_id: str,
        frame_paths: Sequence[Path],
        configuration: Dict[str, Any],
    ) -> str:
        files = [
            {
                "path": str(Path(path).resolve()),
                "size": Path(path).stat().st_size,
                "mtime_ns": Path(path).stat().st_mtime_ns,
            }
            for path in frame_paths
        ]
        payload = {
            "sequence_id": sequence_id,
            "files": files,
            "configuration": configuration,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def load(self, key: str) -> Optional[np.ndarray]:
        path = self.path(key)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            values = archive["embeddings"].astype(np.float32, copy=False)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid embedding cache: {path}")
        return values

    def save(self, key: str, embeddings: np.ndarray) -> Path:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("embeddings must be a finite [time, dimension] array")
        destination = self.path(key)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, embeddings=values)
        temporary.replace(destination)
        return destination


class ResNet18FrameEmbedder:
    """Frozen ImageNet ResNet-18 encoder with deterministic preprocessing."""

    def __init__(self, *, device: Optional[str] = None, batch_size: int = 32) -> None:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError("Install the ML deep extra (torch and torchvision)") from exc
        self.torch = torch
        self.weights = ResNet18_Weights.DEFAULT
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = resnet18(weights=self.weights)
        model.fc = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self.model = model.to(self.device)
        self.transform = self.weights.transforms()
        self.configuration = {
            "encoder": "torchvision_resnet18_imagenet1k_v1",
            "embedding_dimension": 512,
            "preprocessing": "ResNet18_Weights.DEFAULT.transforms",
            "frozen": True,
        }

    def extract_paths(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        if not paths:
            raise ValueError("at least one frame path is required")
        batches = []
        for start in range(0, len(paths), self.batch_size):
            tensors = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as image:
                    tensors.append(self.transform(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                batches.append(self.model(batch).cpu().numpy().astype(np.float32))
        values = np.concatenate(batches, axis=0)
        if values.shape != (len(paths), 512):
            raise RuntimeError(f"Unexpected ResNet embedding shape: {values.shape}")
        return values

    def extract_cached(
        self,
        sequence_id: str,
        paths: Sequence[Path],
        cache: EmbeddingCache,
    ) -> tuple[str, np.ndarray, bool]:
        key = cache.key(sequence_id, paths, self.configuration)
        existing = cache.load(key)
        if existing is not None:
            return key, existing, True
        values = self.extract_paths(paths)
        cache.save(key, values)
        return key, values, False

