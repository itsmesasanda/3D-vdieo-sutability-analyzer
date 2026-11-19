"""Evidence-based estimated viewpoint diversity without reconstruction.

This module deliberately produces a categorical, standalone result. It does not
alter the established heuristic or optional ML score. Image-space motion alone is
never treated as proof of camera translation or object-surface coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig
from .geometry import analyze_frame_pair_geometry, detect_orb_features


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for viewpoint analysis") from exc
    return cv2


def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [
        float(value) for value in values if value is not None and np.isfinite(value)
    ]
    return float(np.median(finite)) if finite else None


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, optionally tied to a source image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: Optional[float] = None
    image_height: Optional[float] = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "CameraIntrinsics":
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(
                "camera_intrinsics is missing required values: " + ", ".join(missing)
            )
        intrinsics = cls(
            fx=float(values["fx"]),
            fy=float(values["fy"]),
            cx=float(values["cx"]),
            cy=float(values["cy"]),
            image_width=(
                float(values["image_width"])
                if values.get("image_width") is not None
                else None
            ),
            image_height=(
                float(values["image_height"])
                if values.get("image_height") is not None
                else None
            ),
        )
        intrinsics.validate()
        return intrinsics

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera fx and fy must be positive")
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError(
                "camera intrinsic image_width and image_height must be supplied together"
            )
        if self.image_width is not None and (
            self.image_width <= 0
            or self.image_height is None
            or self.image_height <= 0
        ):
            raise ValueError("camera intrinsic image dimensions must be positive")

    def for_frame(self, width: int, height: int) -> "CameraIntrinsics":
        """Scale source-pixel intrinsics to the analyzed frame resolution."""

        self.validate()
        if self.image_width is None or self.image_height is None:
            return CameraIntrinsics(self.fx, self.fy, self.cx, self.cy, width, height)
        scale_x = width / self.image_width
        scale_y = height / self.image_height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            image_width=float(width),
            image_height=float(height),
        )

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "fx": round(self.fx, 6),
            "fy": round(self.fy, 6),
            "cx": round(self.cx, 6),
            "cy": round(self.cy, 6),
            "image_width": round(float(self.image_width or 0), 6),
            "image_height": round(float(self.image_height or 0), 6),
        }


def _separated_pair_positions(
    frame_count: int, maximum_pairs: int = 15
) -> List[Tuple[int, int]]:
    """Return a deterministic bounded set of mid-, long-gap, and endpoint pairs."""

    if frame_count < 3:
        return []
    gaps = sorted(
        {
            max(2, round((frame_count - 1) * 0.25)),
            max(2, round((frame_count - 1) * 0.50)),
            frame_count - 1,
        }
    )
    selected: List[Tuple[int, int]] = []
    for gap in gaps:
        possible = frame_count - gap
        if possible <= 0:
            continue
        count = min(6, possible)
        starts = np.rint(np.linspace(0, possible - 1, count)).astype(int)
        selected.extend((int(start), int(start + gap)) for start in starts)
    unique = list(dict.fromkeys(selected))
    if len(unique) <= maximum_pairs:
        return unique
    keep = np.rint(np.linspace(0, len(unique) - 1, maximum_pairs)).astype(int)
    return [unique[int(index)] for index in sorted(set(keep.tolist()))]


def _is_strong_planar_or_rotation(pair: Mapping[str, Any]) -> bool:
    motion = pair.get("median_displacement_normalized")
    homography_ratio = pair.get("homography_inlier_ratio")
    residual = pair.get("homography_residual_p75_normalized")
    return bool(
        motion is not None
        and float(motion) >= 0.003
        and homography_ratio is not None
        and float(homography_ratio) >= 0.92
        and residual is not None
        and float(residual) <= 0.0035
    )


def _has_parallax_proxy(pair: Mapping[str, Any]) -> bool:
    """Conservative uncalibrated evidence that one homography is inadequate."""

    motion = pair.get("median_displacement_normalized")
    residual = pair.get("homography_residual_p75_normalized")
    return bool(
        pair.get("verification_model") == "fundamental_matrix_ransac"
        and int(pair.get("verified_matches") or 0) >= 12
        and float(pair.get("inlier_ratio") or 0.0) >= 0.55
        and float(pair.get("spatial_coverage") or 0.0) >= 0.15
        and motion is not None
        and float(motion) >= 0.004
        and not bool(pair.get("mixed_static_dynamic_evidence", False))
        and not _is_strong_planar_or_rotation(pair)
        and (
            residual is None
            or float(residual) >= 0.0025
            or float(pair.get("homography_inlier_ratio") or 0.0) < 0.85
        )
    )


def _point_coverage(points: np.ndarray, width: int, height: int) -> float:
    """Return coarse spatial support for a selected motion population."""

    if points.size == 0 or width <= 0 or height <= 0:
        return 0.0
    grid_size = 4
    x = np.clip((points[:, 0] / width * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip((points[:, 1] / height * grid_size).astype(int), 0, grid_size - 1)
    return len(set(zip(x.tolist(), y.tolist()))) / float(grid_size * grid_size)


def _motion_field_consistency(
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    frame_shape: Sequence[int],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """Diagnose static-background plus moving-region mixtures.

    This is a rejection signal, not a camera-pose estimate. A large population of
    near-static matches coexisting with a spatially supported population of strongly
    moving matches is consistent with object animation in front of a fixed camera.
    It must not be promoted to camera-translation evidence.
    """

    height, width = int(frame_shape[0]), int(frame_shape[1])
    diagonal = max(1.0, float(np.hypot(width, height)))
    points_a, points_b = _ratio_test_points(
        keypoints_a,
        keypoints_b,
        descriptors_a,
        descriptors_b,
        config.ratio_test,
    )
    result: Dict[str, Any] = {
        "evaluable": False,
        "ratio_test_match_count": int(len(points_a)),
        "near_static_match_fraction": None,
        "strong_motion_match_fraction": None,
        "near_static_spatial_coverage": None,
        "strong_motion_spatial_coverage": None,
        "mixed_static_dynamic_evidence": False,
    }
    if len(points_a) < 16:
        return result
    magnitudes = np.linalg.norm(points_b - points_a, axis=1) / diagonal
    # Pixel-aware floors keep the interpretation stable across analysis sizes.
    static_threshold = max(0.0025, 1.5 / diagonal)
    moving_threshold = max(0.012, 6.0 / diagonal)
    near_static = magnitudes <= static_threshold
    strong_motion = magnitudes >= moving_threshold
    static_fraction = float(np.mean(near_static))
    motion_fraction = float(np.mean(strong_motion))
    static_coverage = _point_coverage(points_a[near_static], width, height)
    motion_coverage = _point_coverage(points_a[strong_motion], width, height)
    mixed = bool(
        int(near_static.sum()) >= 8
        and int(strong_motion.sum()) >= 8
        and static_fraction >= 0.30
        and motion_fraction >= 0.12
        and static_coverage >= 0.20
        and motion_coverage >= 0.0625
    )
    result.update(
        {
            "evaluable": True,
            "near_static_match_fraction": _rounded(static_fraction),
            "strong_motion_match_fraction": _rounded(motion_fraction),
            "near_static_spatial_coverage": _rounded(static_coverage),
            "strong_motion_spatial_coverage": _rounded(motion_coverage),
            "mixed_static_dynamic_evidence": mixed,
        }
    )
    return result


def _is_repeated_view(pair: Mapping[str, Any]) -> bool:
    displacement = pair.get("median_displacement_normalized")
    return bool(
        int(pair.get("verified_matches") or 0) >= 15
        and float(pair.get("verified_match_ratio") or 0.0) >= 0.05
        and displacement is not None
        and float(displacement) <= 0.006
        and float(pair.get("homography_inlier_ratio") or 0.0) >= 0.80
    )


def _has_substantial_view_change(pair: Mapping[str, Any]) -> bool:
    displacement = pair.get("median_displacement_normalized")
    residual = pair.get("homography_residual_p75_normalized")
    return bool(
        int(pair.get("verified_matches") or 0) >= 8
        and (
            (displacement is not None and float(displacement) >= 0.018)
            or (residual is not None and float(residual) >= 0.006)
        )
    )


def _ratio_test_points(
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    ratio_test: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the same ratio-tested ORB point pairs used by geometry analysis."""

    if descriptors_a is None or descriptors_b is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    cv2 = _cv2()
    candidates = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        descriptors_a, descriptors_b, k=2
    )
    good = [
        first
        for pair in candidates
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio_test * second.distance
    ]
    if not good:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    return points_a, points_b


def _within_frame_duplicate_descriptor_fraction(
    descriptors: Optional[np.ndarray],
) -> Optional[float]:
    """Return the feature fraction sharing an identical descriptor in one frame."""

    if descriptors is None or len(descriptors) == 0:
        return None
    _, counts = np.unique(descriptors, axis=0, return_counts=True)
    duplicate_count = int(counts[counts > 1].sum())
    return duplicate_count / len(descriptors)


def _intrinsic_motion_indicator(
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
    config: AnalysisConfig,
) -> Dict[str, Any]:
    """Estimate scale-free relative pose and rotation-compensated bearing change."""

    cv2 = _cv2()
    result: Dict[str, Any] = {
        "essential_matrix_available": False,
        "essential_inliers": 0,
        "recover_pose_inliers": 0,
        "relative_rotation_degrees": None,
        "translation_direction_unit": None,
        "translation_scale_available": False,
        "median_rotation_compensated_parallax_degrees": None,
    }
    points_a, points_b = _ratio_test_points(
        keypoints_a,
        keypoints_b,
        descriptors_a,
        descriptors_b,
        config.ratio_test,
    )
    if len(points_a) < 8:
        return result
    matrix = intrinsics.matrix()
    try:
        essential, mask = cv2.findEssentialMat(
            points_a,
            points_b,
            matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=max(1.0, config.ransac_reprojection_threshold_px),
        )
        if essential is None or mask is None:
            return result
        essential = essential[:3, :3]
        essential_inliers = int(mask.reshape(-1).astype(bool).sum())
        pose_inliers, rotation, translation, pose_mask = cv2.recoverPose(
            essential, points_a, points_b, matrix, mask=mask
        )
    except cv2.error:
        return result

    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_degrees = float(np.degrees(np.arccos(cosine)))
    inverse = np.linalg.inv(matrix)
    homogeneous_a = np.column_stack((points_a, np.ones(len(points_a)))).T
    homogeneous_b = np.column_stack((points_b, np.ones(len(points_b)))).T
    rays_a = inverse @ homogeneous_a
    rays_b = inverse @ homogeneous_b
    rays_a /= np.linalg.norm(rays_a, axis=0, keepdims=True).clip(min=1e-12)
    rays_b /= np.linalg.norm(rays_b, axis=0, keepdims=True).clip(min=1e-12)
    rotated_a = rotation @ rays_a
    cosines = np.clip(np.sum(rotated_a * rays_b, axis=0), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    if pose_mask is not None:
        valid = pose_mask.reshape(-1).astype(bool)
        if valid.any():
            angles = angles[valid]
    translation_vector = translation.reshape(-1)
    norm = float(np.linalg.norm(translation_vector))
    unit = translation_vector / norm if norm > 1e-12 else translation_vector
    result.update(
        {
            "essential_matrix_available": True,
            "essential_inliers": essential_inliers,
            "recover_pose_inliers": int(pose_inliers),
            "relative_rotation_degrees": _rounded(rotation_degrees),
            "translation_direction_unit": [round(float(value), 6) for value in unit],
            "median_rotation_compensated_parallax_degrees": _rounded(
                float(np.median(angles)) if angles.size else None
            ),
        }
    )
    return result


def _classification_guidance(label: str) -> Tuple[str, List[str]]:
    if label == "High":
        return (
            "Strong multi-view change is visible in the sampled sequence, with evidence that is not explained by one global planar transform.",
            [
                "Keep the object static and maintain the same slow, overlapping camera path.",
                "Add a second elevation if top or underside geometry is important; this label does not prove complete object coverage.",
            ],
        )
    if label == "Moderate":
        return (
            "Several distinct views and some parallax-like evidence are present, but the sampled path does not support a strong-diversity result.",
            [
                "Continue farther around the object with small, overlapping lateral steps.",
                "Include another elevation while keeping focus, exposure, and the object fixed.",
            ],
        )
    if label == "Low":
        return (
            "The sampled frames repeatedly show nearly the same view or too little image-space baseline.",
            [
                "Move the camera laterally around the object instead of staying at the front.",
                "Keep roughly 60–80% overlap between neighboring views while extending the path.",
            ],
        )
    return (
        "The footage does not provide enough trustworthy geometric evidence to classify viewpoint diversity.",
        [
            "Retake with sharp focus, diffuse light, distinctive non-repeating detail, and a slow lateral camera path.",
            "Keep the object still; avoid an in-place pan or rotating only the camera, which can mimic viewpoint change without useful translation.",
        ],
    )


def analyze_viewpoint_diversity(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    timestamps_seconds: Sequence[float],
    quality_summary: Mapping[str, Any],
    geometry: Mapping[str, Any],
    config: AnalysisConfig,
    *,
    detected_features: Optional[
        Tuple[List[List[Any]], List[Optional[np.ndarray]]]
    ] = None,
    camera_intrinsics: Optional[Mapping[str, float]] = None,
    maximum_separated_pairs: int = 15,
    analysis_mode: str = "whole_frame",
) -> Dict[str, Any]:
    """Classify estimated viewpoint diversity from bounded multi-gap evidence."""

    if not (len(frames) == len(frame_indices) == len(timestamps_seconds)):
        raise ValueError(
            "frames, frame_indices, and timestamps_seconds must have equal length"
        )
    if not frames:
        raise ValueError("at least one frame is required")
    keypoints, descriptors = detected_features or detect_orb_features(frames, config)
    if len(keypoints) != len(frames) or len(descriptors) != len(frames):
        raise ValueError("detected_features must contain one entry per frame")

    if maximum_separated_pairs < 1 or maximum_separated_pairs > 30:
        raise ValueError("maximum_separated_pairs must be between 1 and 30")
    separated_positions = _separated_pair_positions(
        len(frames), maximum_pairs=maximum_separated_pairs
    )
    separated_pairs: List[Dict[str, Any]] = []
    for start, end in separated_positions:
        pair = analyze_frame_pair_geometry(
            frames[start],
            frames[end],
            keypoints[start],
            keypoints[end],
            descriptors[start],
            descriptors[end],
            frame_indices[start],
            frame_indices[end],
            timestamps_seconds[start],
            timestamps_seconds[end],
            config,
            calculate_optical_flow=False,
        )
        pair["sample_position_start"] = start
        pair["sample_position_end"] = end
        pair["temporal_separation_fraction"] = round(
            (end - start) / max(1, len(frames) - 1), 6
        )
        pair["pair_kind"] = (
            "endpoint"
            if (start, end) == (0, len(frames) - 1)
            else "temporally_separated"
        )
        motion_consistency = _motion_field_consistency(
            keypoints[start],
            keypoints[end],
            descriptors[start],
            descriptors[end],
            frames[start].shape,
            config,
        )
        pair["motion_field_consistency"] = motion_consistency
        pair["mixed_static_dynamic_evidence"] = bool(
            motion_consistency["mixed_static_dynamic_evidence"]
        )
        pair["parallax_proxy_detected"] = _has_parallax_proxy(pair)
        pair["strong_planar_or_rotation_evidence"] = _is_strong_planar_or_rotation(
            pair
        )
        pair["repeated_view_detected"] = _is_repeated_view(pair)
        pair["substantial_view_change_detected"] = _has_substantial_view_change(pair)
        pair["viewpoint_descriptor_ambiguity_fraction"] = pair.get(
            "strict_descriptor_ambiguity_fraction"
        )
        separated_pairs.append(pair)

    adjacent_pairs = list(geometry.get("pairs", []))
    all_pairs = adjacent_pairs + separated_pairs
    valid_pairs = [pair for pair in all_pairs if int(pair.get("verified_matches") or 0) >= 8]
    valid_separated_pairs = [
        pair
        for pair in separated_pairs
        if int(pair.get("verified_matches") or 0) >= 8
    ]
    changed_pairs = [
        pair
        for pair in all_pairs
        if pair.get("median_displacement_normalized") is not None
        and float(pair["median_displacement_normalized"]) >= 0.003
    ]
    repeated_pairs = [
        pair for pair in separated_pairs if pair["repeated_view_detected"]
    ]
    parallax_pairs = [pair for pair in all_pairs if _has_parallax_proxy(pair)]
    separated_parallax_pairs = [
        pair for pair in separated_pairs if pair["parallax_proxy_detected"]
    ]
    changed_separated = [
        pair
        for pair in separated_pairs
        if pair["substantial_view_change_detected"]
    ]
    planar_changed = [
        pair for pair in changed_pairs if _is_strong_planar_or_rotation(pair)
    ]
    ambiguous_pairs = [
        pair
        for pair in separated_pairs
        if pair.get("viewpoint_descriptor_ambiguity_fraction") is not None
        and float(pair["viewpoint_descriptor_ambiguity_fraction"]) >= 0.40
    ]
    mixed_motion_pairs = [
        pair
        for pair in separated_pairs
        if pair.get("mixed_static_dynamic_evidence", False)
    ]
    evaluable_motion_field_pairs = [
        pair
        for pair in separated_pairs
        if pair.get("motion_field_consistency", {}).get("evaluable", False)
    ]
    endpoint_pair = next(
        (pair for pair in separated_pairs if pair["pair_kind"] == "endpoint"), None
    )
    loop_closure = bool(endpoint_pair and endpoint_pair["repeated_view_detected"])

    intrinsic_model: Optional[CameraIntrinsics] = None
    intrinsic_indicators: List[Dict[str, Any]] = []
    if camera_intrinsics is not None:
        source_intrinsics = CameraIntrinsics.from_mapping(camera_intrinsics)
        height, width = frames[0].shape[:2]
        intrinsic_model = source_intrinsics.for_frame(width, height)
        for pair in separated_pairs:
            start = int(pair["sample_position_start"])
            end = int(pair["sample_position_end"])
            indicator = _intrinsic_motion_indicator(
                keypoints[start],
                keypoints[end],
                descriptors[start],
                descriptors[end],
                intrinsic_model,
                config,
            )
            indicator.update(
                {
                    "start_frame_index": int(frame_indices[start]),
                    "end_frame_index": int(frame_indices[end]),
                    "start_time_seconds": round(float(timestamps_seconds[start]), 6),
                    "end_time_seconds": round(float(timestamps_seconds[end]), 6),
                }
            )
            intrinsic_indicators.append(indicator)

    pair_denominator = max(1, len(all_pairs))
    separated_denominator = max(1, len(separated_pairs))
    changed_denominator = max(1, len(changed_pairs))
    geometry_summary = geometry.get("summary", {})
    cumulative_adjacent_motion = float(
        sum(
            float(pair["median_displacement_normalized"])
            for pair in adjacent_pairs
            if pair.get("median_displacement_normalized") is not None
        )
    )
    valid_intrinsic_parallax = [
        float(item["median_rotation_compensated_parallax_degrees"])
        for item in intrinsic_indicators
        if item["median_rotation_compensated_parallax_degrees"] is not None
        and int(item["recover_pose_inliers"]) >= 8
    ]
    self_descriptor_ambiguity = [
        value
        for value in (
            _within_frame_duplicate_descriptor_fraction(descriptor)
            for descriptor in descriptors
        )
        if value is not None
    ]
    separated_descriptor_ambiguity = [
        float(pair["viewpoint_descriptor_ambiguity_fraction"])
        for pair in separated_pairs
        if pair.get("viewpoint_descriptor_ambiguity_fraction") is not None
    ]
    indicators = {
        "sampled_frame_count": len(frames),
        "nearby_pair_count": len(adjacent_pairs),
        "temporally_separated_pair_count": len(separated_pairs),
        "valid_geometry_pair_fraction": round(len(valid_pairs) / pair_denominator, 6),
        "valid_separated_geometry_pair_fraction": round(
            len(valid_separated_pairs) / separated_denominator, 6
        ),
        "median_keypoints_per_frame": _rounded(
            float(geometry_summary.get("median_keypoints_per_frame", 0.0))
        ),
        "near_stationary_nearby_pair_fraction": _rounded(
            float(geometry_summary.get("near_stationary_pair_fraction", 0.0))
        ),
        "cumulative_adjacent_displacement_normalized": round(
            cumulative_adjacent_motion, 6
        ),
        "temporally_separated_repeated_view_fraction": round(
            len(repeated_pairs) / separated_denominator, 6
        ),
        "temporally_separated_substantial_change_fraction": round(
            len(changed_separated) / separated_denominator, 6
        ),
        "parallax_proxy_pair_fraction": round(len(parallax_pairs) / pair_denominator, 6),
        "temporally_separated_parallax_proxy_pair_fraction": round(
            len(separated_parallax_pairs) / separated_denominator, 6
        ),
        "high_descriptor_ambiguity_pair_fraction": round(
            len(ambiguous_pairs) / separated_denominator, 6
        ),
        "mixed_static_dynamic_pair_fraction": round(
            len(mixed_motion_pairs)
            / max(1, len(evaluable_motion_field_pairs)),
            6,
        ),
        "motion_field_evaluable_pair_fraction": round(
            len(evaluable_motion_field_pairs) / separated_denominator, 6
        ),
        "median_within_frame_duplicate_descriptor_fraction": _rounded(
            float(np.median(self_descriptor_ambiguity))
            if self_descriptor_ambiguity
            else None
        ),
        "median_separated_strict_descriptor_ambiguity_fraction": _rounded(
            float(np.median(separated_descriptor_ambiguity))
            if separated_descriptor_ambiguity
            else None
        ),
        "strong_planar_or_rotation_fraction_among_changed_pairs": round(
            len(planar_changed) / changed_denominator, 6
        ),
        "endpoint_repeated_view_detected": loop_closure,
        "median_separated_displacement_normalized": _rounded(
            _median(
                [
                    pair.get("median_displacement_normalized")
                    for pair in separated_pairs
                ]
            )
        ),
        "median_separated_verified_matches": _rounded(
            _median(
                [float(pair.get("verified_matches") or 0) for pair in separated_pairs]
            )
        ),
        "median_rotation_compensated_parallax_degrees": _rounded(
            float(np.median(valid_intrinsic_parallax))
            if valid_intrinsic_parallax
            else None
        ),
    }

    insufficient_reasons: List[str] = []
    blurry_fraction = float(quality_summary.get("blurry_frame_fraction", 0.0))
    low_texture_fraction = float(
        quality_summary.get("low_texture_frame_fraction", 0.0)
    )
    within_frame_ambiguity = float(
        indicators["median_within_frame_duplicate_descriptor_fraction"] or 0.0
    )
    cross_view_ambiguity = float(
        indicators["median_separated_strict_descriptor_ambiguity_fraction"] or 0.0
    )
    if len(frames) < 6:
        insufficient_reasons.append(
            "Fewer than six usable timeline samples are available."
        )
    if blurry_fraction >= 0.60:
        insufficient_reasons.append(
            "Most sampled frames are blurry, so correspondences are not trustworthy."
        )
    if (
        low_texture_fraction >= 0.50
        or float(indicators["median_keypoints_per_frame"] or 0.0) < 40
    ):
        insufficient_reasons.append(
            "The object or scene has too little trackable texture."
        )
    if float(indicators["valid_geometry_pair_fraction"]) < 0.35:
        insufficient_reasons.append(
            "Too few frame pairs have RANSAC-verified geometry."
        )
    if float(indicators["valid_separated_geometry_pair_fraction"]) < 0.35:
        insufficient_reasons.append(
            "Too few temporally separated pairs have at least eight RANSAC-verified matches."
        )
    if float(indicators["median_separated_verified_matches"] or 0.0) < 8.0:
        insufficient_reasons.append(
            "The median temporally separated pair has fewer than eight verified correspondences."
        )
    if within_frame_ambiguity >= 0.10 or cross_view_ambiguity >= 0.65:
        insufficient_reasons.append(
            "Descriptors are frequently ambiguous within or across views, consistent with repetitive or symmetric appearance."
        )
    if (
        len(evaluable_motion_field_pairs) >= 3
        and float(indicators["mixed_static_dynamic_pair_fraction"]) >= 0.30
    ):
        insufficient_reasons.append(
            "Long-gap matches repeatedly split into a broad near-static region and a strongly moving region; object animation or another non-rigid scene change cannot be separated from camera motion."
        )

    motion_present = cumulative_adjacent_motion >= 0.035
    rotation_dominant = bool(
        motion_present
        and float(indicators["strong_planar_or_rotation_fraction_among_changed_pairs"])
        >= 0.65
    )
    if rotation_dominant:
        insufficient_reasons.append(
            "Most changed pairs are explained by a global homography; rotation-only or planar motion cannot be separated reliably from useful translation."
        )

    supporting_evidence: List[str] = []
    limiting_evidence: List[str] = list(insufficient_reasons)
    classification_rule = "insufficient_evidence_gate"
    primary_reason = insufficient_reasons[0] if insufficient_reasons else ""
    if insufficient_reasons:
        label = "Unknown"
    else:
        stationary_fraction = float(
            indicators["near_stationary_nearby_pair_fraction"] or 0.0
        )
        repeat_fraction = float(
            indicators["temporally_separated_repeated_view_fraction"]
        )
        change_fraction = float(
            indicators["temporally_separated_substantial_change_fraction"]
        )
        parallax_fraction = float(
            indicators["temporally_separated_parallax_proxy_pair_fraction"]
        )
        if stationary_fraction >= 0.65 or (
            repeat_fraction >= 0.60 and change_fraction <= 0.20
        ):
            label = "Low"
            classification_rule = "stationary_or_repeated_views"
            primary_reason = (
                "Most nearby or temporally separated evidence repeats nearly the same view."
            )
            limiting_evidence.append(
                "Useful triangulation baseline is inadequate despite usable matches."
            )
        elif (
            loop_closure
            and change_fraction >= 0.40
            and parallax_fraction >= 0.30
            and cumulative_adjacent_motion >= 0.10
        ) or (
            change_fraction >= 0.65
            and parallax_fraction >= 0.45
            and cumulative_adjacent_motion >= 0.22
        ):
            label = "High"
            classification_rule = "broad_change_with_parallax_proxy"
            primary_reason = (
                "Many long-gap pairs show substantial view change with non-planar parallax-like support."
            )
            supporting_evidence.extend(
                [
                    f"Long-gap substantial-change fraction is {change_fraction:.3f}.",
                    f"Long-gap parallax-proxy fraction is {parallax_fraction:.3f}.",
                ]
            )
        elif (
            change_fraction >= 0.25
            and parallax_fraction >= 0.20
            and cumulative_adjacent_motion >= 0.045
        ):
            label = "Moderate"
            classification_rule = "partial_change_with_parallax_proxy"
            primary_reason = (
                "Distinct object views and some parallax-like support are present, but evidence does not meet the High rule."
            )
            supporting_evidence.append(
                f"Long-gap parallax-proxy fraction is {parallax_fraction:.3f}."
            )
        else:
            label = "Low"
            classification_rule = "inadequate_parallax_or_path_extent"
            primary_reason = (
                "Geometric matches exist, but long-gap parallax or path extent is inadequate."
            )
            limiting_evidence.append(
                f"Long-gap parallax-proxy fraction is only {parallax_fraction:.3f}."
            )

    warnings = [
        "Estimated viewpoint diversity is a categorical image-evidence heuristic, not a measurement of exact camera travel, 360-degree coverage, or object-surface coverage.",
        "A moving or rotating object can mimic camera motion; this analysis assumes the filmed object is mostly static.",
    ]
    if camera_intrinsics is None:
        warnings.append(
            "Camera intrinsics were not supplied; metric translation, calibrated relative pose, and rotation-compensated parallax are unavailable."
        )
    else:
        warnings.append(
            "Recovered translation is a direction only; monocular two-view geometry does not provide metric scale."
        )
    if repeated_pairs:
        if loop_closure:
            warnings.append(
                "The first and last sampled views appear repeated. This can be loop closure or a repeated pass; it is not proof of a complete orbit."
            )
        if len(repeated_pairs) / separated_denominator >= 0.50:
            warnings.append(
                "Many temporally separated frames repeat nearly the same view, indicating insufficient viewpoint variation."
            )
    warnings.extend(insufficient_reasons)
    summary, recommendations = _classification_guidance(label)
    if primary_reason:
        summary = f"{summary} Primary decision reason: {primary_reason}"
    return {
        "label": label,
        "evidence_status": "insufficient" if label == "Unknown" else "sufficient",
        "summary": summary,
        "method": {
            "name": "orb_ransac_multigap_viewpoint_heuristic",
            "version": "1.1",
            "trained_model_used": False,
            "colmap_run": False,
            "affects_overall_score": False,
            "intrinsics_used": intrinsic_model is not None,
            "analysis_mode": analysis_mode,
            "object_bbox_used": analysis_mode == "object_focused",
        },
        "decision_explanation": {
            "primary_reason": primary_reason,
            "classification_rule": classification_rule,
            "supporting_evidence": supporting_evidence,
            "limiting_evidence": limiting_evidence,
            "insufficient_evidence_reasons": list(dict.fromkeys(insufficient_reasons)),
        },
        "indicators": indicators,
        "camera_intrinsics_at_analysis_resolution": (
            intrinsic_model.as_dict() if intrinsic_model is not None else None
        ),
        "relative_motion_indicators": intrinsic_indicators,
        "temporally_separated_pair_evidence": separated_pairs,
        "repeated_view_intervals": [
            {
                "start_time_seconds": pair["start_time_seconds"],
                "end_time_seconds": pair["end_time_seconds"],
                "start_frame_index": pair["start_frame_index"],
                "end_frame_index": pair["end_frame_index"],
            }
            for pair in repeated_pairs
        ],
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": recommendations,
    }

