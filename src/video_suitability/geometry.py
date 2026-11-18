"""ORB correspondence, RANSAC verification, overlap, and motion proxies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for geometric analysis") from exc
    return cv2


def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _median(values: Sequence[Optional[float]], default: Optional[float] = 0.0) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(valid)) if valid else default


def _spatial_coverage(
    points: np.ndarray, width: int, height: int, grid_size: int = 4
) -> float:
    if points.size == 0 or width <= 0 or height <= 0:
        return 0.0
    x = np.clip((points[:, 0] / width * grid_size).astype(int), 0, grid_size - 1)
    y = np.clip((points[:, 1] / height * grid_size).astype(int), 0, grid_size - 1)
    occupied = len(set(zip(x.tolist(), y.tolist())))
    return occupied / float(grid_size * grid_size)


def detect_orb_features(
    frames: Sequence[np.ndarray], config: AnalysisConfig
) -> Tuple[List[List[Any]], List[Optional[np.ndarray]]]:
    """Detect ORB features once for reuse by geometry consumers.

    The viewpoint-diversity module shares this cache with the existing adjacent-pair
    analysis. Keeping feature extraction here guarantees identical ORB settings and
    avoids a second detector pass over every sampled frame.
    """

    cv2 = _cv2()
    cv2.setRNGSeed(42)
    orb = cv2.ORB_create(
        nfeatures=config.orb_features,
        fastThreshold=config.orb_fast_threshold,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )
    keypoints: List[List[Any]] = []
    descriptors: List[Optional[np.ndarray]] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        keypoints.append(kp or [])
        descriptors.append(desc)
    return keypoints, descriptors


# Backward-compatible private alias for any internal callers from earlier revisions.
_detect_features = detect_orb_features


def _optical_flow_metrics(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    delta_time_seconds: float,
) -> Dict[str, Any]:
    """Timestamp-aware pyramidal LK flow with forward/backward track validation."""

    cv2 = _cv2()
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    height, width = gray_a.shape
    diagonal = max(1.0, float(np.hypot(width, height)))
    result: Dict[str, Any] = {
        "flow_initial_points": 0,
        "flow_tracked_points": 0,
        "flow_track_retention": 0.0,
        "flow_median_displacement_normalized": None,
        "flow_p90_displacement_normalized": None,
        "flow_speed_normalized_per_second": None,
        "flow_velocity_x_normalized_per_second": None,
        "flow_velocity_y_normalized_per_second": None,
        "affine_rotation_degrees": None,
        "affine_rotation_rate_degrees_per_second": None,
        "affine_scale": None,
        "affine_residual_median_normalized": None,
    }
    corners = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) < 4:
        return result
    result["flow_initial_points"] = int(len(corners))
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, corners, None, **lk
    )
    if next_points is None or forward_status is None:
        return result
    back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_b, gray_a, next_points, None, **lk
    )
    if back_points is None or backward_status is None:
        return result

    original = corners.reshape(-1, 2)
    forward = next_points.reshape(-1, 2)
    backward = back_points.reshape(-1, 2)
    fb_error = np.linalg.norm(backward - original, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(forward).all(axis=1)
        & np.isfinite(backward).all(axis=1)
        & (fb_error <= 1.5)
    )
    tracked = int(valid.sum())
    result["flow_tracked_points"] = tracked
    result["flow_track_retention"] = _rounded(tracked / max(1, len(corners)))
    if tracked < 3:
        return result

    points_a = original[valid].astype(np.float32)
    points_b = forward[valid].astype(np.float32)
    vectors = points_b - points_a
    magnitudes = np.linalg.norm(vectors, axis=1) / diagonal
    median_displacement = float(np.median(magnitudes))
    median_vector = np.median(vectors, axis=0) / diagonal
    dt = max(1e-6, float(delta_time_seconds))
    result.update(
        {
            "flow_median_displacement_normalized": _rounded(median_displacement),
            "flow_p90_displacement_normalized": _rounded(
                float(np.percentile(magnitudes, 90))
            ),
            "flow_speed_normalized_per_second": _rounded(
                median_displacement / dt
            ),
            "flow_velocity_x_normalized_per_second": _rounded(
                float(median_vector[0]) / dt
            ),
            "flow_velocity_y_normalized_per_second": _rounded(
                float(median_vector[1]) / dt
            ),
        }
    )

    if tracked >= 4:
        try:
            affine, affine_mask = cv2.estimateAffinePartial2D(
                points_a,
                points_b,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
            )
            if affine is not None:
                a, b = float(affine[0, 0]), float(affine[0, 1])
                rotation = float(np.degrees(np.arctan2(affine[1, 0], a)))
                scale = float(np.hypot(a, b))
                predicted = cv2.transform(points_a.reshape(-1, 1, 2), affine).reshape(-1, 2)
                residuals = np.linalg.norm(predicted - points_b, axis=1) / diagonal
                if affine_mask is not None:
                    affine_valid = affine_mask.reshape(-1).astype(bool)
                    if affine_valid.any():
                        residuals = residuals[affine_valid]
                result.update(
                    {
                        "affine_rotation_degrees": _rounded(rotation),
                        "affine_rotation_rate_degrees_per_second": _rounded(
                            rotation / dt
                        ),
                        "affine_scale": _rounded(scale),
                        "affine_residual_median_normalized": _rounded(
                            float(np.median(residuals))
                        )
                        if residuals.size
                        else None,
                    }
                )
        except cv2.error:
            pass
    return result


def analyze_frame_pair_geometry(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    start_index: int,
    end_index: int,
    start_time: float,
    end_time: float,
    config: AnalysisConfig,
    calculate_optical_flow: bool = True,
) -> Dict[str, Any]:
    """Match one frame pair and verify it with fundamental/homography RANSAC."""

    cv2 = _cv2()
    height, width = frame_a.shape[:2]
    diagonal = max(1.0, float(np.hypot(width, height)))
    delta_time = max(1e-6, float(end_time) - float(start_time))
    issues: List[str] = []
    flow = (
        _optical_flow_metrics(frame_a, frame_b, delta_time)
        if calculate_optical_flow
        else {
            "flow_initial_points": 0,
            "flow_tracked_points": 0,
            "flow_track_retention": 0.0,
            "flow_median_displacement_normalized": None,
            "flow_p90_displacement_normalized": None,
            "flow_speed_normalized_per_second": None,
            "flow_velocity_x_normalized_per_second": None,
            "flow_velocity_y_normalized_per_second": None,
            "affine_rotation_degrees": None,
            "affine_rotation_rate_degrees_per_second": None,
            "affine_scale": None,
            "affine_residual_median_normalized": None,
        }
    )

    base: Dict[str, Any] = {
        "start_frame_index": int(start_index),
        "end_frame_index": int(end_index),
        "start_time_seconds": round(float(start_time), 6),
        "end_time_seconds": round(float(end_time), 6),
        "delta_time_seconds": round(delta_time, 6),
        "keypoints_start": len(keypoints_a),
        "keypoints_end": len(keypoints_b),
        "ratio_test_matches": 0,
        "verified_matches": 0,
        "verification_model": None,
        "match_ratio": 0.0,
        "descriptor_ambiguity_fraction": None,
        "strict_descriptor_ambiguity_fraction": None,
        "verified_match_ratio": 0.0,
        "inlier_ratio": 0.0,
        "spatial_coverage": 0.0,
        "median_displacement_normalized": None,
        "median_displacement_normalized_per_second": None,
        "p90_displacement_normalized": None,
        "homography_inlier_ratio": None,
        "homography_residual_p75_normalized": None,
        "motion_direction_change_degrees": None,
        "flow_speed_change_ratio": None,
        "issues": issues,
        **flow,
    }
    short_gap_for_flow_reliability = calculate_optical_flow and delta_time <= 0.30
    base["optical_flow_short_gap_evaluable"] = short_gap_for_flow_reliability
    if calculate_optical_flow and not short_gap_for_flow_reliability:
        issues.append("motion_sampling_gap_too_wide")
    if (
        short_gap_for_flow_reliability
        and
        int(flow["flow_initial_points"]) >= 20
        and float(flow["flow_track_retention"] or 0.0) < 0.20
    ):
        issues.append("low_track_retention")
    flow_speed = flow["flow_speed_normalized_per_second"]
    if flow_speed is not None and float(flow_speed) > 0.35:
        issues.append("excessive_camera_speed")
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(keypoints_a) < 2
        or len(keypoints_b) < 2
    ):
        issues.extend(["insufficient_correspondence", "low_overlap"])
        return base

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    candidates = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    valid_candidates = [pair for pair in candidates if len(pair) == 2]
    if valid_candidates:
        distance_ratios = [
            float(pair[0].distance) / max(1.0, float(pair[1].distance))
            for pair in valid_candidates
        ]
        base["descriptor_ambiguity_fraction"] = _rounded(
            float(np.mean(np.asarray(distance_ratios) >= 0.90))
        )
        base["strict_descriptor_ambiguity_fraction"] = _rounded(
            float(
                np.mean(
                    [
                        pair[1].distance <= 2.0 or ratio >= 0.90
                        for pair, ratio in zip(valid_candidates, distance_ratios)
                    ]
                )
            )
        )
    good = [
        m
        for m, n in valid_candidates
        if m.distance < config.ratio_test * n.distance
    ]
    base["ratio_test_matches"] = len(good)
    minimum_keypoints = max(1, min(len(keypoints_a), len(keypoints_b)))
    base["match_ratio"] = _rounded(len(good) / minimum_keypoints)
    if len(good) < 4:
        issues.extend(["insufficient_correspondence", "low_overlap"])
        return base

    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])
    fundamental_mask: Optional[np.ndarray] = None
    if len(good) >= 8:
        try:
            fundamental, mask = cv2.findFundamentalMat(
                points_a,
                points_b,
                cv2.FM_RANSAC,
                config.ransac_reprojection_threshold_px,
                0.99,
            )
            if fundamental is not None and mask is not None:
                fundamental_mask = mask.reshape(-1).astype(bool)
        except cv2.error:
            fundamental_mask = None

    homography: Optional[np.ndarray] = None
    homography_mask: Optional[np.ndarray] = None
    try:
        homography, mask = cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            max(3.0, config.ransac_reprojection_threshold_px * 1.5),
        )
        if homography is not None and mask is not None:
            homography_mask = mask.reshape(-1).astype(bool)
    except cv2.error:
        homography = None
        homography_mask = None

    if fundamental_mask is not None and int(fundamental_mask.sum()) >= 8:
        verified = fundamental_mask
        verification_model = "fundamental_matrix_ransac"
    elif homography_mask is not None and int(homography_mask.sum()) >= 4:
        verified = homography_mask
        verification_model = "homography_ransac_fallback"
    else:
        verified = np.zeros(len(good), dtype=bool)
        verification_model = None

    verified_count = int(verified.sum())
    inlier_ratio = verified_count / max(1, len(good))
    base.update(
        {
            "verified_matches": verified_count,
            "verification_model": verification_model,
            "verified_match_ratio": _rounded(verified_count / minimum_keypoints),
            "inlier_ratio": _rounded(inlier_ratio),
        }
    )

    displacement = np.linalg.norm(points_b - points_a, axis=1) / diagonal
    displacement_values = displacement[verified] if verified_count else displacement
    if displacement_values.size:
        median_displacement = float(np.median(displacement_values))
        base["median_displacement_normalized"] = _rounded(median_displacement)
        base["median_displacement_normalized_per_second"] = _rounded(
            median_displacement / delta_time
        )
        base["p90_displacement_normalized"] = _rounded(
            float(np.percentile(displacement_values, 90))
        )

    coverage_points_a = points_a[verified] if verified_count else points_a
    coverage_points_b = points_b[verified] if verified_count else points_b
    coverage = 0.5 * (
        _spatial_coverage(coverage_points_a, width, height)
        + _spatial_coverage(coverage_points_b, frame_b.shape[1], frame_b.shape[0])
    )
    base["spatial_coverage"] = _rounded(coverage)

    if homography_mask is not None:
        base["homography_inlier_ratio"] = _rounded(float(np.mean(homography_mask)))
    if homography is not None:
        try:
            projected = cv2.perspectiveTransform(points_a.reshape(-1, 1, 2), homography).reshape(-1, 2)
            residuals = np.linalg.norm(projected - points_b, axis=1) / diagonal
            finite = residuals[np.isfinite(residuals)]
            if finite.size:
                base["homography_residual_p75_normalized"] = _rounded(
                    float(np.percentile(finite, 75))
                )
        except cv2.error:
            pass

    motion = base["median_displacement_normalized"]
    if verified_count < 8:
        issues.append("insufficient_correspondence")
    if float(base["verified_match_ratio"] or 0.0) < 0.02 or coverage < 0.10:
        issues.append("low_overlap")
    if motion is not None and motion < 0.003:
        issues.append("little_viewpoint_change")
    if motion is not None and motion > 0.25:
        issues.append("excessive_viewpoint_jump")
    h_ratio = base["homography_inlier_ratio"]
    h_residual = base["homography_residual_p75_normalized"]
    if (
        motion is not None
        and motion >= 0.003
        and h_ratio is not None
        and h_ratio >= 0.80
        and h_residual is not None
        and h_residual < 0.006
    ):
        issues.append("homography_dominant_motion")
    ambiguity = base["descriptor_ambiguity_fraction"]
    if ambiguity is not None and float(ambiguity) >= 0.65:
        issues.append("ambiguous_repetitive_features")
    return base


# Backward-compatible private alias retained for the existing adjacent-pair caller.
_pair_metrics = analyze_frame_pair_geometry


def _annotate_motion_irregularity(pairs: Sequence[Dict[str, Any]]) -> None:
    """Mark sudden changes in timestamp-normalized global flow between pairs."""

    for index in range(1, len(pairs)):
        previous = pairs[index - 1]
        current = pairs[index]
        if not (
            previous.get("optical_flow_short_gap_evaluable", False)
            and current.get("optical_flow_short_gap_evaluable", False)
        ):
            continue
        previous_components = (
            previous.get("flow_velocity_x_normalized_per_second"),
            previous.get("flow_velocity_y_normalized_per_second"),
        )
        current_components = (
            current.get("flow_velocity_x_normalized_per_second"),
            current.get("flow_velocity_y_normalized_per_second"),
        )
        previous_speed = previous.get("flow_speed_normalized_per_second")
        current_speed = current.get("flow_speed_normalized_per_second")
        direction_change: Optional[float] = None
        speed_ratio: Optional[float] = None
        irregular = False

        if previous_speed is not None and current_speed is not None:
            low = max(0.005, min(float(previous_speed), float(current_speed)))
            high = max(float(previous_speed), float(current_speed))
            speed_ratio = high / low
            current["flow_speed_change_ratio"] = _rounded(speed_ratio)
            if speed_ratio >= 3.0 and high >= 0.15:
                irregular = True

        if all(value is not None for value in previous_components + current_components):
            vector_previous = np.asarray(previous_components, dtype=float)
            vector_current = np.asarray(current_components, dtype=float)
            norm_previous = float(np.linalg.norm(vector_previous))
            norm_current = float(np.linalg.norm(vector_current))
            if norm_previous >= 0.02 and norm_current >= 0.02:
                cosine = float(
                    np.clip(
                        np.dot(vector_previous, vector_current)
                        / (norm_previous * norm_current),
                        -1.0,
                        1.0,
                    )
                )
                direction_change = float(np.degrees(np.arccos(cosine)))
                current["motion_direction_change_degrees"] = _rounded(direction_change)
                if direction_change >= 70.0:
                    irregular = True

        previous_rotation = previous.get("affine_rotation_rate_degrees_per_second")
        current_rotation = current.get("affine_rotation_rate_degrees_per_second")
        if previous_rotation is not None and current_rotation is not None:
            if (
                abs(float(current_rotation) - float(previous_rotation)) >= 45.0
                and max(abs(float(current_rotation)), abs(float(previous_rotation))) >= 20.0
            ):
                irregular = True
        if irregular and "irregular_camera_motion" not in current["issues"]:
            current["issues"].append("irregular_camera_motion")


def analyze_geometry(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    timestamps_seconds: Sequence[float],
    config: AnalysisConfig,
    detected_features: Optional[
        Tuple[List[List[Any]], List[Optional[np.ndarray]]]
    ] = None,
) -> Dict[str, Any]:
    """Analyze adjacent uniformly sampled frames using ORB and robust geometry."""

    if not (len(frames) == len(frame_indices) == len(timestamps_seconds)):
        raise ValueError("frames, frame_indices, and timestamps_seconds must have equal length")
    if not frames:
        raise ValueError("at least one frame is required")

    keypoints, descriptors = detected_features or detect_orb_features(frames, config)
    if len(keypoints) != len(frames) or len(descriptors) != len(frames):
        raise ValueError("detected_features must contain one entry per frame")
    pairs = [
        analyze_frame_pair_geometry(
            frames[i],
            frames[i + 1],
            keypoints[i],
            keypoints[i + 1],
            descriptors[i],
            descriptors[i + 1],
            frame_indices[i],
            frame_indices[i + 1],
            timestamps_seconds[i],
            timestamps_seconds[i + 1],
            config,
        )
        for i in range(max(0, len(frames) - 1))
    ]
    _annotate_motion_irregularity(pairs)

    keypoint_counts = [len(kp) for kp in keypoints]
    valid_pairs = [p for p in pairs if p["verified_matches"] >= 8]
    pair_count = len(pairs)

    def pair_fraction(issue: str) -> float:
        return float(np.mean([issue in p["issues"] for p in pairs])) if pairs else 0.0

    moderate_motion = [
        p
        for p in pairs
        if p["median_displacement_normalized"] is not None
        and 0.01 <= p["median_displacement_normalized"] <= 0.18
    ]
    flow_speeds = [
        float(p["flow_speed_normalized_per_second"])
        for p in pairs
        if p.get("flow_speed_normalized_per_second") is not None
    ]
    short_gap_pairs = [
        p for p in pairs if p.get("optical_flow_short_gap_evaluable", False)
    ]
    short_gap_retentions = [
        float(p["flow_track_retention"]) for p in short_gap_pairs
    ]
    speed_median = float(np.median(flow_speeds)) if flow_speeds else 0.0
    speed_mad = (
        float(np.median(np.abs(np.asarray(flow_speeds) - speed_median)))
        if flow_speeds
        else 0.0
    )
    direction_changes = [
        float(p["motion_direction_change_degrees"])
        for p in pairs
        if p.get("motion_direction_change_degrees") is not None
    ]
    rotation_rates = [
        abs(float(p["affine_rotation_rate_degrees_per_second"]))
        for p in pairs
        if p.get("affine_rotation_rate_degrees_per_second") is not None
    ]
    cumulative_motion = float(
        np.sum(
            [
                float(p["flow_median_displacement_normalized"])
                for p in pairs
                if p.get("flow_median_displacement_normalized") is not None
            ]
        )
    )
    rotation_or_planar_fraction = float(
        np.mean(
            [
                p.get("homography_inlier_ratio") is not None
                and float(p["homography_inlier_ratio"]) >= 0.80
                and p.get("homography_residual_p75_normalized") is not None
                and float(p["homography_residual_p75_normalized"]) < 0.006
                for p in pairs
            ]
        )
    ) if pairs else 0.0
    summary = {
        "median_keypoints_per_frame": round(float(np.median(keypoint_counts)), 6),
        "minimum_keypoints_per_frame": int(min(keypoint_counts)),
        "low_feature_frame_fraction": round(
            float(np.mean(np.asarray(keypoint_counts) < 40)), 6
        ),
        "pair_count": pair_count,
        "valid_geometry_pair_count": len(valid_pairs),
        "valid_geometry_pair_fraction": round(
            len(valid_pairs) / max(1, pair_count), 6
        ),
        "median_ratio_test_matches": round(
            float(_median([p["ratio_test_matches"] for p in pairs], 0.0)), 6
        ),
        "median_verified_matches": round(
            float(_median([p["verified_matches"] for p in pairs], 0.0)), 6
        ),
        "median_match_ratio": round(
            float(_median([p["match_ratio"] for p in pairs], 0.0)), 6
        ),
        "median_verified_match_ratio": round(
            float(_median([p["verified_match_ratio"] for p in pairs], 0.0)), 6
        ),
        "median_inlier_ratio": round(
            float(_median([p["inlier_ratio"] for p in valid_pairs], 0.0)), 6
        ),
        "median_spatial_coverage": round(
            float(_median([p["spatial_coverage"] for p in valid_pairs], 0.0)), 6
        ),
        "median_displacement_normalized": _rounded(
            _median([p["median_displacement_normalized"] for p in pairs], None)
        ),
        "median_displacement_normalized_per_second": _rounded(
            _median(
                [p["median_displacement_normalized_per_second"] for p in pairs],
                None,
            )
        ),
        "median_homography_residual_p75_normalized": _rounded(
            _median([p["homography_residual_p75_normalized"] for p in pairs], None)
        ),
        "moderate_motion_pair_fraction": round(
            len(moderate_motion) / max(1, pair_count), 6
        ),
        "near_stationary_pair_fraction": round(pair_fraction("little_viewpoint_change"), 6),
        "excessive_motion_pair_fraction": round(pair_fraction("excessive_viewpoint_jump"), 6),
        "low_overlap_pair_fraction": round(pair_fraction("low_overlap"), 6),
        "homography_dominant_pair_fraction": round(
            pair_fraction("homography_dominant_motion"), 6
        ),
        "rotation_or_planar_pair_fraction": round(
            rotation_or_planar_fraction, 6
        ),
        "median_homography_inlier_ratio": _rounded(
            _median([p["homography_inlier_ratio"] for p in pairs], None)
        ),
        "median_descriptor_ambiguity_fraction": _rounded(
            _median([p["descriptor_ambiguity_fraction"] for p in pairs], None)
        ),
        "high_descriptor_ambiguity_pair_fraction": round(
            pair_fraction("ambiguous_repetitive_features"), 6
        ),
        "median_flow_track_retention": round(
            float(_median([p["flow_track_retention"] for p in pairs], 0.0)), 6
        ),
        "median_flow_track_retention_short_gap": _rounded(
            float(np.median(short_gap_retentions))
            if short_gap_retentions
            else None
        ),
        "optical_flow_short_gap_pair_fraction": round(
            len(short_gap_pairs) / max(1, pair_count), 6
        ),
        "motion_sampling_gap_too_wide_pair_fraction": round(
            pair_fraction("motion_sampling_gap_too_wide"), 6
        ),
        "low_track_retention_pair_fraction": round(
            pair_fraction("low_track_retention"), 6
        ),
        "median_flow_speed_normalized_per_second": round(speed_median, 6),
        "p90_flow_speed_normalized_per_second": round(
            float(np.percentile(flow_speeds, 90)) if flow_speeds else 0.0, 6
        ),
        "flow_speed_robust_cv": round(
            speed_mad / max(0.01, speed_median), 6
        ),
        "excessive_camera_speed_pair_fraction": round(
            pair_fraction("excessive_camera_speed"), 6
        ),
        "irregular_camera_motion_pair_fraction": round(
            pair_fraction("irregular_camera_motion"), 6
        ),
        "irregular_camera_motion_evaluable_pair_fraction": round(
            sum("irregular_camera_motion" in p["issues"] for p in short_gap_pairs)
            / max(1, len(short_gap_pairs)),
            6,
        ),
        "p90_motion_direction_change_degrees": _rounded(
            float(np.percentile(direction_changes, 90))
            if direction_changes
            else None
        ),
        "median_abs_rotation_rate_degrees_per_second": round(
            float(np.median(rotation_rates)) if rotation_rates else 0.0, 6
        ),
        "rotation_rate_mad_degrees_per_second": round(
            float(np.median(np.abs(np.asarray(rotation_rates) - np.median(rotation_rates))))
            if rotation_rates
            else 0.0,
            6,
        ),
        "median_affine_residual_normalized": _rounded(
            _median([p["affine_residual_median_normalized"] for p in pairs], None)
        ),
        "cumulative_flow_motion_normalized": round(cumulative_motion, 6),
    }
    frame_features = [
        {
            "frame_index": int(index),
            "timestamp_seconds": round(float(timestamp), 6),
            "orb_keypoints": int(count),
        }
        for index, timestamp, count in zip(frame_indices, timestamps_seconds, keypoint_counts)
    ]
    return {"summary": summary, "frames": frame_features, "pairs": pairs}
