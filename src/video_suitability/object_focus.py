"""Optional object-region tracking and object-focused viewpoint geometry."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .config import AnalysisConfig
from .geometry import analyze_frame_pair_geometry, analyze_geometry
from .viewpoint import analyze_viewpoint_diversity


NormalizedBox = Tuple[float, float, float, float]
FeatureCache = Tuple[List[List[Any]], List[Optional[np.ndarray]]]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for object-region tracking") from exc
    return cv2


def normalize_object_bbox(
    values: Union[Sequence[float], Mapping[str, float]]
) -> NormalizedBox:
    """Validate a normalized ``(x, y, width, height)`` object box."""

    if isinstance(values, Mapping):
        raw = [values[key] for key in ("x", "y", "width", "height")]
    else:
        raw = list(values)
    if len(raw) != 4:
        raise ValueError("object_bbox must contain x, y, width, and height")
    x, y, width, height = (float(value) for value in raw)
    if not all(np.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("object_bbox values must be finite")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("object_bbox must have a non-negative origin and positive size")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("object_bbox must stay within normalized frame coordinates")
    if width < 0.05 or height < 0.05:
        raise ValueError("object_bbox width and height must each cover at least 5% of the frame")
    return (x, y, width, height)


def _pixel_box(box: NormalizedBox, frame: np.ndarray) -> Tuple[float, float, float, float]:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = box
    return (x * width, y * height, box_width * width, box_height * height)


def _box_iou(first: NormalizedBox, second: NormalizedBox) -> float:
    """Intersection-over-union for normalized tracking cycle validation."""

    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _contains(point: Tuple[float, float], box: NormalizedBox, frame: np.ndarray) -> bool:
    x, y, width, height = _pixel_box(box, frame)
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _reference_foreground_template(
    frame: np.ndarray, box: NormalizedBox
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build one bounded foreground prior from the user-selected reference box.

    GrabCut runs only on the representative frame. The resulting largest connected
    component is warped with the tracked box; it is not rerun on every sample. This
    keeps live CPU cost bounded while preventing obvious background pixels inside a
    generous user box from becoming object-view evidence.
    """

    cv2 = _cv2()
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = _pixel_box(box, frame)
    x0 = max(0, min(frame_width - 2, int(round(x))))
    y0 = max(0, min(frame_height - 2, int(round(y))))
    x1 = max(x0 + 2, min(frame_width, int(round(x + width))))
    y1 = max(y0 + 2, min(frame_height, int(round(y + height))))
    crop_height, crop_width = y1 - y0, x1 - x0
    fallback = np.full((crop_height, crop_width), 255, dtype=np.uint8)
    diagnostics: Dict[str, Any] = {
        "method": "bbox_only_fallback",
        "available": False,
        "foreground_fraction_within_box": 1.0,
        "reason": "grabcut_unavailable_or_implausible",
    }
    if crop_width < 12 or crop_height < 12:
        diagnostics["reason"] = "selected_box_too_small_for_foreground_prior"
        return fallback, diagnostics

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(42)
        cv2.grabCut(
            frame,
            mask,
            (x0, y0, crop_width, crop_height),
            background_model,
            foreground_model,
            1,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return fallback, diagnostics
    crop = mask[y0:y1, x0:x1]
    foreground = np.where(
        (crop == cv2.GC_FGD) | (crop == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8
    )
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        foreground = np.where(labels == largest, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.dilate(foreground, kernel, iterations=1)
    fraction = float(np.mean(foreground > 0))
    if fraction < 0.08 or fraction > 0.92:
        diagnostics["foreground_fraction_within_box"] = round(fraction, 6)
        return fallback, diagnostics
    diagnostics.update(
        {
            "method": "reference_grabcut_largest_component",
            "available": True,
            "foreground_fraction_within_box": round(fraction, 6),
            "reason": "reference_foreground_prior_available",
        }
    )
    return foreground, diagnostics


def _mask_for_box(
    frame: np.ndarray, box: NormalizedBox, template: np.ndarray
) -> np.ndarray:
    """Place a reference foreground template inside a tracked normalized box."""

    cv2 = _cv2()
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = _pixel_box(box, frame)
    x0 = max(0, min(frame_width - 1, int(round(x))))
    y0 = max(0, min(frame_height - 1, int(round(y))))
    x1 = max(x0 + 1, min(frame_width, int(round(x + width))))
    y1 = max(y0 + 1, min(frame_height, int(round(y + height))))
    resized = cv2.resize(
        template,
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = resized
    return mask


def _expanded_box(box: NormalizedBox, scale: float = 1.65) -> NormalizedBox:
    x, y, width, height = box
    new_width = min(1.0, width * scale)
    new_height = min(1.0, height * scale)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return (
        max(0.0, min(1.0 - new_width, center_x - new_width / 2.0)),
        max(0.0, min(1.0 - new_height, center_y - new_height / 2.0)),
        new_width,
        new_height,
    )


def _axis_aligned_box_from_affine(
    source_box: NormalizedBox,
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    affine: np.ndarray,
) -> Tuple[Optional[NormalizedBox], float]:
    """Update center/scale without turning feature rotation into box growth."""

    x, y, width, height = _pixel_box(source_box, source_frame)
    source_height, source_width = source_frame.shape[:2]
    target_height, target_width = target_frame.shape[:2]
    center = np.asarray([x + width / 2.0, y + height / 2.0, 1.0])
    transformed_center = affine @ center
    linear = affine[:, :2]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0:
        return None, 0.0
    measured_scale = float(np.sqrt(determinant))
    # Damping tolerates moderate approach/retreat while avoiding cumulative growth
    # from turntable surface rotation inside an axis-aligned object box.
    applied_scale = float(np.clip(1.0 + 0.60 * (measured_scale - 1.0), 0.80, 1.25))
    new_width = width * (target_width / max(1.0, source_width)) * applied_scale
    new_height = height * (target_height / max(1.0, source_height)) * applied_scale
    if (
        new_width / target_width < 0.04
        or new_height / target_height < 0.04
        or new_width / target_width > 0.95
        or new_height / target_height > 0.95
    ):
        return None, (new_width * new_height) / max(1.0, width * height)
    min_x = float(
        np.clip(transformed_center[0] - new_width / 2.0, 0, target_width - new_width)
    )
    min_y = float(
        np.clip(transformed_center[1] - new_height / 2.0, 0, target_height - new_height)
    )
    area_ratio = (new_width * new_height) / max(1.0, width * height)
    return (
        min_x / target_width,
        min_y / target_height,
        new_width / target_width,
        new_height / target_height,
    ), area_ratio


def _detect_masked_orb(
    frame: np.ndarray, mask: np.ndarray, config: AnalysisConfig
) -> Tuple[List[Any], Optional[np.ndarray]]:
    """Periodically re-detect ORB features inside an explicit region mask."""

    cv2 = _cv2()
    cv2.setRNGSeed(42)
    orb = cv2.ORB_create(
        nfeatures=config.orb_features,
        fastThreshold=config.orb_fast_threshold,
        scoreType=cv2.ORB_HARRIS_SCORE,
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return keypoints or [], descriptors


def _subset_features_by_mask(
    keypoints: Sequence[Any],
    descriptors: Optional[np.ndarray],
    mask: np.ndarray,
    *,
    inside: bool,
) -> Tuple[List[Any], Optional[np.ndarray]]:
    if descriptors is None or not keypoints:
        return [], None
    height, width = mask.shape[:2]
    indices: List[int] = []
    for index, keypoint in enumerate(keypoints):
        x = min(width - 1, max(0, int(round(keypoint.pt[0]))))
        y = min(height - 1, max(0, int(round(keypoint.pt[1]))))
        if bool(mask[y, x] > 0) is inside:
            indices.append(index)
    if not indices:
        return [], None
    return [keypoints[index] for index in indices], descriptors[np.asarray(indices)]


def _propagate_box(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    source_keypoints: Sequence[Any],
    target_keypoints: Sequence[Any],
    source_descriptors: Optional[np.ndarray],
    target_descriptors: Optional[np.ndarray],
    source_box: NormalizedBox,
    config: AnalysisConfig,
    source_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[NormalizedBox], Dict[str, Any]]:
    """Propagate one box using source-in-box ORB matches and affine RANSAC."""

    cv2 = _cv2()
    result: Dict[str, Any] = {
        "ratio_test_matches": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "status": "failed",
        "reason": "insufficient_descriptors",
    }
    if source_descriptors is None or target_descriptors is None:
        return None, result
    if source_mask is None:
        source_indices = [
            index
            for index, keypoint in enumerate(source_keypoints)
            if _contains(keypoint.pt, source_box, source_frame)
        ]
    else:
        mask_height, mask_width = source_mask.shape[:2]
        source_indices = []
        for index, keypoint in enumerate(source_keypoints):
            point_x = min(mask_width - 1, max(0, int(round(keypoint.pt[0]))))
            point_y = min(mask_height - 1, max(0, int(round(keypoint.pt[1]))))
            if source_mask[point_y, point_x] > 0:
                source_indices.append(index)
    if len(source_indices) < 8 or len(target_keypoints) < 8:
        result["reason"] = "insufficient_in_box_features"
        return None, result
    query_descriptors = source_descriptors[np.asarray(source_indices)]
    candidates = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        query_descriptors, target_descriptors, k=2
    )
    good = [
        first
        for pair in candidates
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < config.ratio_test * second.distance
    ]
    result["ratio_test_matches"] = len(good)
    if len(good) < 8:
        result["reason"] = "insufficient_ratio_test_matches"
        return None, result
    points_source = np.float32(
        [source_keypoints[source_indices[match.queryIdx]].pt for match in good]
    )
    points_target = np.float32([target_keypoints[match.trainIdx].pt for match in good])
    try:
        affine, mask = cv2.estimateAffinePartial2D(
            points_source,
            points_target,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(2.5, config.ransac_reprojection_threshold_px),
            maxIters=2000,
            confidence=0.99,
        )
    except cv2.error:
        affine, mask = None, None
    if affine is None or mask is None:
        result["reason"] = "affine_ransac_failed"
        return None, result
    inliers = int(mask.reshape(-1).astype(bool).sum())
    inlier_ratio = inliers / max(1, len(good))
    result.update(
        {
            "ransac_inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 6),
        }
    )
    if inliers < 6 or inlier_ratio < 0.45:
        result["reason"] = "unreliable_affine_consensus"
        return None, result
    box, area_ratio = _axis_aligned_box_from_affine(
        source_box, source_frame, target_frame, affine
    )
    result["area_ratio"] = round(area_ratio, 6)
    if box is None or not 0.45 <= area_ratio <= 2.20:
        result["reason"] = "implausible_tracked_box"
        return None, result
    result.update({"status": "tracked", "reason": "affine_ransac_consensus"})
    return box, result


def _propagate_box_klt(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    source_box: NormalizedBox,
    source_mask: np.ndarray,
    config: AnalysisConfig,
) -> Tuple[Optional[NormalizedBox], Dict[str, Any]]:
    """Propagate a box using pyramidal KLT with point-level FB consistency."""

    cv2 = _cv2()
    result: Dict[str, Any] = {
        "klt_initial_points": 0,
        "klt_forward_backward_points": 0,
        "ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "spatial_support": 0.0,
        "median_forward_backward_error_px": None,
        "status": "failed",
        "reason": "insufficient_klt_points",
    }
    gray_source = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        gray_source,
        mask=source_mask,
        maxCorners=350,
        qualityLevel=0.008,
        minDistance=4,
        blockSize=7,
    )
    if points is None or len(points) < 10:
        return None, result
    result["klt_initial_points"] = int(len(points))
    lk = {
        "winSize": (31, 31),
        "maxLevel": 3,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_source, gray_target, points, None, **lk
    )
    if forward is None or status_forward is None:
        result["reason"] = "klt_forward_failed"
        return None, result
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray_target, gray_source, forward, None, **lk
    )
    if backward is None or status_backward is None:
        result["reason"] = "klt_backward_failed"
        return None, result
    source_points = points.reshape(-1, 2)
    target_points = forward.reshape(-1, 2)
    returned_points = backward.reshape(-1, 2)
    diagonal = max(1.0, float(np.hypot(*gray_source.shape[::-1])))
    fb_threshold = min(4.0, max(1.5, 0.004 * diagonal))
    fb_error = np.linalg.norm(source_points - returned_points, axis=1)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(target_points).all(axis=1)
        & (fb_error <= fb_threshold)
    )
    source_points = source_points[valid]
    target_points = target_points[valid]
    retained_errors = fb_error[valid]
    result["klt_forward_backward_points"] = int(len(source_points))
    result["median_forward_backward_error_px"] = (
        round(float(np.median(retained_errors)), 6)
        if len(retained_errors)
        else None
    )
    if len(source_points) < 10:
        result["reason"] = "insufficient_forward_backward_klt_points"
        return None, result
    try:
        affine, mask = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(
                2.5, config.ransac_reprojection_threshold_px
            ),
            maxIters=2000,
            confidence=0.99,
        )
    except cv2.error:
        affine, mask = None, None
    if affine is None or mask is None:
        result["reason"] = "klt_affine_ransac_failed"
        return None, result
    inlier_mask = mask.reshape(-1).astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / max(1, len(source_points))
    x, y, width, height = _pixel_box(source_box, source_frame)
    inlier_points = source_points[inlier_mask]
    grid_x = np.clip(((inlier_points[:, 0] - x) / max(1.0, width) * 4).astype(int), 0, 3)
    grid_y = np.clip(((inlier_points[:, 1] - y) / max(1.0, height) * 4).astype(int), 0, 3)
    spatial_support = len(set(zip(grid_x.tolist(), grid_y.tolist()))) / 16.0
    result.update(
        {
            "ransac_inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 6),
            "spatial_support": round(spatial_support, 6),
        }
    )
    median_fb_error = result["median_forward_backward_error_px"]
    standard_consensus = bool(
        inliers >= 8 and inlier_ratio >= 0.50 and spatial_support >= 0.125
    )
    distributed_partial_consensus = bool(
        inliers >= 20
        and inlier_ratio >= 0.40
        and spatial_support >= 0.25
        and median_fb_error is not None
        and float(median_fb_error) <= 1.0
    )
    if not (standard_consensus or distributed_partial_consensus):
        result["reason"] = "unreliable_klt_affine_consensus"
        return None, result
    result["consensus_rule"] = (
        "standard"
        if standard_consensus
        else "distributed_partial_occlusion_or_view_change"
    )
    box, area_ratio = _axis_aligned_box_from_affine(
        source_box, source_frame, target_frame, affine
    )
    result["area_ratio"] = round(area_ratio, 6)
    if box is None or not 0.40 <= area_ratio <= 2.50:
        result["reason"] = "implausible_klt_tracked_box"
        return None, result
    result.update({"status": "tracked", "reason": "klt_fb_affine_consensus"})
    return box, result


def _propagate_box_hybrid(
    source_frame: np.ndarray,
    target_frame: np.ndarray,
    source_box: NormalizedBox,
    foreground_template: np.ndarray,
    config: AnalysisConfig,
) -> Tuple[Optional[NormalizedBox], Dict[str, Any]]:
    """Combine FB-KLT propagation with periodic object-masked ORB re-detection."""

    source_mask = _mask_for_box(source_frame, source_box, foreground_template)
    search_box = _expanded_box(source_box)
    search_template = np.full((32, 32), 255, dtype=np.uint8)
    target_mask = _mask_for_box(target_frame, search_box, search_template)
    source_keypoints, source_descriptors = _detect_masked_orb(
        source_frame, source_mask, config
    )
    target_keypoints, target_descriptors = _detect_masked_orb(
        target_frame, target_mask, config
    )
    orb_box, orb = _propagate_box(
        source_frame,
        target_frame,
        source_keypoints,
        target_keypoints,
        source_descriptors,
        target_descriptors,
        source_box,
        config,
        source_mask=source_mask,
    )
    klt_box, klt = _propagate_box_klt(
        source_frame,
        target_frame,
        source_box,
        source_mask,
        config,
    )
    result: Dict[str, Any] = {
        "status": "failed",
        "reason": "hybrid_no_reliable_candidate",
        "tracker_method": None,
        "orb_status": orb["status"],
        "orb_reason": orb["reason"],
        "orb_ratio_test_matches": orb.get("ratio_test_matches", 0),
        "orb_ransac_inliers": orb.get("ransac_inliers", 0),
        "orb_inlier_ratio": orb.get("inlier_ratio", 0.0),
        "klt_status": klt["status"],
        "klt_reason": klt["reason"],
        "klt_initial_points": klt.get("klt_initial_points", 0),
        "klt_forward_backward_points": klt.get(
            "klt_forward_backward_points", 0
        ),
        "klt_ransac_inliers": klt.get("ransac_inliers", 0),
        "klt_inlier_ratio": klt.get("inlier_ratio", 0.0),
        "klt_spatial_support": klt.get("spatial_support", 0.0),
        "klt_median_forward_backward_error_px": klt.get(
            "median_forward_backward_error_px"
        ),
        "redetected_source_object_keypoints": len(source_keypoints),
        "redetected_target_search_keypoints": len(target_keypoints),
    }
    if orb_box is not None and klt_box is not None:
        agreement = _box_iou(orb_box, klt_box)
        result["orb_klt_box_iou"] = round(agreement, 6)
        if agreement < 0.35:
            result["reason"] = "orb_klt_disagreement"
            return None, result
        orb_strength = float(orb.get("inlier_ratio") or 0.0) * min(
            1.0, float(orb.get("ransac_inliers") or 0) / 20.0
        )
        klt_strength = float(klt.get("inlier_ratio") or 0.0) * min(
            1.0, float(klt.get("ransac_inliers") or 0) / 20.0
        )
        selected_box, selected, method = (
            (klt_box, klt, "klt_with_orb_agreement")
            if klt_strength >= orb_strength
            else (orb_box, orb, "orb_with_klt_agreement")
        )
    elif klt_box is not None:
        if len(source_keypoints) < 8:
            result["reason"] = "insufficient_redetected_object_features"
            return None, result
        selected_box, selected, method = klt_box, klt, "klt_fb_affine"
    elif orb_box is not None:
        selected_box, selected, method = orb_box, orb, "masked_orb_redetection"
    else:
        return None, result
    result.update(selected)
    result.update(
        {
            "status": "tracked",
            "reason": "hybrid_consensus",
            "selected_tracker_reason": selected.get("reason"),
            "tracker_method": method,
        }
    )
    return selected_box, result


def track_object_regions(
    frames: Sequence[np.ndarray],
    detected_features: FeatureCache,
    initial_box: Union[Sequence[float], Mapping[str, float]],
    representative_position: Optional[int],
    config: AnalysisConfig,
    *,
    foreground_template: Optional[np.ndarray] = None,
    foreground_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Track a user-selected object box forward and backward across samples."""

    if not frames:
        raise ValueError("at least one frame is required for object tracking")
    keypoints, descriptors = detected_features
    if len(keypoints) != len(frames) or len(descriptors) != len(frames):
        raise ValueError("detected_features must contain one entry per frame")
    reference = (
        len(frames) // 2 if representative_position is None else representative_position
    )
    if reference < 0 or reference >= len(frames):
        raise ValueError("representative_position is outside the sampled frames")
    box = normalize_object_bbox(initial_box)
    if foreground_template is None:
        foreground_template, generated_diagnostics = _reference_foreground_template(
            frames[reference], box
        )
        foreground_diagnostics = generated_diagnostics
    elif foreground_diagnostics is None:
        foreground_diagnostics = {
            "method": "supplied_foreground_template",
            "available": True,
            "foreground_fraction_within_box": round(
                float(np.mean(foreground_template > 0)), 6
            ),
            "reason": "foreground_template_supplied",
        }
    boxes: List[Optional[NormalizedBox]] = [None] * len(frames)
    boxes[reference] = box
    steps: List[Dict[str, Any]] = [
        {
            "source_position": reference,
            "target_position": reference,
            "status": "selected",
            "reason": "user_selected_reference_box",
            "ratio_test_matches": None,
            "ransac_inliers": None,
            "inlier_ratio": None,
        }
    ]
    for direction in (1, -1):
        source = reference
        target = source + direction
        consecutive_failures = 0
        while 0 <= target < len(frames):
            if boxes[source] is None:
                break
            propagated, evidence = _propagate_box_hybrid(
                frames[source],
                frames[target],
                boxes[source],
                foreground_template,
                config,
            )
            evidence.update(
                {
                    "source_position": source,
                    "target_position": target,
                    "sample_gap": abs(target - source),
                }
            )
            if propagated is not None:
                reverse_box, reverse_evidence = _propagate_box_hybrid(
                    frames[target],
                    frames[source],
                    propagated,
                    foreground_template,
                    config,
                )
                cycle_iou = (
                    _box_iou(boxes[source], reverse_box)
                    if reverse_box is not None and boxes[source] is not None
                    else 0.0
                )
                evidence.update(
                    {
                        "reverse_status": reverse_evidence["status"],
                        "reverse_reason": reverse_evidence["reason"],
                        "cycle_iou": round(cycle_iou, 6),
                        "cycle_verified": (
                            reverse_box is not None and cycle_iou >= 0.40
                        ),
                    }
                )
                if (reverse_box is not None and cycle_iou < 0.40) or (
                    reverse_box is None
                    and float(evidence.get("inlier_ratio") or 0.0) < 0.65
                ):
                    propagated = None
                    evidence["status"] = "failed"
                    evidence["reason"] = "forward_backward_cycle_inconsistent"
            steps.append(evidence)
            if propagated is None:
                consecutive_failures += 1
                # Keep the missing sample missing, but permit one bounded ORB/KLT
                # re-detection attempt after a partial occlusion. This cannot inflate
                # tracked_sample_fraction because skipped samples remain None.
                if consecutive_failures >= 2:
                    break
                target += direction
                continue
            boxes[target] = propagated
            source = target
            target = source + direction
            consecutive_failures = 0
    tracked_count = sum(item is not None for item in boxes)
    tracked_fraction = tracked_count / len(frames)
    propagation_steps = [
        item for item in steps if item.get("status") != "selected"
    ]
    cycle_attempts = [
        item for item in propagation_steps if "cycle_verified" in item
    ]
    verified_cycles = [item for item in cycle_attempts if item["cycle_verified"]]
    return {
        "representative_sample_position": reference,
        "initial_normalized_bbox": [round(value, 6) for value in box],
        "normalized_bboxes": [
            [round(value, 6) for value in item] if item is not None else None
            for item in boxes
        ],
        "tracked_sample_count": tracked_count,
        "sample_count": len(frames),
        "tracked_sample_fraction": round(tracked_fraction, 6),
        "forward_reached_end": boxes[-1] is not None,
        "backward_reached_start": boxes[0] is not None,
        "cycle_check_attempt_count": len(cycle_attempts),
        "cycle_verified_step_fraction": round(
            len(verified_cycles) / max(1, len(cycle_attempts)), 6
        ),
        "median_verified_cycle_iou": round(
            float(np.median([item["cycle_iou"] for item in verified_cycles])), 6
        )
        if verified_cycles
        else None,
        "foreground_separation": dict(foreground_diagnostics),
        "tracker_version": "hybrid_klt_masked_orb_fb_v1",
        "steps": sorted(steps, key=lambda item: item["target_position"]),
    }


def build_region_feature_caches(
    frames: Sequence[np.ndarray],
    detected_features: FeatureCache,
    normalized_bboxes: Sequence[Optional[Sequence[float]]],
    config: AnalysisConfig,
    foreground_template: Optional[np.ndarray] = None,
) -> Tuple[FeatureCache, FeatureCache, Dict[str, Any]]:
    """Split the shared feature cache into object and background caches."""

    keypoints, descriptors = detected_features
    object_keypoints: List[List[Any]] = []
    object_descriptors: List[Optional[np.ndarray]] = []
    background_keypoints: List[List[Any]] = []
    background_descriptors: List[Optional[np.ndarray]] = []
    locally_redetected_counts: List[int] = []
    for frame, frame_keypoints, frame_descriptors, raw_box in zip(
        frames, keypoints, descriptors, normalized_bboxes
    ):
        box = tuple(raw_box) if raw_box is not None else None
        if box is None:
            object_items = ([], None)
            background_items = ([], None)
        else:
            template = (
                foreground_template
                if foreground_template is not None
                else np.full((32, 32), 255, dtype=np.uint8)
            )
            object_mask = _mask_for_box(frame, box, template)
            object_items = _detect_masked_orb(frame, object_mask, config)
            background_items = _subset_features_by_mask(
                frame_keypoints,
                frame_descriptors,
                object_mask,
                inside=False,
            )
        locally_redetected_counts.append(len(object_items[0]))
        object_keypoints.append(object_items[0])
        object_descriptors.append(object_items[1])
        background_keypoints.append(background_items[0])
        background_descriptors.append(background_items[1])
    return (
        (object_keypoints, object_descriptors),
        (background_keypoints, background_descriptors),
        {
            "method": "local_masked_orb_redetection",
            "median_object_keypoints": round(
                float(np.median(locally_redetected_counts)), 6
            )
            if locally_redetected_counts
            else 0.0,
            "untracked_frames_have_object_features": False,
        },
    )


def _background_comparison(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    timestamps_seconds: Sequence[float],
    background_features: FeatureCache,
    normalized_bboxes: Sequence[Optional[Sequence[float]]],
    object_pairs: Sequence[Mapping[str, Any]],
    config: AnalysisConfig,
) -> Dict[str, Any]:
    keypoints, descriptors = background_features
    rows: List[Dict[str, Any]] = []
    for object_pair in object_pairs:
        start = int(object_pair["sample_position_start"])
        end = int(object_pair["sample_position_end"])
        if normalized_bboxes[start] is None or normalized_bboxes[end] is None:
            rows.append(
                {
                    "start_frame_index": int(frame_indices[start]),
                    "end_frame_index": int(frame_indices[end]),
                    "evaluated": False,
                    "reason": "object_region_untracked_for_pair",
                    "object_verified_matches": int(
                        object_pair.get("verified_matches") or 0
                    ),
                    "background_verified_matches": None,
                    "object_verified_match_ratio": round(
                        float(object_pair.get("verified_match_ratio") or 0.0), 6
                    ),
                    "background_verified_match_ratio": None,
                    "background_dominated": None,
                    "object_motion_normalized": object_pair.get(
                        "median_displacement_normalized"
                    ),
                    "background_motion_normalized": None,
                    "moving_object_with_static_background": None,
                }
            )
            continue
        background = analyze_frame_pair_geometry(
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
        object_ratio = float(object_pair.get("verified_match_ratio") or 0.0)
        background_ratio = float(background.get("verified_match_ratio") or 0.0)
        dominated = bool(
            int(background.get("verified_matches") or 0) >= 12
            and (
                int(object_pair.get("verified_matches") or 0) < 8
                or background_ratio >= max(0.02, object_ratio * 1.75)
            )
        )
        object_motion = object_pair.get("median_displacement_normalized")
        background_motion = background.get("median_displacement_normalized")
        moving_object_with_static_background = bool(
            int(object_pair.get("verified_matches") or 0) >= 8
            and int(background.get("verified_matches") or 0) >= 12
            and object_motion is not None
            and float(object_motion) >= 0.012
            and background_motion is not None
            and float(background_motion) <= 0.004
        )
        rows.append(
            {
                "start_frame_index": int(frame_indices[start]),
                "end_frame_index": int(frame_indices[end]),
                "evaluated": True,
                "reason": "object_and_background_regions_available",
                "object_verified_matches": int(
                    object_pair.get("verified_matches") or 0
                ),
                "background_verified_matches": int(
                    background.get("verified_matches") or 0
                ),
                "object_verified_match_ratio": round(object_ratio, 6),
                "background_verified_match_ratio": round(background_ratio, 6),
                "background_dominated": dominated,
                "object_motion_normalized": object_motion,
                "background_motion_normalized": background_motion,
                "moving_object_with_static_background": (
                    moving_object_with_static_background
                ),
            }
        )
    evaluated = [row for row in rows if row["evaluated"]]
    dominated_fraction = (
        float(np.mean([row["background_dominated"] for row in evaluated]))
        if evaluated
        else None
    )
    static_background_motion_fraction = (
        float(
            np.mean(
                [row["moving_object_with_static_background"] for row in evaluated]
            )
        )
        if evaluated
        else None
    )
    return {
        "pair_count": len(rows),
        "evaluated_pair_count": len(evaluated),
        "unevaluated_pair_count": len(rows) - len(evaluated),
        "background_dominated_pair_fraction": (
            round(dominated_fraction, 6) if dominated_fraction is not None else None
        ),
        "moving_object_with_static_background_pair_fraction": (
            round(static_background_motion_fraction, 6)
            if static_background_motion_fraction is not None
            else None
        ),
        "median_object_verified_matches": round(
            float(np.median([row["object_verified_matches"] for row in evaluated])), 6
        )
        if evaluated
        else None,
        "median_background_verified_matches": round(
            float(
                np.median(
                    [row["background_verified_matches"] for row in evaluated]
                )
            ),
            6,
        )
        if evaluated
        else None,
        "pairs": rows,
    }


def analyze_object_focused_viewpoint(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    timestamps_seconds: Sequence[float],
    quality_summary: Mapping[str, Any],
    full_frame_geometry: Mapping[str, Any],
    detected_features: FeatureCache,
    object_bbox: Union[Sequence[float], Mapping[str, float]],
    config: AnalysisConfig,
    *,
    camera_intrinsics: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run object-only viewpoint evidence while retaining background diagnostics."""

    normalized_box = normalize_object_bbox(object_bbox)
    representative_position = len(frames) // 2
    foreground_template, foreground_diagnostics = _reference_foreground_template(
        frames[representative_position], normalized_box
    )
    tracking = track_object_regions(
        frames,
        detected_features,
        normalized_box,
        representative_position=representative_position,
        config=config,
        foreground_template=foreground_template,
        foreground_diagnostics=foreground_diagnostics,
    )
    object_features, background_features, feature_separation = build_region_feature_caches(
        frames,
        detected_features,
        tracking["normalized_bboxes"],
        config,
        foreground_template=foreground_template,
    )
    tracking["feature_separation"] = feature_separation
    object_geometry = analyze_geometry(
        frames,
        frame_indices,
        timestamps_seconds,
        config,
        detected_features=object_features,
    )
    result = analyze_viewpoint_diversity(
        frames,
        frame_indices,
        timestamps_seconds,
        quality_summary,
        object_geometry,
        config,
        detected_features=object_features,
        camera_intrinsics=camera_intrinsics,
        maximum_separated_pairs=9,
        analysis_mode="object_focused",
    )
    background = _background_comparison(
        frames,
        frame_indices,
        timestamps_seconds,
        background_features,
        tracking["normalized_bboxes"],
        result["temporally_separated_pair_evidence"],
        config,
    )
    insufficient = list(
        result.get("decision_explanation", {}).get(
            "insufficient_evidence_reasons", []
        )
    )
    if not bool(tracking["foreground_separation"].get("available", False)):
        insufficient.insert(
            0,
            "The selected object foreground could not be separated reliably from background pixels inside the selection box.",
        )
    if float(tracking["tracked_sample_fraction"]) < 0.70:
        insufficient.insert(
            0,
            "The selected object region could not be tracked reliably across at least 70% of sampled frames."
        )
    tracking_endpoint_warning = None
    if not tracking["forward_reached_end"] or not tracking["backward_reached_start"]:
        tracking_endpoint_warning = (
            "Object-region tracking did not reach both ends of the sampled timeline; untracked endpoint pairs were excluded."
        )
    background_dominated_fraction = background[
        "background_dominated_pair_fraction"
    ]
    if (
        background_dominated_fraction is not None
        and float(background_dominated_fraction) >= 0.60
    ):
        insufficient.insert(
            0,
            "Background-only correspondences dominate most evaluated long-gap pairs."
        )
    animated_object_fraction = background.get(
        "moving_object_with_static_background_pair_fraction"
    )
    if (
        background["evaluated_pair_count"] >= 3
        and animated_object_fraction is not None
        and float(animated_object_fraction) >= 0.30
    ):
        insufficient.append(
            "The selected object moves while the surrounding background remains nearly static in multiple long-gap pairs; this is consistent with object animation rather than a camera orbit."
        )
    insufficient = list(dict.fromkeys(insufficient))
    if insufficient:
        result["label"] = "Unknown"
        result["evidence_status"] = "insufficient"
        result["summary"] = (
            "The selected object region does not provide enough reliable object-only geometry to classify viewpoint diversity."
        )
        result["warnings"] = list(dict.fromkeys(result["warnings"] + insufficient))
        decision = result.setdefault("decision_explanation", {})
        decision["primary_reason"] = insufficient[0]
        decision["insufficient_evidence_reasons"] = insufficient
        decision["classification_rule"] = "object_focused_insufficient_evidence_gate"
    if tracking_endpoint_warning is not None:
        result["warnings"] = list(
            dict.fromkeys(result["warnings"] + [tracking_endpoint_warning])
        )
        result.setdefault("decision_explanation", {}).setdefault(
            "limiting_evidence", []
        ).append(tracking_endpoint_warning)
    result["method"]["analysis_mode"] = "object_focused"
    result["method"]["object_bbox_used"] = True
    result["object_tracking"] = tracking
    result["background_diagnostics"] = background
    result["whole_frame_context"] = {
        "median_keypoints_per_frame": full_frame_geometry["summary"].get(
            "median_keypoints_per_frame"
        ),
        "median_verified_matches": full_frame_geometry["summary"].get(
            "median_verified_matches"
        ),
        "valid_geometry_pair_fraction": full_frame_geometry["summary"].get(
            "valid_geometry_pair_fraction"
        ),
        "note": "Whole-frame long-gap classification was skipped to avoid duplicate live-path work.",
    }
    return result, object_geometry

