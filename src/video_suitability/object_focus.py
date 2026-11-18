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


def _contains(point: Tuple[float, float], box: NormalizedBox, frame: np.ndarray) -> bool:
    x, y, width, height = _pixel_box(box, frame)
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _subset_features(
    keypoints: Sequence[Any],
    descriptors: Optional[np.ndarray],
    box: Optional[NormalizedBox],
    frame: np.ndarray,
    *,
    inside: bool,
) -> Tuple[List[Any], Optional[np.ndarray]]:
    if box is None or descriptors is None or not keypoints:
        return [], None
    indices = [
        index
        for index, keypoint in enumerate(keypoints)
        if _contains(keypoint.pt, box, frame) is inside
    ]
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
    source_indices = [
        index
        for index, keypoint in enumerate(source_keypoints)
        if _contains(keypoint.pt, source_box, source_frame)
    ]
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
    x, y, width, height = _pixel_box(source_box, source_frame)
    corners = np.float32(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    ).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, affine).reshape(-1, 2)
    target_height, target_width = target_frame.shape[:2]
    min_x = float(np.clip(np.min(transformed[:, 0]), 0, target_width - 1))
    max_x = float(np.clip(np.max(transformed[:, 0]), 1, target_width))
    min_y = float(np.clip(np.min(transformed[:, 1]), 0, target_height - 1))
    max_y = float(np.clip(np.max(transformed[:, 1]), 1, target_height))
    new_width = max_x - min_x
    new_height = max_y - min_y
    old_area = max(1.0, width * height)
    new_area = new_width * new_height
    area_ratio = new_area / old_area
    result["area_ratio"] = round(area_ratio, 6)
    if (
        new_width / target_width < 0.04
        or new_height / target_height < 0.04
        or new_width / target_width > 0.95
        or new_height / target_height > 0.95
        or not 0.45 <= area_ratio <= 2.20
    ):
        result["reason"] = "implausible_tracked_box"
        return None, result
    box = (
        min_x / target_width,
        min_y / target_height,
        new_width / target_width,
        new_height / target_height,
    )
    result.update({"status": "tracked", "reason": "affine_ransac_consensus"})
    return box, result


def track_object_regions(
    frames: Sequence[np.ndarray],
    detected_features: FeatureCache,
    initial_box: Union[Sequence[float], Mapping[str, float]],
    representative_position: Optional[int],
    config: AnalysisConfig,
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
        while 0 <= source + direction < len(frames):
            target = source + direction
            if boxes[source] is None:
                break
            propagated, evidence = _propagate_box(
                frames[source],
                frames[target],
                keypoints[source],
                keypoints[target],
                descriptors[source],
                descriptors[target],
                boxes[source],
                config,
            )
            evidence.update(
                {
                    "source_position": source,
                    "target_position": target,
                }
            )
            steps.append(evidence)
            if propagated is None:
                break
            boxes[target] = propagated
            source = target
    tracked_count = sum(item is not None for item in boxes)
    tracked_fraction = tracked_count / len(frames)
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
        "steps": sorted(steps, key=lambda item: item["target_position"]),
    }


def build_region_feature_caches(
    frames: Sequence[np.ndarray],
    detected_features: FeatureCache,
    normalized_bboxes: Sequence[Optional[Sequence[float]]],
) -> Tuple[FeatureCache, FeatureCache]:
    """Split the shared feature cache into object and background caches."""

    keypoints, descriptors = detected_features
    object_keypoints: List[List[Any]] = []
    object_descriptors: List[Optional[np.ndarray]] = []
    background_keypoints: List[List[Any]] = []
    background_descriptors: List[Optional[np.ndarray]] = []
    for frame, frame_keypoints, frame_descriptors, raw_box in zip(
        frames, keypoints, descriptors, normalized_bboxes
    ):
        box = tuple(raw_box) if raw_box is not None else None
        object_items = _subset_features(
            frame_keypoints, frame_descriptors, box, frame, inside=True
        )
        background_items = _subset_features(
            frame_keypoints, frame_descriptors, box, frame, inside=False
        )
        object_keypoints.append(object_items[0])
        object_descriptors.append(object_items[1])
        background_keypoints.append(background_items[0])
        background_descriptors.append(background_items[1])
    return (
        (object_keypoints, object_descriptors),
        (background_keypoints, background_descriptors),
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
            }
        )
    evaluated = [row for row in rows if row["evaluated"]]
    dominated_fraction = (
        float(np.mean([row["background_dominated"] for row in evaluated]))
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

    tracking = track_object_regions(
        frames,
        detected_features,
        object_bbox,
        representative_position=len(frames) // 2,
        config=config,
    )
    object_features, background_features = build_region_feature_caches(
        frames, detected_features, tracking["normalized_bboxes"]
    )
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
        insufficient.append(
            "Background-only correspondences dominate most evaluated long-gap pairs."
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

