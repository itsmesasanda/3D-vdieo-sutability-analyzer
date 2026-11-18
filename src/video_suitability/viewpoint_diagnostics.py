"""Diagnostic pair selection and rendering for viewpoint validation audits."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for viewpoint diagnostics") from exc
    return cv2


def select_diagnostic_pairs(
    viewpoint_result: Mapping[str, Any], maximum_pairs: int = 3
) -> List[Dict[str, Any]]:
    """Select bounded, distinct pairs that expose the most useful evidence modes."""

    if maximum_pairs < 1:
        return []
    pairs = list(viewpoint_result.get("temporally_separated_pair_evidence", []))
    if not pairs:
        return []
    selected: List[Dict[str, Any]] = []
    used = set()

    def add(reason: str, candidates: Sequence[Mapping[str, Any]], key) -> None:
        if not candidates or len(selected) >= maximum_pairs:
            return
        pair = max(candidates, key=key)
        identity = (
            int(pair["sample_position_start"]),
            int(pair["sample_position_end"]),
        )
        if identity in used:
            return
        selected.append({"reason": reason, "pair": dict(pair)})
        used.add(identity)

    add(
        "repeated_view",
        [pair for pair in pairs if pair.get("repeated_view_detected")],
        lambda pair: int(pair.get("verified_matches") or 0),
    )
    add(
        "rotation_or_planar",
        [pair for pair in pairs if pair.get("strong_planar_or_rotation_evidence")],
        lambda pair: float(pair.get("homography_inlier_ratio") or 0.0),
    )
    add(
        "parallax_proxy",
        [pair for pair in pairs if pair.get("parallax_proxy_detected")],
        lambda pair: float(pair.get("homography_residual_p75_normalized") or 0.0),
    )
    add(
        "largest_view_change",
        pairs,
        lambda pair: float(pair.get("median_displacement_normalized") or 0.0),
    )
    if len(selected) < maximum_pairs:
        weakest = sorted(
            pairs,
            key=lambda pair: (
                int(pair.get("verified_matches") or 0),
                float(pair.get("spatial_coverage") or 0.0),
            ),
        )
        for pair in weakest:
            identity = (
                int(pair["sample_position_start"]),
                int(pair["sample_position_end"]),
            )
            if identity in used:
                continue
            selected.append({"reason": "weakest_geometry", "pair": dict(pair)})
            used.add(identity)
            if len(selected) >= maximum_pairs:
                break
    return selected[:maximum_pairs]


def _verified_matches(
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    config: AnalysisConfig,
) -> Tuple[List[Any], np.ndarray, str]:
    """Recreate the production ratio test and RANSAC model choice for rendering."""

    cv2 = _cv2()
    if descriptors_a is None or descriptors_b is None:
        return [], np.zeros(0, dtype=bool), "unavailable"
    candidates = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(
        descriptors_a, descriptors_b, k=2
    )
    good = [
        first
        for pair in candidates
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < config.ratio_test * second.distance
    ]
    if len(good) < 4:
        return good, np.zeros(len(good), dtype=bool), "unavailable"
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    fundamental_mask = None
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
    homography_mask = None
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
        homography_mask = None
    if fundamental_mask is not None and int(fundamental_mask.sum()) >= 8:
        return good, fundamental_mask, "fundamental_matrix_ransac"
    if homography_mask is not None and int(homography_mask.sum()) >= 4:
        return good, homography_mask, "homography_ransac_fallback"
    return good, np.zeros(len(good), dtype=bool), "unavailable"


def render_pair_diagnostic(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    keypoints_a: Sequence[Any],
    keypoints_b: Sequence[Any],
    descriptors_a: Optional[np.ndarray],
    descriptors_b: Optional[np.ndarray],
    pair_evidence: Mapping[str, Any],
    reason: str,
    config: AnalysisConfig,
    maximum_rendered_matches: int = 80,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render verified correspondences and image-space motion arrows.

    Green lines are RANSAC-verified matches, gray lines are rejected ratio-test
    matches, and yellow arrows show raw image displacement on the first frame. The
    arrows are diagnostics only; they do not assert camera translation.
    """

    cv2 = _cv2()
    if frame_a.ndim != 3 or frame_b.ndim != 3:
        raise ValueError("diagnostic frames must be BGR color images")
    height = max(frame_a.shape[0], frame_b.shape[0])
    width_a, width_b = frame_a.shape[1], frame_b.shape[1]
    banner_height = 94
    footer_height = 30
    canvas = np.full(
        (banner_height + height + footer_height, width_a + width_b, 3),
        24,
        dtype=np.uint8,
    )
    canvas[banner_height : banner_height + frame_a.shape[0], :width_a] = frame_a
    canvas[
        banner_height : banner_height + frame_b.shape[0],
        width_a : width_a + width_b,
    ] = frame_b
    matches, verified, model = _verified_matches(
        keypoints_a,
        keypoints_b,
        descriptors_a,
        descriptors_b,
        config,
    )
    verified_indices = np.flatnonzero(verified).tolist()
    rejected_indices = np.flatnonzero(~verified).tolist()
    ordered = verified_indices + rejected_indices
    if len(ordered) > maximum_rendered_matches:
        positions = np.rint(
            np.linspace(0, len(ordered) - 1, maximum_rendered_matches)
        ).astype(int)
        ordered = [ordered[int(position)] for position in positions]
    for index in ordered:
        match = matches[index]
        point_a = np.asarray(keypoints_a[match.queryIdx].pt, dtype=float)
        point_b = np.asarray(keypoints_b[match.trainIdx].pt, dtype=float)
        start = (int(round(point_a[0])), int(round(point_a[1])) + banner_height)
        end = (
            int(round(point_b[0])) + width_a,
            int(round(point_b[1])) + banner_height,
        )
        color = (70, 220, 90) if verified[index] else (105, 105, 105)
        thickness = 1 if verified[index] else 1
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, start, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 2, color, -1, cv2.LINE_AA)

    arrow_indices = verified_indices
    if len(arrow_indices) > 20:
        positions = np.rint(np.linspace(0, len(arrow_indices) - 1, 20)).astype(int)
        arrow_indices = [arrow_indices[int(position)] for position in positions]
    for index in arrow_indices:
        match = matches[index]
        point_a = np.asarray(keypoints_a[match.queryIdx].pt, dtype=float)
        point_b = np.asarray(keypoints_b[match.trainIdx].pt, dtype=float)
        motion_end = point_a + (point_b - point_a)
        motion_end[0] = np.clip(motion_end[0], 0, width_a - 1)
        motion_end[1] = np.clip(motion_end[1], 0, frame_a.shape[0] - 1)
        cv2.arrowedLine(
            canvas,
            (int(round(point_a[0])), int(round(point_a[1])) + banner_height),
            (
                int(round(motion_end[0])),
                int(round(motion_end[1])) + banner_height,
            ),
            (40, 220, 240),
            1,
            cv2.LINE_AA,
            tipLength=0.25,
        )

    start_time = float(pair_evidence.get("start_time_seconds") or 0.0)
    end_time = float(pair_evidence.get("end_time_seconds") or 0.0)
    displacement = pair_evidence.get("median_displacement_normalized")
    displacement_text = "n/a" if displacement is None else f"{float(displacement):.4f}"
    lines = [
        f"{reason} | {start_time:.3f}s -> {end_time:.3f}s | {model}",
        f"ratio matches {len(matches)} | verified {int(verified.sum())} | normalized displacement {displacement_text}",
        "parallax proxy {parallax} | repeated {repeated} | planar/rotation {planar}".format(
            parallax="yes" if pair_evidence.get("parallax_proxy_detected") else "no",
            repeated="yes" if pair_evidence.get("repeated_view_detected") else "no",
            planar=(
                "yes"
                if pair_evidence.get("strong_planar_or_rotation_evidence")
                else "no"
            ),
        ),
    ]
    for offset, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (10, 22 + offset * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "green: verified feature match | gray: rejected | yellow: image motion (not proof of translation)",
        (10, banner_height + height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    metadata = {
        "selection_reason": reason,
        "ratio_test_matches": len(matches),
        "verified_matches": int(verified.sum()),
        "verification_model": model,
        "rendered_match_count": len(ordered),
        "motion_arrow_count": len(arrow_indices),
    }
    return canvas, metadata

