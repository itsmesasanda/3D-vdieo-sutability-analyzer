"""Per-frame visual quality measurements and timestamped issue flags."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for quality analysis") from exc
    return cv2


def _gray_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = hist / max(1.0, hist.sum())
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))


def measure_frame_quality(
    frame_bgr: np.ndarray, frame_index: int, timestamp_seconds: float
) -> Dict[str, Any]:
    """Compute deterministic quality proxies for one BGR frame."""

    cv2 = _cv2()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luminance = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = float(np.mean(cv2.magnitude(sobel_x, sobel_y)))
    dark_fraction = float(np.mean(gray <= 20))
    bright_fraction = float(np.mean(gray >= 235))
    entropy = _gray_entropy(gray)

    issues: List[str] = []
    if luminance < 55.0 or dark_fraction > 0.45:
        issues.append("underexposure")
    if luminance > 205.0 or bright_fraction > 0.35:
        issues.append("overexposure")
    if contrast < 22.0:
        issues.append("low_contrast")
    if laplacian_variance < 45.0:
        issues.append("blur")
    if entropy < 3.2 or gradient_magnitude < 5.0:
        issues.append("low_texture")

    return {
        "frame_index": int(frame_index),
        "timestamp_seconds": round(float(timestamp_seconds), 6),
        "luminance_mean": round(luminance, 6),
        "contrast_std": round(contrast, 6),
        "laplacian_variance": round(laplacian_variance, 6),
        "gradient_magnitude": round(gradient_magnitude, 6),
        "entropy_bits": round(entropy, 6),
        "dark_pixel_fraction": round(dark_fraction, 6),
        "bright_pixel_fraction": round(bright_fraction, 6),
        "issues": issues,
    }


def _median(metrics: Sequence[Dict[str, Any]], key: str) -> float:
    return float(np.median([float(item[key]) for item in metrics]))


def analyze_quality(
    frames: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    timestamps_seconds: Sequence[float],
) -> Dict[str, Any]:
    """Measure all frames and aggregate robust video-level quality statistics."""

    if not (len(frames) == len(frame_indices) == len(timestamps_seconds)):
        raise ValueError("frames, frame_indices, and timestamps_seconds must have equal length")
    if not frames:
        raise ValueError("at least one frame is required")

    frame_metrics = [
        measure_frame_quality(frame, index, timestamp)
        for frame, index, timestamp in zip(frames, frame_indices, timestamps_seconds)
    ]
    luminances = np.asarray([m["luminance_mean"] for m in frame_metrics], dtype=float)

    def issue_fraction(issue: str) -> float:
        return float(np.mean([issue in m["issues"] for m in frame_metrics]))

    summary = {
        "median_luminance": round(_median(frame_metrics, "luminance_mean"), 6),
        "luminance_std_across_frames": round(float(np.std(luminances)), 6),
        "median_contrast_std": round(_median(frame_metrics, "contrast_std"), 6),
        "median_laplacian_variance": round(
            _median(frame_metrics, "laplacian_variance"), 6
        ),
        "median_gradient_magnitude": round(
            _median(frame_metrics, "gradient_magnitude"), 6
        ),
        "median_entropy_bits": round(_median(frame_metrics, "entropy_bits"), 6),
        "median_dark_pixel_fraction": round(
            _median(frame_metrics, "dark_pixel_fraction"), 6
        ),
        "median_bright_pixel_fraction": round(
            _median(frame_metrics, "bright_pixel_fraction"), 6
        ),
        "underexposed_frame_fraction": round(issue_fraction("underexposure"), 6),
        "overexposed_frame_fraction": round(issue_fraction("overexposure"), 6),
        "blurry_frame_fraction": round(issue_fraction("blur"), 6),
        "low_contrast_frame_fraction": round(issue_fraction("low_contrast"), 6),
        "low_texture_frame_fraction": round(issue_fraction("low_texture"), 6),
    }
    return {"summary": summary, "frames": frame_metrics}
