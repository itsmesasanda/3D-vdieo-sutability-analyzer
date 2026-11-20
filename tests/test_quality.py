import numpy as np

from video_suitability.quality import analyze_quality, measure_frame_quality


def _checkerboard(size=256, tile=16):
    y, x = np.indices((size, size))
    gray = (((x // tile + y // tile) % 2) * 180 + 35).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_blur_and_darkness_are_measured():
    import cv2

    sharp = _checkerboard()
    blurry = cv2.GaussianBlur(sharp, (21, 21), 7.0)
    dark = np.clip(sharp.astype(np.float32) * 0.1, 0, 255).astype(np.uint8)

    sharp_metrics = measure_frame_quality(sharp, 0, 0.0)
    blur_metrics = measure_frame_quality(blurry, 1, 1.0)
    dark_metrics = measure_frame_quality(dark, 2, 2.0)

    assert sharp_metrics["laplacian_variance"] > blur_metrics["laplacian_variance"] * 5
    assert dark_metrics["luminance_mean"] < sharp_metrics["luminance_mean"] * 0.2
    assert "underexposure" in dark_metrics["issues"]
    assert "blur" in blur_metrics["issues"]


def test_quality_summary_has_fractions():
    bright = np.full((80, 120, 3), 250, dtype=np.uint8)
    normal = _checkerboard(80, 10)[:, :120]
    result = analyze_quality([bright, normal], [0, 1], [0.0, 1.0])
    assert result["summary"]["overexposed_frame_fraction"] == 0.5
    assert len(result["frames"]) == 2
    assert result["summary"]["exposure_measurement"]["scope"] == "whole_frame"
    assert (
        result["summary"]["exposure_measurement"]["foreground_status"]
        == "not_available"
    )


def test_smooth_low_detail_is_not_presented_as_proven_optical_blur():
    smooth = np.full((100, 140, 3), 128, dtype=np.uint8)
    metric = measure_frame_quality(smooth, 4, 0.5)
    assert "blur" in metric["issues"]
    assert "low_texture" in metric["issues"]
    assert metric["sharpness_proxy_status"] == "ambiguous_with_low_texture"

    result = analyze_quality([smooth], [4], [0.5])
    assert result["summary"]["blur_and_low_texture_frame_fraction"] == 1.0
    assert "not proof of optical blur" in result["summary"][
        "sharpness_measurement"
    ]["interpretation"]

