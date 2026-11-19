"""Deterministic synthetic MP4 fixtures for local validation and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to generate synthetic videos") from exc
    return cv2


def _build_layers(width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2 = _cv2()
    rng = np.random.default_rng(20261117)
    background = np.full((height, width, 3), (205, 215, 225), dtype=np.uint8)

    for x in range(0, width, 36):
        cv2.line(background, (x, 0), (x, height - 1), (175, 185, 195), 1)
    for y in range(0, height, 36):
        cv2.line(background, (0, y), (width - 1, y), (175, 185, 195), 1)
    for _ in range(180):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(1, 5))
        color = tuple(int(v) for v in rng.integers(40, 190, size=3))
        cv2.circle(background, center, radius, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        background,
        "STATIC BACKGROUND",
        (18, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (70, 80, 90),
        2,
        cv2.LINE_AA,
    )

    object_layer = np.zeros_like(background)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(45, width // 5), max(55, height // 3))
    cv2.ellipse(object_mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    object_layer[:] = (35, 60, 110)
    for offset in range(-axes[1], axes[1] + 1, 18):
        color = (50 + (offset % 60), 100 + (offset % 80), 170 + (offset % 70))
        cv2.line(
            object_layer,
            (center[0] - axes[0], center[1] + offset),
            (center[0] + axes[0], center[1] + offset + 35),
            color,
            4,
            cv2.LINE_AA,
        )
    for _ in range(100):
        point = (
            int(rng.integers(center[0] - axes[0], center[0] + axes[0] + 1)),
            int(rng.integers(center[1] - axes[1], center[1] + axes[1] + 1)),
        )
        if object_mask[point[1], point[0]]:
            color = tuple(int(v) for v in rng.integers(80, 255, size=3))
            cv2.circle(object_layer, point, int(rng.integers(2, 6)), color, -1, cv2.LINE_AA)
    cv2.putText(
        object_layer,
        "3D",
        (center[0] - 48, center[1] + 18),
        cv2.FONT_HERSHEY_DUPLEX,
        1.6,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    return background, object_layer, object_mask


def make_orbit_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Simulate a steady orbit using different foreground/background image motion."""

    cv2 = _cv2()
    background, object_layer, object_mask = _build_layers(width, height)
    center = (width / 2.0, height / 2.0)
    frames: List[np.ndarray] = []
    phases = np.linspace(-1.0, 1.0, frame_count)
    for phase in phases:
        bg_matrix = cv2.getRotationMatrix2D(center, float(phase * 1.5), 1.0)
        bg_matrix[0, 2] += float(phase * 7.0)
        bg_matrix[1, 2] += float(np.sin(phase * np.pi) * 2.0)
        warped_bg = cv2.warpAffine(
            background,
            bg_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        fg_matrix = cv2.getRotationMatrix2D(
            center, float(-phase * 7.0), float(1.0 + 0.035 * np.cos(phase * np.pi))
        )
        fg_matrix[0, 2] += float(phase * 24.0)
        fg_matrix[1, 2] += float(np.sin(phase * np.pi) * 5.0)
        warped_fg = cv2.warpAffine(
            object_layer,
            fg_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = cv2.warpAffine(
            object_mask,
            fg_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha = warped_mask.astype(np.float32)[..., None] / 255.0
        frame = warped_bg.astype(np.float32) * (1.0 - alpha) + warped_fg.astype(np.float32) * alpha
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def make_smooth_parallax_orbit_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Create smooth multi-layer relative motion as a controlled parallax fixture."""

    cv2 = _cv2()
    background, object_layer, object_mask = _build_layers(width, height)
    center = (width / 2.0, height / 2.0)
    frames: List[np.ndarray] = []
    phases = np.linspace(-1.0, 1.0, frame_count)
    band_edges = np.linspace(0, width, 4).astype(int)
    shifts = (-24.0, 0.0, 24.0)
    for phase in phases:
        layered = np.empty_like(background)
        for band in range(3):
            matrix = cv2.getRotationMatrix2D(
                center, float(phase * (band - 1) * 1.5), 1.0
            )
            matrix[0, 2] += float(phase * shifts[band])
            warped = cv2.warpAffine(
                background,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            start, end = band_edges[band], band_edges[band + 1]
            layered[:, start:end] = warped[:, start:end]

        foreground_matrix = cv2.getRotationMatrix2D(
            center, float(-phase * 6.0), float(1.0 + 0.04 * np.cos(phase * np.pi))
        )
        foreground_matrix[0, 2] += float(phase * 38.0)
        foreground_matrix[1, 2] += float(np.sin(phase * np.pi) * 4.0)
        foreground = cv2.warpAffine(
            object_layer,
            foreground_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpAffine(
            object_mask,
            foreground_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha = mask.astype(np.float32)[..., None] / 255.0
        frame = layered.astype(np.float32) * (1.0 - alpha) + foreground.astype(
            np.float32
        ) * alpha
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def make_pure_rotation_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Rotate one textured planar view, creating homography-only motion."""

    cv2 = _cv2()
    background, object_layer, object_mask = _build_layers(width, height)
    alpha = object_mask.astype(np.float32)[..., None] / 255.0
    base = np.clip(
        background.astype(np.float32) * (1.0 - alpha)
        + object_layer.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    center = (width / 2.0, height / 2.0)
    frames: List[np.ndarray] = []
    for angle in np.linspace(-10.0, 10.0, frame_count):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        frames.append(
            cv2.warpAffine(
                base,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return frames


def make_animated_object_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Move a textured object in front of a fixed, feature-rich background.

    This negative control intentionally provides strong matches and visible motion
    without camera translation or new viewpoints of the object.
    """

    cv2 = _cv2()
    background, object_layer, object_mask = _build_layers(width, height)
    center = (width / 2.0, height / 2.0)
    frames: List[np.ndarray] = []
    for phase in np.linspace(0.0, 2.0 * np.pi, frame_count):
        matrix = cv2.getRotationMatrix2D(
            center, float(7.0 * np.sin(phase * 1.5)), 1.0
        )
        matrix[0, 2] += float(28.0 * np.sin(phase))
        matrix[1, 2] += float(7.0 * np.sin(phase * 2.0))
        moving_object = cv2.warpAffine(
            object_layer,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        moving_mask = cv2.warpAffine(
            object_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha = moving_mask.astype(np.float32)[..., None] / 255.0
        frame = background.astype(np.float32) * (1.0 - alpha) + moving_object.astype(
            np.float32
        ) * alpha
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def make_viewpoint_sweep_frames(
    sweep_degrees: float,
    width: int = 480,
    height: int = 360,
    frame_count: int = 24,
) -> List[np.ndarray]:
    """Create a periodic, multi-depth 2D proxy for a controlled camera sweep.

    This is a plumbing/regression fixture, not a physically rendered object or
    reconstruction benchmark. A 360-degree sweep returns to the initial synthetic
    view, while intermediate layers move at different rates to provide parallax-like
    evidence. A 90-degree sweep covers only one quarter of the same periodic path.
    """

    cv2 = _cv2()
    background, object_layer, object_mask = _build_layers(width, height)
    center = (width / 2.0, height / 2.0)
    angles = np.linspace(0.0, np.radians(float(sweep_degrees)), frame_count)
    band_edges = np.linspace(0, width, 5).astype(int)
    depth_shifts = (-18.0, -6.0, 8.0, 22.0)
    frames: List[np.ndarray] = []
    for angle in angles:
        sine = float(np.sin(angle))
        cosine = float(np.cos(angle))
        layered = np.empty_like(background)
        for band, shift in enumerate(depth_shifts):
            matrix = cv2.getRotationMatrix2D(center, sine * (band - 1.5), 1.0)
            matrix[0, 2] += sine * shift
            matrix[1, 2] += (1.0 - cosine) * (band - 1.5) * 2.0
            warped = cv2.warpAffine(
                background,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            start, end = band_edges[band], band_edges[band + 1]
            layered[:, start:end] = warped[:, start:end]

        object_matrix = cv2.getRotationMatrix2D(
            center,
            -sine * 10.0,
            1.0 + (1.0 - cosine) * 0.035,
        )
        object_matrix[0, 2] += sine * 46.0
        object_matrix[1, 2] += (1.0 - cosine) * 7.0
        foreground = cv2.warpAffine(
            object_layer,
            object_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpAffine(
            object_mask,
            object_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha = mask.astype(np.float32)[..., None] / 255.0
        frame = layered.astype(np.float32) * (1.0 - alpha) + foreground.astype(
            np.float32
        ) * alpha
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def make_repetitive_ambiguous_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Create sharp tiled/symmetric footage with deliberately ambiguous ORB matches."""

    cv2 = _cv2()
    tile = np.zeros((40, 40, 3), dtype=np.uint8)
    for y in range(0, 40, 10):
        for x in range(0, 40, 10):
            tile[y : y + 10, x : x + 10] = (
                220 if (x // 10 + y // 10) % 2 else 35
            )
    repeats_y = int(np.ceil(height / tile.shape[0]))
    repeats_x = int(np.ceil(width / tile.shape[1]))
    base = np.tile(tile, (repeats_y, repeats_x, 1))[:height, :width]
    frames = []
    for shift in np.linspace(-20.0, 20.0, frame_count):
        matrix = np.float32([[1.0, 0.0, shift], [0.0, 1.0, 0.0]])
        frames.append(
            cv2.warpAffine(
                base,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        )
    return frames


def make_low_texture_frames(
    width: int = 480, height: int = 360, frame_count: int = 24
) -> List[np.ndarray]:
    """Create a smooth, nearly featureless centered-object fixture."""

    cv2 = _cv2()
    base = np.full((height, width, 3), (178, 182, 185), dtype=np.uint8)
    cv2.ellipse(
        base,
        (width // 2, height // 2),
        (max(30, width // 5), max(40, height // 3)),
        0,
        0,
        360,
        (158, 162, 165),
        -1,
        cv2.LINE_AA,
    )
    return [base.copy() for _ in range(frame_count)]


def make_shaky_orbit_frames(
    width: int = 480,
    height: int = 360,
    frame_count: int = 24,
    seed: int = 31415,
) -> List[np.ndarray]:
    """Apply deterministic abrupt jitter to an otherwise smooth orbit."""

    cv2 = _cv2()
    rng = np.random.default_rng(seed)
    frames = make_smooth_parallax_orbit_frames(width, height, frame_count)
    center = (width / 2.0, height / 2.0)
    output: List[np.ndarray] = []
    for index, frame in enumerate(frames):
        # Alternating impulses force direction and speed discontinuities.
        impulse = -1.0 if index % 2 else 1.0
        rotation = impulse * float(rng.uniform(2.0, 6.0))
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        matrix[0, 2] += impulse * float(rng.uniform(8.0, 22.0))
        matrix[1, 2] += float(rng.uniform(-14.0, 14.0))
        output.append(
            cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return output


def make_fast_motion_frames(
    width: int = 480, height: int = 360, frame_count: int = 48
) -> List[np.ndarray]:
    """Create a fast, directionally blurred traverse with otherwise smooth motion."""

    cv2 = _cv2()
    frames = make_smooth_parallax_orbit_frames(width, height, frame_count)
    output: List[np.ndarray] = []
    translations = np.linspace(-140.0, 140.0, frame_count)
    for frame, translation in zip(frames, translations):
        matrix = np.float32([[1.0, 0.0, translation], [0.0, 1.0, 0.0]])
        moved = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        output.append(cv2.GaussianBlur(moved, (11, 3), 3.0))
    return output


def write_mp4(path: Path, frames: List[np.ndarray], fps: float = 12.0) -> Path:
    cv2 = _cv2()
    if not frames:
        raise ValueError("frames cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV could not initialize the mp4v video writer")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("all frames must have the same dimensions")
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no data: {path}")
    return path


def generate_test_videos(
    output_dir: Path,
    width: int = 480,
    height: int = 360,
    frame_count: int = 24,
    fps: float = 12.0,
) -> Dict[str, Path]:
    """Generate orbit, stationary, blurry, and dark deterministic MP4 clips."""

    cv2 = _cv2()
    output_dir = Path(output_dir)
    orbit = make_orbit_frames(width=width, height=height, frame_count=frame_count)
    smooth_orbit = make_smooth_parallax_orbit_frames(
        width=width, height=height, frame_count=frame_count
    )
    quarter_orbit = make_viewpoint_sweep_frames(
        90.0, width=width, height=height, frame_count=frame_count
    )
    full_orbit = make_viewpoint_sweep_frames(
        360.0, width=width, height=height, frame_count=frame_count
    )
    middle = smooth_orbit[len(smooth_orbit) // 2]
    low_resolution = [
        cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        for frame in smooth_orbit
    ]
    variants = {
        "orbit_good": (orbit, fps),
        "smooth_orbit": (smooth_orbit, fps),
        "quarter_orbit": (quarter_orbit, fps),
        "full_orbit": (full_orbit, fps),
        "front_only": ([middle.copy() for _ in range(frame_count)], fps),
        "stationary": ([middle.copy() for _ in range(frame_count)], fps),
        "repetitive_ambiguous": (
            make_repetitive_ambiguous_frames(width, height, frame_count),
            fps,
        ),
        "low_texture": (
            make_low_texture_frames(width, height, frame_count),
            fps,
        ),
        "pure_rotation": (
            make_pure_rotation_frames(width, height, frame_count),
            fps,
        ),
        "animated_object": (
            make_animated_object_frames(width, height, frame_count),
            fps,
        ),
        "shaky_orbit": (
            make_shaky_orbit_frames(width, height, frame_count),
            fps,
        ),
        "fast_motion": (
            make_fast_motion_frames(width, height, max(48, frame_count * 2)),
            48.0,
        ),
        "low_resolution": (low_resolution, fps),
        "blurry": (
            [
                cv2.GaussianBlur(frame, (21, 21), 7.0)
                for frame in smooth_orbit
            ],
            fps,
        ),
        "dark": (
            [
                np.clip(frame.astype(np.float32) * 0.14, 0, 255).astype(np.uint8)
                for frame in smooth_orbit
            ],
            fps,
        ),
        "insufficient_coverage": (smooth_orbit[:5], fps),
    }
    return {
        name: write_mp4(output_dir / f"{name}.mp4", frames, fps=variant_fps)
        for name, (frames, variant_fps) in variants.items()
    }

