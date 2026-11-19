"""Exact legacy eight-feature function and an experimental live-video adapter.

The trained historical Ridge model consumed features calculated from directories of
60 already-prepared JPEGs.  The image-directory extractor below reproduces that
function. The uploaded-video adapter is a new inference policy: it preserves the
legacy feature calculations, but equivalence to the CO3D training-input preparation
has not been established.
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from ..video import VideoValidationError


HISTORICAL_FEATURE_NAMES: Tuple[str, ...] = (
    "brightness_mean",
    "brightness_variation",
    "contrast_mean",
    "sharpness_mean",
    "sharpness_variation",
    "dark_fraction",
    "orb_features_mean",
    "orb_matches_mean",
)
HISTORICAL_EXTRACTOR_SOURCE_SHA256 = (
    "ac727c93fc14ffec7dd8079fb53f876f8a8b71560237ff9e319692d4288a2be0"
)
HISTORICAL_INPUT_FRAME_COUNT = 60
HISTORICAL_FEATURE_SAMPLE_COUNT = 12
LIVE_VIDEO_ADAPTER_POLICY = "uploaded_video_uniform60_pillow_jpeg90_v1"


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for historical features") from exc
    return cv2


def _extract_historical_features_from_bgr_frames(
    frames: List[np.ndarray],
) -> Tuple[Dict[str, float], List[List[int]]]:
    """Run the exact historical calculations on the 12 already-selected JPEGs."""

    if len(frames) != HISTORICAL_FEATURE_SAMPLE_COUNT:
        raise VideoValidationError(
            f"Expected 12 sampled prepared frames, got {len(frames)}"
        )
    cv2 = _cv2()
    brightness: List[float] = []
    contrast: List[float] = []
    blur: List[float] = []
    darkness: List[float] = []
    orb_counts: List[float] = []
    match_counts: List[float] = []
    previous_desc = None
    orb = cv2.ORB_create(nfeatures=400)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    resized_shapes = []
    for frame in frames:
        if frame is None:
            raise VideoValidationError("Cannot read a prepared frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if max(height, width) > 480:
            scale = 480 / max(height, width)
            gray = cv2.resize(gray, (round(width * scale), round(height * scale)))
        resized_shapes.append([int(gray.shape[1]), int(gray.shape[0])])
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        blur.append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())))
        darkness.append(float((gray < 32).mean()))
        keypoints, desc = orb.detectAndCompute(gray, None)
        orb_counts.append(float(len(keypoints)))
        if previous_desc is not None:
            if desc is not None and len(desc) >= 2:
                matches = matcher.knnMatch(previous_desc, desc, k=2)
                good = sum(
                    1
                    for pair in matches
                    if len(pair) == 2
                    and pair[0].distance < 0.75 * pair[1].distance
                )
                match_counts.append(float(good))
            else:
                match_counts.append(0.0)
        previous_desc = desc
    features = {
        "brightness_mean": float(np.mean(brightness)),
        "brightness_variation": float(np.std(brightness)),
        "contrast_mean": float(np.mean(contrast)),
        "sharpness_mean": float(np.mean(blur)),
        "sharpness_variation": float(np.std(blur)),
        "dark_fraction": float(np.mean(darkness)),
        "orb_features_mean": float(np.mean(orb_counts)),
        "orb_matches_mean": float(np.mean(match_counts)),
    }
    if tuple(features) != HISTORICAL_FEATURE_NAMES:
        raise RuntimeError("Historical feature order changed")
    return features, resized_shapes


def extract_historical_features_from_image_directory(
    image_directory: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Reproduce the notebook's legacy extractor on exactly 60 prepared JPEGs."""

    cv2 = _cv2()
    image_root = Path(image_directory).expanduser().resolve()
    paths = sorted(image_root.glob("*.jpg"))
    if len(paths) != HISTORICAL_INPUT_FRAME_COUNT:
        raise VideoValidationError(
            f"Expected 60 prepared JPEG frames in {image_root}, got {len(paths)}"
        )
    indices = np.linspace(
        0, len(paths) - 1, HISTORICAL_FEATURE_SAMPLE_COUNT, dtype=int
    )
    frames: List[np.ndarray] = []
    for index in indices:
        frame = cv2.imread(str(paths[index]))
        if frame is None:
            raise VideoValidationError(f"Cannot read prepared frame {paths[index]}")
        frames.append(frame)
    features, resized_shapes = _extract_historical_features_from_bgr_frames(frames)
    return features, {
        "input_adapter": "prepared_60_jpeg_directory",
        "sampled_positions_within_60": [int(value) for value in indices],
        "resized_gray_shapes": resized_shapes,
        "source_implementation_sha256": HISTORICAL_EXTRACTOR_SOURCE_SHA256,
    }


def extract_historical_features_from_uploaded_video(
    video_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Apply the explicit experimental live-video policy and legacy extractor.

    A logical ordered sequence of 60 frames is sampled uniformly across the codec's
    declared frame range. The legacy extractor only ever reads 12 fixed positions
    from that sequence, so only those 12 frames are decoded and JPEG-round-tripped.
    This is numerically equivalent to materializing all 60 adapter JPEGs, while
    avoiding 48 unnecessary decodes and writes. It does *not* establish equivalence
    to the historical CO3D input-generation domain.
    """

    cv2 = _cv2()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for live historical features") from exc

    source = Path(video_path).expanduser().resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"Could not open video: {source}")
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(raw_count) or raw_count < HISTORICAL_INPUT_FRAME_COUNT:
        capture.release()
        raise VideoValidationError(
            "Experimental Ridge adapter requires a reliable frame count of at least 60"
        )
    frame_count = int(round(raw_count))
    logical_positions = np.linspace(
        0, frame_count - 1, HISTORICAL_INPUT_FRAME_COUNT, dtype=int
    ).tolist()
    if len(set(logical_positions)) != HISTORICAL_INPUT_FRAME_COUNT:
        capture.release()
        raise VideoValidationError("Video does not yield 60 unique adapter positions")
    sampled_slots = np.linspace(
        0, HISTORICAL_INPUT_FRAME_COUNT - 1, HISTORICAL_FEATURE_SAMPLE_COUNT, dtype=int
    ).tolist()
    requested_positions = [logical_positions[index] for index in sampled_slots]
    prepared_frames: List[np.ndarray] = []
    decoded_positions: List[int] = []
    decode_wall_started = time.perf_counter()
    decode_cpu_started = time.process_time()
    try:
        for position in requested_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise VideoValidationError(f"Could not decode adapter frame {position}")
            reported_next = capture.get(cv2.CAP_PROP_POS_FRAMES)
            actual = int(round(reported_next)) - 1 if reported_next > 0 else position
            if actual != position:
                raise VideoValidationError(
                    f"Codec seek returned frame {actual}, expected {position}"
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB")
            factor = min(1.0, 800 / max(image.size))
            size = tuple(max(1, round(side * factor)) for side in image.size)
            if size != image.size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            image.save(encoded, "JPEG", quality=90)
            jpeg_array = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
            prepared = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
            if prepared is None:
                raise VideoValidationError(f"Could not decode prepared frame {position}")
            prepared_frames.append(prepared)
            decoded_positions.append(actual)
    finally:
        capture.release()
    decode_wall = time.perf_counter() - decode_wall_started
    decode_cpu = time.process_time() - decode_cpu_started
    feature_wall_started = time.perf_counter()
    feature_cpu_started = time.process_time()
    features, resized_shapes = _extract_historical_features_from_bgr_frames(
        prepared_frames
    )
    return features, {
        "input_adapter": LIVE_VIDEO_ADAPTER_POLICY,
        "adapter_status": "new_live_policy_training_input_equivalence_not_established",
        "source_frame_count": frame_count,
        "logical_prepared_frame_count": HISTORICAL_INPUT_FRAME_COUNT,
        "logical_uniform_positions": logical_positions,
        "sampled_positions_within_60": sampled_slots,
        "requested_source_positions": requested_positions,
        "decoded_source_positions": decoded_positions,
        "materialized_feature_frame_count": len(prepared_frames),
        "materialization_optimization": (
            "Only the 12 positions consumed by the historical extractor are decoded; "
            "the other 48 logical prepared frames cannot affect any feature."
        ),
        "preparation": (
            "codec-decoded BGR -> RGB -> Pillow RGB; proportional max-edge 800 "
            "LANCZOS resize; Pillow JPEG quality 90; OpenCV JPEG decode; then exact "
            "historical max-edge 480/default-resize grayscale calculations"
        ),
        "image_ordering": "ascending source frame position via logical names 0000.jpg..0059.jpg",
        "resized_gray_shapes": resized_shapes,
        "source_implementation_sha256": HISTORICAL_EXTRACTOR_SOURCE_SHA256,
        "decode_and_prepare_wall_seconds": round(decode_wall, 6),
        "decode_and_prepare_cpu_seconds": round(decode_cpu, 6),
        "feature_wall_seconds": round(time.perf_counter() - feature_wall_started, 6),
        "feature_cpu_seconds": round(time.process_time() - feature_cpu_started, 6),
    }


def prepare_video_for_historical_probe(video_path: Path, output_directory: Path) -> Dict[str, Any]:
    """Create an analogous 60-JPEG input solely for compatibility measurement.

    This is not claimed to be a recovered historical video adapter. Historical
    training began with ordered CO3D source JPEGs, while this function begins with
    codec-decoded video frames.
    """

    cv2 = _cv2()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for the historical video probe") from exc
    source = Path(video_path).expanduser().resolve()
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(f"Could not open video: {source}")
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(raw_count) or raw_count < HISTORICAL_INPUT_FRAME_COUNT:
        capture.release()
        raise VideoValidationError(
            "Historical probe requires a reliable frame count of at least 60"
        )
    frame_count = int(round(raw_count))
    positions = np.linspace(
        0, frame_count - 1, HISTORICAL_INPUT_FRAME_COUNT, dtype=int
    ).tolist()
    if len(set(positions)) != HISTORICAL_INPUT_FRAME_COUNT:
        capture.release()
        raise VideoValidationError("Video does not yield 60 unique probe positions")
    actual_positions = []
    for sequence_number, position in enumerate(positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(position))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise VideoValidationError(f"Could not decode probe frame {position}")
        reported_next = capture.get(cv2.CAP_PROP_POS_FRAMES)
        actual = int(round(reported_next)) - 1 if reported_next > 0 else position
        if actual != position:
            capture.release()
            raise VideoValidationError(
                f"Codec seek returned frame {actual}, expected {position}"
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGB")
        factor = min(1.0, 800 / max(image.size))
        size = tuple(max(1, round(side * factor)) for side in image.size)
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        image.save(root / f"{sequence_number:04d}.jpg", "JPEG", quality=90)
        actual_positions.append(actual)
    capture.release()
    return {
        "adapter_status": "probe_only_not_historically_verified",
        "source_frame_count": frame_count,
        "requested_positions": positions,
        "decoded_positions": actual_positions,
        "prepared_frame_count": HISTORICAL_INPUT_FRAME_COUNT,
        "preparation": "decoded BGR -> RGB, max-edge 800 LANCZOS, JPEG quality 90",
    }


def extract_historical_features_from_video_probe(
    video_path: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Measure the candidate adapter without authorizing Ridge live inference."""

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with tempfile.TemporaryDirectory(prefix="historical-video-probe-") as temporary:
        prepared = Path(temporary) / "images"
        preparation = prepare_video_for_historical_probe(video_path, prepared)
        feature_started_wall = time.perf_counter()
        feature_started_cpu = time.process_time()
        features, diagnostics = extract_historical_features_from_image_directory(
            prepared
        )
        diagnostics.update(
            {
                "video_adapter": preparation,
                "feature_wall_seconds": round(
                    time.perf_counter() - feature_started_wall, 6
                ),
                "feature_cpu_seconds": round(
                    time.process_time() - feature_started_cpu, 6
                ),
                "total_wall_seconds": round(time.perf_counter() - wall_started, 6),
                "total_cpu_seconds": round(time.process_time() - cpu_started, 6),
                "live_inference_authorized": False,
            }
        )
        return features, diagnostics
