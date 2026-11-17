"""Input validation and memory-bounded whole-video temporal sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig


class VideoError(RuntimeError):
    """Base error for invalid or undecodable video input."""


class VideoDependencyError(VideoError):
    """Raised when the OpenCV runtime dependency is unavailable."""


class VideoValidationError(VideoError):
    """Raised when the input path is not a usable local video."""


@dataclass
class SampledVideo:
    """Decoded analysis frames and their source positions."""

    frames: List[np.ndarray]
    frame_indices: List[int]
    timestamps_seconds: List[float]
    metadata: Dict[str, Any]
    warnings: List[str]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in dependency-less installs
        raise VideoDependencyError(
            "OpenCV is required. Install the project with `python -m pip install -e .`."
        ) from exc
    return cv2


def uniform_sample_indices(frame_count: int, sample_count: int) -> List[int]:
    """Return unique, ordered indices spanning the complete known frame range."""

    if frame_count <= 0:
        return []
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if frame_count <= sample_count:
        return list(range(frame_count))
    indices = np.rint(np.linspace(0, frame_count - 1, sample_count)).astype(int)
    return sorted(set(int(i) for i in indices))


def _resize_for_analysis(frame: np.ndarray, max_side: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return frame
    scale = max_side / float(largest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _decode_at_indices(
    path: Path, indices: Sequence[int], fps: float, max_side: int
) -> Tuple[List[np.ndarray], List[int], List[float], int]:
    """Seek to known indices; return frames plus a count of failed seeks."""

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(f"Could not open video with OpenCV: {path}")

    frames: List[np.ndarray] = []
    actual_indices: List[int] = []
    timestamps: List[float] = []
    failed = 0
    for requested in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(requested))
        ok, frame = cap.read()
        if not ok or frame is None:
            failed += 1
            continue
        reported_next = cap.get(cv2.CAP_PROP_POS_FRAMES)
        actual = max(0, int(round(reported_next)) - 1) if reported_next > 0 else requested
        if actual_indices and actual == actual_indices[-1]:
            failed += 1
            continue
        timestamp = (actual / fps) if fps > 0 else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames.append(_resize_for_analysis(frame, max_side))
        actual_indices.append(actual)
        timestamps.append(max(0.0, float(timestamp)))
    cap.release()
    return frames, actual_indices, timestamps, failed


def _decode_sequential_at_indices(
    path: Path, indices: Sequence[int], fps: float, max_side: int
) -> Tuple[List[np.ndarray], List[int], List[float], int]:
    """Decode one forward pass and retrieve only requested frames.

    Repeated random seeking in long-GOP HEVC video can decode the same dependency
    frames many times. A single pass is materially faster when requested samples are
    relatively dense, while still retaining only the bounded set of resized frames.
    """

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(f"Could not open video with OpenCV: {path}")
    requested = sorted(set(int(index) for index in indices))
    targets = set(requested)
    frames: List[np.ndarray] = []
    actual_indices: List[int] = []
    timestamps: List[float] = []
    if not requested:
        cap.release()
        return frames, actual_indices, timestamps, 0
    final_index = requested[-1]
    for frame_index in range(final_index + 1):
        if not cap.grab():
            break
        if frame_index not in targets:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        timestamp = (
            frame_index / fps
            if fps > 0
            else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        )
        frames.append(_resize_for_analysis(frame, max_side))
        actual_indices.append(frame_index)
        timestamps.append(max(0.0, float(timestamp)))
    cap.release()
    failed = len(requested) - len(actual_indices)
    return frames, actual_indices, timestamps, failed


def _prefer_sequential_decode(indices: Sequence[int]) -> bool:
    """Choose one-pass decoding only when uniform samples are relatively dense."""

    if len(indices) < 2:
        return False
    average_gap = (indices[-1] - indices[0]) / max(1, len(indices) - 1)
    return average_gap <= 64.0


def _count_frames_without_metadata(path: Path, limit: int) -> Tuple[int, bool]:
    """Count frames using grab(), bounded for malformed/stream-like inputs."""

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    count = 0
    reached_limit = False
    while count < limit and cap.grab():
        count += 1
    if count == limit:
        reached_limit = bool(cap.grab())
    cap.release()
    return count, reached_limit


def _codec_name(raw_fourcc: float) -> str:
    value = int(raw_fourcc)
    chars = [chr((value >> (8 * i)) & 0xFF) for i in range(4)]
    codec = "".join(c for c in chars if c.isprintable()).strip("\x00 ")
    return codec or "unknown"


def sample_video(path: Path, config: AnalysisConfig) -> SampledVideo:
    """Validate and sample frames uniformly over the full declared video duration.

    Only sampled, downscaled frames are retained, so peak frame memory is bounded by
    ``max_samples * max_analysis_side`` rather than by video length. When a codec does
    not expose a frame count, the function performs a bounded metadata pass before
    sampling and reports if full temporal coverage could not be guaranteed.
    """

    config.validate()
    cv2 = _cv2()
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise VideoValidationError(f"Video does not exist: {input_path}")
    if not input_path.is_file():
        raise VideoValidationError(f"Video path is not a file: {input_path}")
    if input_path.stat().st_size <= 0:
        raise VideoValidationError(f"Video file is empty: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(
            f"Could not open video: {input_path}. The file may be corrupt or use an unsupported codec."
        )

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps_raw if np.isfinite(fps_raw) and fps_raw > 1e-3 else 0.0
    frame_count_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = (
        int(round(frame_count_raw))
        if np.isfinite(frame_count_raw) and frame_count_raw >= 1
        else 0
    )
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    codec = _codec_name(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    warnings: List[str] = []
    metadata_frame_count_available = frame_count > 0
    full_coverage_known = True
    if frame_count <= 0:
        frame_count, truncated = _count_frames_without_metadata(
            input_path, config.max_decode_frames_fallback
        )
        full_coverage_known = not truncated
        warnings.append(
            "The codec did not report a frame count; a bounded frame-count pass was used."
        )
        if truncated:
            warnings.append(
                "Frame counting reached the configured fallback limit, so analysis may not cover the video tail."
            )
    if frame_count <= 0:
        raise VideoValidationError(f"No decodable frames found in video: {input_path}")

    requested = uniform_sample_indices(frame_count, config.max_samples)
    decode_strategy = "sparse_random_seek"
    if _prefer_sequential_decode(requested):
        decode_strategy = "single_forward_pass"
        frames, actual_indices, timestamps, failed = _decode_sequential_at_indices(
            input_path, requested, fps, config.max_analysis_side
        )
        if failed:
            # A full sparse-seek retry avoids silently losing coverage on codecs
            # whose grab/retrieve behavior is unreliable.
            frames, actual_indices, timestamps, failed = _decode_at_indices(
                input_path, requested, fps, config.max_analysis_side
            )
            decode_strategy = "sparse_random_seek_after_forward_pass_fallback"
    else:
        frames, actual_indices, timestamps, failed = _decode_at_indices(
            input_path, requested, fps, config.max_analysis_side
        )
    if not frames:
        raise VideoValidationError(f"No sampled frames could be decoded from: {input_path}")
    if failed:
        warnings.append(
            f"{failed} of {len(requested)} requested sample positions could not be decoded uniquely."
        )

    coverage_fraction = (
        actual_indices[-1] / max(1, frame_count - 1) if actual_indices else 0.0
    )
    if full_coverage_known and coverage_fraction < 0.90 and frame_count > 1:
        warnings.append(
            "Decoded samples did not reach the final 10% of the declared video; codec seeking may be unreliable."
        )
        full_coverage_known = False

    duration = (frame_count / fps) if fps > 0 else None
    file_size_bytes = int(input_path.stat().st_size)
    bitrate_mbps = (
        file_size_bytes * 8.0 / duration / 1_000_000.0
        if duration is not None and duration > 0
        else None
    )
    compressed_bits_per_pixel_per_frame = (
        file_size_bytes * 8.0 / frame_count / width / height
        if frame_count > 0 and width > 0 and height > 0
        else None
    )
    source_megapixels = width * height / 1_000_000.0 if width > 0 and height > 0 else None
    analyzed_shape = list(frames[0].shape[:2][::-1])
    analysis_scale = (
        min(analyzed_shape[0] / width, analyzed_shape[1] / height)
        if width > 0 and height > 0
        else None
    )
    metadata: Dict[str, Any] = {
        "path": str(input_path.resolve()),
        "file_size_bytes": file_size_bytes,
        "codec": codec,
        "fps": round(fps, 6) if fps > 0 else None,
        "frame_count": int(frame_count),
        "frame_count_from_codec_metadata": metadata_frame_count_available,
        "duration_seconds": round(float(duration), 6) if duration is not None else None,
        "source_width": width or None,
        "source_height": height or None,
        "source_megapixels": round(float(source_megapixels), 6)
        if source_megapixels is not None
        else None,
        "estimated_bitrate_mbps": round(float(bitrate_mbps), 6)
        if bitrate_mbps is not None
        else None,
        "compressed_bits_per_pixel_per_frame": round(
            float(compressed_bits_per_pixel_per_frame), 9
        )
        if compressed_bits_per_pixel_per_frame is not None
        else None,
        "analysis_width": analyzed_shape[0],
        "analysis_height": analyzed_shape[1],
        "analysis_linear_scale_from_source": round(float(analysis_scale), 6)
        if analysis_scale is not None
        else None,
        "requested_sample_count": len(requested),
        "decoded_sample_count": len(frames),
        "decode_strategy": decode_strategy,
        "sampling_strategy": "uniform indices across the complete declared frame range",
        "temporal_coverage_fraction": round(float(np.clip(coverage_fraction, 0.0, 1.0)), 6),
        "full_temporal_coverage_confirmed": bool(full_coverage_known),
    }
    return SampledVideo(frames, actual_indices, timestamps, metadata, warnings)
