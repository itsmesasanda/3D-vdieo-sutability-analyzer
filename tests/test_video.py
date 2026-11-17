from pathlib import Path

import pytest

from video_suitability.config import AnalysisConfig
from video_suitability.video import VideoValidationError, sample_video, uniform_sample_indices
from video_suitability.video import _prefer_sequential_decode


def test_uniform_indices_span_entire_video():
    assert uniform_sample_indices(101, 5) == [0, 25, 50, 75, 100]
    assert uniform_sample_indices(3, 8) == [0, 1, 2]


def test_decode_strategy_uses_forward_pass_only_for_dense_sampling():
    assert _prefer_sequential_decode([0, 32, 64, 96]) is True
    assert _prefer_sequential_decode([0, 200, 400, 600]) is False


def test_video_sampling_reaches_start_and_end(synthetic_videos):
    sampled = sample_video(
        synthetic_videos["orbit_good"],
        AnalysisConfig(max_samples=8, max_analysis_side=320),
    )
    assert len(sampled.frames) == 8
    assert sampled.frame_indices[0] <= 1
    assert sampled.frame_indices[-1] >= sampled.metadata["frame_count"] - 2
    assert sampled.timestamps_seconds == sorted(sampled.timestamps_seconds)
    assert sampled.metadata["temporal_coverage_fraction"] >= 0.95
    assert sampled.metadata["decode_strategy"] == "single_forward_pass"
    assert max(sampled.frames[0].shape[:2]) <= 320


def test_missing_and_empty_video_fail_cleanly(tmp_path):
    config = AnalysisConfig(max_samples=8)
    with pytest.raises(VideoValidationError, match="does not exist"):
        sample_video(tmp_path / "missing.mp4", config)
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(VideoValidationError, match="empty"):
        sample_video(empty, config)

