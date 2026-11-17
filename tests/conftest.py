from pathlib import Path

import pytest

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.synthetic import generate_test_videos


@pytest.fixture(scope="session")
def synthetic_videos(tmp_path_factory):
    output = tmp_path_factory.mktemp("synthetic_videos")
    return generate_test_videos(output, width=400, height=300, frame_count=24, fps=12.0)


@pytest.fixture(scope="session")
def analysis_reports(synthetic_videos):
    config = AnalysisConfig(max_samples=16, max_analysis_side=400)
    return {
        name: analyze_video(str(path), config=config)
        for name, path in synthetic_videos.items()
    }

