import csv
import json
from pathlib import Path

import numpy as np

from video_suitability.reconstruction import (
    analyze_colmap_text_model,
    extract_reconstruction_frames,
    extraction_indices,
    render_sparse_preview,
    run_sparse_colmap,
    sha256_file,
)


def test_extraction_indices_cover_complete_timeline():
    indices = extraction_indices(frame_count=121, fps=30.0, sample_fps=2.0)
    assert indices[0] == 0
    assert indices[-1] == 120
    assert indices == sorted(set(indices))


def test_timestamped_extraction_preserves_source_and_orientation(synthetic_videos, tmp_path):
    source = synthetic_videos["smooth_orbit"]
    before = sha256_file(source)
    output = tmp_path / "extracted"
    metadata = extract_reconstruction_frames(
        source, output, sample_fps=3.0, max_side=320, jpeg_quality=90
    )
    after = sha256_file(source)

    assert before == after == metadata["source_sha256_after"]
    assert metadata["source_unchanged_after_extraction"] is True
    assert metadata["temporal_coverage_fraction"] == 1.0
    assert metadata["extracted_frame_count"] >= 2
    with (output / "frames_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == metadata["extracted_frame_count"]
    assert float(rows[0]["timestamp_seconds"]) == 0.0
    assert float(rows[-1]["timestamp_seconds"]) > 0.0
    assert all((output / "images" / row["filename"]).is_file() for row in rows)


def _write_tiny_colmap_model(directory: Path) -> None:
    directory.mkdir()
    (directory / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 500 500 320 240\n", encoding="utf-8"
    )
    (directory / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame_000.jpg\n"
        "10 20 1 30 40 2\n"
        "2 1 0 0 0 -1 0 0 1 frame_001.jpg\n"
        "12 22 1\n",
        encoding="utf-8",
    )
    (directory / "points3D.txt").write_text(
        "# 3D point list\n"
        "1 0 0 1 255 0 0 0.5 1 0 2 0\n"
        "2 1 0 1 0 255 0 1.5 1 1\n",
        encoding="utf-8",
    )


def test_colmap_text_metrics_and_preview_are_objective(tmp_path):
    model = tmp_path / "model"
    _write_tiny_colmap_model(model)
    metrics = analyze_colmap_text_model(model, total_input_images=4)
    assert metrics["registered_image_count"] == 2
    assert metrics["registered_image_ratio"] == 0.5
    assert metrics["reconstructed_point_count"] == 2
    assert metrics["mean_reprojection_error_pixels"] == 1.0
    assert metrics["mean_track_length_images"] == 1.5
    assert metrics["camera_center_max_baseline_arbitrary_units"] == 1.0
    assert metrics["camera_centers"][0]["qvec_world_to_camera_qw_qx_qy_qz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics["camera_centers"][1]["tvec_world_to_camera_xyz"] == [
        -1.0,
        0.0,
        0.0,
    ]
    assert "do not measure object surface completeness" in metrics["coverage_interpretation"]
    preview = render_sparse_preview(model, tmp_path / "preview.png", size=300)
    assert preview.is_file() and preview.stat().st_size > 0


def test_missing_colmap_is_infrastructure_unavailable_not_bad_video(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    result = run_sparse_colmap(
        images,
        tmp_path / "workspace",
        colmap_executable="definitely-not-a-colmap-executable",
    )
    assert result["status"] == "infrastructure_unavailable"
    assert result["objective_metrics"] is None
    assert result["mapper_runtime_seconds"] is None
    persisted = json.loads(
        (tmp_path / "workspace" / "reconstruction_result.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "infrastructure_unavailable"

