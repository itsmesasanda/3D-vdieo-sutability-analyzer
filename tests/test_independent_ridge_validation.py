import csv
from pathlib import Path

import pytest

from video_suitability.ml import independent_validation as validation


def _colmap_result(status, stage=None, registered=None, command_codes=(0, 0, 0)):
    return {
        "status": status,
        "failure_stage": stage,
        "failure_reason": "test failure" if status != "succeeded" else None,
        "commands": [{"returncode": value} for value in command_codes],
        "objective_metrics": (
            {"registered_image_count": registered}
            if registered is not None
            else None
        ),
    }


def test_manifest_resolves_relative_videos_and_frozen_colmap_policy(tmp_path):
    video = tmp_path / "front.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "videos.csv"
    manifest.write_text(
        "video_id,role,video_path\nfront,front_only,front.mp4\n",
        encoding="utf-8",
    )
    records = validation.load_validation_manifest(manifest)
    assert records == [
        {"video_id": "front", "role": "front_only", "video_path": video.resolve()}
    ]
    assert validation.FROZEN_COLMAP_SETTINGS == {
        "camera_model": "SIMPLE_RADIAL",
        "single_camera": True,
        "matcher": "sequential",
        "sequential_overlap": 6,
        "retry_exhaustive": False,
        "use_gpu": False,
        "max_image_size": 800,
        "max_num_features": 2048,
        "num_threads": 2,
    }


def test_missing_explicit_model_is_setup_failure_without_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(validation.IndependentValidationError, match="artifact is missing"):
        validation.run_independent_validation(
            [], output, model_artifact=tmp_path / "missing.joblib"
        )
    assert not output.exists()


def test_colmap_outcome_classification_never_converts_infrastructure_to_zero():
    success = validation.classify_colmap_result(
        _colmap_result("succeeded", registered=47)
    )
    mapper_zero = validation.classify_colmap_result(
        _colmap_result("reconstruction_failed", stage="sparse_mapping")
    )
    infrastructure = validation.classify_colmap_result(
        _colmap_result("infrastructure_unavailable", stage="environment")
    )
    mapper_crash = validation.classify_colmap_result(
        _colmap_result(
            "reconstruction_failed", stage="mapper_sequential", command_codes=(0, 0, 1)
        )
    )

    assert success["ground_truth_valid"] is True
    assert success["actual_registered_images"] == 47
    assert mapper_zero["reconstruction_status"] == "completed_mapper_failure"
    assert mapper_zero["ground_truth_valid"] is True
    assert mapper_zero["actual_registered_images"] == 0
    assert infrastructure["reconstruction_status"] == "infrastructure_failure"
    assert infrastructure["ground_truth_valid"] is False
    assert infrastructure["actual_registered_images"] is None
    assert mapper_crash["reconstruction_status"] == "pipeline_failure"
    assert mapper_crash["ground_truth_valid"] is False


def test_report_mae_uses_only_valid_completed_pairs(tmp_path):
    rows = [
        {
            "video_id": "good",
            "role": "good_orbit",
            "predicted_registration_percent": 80.0,
            "actual_registered_images": 45,
            "actual_registration_percent": 75.0,
            "absolute_error_percentage_points": 5.0,
            "ground_truth_valid": True,
            "reconstruction_status": "succeeded",
            "total_execution_seconds": 12.0,
        },
        {
            "video_id": "missing",
            "role": "front_only",
            "predicted_registration_percent": 70.0,
            "actual_registered_images": None,
            "actual_registration_percent": None,
            "absolute_error_percentage_points": None,
            "ground_truth_valid": False,
            "reconstruction_status": "infrastructure_failure",
            "total_execution_seconds": 1.0,
        },
        {
            "video_id": "zero",
            "role": "animated_control",
            "predicted_registration_percent": 20.0,
            "actual_registered_images": 0,
            "actual_registration_percent": 0.0,
            "absolute_error_percentage_points": 20.0,
            "ground_truth_valid": True,
            "reconstruction_status": "completed_mapper_failure",
            "total_execution_seconds": 8.0,
        },
    ]
    summary = validation.write_validation_report(tmp_path / "report.md", rows)
    assert summary["valid_ground_truth_count"] == 2
    assert summary["valid_error_count"] == 2
    assert summary["mae_percentage_points"] == 12.5
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "initial diagnostic study" in report
    assert "12.50 percentage points" in report


def test_runner_writes_required_csv_from_shared_60_images(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"read-only-video-fixture")

    def fake_prepare(_video, images):
        images.mkdir()
        prepared = []
        for index in range(60):
            name = f"{index:04d}.jpg"
            (images / name).write_bytes(b"jpeg" + bytes([index]))
            prepared.append(
                {
                    "filename": name,
                    "source_frame_position": index,
                    "sha256": "test",
                    "size_bytes": 5,
                }
            )
        return {
            "prepared_frame_count": 60,
            "prepared_images": prepared,
            "adapter_contract_sha256": validation.LIVE_VIDEO_ADAPTER_CONTRACT_SHA256,
        }

    monkeypatch.setattr(validation, "prepare_video_to_historical_60_jpegs", fake_prepare)
    monkeypatch.setattr(
        validation,
        "predict_experimental_ridge_from_prepared_images",
        lambda *_args, **_kwargs: {
            "status": "available",
            "predicted_registration_percentage": 80.0,
        },
    )
    monkeypatch.setattr(
        validation,
        "run_sparse_colmap",
        lambda *_args, **_kwargs: _colmap_result("succeeded", registered=45),
    )
    monkeypatch.setattr(validation, "_colmap_version", lambda _value: "COLMAP test")

    output = tmp_path / "output"
    result = validation.run_independent_validation(
        [{"video_id": "pilot", "role": "good_orbit", "video_path": source}],
        output,
    )
    assert result["summary"]["mae_percentage_points"] == 5.0
    with (output / "validation_results.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["predicted_registration_percent"] == "80.0"
    assert row["actual_registered_images"] == "45"
    assert row["actual_registration_percent"] == "75.0"
    assert row["absolute_error_percentage_points"] == "5.0"
    assert row["reconstruction_status"] == "succeeded"
    assert len(list((output / "runs/pilot/images").glob("*.jpg"))) == 60
