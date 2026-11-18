import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from video_suitability.ml.artifacts import sha256_file
from video_suitability.ml.registration_features import (
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features_from_frame_manifest,
    load_feature_spec,
)
from video_suitability.ml.registration_rf import verify_training_feature_parity
from video_suitability.ml.registration_training import (
    LABEL_SCHEMA_VERSION,
    REQUIRED_LABEL_COLUMNS,
    prepare_registration_training_bundle,
    registration_sampling_configuration,
)
from video_suitability.video import VideoValidationError


def _feature_spec():
    root = Path(__file__).resolve().parents[1]
    return load_feature_spec(
        root
        / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
    )


def _frame_input(root: Path, name: str):
    import cv2

    sequence = root / name
    images = sequence / "images"
    images.mkdir(parents=True)
    rows = []
    for index in range(60):
        y, x = np.indices((72, 96))
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        frame[..., 0] = (x * 3 + index * 5) % 256
        frame[..., 1] = (y * 4 + index * 7) % 256
        frame[..., 2] = ((x + y) * 2 + index * 11) % 256
        cv2.circle(frame, (12 + index % 65, 36), 7, (255, 255, 255), -1)
        filename = f"frame_{index:06d}.jpg"
        path = images / filename
        assert cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rows.append(
            {
                "sequence_number": index,
                "filename": filename,
                "requested_source_frame_index": index,
                "actual_source_frame_index": index,
                "timestamp_seconds": index / 10.0,
                "width": 96,
                "height": 72,
                "sha256": sha256_file(path),
            }
        )
    manifest = sequence / "frames_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_hash = sha256_file(images / rows[0]["filename"])
    metadata = {
        "schema_version": "1.0",
        "source_codec_fourcc": "test",
        "source_fps": 10.0,
        "source_frame_count": 60,
        "source_duration_seconds": 6.0,
        "source_sha256_before": source_hash,
        "source_sha256_after": source_hash,
        "source_unchanged_after_extraction": True,
        "orientation_metadata_degrees": 0.0,
        "opencv_orientation_auto_enabled": True,
        "requested_sample_fps": 10.0,
        "sampling_strategy": "uniform exact 60-frame test protocol",
        "requested_frame_count": 60,
        "extracted_frame_count": 60,
        "failed_source_frame_indices": [],
        "first_timestamp_seconds": 0.0,
        "last_timestamp_seconds": 5.9,
        "temporal_coverage_fraction": 1.0,
        "output_width": 96,
        "output_height": 72,
        "linear_resize_scale": 1.0,
        "jpeg_quality": 95,
    }
    metadata_path = sequence / "extraction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest, images, metadata_path


def _label_row(
    root: Path,
    *,
    record_id: str,
    group: str,
    variant: str,
    role: str,
    registered: int,
    status: str = "succeeded",
    failure_stage: str = "",
    failure_verified: bool = False,
):
    manifest, images, metadata = _frame_input(root, record_id)
    result = root / record_id / "reconstruction_result.json"
    if status == "succeeded":
        artifact = {
            "status": "succeeded",
            "failure_stage": None,
            "commands": [{"stage": "mapper"}],
            "total_input_image_count": 60,
            "objective_metrics": {
                "registered_image_count": registered,
                "total_input_image_count": 60,
                "registered_image_ratio": registered / 60.0,
            },
        }
    else:
        artifact = {
            "status": "reconstruction_failed",
            "failure_stage": failure_stage,
            "commands": [{"stage": "mapper", "returncode": 1}],
            "total_input_image_count": 60,
            "objective_metrics": None,
        }
    result.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    row = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "batch_id": "batch-a" if variant != "quarter" else "batch-b",
        "record_id": record_id,
        "sequence_id": record_id,
        "source_group": group,
        "source_group_provenance": "test_physical_object_mapping",
        "variant": variant,
        "evaluation_role": role,
        "reconstruction_status": status,
        "failure_stage": failure_stage,
        "failure_verified": str(failure_verified).lower(),
        "registered_image_count": str(registered),
        "total_input_image_count": "60",
        "registration_percentage": str(registered / 60.0 * 100.0),
        "label_provenance_kind": "colmap_sparse_reconstruction",
        "label_protocol_version": "colmap-60/v1",
        "label_artifact": str(result),
        "label_artifact_sha256": sha256_file(result),
        "label_generated_at_utc": "2026-11-18T00:00:00+00:00",
        "frames_manifest": str(manifest),
        "frames_manifest_sha256": sha256_file(manifest),
        "images_directory": str(images),
        "extraction_metadata": str(metadata),
        "extraction_metadata_sha256": sha256_file(metadata),
        "sampling_protocol_id": "colmap-exact-60/v1",
        "sampling_configuration_sha256": "",
        "input_fidelity": "exact_labeled_frames",
        "input_hash_provenance": "original_notebook_verified",
        "regeneration_provenance": "",
    }
    row["sampling_configuration_sha256"] = hashlib.sha256(
        json.dumps(
            registration_sampling_configuration(
                json.loads(metadata.read_text(encoding="utf-8"))
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return row


def _infrastructure_row(record_id: str):
    row = {name: "" for name in REQUIRED_LABEL_COLUMNS}
    row.update(
        {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "batch_id": "batch-infra",
            "record_id": record_id,
            "sequence_id": record_id,
            "source_group": "infra-group",
            "source_group_provenance": "test_fixture",
            "variant": "front",
            "evaluation_role": "development",
            "reconstruction_status": "infrastructure_unavailable",
            "failure_stage": "environment",
            "failure_verified": "false",
            "registration_percentage": "0",
        }
    )
    return row


def _write_batch(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_exact_60_frame_adapter_is_deterministic_and_refuses_missing_frame(tmp_path):
    manifest, images, _ = _frame_input(tmp_path, "sequence")
    first, diagnostics = extract_registration_features_from_frame_manifest(
        manifest, _feature_spec(), images_dir=images
    )
    second, _ = extract_registration_features_from_frame_manifest(
        manifest, _feature_spec(), images_dir=images
    )
    assert tuple(first) == REGISTRATION_FEATURE_NAMES
    np.testing.assert_allclose(list(first.values()), list(second.values()), rtol=0, atol=0)
    assert diagnostics["input_adapter"] == "colmap_frame_manifest"
    assert diagnostics["frame_count"] == 60
    assert diagnostics["sample_count"] == 16
    assert diagnostics["all_input_hashes_verified"] is True

    (images / "frame_000059.jpg").unlink()
    with pytest.raises(VideoValidationError, match="missing"):
        extract_registration_features_from_frame_manifest(
            manifest, _feature_spec(), images_dir=images
        )


def test_multi_batch_bundle_preserves_groups_holdout_failures_and_parity(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    batch_one_rows = [
        _label_row(
            inputs,
            record_id="object-a-front",
            group="physical-object-a",
            variant="front",
            role="development",
            registered=48,
        ),
        _label_row(
            inputs,
            record_id="object-b-failed",
            group="physical-object-b",
            variant="full",
            role="development",
            registered=0,
            status="reconstruction_failed",
            failure_stage="sparse_mapping",
            failure_verified=True,
        ),
        _infrastructure_row("environment-failure"),
    ]
    batch_two_rows = [
        _label_row(
            inputs,
            record_id="object-a-quarter",
            group="physical-object-a",
            variant="quarter",
            role="development",
            registered=42,
        ),
        _label_row(
            inputs,
            record_id="object-holdout",
            group="physical-object-holdout",
            variant="full",
            role="independent_holdout",
            registered=55,
        ),
    ]
    batch_one = tmp_path / "batch_one.csv"
    batch_two = tmp_path / "batch_two.csv"
    _write_batch(batch_one, batch_one_rows)
    _write_batch(batch_two, batch_two_rows)
    output = tmp_path / "bundle"
    root = Path(__file__).resolve().parents[1]
    report = prepare_registration_training_bundle(
        [batch_one, batch_two],
        output,
        root
        / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json",
        bundle_version="labels-merged-v1",
        fold_count=2,
        parity_case_count=1,
    )
    assert report["training_ready"] is True
    assert report["input_row_count"] == 5
    assert report["compatible_record_count"] == 4
    assert report["excluded_record_count"] == 1
    assert report["incompatible_record_count"] == 0
    assert "No numerical target was created" in report["exclusions"][0]["reason"]

    split = json.loads((output / "grouped_cv_manifest.json").read_text())
    assert split["independent_holdout_excluded_from_model_selection"] is True
    assert split["independent_holdout_record_ids"] == ["object-holdout"]
    assert set(split["development_group_fold_assignments"]) == {
        "physical-object-a",
        "physical-object-b",
    }

    records = [
        json.loads(line)
        for line in (output / "training_records.jsonl").read_text().splitlines()
    ]
    by_id = {record["record_id"]: record for record in records}
    assert by_id["object-a-front"]["source_group"] == "physical-object-a"
    assert by_id["object-a-quarter"]["variant"] == "quarter"
    assert by_id["object-b-failed"]["target"]["value"] == 0.0
    assert by_id["object-b-failed"]["target"]["failure_verified"] is True
    assert by_id["object-holdout"]["evaluation_role"] == "independent_holdout"
    assert by_id["object-a-front"]["input_provenance"]["sampling_configuration"]

    parity = json.loads((output / "parity_examples.json").read_text())
    assert parity["cases"][0]["input_type"] == "colmap_frame_manifest"
    parity_result = verify_training_feature_parity(output, _feature_spec(), parity)
    assert parity_result["status"] == "verified"
    assert parity_result["case_count"] == 1

    bundle_manifest = json.loads(
        (output / "training_bundle_manifest.json").read_text()
    )
    assert bundle_manifest["contains_trained_model"] is False
    assert bundle_manifest["activates_production_inference"] is False
    assert not (output / "model.joblib").exists()
    assert not (output / "artifact_manifest.json").exists()


def test_preprocessing_mismatch_is_reported_without_fabricated_features(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    valid_rows = [
        _label_row(
            inputs,
            record_id="dev-a",
            group="group-a",
            variant="front",
            role="development",
            registered=30,
        ),
        _label_row(
            inputs,
            record_id="dev-b",
            group="group-b",
            variant="full",
            role="development",
            registered=40,
        ),
        _label_row(
            inputs,
            record_id="holdout",
            group="group-h",
            variant="full",
            role="independent_holdout",
            registered=50,
        ),
    ]
    broken = _label_row(
        inputs,
        record_id="broken-input",
        group="group-c",
        variant="blurred",
        role="development",
        registered=10,
    )
    metadata_path = Path(broken["extraction_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["extracted_frame_count"] = 59
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    broken["extraction_metadata_sha256"] = sha256_file(metadata_path)
    batch = tmp_path / "batch.csv"
    _write_batch(batch, [*valid_rows, broken])
    root = Path(__file__).resolve().parents[1]
    report = prepare_registration_training_bundle(
        [batch],
        tmp_path / "bundle",
        root
        / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json",
        bundle_version="incompatible-v1",
        fold_count=2,
    )
    assert report["training_ready"] is False
    assert report["compatible_record_count"] == 3
    assert report["incompatible_record_count"] == 1
    assert "exactly 60 extracted frames" in report["incompatibilities"][0]["reason"]
    records = (tmp_path / "bundle" / "training_records.jsonl").read_text()
    assert "broken-input" not in records

