import csv
import gzip
import json
from pathlib import Path

import numpy as np

from video_suitability.ml.artifacts import sha256_file
from video_suitability.ml.colab_label_import import import_colab_registration_labels
from video_suitability.ml.registration_training import prepare_registration_training_bundle


def _write_co3d_sequence(root: Path, *, category: str, sequence: str, count: int) -> None:
    import cv2

    category_dir = root / category
    images_dir = category_dir / sequence / "images"
    images_dir.mkdir(parents=True)
    annotations = []
    for index in range(count):
        y, x = np.indices((48, 64))
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[..., 0] = (x * 5 + index * 3) % 256
        image[..., 1] = (y * 7 + index * 5) % 256
        image[..., 2] = ((x + y) * 3 + index * 11) % 256
        cv2.circle(image, (8 + index % 48, 24), 5, (255, 255, 255), -1)
        filename = f"frame{index:06d}.jpg"
        assert cv2.imwrite(str(images_dir / filename), image)
        annotations.append(
            {
                "sequence_name": sequence,
                "frame_number": index,
                "frame_timestamp": index / 30.0,
                "image": {
                    "path": f"{category}/{sequence}/images/{filename}",
                    "size": [48, 64],
                },
            }
        )
    with gzip.open(category_dir / "frame_annotations.jgz", "wt", encoding="utf-8") as handle:
        json.dump(annotations, handle)
    with gzip.open(category_dir / "sequence_annotations.jgz", "wt", encoding="utf-8") as handle:
        json.dump([{"sequence_name": sequence, "object_id": "physical-object-1"}], handle)


def _write_source_result(path: Path, *, total: int, registered: int, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reconstruction_status": status,
        "n_sampled_frames": total,
        "n_registered_cameras": registered,
        "registration_ratio": registered / total,
        "n_points_3d": 1200 if registered else 0,
        "mean_reprojection_error": 0.8 if registered else None,
        "mean_track_length": 3.1 if registered else None,
        "commands": [{"stage": "mapper", "returncode": 0}],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_completed_batch_fixture(root: Path) -> tuple[Path, Path, Path]:
    results = root / "results"
    rows = []
    cases = [
        ("accepted_exact_60", 60, 45, "success"),
        ("legacy_16_frames", 16, 12, "success"),
        ("infrastructure_missing", 60, 0, "unavailable"),
    ]
    for sequence, total, registered, status in cases:
        result = results / f"toy__{sequence}.json"
        _write_source_result(
            result, total=total, registered=registered, status=status
        )
        rows.append(
            {
                "category": "toy",
                "sequence_name": sequence,
                "n_sampled_frames": total,
                "n_registered_cameras": registered,
                "registration_ratio": registered / total,
                "mean_reprojection_error": 0.8 if registered else "",
                "n_points_3d": 1200 if registered else 0,
                "mean_track_length": 3.1 if registered else "",
                "quality_score": 88.0 if registered else 0.0,
                "label": "Good" if registered else "Poor",
                "reconstruction_status": status,
                "error": "COLMAP executable missing" if status == "unavailable" else "",
                "sampled_frame_paths": "[]",
                "reconstruction_result": str(result),
            }
        )
    csv_path = root / "sequence_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_map = root / "group_map.csv"
    with group_map.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "category",
            "sequence_name",
            "source_group",
            "source_group_provenance",
            "variant",
            "evaluation_role",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sequence, *_ in cases:
            writer.writerow(
                {
                    "category": "toy",
                    "sequence_name": sequence,
                    "source_group": f"physical:{sequence}",
                    "source_group_provenance": "test_reviewed_mapping",
                    "variant": "full_orbit",
                    "evaluation_role": "development",
                }
            )
    evidence = root / "completed_batch_notebook.ipynb"
    evidence.write_text(
        json.dumps({"sampling": "np.linspace(...).astype(int)", "count": 60}) + "\n",
        encoding="utf-8",
    )
    return csv_path, group_map, evidence


def test_completed_legacy_batch_import_accepts_only_evidenced_exact_60(tmp_path):
    batch = tmp_path / "completed_batch_c_fixture"
    batch.mkdir()
    csv_path, group_map, evidence = _write_completed_batch_fixture(batch)
    source_hash = sha256_file(csv_path)
    co3d = tmp_path / "co3d"
    _write_co3d_sequence(
        co3d, category="toy", sequence="accepted_exact_60", count=80
    )

    imported = tmp_path / "imported"
    report = import_colab_registration_labels(
        [csv_path],
        imported,
        batch_roots=[batch],
        co3d_root=co3d,
        group_map_csv=group_map,
        regeneration_sampling_strategy="legacy_floor_linspace",
        regeneration_evidence_path=evidence,
        batch_id_prefix="completed-batch-c-fixture",
    )

    assert sha256_file(csv_path) == source_hash
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 2
    assert report["source_csvs"][0]["unchanged"] is True
    assert report["accepted"][0]["sequence_name"] == "accepted_exact_60"
    assert report["accepted"][0]["input_fidelity"] == "faithfully_regenerated"
    assert (
        report["accepted"][0]["input_hash_provenance"]
        == "faithfully_regenerated_from_co3d"
    )
    reasons = {item["sequence_name"]: item["reason"] for item in report["rejected"]}
    assert "requires exactly 60" in reasons["legacy_16_frames"]
    assert "no training target" in reasons["infrastructure_missing"]

    canonical_csv = Path(report["canonical_labels_csv"])
    with canonical_csv.open(newline="", encoding="utf-8") as handle:
        canonical = list(csv.DictReader(handle))
    assert canonical[0]["registration_percentage"] == "75.0"
    assert canonical[0]["source_group_provenance"] == "test_reviewed_mapping"
    assert canonical[0]["label_generated_at_utc"] == "original_label_generation_time_not_recorded"
    provenance = json.loads(canonical[0]["regeneration_provenance"])
    assert provenance["sampling_declaration_evidence_sha256"] == sha256_file(evidence)

    repository = Path(__file__).resolve().parents[1]
    compatibility = prepare_registration_training_bundle(
        [canonical_csv],
        tmp_path / "training_bundle",
        repository
        / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json",
        bundle_version="import-fixture-v1",
        fold_count=2,
    )
    assert compatibility["compatible_record_count"] == 1
    assert compatibility["incompatible_record_count"] == 0
    assert compatibility["model_trained"] is False
    assert compatibility["training_ready"] is False
    assert "independent holdout" in compatibility["grouped_cv_error"]
