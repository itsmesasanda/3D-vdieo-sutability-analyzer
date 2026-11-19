import csv
import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2

from scripts.validate_viewpoint_real_videos import (
    CSV_COLUMNS,
    misleading_risk_flags,
    run_validation,
)
from video_suitability.config import AnalysisConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_folder_validator_writes_csv_reports_and_diagnostics(
    synthetic_videos, tmp_path
):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    selected = {}
    for name in ("front_only", "full_orbit"):
        target = input_dir / f"{name}.mp4"
        shutil.copyfile(synthetic_videos[name], target)
        selected[name] = (target, _sha256(target))

    result = run_validation(
        input_dir,
        output_dir,
        AnalysisConfig(max_samples=16, max_analysis_side=400),
        maximum_diagnostic_pairs=2,
    )
    assert result == 0
    with (output_dir / "validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(CSV_COLUMNS) == set(rows[0])
    by_name = {row["filename"]: row for row in rows}
    assert by_name["front_only.mp4"]["predicted_viewpoint_category"] == "Low"
    assert by_name["full_orbit.mp4"]["predicted_viewpoint_category"] == "High"
    for row in rows:
        assert row["processing_status"] == "success"
        assert float(row["total_processing_cpu_seconds"]) > 0
        assert row["source_unchanged_after_processing"] == "True"
        diagnostics = json.loads(row["diagnostic_images_json"])
        assert len(diagnostics) == 2
        for diagnostic in diagnostics:
            image = cv2.imread(diagnostic["path"])
            assert image is not None
            assert image.shape[0] > 100 and image.shape[1] > 200
            assert diagnostic["rendered_match_count"] >= 0
    assert (output_dir / "REPORT.md").exists()
    assert "do not establish real-world accuracy" in (
        output_dir / "REPORT.md"
    ).read_text(encoding="utf-8")
    for target, before in selected.values():
        assert _sha256(target) == before


def test_misleading_risks_cover_rotation_background_and_intrinsics(analysis_reports):
    pure_rotation = misleading_risk_flags(
        analysis_reports["pure_rotation"]["viewpoint_diversity"]
    )
    assert "rotation_or_planar_motion" in pure_rotation
    assert "background_feature_dominance_unresolved" in pure_rotation
    assert "object_motion_unresolved" in pure_rotation
    assert "intrinsics_unavailable_for_pose_disambiguation" in pure_rotation

    with_intrinsics = copy.deepcopy(
        analysis_reports["pure_rotation"]["viewpoint_diversity"]
    )
    with_intrinsics["method"]["intrinsics_used"] = True
    calibrated_risks = misleading_risk_flags(with_intrinsics)
    assert "intrinsics_accuracy_unverified" in calibrated_risks
    assert "intrinsics_unavailable_for_pose_disambiguation" not in calibrated_risks

    animated = misleading_risk_flags(
        analysis_reports["animated_object"]["viewpoint_diversity"]
    )
    assert "object_animation_evidence" in animated

