import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from video_suitability.ml.r1_regeneration import (
    DEFAULT_R1_PILOT_TAGS,
    R1_VARIANTS,
    R1RegenerationError,
    r1_frame_indices,
    regenerate_r1_experiment,
    select_r1_label_records,
    verify_executed_r1_notebook,
)
from video_suitability.ml.registration_features import (
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features_from_frame_manifest,
    load_feature_spec,
    verify_registration_frame_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_AUDIT = ROOT / "evaluation/ml/colmap_csv_archive_real_audit/record_audit.csv"
FEATURE_SPEC = (
    ROOT / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
)


def _write_sequence(root: Path, category: str, sequence_id: str, count: int = 120):
    images = root / category / sequence_id / "images"
    images.mkdir(parents=True)
    for index in range(count):
        yy, xx = np.indices((48, 64))
        array = np.stack(
            [
                (xx * 7 + index * 3) % 256,
                (yy * 11 + index * 5) % 256,
                ((xx + yy) * 13 + index * 17) % 256,
            ],
            axis=-1,
        ).astype(np.uint8)
        Image.fromarray(array).save(images / f"frame{index:04d}.jpg")


def test_r1_camera_path_indices_match_executed_formulas():
    window = r1_frame_indices(120, "window_offset_p10_60")
    assert window == list(range(6, 66))

    span = r1_frame_indices(120, "span_p03_p84")
    assert span == np.linspace(round(119 * 0.03), round(119 * 0.84), 60, dtype=int).tolist()
    assert len(span) == len(set(span)) == 60
    assert span == sorted(span)
    all_variants = [tuple(r1_frame_indices(120, variant)) for variant in R1_VARIANTS]
    assert len(all_variants) == len(set(all_variants)) == 16

    try:
        r1_frame_indices(100, "span_p03_p84")
    except R1RegenerationError as exc:
        assert "Need >=110" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Short span sequence was accepted")


def test_three_selected_rows_are_real_numeric_r1_sparse_results():
    records = select_r1_label_records(REAL_AUDIT, tags=DEFAULT_R1_PILOT_TAGS)
    assert [row["tag"] for row in records] == list(DEFAULT_R1_PILOT_TAGS)
    assert len({row["source_group"] for row in records}) == 3
    assert [row["registered_images"] for row in records] == [60, 29, 2]
    assert np.allclose(
        [row["target_registration_0_100"] for row in records],
        [100.0, 29 * 100.0 / 60.0, 2 * 100.0 / 60.0],
    )
    expansion = select_r1_label_records(REAL_AUDIT, all_sparse_records=True)
    assert len(expansion) == 72
    assert len({row["source_group"] for row in expansion}) == 5


def test_r1_regeneration_produces_verified_manifest_and_production_features(tmp_path):
    co3d_root = tmp_path / "co3d"
    _write_sequence(co3d_root, "apple", "110_13051_23361")
    record = {
        "tag": "apple_110_13051_23361__span_p03_p84",
        "source_group": "apple_110_13051_23361",
        "category": "apple",
        "variant": "span_p03_p84",
    }
    output = tmp_path / "regenerated"
    result = regenerate_r1_experiment(
        record=record,
        co3d_root=co3d_root,
        output_dir=output,
        executed_notebook_evidence={"implementation_verified": True},
    )

    rows, image_root = verify_registration_frame_manifest(
        result["frames_manifest"],
        images_dir=result["images_directory"],
        expected_input_frame_count=60,
    )
    assert len(rows) == 60
    assert image_root == result["images_directory"].resolve()
    assert [int(row["actual_source_frame_index"]) for row in rows] == r1_frame_indices(
        120, "span_p03_p84"
    )
    provenance = json.loads(
        Path(result["regeneration_provenance"]).read_text(encoding="utf-8")
    )
    assert provenance["original_hashes_verified"] is False
    assert provenance["fidelity_tier"] == "executed_contract_reproduced"
    assert provenance["transformation"]["max_edge"] == 800

    features, diagnostics = extract_registration_features_from_frame_manifest(
        result["frames_manifest"],
        load_feature_spec(FEATURE_SPEC),
        images_dir=result["images_directory"],
        expected_input_frame_count=60,
    )
    assert tuple(features) == REGISTRATION_FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()
    assert diagnostics["sample_count"] == 16
    assert diagnostics["pair_count"] == 15


def test_executed_notebook_without_execution_evidence_is_rejected(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "source": ["FRAME_COUNT=60\nVARIANTS=(\n"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "source": ["def _indices_for_view_variant(): pass\ndef make_variant(): pass\n"],
            },
        ]
    }
    path = tmp_path / "unexecuted.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    try:
        verify_executed_r1_notebook(path)
    except R1RegenerationError as exc:
        assert "differs from the inspected code" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Unverified notebook was accepted")
