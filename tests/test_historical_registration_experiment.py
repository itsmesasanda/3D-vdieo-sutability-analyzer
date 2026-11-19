import csv
import json
from pathlib import Path

from video_suitability.ml.registration_features import REGISTRATION_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evaluation/ml/historical_registration_experiment/historical_opencv8_v2"


def test_historical_experiment_is_offline_only_and_complete():
    manifest = json.loads(
        (ARTIFACT / "experimental_artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_rows"] == 647
    assert manifest["source_groups"] == 49
    assert manifest["selected_model"] == "ridge"
    assert manifest["experimental"] is True
    assert manifest["offline_only"] is True
    assert manifest["activation_allowed"] is False
    assert manifest["production_feature_compatible"] is False
    assert not (ARTIFACT / "artifact_manifest.json").exists()
    assert (ARTIFACT / "experimental_model.joblib").is_file()


def test_all_four_models_share_grouped_cv_and_production_features_differ():
    with (ARTIFACT / "grouped_cv_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model"] for row in rows} == {
        "dummy_mean",
        "ridge",
        "random_forest",
        "gradient_boosting",
    }
    assert min(rows, key=lambda row: float(row["mae"]))["model"] == "ridge"

    splits = json.loads((ARTIFACT / "grouped_cv_splits.json").read_text())
    assert len(splits["splits"]) == 5
    for fold in splits["splits"]:
        assert not set(fold["train_source_groups"]) & set(
            fold["validation_source_groups"]
        )

    definition = json.loads(
        (ARTIFACT / "historical_feature_definition.json").read_text()
    )
    assert definition["production_extractor_compatible"] is False
    assert tuple(definition["feature_names_in_order"]) != REGISTRATION_FEATURE_NAMES


def test_training_records_exclude_failures_and_reserved_categories():
    heldout_categories = {"bicycle", "bowl", "cake", "plant", "toyplane"}
    with (ARTIFACT / "training_records.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 647
    assert len({row["tag"] for row in rows}) == 647
    assert len({row["source_group"] for row in rows}) == 49
    assert {row["status"] for row in rows} == {"sparse_model_created"}
    assert not {row["category"] for row in rows} & heldout_categories
