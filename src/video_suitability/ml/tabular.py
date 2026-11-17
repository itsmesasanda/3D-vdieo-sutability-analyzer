"""Random-Forest baseline on versioned video-level numerical CV features."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .artifacts import create_artifact_directory, write_manifest
from .metrics import compare_with_heuristic, evaluate_predictions
from .schema import DatasetRecord
from .splits import SplitManifest, assert_no_group_leakage


MODEL_TYPE = "random_forest_numeric_cv"
MODEL_VERSION = "1.0"


def feature_matrix(
    records: Sequence[DatasetRecord], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [
                np.nan
                if record.numeric_features.get(name) is None
                else float(record.numeric_features[name])
                for name in feature_names
            ]
            for record in records
        ],
        dtype=np.float64,
    )


def _records_for_split(
    records: Sequence[DatasetRecord], manifest: SplitManifest, name: str
) -> List[DatasetRecord]:
    return [record for record in records if manifest.split_for(record.sequence_id) == name]


def _target_vector(records: Sequence[DatasetRecord]) -> np.ndarray:
    return np.asarray([float(record.target.target_score) for record in records], dtype=float)


def train_random_forest(
    records: Sequence[DatasetRecord],
    split_manifest: SplitManifest,
    artifact_root: Path,
    *,
    allow_synthetic_targets: bool = False,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    random_state: int = 20261117,
    classification_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Train and export a label-gated Random Forest.

    Synthetic targets require an explicit flag and produce an artifact permanently
    marked as smoke-test-only. Heuristic targets are rejected by schema validation.
    """

    try:
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError("Install the ML tabular extra (scikit-learn and joblib)") from exc
    if n_estimators <= 0 or min_samples_leaf <= 0:
        raise ValueError("Random Forest size parameters must be positive")
    for record in records:
        record.validate(
            require_target=True, allow_synthetic_target=allow_synthetic_targets
        )
    assert_no_group_leakage(records, split_manifest)
    synthetic_flags = {record.target.provenance.is_synthetic for record in records}
    if len(synthetic_flags) != 1:
        raise ValueError("Do not mix synthetic and genuine targets in one training run")
    is_synthetic = synthetic_flags == {True}
    if is_synthetic and not allow_synthetic_targets:
        raise ValueError("Synthetic-target training was not explicitly enabled")

    train = _records_for_split(records, split_manifest, "train")
    validation = _records_for_split(records, split_manifest, "validation")
    test = _records_for_split(records, split_manifest, "test")
    if not train or not validation or not test:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    candidate_features = sorted(
        {name for record in train for name in record.numeric_features}
    )
    if not candidate_features:
        raise ValueError("No numerical features are available")
    raw_train = feature_matrix(train, candidate_features)
    usable = ~np.all(np.isnan(raw_train), axis=0)
    feature_names = [
        name for name, keep in zip(candidate_features, usable.tolist()) if keep
    ]
    excluded_all_missing = [
        name for name, keep in zip(candidate_features, usable.tolist()) if not keep
    ]
    if not feature_names:
        raise ValueError("All training features are missing")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(feature_matrix(train, feature_names), _target_vector(train))

    metrics: Dict[str, Any] = {}
    predictions: Dict[str, List[float]] = {}
    train_mean = float(np.mean(_target_vector(train)))
    for name, split_records in [
        ("validation", validation),
        ("test", test),
    ]:
        targets = _target_vector(split_records)
        predicted = np.clip(
            pipeline.predict(feature_matrix(split_records, feature_names)), 0.0, 100.0
        )
        predictions[name] = predicted.tolist()
        binary_truth = (
            [bool(record.target.binary_suitable) for record in split_records]
            if all(record.target.binary_suitable is not None for record in split_records)
            else None
        )
        metrics[name] = {
            "random_forest": compare_with_heuristic(
                targets,
                predicted,
                [record.heuristic_v21_score for record in split_records],
                classification_threshold=classification_threshold,
                binary_truth=binary_truth,
            ),
            "train_mean_baseline": evaluate_predictions(
                targets,
                np.full_like(targets, train_mean),
                classification_threshold=classification_threshold,
                binary_truth=binary_truth,
            ),
            "sequence_ids": [record.sequence_id for record in split_records],
        }

    artifact_dir = create_artifact_directory(artifact_root, MODEL_TYPE, MODEL_VERSION)
    joblib.dump(pipeline, artifact_dir / "model.joblib")
    preprocessing = {
        "feature_schema": records[0].feature_schema_version,
        "feature_names_in_order": feature_names,
        "excluded_all_missing_train_features": excluded_all_missing,
        "missing_value_strategy": "median fitted on train split only",
        "prediction_clip": [0.0, 100.0],
    }
    (artifact_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    split_manifest.write(artifact_dir / "split_manifest.json")
    training_scope = "synthetic_smoke_test_only" if is_synthetic else "genuine_reconstruction_labels"
    write_manifest(
        artifact_dir,
        model_type=MODEL_TYPE,
        model_version=MODEL_VERSION,
        training_scope=training_scope,
        files=["model.joblib", "preprocessing.json", "metrics.json", "split_manifest.json"],
        metadata={
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "record_count": len(records),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "target_provenance_kinds": sorted(
                {record.target.provenance.kind for record in records}
            ),
            "classification_threshold": classification_threshold,
            "synthetic_metrics_are_not_real_accuracy": bool(is_synthetic),
        },
    )
    return {
        "artifact_dir": str(artifact_dir),
        "training_scope": training_scope,
        "metrics": metrics,
        "predictions": predictions,
        "feature_names": feature_names,
    }

