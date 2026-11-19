#!/usr/bin/env python3
"""Train an offline registration experiment from paired historical feature CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "brightness_mean",
    "brightness_variation",
    "contrast_mean",
    "sharpness_mean",
    "sharpness_variation",
    "dark_fraction",
    "orb_features_mean",
    "orb_matches_mean",
]
HISTORICAL_EXTRACTOR_SOURCE_SHA256 = (
    "ac727c93fc14ffec7dd8079fb53f876f8a8b71560237ff9e319692d4288a2be0"
)
HELDOUT_CATEGORIES = {"bicycle", "bowl", "cake", "plant", "toyplane"}
RANDOM_SEED = 20261119

# Each chosen file has the same tag set as its selected batch in record_audit.csv.
# This avoids mixing cumulative exports or resolving conflicting duplicate features.
BATCH_FEATURE_FILES = {
    "A": "new_features_50 (1).csv",
    "B": "new_features_50 (2).csv",
    "C": "new_features_50-2.csv",
    "D": "new_features_50 (8).csv",
    "E": "new_features_50 (7).csv",
    "F": "new_features_50 (6).csv",
    "R1": "new_features_50 (5).csv",
    "R2": "new_features_50.csv",
    "R3": "new_features_50-4.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_factories() -> Dict[str, Any]:
    return {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="squared_error",
            random_state=RANDOM_SEED,
        ),
    }


def _normalize_feature_file(path: Path, batch_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"tag", "source_group", "target", *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if frame["tag"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate experiment tags")
    result = frame[["tag", "source_group", "target", *FEATURE_COLUMNS]].copy()
    result["batch_id"] = batch_id
    result["feature_source_file"] = path.name
    result["feature_source_sha256"] = sha256_file(path)
    return result


def build_training_table(
    *,
    record_audit: Path,
    archive_directory: Path,
    original_features: Path,
    original_labels: Path,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    audit = pd.read_csv(record_audit, dtype=str, keep_default_na=False)
    if audit["tag"].duplicated().any():
        raise ValueError("record_audit.csv must contain unique experiment tags")

    sources: List[Dict[str, Any]] = []
    frames = []
    original = _normalize_feature_file(original_features, "Original")
    original_label_frame = pd.read_csv(original_labels, dtype=str, keep_default_na=False)
    if set(original["tag"]) != set(original_label_frame["tag"]):
        raise ValueError("Original feature and 60-frame label tag sets differ")
    frames.append(original)
    sources.append(
        {
            "batch_id": "Original",
            "feature_file": str(original_features.resolve()),
            "feature_sha256": sha256_file(original_features),
            "label_file": str(original_labels.resolve()),
            "label_sha256": sha256_file(original_labels),
            "row_count": len(original),
        }
    )

    for batch_id, filename in BATCH_FEATURE_FILES.items():
        feature_path = archive_directory / filename
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        batch = _normalize_feature_file(feature_path, batch_id)
        expected = audit.loc[audit["batch_id"] == batch_id]
        if len(expected) != len(batch) or set(expected["tag"]) != set(batch["tag"]):
            raise ValueError(f"{feature_path.name} does not match audit batch {batch_id}")
        selected_exports = expected["selected_export"].unique().tolist()
        if len(selected_exports) != 1:
            raise ValueError(f"Batch {batch_id} has ambiguous label exports")
        label_path = archive_directory / selected_exports[0]
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        raw_labels = pd.read_csv(label_path, dtype=str, keep_default_na=False)
        if not set(batch["tag"]) <= set(raw_labels["tag"]):
            raise ValueError(f"{feature_path.name} tags are absent from {label_path.name}")
        frames.append(batch)
        sources.append(
            {
                "batch_id": batch_id,
                "feature_file": str(feature_path.resolve()),
                "feature_sha256": sha256_file(feature_path),
                "label_file": str(label_path.resolve()),
                "label_sha256": sha256_file(label_path),
                "row_count": len(batch),
            }
        )

    historical = pd.concat(frames, ignore_index=True)
    if len(historical) != 675 or historical["tag"].nunique() != 675:
        raise ValueError(
            f"Expected 675 uniquely paired historical feature rows, got {len(historical)} "
            f"rows and {historical['tag'].nunique()} tags"
        )
    joined = audit.merge(
        historical,
        on=["tag", "batch_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_label", "_feature"),
        indicator=True,
    )
    exclusions: Dict[str, int] = {}
    eligible = pd.Series(True, index=joined.index)

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal eligible
        applied = eligible & mask
        exclusions[reason] = int(applied.sum())
        eligible &= ~mask

    exclude(joined["_merge"] != "both", "missing_or_ambiguous_feature_join")
    exclude(
        joined["classification"] != "recoverable_success_missing_evidence",
        "mapper_failure_interruption_or_invalid_label",
    )
    exclude(joined["status"] != "sparse_model_created", "not_sparse_model_created")
    exclude(joined["target_registration_0_100"] == "", "missing_numeric_target")
    exclude(joined["category"].isin(HELDOUT_CATEGORIES), "reserved_heldout_category")
    exclude(
        joined["source_group_label"].astype(str)
        != joined["source_group_feature"].astype(str),
        "source_group_mismatch",
    )
    feature_target = pd.to_numeric(joined["target"], errors="coerce")
    label_target = pd.to_numeric(joined["target_registration_0_100"], errors="coerce")
    exclude(
        ~(np.isfinite(feature_target) & np.isfinite(label_target)),
        "non_numeric_feature_or_label_target",
    )
    exclude(
        pd.Series(
            ~np.isclose(feature_target, label_target, atol=0.005001, rtol=0.0),
            index=joined.index,
        ),
        "feature_label_target_mismatch",
    )
    feature_matrix = joined[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    exclude(
        pd.Series(~np.isfinite(feature_matrix.to_numpy(dtype=float)).all(axis=1), index=joined.index),
        "non_finite_feature",
    )

    training = joined.loc[eligible].copy()
    training[FEATURE_COLUMNS] = feature_matrix.loc[eligible]
    training["target_registration_0_100"] = label_target.loc[eligible]
    training["source_group"] = training["source_group_label"].astype(str)
    keep = [
        "tag",
        "batch_id",
        "selected_export",
        "feature_source_file",
        "feature_source_sha256",
        "source_group",
        "category",
        "variant",
        "status",
        "sampled_images",
        "registered_images",
        "target_registration_0_100",
        *FEATURE_COLUMNS,
    ]
    training = training[keep].sort_values("tag").reset_index(drop=True)
    if training["tag"].duplicated().any():
        raise RuntimeError("Accepted training table contains duplicate experiments")
    if set(training["category"]) & HELDOUT_CATEGORIES:
        raise RuntimeError("Reserved held-out category entered training")
    if len(training) != 647:
        raise RuntimeError(f"Expected 647 valid sparse-model rows, accepted {len(training)}")

    join_report = {
        "input_unique_experiments": len(joined),
        "accepted_training_rows": len(training),
        "independent_source_groups": int(training["source_group"].nunique()),
        "exclusion_counts_sequential": exclusions,
        "heldout_categories_excluded": sorted(HELDOUT_CATEGORIES),
        "exact_heldout_source_ids_available": False,
        "independent_holdout_evaluation_allowed": False,
    }
    return training, sources, join_report


def train_and_evaluate(training: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    X = training[FEATURE_COLUMNS]
    y = training["target_registration_0_100"].to_numpy(dtype=float)
    groups = training["source_group"].to_numpy(dtype=str)
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(X, y, groups))
    split_document = []
    for fold, (train_indices, validation_indices) in enumerate(splits):
        train_groups = sorted(set(groups[train_indices]))
        validation_groups = sorted(set(groups[validation_indices]))
        if set(train_groups) & set(validation_groups):
            raise RuntimeError("Source-group leakage in GroupKFold")
        split_document.append(
            {
                "fold": fold,
                "train_source_groups": train_groups,
                "validation_source_groups": validation_groups,
                "train_record_ids": training.iloc[train_indices]["tag"].tolist(),
                "validation_record_ids": training.iloc[validation_indices]["tag"].tolist(),
            }
        )
    _json_write(
        output_dir / "grouped_cv_splits.json",
        {
            "schema_version": "historical_registration_groupkfold/1.0",
            "group_key": "physical_co3d_source_group",
            "fold_count": 5,
            "splits": split_document,
        },
    )

    predictions: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    factories = _model_factories()
    for model_name, factory in factories.items():
        for fold, (train_indices, validation_indices) in enumerate(splits):
            model = factory()
            model.fit(X.iloc[train_indices], y[train_indices])
            raw_predicted = np.asarray(
                model.predict(X.iloc[validation_indices]), dtype=float
            )
            predicted = np.clip(raw_predicted, 0.0, 100.0)
            actual = y[validation_indices]
            error = predicted - actual
            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                }
            )
            for offset, row_index in enumerate(validation_indices):
                source = training.iloc[row_index]
                predictions.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "tag": source["tag"],
                        "source_group": source["source_group"],
                        "batch_id": source["batch_id"],
                        "variant": source["variant"],
                        "actual": float(actual[offset]),
                        "predicted": float(predicted[offset]),
                        "raw_predicted": float(raw_predicted[offset]),
                        "error": float(error[offset]),
                        "absolute_error": float(abs(error[offset])),
                    }
                )

    prediction_frame = pd.DataFrame(predictions)
    fold_frame = pd.DataFrame(fold_metrics)
    summaries = []
    for model_name, rows in prediction_frame.groupby("model"):
        error = rows["predicted"] - rows["actual"]
        summaries.append(
            {
                "model": model_name,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "prediction_min": float(rows["predicted"].min()),
                "prediction_mean": float(rows["predicted"].mean()),
                "prediction_max": float(rows["predicted"].max()),
                "raw_out_of_range_count": int(
                    ((rows["raw_predicted"] < 0) | (rows["raw_predicted"] > 100)).sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["mae", "rmse", "model"])
    selected_name = str(summary.iloc[0]["model"])
    prediction_frame.to_csv(output_dir / "grouped_cv_predictions.csv", index=False)
    fold_frame.to_csv(output_dir / "grouped_cv_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "grouped_cv_summary.csv", index=False)

    selected_model = factories[selected_name]()
    selected_model.fit(X, y)
    joblib.dump(selected_model, output_dir / "experimental_model.joblib")
    reloaded = joblib.load(output_dir / "experimental_model.joblib")
    parity_indices = training.groupby("source_group", sort=True).head(1).index[:3]
    parity = training.loc[parity_indices, ["tag", "source_group", *FEATURE_COLUMNS]].copy()
    expected = np.clip(
        np.asarray(reloaded.predict(parity[FEATURE_COLUMNS]), dtype=float), 0.0, 100.0
    )
    parity["expected_prediction_0_100"] = expected
    parity.to_csv(output_dir / "experimental_prediction_parity.csv", index=False)
    selection = {
        "selected_model": selected_name,
        "selection_source": "five_fold_physical_source_group_cv_only",
        "primary_metric": "oof_mae",
        "tie_breakers": ["oof_rmse", "model_name"],
        "prediction_postprocessing": "clip_0_100_before_metrics",
        "models": summary.to_dict(orient="records"),
        "independent_holdout_evaluated": False,
    }
    _json_write(output_dir / "model_selection.json", selection)
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-audit", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--original-features", type=Path, required=True)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {output}")
    output.mkdir(parents=True)

    training, sources, join_report = build_training_table(
        record_audit=args.record_audit,
        archive_directory=args.archive_dir,
        original_features=args.original_features,
        original_labels=args.original_labels,
    )
    training.to_csv(output / "training_records.csv", index=False)
    _json_write(output / "input_sources.json", sources)
    _json_write(output / "join_report.json", join_report)
    feature_definition = {
        "schema_version": "historical_registration_opencv8/1.0",
        "implementation_source_sha256": HISTORICAL_EXTRACTOR_SOURCE_SHA256,
        "feature_names_in_order": FEATURE_COLUMNS,
        "frame_input": "60 prepared JPEG images sorted by filename",
        "frame_sampling": "np.linspace(0,59,12,dtype=int)",
        "resize": "if max edge >480, proportional cv2.resize with default interpolation",
        "grayscale": "cv2.COLOR_BGR2GRAY",
        "orb": "cv2.ORB_create(nfeatures=400), remaining parameters/OpenCV runtime defaults",
        "matching": "BFMatcher(NORM_HAMMING).knnMatch(k=2), Lowe ratio 0.75",
        "features": {
            "brightness_mean": "mean of 12 per-frame gray means",
            "brightness_variation": "std of 12 per-frame gray means",
            "contrast_mean": "mean of 12 per-frame gray standard deviations",
            "sharpness_mean": "mean log1p(var(Laplacian(gray,CV_64F)))",
            "sharpness_variation": "std log1p(var(Laplacian(gray,CV_64F)))",
            "dark_fraction": "mean per-frame fraction(gray < 32)",
            "orb_features_mean": "mean ORB keypoint count",
            "orb_matches_mean": "mean consecutive Lowe-ratio good-match count",
        },
        "production_extractor_compatible": False,
        "production_mismatches": [
            "12 sampled prepared frames versus production 16",
            "max-edge-480 default resize versus fixed 320x240 INTER_AREA",
            "ORB nfeatures=400/defaults versus production nfeatures=1000/explicit parameters and RNG seed",
            "historical brightness/sharpness variation and dark fraction are absent in production",
            "production frame difference, match ratio, and homography-RANSAC inlier ratio are absent historically",
        ],
    }
    _json_write(output / "historical_feature_definition.json", feature_definition)
    selection = train_and_evaluate(training, output)

    files = [path for path in sorted(output.iterdir()) if path.is_file()]
    manifest = {
        "schema_version": "historical_registration_experimental_artifact/1.0",
        "experimental": True,
        "offline_only": True,
        "activation_allowed": False,
        "production_feature_compatible": False,
        "reason": "Historical eight-feature preprocessing differs from the deployed extractor",
        "training_rows": len(training),
        "source_groups": int(training["source_group"].nunique()),
        "selected_model": selection["selected_model"],
        "prediction_postprocessing": "clip_0_100",
        "independent_holdout_evaluated": False,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "files_sha256": {path.name: sha256_file(path) for path in files},
    }
    _json_write(output / "experimental_artifact_manifest.json", manifest)
    report = {
        **manifest,
        "join": join_report,
        "feature_sources": sources,
        "grouped_cv": selection,
        "safe_to_integrate_live": False,
        "production_artifact_manifest_created": False,
    }
    _json_write(output / "evaluation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
