"""Integrity-gated, opt-in inference for the offline historical Ridge model.

This module intentionally remains separate from the production Random Forest path.
Passing these checks proves artifact integrity and historical feature/model parity;
it does not prove that uploaded videos share the CO3D training-input domain.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .historical_registration_features import (
    HISTORICAL_EXTRACTOR_SOURCE_SHA256,
    HISTORICAL_FEATURE_NAMES,
    LIVE_VIDEO_ADAPTER_POLICY,
    extract_historical_features_from_uploaded_video,
)


MODEL_SHA256 = "46112ef74ffe29896f892baf6dd037200271d14aa3da5f4f13901eac1af988c6"
FEATURE_DEFINITION_SHA256 = (
    "492b28f1d5973012d56190d7ec7a6759cf747f4145e8fc9bb854fba04fa9282e"
)
PARITY_CSV_SHA256 = "7ae35b717ac3060bba6b6158174a6d7947d6c20ef1d1dfdbfaa7a58167b28b4b"
ARTIFACT_SCHEMA = "historical_registration_experimental_artifact/1.0"
MODEL_FILENAME = "experimental_model.joblib"
ARTIFACT_RELATIVE_PATH = Path(
    "evaluation/ml/historical_registration_experiment/historical_opencv8_v2"
)


class ExperimentalRidgeError(RuntimeError):
    """Raised when the experimental artifact or adapter is not safe to use."""


def default_artifact_directory() -> Path:
    return Path(__file__).resolve().parents[3] / ARTIFACT_RELATIVE_PATH


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_verified_bytes(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise ExperimentalRidgeError(f"Required artifact file is missing: {path.name}")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ExperimentalRidgeError(
            f"Integrity check failed for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return payload


def _load_supporting_metadata(directory: Path) -> Tuple[Dict[str, Any], list[Dict[str, str]]]:
    manifest_path = directory / "experimental_artifact_manifest.json"
    if not manifest_path.is_file():
        raise ExperimentalRidgeError("Experimental artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ExperimentalRidgeError("Unexpected experimental artifact schema")
    if manifest.get("selected_model") != "ridge" or manifest.get("experimental") is not True:
        raise ExperimentalRidgeError("Artifact is not the expected experimental Ridge model")
    if manifest.get("training_rows") != 647 or manifest.get("source_groups") != 49:
        raise ExperimentalRidgeError("Artifact training provenance metadata changed")
    declared = manifest.get("files_sha256", {})
    pinned = {
        MODEL_FILENAME: MODEL_SHA256,
        "historical_feature_definition.json": FEATURE_DEFINITION_SHA256,
        "experimental_prediction_parity.csv": PARITY_CSV_SHA256,
    }
    for name, digest in pinned.items():
        if declared.get(name) != digest:
            raise ExperimentalRidgeError(f"Manifest digest changed for {name}")

    definition_payload = _read_verified_bytes(
        directory / "historical_feature_definition.json", FEATURE_DEFINITION_SHA256
    )
    definition = json.loads(definition_payload.decode("utf-8"))
    if tuple(definition.get("feature_names_in_order", ())) != HISTORICAL_FEATURE_NAMES:
        raise ExperimentalRidgeError("Historical feature names/order do not match code")
    if definition.get("implementation_source_sha256") != HISTORICAL_EXTRACTOR_SOURCE_SHA256:
        raise ExperimentalRidgeError("Historical feature implementation hash changed")

    parity_payload = _read_verified_bytes(
        directory / "experimental_prediction_parity.csv", PARITY_CSV_SHA256
    )
    parity_rows = list(csv.DictReader(io.StringIO(parity_payload.decode("utf-8"))))
    if not parity_rows:
        raise ExperimentalRidgeError("Artifact prediction parity examples are empty")
    return manifest, parity_rows


def _model_bytes(directory: Path, supplied_bytes: Optional[bytes]) -> bytes:
    payload = supplied_bytes
    if payload is None:
        model_path = directory / MODEL_FILENAME
        if not model_path.is_file():
            raise ExperimentalRidgeError(
                "The trusted Ridge model is not installed. Configure the hash-verified "
                "Streamlit secret or install the local artifact."
            )
        payload = model_path.read_bytes()
    if _sha256_bytes(payload) != MODEL_SHA256:
        raise ExperimentalRidgeError("Ridge model integrity check failed; refusing deserialization")
    return payload


def _predict_array(model: Any, values: list[list[float]]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.asarray(model.predict(np.asarray(values, dtype=np.float64)), dtype=float)


def load_experimental_ridge(
    artifact_directory: Optional[Path] = None,
    model_bytes: Optional[bytes] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Verify trusted metadata/model bytes and reproduce saved prediction parity."""

    directory = Path(artifact_directory or default_artifact_directory()).resolve()
    manifest, parity_rows = _load_supporting_metadata(directory)
    payload = _model_bytes(directory, model_bytes)
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ExperimentalRidgeError(
            "Experimental Ridge dependencies are not installed"
        ) from exc
    # joblib/pickle is invoked only after the payload matches the pinned trusted hash.
    model = joblib.load(io.BytesIO(payload))
    declared_names = tuple(str(value) for value in getattr(model, "feature_names_in_", ()))
    if declared_names != HISTORICAL_FEATURE_NAMES:
        raise ExperimentalRidgeError("Model feature names/order do not match historical contract")
    if int(getattr(model, "n_features_in_", -1)) != len(HISTORICAL_FEATURE_NAMES):
        raise ExperimentalRidgeError("Model feature count does not match historical contract")
    steps = getattr(model, "named_steps", {})
    if steps.get("model", object()).__class__.__name__ != "Ridge":
        raise ExperimentalRidgeError("Serialized estimator is not Ridge")

    parity_values = [
        [float(row[name]) for name in HISTORICAL_FEATURE_NAMES] for row in parity_rows
    ]
    expected = np.asarray(
        [float(row["expected_prediction_0_100"]) for row in parity_rows], dtype=float
    )
    actual = _predict_array(model, parity_values)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ExperimentalRidgeError("Saved Ridge prediction parity check failed")
    return model, {
        "status": "verified",
        "model_sha256": MODEL_SHA256,
        "feature_definition_sha256": FEATURE_DEFINITION_SHA256,
        "prediction_parity_sha256": PARITY_CSV_SHA256,
        "prediction_parity_case_count": len(parity_rows),
        "training_rows": manifest["training_rows"],
        "source_groups": manifest["source_groups"],
        "grouped_cv_mae_percentage_points": 19.6249,
        "independent_holdout_evaluated": False,
        "historical_manifest_offline_only": bool(manifest.get("offline_only")),
        "historical_manifest_activation_allowed": bool(manifest.get("activation_allowed")),
    }


def predict_experimental_ridge_registration(
    video_path: Path,
    artifact_directory: Optional[Path] = None,
    model_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Return an isolated prediction or an explicit unavailable result."""

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    base: Dict[str, Any] = {
        "status": "unavailable",
        "trained_model_used": False,
        "colmap_run": False,
        "affects_suitability_score": False,
        "affects_viewpoint_diversity": False,
        "experimental": True,
        "independently_validated": False,
        "adapter_policy": LIVE_VIDEO_ADAPTER_POLICY,
    }
    try:
        model, integrity = load_experimental_ridge(
            artifact_directory=artifact_directory, model_bytes=model_bytes
        )
        features, adapter = extract_historical_features_from_uploaded_video(video_path)
        if tuple(features) != HISTORICAL_FEATURE_NAMES:
            raise ExperimentalRidgeError("Live feature order changed")
        ordered = [[float(features[name]) for name in HISTORICAL_FEATURE_NAMES]]
        raw = float(_predict_array(model, ordered)[0])
        if not np.isfinite(raw):
            raise ExperimentalRidgeError("Ridge produced a non-finite prediction")
        clipped = float(np.clip(raw, 0.0, 100.0))
        base.update(
            {
                "status": "available",
                "trained_model_used": True,
                "predicted_registration_percentage": round(clipped, 6),
                "raw_prediction_percentage": round(raw, 6),
                "prediction_was_clipped": bool(clipped != raw),
                "feature_names_in_order": list(HISTORICAL_FEATURE_NAMES),
                "features": {name: float(features[name]) for name in HISTORICAL_FEATURE_NAMES},
                "artifact_integrity": integrity,
                "adapter_diagnostics": adapter,
                "validation_warning": (
                    "This uploaded-video adapter has not been independently validated and "
                    "is not proven equivalent to the historical CO3D training-input generation."
                ),
                "interpretation_warning": (
                    "Predicted COLMAP image registration is not a guarantee of successful "
                    "3D reconstruction or complete object coverage."
                ),
            }
        )
    except Exception as exc:
        base["reason"] = str(exc) or exc.__class__.__name__
    base["performance_profile"] = {
        "wall_seconds": round(time.perf_counter() - wall_started, 6),
        "cpu_seconds": round(time.process_time() - cpu_started, 6),
    }
    return base


def attach_experimental_ridge_prediction(
    report: Mapping[str, Any],
    video_path: Path,
    artifact_directory: Optional[Path] = None,
    model_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    attached = dict(report)
    attached["experimental_ridge_prediction"] = predict_experimental_ridge_registration(
        video_path,
        artifact_directory=artifact_directory,
        model_bytes=model_bytes,
    )
    return attached
