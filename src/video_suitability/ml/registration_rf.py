"""Parity-gated Random Forest inference for COLMAP registration percentage."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import load_and_validate_manifest, sha256_file
from .registration_features import (
    FEATURE_EXTRACTOR_SCHEMA_VERSION,
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features,
    extract_registration_features_from_frame_manifest,
    feature_vector,
    load_feature_spec,
)


MODEL_TYPE = "random_forest_colmap_registration_percentage"
TARGET_NAME = "colmap_registered_image_percentage"
ALLOWED_TRAINING_SCOPES = {"genuine_colmap_registration_labels"}
DEFAULT_ARTIFACT_DIRECTORY = Path("models/registration_rf_current")
ARTIFACT_ENVIRONMENT_VARIABLE = "VIDEO_SUITABILITY_RF_ARTIFACT_DIR"


class RegistrationArtifactIncompatible(ValueError):
    """Raised when an artifact cannot prove preprocessing compatibility."""


class RegistrationFeatureParityError(RegistrationArtifactIncompatible):
    """Raised when saved training examples disagree with local extraction."""


def configured_artifact_directory() -> Optional[Path]:
    configured = os.environ.get(ARTIFACT_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_ARTIFACT_DIRECTORY


def artifact_cache_identity(artifact_dir: Optional[Path] = None) -> Tuple[Any, ...]:
    """Return an identity that changes when a deployed manifest is replaced."""

    directory = (
        Path(artifact_dir)
        if artifact_dir is not None
        else configured_artifact_directory()
    )
    if directory is None:
        return ("unconfigured",)
    manifest = directory / "artifact_manifest.json"
    try:
        stat = manifest.stat()
        digest = sha256_file(manifest)
    except OSError:
        return (str(directory.expanduser().resolve()), "missing")
    return (
        str(directory.expanduser().resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        digest,
    )


def _safe_artifact_relative_path(directory: Path, value: str) -> Path:
    candidate = (directory / value).resolve()
    root = directory.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RegistrationArtifactIncompatible(
            "Parity video path escapes the artifact directory"
        ) from exc
    return candidate


def _validate_artifact_contract(
    directory: Path, manifest: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if manifest.get("model_type") != MODEL_TYPE:
        raise RegistrationArtifactIncompatible(
            f"Expected model_type={MODEL_TYPE!r}"
        )
    if manifest.get("training_scope") not in ALLOWED_TRAINING_SCOPES:
        raise RegistrationArtifactIncompatible(
            "Only genuinely labeled COLMAP-registration artifacts are accepted"
        )
    metadata = manifest.get("metadata", {})
    if metadata.get("target_name") != TARGET_NAME:
        raise RegistrationArtifactIncompatible(
            "Artifact target is not COLMAP registered-image percentage"
        )
    if metadata.get("target_units") != "percent_0_to_100":
        raise RegistrationArtifactIncompatible("Artifact target units are incompatible")
    if int(metadata.get("feature_count", -1)) != 8:
        raise RegistrationArtifactIncompatible("Artifact does not declare eight features")
    hashes = manifest.get("files_sha256", {})
    required = {
        "model.joblib",
        "feature_extractor.json",
        "parity_examples.json",
    }
    missing_hashes = required - set(hashes)
    if missing_hashes:
        raise RegistrationArtifactIncompatible(
            "Required artifact files are not checksum-protected: "
            + ", ".join(sorted(missing_hashes))
        )
    try:
        spec = load_feature_spec(directory / "feature_extractor.json")
        parity = json.loads(
            (directory / "parity_examples.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RegistrationArtifactIncompatible(
            f"Preprocessing/parity metadata is invalid: {exc}"
        ) from exc
    if parity.get("schema_version") != "registration_feature_parity/1.0":
        raise RegistrationArtifactIncompatible("Unsupported parity-example schema")
    cases = parity.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RegistrationArtifactIncompatible(
            "At least one saved training parity example is required"
        )
    return spec, parity


def verify_training_feature_parity(
    artifact_dir: Path,
    spec: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-extract saved training examples and require numerical agreement."""

    directory = Path(artifact_dir)
    atol = float(parity.get("absolute_tolerance", 1e-6))
    rtol = float(parity.get("relative_tolerance", 1e-6))
    if atol < 0 or rtol < 0:
        raise RegistrationArtifactIncompatible("Parity tolerances cannot be negative")
    checked = []
    maximum_absolute_error = 0.0
    for case in parity["cases"]:
        if not isinstance(case, dict):
            raise RegistrationArtifactIncompatible("Invalid parity case")
        case_id = str(case.get("case_id", "unnamed"))
        expected_mapping = case.get("expected_features")
        if not isinstance(expected_mapping, dict) or tuple(expected_mapping) != REGISTRATION_FEATURE_NAMES:
            raise RegistrationArtifactIncompatible(
                f"Parity case {case_id} does not preserve exact feature order"
            )
        try:
            input_type = case.get("input_type", "video")
            if input_type == "video":
                relative_path = case.get("video")
                if not isinstance(relative_path, str) or not relative_path:
                    raise RegistrationArtifactIncompatible(
                        f"Parity case {case_id} has no video path"
                    )
                video_path = _safe_artifact_relative_path(directory, relative_path)
                if not video_path.is_file():
                    raise RegistrationArtifactIncompatible(
                        f"Parity video is missing for case {case_id}: {relative_path}"
                    )
                expected_hash = case.get("video_sha256")
                if (
                    not isinstance(expected_hash, str)
                    or sha256_file(video_path) != expected_hash
                ):
                    raise RegistrationArtifactIncompatible(
                        f"Parity video checksum mismatch for case {case_id}"
                    )
                actual_mapping, _ = extract_registration_features(video_path, spec)
            elif input_type == "colmap_frame_manifest":
                relative_manifest = case.get("frames_manifest")
                if not isinstance(relative_manifest, str) or not relative_manifest:
                    raise RegistrationArtifactIncompatible(
                        f"Parity case {case_id} has no frame-manifest path"
                    )
                manifest_path = _safe_artifact_relative_path(
                    directory, relative_manifest
                )
                expected_hash = case.get("frames_manifest_sha256")
                if (
                    not manifest_path.is_file()
                    or not isinstance(expected_hash, str)
                    or sha256_file(manifest_path) != expected_hash
                ):
                    raise RegistrationArtifactIncompatible(
                        f"Parity frame-manifest checksum mismatch for case {case_id}"
                    )
                relative_images = case.get("images_directory")
                if not isinstance(relative_images, str) or not relative_images:
                    raise RegistrationArtifactIncompatible(
                        f"Parity case {case_id} has no images directory"
                    )
                images_path = _safe_artifact_relative_path(
                    directory, relative_images
                )
                actual_mapping, _ = (
                    extract_registration_features_from_frame_manifest(
                        manifest_path,
                        spec,
                        images_dir=images_path,
                        expected_input_frame_count=int(
                            case.get("expected_input_frame_count", 60)
                        ),
                    )
                )
            else:
                raise RegistrationArtifactIncompatible(
                    f"Unsupported parity input_type for case {case_id}: {input_type}"
                )
        except Exception as exc:
            if isinstance(exc, RegistrationArtifactIncompatible):
                raise
            raise RegistrationFeatureParityError(
                f"Could not reproduce training parity case {case_id}: {exc}"
            ) from exc
        expected = feature_vector(expected_mapping)
        actual = feature_vector(actual_mapping)
        absolute_error = np.abs(actual - expected)
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(absolute_error))
        )
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            failing = [
                {
                    "feature": name,
                    "expected": float(expected[index]),
                    "actual": float(actual[index]),
                    "absolute_error": float(absolute_error[index]),
                }
                for index, name in enumerate(REGISTRATION_FEATURE_NAMES)
                if not np.isclose(actual[index], expected[index], atol=atol, rtol=rtol)
            ]
            raise RegistrationFeatureParityError(
                f"Training feature parity failed for {case_id}: {failing}"
            )
        checked.append(case_id)
    return {
        "status": "verified",
        "case_count": len(checked),
        "case_ids": checked,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "maximum_absolute_error": round(maximum_absolute_error, 12),
        "feature_schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
    }


def load_registration_model(path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Install the ml-tabular extra to load the Random Forest artifact"
        ) from exc
    return joblib.load(path)


def _base_unavailable(reason: str, *, status: str = "unavailable") -> Dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "trained_model_used": False,
        "target_name": TARGET_NAME,
        "affects_suitability_score": False,
        "affects_viewpoint_diversity": False,
        "colmap_run": False,
    }


def predict_colmap_registration_percentage(
    video_path: Path,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Predict registration percentage only after manifest and parity validation."""

    total_wall_started = time.perf_counter()
    total_cpu_started = time.process_time()
    directory = (
        Path(artifact_dir)
        if artifact_dir is not None
        else configured_artifact_directory()
    )
    if directory is None or not directory.is_dir():
        location = str(directory) if directory is not None else "not configured"
        return _base_unavailable(
            "No compatible Random Forest artifact is installed. Expected a validated "
            f"artifact at {location!r} or set {ARTIFACT_ENVIRONMENT_VARIABLE}."
        )
    try:
        try:
            manifest = load_and_validate_manifest(directory)
        except (OSError, ValueError) as exc:
            raise RegistrationArtifactIncompatible(
                f"Artifact manifest/hash validation failed: {exc}"
            ) from exc
        spec, parity = _validate_artifact_contract(directory, manifest)
        parity_wall_started = time.perf_counter()
        parity_cpu_started = time.process_time()
        parity_result = verify_training_feature_parity(directory, spec, parity)
        parity_timing = {
            "wall_seconds": round(time.perf_counter() - parity_wall_started, 6),
            "cpu_seconds": round(time.process_time() - parity_cpu_started, 6),
        }
        model = load_registration_model(directory / "model.joblib")
        feature_wall_started = time.perf_counter()
        feature_cpu_started = time.process_time()
        features, diagnostics = extract_registration_features(Path(video_path), spec)
        feature_timing = {
            "wall_seconds": round(time.perf_counter() - feature_wall_started, 6),
            "cpu_seconds": round(time.process_time() - feature_cpu_started, 6),
        }
        values = feature_vector(features).reshape(1, -1)
        if getattr(model, "n_features_in_", 8) != 8:
            raise RegistrationArtifactIncompatible(
                "Loaded model does not declare eight input features"
            )
        declared_names = getattr(model, "feature_names_in_", None)
        if declared_names is not None and tuple(
            str(name) for name in declared_names
        ) != REGISTRATION_FEATURE_NAMES:
            raise RegistrationArtifactIncompatible(
                "Loaded model feature names/order do not match the eight-feature contract"
            )
        prediction_wall_started = time.perf_counter()
        prediction_cpu_started = time.process_time()
        predicted = np.asarray(model.predict(values), dtype=float).reshape(-1)
        prediction_timing = {
            "wall_seconds": round(time.perf_counter() - prediction_wall_started, 6),
            "cpu_seconds": round(time.process_time() - prediction_cpu_started, 6),
        }
        if predicted.size != 1 or not np.isfinite(predicted[0]):
            raise RuntimeError("Random Forest returned an invalid prediction")
        percentage = float(np.clip(predicted[0], 0.0, 100.0))
        return {
            "status": "available",
            "trained_model_used": True,
            "model_type": manifest["model_type"],
            "model_version": manifest["model_version"],
            "training_scope": manifest["training_scope"],
            "target_name": TARGET_NAME,
            "predicted_registration_percentage": round(percentage, 2),
            "interpretation": (
                "Estimated percentage of sampled COLMAP input images expected to register; "
                "not a reconstruction-success probability or object-completeness score."
            ),
            "feature_parity": parity_result,
            "feature_names_in_order": list(REGISTRATION_FEATURE_NAMES),
            "features": {name: round(float(value), 9) for name, value in features.items()},
            "feature_diagnostics": diagnostics,
            "affects_suitability_score": False,
            "affects_viewpoint_diversity": False,
            "colmap_run": False,
            "performance_profile": {
                "training_example_parity": parity_timing,
                "video_feature_extraction": feature_timing,
                "model_prediction": prediction_timing,
                "total_wall_seconds": round(
                    time.perf_counter() - total_wall_started, 6
                ),
                "total_cpu_seconds": round(
                    time.process_time() - total_cpu_started, 6
                ),
            },
        }
    except RegistrationArtifactIncompatible as exc:
        return _base_unavailable(
            f"Random Forest artifact is incompatible: {exc}", status="incompatible"
        )
    except Exception as exc:
        return _base_unavailable(
            f"Random Forest prediction is unavailable: {type(exc).__name__}: {exc}"
        )


def attach_registration_prediction(
    report: Mapping[str, Any],
    video_path: Path,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output = dict(report)
    output["registration_prediction"] = predict_colmap_registration_percentage(
        video_path, artifact_dir
    )
    return output
