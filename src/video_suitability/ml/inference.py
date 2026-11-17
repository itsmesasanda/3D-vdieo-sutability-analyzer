"""Additive trained-model adapter that preserves heuristic fallback semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .artifacts import load_and_validate_manifest
from .features import numeric_features_from_report


def optional_ml_prediction(
    heuristic_report: Dict[str, Any], artifact_dir: Optional[Path]
) -> Dict[str, Any]:
    """Return a separate ML result; never replace the heuristic overall score."""

    if artifact_dir is None:
        return {
            "status": "unavailable",
            "reason": "No trained model artifact was configured.",
            "trained_model_used": False,
        }
    directory = Path(artifact_dir)
    try:
        manifest = load_and_validate_manifest(directory)
        if manifest["model_type"] != "random_forest_numeric_cv":
            raise ValueError(f"Unsupported inference model type: {manifest['model_type']}")
        if manifest["training_scope"] != "genuine_reconstruction_labels":
            return {
                "status": "unavailable",
                "reason": "Synthetic smoke-test artifacts are never exposed as production predictions.",
                "trained_model_used": False,
                "artifact_training_scope": manifest["training_scope"],
            }
        import joblib

        preprocessing = json.loads(
            (directory / "preprocessing.json").read_text(encoding="utf-8")
        )
        model = joblib.load(directory / "model.joblib")
        features = numeric_features_from_report(heuristic_report)
        names = preprocessing["feature_names_in_order"]
        values = np.asarray(
            [[np.nan if features.get(name) is None else features[name] for name in names]],
            dtype=float,
        )
        score = float(np.clip(model.predict(values)[0], 0.0, 100.0))
        return {
            "status": "available",
            "trained_model_used": True,
            "model_type": manifest["model_type"],
            "model_version": manifest["model_version"],
            "training_scope": manifest["training_scope"],
            "score": round(score, 2),
            "interpretation": "Separate learned reconstruction-quality estimate; not a calibrated probability.",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"Model artifact could not be used: {exc}",
            "trained_model_used": False,
        }


def attach_optional_ml_prediction(
    heuristic_report: Dict[str, Any], artifact_dir: Optional[Path]
) -> Dict[str, Any]:
    """Mutate only an additive field; top-level heuristic score remains authoritative."""

    output = dict(heuristic_report)
    output["optional_ml_prediction"] = optional_ml_prediction(
        heuristic_report, artifact_dir
    )
    return output


def temporal_prediction_from_embeddings(
    embeddings: np.ndarray,
    artifact_dir: Optional[Path],
    *,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a genuine temporal artifact on already cached ResNet-18 embeddings."""

    if artifact_dir is None:
        return {
            "status": "unavailable",
            "reason": "No temporal model artifact was configured.",
            "trained_model_used": False,
        }
    try:
        directory = Path(artifact_dir)
        manifest = load_and_validate_manifest(directory)
        if manifest["model_type"] != "frozen_resnet18_temporal_lstm":
            raise ValueError(f"Unsupported temporal model type: {manifest['model_type']}")
        if manifest["training_scope"] != "genuine_reconstruction_labels":
            return {
                "status": "unavailable",
                "reason": "Synthetic smoke-test artifacts are never exposed as production predictions.",
                "trained_model_used": False,
                "artifact_training_scope": manifest["training_scope"],
            }
        import torch

        from .temporal import TemporalLSTMRegressor

        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("embeddings must be a finite non-empty [time, dimension] matrix")
        with np.load(directory / "embedding_normalization.npz", allow_pickle=False) as archive:
            mean = archive["mean"].astype(np.float32)
            scale = archive["scale"].astype(np.float32)
        if values.shape[1:] != mean.shape or mean.shape != scale.shape:
            raise ValueError("Embedding dimension does not match artifact preprocessing")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(directory / "model.pt", map_location=selected_device)
        model = TemporalLSTMRegressor(**checkpoint["model_config"]).to(selected_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        tensor = torch.from_numpy(((values - mean) / scale)[None]).to(selected_device)
        mask = torch.ones((1, len(values)), dtype=torch.bool, device=selected_device)
        with torch.inference_mode():
            score = float(model(tensor, mask).cpu().item())
        return {
            "status": "available",
            "trained_model_used": True,
            "model_type": manifest["model_type"],
            "model_version": manifest["model_version"],
            "training_scope": manifest["training_scope"],
            "score": round(score, 2),
            "interpretation": "Separate learned reconstruction-quality estimate; not a calibrated probability.",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"Temporal model artifact could not be used: {exc}",
            "trained_model_used": False,
        }

