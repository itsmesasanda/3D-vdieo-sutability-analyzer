"""Dependency-light regression, ranking, and binary evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average
        position = end
    return ranks


def spearman_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) < 2 or np.std(true) < 1e-12 or np.std(pred) < 1e-12:
        return None
    return float(np.corrcoef(_ranks(true), _ranks(pred))[0, 1])


def _roc_auc(binary: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = binary == 1
    negatives = binary == 0
    n_positive, n_negative = int(positives.sum()), int(negatives.sum())
    if not n_positive or not n_negative:
        return None
    ranks = _ranks(scores)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def evaluate_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    classification_threshold: Optional[float] = 70.0,
    binary_truth: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.ndim != 1 or len(true) == 0:
        raise ValueError("y_true and y_pred must be non-empty equal-length vectors")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("metrics require finite values")
    errors = pred - true
    result: Dict[str, Any] = {
        "sample_count": int(len(true)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "spearman_rank_correlation": spearman_correlation(true, pred),
        "classification": None,
    }
    if classification_threshold is None:
        return result
    if binary_truth is None:
        truth = (true >= classification_threshold).astype(int)
        truth_source = "target_score_threshold"
    else:
        truth = np.asarray(binary_truth, dtype=int)
        if truth.shape != true.shape or not np.isin(truth, [0, 1]).all():
            raise ValueError("binary_truth must be an equal-length boolean vector")
        truth_source = "independent_binary_target"
    predicted = (pred >= classification_threshold).astype(int)
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both_classes = bool(np.unique(truth).size == 2)
    result["classification"] = {
        "threshold": float(classification_threshold),
        "truth_source": truth_source,
        "positive_count": int(truth.sum()),
        "negative_count": int((truth == 0).sum()),
        "accuracy": (tp + tn) / len(truth),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "balanced_accuracy": 0.5 * (recall + tn / (tn + fp))
        if both_classes
        else None,
        "roc_auc": _roc_auc(truth, pred),
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
        "both_classes_present": both_classes,
    }
    return result


def compare_with_heuristic(
    y_true: Sequence[float],
    model_predictions: Sequence[float],
    heuristic_predictions: Sequence[Optional[float]],
    *,
    classification_threshold: float = 70.0,
    binary_truth: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    true = np.asarray(y_true, dtype=float)
    model = np.asarray(model_predictions, dtype=float)
    available = np.asarray([value is not None for value in heuristic_predictions])
    heuristic = np.asarray(
        [float(value) if value is not None else np.nan for value in heuristic_predictions]
    )
    result = {
        "model": evaluate_predictions(
            true,
            model,
            classification_threshold=classification_threshold,
            binary_truth=binary_truth,
        ),
        "heuristic_v2_1": None,
    }
    if available.any():
        result["heuristic_v2_1"] = evaluate_predictions(
            true[available],
            heuristic[available],
            classification_threshold=classification_threshold,
            binary_truth=(
                np.asarray(binary_truth, dtype=bool)[available]
                if binary_truth is not None
                else None
            ),
        )
    return result

