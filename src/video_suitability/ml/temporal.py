"""Lightweight bounded LSTM head over cached frozen ResNet-18 embeddings."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import create_artifact_directory, write_manifest
from .metrics import compare_with_heuristic
from .schema import DatasetRecord
from .splits import SplitManifest, assert_no_group_leakage


MODEL_TYPE = "frozen_resnet18_temporal_lstm"
MODEL_VERSION = "1.0"

try:  # Optional locally; available in the Colab training runtime.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency-less installations
    torch = None
    nn = None


if nn is not None:

    class TemporalLSTMRegressor(nn.Module):
        """Mask-aware LSTM with a bounded 0–100 regression head."""

        def __init__(
            self,
            input_dim: int = 512,
            hidden_dim: int = 128,
            num_layers: int = 1,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.config = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            }
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, hidden_dim // 2), 1),
            )

        def forward(self, embeddings: Any, valid_mask: Any) -> Any:
            if embeddings.ndim != 3 or valid_mask.ndim != 2:
                raise ValueError("Expected embeddings [B,T,D] and valid_mask [B,T]")
            if embeddings.shape[:2] != valid_mask.shape:
                raise ValueError("Embedding and mask batch/time dimensions differ")
            lengths = valid_mask.long().sum(dim=1)
            if torch.any(lengths <= 0):
                raise ValueError("Every sequence needs at least one valid embedding")
            expected = (
                torch.arange(valid_mask.shape[1], device=valid_mask.device)[None, :]
                < lengths[:, None]
            )
            if not torch.equal(expected, valid_mask.bool()):
                raise ValueError("Valid masks must be contiguous prefixes")
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            logits = self.head(hidden[-1]).squeeze(-1)
            return torch.sigmoid(logits) * 100.0

else:

    class TemporalLSTMRegressor:  # pragma: no cover
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Install torch to construct TemporalLSTMRegressor")


def pad_embedding_batch(
    sequences: Sequence[np.ndarray], targets: Sequence[float]
) -> Tuple[Any, Any, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Install torch for temporal batches")
    if not sequences or len(sequences) != len(targets):
        raise ValueError("sequences and targets must be non-empty and equal length")
    dimensions = {array.shape[1] for array in sequences if array.ndim == 2}
    if len(dimensions) != 1 or len(dimensions) == 0:
        raise ValueError("All embeddings must be [time, same_dimension]")
    max_time = max(len(array) for array in sequences)
    if max_time <= 0:
        raise ValueError("Sequences cannot be empty")
    values = np.zeros((len(sequences), max_time, next(iter(dimensions))), np.float32)
    mask = np.zeros((len(sequences), max_time), bool)
    for index, array in enumerate(sequences):
        if array.ndim != 2 or not np.isfinite(array).all() or len(array) == 0:
            raise ValueError("Embeddings must be finite non-empty matrices")
        values[index, : len(array)] = array
        mask[index, : len(array)] = True
    return (
        torch.from_numpy(values),
        torch.from_numpy(mask),
        torch.tensor(targets, dtype=torch.float32),
    )


def train_temporal_lstm(
    records: Sequence[DatasetRecord],
    split_manifest: SplitManifest,
    embeddings_by_sequence: Mapping[str, np.ndarray],
    artifact_root: Path,
    *,
    allow_synthetic_targets: bool = False,
    hidden_dim: int = 128,
    batch_size: int = 16,
    max_epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 1e-3,
    random_state: int = 20261117,
    classification_threshold: float = 70.0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train only the temporal head; embeddings and normalization are cached/exported."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Install torch and torchvision for temporal training")
    for record in records:
        record.validate(
            require_target=True, allow_synthetic_target=allow_synthetic_targets
        )
        if record.sequence_id not in embeddings_by_sequence:
            raise ValueError(f"Missing embeddings for {record.sequence_id}")
    assert_no_group_leakage(records, split_manifest)
    synthetic_flags = {record.target.provenance.is_synthetic for record in records}
    if len(synthetic_flags) != 1:
        raise ValueError("Do not mix synthetic and genuine targets")
    is_synthetic = synthetic_flags == {True}
    if is_synthetic and not allow_synthetic_targets:
        raise ValueError("Synthetic-target training was not explicitly enabled")
    by_split = {
        name: [
            record
            for record in records
            if split_manifest.split_for(record.sequence_id) == name
        ]
        for name in ("train", "validation", "test")
    }
    if any(not values for values in by_split.values()):
        raise ValueError("Train, validation and test splits must all be non-empty")

    train_embeddings = np.concatenate(
        [embeddings_by_sequence[record.sequence_id] for record in by_split["train"]],
        axis=0,
    ).astype(np.float32)
    mean = train_embeddings.mean(axis=0)
    scale = train_embeddings.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    mean = mean.astype(np.float32)
    input_dim = int(train_embeddings.shape[1])

    def normalized(record: DatasetRecord) -> np.ndarray:
        value = np.asarray(embeddings_by_sequence[record.sequence_id], np.float32)
        if value.ndim != 2 or value.shape[1] != input_dim or not np.isfinite(value).all():
            raise ValueError(f"Invalid embeddings for {record.sequence_id}: {value.shape}")
        return (value - mean) / scale

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_state)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalLSTMRegressor(input_dim=input_dim, hidden_dim=hidden_dim).to(
        selected_device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.HuberLoss(delta=10.0)
    artifact_dir = create_artifact_directory(artifact_root, MODEL_TYPE, MODEL_VERSION)
    checkpoints = artifact_dir / "checkpoints"
    checkpoints.mkdir()
    np.savez_compressed(artifact_dir / "embedding_normalization.npz", mean=mean, scale=scale)

    def batches(split_records: Sequence[DatasetRecord], shuffle: bool) -> List[List[DatasetRecord]]:
        indices = np.arange(len(split_records))
        if shuffle:
            np.random.shuffle(indices)
        return [
            [split_records[int(index)] for index in indices[start : start + batch_size]]
            for start in range(0, len(indices), batch_size)
        ]

    def run_split(split_records: Sequence[DatasetRecord], training: bool) -> Dict[str, Any]:
        model.train(training)
        losses, predictions, targets = [], [], []
        for batch_records in batches(split_records, training):
            embedded = [normalized(record) for record in batch_records]
            target_values = [float(record.target.target_score) for record in batch_records]
            values, mask, target = pad_embedding_batch(embedded, target_values)
            values, mask, target = (
                values.to(selected_device),
                mask.to(selected_device),
                target.to(selected_device),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                predicted = model(values, mask)
                loss = criterion(predicted, target)
                if training:
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predictions.extend(predicted.detach().cpu().numpy().tolist())
            targets.extend(target.detach().cpu().numpy().tolist())
        return {
            "loss": float(np.mean(losses)),
            "predictions": predictions,
            "targets": targets,
        }

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_result = run_split(by_split["train"], True)
        with torch.no_grad():
            validation_result = run_split(by_split["validation"], False)
        entry = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "validation_loss": validation_result["loss"],
        }
        history.append(entry)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config,
            "epoch": epoch,
            "history": history,
            "random_state": random_state,
        }
        torch.save(checkpoint, checkpoints / f"epoch_{epoch:03d}.pt")
        if validation_result["loss"] < best_loss - 1e-8:
            best_loss = validation_result["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, artifact_dir / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    best = torch.load(artifact_dir / "model.pt", map_location=selected_device)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    metrics: Dict[str, Any] = {}
    with torch.no_grad():
        for split_name in ("validation", "test"):
            result = run_split(by_split[split_name], False)
            binary_truth = (
                [bool(record.target.binary_suitable) for record in by_split[split_name]]
                if all(
                    record.target.binary_suitable is not None
                    for record in by_split[split_name]
                )
                else None
            )
            metrics[split_name] = compare_with_heuristic(
                result["targets"],
                result["predictions"],
                [record.heuristic_v21_score for record in by_split[split_name]],
                classification_threshold=classification_threshold,
                binary_truth=binary_truth,
            )
            metrics[split_name]["sequence_ids"] = [
                record.sequence_id for record in by_split[split_name]
            ]
    preprocessing = {
        "embedding_encoder": "frozen_resnet18_imagenet1k_v1",
        "embedding_dimension": input_dim,
        "normalization": "per-dimension mean/std fitted on training embeddings only",
        "mask_rule": "contiguous valid prefix",
        "output_range": [0.0, 100.0],
    }
    (artifact_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
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
        files=[
            "model.pt",
            "embedding_normalization.npz",
            "preprocessing.json",
            "history.json",
            "metrics.json",
            "split_manifest.json",
        ],
        metadata={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "selected_device": selected_device,
            "record_count": len(records),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "target_provenance_kinds": sorted(
                {record.target.provenance.kind for record in records}
            ),
            "synthetic_metrics_are_not_real_accuracy": bool(is_synthetic),
        },
    )
    return {
        "artifact_dir": str(artifact_dir),
        "training_scope": training_scope,
        "best_epoch": best_epoch,
        "history": history,
        "metrics": metrics,
    }

