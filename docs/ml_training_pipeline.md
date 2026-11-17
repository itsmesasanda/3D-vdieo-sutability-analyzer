# Reconstruction-grounded ML pipeline

Milestone 4A prepares training infrastructure; it does not supply genuine labels or
claim model accuracy. The production heuristic v2.1, historical reports, sparse
reconstruction workflow, and Streamlit contract remain unchanged.

## Exact inputs required for genuine training

1. A local CO3D subset with each category's `frame_annotations.jgz`,
   `sequence_annotations.jgz`, and referenced image files. The builder never
   downloads it.
2. Stable sequence IDs matching `co3d:<category>:<sequence_name>` and physical
   object or scene identifiers where annotations/supporting curation can verify
   them. At least three independent groups are technically required for three
   splits; meaningful accuracy claims require a much broader independently labeled
   sample and class/score coverage.

If the annotations do not expose shared physical-object identity, copy
[`split_group_map_template.csv`](../evaluation/ml/split_group_map_template.csv),
manually verify which sequences depict the same object/scene, and pass it with
`--group-map-csv`. Unresolved sequences remain conservatively grouped by sequence;
accuracy claims must disclose that cross-sequence identity leakage could not be
excluded.
3. One reconstruction target CSV row per labeled sequence, based on a versioned,
   documented protocol. Start from
   [`reconstruction_target_template.csv`](../evaluation/ml/reconstruction_target_template.csv).
   Required columns are `sequence_id`, `target_score`, `binary_suitable`,
   `reconstruction_status`, `target_provenance_kind`,
   `target_protocol_version`, `source_artifact`, and `source_sha256`.
4. `target_score` must be an independently defined 0–100 reconstruction-quality
   outcome. It must not be copied from heuristic v1/v2/v2.1 or derived from the same
   CV features used for training. Blank scores remain missing and cannot train.
5. Preserve objective raw measurements—registered-image ratio, sparse point count,
   reprojection error, and track length—plus human object-completeness, geometric-
   plausibility, and actual-usability review where available. Sparse registration
   alone does not prove a complete/useful object reconstruction.
6. Preserve reconstruction status and evidence hashes. `unavailable`, missing, or
   not-run infrastructure is not a bad-video label and cannot carry a numerical
   target. A genuine algorithmic reconstruction failure is valid evidence only
   under the declared target protocol.
7. Optional heuristic v2.1 predictions may be stored as baselines for comparison,
   never as supervision. A classification threshold must be fixed by the target
   protocol before inspecting the test set.

For automatic comparison, copy
[`heuristic_v21_baseline_template.csv`](../evaluation/ml/heuristic_v21_baseline_template.csv)
and provide the frozen score plus `scoring_version=heuristic_v2` and
`scoring_revision=2.1`. The importer rejects other revisions and stores the value in
the separate `heuristic_v21_score` field.

## Versioned dataset contract

The JSONL schema is `video_suitability_ml_dataset/1.0`, one record per complete
sequence. It records:

- sequence/category/object/scene IDs and the split-group ID with provenance;
- full frame count, uniformly sampled positions, and dataset-root-relative paths;
- nullable, named numerical quality/geometry/media features under
  `cv_numeric_features/2.1`;
- optional ResNet cache key and embedding dimension (arrays stay in the cache);
- nullable reconstruction metrics, human review, status, target score, and exact
  provenance;
- optional heuristic v2.1 comparison score; and
- explicit JSON `null` missing values—never sentinel zeros.

Build a manifest after placing an approved subset locally:

```bash
python scripts/ml/build_dataset.py \
  --co3d-root /path/to/co3d \
  --category apple \
  --sample-count 32 \
  --group-map-csv evaluation/ml/split_group_map.csv \
  --labels-csv evaluation/ml/reconstruction_targets.csv \
  --heuristic-baselines-csv evaluation/ml/heuristic_v21_baselines.csv \
  --out data/ml/co3d_dataset_v1.jsonl
```

Use `--metadata-only` for a fast layout/path audit. Omitting `--labels-csv` creates
an intentionally unlabeled feature manifest, not fabricated targets.

## Leakage-safe evaluation

`split_group_id` prefers a verified physical-object ID, then a verified scene ID,
then conservatively falls back to sequence ID. All frames from a sequence and all
sequences with the same known group stay together. A persisted manifest contains a
dataset fingerprint and is rejected if records/groups change.

The Random Forest uses median imputation fit on training data only. The LSTM uses
per-embedding-dimension mean/std fit on training embeddings only. Validation selects
models/epochs; the test split is evaluated once. Results include counts, MAE, RMSE,
Spearman rank correlation, fixed-threshold accuracy/precision/recall/F1/balanced
accuracy/AUC, class counts, and heuristic v2.1/train-mean comparisons. Metrics that
need two classes are `null` when one class is absent.

Local genuine-label tabular command:

```bash
python -m pip install -e '.[ml-tabular]'
python scripts/ml/train_tabular.py \
  --dataset data/ml/co3d_dataset_v1.jsonl \
  --split-manifest evaluation/ml/split_v1.json \
  --artifacts evaluation/ml/artifacts/run_001
```

There is intentionally no synthetic-target option in the production CLI. Unit
tests call the lower-level API with an explicit flag; resulting manifests say
`synthetic_smoke_test_only`, and the Streamlit/inference adapter refuses to expose
them as predictions.

## Models and artifacts

- `random_forest_numeric_cv/1.0`: small CPU baseline using only named video-level
  numerical features. Train this first.
- `frozen_resnet18_temporal_lstm/1.0`: official torchvision ImageNet ResNet-18
  transforms and frozen 512-D frame encoder, followed by a small mask-aware LSTM
  head. Only the head trains. The output is bounded to 0–100.
- ResNet caches are atomic `.npz` files keyed by sequence, frame file sizes/mtimes,
  and extractor configuration. Interrupted Colab extraction resumes by cache hit.
- Artifact manifests SHA-256 hash the model, preprocessing, metric, history, and
  split files and record package versions, device, provenance kinds, and training
  scope. Never load an artifact from an untrusted source (`joblib`/PyTorch files are
  executable serialization formats).

The additive adapter is available without changing the existing report contract:

```python
from video_suitability.ml.inference import attach_optional_ml_prediction

report = attach_optional_ml_prediction(heuristic_report, artifact_directory)
```

The top-level heuristic `overall_score` is retained. With no artifact, a damaged
artifact, a missing dependency, or a synthetic artifact, the added field reports
`status: unavailable` and the heuristic path continues.
`temporal_prediction_from_embeddings` provides the equivalent guarded adapter for
an already extracted `[frames, 512]` ResNet sequence; it applies the exported
train-only normalization and mask-aware head without replacing the heuristic score.

## Google Colab Free instructions

1. Do not download the full CO3D dataset for this milestone. Place only an approved
   subset and its genuine labeled JSONL in Google Drive.
2. Copy/clone this repository to
   `MyDrive/3D-video-suitability-analyzer`, upload the JSONL to
   `MyDrive/video_suitability_ml/dataset_v1.jsonl`, and place images under the
   configured `CO3D_ROOT` while preserving the relative paths stored in JSONL.
3. Open [`ml_training_colab.ipynb`](../notebooks/ml_training_colab.ipynb) in Colab,
   choose a free GPU runtime only for ResNet/LSTM work, and edit the configuration
   cell. The Random Forest runs on CPU.
4. Run through the environment, label-gate, split, and Random Forest cells first.
   The notebook prints disk, memory, library, CUDA, GPU, record, group, and target-
   provenance information.
5. Review the tabular artifact, then set `RUN_EMBEDDING_EXTRACTION=True`. Cache files
   persist in Drive, so reruns skip completed sequences. ResNet weights are about
   tens of megabytes and may be downloaded by torchvision only when this stage is
   enabled.
6. Set `RUN_TEMPORAL_LSTM=True` only after embeddings succeed. Epoch checkpoints
   are written directly to Drive. An interrupted incomplete run is preserved for
   diagnosis; choose a new `RUN_ID` after resolving it rather than overwriting it.
7. Keep `ALLOW_SYNTHETIC_SMOKE=False` for genuine training. Every completed model
   manifest must say `genuine_reconstruction_labels` before integration.

The notebook is generated reproducibly with:

```bash
python scripts/ml/build_colab_notebook.py
```

## Storage and hardware planning

The official CO3Dv2 `--single_sequence_subset` cited in the project instructions is
approximately 8.9 GB compressed. Extraction, image caches, embeddings, and
checkpoints require additional space; reserve roughly 20–30 GB of Drive for that
subset and verify the actual archive/category footprint before download. Larger
category/full-dataset storage is not estimated here and requires explicit approval.

The Random Forest needs ordinary CPU/RAM. CV feature extraction is CPU-capable but
can be slow. Frozen ResNet extraction is practical on a free Colab GPU and can fall
back to CPU; use configurable batches (start at 16–32 and reduce on OOM). The LSTM
head is lightweight, but dataset size and cache I/O still govern runtime. Milestone
4A did not download data, encoder weights, or begin genuine/expensive training.

