# Milestone 4A audit of the reference CO3D notebook

Audit date: 2026-11-17. This ML-focused audit builds on the complete 50-cell JSON,
source, metadata, execution-output, and artifact audit in
[`notebook_audit.md`](notebook_audit.md). No reference cell was executed: it includes
Colab environment mutations, an interrupted multi-gigabyte download, and training
code with no genuine labels.

## What is reusable

| Reference cells | Reusable idea | Milestone 4A implementation |
|---|---|---|
| 5–12 | Read CO3D `.jgz` annotations, preserve frame-number order, inspect missing metadata | `datasets/co3d.py` validates gzip/JSON structure, path containment, image existence, unique frame identities, sequence metadata, and deterministic order. |
| 20–21 | Group sequences before splitting | `ml/splits.py` freezes complete physical-object/scene groups into one split, fingerprints the dataset, and checks sequence coverage and leakage. Sequence ID is a documented conservative fallback when object identity is absent; a reviewed group-map CSV can override it. |
| 23–24 | Uniform whole-sequence frame sampling and validity masks | The CO3D loader uses the production uniform sampler. Cached embeddings store only valid decoded frames; the LSTM pads at batch time and requires a contiguous validity prefix. |
| 26–28 | Frozen ImageNet ResNet-18 frame features and caching | `ResNet18FrameEmbedder` uses the official torchvision weight transforms, freezes every encoder parameter, produces 512-D embeddings, and caches by source-file metadata plus extractor configuration. |
| 30–35 | A small temporal LSTM regressor | `TemporalLSTMRegressor` is mask-aware and uses a sigmoid-bounded 0–100 head. Normalization is fit on training embeddings only, checkpoints are versioned, and validation—not test—selects the best epoch. |
| 36–40 | MAE, rank metrics, baselines, and independent test evaluation | `ml/metrics.py` reports sample counts, MAE, RMSE, Spearman correlation, and classification metrics only with their class-support status. Model results are paired with heuristic v2.1 and train-mean baselines. |
| 41–43 | Checkpoint/preprocessing exports | Artifacts contain hashed model, preprocessing, split, history, and metric files plus library versions and immutable training scope. |
| 45–46 | A trained-model inference entry point | `ml/inference.py` adds a separate optional prediction while retaining the existing heuristic score and returning an explicit fallback when a genuine artifact is absent. |

## What was rejected or repaired

- The notebook generated no COLMAP labels, extracted features, trained sequence
  model, evaluation result, or saved checkpoint. ImageNet encoder weights are not a
  trained video-quality predictor.
- The interrupted category downloader is not reused. Milestone 4A downloads no
  CO3D data.
- Infrastructure-unavailable COLMAP runs cannot become score zero. They remain
  missing targets. A documented failed reconstruction may be a genuine outcome,
  but only its label protocol may decide its score.
- Heuristic v1/v2/v2.1 outputs are rejected as target provenance. They are retained
  only as comparison baselines.
- The notebook's unaligned CO3D/reference point-cloud coverage is not imported as a
  target. Registration, scale, and alignment must first be validated.
- The notebook's frame-level split and broken overlap-check loop are replaced by a
  record-per-sequence, group-frozen manifest with a dataset fingerprint.
- Its temporal feature scaler was fitted after model training and applied only at
  inference. Both tabular imputation and embedding normalization now fit on the
  train split and are exported with the model.
- Its 12-element video vector was incorrectly treated as 12 time steps. The new
  LSTM consumes one 512-D ResNet embedding per sampled frame; numerical video-level
  features go only to the tabular baseline.
- Non-prefix validity masks could be packed incorrectly. The new batch builder
  compacts only successfully extracted frames and rejects masks with internal holes.
- The unbounded `score * 100` neural head is replaced by `sigmoid(logit) * 100` and
  tested for range.
- The reference inference reader examined only the first 512 frames. Dataset frame
  lists now span every sequence uniformly and are persisted in the schema.
- CO3D sequences without trustworthy timestamps do not receive fabricated
  timestamp-normalized speed/shake values. Those feature fields are JSON `null`;
  non-temporal quality and correspondence metrics remain available.
- F1 threshold search against a thresholded version of the same composite label is
  not reused. Classification thresholds must be predeclared by the independent
  label protocol, and one-class test sets explicitly report unavailable balanced
  accuracy/AUC.

This audit identifies an implementable foundation, not evidence that the reference
model worked or that the new models are accurate.

