# Registration Random Forest training-data compatibility workflow

## Purpose and non-goals

This workflow prepares final COLMAP registration labels and their exact image inputs
for later Random Forest training. It does not fit an estimator, write `model.joblib`,
create a deployable model manifest, activate Streamlit inference, run COLMAP, or
change the heuristic/viewpoint paths.

The batch builder imports the production extractor from
`video_suitability.ml.registration_features`. Video inference and training-manifest
inputs share the same resize, grayscale, ORB, matching, RANSAC, aggregation, and
feature-order implementation. The only input adapter difference is explicit:

- live inference uniformly samples the uploaded video's complete declared frame
  range;
- training uniformly samples the exact ordered 60-image manifest that was supplied
  to COLMAP for that label.

All 60 training images are existence- and SHA-256-verified before the configured 16
positions are decoded. Missing, reordered, substituted, or checksum-mismatched
frames reject that record; no feature value is imputed.

## Batch label CSV contract

Start each labeling batch from
`evaluation/ml/registration_label_batch_template.csv`. Pass any number of completed
batch CSVs by repeating `--labels-csv`. Paths may be absolute (convenient in Drive)
or relative to that CSV.

Identity and split fields:

- `batch_id`: immutable label-batch identifier;
- `record_id`: unique capture/variant record across all batches;
- `sequence_id`: capture sequence identifier;
- `source_group`: physical source sequence/object identity used for leakage control;
- `variant`: front-only, quarter-orbit, full-orbit, rotation-only, blurred, or the
  batch's declared variant;
- `evaluation_role`: either `development` or `independent_holdout`. This assignment
  must be decided before model selection. A `source_group` cannot appear in both.

Label evidence fields:

- `reconstruction_status`;
- `failure_stage` and `failure_verified`;
- `registered_image_count`, `total_input_image_count`, and
  `registration_percentage`;
- `label_provenance_kind=colmap_sparse_reconstruction`;
- `label_protocol_version`, `label_generated_at_utc`;
- `label_artifact` and `label_artifact_sha256`, pointing to the original
  `reconstruction_result.json`.

Input evidence fields:

- `frames_manifest`, `frames_manifest_sha256`, and `images_directory` for the exact
  60 images used by COLMAP;
- `extraction_metadata` and `extraction_metadata_sha256`;
- `sampling_protocol_id` and `sampling_configuration_sha256`;
- `input_fidelity=exact_labeled_frames`, or `faithfully_regenerated` with a nonblank
  `regeneration_provenance` explaining how hashes/configuration were reproduced.

The sampling-configuration hash is SHA-256 over compact, key-sorted JSON returned by
`registration_sampling_configuration(metadata)`. It covers source cadence/count,
orientation handling, requested sampling, exact frame counts, failures, temporal
coverage, output dimensions/scale, and JPEG quality. The builder recomputes it from
the hashed extraction metadata and reports any mismatch.

## Label acceptance rules

A successful label is accepted only when all of these agree:

1. the row status is `succeeded` or `measured_success`;
2. the hashed COLMAP artifact says `succeeded` and contains objective metrics;
3. registered count, total count, ratio, and percentage agree numerically;
4. the total input count is exactly 60;
5. the hashed frame manifest contains exactly 60 unique chronological source-frame
   entries and all 60 files match their hashes;
6. extraction metadata records 60 requested and 60 extracted frames, no failed
   source indices, unchanged source hashes, and complete timeline coverage.

A failed reconstruction is accepted as a 0% measured target only when the CSV
explicitly records 0/60, `failure_verified=true`, the hashed artifact says
`reconstruction_failed`, recorded COLMAP commands exist, and the failure stage is a
mapper/no-sparse-model stage (`mapper_sequential`, `mapper_exhaustive`, or
`sparse_mapping`). The builder never derives zero merely from a generic failure
string.

`infrastructure_unavailable`, `not_run`, missing, cancelled, interrupted, or
pilot-gated rows are written to `compatibility_report.json` as exclusions. They do
not receive features or numerical targets and do not by themselves prevent an
otherwise valid bundle. Unknown statuses, bad hashes, incompatible preprocessing,
duplicate record IDs, or missing frames are incompatibilities and set
`training_ready=false`.

## Colab-compatible batch command

In Colab, mount Drive, clone or upload this exact repository revision, and install
the local package. No sklearn is required for data preparation:

```bash
%cd /content/3D-vdieo-sutability-analyzer
!python -m pip install -e .
```

Build the immutable merged bundle into a new empty Drive directory:

```bash
!python scripts/ml/build_registration_training_data.py \
  --labels-csv /content/drive/MyDrive/labels/batch_01.csv \
  --labels-csv /content/drive/MyDrive/labels/batch_02.csv \
  --labels-csv /content/drive/MyDrive/labels/batch_03.csv \
  --out-dir /content/drive/MyDrive/training/registration_labels_final_v1 \
  --bundle-version registration-labels-final-v1 \
  --fold-count 5 \
  --parity-case-count 2
```

Exit code 0 means every non-infrastructure record is compatible, grouped CV is
possible, and at least one independent holdout group exists. Exit code 2 means the
audit files were written but training must not start. Read
`compatibility_report.json` for exact row/file reasons.

## Generated training bundle

The output is versioned and checksum-protected:

```text
feature_extractor.json
training_records.jsonl
training_records.csv
development_records.csv
independent_holdout_records.csv
grouped_cv_manifest.json
compatibility_report.json
parity_examples.json
parity/<record-id>/frames_manifest.csv
parity/<record-id>/images/<60 exact files>
training_bundle_manifest.json
```

Every training record preserves `source_group`, `variant`, batch, reconstruction
status, target counts, complete label provenance, input fidelity, extraction hashes,
sampling configuration, feature contract, sampled positions, and the ordered eight
fresh features.

`grouped_cv_manifest.json` assigns complete physical `source_group` values to folds;
all variants of a source stay together. Predeclared independent holdout groups are
listed separately with `independent_holdout_excluded_from_model_selection=true` and
receive no fold assignment. The manifest stores a dataset fingerprint.

`parity_examples.json` uses the same schema consumed by the production model loader.
The final model packager copies/hashes every parity manifest and image, then the
loader re-extracts those saved training features before loading joblib.

The bundle deliberately contains neither `model.joblib` nor
`artifact_manifest.json`. `training_bundle_manifest.json` records
`contains_trained_model=false` and `activates_production_inference=false`.

## Later training handoff

When all batches pass review, train only on `development_records.csv`, using the
source-group folds in `grouped_cv_manifest.json`. Do not inspect or tune against
`independent_holdout_records.csv`; use it once for the final independent evaluation.

After training, copy the trusted model into a new versioned directory alongside the
bundle's `feature_extractor.json`, `parity_examples.json`, and `parity/` directory.
Then run `scripts/ml/package_registration_rf.py`. That later step validates feature
parity, model feature count/order, target semantics, and hashes. It is intentionally
outside this data-preparation workflow.

