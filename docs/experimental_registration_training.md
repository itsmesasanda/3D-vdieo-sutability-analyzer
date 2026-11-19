# Experimental registration-model training workflow

## Current status

The runnable Colab notebook is
`notebooks/experimental_registration_training_colab.ipynb`. It does not contain
invented transformation settings and cannot be executed to completion until the
executed labeling notebook/configuration, exact five held-out source IDs, and CO3D
source tree are supplied in Drive.

The workflow starts from the 647 audited `sparse_model_created` records. It never
uses the 12 provisional mapper-failure zeros, five unlabeled mapper failures, nine
budget pauses, or two matching exceptions.

No model has been trained by adding this workflow. No evaluation metric in this
document is a result.

## Required Drive inputs

1. This repository revision, including the production feature specification.
2. The audited `record_audit.csv` with the 675 deduplicated experiments.
3. The local CO3D image/annotation tree used by labeling.
4. The actually executed labeling notebook/script and batch configuration files.
5. A completed regeneration adapter copied from that executed implementation.
6. A regeneration contract containing hashes of the adapter and executed evidence,
   exact NumPy/OpenCV versions, and a per-variant fidelity decision.
7. A JSON file with exactly five physical `source_group` IDs reserved for independent
   holdout evaluation—not merely category names.
8. Optionally, a compatible held-out label inventory. If absent, the notebook reports
   held-out evaluation unavailable and does not invent it.

Start from:

- `evaluation/ml/experimental_registration_training/regeneration_contract_template.json`
- `scripts/ml/executed_regeneration_adapter_template.py`

The templates deliberately do not implement any transformation. Fill them only from
executed evidence. Hash the completed files after they are frozen.

## Fidelity decisions

Every variant must declare one tier:

- `original_hash_verified`: regenerated files match all original labeling-time
  hashes;
- `executed_contract_reproduced`: exact executed code/config and runtime are
  available, but original output hashes are absent. These records may participate
  only in the explicitly experimental workflow;
- `insufficient_evidence`: excluded.

The notebook independently verifies that sampling, transformation, and determinism
are fully specified. An unknown variant or adapter failure becomes an exclusion. It
never falls back to a similarly named transformation.

## Colab execution

Upload/open the notebook in Google Colab, select a CPU runtime, mount the Drive that
contains the evidence, and edit only the configuration cell. In particular, set:

```python
DRIVE_PROJECT_ROOT
CO3D_ROOT
REGENERATION_CONTRACT_JSON
REGENERATION_ADAPTER_PY
HOLDOUT_SOURCE_GROUPS_JSON
OUTPUT_PARENT
```

Use a new output directory/run ID for every execution. The notebook refuses to
overwrite an existing run.

## Evaluation protocol

After exact-manifest verification and production feature extraction, the notebook:

1. records accepted, uncertain, and excluded inputs;
2. freezes one five-fold `GroupKFold` plan keyed by physical `source_group`;
3. uses the identical fold indices for Dummy mean, Ridge, Random Forest, and
   Gradient Boosting;
4. keeps Ridge scaling inside each training fold;
5. saves fold and aggregate MAE/RMSE, out-of-fold predictions, prediction
   distributions, and errors by degradation family;
6. selects by grouped-CV MAE, then RMSE and model name as deterministic tie-breakers;
7. fits the selected model on all development records;
8. evaluates it once on compatible held-out records, if available.

The holdout is never used for preprocessing, hyperparameters, model selection, or
the tie-breaker.

## Versioned outputs

Each Drive run contains:

```text
runtime_environment.json
executed_implementation_inspection.json
regeneration_contract_audit.json
input_reproduction_decisions.csv
input_reproduction_summary.json
feature_records.csv
grouped_cv_splits.json
grouped_cv_fold_metrics.csv
grouped_cv_predictions.csv
grouped_cv_model_summary.csv
prediction_distributions.csv
errors_by_degradation_type.csv
model_selection.json
independent_holdout_evaluation.json
candidate_models/*.joblib
artifact_<version>/
  model.joblib
  feature_extractor.json
  parity_examples.json
  parity/<case>/frames_manifest.csv
  parity/<case>/images/<60 files>
  experimental_training_manifest.json
  experimental_files_sha256.json
deployment_status.json
evaluation_report.json
```

`experimental_training_manifest.json` records incomplete provenance, decision and
fidelity counts, selected model, CV metrics, and holdout status. The artifact stays
experimental unless every legitimate deployment gate passes.

## Deployment gate

The notebook does not activate a model by default. Packaging requires all of:

- an explicit request in the configuration cell;
- completed label/input provenance;
- Random Forest selected by grouped CV (the current loader is RF-specific);
- a completed independent holdout evaluation;
- all inputs verified against original hashes;
- saved parity cases that re-extract identically.

Only then does it invoke `scripts/ml/package_registration_rf.py`, which creates the
hash-protected production `artifact_manifest.json` and reruns feature parity. If any
gate fails, no production manifest is created and the existing application remains
inactive. Never copy a merely experimental artifact into
`models/registration_rf_current`.
