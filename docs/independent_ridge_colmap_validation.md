# Independent Ridge/COLMAP diagnostic validation

This workflow compares the frozen experimental Ridge estimate with an objective
COLMAP registered-image count on a very small set of unseen uploaded videos. It does
not retrain the model, affect Streamlit, change either analyzer, or claim accuracy.
Three videos are only an initial diagnostic study.

## Completed four-video evaluation

The final diagnostic was expanded to four uploaded videos. It found Ridge MAE
**58.743399 percentage points**, including three overpredictions greater than 60
points. Results, methodology, the unresolved 60/60 front-only geometry question and
presentation-ready limitations are preserved in
[`FINAL_EVALUATION_REPORT.md`](../evaluation/ml/independent_ridge_validation/FINAL_EVALUATION_REPORT.md).

This result supersedes the earlier statement that no independent live-video evidence
existed. It remains a small diagnostic rather than a statistically sufficient
accuracy evaluation.

## Inputs

Prepare three videos that were not used in the historical experiments:

1. a mostly static object filmed with a good orbit;
2. a front-only recording of a mostly static object;
3. the waving or animated-object negative control.

Create a CSV using
`evaluation/ml/independent_ridge_validation/video_manifest.example.csv` as the
template. Required columns are `video_id`, `role`, and `video_path`. Video IDs must
be unique and may contain letters, numbers, dots, underscores and hyphens.

## Frozen input policy

Every video is converted into exactly 60 ordered JPEGs by the same deployed adapter:

- source positions: `np.linspace(0, frame_count - 1, 60, dtype=int)`;
- exact frame seeks are verified;
- OpenCV BGR decode, then RGB Pillow image;
- proportional max-edge-800 LANCZOS resize without upscaling;
- Pillow JPEG quality 90 with library defaults;
- filenames `0000.jpg` through `0059.jpg` in temporal order.

The frozen contract SHA-256 is
`eac0ba4e28938adaa03a866c77ed7f2393fe5f45985a345b8f82ac4ac25dd9ef`.
The preparation manifest records every source position and resulting JPEG hash. The
same 60 JPEG files feed both Ridge and COLMAP.

## COLMAP policy and historical differences

The pilot uses CPU SIFT, one shared `SIMPLE_RADIAL` camera, maximum image size 800,
maximum 2048 SIFT features, two threads, sequential matching with overlap 6, and no
automatic exhaustive fallback. Optional flags are applied only when exposed by the
installed COLMAP version and every exact command is retained in the run JSON/logs.

Batch C and R1 used CPU SIFT, max image size 800, max 2048 features, two threads,
one shared camera and sequential overlap 6. Important differences remain:

- historical inputs were prepared CO3D images, whereas this study starts from
  codec-decoded uploaded video and re-encodes JPEGs;
- historical notebooks relied on the installed default camera model; this runner
  freezes `SIMPLE_RADIAL` explicitly;
- the installed COLMAP build may differ from the historical Colab build;
- model conversion/inspection is added after mapping but does not change registration.

These differences mean the study validates the deployed uploaded-video policy, not
byte-level equivalence to the old CO3D labeling pipeline.

## Local command

If COLMAP is already installed:

```bash
python -m pip install -e ".[app-ml]"
python scripts/ml/validate_independent_ridge_colmap.py \
  --manifest /absolute/path/validation_videos.csv \
  --output-dir /absolute/path/independent_validation_run \
  --model-artifact evaluation/ml/historical_registration_experiment/historical_opencv8_v2/experimental_model.joblib
```

The output directory must be new or empty. The source videos are read-only. Each
run retains its prepared JPEGs, hashes, Ridge output, exact COLMAP commands/logs and
reconstruction status. Top-level outputs are:

- `validation_results.csv`;
- `VALIDATION_REPORT.md`;
- `validation_summary.json`;
- `validation_config.json`.

## Google Colab CPU instructions

COLMAP is not installed on the current local machine. Run these cells in a fresh
Google Colab CPU runtime after the implementation is pushed to GitHub.

For the minimal executable workflow, upload and run
`notebooks/independent_ridge_colmap_validation_colab.ipynb`. It clones the latest
committed `main` branch, verifies the pinned Ridge artifact and feature parity,
generates the documented manifest, invokes the existing validation CLI unchanged,
and synchronizes concise evidence to `MyDrive/validation/results/`. The manual cells
below remain as a command-level reference.

```python
!apt-get update -qq
!DEBIAN_FRONTEND=noninteractive apt-get install -y -qq colmap
!git clone https://github.com/itsmesasanda/3D-vdieo-sutability-analyzer.git
%cd /content/3D-vdieo-sutability-analyzer
!python -m pip install -q -e ".[app-ml]"
!colmap -h | head
```

Mount Drive so larger videos do not need browser uploads:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Copy three independent videos into `/content/validation_inputs` and create the
manifest. Adjust the Drive source paths to the real files:

```python
from pathlib import Path
import shutil

inputs = Path('/content/validation_inputs')
inputs.mkdir(exist_ok=True)
shutil.copy2('/content/drive/MyDrive/validation/good_orbit.mp4', inputs / 'good_orbit.mp4')
shutil.copy2('/content/drive/MyDrive/validation/front_only.mp4', inputs / 'front_only.mp4')
shutil.copy2('/content/drive/MyDrive/validation/waving_control.mp4', inputs / 'waving_control.mp4')

Path('/content/validation_videos.csv').write_text(
    'video_id,role,video_path\n'
    'good_orbit,good_orbit,/content/validation_inputs/good_orbit.mp4\n'
    'front_only,front_only,/content/validation_inputs/front_only.mp4\n'
    'waving_control,animated_object_negative_control,/content/validation_inputs/waving_control.mp4\n',
    encoding='utf-8',
)
```

The ignored model binary is intentionally not stored in Git. Copy the exact local
artifact to Drive once, then use that Drive path below. Its required SHA-256 is
`46112ef74ffe29896f892baf6dd037200271d14aa3da5f4f13901eac1af988c6`.

```python
MODEL = '/content/drive/MyDrive/validation/experimental_model.joblib'
!sha256sum "$MODEL"
!python scripts/ml/validate_independent_ridge_colmap.py \
  --manifest /content/validation_videos.csv \
  --output-dir /content/independent_validation_run \
  --model-artifact "$MODEL"
```

The command returns exit code 3 if zero valid COLMAP ground-truth evaluations were
completed; that is an evidence status, not a script crash. Download only the concise
presentation outputs first:

```python
from google.colab import files
files.download('/content/independent_validation_run/validation_results.csv')
files.download('/content/independent_validation_run/VALIDATION_REPORT.md')
files.download('/content/independent_validation_run/validation_summary.json')
files.download('/content/independent_validation_run/validation_config.json')
```

Retain the per-video `runs/<video_id>/validation_record.json`,
`preparation_manifest.json`, `ridge_prediction.json`, COLMAP
`reconstruction_result.json`, and logs as evidence. Large databases, sparse models,
PLY files and prepared images do not need to be downloaded for the presentation.

## Validity rules

- `succeeded`: registration count is read from the selected exported sparse model.
- `completed_mapper_failure`: all COLMAP stages returned successfully but mapper
  produced no sparse model; this is a verified 0/60 result.
- `infrastructure_failure`: COLMAP missing or unavailable; no target and no MAE row.
- `pipeline_failure`: command crash, interruption, export problem or malformed
  result; no target and no MAE row.

MAE includes only rows with both a verified completed ground-truth result and an
available Ridge prediction. Registration percentage is not proof of useful geometry,
surface coverage, or successful downstream reconstruction.
