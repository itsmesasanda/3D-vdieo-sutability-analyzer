# 3D Reconstruction Video Suitability Analyzer

This repository contains a CPU-runnable MVP that analyzes a video of a mostly static physical object filmed from multiple viewpoints. It produces an explicitly **heuristic** 0–100 suitability score, five subscores, timestamped problems, capture recommendations, raw evidence, and uncertainty warnings.

It also reports a separate **Estimated Viewpoint Diversity** category (`Low`,
`Moderate`, `High`, or `Unknown`). This category uses bounded multi-gap ORB/RANSAC
evidence and does not change the heuristic score or optional Random Forest result.
An optional middle-frame object box focuses this category on tracked object features
while preserving background matches as separate diagnostics.

The public Streamlit deployment hard-disables both learned predictors. No Ridge or
Random Forest artifact has passed video-input feature parity, so the UI explicitly
reports live ML as `unavailable`; neither model can be activated by an environment
variable or a stray artifact. The optional CLI Random Forest path retains its strict
compatibility gates for a future verified artifact.

The final-label preparation path is also available without training. It merges
multiple COLMAP label batches, verifies the exact 60-image inputs and provenance,
extracts fresh production-compatible features, creates physical-source grouped CV
folds and a predeclared independent holdout, and packages loader-compatible parity
examples. Infrastructure failures are excluded rather than converted to zero. See
[the compatibility workflow](docs/registration_training_compatibility.md). Historical
Colab outputs can first be audited and normalized with the read-only
[Colab label importer](docs/colab_label_compatibility_audit.md). The supplied real
60-frame archive and its current evidence gaps are documented in the
[real dataset audit](docs/real_colmap_csv_dataset_audit.md).

The production heuristic does **not** reconstruct a 3D model, upload the video, or
claim an ML prediction. A separate Milestone 3A experiment can run an installed
COLMAP executable for reconstruction-grounded validation; it never changes the
heuristic result. No trained/validated model artifact currently exists.

## Install

Python 3.9+ is supported. The runtime uses only NumPy and headless OpenCV.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No CO3D data, CUDA installation, GPU, paid API, or network service is needed after installation.

For the Streamlit UI (live ML remains disabled):

```bash
python -m pip install -e ".[ui]"
```

## Run the CLI

```bash
python -m video_suitability.infer \
  --video /path/to/object_capture.mp4 \
  --out report.json
```

The installed command is equivalent:

```bash
video-suitability --video /path/to/object_capture.mp4 --out report.json
```

Useful bounded-resource options:

```bash
video-suitability \
  --video capture.mp4 \
  --out report.json \
  --max-samples 32 \
  --max-side 640
```

Optionally focus only the separate viewpoint category on a normalized middle-frame
object box and include stage timings:

```bash
video-suitability --video capture.mov --out report.json \
  --object-bbox 0.25,0.10,0.50,0.80 --profile
```

The box never changes the main suitability score. If affine-RANSAC tracking cannot
retain it across at least 70% of samples, the viewpoint result is `Unknown`.

After a compatible final Random Forest artifact is installed, the CLI can attach
its independent registration prediction:

```bash
video-suitability --video capture.mp4 --out report.json \
  --registration-artifact /absolute/path/to/versioned_artifact
```

If calibrated pinhole intrinsics are known, they can be supplied for scale-free
relative-pose and rotation-compensated parallax indicators. Values are in pixels;
give the image size on which they were calibrated so they can be scaled to analysis
resolution:

```bash
video-suitability --video capture.mp4 --out report.json \
  --camera-fx 1220 --camera-fy 1215 --camera-cx 960 --camera-cy 540 \
  --intrinsics-width 1920 --intrinsics-height 1080
```

The recovered translation is a unit direction only. Monocular two-view geometry
does not recover metric scale.

If `--out` is omitted, JSON is written to stdout. Invalid/missing/undecodable videos return exit code 2 with a concise error on stderr.

## Report contract

The stable top-level fields are:

- `analysis_method`: identifies `explainable_cpu_heuristic`, with `trained_model_used: false` and `reconstruction_run: false`.
- `scoring_version`: currently `heuristic_v2`.
- `scoring_revision`: currently `2.1`; this additive field identifies controlled
  regression repairs without breaking the established v2 contract.
- `overall_score`: bounded 0–100 v2 heuristic score.
- `baseline_v1` and `score_comparison`: preserve the original formula and expose the exact before/after delta.
- `confidence_or_uncertainty`: evidence level and limitations; this is not statistical confidence.
- `subscores`: visual quality, exposure/lighting, feature trackability, geometric consistency, and viewpoint motion.
- `raw_metrics`: video/sampling metadata, per-frame quality, ORB keypoints, and adjacent-pair RANSAC metrics.
- `problem_segments`: sampled evidence grouped into timestamped intervals.
- `recommendations`: capture changes tied to detected problems.
- `warnings`: method limitations and degraded-analysis conditions.
- `critical_failure_gate_evaluations`: every documented physical gate, whether it triggered, its evidence, and its maximum score.
- `geometry_reliability`: visible overlap, track-retention, rotation/planar, and cadence uncertainty.
- `viewpoint_diversity`: independent Low/Moderate/High/Unknown multi-gap result,
  raw pair evidence, repeated-view intervals, optional intrinsic-calibrated motion
  indicators, optional object tracking and separate background diagnostics,
  decision explanations, limitations, and filming recommendations. It explicitly records
  `affects_overall_score: false` and `colmap_run: false`.
- `registration_prediction` in Streamlit or the opt-in CLI reports a separate
  predicted COLMAP registered-image percentage only after artifact hashes, target
  semantics, the exact eight-feature contract, and saved-training-example parity
  all pass. It never affects either heuristic result and does not run COLMAP.

The overall score uses fixed, declared weights:

| Subscore | Weight |
|---|---:|
| Visual quality | 12% |
| Exposure and lighting | 8% |
| Feature trackability | 15% |
| Geometric consistency | 35% |
| Viewpoint motion | 30% |

V2 combines these using a weighted geometric mean so saturated appearance metrics cannot fully compensate for missing geometric/viewpoint evidence. Scores at or above 70 are labeled `likely_suitable`, scores from 45 to below 70 are `review_recommended`, and lower scores are `unlikely_suitable`. These thresholds remain engineering defaults, **not calibrated against independent reconstruction ground truth**. Documented caps cover insufficient temporal evidence, missing geometry, stationary views, reliable rotation/planar cues, persistent overlap loss, fast/irregular motion with continuity failures, severe exposure, and pervasive objectively severe blur.

## What is measured

- Frames are sampled uniformly from the complete codec-declared frame range, not just from the beginning. Only downscaled samples are retained in memory.
- Exposure, clipped pixels, contrast, Laplacian sharpness, gradient magnitude, and entropy are measured per frame.
- ORB features are matched between adjacent sampled frames. Fundamental-matrix RANSAC is preferred; homography RANSAC is an explicit fallback.
- The viewpoint module reuses those ORB features, adds a bounded set of quarter-gap,
  half-gap, and endpoint pairs, and detects likely repeated views. Its uncalibrated
  parallax proxy now requires spatially distributed RANSAC consensus; high overlap or
  optical flow alone is never diversity evidence. Long-gap matches that repeatedly
  split into broad near-static and strongly moving populations force `Unknown` as
  possible object animation. A repeated endpoint can suggest loop closure but is
  never presented as proof of a complete orbit.
- With an optional object box, object-internal ORB matches propagate the box using
  affine RANSAC with forward/backward cycle diagnostics; object and background
  feature caches are then evaluated separately. A moving object against a nearly
  static matched background, failed tracking, or background-dominated evaluable pairs
  produce `Unknown`.
- When camera intrinsics are supplied, essential-matrix RANSAC and `recoverPose`
  provide relative rotation, translation direction (without scale), and a
  rotation-compensated bearing-change indicator. Without intrinsics these fields are
  explicitly unavailable.
- Forward/backward Lucas–Kanade flow measures short-gap track retention and timestamp-normalized speed. Direction/speed discontinuities are used only when sample gaps are at most 0.30 seconds; otherwise shake is reported as unresolved and ORB continuity is the fallback.
- Verified match count/ratio, spatial spread, overlap failures, descriptor ambiguity, motion regularity, and homography residuals feed cautious geometry and viewpoint proxies.
- Source resolution, approximate bitrate/bits-per-pixel/frame, and sharpness at analysis resolution are reported. Low pixel resolution alone never triggers failure.

Image displacement is not proof of 3D parallax. Background features, a moving object, a planar object, or camera rotation can all affect it, so the JSON preserves the raw evidence and warnings.

## Generate deterministic validation videos

```bash
python scripts/generate_test_videos.py --out-dir data/samples
```

This creates deterministic local MP4 fixtures (ignored by Git) for:

- smooth multi-layer orbit;
- front-only, quarter-orbit, and periodic full-orbit viewpoint scenarios;
- stationary view and pure rotation;
- low-texture and deliberately repetitive/ambiguous appearance;
- shaky orbit and fast motion;
- low resolution, blur, and severe darkness;
- insufficient temporal coverage;
- the original homography-dominant `orbit_good.mp4` regression case.

They test failure detection and plumbing; their scores are not benchmark metrics or real-world validation.

## Test

```bash
python -m pytest -q
```

The suite covers uniform full-video sampling, decoder strategy, invalid inputs,
blur/exposure metrics, stationary/pure-rotation/smooth/shaky/fast/low-resolution/
short-coverage cases, the five requested viewpoint-diversity scenarios plus
low-texture/repetitive ambiguity gates, optional intrinsic
scaling/relative-pose fields, repeated-view warnings, score-path isolation, v1
preservation, JSON serialization, problem timestamps,
score bounds, gates, report-to-timeline/frame integration, evaluation missing values,
object-box validation/tracking/background separation, Streamlit rendering, and a
real CLI subprocess writing JSON.

## Batch validation of local real videos

The batch validator treats input media as read-only, verifies SHA-256 before/after each analysis, continues across per-file failures, and writes per-video JSON plus review artifacts:

```bash
python scripts/validate_real_videos.py \
  --input-dir data/real_tests \
  --output-dir reports/real_tests \
  --max-samples 32 \
  --max-side 640
```

Generated artifacts are:

- `<original-filename>.json` for every supported video, including failure JSON if decoding fails.
- `summary.csv` with metadata, scores, subscores, selected raw metrics, full raw-metric JSON, warnings, and detected problem JSON.
- `comparison_report.md` with evidence-based false-positive/false-negative review risks; filenames are never treated as ground truth.
- `before_after_components.csv` with v1/v2 score, label, subscore, gate, and warning changes.
- `human_review_template.csv` for independent review observations.

Reports and source videos are ignored by Git to avoid committing private or large media-derived artifacts.

## Real-video viewpoint validation

The viewpoint-specific validator reuses `analyze_video`, keeps both suitability
scores untouched, records wall and process CPU time, and produces selected
ORB/RANSAC match overlays with image-motion arrows:

```bash
python scripts/validate_viewpoint_real_videos.py \
  --input-dir data/samples/real_tests \
  --output-dir evaluation/viewpoint_real_validation \
  --max-samples 32 --max-side 640 \
  --max-diagnostic-pairs 3 \
  --review-csv evaluation/real_video_human_review.csv
```

Outputs include `validation.csv`, a Markdown audit, per-video JSON, and JPEG
diagnostics. Full JSON and images are ignored because they can reveal source-media
details; the derived CSV and audit remain trackable. Green diagnostic lines are
RANSAC-verified matches, gray lines are rejected ratio-test matches, and yellow
arrows show image-space displacement—not proven camera translation.

Optional intrinsics are supplied in a filename-keyed JSON object through
`--intrinsics-json`. They are never guessed. The validator flags calibration as
unverified because crop, stabilization, digital zoom, focal-length errors, and an
incorrect principal point can bias essential-matrix results.

The current audit found three real phone MOVs and one synthetic regression MP4 in
the historical folder. All phone videos return `Unknown` because long-gap match
sets are highly ambiguous across the patterned background, object, and furniture;
the synthetic MP4 returns `Moderate`. These are diagnostic outcomes, not a
real-world accuracy estimate. See
[`evaluation/viewpoint_real_validation/REPORT.md`](evaluation/viewpoint_real_validation/REPORT.md).

## Object-focused comparison validation

The optional object-focused comparator consumes a CSV manifest with a representative
box for each video and reports whole-frame versus object-only classifications,
geometric evidence, tracking/background diagnostics, actual wall/CPU timings, and
an explicit score-invariance check:

```bash
python scripts/compare_object_focused_validation.py \
  --input-dir data/samples/real_tests \
  --manifest evaluation/object_focused_validation/real_samples_manifest.csv \
  --output-dir evaluation/object_focused_validation \
  --max-samples 16 --max-side 480
```

A five-scenario same-physical-object protocol is prepared in
`evaluation/object_focused_validation/physical_object_manifest_template.csv`; use
`--require-complete-scenario-set` to enforce front-only, quarter-orbit, full-orbit,
rotation-only, and blurred recordings with one object ID. The accessible three MOVs
remain `Unknown`: each selected bottle region tracks only 1/16 samples and provides
zero verified object-only long-gap matches. Their protocol labels and reconstruction
outcomes are unavailable, so this is not a misclassification-rate or accuracy
claim. See [the object-focused audit](docs/object_focused_validation.md).

## Python API

```python
from video_suitability.infer import analyze_video

report = analyze_video("capture.mp4")
print(report["overall_score"], report["suitability_label"])
print(report["baseline_v1"]["overall_score"], report["score_comparison"])
```

## Streamlit UI

```bash
python -m pip install -e ".[ui]"
streamlit run app.py
```

Open the printed local URL (normally `http://localhost:8501`). The competition MVP
provides:

- a prominent 0–100 score, suitability band, v1 comparison, five explained
  components, and confidence/uncertainty banner;
- a separate Estimated Viewpoint Diversity panel with Low/Moderate/High/Unknown,
  evidence counts, limitations, and actionable capture guidance when the optional
  experimental checkbox is enabled;
- an explicit notice that live Ridge and Random Forest inference is unavailable;
- optional middle-frame object selection, tracked object-only viewpoint evidence,
  background-contamination diagnostics, and a clear Unknown reason on track loss;
- a filterable timestamped problem timeline and representative problem frames;
- what/where/why/action guidance for blur, exposure, overlap, tracking, shake,
  excessive motion, weak viewpoint change, and other detected issues;
- technical gates, warnings, complete JSON, and report download;
- a preserved v1/v2/human-feedback evaluation view that explicitly avoids accuracy
  claims and keeps missing human numbers missing.

Uploaded content is analyzed locally. Analysis is button-triggered and session state
caches an upload by SHA-256 plus object box so ordinary widget reruns do not repeat
the work. Problem-frame rendering is deferred to a separate button. See
[`docs/ui_mvp_audit.md`](docs/ui_mvp_audit.md) for verified behavior and limitations.

### Streamlit Community Cloud deployment

- Repository: `itsmesasanda/3D-vdieo-sutability-analyzer`
- Branch: `main`
- Application entry point: `app.py`
- Recommended Python version: `3.11`
- Secrets: none

Commit and push the deployment files, then open `https://share.streamlit.io`, select
**Create app**, choose the repository and `main` branch, and enter `app.py` as the
file path. In Advanced settings select Python 3.11 and leave Secrets empty. Community
Cloud installs the root `requirements.txt`, uses `packages.txt` for Debian `ffmpeg`,
and reads `.streamlit/config.toml` for the 150 MB upload limit.

The public-cloud default runs the established heuristic path only. Estimated
Viewpoint Diversity is opt-in; an exception becomes an isolated `Unknown` result and
cannot abort or change the main score. Uploaded bytes are rejected above 150 MB,
written to a unique temporary file, and removed in a `finally` block. The analyzer
retains at most 32 downscaled analysis samples by default.

Local preflight:

```bash
python -m pip install -r requirements.txt
python scripts/check_streamlit_deployment.py
python -m pytest -q
streamlit run app.py --server.headless true
```

After deployment, test one short MP4 and one iPhone/Android MOV. Leave experimental
viewpoint analysis off for the first run, confirm the heuristic score and JSON
download, then repeat with the experimental checkbox if desired. Live Ridge and
Random Forest output must remain unavailable.

## 4K performance

For the accessible 2160×3840 HEVC `IMG_3913.MOV`, two end-to-end runs improved from
a 27.518-second mean to 14.443 seconds (1.905×, 47.516% lower). Dense uniform sample
positions use a single forward decode pass; long videos with sparse positions retain
random seeking. All 32 resized frames were pixel-identical in a direct comparison,
and score, components, quality/geometry records, and problem segments matched the
preserved report. See [`docs/performance_benchmark.md`](docs/performance_benchmark.md)
and `scripts/benchmark_video_analysis.py`. These local two-run timings are not a
cross-platform guarantee.

The exact Streamlit payload path is benchmarked separately from diagnostic
rendering by `scripts/benchmark_streamlit_upload.py`. On one local `IMG_3914.MOV`
run, upload-to-result was 8.779 s wall for whole-frame and 9.152 s for object-focused
analysis; rendering six diagnostics later added 4.405 s and 5.634 s. Decoding used
6.860–7.127 s and dominated both runs. Raw stage/CPU timings are in
`benchmarks/streamlit_upload_real_*.json`.

## Random Forest registration prediction

Set `VIDEO_SUITABILITY_RF_ARTIFACT_DIR` to a trusted versioned artifact or install
it at `models/registration_rf_current`. Replacing that directory supports the final
retrained model without changing UI or scoring code. The loader requires a hashed
eight-feature OpenCV specification and at least one hashed training video whose
locally re-extracted vector matches the value saved by Colab. Missing or
incompatible artifacts never produce a prediction.

The current repository has no saved model, exact historical eight-feature export,
or parity examples. It therefore does not claim to reproduce that missing Colab run
or report a learned value. The proposed final-retraining contract, exact formulas,
deployment steps, tests, and CPU extraction benchmark are documented in
[the Random Forest integration audit](docs/registration_random_forest.md).

## Sparse reconstruction validation (Milestone 3A)

The optional experiment extracts timestamped frames, verifies source checksums,
runs CPU sparse COLMAP when available, parses objective sparse-model metrics, and
keeps subjective human review separate:

```bash
python scripts/reconstruction/audit_environment.py \
  --out evaluation/reconstruction/environment_audit.json
python scripts/reconstruction/extract_frames.py \
  --video data/samples/real_tests/IMG_3913.MOV \
  --output-dir evaluation/reconstruction/IMG_3913 \
  --sample-fps 2 --max-side 1600
python scripts/reconstruction/run_colmap.py \
  --images-dir evaluation/reconstruction/IMG_3913/images \
  --workspace-dir evaluation/reconstruction/IMG_3913/colmap \
  --camera-model OPENCV --retry-exhaustive
```

The current macOS host does not have COLMAP and lacks enough free disk for a safe
Homebrew install. The pilot extraction is complete, while reconstruction fields
remain explicitly unavailable. See
[the experiment report](evaluation/reconstruction/REPORT.md) and
[the Linux/Colab runbook](evaluation/reconstruction/COLAB_LINUX_RUNBOOK.md). Sparse
registration and point counts are not substitutes for object-completeness review.

For cloud execution, open
[`notebooks/colmap_reconstruction_colab.ipynb`](notebooks/colmap_reconstruction_colab.ipynb)
in Google Colab Free. It accepts a Drive folder or uploaded ZIP, verifies the 35
pilot frame hashes, installs and inspects the runtime's COLMAP build, runs a
CPU-compatible sparse pipeline, and persists every output to Drive. GPU use requires
compatible CLI flags plus a successful two-frame GPU smoke test. Conservative
fallback and the remaining three sequences are disabled until explicitly enabled
after pilot review.

## Reconstruction-grounded ML preparation (Milestone 4A)

The optional ML package is implemented but intentionally untrained. It provides a
validated CO3D sequence reader, versioned nullable JSONL records, leakage-safe
object/scene group splits, a numerical Random Forest baseline, cached frozen
ResNet-18 embeddings, a bounded mask-aware temporal LSTM, metrics, hashed artifacts,
and a heuristic-safe optional inference adapter.

No CO3D data or encoder weights were downloaded and no genuine model accuracy is
reported. Training rejects heuristic scores as targets and distinguishes unavailable
reconstruction infrastructure from a genuine failed reconstruction. See the
[ML pipeline runbook](docs/ml_training_pipeline.md) and
[ML notebook audit](docs/ml_notebook_audit.md).

After genuine reconstruction-quality targets exist, build and train the CPU
baseline with:

```bash
python -m pip install -e '.[ml-tabular]'
python scripts/ml/build_dataset.py \
  --co3d-root /path/to/co3d_subset \
  --labels-csv evaluation/ml/reconstruction_targets.csv \
  --out data/ml/co3d_dataset_v1.jsonl
python scripts/ml/train_tabular.py \
  --dataset data/ml/co3d_dataset_v1.jsonl \
  --split-manifest evaluation/ml/split_v1.json \
  --artifacts evaluation/ml/artifacts/run_001
```

For GPU-optional feature extraction and LSTM training, open
[`notebooks/ml_training_colab.ipynb`](notebooks/ml_training_colab.ipynb) in Colab
Free. Drive-backed feature caches are resumable; synthetic smoke-test artifacts are
permanently marked and refused by the production inference adapter.

## Project layout

```text
src/video_suitability/
  video.py       validation, metadata, bounded whole-video sampling
  quality.py     exposure, contrast, blur and texture measurements
  geometry.py    ORB matching, RANSAC verification, overlap/motion proxies
  viewpoint.py   independent multi-gap viewpoint-diversity evidence and category
  object_focus.py  optional box tracking, object/background feature separation
  scoring.py     declared heuristic, uncertainty, segments, recommendations
  presentation.py  tested UI view models, guidance, evaluation join, problem frames
  infer.py       stable API and JSON CLI
  synthetic.py   deterministic MP4 fixtures
  datasets/co3d.py  validated local CO3D annotations/sequences
  ml/             schemas, splits, metrics, RF, ResNet cache, LSTM, artifacts
    registration_features.py  strict eight-feature OpenCV contract
    registration_rf.py  artifact hashes, parity gate, loading and prediction
    r1_regeneration.py  executed R1 60-frame camera-path reproduction
    colab_label_import.py  evidence-gated historical Colab label adapter
    colmap_archive_audit.py  read-only real 60-frame archive inventory
scripts/generate_test_videos.py
scripts/benchmark_video_analysis.py
scripts/benchmark_viewpoint_diversity.py
scripts/benchmark_object_focused.py
scripts/benchmark_streamlit_upload.py
scripts/benchmark_registration_rf.py
scripts/validate_viewpoint_real_videos.py
scripts/compare_object_focused_validation.py
scripts/ml/                        dataset, training, Colab notebook tools
  audit_colmap_csv_archive.py  reproduce the real archive audit/manifests
  import_colab_registration_labels.py  historical output audit/import
  build_registration_training_data.py  merged-label audit and exact feature bundle
  build_experimental_registration_training_notebook.py  reproducible final Colab workflow
  executed_regeneration_adapter_template.py  deliberately unimplemented evidence adapter
  run_r1_feature_pilot.py  three-record R1 regeneration and production features
  train_historical_registration_experiment.py  offline legacy-feature grouped CV
scripts/reconstruction/        frame extraction, environment audit, sparse COLMAP runner
notebooks/colmap_reconstruction_colab.ipynb
notebooks/ml_training_colab.ipynb
notebooks/experimental_registration_training_colab.ipynb
notebooks/r1_regeneration_pilot_colab.ipynb
tests/
docs/notebook_audit.md
docs/scoring_v2_audit.md
docs/ui_mvp_audit.md
docs/performance_benchmark.md
docs/viewpoint_diversity.md
docs/object_focused_validation.md
docs/registration_random_forest.md
docs/registration_training_compatibility.md
docs/colab_label_compatibility_audit.md
docs/real_colmap_csv_dataset_audit.md
docs/experimental_registration_training.md
docs/r1_regeneration_pilot.md
docs/historical_registration_experiment.md
docs/historical_ridge_live_compatibility.md
evaluation/viewpoint_real_validation/  derived real-video CSV and audit
evaluation/object_focused_validation/  manifests and whole/object comparison
evaluation/
app.py
reference/co3d_video_quality_assessment.ipynb
STATUS.md
```

## Current limitations

- The heuristic has not yet been calibrated against independent downstream reconstruction outcomes.
- It cannot reliably separate object motion from camera motion or prove camera translation.
- Estimated viewpoint diversity does not measure exact orbit angle, prove 360°
  capture, or estimate the percentage of object surface recorded. Blur, low texture,
  repetitive/symmetric appearance, sparse geometry, and rotation-dominant motion
  intentionally produce `Unknown` when evidence is insufficient.
- Object focus is a tracked rectangle, not semantic segmentation. A loose box can
  retain background features; occlusion, specularity, blur, or large appearance
  changes can break tracking and intentionally return `Unknown`.
- Reflective, transparent, textureless, very small, or heavily occluded objects remain difficult.
- Uniform sampling can miss short defects between samples, especially in long videos. Long gaps explicitly mark shake/LK retention as unresolved.
- Four real captures have independent qualitative review, but only one has an approximate numerical score and none has a downstream reconstruction outcome.
- No additional held-out real video is currently accessible, so real-world generalization has not been tested.
- Reconstruction-grounded results are still unavailable on the audited host because
  COLMAP is absent; no registration or sparse-geometry metric has been fabricated.
- No supervised model has been trained on genuine reconstruction-quality labels;
  Milestone 4A provides label-gated infrastructure only. ML predictions remain
  unavailable until a hashed artifact records genuine target provenance.
- The requested registration Random Forest and its historical eight-feature
  preprocessing export are absent. The app refuses prediction until a final trusted
  artifact passes saved-training-example parity; the template extractor is not
  assumed equivalent to an undocumented earlier Colab procedure.

See [the viewpoint-diversity audit](docs/viewpoint_diversity.md),
[the object-focused validation audit](docs/object_focused_validation.md),
[the Random Forest integration audit](docs/registration_random_forest.md),
[the training-data compatibility workflow](docs/registration_training_compatibility.md),
[the Colab label compatibility audit](docs/colab_label_compatibility_audit.md),
[the real 60-frame CSV dataset audit](docs/real_colmap_csv_dataset_audit.md),
[the experimental registration-model training runbook](docs/experimental_registration_training.md),
[the R1 regeneration pilot](docs/r1_regeneration_pilot.md),
[the fast historical-feature experiment](docs/historical_registration_experiment.md),
[the historical Ridge live-compatibility decision](docs/historical_ridge_live_compatibility.md),
[the notebook audit](docs/notebook_audit.md),
[the v2 scoring audit](docs/scoring_v2_audit.md), and [STATUS.md](STATUS.md) for
actual executed commands and results.

