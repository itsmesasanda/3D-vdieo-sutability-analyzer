# 3D Reconstruction Video Suitability Analyzer

This repository contains a CPU-runnable MVP that analyzes a video of a mostly static physical object filmed from multiple viewpoints. It produces an explicitly **heuristic** 0–100 suitability score, five subscores, timestamped problems, capture recommendations, raw evidence, and uncertainty warnings.

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
- stationary view and pure rotation;
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
short-coverage cases, v1 preservation, JSON serialization, problem timestamps,
score bounds, gates, report-to-timeline/frame integration, evaluation missing values,
Streamlit rendering, and a real CLI subprocess writing JSON.

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
- a filterable timestamped problem timeline and representative problem frames;
- what/where/why/action guidance for blur, exposure, overlap, tracking, shake,
  excessive motion, weak viewpoint change, and other detected issues;
- technical gates, warnings, complete JSON, and report download;
- a preserved v1/v2/human-feedback evaluation view that explicitly avoids accuracy
  claims and keeps missing human numbers missing.

Uploaded content is analyzed locally. Streamlit session state caches an upload by
SHA-256 so ordinary widget reruns do not repeat the analysis. See
[`docs/ui_mvp_audit.md`](docs/ui_mvp_audit.md) for verified behavior and limitations.

## 4K performance

For the accessible 2160×3840 HEVC `IMG_3913.MOV`, two end-to-end runs improved from
a 27.518-second mean to 14.443 seconds (1.905×, 47.516% lower). Dense uniform sample
positions use a single forward decode pass; long videos with sparse positions retain
random seeking. All 32 resized frames were pixel-identical in a direct comparison,
and score, components, quality/geometry records, and problem segments matched the
preserved report. See [`docs/performance_benchmark.md`](docs/performance_benchmark.md)
and `scripts/benchmark_video_analysis.py`. These local two-run timings are not a
cross-platform guarantee.

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
  scoring.py     declared heuristic, uncertainty, segments, recommendations
  presentation.py  tested UI view models, guidance, evaluation join, problem frames
  infer.py       stable API and JSON CLI
  synthetic.py   deterministic MP4 fixtures
  datasets/co3d.py  validated local CO3D annotations/sequences
  ml/             schemas, splits, metrics, RF, ResNet cache, LSTM, artifacts
scripts/generate_test_videos.py
scripts/benchmark_video_analysis.py
scripts/ml/                        dataset, training, Colab notebook tools
scripts/reconstruction/        frame extraction, environment audit, sparse COLMAP runner
notebooks/colmap_reconstruction_colab.ipynb
notebooks/ml_training_colab.ipynb
tests/
docs/notebook_audit.md
docs/scoring_v2_audit.md
docs/ui_mvp_audit.md
docs/performance_benchmark.md
evaluation/
app.py
reference/co3d_video_quality_assessment.ipynb
STATUS.md
```

## Current limitations

- The heuristic has not yet been calibrated against independent downstream reconstruction outcomes.
- It cannot reliably separate object motion from camera motion or prove camera translation.
- Reflective, transparent, textureless, very small, or heavily occluded objects remain difficult.
- Uniform sampling can miss short defects between samples, especially in long videos. Long gaps explicitly mark shake/LK retention as unresolved.
- Four real captures have independent qualitative review, but only one has an approximate numerical score and none has a downstream reconstruction outcome.
- No additional held-out real video is currently accessible, so real-world generalization has not been tested.
- Reconstruction-grounded results are still unavailable on the audited host because
  COLMAP is absent; no registration or sparse-geometry metric has been fabricated.
- No supervised model has been trained on genuine reconstruction-quality labels;
  Milestone 4A provides label-gated infrastructure only. ML predictions remain
  unavailable until a hashed artifact records genuine target provenance.

See [the notebook audit](docs/notebook_audit.md), [the v2 scoring audit](docs/scoring_v2_audit.md), and [STATUS.md](STATUS.md) for actual executed commands and results.

