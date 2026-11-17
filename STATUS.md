# Project status

## Milestone 1 — complete (2026-11-17)

The repository now has a complete CPU-only heuristic path from local MP4 input to a JSON suitability report. It does not require CO3D, COLMAP, PyTorch, a GPU, or trained weights.

### Files added and why

- `pyproject.toml`, `requirements.txt`, `.gitignore`: reproducible packaging/dependencies and protection for local media, datasets, weights, and reports.
- `src/video_suitability/config.py`: immutable resource/scoring configuration.
- `src/video_suitability/video.py`: validation, metadata, uniform full-range sampling, timestamps, resize/memory bounds, and fallback warnings.
- `src/video_suitability/quality.py`: per-frame exposure, clipping, contrast, sharpness, gradients, entropy, and issue flags.
- `src/video_suitability/geometry.py`: ORB extraction/matching, fundamental/homography RANSAC, overlap/spatial coverage, motion, and homography-residual proxies.
- `src/video_suitability/scoring.py`: explainable fixed heuristic, five subscores, evidence caps, uncertainty, timestamped problem grouping, and capture recommendations.
- `src/video_suitability/infer.py`, `__main__.py`: stable Python API and documented JSON CLI.
- `src/video_suitability/synthetic.py`, `scripts/generate_test_videos.py`: deterministic MP4 fixtures for smooth orbit, stationary view, blur, and darkness.
- `tests/`: unit and end-to-end coverage, including a real CLI subprocess.
- `docs/notebook_audit.md`: complete audit of all 50 reference-notebook cells and stored outputs.
- `README.md`: install/run/test guide, report contract, scoring declaration, and limitations.

The user-provided `AGENTS.md` and reference notebook were preserved unchanged.

### Commands executed and actual results

Repository/notebook inspection:

```bash
pwd
rg --files -g 'AGENTS.md' -g '!**/.git/**'
git status --short
wc -l AGENTS.md
sed -n '1,260p' AGENTS.md
wc -c -l reference/co3d_video_quality_assessment.ipynb
```

Observed: only `AGENTS.md` and the 256,749-byte, 4,685-line reference notebook initially existed; both were untracked. `jq` was not installed, so a read-only Python JSON parser was used to enumerate/read all 50 cells and stored outputs. The audit findings are in `docs/notebook_audit.md`.

Initial syntax command:

```bash
python3 -m compileall -q src scripts tests
```

Observed failure: macOS Python attempted to write bytecode under `~/Library/Caches/com.apple.python/...`, which the managed filesystem denied. This was an environment write-location failure, not a source syntax error.

Corrected syntax command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache python3 -m compileall -q src scripts tests
```

Observed: success with no syntax errors.

Environment setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Observed failure: the newly created environment had pip 21.2.4, which could not perform this PEP 660 editable install.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Observed first failure inside the restricted sandbox: DNS/network access was unavailable. After the required network approval, the same command installed NumPy 2.0.2, OpenCV headless 4.14.0.94, pytest 8.4.2, and their small dependencies.

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps -e .
```

Observed: pip upgraded to 26.0.1; editable package build and installation of `video-suitability 0.1.0` succeeded.

Synthetic video generation and complete test run:

```bash
.venv/bin/python scripts/generate_test_videos.py --out-dir data/samples
.venv/bin/python -m pytest -q
```

Observed generated files:

- `orbit_good.mp4`: 265,379 bytes
- `stationary.mp4`: 62,064 bytes
- `blurry.mp4`: 34,526 bytes
- `dark.mp4`: 29,509 bytes

Observed tests: `9 passed in 1.60s`.

After adding the CLI input-overwrite guard, the final regression run was:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache .venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
```

Observed: syntax check succeeded and `10 passed in 1.74s`.

The installed console entry point was also exercised:

```bash
.venv/bin/video-suitability --video data/samples/orbit_good.mp4 --out reports/console_entrypoint.json --max-samples 12 --max-side 320
```

Observed: report schema `1.0`, score `90.82`, label `likely_suitable`, with one timestamped caution segment.

Actual CLI validation commands:

```bash
.venv/bin/python -m video_suitability.infer --video data/samples/orbit_good.mp4 --out reports/orbit_good.json --max-samples 16 --max-side 400
.venv/bin/python -m video_suitability.infer --video data/samples/stationary.mp4 --out reports/stationary.json --max-samples 16 --max-side 400
.venv/bin/python -m video_suitability.infer --video data/samples/blurry.mp4 --out reports/blurry.json --max-samples 16 --max-side 400
.venv/bin/python -m video_suitability.infer --video data/samples/dark.mp4 --out reports/dark.json --max-samples 16 --max-side 400
```

Observed results (synthetic validation only, not benchmark performance):

| Fixture | Heuristic score | Label | Expected detected issue |
|---|---:|---|---|
| orbit_good | 90.21 | `likely_suitable` | homography-dominant caution |
| stationary | 44.00 | `unlikely_suitable` | little viewpoint change |
| blurry | 61.13 | `review_recommended` | blur |
| dark | 53.72 | `review_recommended` | underexposure (plus low contrast/texture) |

All reports serialized with `allow_nan=False`; the known degraded fixtures scored below the orbit fixture and contained timestamped expected problems.

### Acceptance status

- Notebook completely audited: **yes**.
- Reusable, repaired modules extracted: **yes**.
- CPU-runnable video pipeline: **yes**.
- Explainable bounded heuristic: **yes**.
- Programmatic test MP4s: **yes**.
- Tests actually run and passing: **yes, 10/10**.
- Real MP4 CLI to JSON: **yes** (real encoded synthetic MP4s exercised through subprocess and direct CLI runs).
- Operates without CO3D: **yes**.
- Operates without GPU: **yes**.
- CNN-LSTM/data download started: **no**, intentionally.

### Blockers and next independently executable step

There is no blocker for Milestone 1. The next safe step is manual validation on one or more user-supplied real object-orbit videos, followed by the minimal Streamlit UI in Milestone 4. Milestone 2 dataset/COLMAP work remains gated on a local CO3D subset or explicit approval for a download and storage budget.

## Real-video validation stage — complete (2026-11-17)

The requested `data/real_tests/` directory was not present. A repository-wide supported-video search found the accessible test set at `data/samples/real_tests/`, containing three phone MOV files and one MP4. All four were processed without changing their contents.

### Added automation and artifacts

- `scripts/validate_real_videos.py`: recursive supported-video discovery, SHA-256 before/after verification, timed analysis, per-video JSON, a flattened/full-raw-metrics CSV, a comparison report, and a blank human-review template.
- `reports/real_tests/*.json`: four successful individual reports.
- `reports/real_tests/summary.csv`: 4 rows and 45 columns, including subscores, selected raw metrics, full raw-metric JSON, warnings, and problem segments.
- `reports/real_tests/comparison_report.md`: measured comparison and proxy-related false-positive/false-negative risks.
- `reports/real_tests/human_review_template.csv`: filename plus object, camera motion, lighting, blur, expected suitability, notes, and reviewer columns.

The report artifacts remain Git-ignored because they may encode details from private source media.

### Commands and observed results

Read-only metadata inspection found:

| Filename | Size (bytes) | Codec | FPS | Frames | Duration | Resolution |
|---|---:|---|---:|---:|---:|---|
| IMG_3913.MOV | 114,732,613 | HEVC | 59.994 | 997 | 16.618 s | 2160×3840 |
| IMG_3914.MOV | 46,175,861 | HEVC | 60.000 | 401 | 6.683 s | 2160×3840 |
| IMG_3915.MOV | 48,041,199 | HEVC | 60.000 | 415 | 6.917 s | 2160×3840 |
| orbit_good.mp4 | 265,379 | FMP4 | 12.000 | 24 | 2.000 s | 480×360 |

Pre-run verification:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache .venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
```

Observed: syntax check succeeded and `10 passed in 1.97s`.

Batch analysis:

```bash
.venv/bin/python scripts/validate_real_videos.py \
  --input-dir data/samples/real_tests \
  --output-dir reports/real_tests \
  --max-samples 32 \
  --max-side 640
```

Observed results:

| Filename | Score | Label | Analysis time | Principal detected evidence |
|---|---:|---|---:|---|
| IMG_3913.MOV | 92.82 | likely suitable | 21.395 s | low overlap on 12.90% of sampled pairs |
| IMG_3914.MOV | 85.44 | likely suitable | 21.089 s | blur, excessive jump, low overlap on 38.71% of pairs |
| IMG_3915.MOV | 73.85 | likely suitable | 21.578 s | blur, excessive jumps, insufficient correspondence, low overlap on 54.84% of pairs |
| orbit_good.mp4 | 88.57 | likely suitable | 0.368 s | homography-dominant motion on 95.65% of pairs; viewpoint-motion subscore 29.72 |

Total measured analysis time was 64.430 seconds. Four succeeded and zero failed. Every report recorded `source_unchanged_after_processing: true`; post-run SHA-256 values matched the pre-run values.

No thresholds or weights were changed. `IMG_3915.MOV` and `orbit_good.mp4` are explicitly highlighted as potential high-score false-positive risks based on weak geometry/overlap and homography-dominant low-motion evidence, respectively. The cohort has no low-score prediction, so observed false-negative behavior cannot be assessed from these files alone.

## Heuristic v2 and human-feedback audit — complete (2026-11-17)

Independent feedback is stored without imputation in `evaluation/real_video_human_review.csv`. Only `IMG_3913.MOV` has a numerical human estimate (approximately 80); the other three numerical fields are explicitly `missing_not_provided`. `evaluation/baseline_v1_manifest.csv` records source/report hashes and the four v1 outputs so later reports cannot silently replace the baseline.

### Duration verification

A complete decode of `orbit_good.mp4` observed exactly 24 frames at 12.0 FPS, first timestamp 0.0 s and last timestamp 1.916667 s. Nominal duration is therefore 2.0 seconds for the accessible file. The human recollection of roughly four seconds remains in the review CSV as separate evidence.

### V2 implementation

- Timestamp-aware forward/backward LK optical flow, track retention, normalized speed, direction/speed discontinuity, affine rotation, and residual metrics.
- Descriptor-ambiguity, rotation/planar, low-overlap, and short-cadence reliability evidence.
- Shake/LK retention is evaluated only for ≤0.30-second sample gaps; longer gaps remain visible and use ORB continuity rather than a false shake penalty.
- Source megapixels, approximate bitrate, compressed bits/pixel/frame, scale-to-analysis resolution, and actual sharpness are reported. Resolution alone cannot trigger a failure gate.
- V1 remains embedded as `baseline_v1`; top-level output is explicitly `heuristic_v2` with score/subscore deltas.
- Weighted geometric-mean v2 plus fully visible gate evaluations. Classification thresholds stayed at 70/45 and were not fitted to feedback.
- The additive JSON contract, CLI, batch reports, and a minimal optional Streamlit UI remain compatible.

Full rationale, formulas, gates, component changes, synthetic regression results, and limitations are in `docs/scoring_v2_audit.md`.

### Executed tests and observed failures

The first v2 test run found one controlled regression:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache .venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
```

Observed: `1 failed, 9 passed`. The original `orbit_good` fixture is itself homography-dominant, so it was no longer a valid smooth-orbit positive. A separate multi-layer parallax fixture was added instead of weakening the rotation gate.

The next run found that pure rotation was not always counted because the v1 homography issue required a displacement threshold: `1 failed, 11 passed`. V2 now records raw rotation/planar dominance independently while leaving v1 fields unchanged.

Final controlled suite after adding stationary, pure rotation, smooth orbit, shaky orbit, fast motion, low resolution, blur, darkness, and insufficient-coverage cases:

```text
12 passed in 4.42s
```

### Final reviewed-video comparison

Command:

```bash
.venv/bin/python scripts/validate_real_videos.py \
  --input-dir data/samples/real_tests \
  --output-dir reports/real_tests/v2 \
  --max-samples 32 \
  --max-side 640
```

| File | Baseline v1 | V2 | Change | V2 label | V2 analysis time |
|---|---:|---:|---:|---|---:|
| IMG_3913.MOV | 92.82 | 78.54 | -14.28 | likely suitable | 22.343 s |
| IMG_3914.MOV | 85.44 | 60.00 | -25.44 | review recommended | 22.651 s |
| IMG_3915.MOV | 73.85 | 48.51 | -25.34 | review recommended | 24.580 s |
| orbit_good.mp4 | 88.57 | 44.00 | -44.57 | unlikely suitable | 0.557 s |

Total final analysis time: 70.131 seconds. Four succeeded, zero failed, and source hashes were unchanged. The numerical changes are qualitatively consistent with supplied comments, but **do not establish accuracy** because the reviewed files influenced the failure-mode audit and only one has a numerical human estimate.

No additional real videos exist in the accessible workspace, so the requested held-out real-video evaluation remains blocked by data availability. Controlled synthetic tests were run, but they are not presented as a substitute for held-out real evidence. No ML model was trained and no heuristic score was used as a target.

### Final verification

An attempted editable reinstall with normal build isolation failed because the sandbox could not fetch `setuptools>=68`. Retrying offline with `--no-build-isolation` also failed because the older local setuptools lacks the PEP 660 `build_editable` hook. The already-working editable mapping was left intact; source execution was unaffected.

Final commands:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python -m pytest -q
.venv/bin/video-suitability --version
```

Observed:

```text
12 passed in 4.28s
video-suitability 0.2.0
evidence validation: 4 reviews (3 numeric missing), 4 unchanged v1 reports, 4 v2 reports
```

## Milestone 3A — reconstruction-grounded validation started (2026-11-17)

Milestone 3A is implemented through the local infrastructure boundary. The heuristic
and its reports were not changed. The four source checksums, four baseline-v1 report
checksums, and four v2 report checksums are preserved in
`evaluation/reconstruction/results.csv`.

### Host audit and decision

The final read-only audit measured a MacBook Air with an Apple M3 (8 CPU cores: 4
performance and 4 efficiency), 8 GiB RAM, and an integrated 8-core Metal GPU. CUDA
is unavailable and not required for the prepared CPU workflow. OpenCV 4.14.0 and
Homebrew 7.0.9 are present; COLMAP, ffmpeg, ffprobe, and pycolmap are absent. Free
disk was 1.820 GiB of 228.274 GiB total in the final recorded audit.

Homebrew exposes COLMAP, but the formula has a large dependency tree and the free
disk margin is insufficient for a safe install plus reconstruction artifacts. No
package installation, large download, paid runtime, or remote upload was attempted.
An exact Linux/Colab CPU workflow is documented in
`evaluation/reconstruction/COLAB_LINUX_RUNBOOK.md`.

### Files added and why

- `src/video_suitability/reconstruction.py`: immutable frame extraction, checksums,
  CPU COLMAP orchestration, version-aware GPU-disable flags, text-model parsing,
  objective sparse metrics, and OpenCV sparse previews.
- `scripts/reconstruction/`: CLIs for environment audit, extraction, COLMAP mapping,
  standalone model analysis, and evidence-table generation.
- `tests/test_reconstruction.py`: full-timeline extraction/source-preservation,
  known COLMAP-text metrics, sparse preview, and unavailable-infrastructure tests.
- `evaluation/reconstruction/environment_audit.json`: measured host evidence.
- `evaluation/reconstruction/IMG_3913/`: pilot extraction manifest/metadata and an
  explicit COLMAP-unavailable JSON. Generated JPEG inputs are Git-ignored but kept
  locally.
- `evaluation/reconstruction/results.csv`: four-row v1/v2/objective-results table;
  unavailable objective fields are blank, never zero-filled.
- `evaluation/reconstruction/human_review_form.csv`: blank, structured review fields
  for completeness, plausibility, missing regions, artifacts, and actual usability.
- `evaluation/reconstruction/REPORT.md`: measured results, commands, limitations,
  interpretation, and next gate.
- `.gitignore`: excludes generated reconstruction JPEGs, databases, sparse binaries,
  and PLY files while leaving manifests, JSON, CSV, logs, and docs trackable.

### Commands and observed results

Initial syntax and complete tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
```

Observed: syntax succeeded and `16 passed in 2.90s`.

The first audit execution failed with a `ValueError` because sandbox-denied `sysctl`
printed an error string. The audit helper was repaired to discard nonzero command
output and derive memory from `system_profiler`; its rerun succeeded:

```bash
.venv/bin/python scripts/reconstruction/audit_environment.py \
  --out evaluation/reconstruction/environment_audit.json
```

Pilot extraction:

```bash
.venv/bin/python scripts/reconstruction/extract_frames.py \
  --video data/samples/real_tests/IMG_3913.MOV \
  --output-dir evaluation/reconstruction/IMG_3913 \
  --sample-fps 2 --max-side 1600 --jpeg-quality 95
```

Observed: 35/35 frames at 900×1600, 0.000–16.602 s, temporal coverage 1.0,
11.517968 s runtime, no failed positions, source hash unchanged. Orientation auto
was enabled, the container reported 90 degrees, and a generated frame was visually
confirmed upright.

Pilot COLMAP command:

```bash
.venv/bin/python scripts/reconstruction/run_colmap.py \
  --images-dir evaluation/reconstruction/IMG_3913/images \
  --workspace-dir evaluation/reconstruction/IMG_3913/colmap \
  --camera-model OPENCV --sequential-overlap 10 --retry-exhaustive
```

Observed: exit 3 with `infrastructure_unavailable`; no COLMAP subprocess ran,
objective metrics are null, and mapper runtime is null. This is not a failed-video
label. The three remaining videos were not launched because the pilot did not pass
the requested success gate.

Evaluation consolidation:

```bash
.venv/bin/python scripts/reconstruction/build_evaluation.py
```

Observed: four results rows and one blank four-row human-review form. Preserved v2
scores are 78.54, 60.00, 48.51, and 44.00; preserved baseline v1 scores are 92.82,
85.44, 73.85, and 88.57.

Final verification reran syntax checks, all tests, evaluation consolidation, and
SHA-256 for all four videos plus all eight v1/v2 reports. The final test-only rerun
observed `16 passed in 4.55s`; all hashes matched the preserved values. `shasum` printed a harmless macOS
locale fallback warning, not a checksum or test failure.

### Current blocker and next executable step

Actual reconstruction validation needs COLMAP on a host with adequate free storage.
Run the prepared pilot on a free Linux/Colab CPU environment (no paid runtime
without approval) or free several additional GiB and explicitly approve a local
install. Only after the pilot produces a sparse model should the other three videos
be processed. No ML training has begun.

## Milestone 3B — Colab sparse-reconstruction package ready (2026-11-17)

Cloud execution has intentionally not been started by the agent. The repository now
contains a clean Google Colab notebook at
`notebooks/colmap_reconstruction_colab.ipynb`; the next action is a human opening it
in Colab Free, authorizing Drive, and running the pilot cells.

### Implemented behavior

- Supports a mounted Drive tree or one uploaded ZIP containing project source,
  frozen score evidence, and the 35 extracted `IMG_3913` frames.
- Installs COLMAP, ffmpeg, and `requirements.txt`, with setup output persisted.
- Records platform, Python, CPU count, memory, disk, COLMAP/ffmpeg versions, detected
  old/new GPU flag names, NVIDIA state, selected device, and command runtimes.
- Enables GPU only after hardware/flag detection and a successful two-frame GPU
  feature-extraction plus matching smoke test. Otherwise it selects CPU. Requiring
  an unverified GPU stops explicitly.
- Reuses `run_sparse_colmap`. Its additive `use_gpu` option preserves all existing
  CPU calls, and model analysis now exposes COLMAP quaternion, translation, and
  camera-center values for `camera_poses.csv`.
- Runs the pilot with one shared `OPENCV` camera, sequential overlap 10, and an
  exhaustive retry only when sequential mapping produces no model. Input hashes are
  verified before and after.
- Runs in temporary `/content` storage for performance, then copies each complete
  attempt—database, logs, sparse models, text export, PLY, poses, preview, and JSON—
  to a new timestamped Drive directory. Persistent JSON paths point to Drive.
- Produces a summary CSV with registered images/ratio, sparse points, mean
  reprojection error, mean track length, mapper/total runtime, failure stage/reason,
  and blank unavailable measurements. `completeness_claimed` is always false.
- Keeps CPU `SIMPLE_RADIAL` fallback off until primary diagnostics are reviewed.
  Processing the remaining three isolated sequences is also disabled and gated on a
  successful pilot.
- Copies frozen v1/v2 evidence rather than recomputing it. No scoring weight, source
  frame, video, existing report, or local reconstruction result was changed.

### Local validation actually run

The generator parses every code cell with Python's AST before writing. The generated
notebook was independently loaded as JSON and its code cells parsed again. Tests
verify clean execution state, persistence and safety gates, objective metric fields,
pose/PLY/preview outputs, and completeness cautions.

```bash
.venv/bin/python scripts/reconstruction/build_colab_notebook.py
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
```

Observed: 12-cell nbformat 4.5 notebook written successfully and final verification
reported `18 passed in 4.33s`. COLMAP was not installed locally; no dataset, paid runtime, or lengthy cloud
execution was started.

## Competition Streamlit MVP and performance revision — complete (2026-11-17)

COLMAP execution remains deferred and its code/evidence was not changed during this
stage. The production Streamlit app is now a polished analysis, inspection, capture-
guidance, and evaluation experience on top of the existing report contract.

### UI and presentation implementation

- `app.py` now has a responsive visual system, prominent circular 0–100 score,
  suitability interpretation, preserved-v1 comparison, five explained components,
  and evidence-level warnings.
- `src/video_suitability/presentation.py` builds testable view models without a
  Streamlit dependency: problem labels, reconstruction impact, recording action,
  timeline positions, representative-frame selection, and evaluation joins.
- Timestamped problem rows cover blur, under/overexposure, contrast/texture, low
  overlap, track loss, camera shake, excessive speed/jumps, weak viewpoint change,
  homography dominance, repetitive texture, and unresolved sampling cadence.
- The gallery decodes up to six deterministic representative source frames. It
  explicitly distinguishes directly visible appearance defects from motion/overlap
  evidence that applies to a surrounding frame pair.
- Recommendation cards answer what is wrong, where it occurs, why reconstruction is
  affected, and how to record differently.
- The evaluation view joins four frozen v1/v2 rows with structured human review.
  Only one numerical human score is displayed; the other three remain “Not
  provided.” It states that the reviewed set influenced the audit and has no
  reconstruction ground truth.
- Uploads are cached in Streamlit session state by SHA-256, avoiding repeated CPU
  analysis on ordinary widget reruns. Temporary upload files are removed in a
  `finally` block after report/frame extraction.

The full audit is in `docs/ui_mvp_audit.md`.

### Blurred-orbit regression investigation

The existing controlled blurred orbit reproduced the defect: with 16 samples v1 was
61.69 and v2 was 67.75 even though 100% of frames were blurred and median Laplacian
variance was 4.78. Strong coarse correspondences compensated for a very weak visual
score.

Revision 2.1 keeps all weights and 70/45 classification thresholds. It adds a
`persistent_severe_blur` cap of 55 only when at least 80% of frames are flagged and
median Laplacian variance is below 15. It also requires at least 0.90 median
homography inlier ratio before treating otherwise reliable homography evidence as a
rotation/planar failure. This avoids a dense-sampling false positive on the
multilayer-parallax fixture while retaining pure-rotation failure.

Controlled results at 16 samples: smooth orbit 93.89 (no gate), pure rotation 44.00
(rotation gate), blurred orbit 55.00 (severe-blur gate), sharp low-resolution orbit
90.51 (no gate). With default dense sampling: smooth orbit 92.71, pure rotation
35.00, blurred orbit 55.00. These are synthetic regression tests, not real-world
accuracy evidence. Existing real-video reports/evaluation CSVs were not overwritten.
The compatible `scoring_version` remains `heuristic_v2`; additive
`scoring_revision: "2.1"` identifies the repair.

Replaying the four preserved real-report metric summaries through revision 2.1
produced 78.54, 60.00, 48.51, and 44.00—delta 0.00 in every case. This was a
non-writing regression check, not a claim of accuracy.

### 4K MOV optimization and measured benchmark

Profiling `IMG_3913.MOV` showed random HEVC seeks dominated runtime. Two isolated
sampling runs took 27.264 and 26.194 seconds (mean 26.729), while quality plus
geometry took about 1.29 seconds.

For uniform samples averaging no more than 64 source frames apart, `video.py` now
uses one forward `grab()` pass and retrieves only requested frames. Wider-spaced
samples retain random seeking, and a missed forward-pass position triggers the old
seek path rather than reduced coverage.

Two end-to-end baseline runs were 29.478 and 25.558 seconds (mean 27.518). Optimized
runs were 15.103 and 13.782 seconds (mean 14.443): 1.905× speedup and 47.516% mean
runtime reduction. A direct decoder comparison confirmed the same 32 indices and
pixel-identical resized frames. Against the frozen report, score 78.54, all
subscores, all quality/geometry summaries and records, and all problem segments were
identical. Raw data is in `benchmarks/4k_mov_before_after.json`; methodology and
limitations are in `docs/performance_benchmark.md`.

### UI installation and actual smoke tests

The first optional Streamlit installation attempt failed because the restricted
sandbox had no DNS access. The approved retry installed Streamlit 1.50.0 and its
dependencies. The first server bind inside the sandbox failed with
`PermissionError: [Errno 1]`; the approved localhost-only retry started on
`127.0.0.1:8503`. `/_stcore/health` returned `ok`, the root HTML was served, and the
server stopped cleanly.

`streamlit.testing.v1.AppTest` rendered the app with no exceptions and found all
three primary tabs, one uploader, the evaluation metrics, and human-review
expanders. Pure integration tests also exercised report → timeline → guidance →
representative-frame decoding and verified the synthetic source hash stayed
unchanged.

### Final commands and observed results

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python -m pytest -q
.venv/bin/python -m video_suitability.infer \
  --video /private/tmp/v21_regression/blurry.mp4 \
  --out /private/tmp/blurry_v21_cli.json \
  --max-samples 16 --max-side 400
```

Observed: `26 passed in 6.98s`. The unchanged CLI wrote valid JSON with
`scoring_version=heuristic_v2`, `scoring_revision=2.1`, score 55.00,
`review_recommended`, and the severe-blur gate triggered. No dataset was downloaded,
no neural network was trained, and no reconstruction evidence was altered.

## Milestone 4A — ML training pipeline preparation (2026-11-17)

Status: infrastructure complete and locally verified; genuine training correctly
blocked pending reconstruction-grounded labels.

### Implemented

- `datasets/co3d.py`: strict local `.jgz`/sequence reader, safe referenced paths,
  duplicate/orphan validation, chronological frames, stable sequence IDs, metadata,
  object/scene group provenance, and whole-sequence uniform sampling.
- `ml/schema.py` and `ml/labels.py`: versioned nullable JSONL records, raw
  reconstruction/human-review fields, exact provenance, explicit synthetic gates,
  rejection of heuristic targets, separate frozen-v2.1 comparison imports, and no
  conversion of unavailable reconstruction to score zero.
- `ml/features.py`: named v2.1 numerical CV features, CO3D frame extraction with
  real annotation timing where available, missing timestamp-aware values otherwise,
  official frozen ResNet-18 embeddings, and atomic content/configuration caches.
- `ml/splits.py`: deterministic physical-object/scene group splits with a full
  dataset fingerprint and train/validation/test leakage checks.
- `ml/tabular.py`, `ml/temporal.py`, and `ml/metrics.py`: train-only preprocessing,
  Random Forest baseline, bounded mask-aware temporal LSTM, validation selection,
  MAE/RMSE/Spearman and support-aware classification metrics, heuristic v2.1 and
  train-mean comparisons, checkpoints, and synthetic/genuine scope separation.
- `ml/artifacts.py` and `ml/inference.py`: SHA-256 model bundles plus guarded
  Random-Forest and temporal adapters. The heuristic score remains authoritative;
  missing, corrupt, unsupported, or synthetic model artifacts degrade explicitly to
  unavailable.
- `scripts/ml/`: dataset builder, genuine-only tabular training CLI, and reproducible
  10-cell Colab notebook generator. `notebooks/ml_training_colab.ipynb` checks
  Drive/dataset paths, runtime memory/disk/CUDA/library versions, label provenance,
  frozen group splits, resumable Drive embedding caches, persistent checkpoints,
  and artifact inventories.
- `evaluation/ml/`: blank reconstruction-target, verified object/scene group-map,
  and heuristic-v2.1 baseline CSV templates. No evaluation evidence was overwritten.
- `docs/ml_notebook_audit.md` and `docs/ml_training_pipeline.md`: notebook reuse/
  rejection decisions, schemas, commands, exact label inputs, storage/hardware
  planning, Colab procedure, limitations, and integration contract.

### Actual local verification

The project virtual environment intentionally has only core/UI dependencies. The
full suite therefore skips optional scikit-learn/PyTorch tests there, then those
tests are run separately against already installed user-site scikit-learn 1.6.1 and
PyTorch 2.8.0. No package or model-weight download was needed.

```bash
.venv/bin/python scripts/ml/build_colab_notebook.py
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python -m pytest -q
PYTHONPATH=src:/Users/sasanda/Library/Python/3.9/lib/python/site-packages \
  PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m pytest -q tests/test_ml_training.py
```

Observed in the final pass: the notebook generator wrote a clean nbformat 4.5
notebook with 10 cells and every code cell parsed with `ast`; the core environment
reported `40 passed, 3 skipped in 5.77s`; the optional ML run reported `3 passed in
4.10s`. The optional run actually fit/exported a small Random
Forest and two-epoch CPU LSTM on explicitly synthetic targets, verified bounded
outputs/masks/checkpoints/hashes, and confirmed both artifacts were refused by the
production adapter. Those ephemeral scores are not recorded as accuracy.

The host had 438 MiB free at the dependency audit and 390 MiB at final inventory;
torchvision was unavailable in the
tested environments, and no safe local ResNet weight extraction was attempted.
Embedding cache behavior and interfaces are unit-tested; real encoder extraction is
deferred to the prepared Colab GPU/CPU workflow. No CO3D data, ImageNet weights, or
genuine model checkpoint was downloaded or created. Scoring, UI, real reports, and
reconstruction outputs were not modified.

The unchanged CLI was also run on `data/samples/good_orbit.mp4` with 8 samples at a
320-pixel analysis side. It wrote valid JSON to `/private/tmp/m4a_cli_report.json`
with score 45.32, `scoring_version=heuristic_v2`, and `scoring_revision=2.1`. This is
a compatibility smoke result under reduced sampling, not a new accuracy claim or a
replacement for a preserved report.

### Exact start condition for genuine training

Provide the approved local/Drive CO3D subset, a one-record-per-sequence JSONL built
from it, verified object/scene grouping, and a CSV of independently defined 0–100
reconstruction-quality targets with protocol version, source artifact/hash,
reconstruction status/raw metrics, and human completeness/plausibility/usability
review where available. Blank/unavailable rows remain non-training records. A
predeclared binary threshold and frozen v2.1 predictions are optional comparison
inputs, not supervision. See `docs/ml_training_pipeline.md` for the exact columns and
commands.

