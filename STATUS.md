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

## Estimated Viewpoint Diversity v1 — complete (2026-11-18)

Status: implemented as an additive CPU-only result. The established heuristic v1,
heuristic v2.1, and optional Random Forest code paths were not changed. No COLMAP,
training, labels, dataset, network service, or GPU was used.

### Implementation

- `src/video_suitability/viewpoint.py` adds bounded nearby/quarter-gap/half-gap/
  endpoint evidence, repeated-view detection, conservative parallax-like evidence,
  rotation/planar ambiguity handling, and Low/Moderate/High/Unknown categories.
- ORB detection is now a reusable public helper in `geometry.py`. One feature cache
  feeds both the unchanged adjacent geometry analysis and the new module. The
  existing pair matcher/RANSAC path is reused for at most 15 temporally separated
  pairs, with expensive LK optical flow disabled only for those new long-gap pairs.
- Optional calibrated intrinsics are accepted through the Python API and CLI. They
  are scaled from their supplied image dimensions to the analysis frame, then used
  for essential-matrix RANSAC, relative rotation, unit translation direction, and a
  rotation-compensated bearing-change indicator. Translation scale remains
  explicitly unavailable.
- Blur, low texture, sparse verified geometry, descriptor ambiguity consistent with
  repetitive/symmetric appearance, and homography-dominant changed pairs cause
  `Unknown`/insufficient evidence instead of a viewpoint claim.
- The JSON `viewpoint_diversity` block records `affects_overall_score: false`,
  `trained_model_used: false`, and `colmap_run: false`, plus raw pair evidence,
  repeated intervals, warnings, and actions. It explicitly rejects exact orbit-angle,
  360-degree, and object-surface-coverage interpretations.
- `app.py` shows a separate Estimated Viewpoint Diversity panel and expandable
  filming guidance in the Overview tab. `presentation.py` keeps this view model
  testable without Streamlit.
- `synthetic.py` adds deterministic front-only, quarter-sweep, and periodic
  full-sweep videos. These layered 2D fixtures test gates and plumbing, not real-world
  accuracy.
- `scripts/benchmark_viewpoint_diversity.py`,
  `benchmarks/viewpoint_diversity_synthetic.json`, and
  `docs/viewpoint_diversity.md` provide a reproducible scenario/timing audit.

Score isolation is covered by replacing the complete viewpoint function in a test
and confirming that overall score, all five subscores, and the v1 baseline remain
identical. No file under `src/video_suitability/ml/` changed.

### Controlled scenario outcomes and CPU timing

With 16 uniformly sampled frames at a maximum 400-pixel side, all five declared
synthetic expectations matched:

| Scenario | Expected | Observed | Wall seconds | Process CPU seconds |
|---|---|---|---:|---:|
| Front-only | Low | Low | 0.566664 | 0.789746 |
| Quarter-orbit proxy | Moderate | Moderate | 0.360235 | 0.772738 |
| Periodic full-orbit proxy | High | High | 0.422247 | 0.848246 |
| Rotation-only | Unknown | Unknown | 0.355103 | 0.773124 |
| Blurry orbit | Unknown | Unknown | 0.188688 | 0.329715 |

The complete five-video end-to-end runs took 1.892937 seconds wall and 3.513569
seconds process CPU on macOS 14.5 arm64 with Python 3.9.6 and OpenCV 4.14.0. With
decoded frames, quality, the shared ORB cache, and adjacent geometry already
available, the new module itself took 0.349794 seconds wall and 0.833215 seconds
process CPU across all five videos. OpenCV used multiple CPU threads, so process CPU
exceeds wall time. This single local run is a resource check, not a cross-platform
performance guarantee or accuracy benchmark.

### Commands actually run and observed results

Before implementation:

```bash
.venv/bin/python -m pytest -q
```

Observed: `40 passed, 3 skipped in 6.01s`; skips were the already-missing optional
scikit-learn/PyTorch dependencies.

Final validation:

```bash
.venv/bin/python scripts/benchmark_viewpoint_diversity.py \
  --out benchmarks/viewpoint_diversity_synthetic.json
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python -m pytest -q
```

Observed: all five benchmark expectations matched; compileall exited 0 without
output; the final pytest pass reported `46 passed, 3 skipped in 7.16s`. A focused
viewpoint, geometry, and integration run reported `15 passed in 9.12s`; the
viewpoint-only suite reported `6 passed in 5.41s`.

The CLI/intrinsics smoke test generated the deterministic videos under
`/private/tmp/viewpoint_cli_v1` and ran:

```bash
.venv/bin/python -m video_suitability.infer \
  --video /private/tmp/viewpoint_cli_v1/full_orbit.mp4 \
  --out /private/tmp/viewpoint_cli_v1/full_orbit_report.json \
  --max-samples 16 --max-side 320 \
  --camera-fx 420 --camera-fy 420 --camera-cx 240 --camera-cy 180 \
  --intrinsics-width 480 --intrinsics-height 360
```

Observed: valid heuristic v2/revision 2.1 JSON, unchanged score path at 91.72,
viewpoint category `High`, 13 separated pairs, 13 relative-motion records, median
rotation-compensated bearing change 1.793203 degrees, `intrinsics_used: true`,
`affects_overall_score: false`, and `colmap_run: false`.

## Real-world viewpoint validation preparation — complete (2026-11-18)

Status: the folder validator, CSV audit, feature/motion diagnostics, real-sample run,
and conservative ambiguity repair are complete. No ML training, COLMAP execution,
score change, or guessed camera calibration occurred.

### Added validation workflow

- `scripts/validate_viewpoint_real_videos.py` recursively evaluates supported local
  videos with the existing `analyze_video` API. It preserves pre/post SHA-256,
  records separate analysis/diagnostic/total wall and process CPU times, supports an
  optional filename-keyed intrinsics JSON, and continues across per-file failures.
- `src/video_suitability/viewpoint_diagnostics.py` selects repeated, planar/rotation,
  parallax-proxy, largest-change, and weakest-geometry pairs. JPEG overlays show
  RANSAC-verified matches in green, rejected ratio matches in gray, and raw
  image-motion arrows in yellow with an explicit “not proof of translation” legend.
- `evaluation/viewpoint_real_validation/validation.csv` contains filename,
  viewpoint category/evidence status, unchanged suitability score, all requested
  match/parallax/repeated-view indicators, risk flags, warnings, diagnostic paths,
  and actual CPU/wall timings.
- `evaluation/viewpoint_real_validation/REPORT.md` records the cohort limits,
  results, independent capture-quality context, per-file misclassification review,
  and local diagnostic links. Full JSON and diagnostic images are locally available
  but ignored because they reveal source-media content.
- `.gitignore`, `README.md`, and `docs/viewpoint_diversity.md` document the workflow,
  privacy boundary, optional-intrinsics risk, and non-accuracy interpretation.
- `tests/test_real_viewpoint_validation.py` exercises a two-video folder run, CSV
  schema, nonzero CPU timings, readable diagnostic JPEGs, source-hash preservation,
  reporting language, and misleading-risk flags.

### Real inputs found and first-pass audit

The historical `data/samples/real_tests` folder contains three real phone HEVC MOV
captures and one synthetic MP4 regression fixture. The first validation pass labeled
all three phone captures `High`. Inspection of the generated overlays showed matches
spanning the patterned tablecloth, chair slats, background, and an approximately
symmetric dark object. Median separated-pair verified matches were only 10–11, while
median strict cross-view descriptor ambiguity was 0.786–0.805.

The viewpoint-only evidence gate now returns `Unknown` when the median strict
separated-pair ambiguity is at least 0.65. This metric treats close second-best ORB
descriptors and exact ties as ambiguity. The preserved scoring ambiguity metric,
heuristic score formula, thresholds, and all ML files remain unchanged. Controlled
front-only, quarter-orbit, full-orbit, rotation-only, and blur outcomes still match
their declared expectations.

### Final real-folder results

The final run used 32 samples, a 640-pixel maximum analysis side, and up to three
diagnostic pairs. No intrinsics were available or inferred.

| Video | Viewpoint result | Evidence | Nearby / separated median verified matches | Parallax proxy fraction | Repeated pairs | Strict ambiguity | Total process CPU (s) | Total wall (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IMG_3913.MOV | Unknown | insufficient | 76 / 11 | 0.307692 | 0 | 0.805333 | 154.904683 | 68.200729 |
| IMG_3914.MOV | Unknown | insufficient | 42 / 11 | 0.230769 | 0 | 0.786000 | 70.396004 | 19.159773 |
| IMG_3915.MOV | Unknown | insufficient | 17 / 10 | 0.230769 | 0 | 0.804000 | 73.194364 | 49.983723 |
| orbit_good.mp4 | Moderate | sufficient | 917 / 456 | 1.000000 | 0 | 0.372667 | 1.587537 | 19.020297 |

OpenCV used multiple threads, so summed process CPU exceeds wall time. CPU totals
include source hashing, the production analysis, diagnostic re-sampling/rendering,
and post-run hash verification. Wall time varied substantially across repeated runs
under local system load; the table preserves the final actual run rather than an
average. All four source hashes remained unchanged. The parallax fraction in this
table is long-gap-only; the CSV also retains the all-pair fraction separately.

The three phone `Unknown` results do not mean low diversity. `IMG_3913` may be a
conservative false negative because the independent review calls it good, but its
long-gap evidence is too ambiguous for the current object-agnostic method.
`IMG_3914` and `IMG_3915` also have independent excessive-speed/shake concerns and
selected long-gap pairs with only 4–5 verified matches. `orbit_good.mp4` is not
real-world evidence: it is a synthetic fixture whose textured background supplies
hundreds of matches. Its `Moderate` diversity result does not contradict the “very
poor” capture-quality review or its unchanged suitability score of 44.

The production suitability values were compared with the frozen v2.1 evidence and
matched exactly: 78.54, 60.00, 48.51, and 44.00. Random Forest and ML files were not
modified.

### Commands and observed results

Focused validation tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_real_viewpoint_validation.py tests/test_viewpoint.py
```

Observed before the final full-suite pass: `8 passed in 7.26s`.

Real-folder command:

```bash
.venv/bin/python scripts/validate_viewpoint_real_videos.py \
  --input-dir data/samples/real_tests \
  --output-dir evaluation/viewpoint_real_validation \
  --max-samples 32 --max-side 640 --max-diagnostic-pairs 3 \
  --review-csv evaluation/real_video_human_review.csv
```

Observed: four succeeded, zero failed. The source-score comparison printed
`unchanged True` against `evaluation/reconstruction/results.csv`.

Final verification:

```bash
git diff --check
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python -m pytest -q
```

Observed: diff check and compileall exited 0 without output; pytest reported
`48 passed, 3 skipped in 12.20s`. The three skips are the unchanged optional
scikit-learn/PyTorch tests in the core environment.

### Remaining validation boundary

There is still no independently assigned viewpoint-diversity band or downstream
reconstruction result for the real MOV files, so accuracy, precision, recall, and a
misclassification rate cannot be reported. The next independent step is a held-out
set with object masks or object-focused review, verified static-subject status,
calibrated/crop-correct intrinsics where available, and viewpoint labels assigned
without seeing this module's predictions.

## Object-focused viewpoint validation revision — complete (2026-11-18)

Status: optional object selection, conservative tracking, object/background evidence
separation, decision explanations, live-path profiling, controlled comparisons, and
the accessible real-MOV audit are implemented. No score formula, threshold, Random
Forest/ML file, training path, or COLMAP live behavior changed.

### Files and behavior added

- `src/video_suitability/object_focus.py` validates a normalized middle-frame box,
  propagates it forward/backward with in-box ORB matches plus affine RANSAC, splits
  the shared ORB cache into object/background features, runs object-only geometry,
  and retains background-only comparisons. Tracking below 70% or background
  dominance on at least 60% of evaluable long-gap pairs forces `Unknown`.
- Failed tracking stops propagation in that direction. Pairs without two tracked
  boxes mark background dominance unevaluable instead of reporting a misleading
  zero.
- `viewpoint.py` now emits `decision_explanation` with the primary reason, named
  rule, supporting/limiting evidence, and all insufficiency reasons. Match scarcity,
  descriptor repetition/symmetry, homography/pure-rotation dominance, inadequate
  parallax/path extent, blur/low texture, tracking loss, and background dominance
  are distinct.
- `infer.py` accepts `object_bbox` and optional performance profiling. It extracts
  ORB once, reuses it for the unchanged adjacent-score geometry and object focus,
  and limits the object live path to nine long-gap comparisons. The main score
  always uses the original whole-frame geometry.
- `app.py` provides a middle-frame box preview, explicit Analyze button, cache key
  comprising upload hash plus box, a separate decision explanation, and deferred
  problem-frame rendering. `_analyze_upload_payload` is shared with the benchmark so
  the measured route is the route used by Streamlit.
- `scripts/compare_object_focused_validation.py` evaluates whole/object modes from a
  manifest and writes categories, evidence, tracking/background diagnostics,
  separate CPU/wall timings, JSON reports, and a score-invariance check.
- `evaluation/object_focused_validation/physical_object_manifest_template.csv`
  prepares front-only, quarter-orbit, full-orbit, rotation-only, and blurred
  recordings of one physical object. `--require-complete-scenario-set` rejects an
  incomplete set or mixed object IDs.
- `scripts/benchmark_object_focused.py` and
  `benchmarks/object_focused_synthetic_comparison.json` record whole/object
  controlled evidence and timings. `scripts/benchmark_streamlit_upload.py` and the
  two `streamlit_upload_real_*.json` files split upload-to-result from diagnostics.
- `tests/test_object_focus.py` covers box validation, the five scenarios, score-path
  identity, separate object/background evidence, tracking failure, explanation
  modes, and production-stage profiling.
- `docs/object_focused_validation.md`, `docs/viewpoint_diversity.md`, and `README.md`
  document the API/UI, validation protocol, measured outcomes, non-claims, and
  remaining risks.

### Controlled whole/object results

The recorded run used 16 samples, a 400-pixel maximum side, and normalized object
box `[0.25, 0.10, 0.50, 0.80]`:

| Scenario | Whole | Object | Track fraction | Median separated verified matches whole/object | Parallax fraction whole/object | Wall seconds whole/object |
|---|---|---|---:|---:|---:|---:|
| front-only | Low | Low | 1.0000 | 1268 / 1143 | 0.000000 / 0.000000 | 3.932140 / 5.027871 |
| quarter-orbit proxy | Moderate | Moderate | 1.0000 | 561 / 524 | 0.923077 / 1.000000 | 3.648086 / 5.142417 |
| periodic full-orbit proxy | High | Moderate | 0.9375 | 426 / 432 | 0.923077 / 0.333333 | 2.751614 / 4.214960 |
| pure rotation | Unknown | Unknown | 1.0000 | 827 / 769 | 0.000000 / 0.000000 | 5.851167 / 3.526223 |
| blurred | Unknown | Unknown | 1.0000 | 29 / 29 | 1.000000 / 1.000000 | 0.338617 / 0.584740 |

The object-focused full-orbit proxy remains `Moderate`, rather than weakening the
High rule to fit the fixture name: only one third of evaluated object long-gap pairs
retain the parallax proxy and tracking misses one endpoint. The blurred proxy remains
`Unknown` even though its surviving match pairs have parallax-like evidence, because
the blur gate makes those correspondences untrustworthy. All five overall scores,
subscores, and baseline-v1 blocks were exactly identical between modes. These are
layered 2D regressions, not physical-object evidence or an accuracy estimate.

### Accessible real-video comparison

The existing folder does not contain the required controlled five-video physical
protocol. It contains three unlabeled MOV recordings of the same bottle. Manually
reviewed middle-frame boxes are stored in `real_samples_manifest.csv`. The comparison
used 16 samples at 480 pixels:

| Video | Whole -> object | Track fraction | Median verified matches whole/object | CPU seconds whole/object | Wall seconds whole/object | Main score unchanged |
|---|---|---:|---:|---:|---:|---|
| IMG_3913.MOV | Unknown -> Unknown | 0.0625 | 11 / 0 | 65.595767 / 65.093366 | 16.462443 / 14.965045 | yes, 60.75 |
| IMG_3914.MOV | Unknown -> Unknown | 0.0625 | 5 / 0 | 36.171592 / 36.405832 | 8.649205 / 25.018755 | yes, 51.26 |
| IMG_3915.MOV | Unknown -> Unknown | 0.0625 | 11 / 0 | 36.364412 / 35.443833 | 48.550726 / 6.992575 | yes, 49.94 |

All object tracks stop on the first propagation step because the cropped/dark bottle
regions are blurred or too low-texture for reliable consensus. Object long-gap
verified matches are therefore zero and background dominance is `n/a`, because no
pair has two tracked boxes. Whole-frame primary reasons are insufficient matches or
high cross-view ambiguity; object mode more specifically reports low object texture
and unreliable tracking. These Unknown results are intentional insufficient
evidence, not proof of Low diversity. No independent viewpoint bands or downstream
reconstructions exist, so no misclassification rate or real-world accuracy is
reported.

The first batch comparison attempt encountered an OpenCV/FFmpeg 33-second timeout
and `moov atom not found` while reopening `IMG_3913.MOV`; the runner recorded a
decode failure instead of turning it into geometric evidence. An isolated retry
succeeded, and the final three-row comparison then completed successfully. This
intermittent decoder behavior remains a deployment risk worth monitoring.
The final offline runner permits exactly one bounded retry per analysis and records
the retry count; `IMG_3915` whole-frame validation used one retry, which is included
in its 48.550726-second wall time. Live Streamlit analysis still fails promptly and
does not silently extend upload latency.

### Exact Streamlit-path performance

Commands:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_streamlit_upload.py \
  --video data/samples/real_tests/IMG_3914.MOV \
  --out benchmarks/streamlit_upload_real_whole.json --runs 1
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_streamlit_upload.py \
  --video data/samples/real_tests/IMG_3914.MOV \
  --out benchmarks/streamlit_upload_real_object.json --runs 1 \
  --object-bbox 0.0,0.30,0.43,0.69
```

Production-default observed timings:

| Mode | Upload-to-result wall / CPU | Decode wall / CPU | ORB wall | Adjacent geometry wall | Viewpoint wall | Deferred diagnostics wall / CPU |
|---|---:|---:|---:|---:|---:|---:|
| whole | 8.779270 / 36.836386 | 6.859514 / 33.833920 | 0.369364 | 0.973989 | 0.361231 | 4.405434 / 23.829048 |
| object | 9.151934 / 38.103702 | 7.127390 / 34.107377 | 0.359770 | 1.015382 | 0.464500 | 5.634101 / 25.983966 |

Both source hashes remained unchanged and both scores were the established 60.00.
Diagnostic rendering was not part of upload-to-result. Decoding is the clear
bottleneck. ORB was extracted once; object focus added under 0.5 seconds wall time
and did not duplicate whole-frame long-gap classification. Process CPU exceeds wall
time because OpenCV/FFmpeg uses multiple threads. These are single-host runs, not a
cross-platform latency guarantee.

### Commands, failures, and final verification

During verification, bare `python` was unavailable (`command not found`); all valid
project commands use `.venv/bin/python`. The first compile attempt also tried the
protected macOS bytecode-cache location and produced `PermissionError`; setting a
task-specific `PYTHONPYCACHEPREFIX` resolved it. A targeted command referenced a
nonexistent `tests/test_app.py`, then the actual Streamlit coverage was confirmed in
`tests/test_presentation.py`.

Affected-area tests:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_object_focus.py tests/test_viewpoint.py \
  tests/test_integration.py tests/test_presentation.py
```

Observed: `27 passed in 13.67s` on the final affected-area pass.

Real and controlled comparison commands:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/compare_object_focused_validation.py \
  --input-dir data/samples/real_tests \
  --manifest evaluation/object_focused_validation/real_samples_manifest.csv \
  --output-dir evaluation/object_focused_validation \
  --max-samples 16 --max-side 480
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_object_focused.py \
  --out benchmarks/object_focused_synthetic_comparison.json \
  --max-samples 16 --max-side 400
```

Final verification:

```bash
git diff --check
VIEWPOINT_PYCACHE=/private/tmp/viewpoint_pycache \
  PYTHONPYCACHEPREFIX="$VIEWPOINT_PYCACHE" \
  .venv/bin/python -m compileall -q app.py src scripts tests
PYTHONPATH=src:. .venv/bin/pytest -q
```

Observed: diff check and compileall exited 0 without output. Pytest reported
`56 passed, 3 skipped in 15.61s`; the three skips are the unchanged optional
scikit-learn/PyTorch tests missing from the core environment. Streamlit's `AppTest`
ran as part of the passing suite. No GPU, network, COLMAP, dataset, or model training
was used.

Final CLI smoke test:

```bash
PYTHONPATH=src:. .venv/bin/python -m video_suitability.infer \
  --video data/samples/real_tests/orbit_good.mp4 \
  --out /private/tmp/object_focus_cli_report.json \
  --max-samples 8 --max-side 240 \
  --object-bbox 0.25,0.10,0.50,0.80 --profile
```

Observed: exit 0, established suitability score 44.00, separate object-focused
viewpoint `Moderate`, tracking fraction 1.0, and all six expected production timing
stages in the JSON.

## Random Forest Streamlit integration readiness — complete, artifact pending (2026-11-18)

The Streamlit and opt-in CLI paths now support a separately displayed Random Forest
estimate of COLMAP registered-image percentage. The existing heuristic score,
subscores, baseline v1, scoring gates, and experimental viewpoint category are not
inputs to or modified by this result. Live inference does not invoke COLMAP.

### Artifact and preprocessing audit

A repository-wide inspection of tracked, ignored, and historical files found no
saved registration Random Forest (`.joblib`/`.pkl`), no export of the historical
eight-feature preprocessing, and no saved training examples with expected feature
vectors. `models/` contains only `.gitkeep`. The reference notebook contains a
different 12-feature temporal/CNN experiment, and the newer generic tabular
infrastructure uses a variable-width `cv_numeric_features/2.1` schema. Neither was
silently substituted. No model was retrained on the existing 45 labels.

Consequently, the current UI truthfully reports the registration prediction as
`unavailable`. A proposed, fully specified eight-feature OpenCV contract is provided
for the final retraining run, but is not claimed to reproduce an undocumented Colab
run. A deployable artifact must include a checksum-protected feature specification
and at least one checksum-protected training video plus its saved eight-feature
vector. The application re-extracts those features and refuses to load joblib if
parity fails.

### Files and behavior added

- `src/video_suitability/ml/registration_features.py`: deterministic versioned
  eight-feature sampling, resize, grayscale, ORB/KNN, homography-RANSAC, formula,
  aggregation, and feature-order contract.
- `src/video_suitability/ml/registration_rf.py`: strict target/training-provenance
  checks, SHA-256 manifest validation, saved-training parity gate, lazy trusted
  joblib loading, bounded percentage prediction, graceful fallback, timings, and a
  content-sensitive deployment-artifact cache identity.
- `app.py` and `presentation.py`: separate registration-result panel and technical
  timing output. Missing/incompatible artifacts do not prevent the heuristic or
  viewpoint results from rendering.
- `infer.py`: optional `--registration-artifact`; the default heuristic API and CLI
  output remain unchanged unless explicitly requested.
- `scripts/ml/package_registration_rf.py`: validates and hashes the final artifact
  without training it. Replacing a versioned artifact directory is sufficient for
  the next labeling/retraining batch.
- `scripts/benchmark_registration_rf.py`: CPU feature-only or complete-artifact
  benchmark with before/after video hashes.
- `evaluation/ml/registration_rf_artifact_template/`: proposed final-training
  feature spec and parity schema; it contains no model.
- `tests/test_registration_rf.py`: exact feature order/determinism, missing artifact,
  replacement identity, successful parity-gated prediction, parity refusal before
  joblib loading, score isolation, presentation, and actual upload-path fallback.
- `pyproject.toml`: an `app-ml` deployment extra for Streamlit, scikit-learn, and
  joblib; model dependencies remain optional for the core heuristic.
- `docs/registration_random_forest.md` and `README.md`: artifact handoff, replacement,
  failure modes, semantics, benchmark, and deployment runbook.

### Commands and observed results

The first syntax pass attempted to write bytecode to the protected macOS cache and
failed with `PermissionError`; no test was executed by that chained command. The
corrected sandbox-safe check was:

```bash
PYTHONPYCACHEPREFIX=/tmp/video-suitability-pycache \
  PYTHONPATH=src:. .venv/bin/python -m compileall -q app.py src scripts tests
PYTHONPATH=src:. .venv/bin/python scripts/ml/package_registration_rf.py --help
git diff --check
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_registration_rf.py tests/test_presentation.py tests/test_integration.py
```

Observed: compile and diff checks exited 0; the packager help rendered; focused tests
reported `20 passed in 11.94s`.

Full regression command:

```bash
PYTHONPATH=src:. .venv/bin/pytest -q
```

The first full run observed `63 passed, 3 skipped in 16.66s`. After adding the final
declared-feature-order guard and its regression test, the final compile/diff/targeted/
full sequence observed `8 passed in 4.32s` for the registration file and
`64 passed, 3 skipped in 25.04s` repository-wide. The skips are the unchanged
optional scikit-learn/PyTorch training tests because those dependencies are not
installed in the core environment. Streamlit coverage ran as part of the passing
suite. No GPU, network, dataset download, COLMAP execution, or model training was
used.

Feature-only CPU benchmark commands used the proposed final-training contract:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_registration_rf.py \
  --video data/samples/real_tests/orbit_good.mp4 \
  --feature-spec evaluation/ml/registration_rf_artifact_template/feature_extractor.json \
  --out benchmarks/registration_rf_cpu_untrained.json --runs 3
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_registration_rf.py \
  --video data/samples/real_tests/IMG_3913.MOV \
  --feature-spec evaluation/ml/registration_rf_artifact_template/feature_extractor.json \
  --out benchmarks/registration_rf_cpu_real_mov_untrained.json --runs 3
```

Observed means:

| Input | Samples / pairs | Wall seconds | Process CPU seconds | Deterministic | Source unchanged |
|---|---:|---:|---:|---|---|
| `orbit_good.mp4` | 16 / 15 | 0.260569 | 0.347153 | yes | yes |
| `IMG_3913.MOV` (4K HEVC) | 16 / 15 | 12.873245 | 59.008896 | yes | yes |

These are feature-extraction timings, not trained-model timings or accuracy results.
The real MOV result shows sparse HEVC decoding is the dominant cost and must be
reassessed with the final artifact before deployment. The remaining blocker is the
trusted final model bundle and saved training parity cases from the final labeling
run.

## Final RF training-data compatibility workflow — ready, no model trained (2026-11-18)

The repository now has a Colab-compatible preparation path for merging final COLMAP
label batches, validating label/input provenance, extracting the production eight
features from the exact 60 labeled input images, defining leakage-safe evaluation,
and packaging later-loader parity evidence. It contains no estimator fitting or
activation step and does not modify any heuristic or viewpoint score.

### Production extractor and parity changes

`registration_features.py` now has two input adapters feeding one shared feature
computation. The live video adapter is unchanged in meaning. The new COLMAP-frame
adapter verifies an ordered 60-row manifest, chronological source indices, every
image's presence and SHA-256, then uniformly selects the configured 16 positions and
runs the exact same fixed resize, grayscale, ORB, KNN ratio, homography RANSAC,
formula, aggregation, and feature ordering as production.

The existing model loader and final packager remain backward-compatible with video
parity cases and now also accept `colmap_frame_manifest` parity cases. A manifest
case preserves all 60 exact labeled input images; the loader re-extracts the saved
training vector before joblib can load.

### Training-data bundle workflow

Added:

- `src/video_suitability/ml/registration_training.py`: multi-batch merge, hashed
  label/result/input validation, fresh feature extraction, exclusions and
  incompatibility reporting, grouped folds, holdout isolation, parity copying, and
  a checksum-protected versioned bundle manifest.
- `scripts/ml/build_registration_training_data.py`: dependency-light Colab/CLI
  entry point. Repeating `--labels-csv` merges batches; it exits 2 when the audit is
  written but training readiness gates fail.
- `evaluation/ml/registration_label_batch_template.csv`: required batch schema for
  physical `source_group`, capture `variant`, predeclared evaluation role, label
  provenance, reconstruction evidence, exact frame inputs, fidelity, and canonical
  sampling-configuration hash.
- `docs/registration_training_compatibility.md`: Colab commands, schema, acceptance
  rules, outputs, split policy, and later training handoff.
- `tests/test_registration_training_data.py`: exact 60-frame determinism/hash gate,
  missing-frame refusal, two-batch merge, measured success, explicitly verified
  mapper failure, infrastructure exclusion, source-group folds, independent
  holdout, loader parity, and preprocessing mismatch reporting.

Successful labels must reconcile their percentage with explicit counts and the
hashed COLMAP objective metrics. A reconstruction failure is accepted as 0% only
when the row explicitly says 0/60, `failure_verified=true`, the hashed result says
`reconstruction_failed`, actual COLMAP commands are recorded, and the stage is a
mapper/no-sparse-model stage. Infrastructure/unavailable/not-run rows are reported
as exclusions and never receive zero or fabricated features. Missing frames, bad
hashes, duplicate records, sampling mismatches, or group leakage make the bundle
not ready.

All variants from a physical `source_group` remain in one development fold.
`independent_holdout` groups are predeclared, cannot overlap development groups,
receive no CV fold, and are explicitly marked excluded from model selection. The
output separates `development_records.csv` and `independent_holdout_records.csv` and
stores a dataset fingerprint.

The generated `training_bundle_manifest.json` records
`contains_trained_model=false` and `activates_production_inference=false`. No
`model.joblib` or production `artifact_manifest.json` is created.

### Current real-data readiness

The accessible repository has no completed merged label batch. Its only frame
manifest has 35 data rows, and the corresponding extraction metadata says 35
requested/extracted frames. Its COLMAP result is `infrastructure_unavailable`.
Therefore it is intentionally ineligible for this 60-frame training workflow and no
real training bundle or numerical target was fabricated from it.

Audit command and observed result:

```bash
find evaluation -name 'frames_manifest.csv' -print -exec wc -l {} \;
find evaluation -type f \( -iname '*label*.csv' -o -iname '*target*.csv' \) -print
rg -n '"status":|"extracted_frame_count":|"requested_frame_count":' \
  evaluation/reconstruction/IMG_3913/colmap/reconstruction_result.json \
  evaluation/reconstruction/IMG_3913/extraction_metadata.json
```

Observed: one 36-line manifest (header plus 35 frames), no populated merged label
CSV, requested/extracted counts of 35, and status `infrastructure_unavailable`.

### Commands and verification

Initial affected-area regression after sharing the production computation:

```bash
PYTHONPYCACHEPREFIX=/tmp/video-suitability-pycache PYTHONPATH=src:. \
  .venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_registration_training_data.py tests/test_registration_rf.py
```

The first run exposed that sorted JSON serialization changed parity feature-key
order. The writer was corrected to preserve the production order; the rerun reported
`11 passed in 4.85s`.

Final verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/video-suitability-pycache PYTHONPATH=src:. \
  .venv/bin/python -m compileall -q app.py src scripts tests
PYTHONPATH=src:. .venv/bin/python \
  scripts/ml/build_registration_training_data.py --help
git diff --check
PYTHONPATH=src:. .venv/bin/pytest -q
```

Observed: compileall and diff checks exited 0; the Colab-compatible CLI help rendered;
and pytest reported `67 passed, 3 skipped in 18.81s`. The unchanged skips are the
optional scikit-learn/PyTorch training tests because those packages are absent from
the core environment. No model was trained or activated, no COLMAP run occurred, and
no GPU/network/dataset download was used.

