# Project status

## Rice-cooker tracker reproducibility fix (2026-11-20)

The inconsistent 100%, 50%, and 16% object-tracking reports were investigated on
the same local rice-cooker file. Its SHA256 is
`bf90a2d662b2465e897ff2347792063f59a1a0c8ed8274d6e4022087e1e7783b` (3,228,212
bytes). The 100% local run used normalized ROI `[0.20, 0.10, 0.55, 0.82]`; the
Streamlit defaults are `[0.25, 0.10, 0.50, 0.80]`, which yield 1/32 accepted
samples on this file. ROI `[0.23, 0.08, 0.53, 0.84]` reproduces 5/32 (15.625%).
Thus these were not identical tracker inputs. The reference GrabCut foreground
prior changes substantially between these boxes; the 100% mask touches its left
selection border and is not evidence that the entire cooker was tracked accurately.

The Streamlit preview also used source frame 119 at 720x405 while analysis
initialized tracking on uniform sample source frame 123 at 640x360. The preview
now decodes the exact analysis sample and size; a pixel comparison on the real file
was identical (`max_abs_diff=0`). Reports now contain video SHA, byte count, ROI,
sample indices, analysis dimensions, tracker settings, Git/source identity, and
per-sample/per-transition diagnostics. The session cache now includes tracker/code
identity, preventing a stale pre-deploy result from being reused after code changes.
No tracking threshold, heuristic score, ML feature, or model was changed.

An exact Streamlit replay can be run with:

```bash
.venv/bin/python scripts/debug_object_tracking_replay.py \
  --video /path/to/video.mp4 \
  --bbox 0.23,0.08,0.53,0.84 \
  --out /tmp/object-tracking-replay.json
```

Focused verification: `18 passed in 22.27s`. Full regression verification:
`125 passed, 2 skipped in 371.45s`; both skips are the existing optional PyTorch
tests because PyTorch is not installed in the CPU deployment environment.

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

## Historical Colab label compatibility audit — implemented (2026-11-19)

The reference CO3D labeling notebook, persistent reconstruction notebook, production
60-frame contract, locally accessible reconstruction artifacts, and connected Drive
inventory were audited. The checked-in reference notebook uses 16 COLMAP images,
floor-truncated linspace sampling, no frame manifest/hashes, and a composite
`quality_score` distinct from the registration target. The persistent phone-video
notebook verifies hashes and saves result evidence but uses 35 images. Neither is
directly eligible for the exact 60-image final target.

The connected Drive account did not expose the original 45 labels or A–G/R1–R3
batch folders, so no result has been claimed for actual Batch C. The limitation and
the exact Drive collection procedure are recorded in
`docs/colab_label_compatibility_audit.md`.

Added `colab_label_import.py` and
`scripts/ml/import_colab_registration_labels.py`. The importer keeps source files
read-only, reconciles CSV metrics with per-record reconstruction JSON, requires
60 inputs, excludes infrastructure failures, retains physical-source/variant/label
provenance, distinguishes original notebook hashes from import-time hashes and
CO3D regeneration, and emits canonical label rows plus an acceptance audit. Missing
CO3D inputs can be regenerated only with a declared supported algorithm and a
hashed notebook/config file proving it; the result is never described as original
hash verification.

The production label contract now makes `source_group_provenance` and
`input_hash_provenance` mandatory. A schema-faithful completed-batch fixture produced
one accepted row (45/60 = 75%), one incompatible 16-frame rejection, and one
infrastructure exclusion. The accepted regenerated row passed the downstream
production manifest/hash/sampling/eight-feature checks. This fixture is a contract
test, not evidence that inaccessible Batch C passes.

Final verification commands:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
.venv/bin/python scripts/ml/import_colab_registration_labels.py --help
.venv/bin/python scripts/ml/build_registration_training_data.py --help
git diff --check
.venv/bin/python -m pytest -q
```

Observed: both CLI help commands rendered, compileall and `git diff --check` exited
0, and the full suite reported `69 passed, 3 skipped in 18.82s`. The unchanged skips
are optional scikit-learn/PyTorch tests because those packages are absent. No model
was trained or activated, COLMAP was not run, source label/reconstruction files were
not modified, and no CO3D or other large data was downloaded.

## Real 60-frame COLMAP CSV archive audit — complete, evidence retrieval pending (2026-11-19)

The supplied `csv files.zip`, preliminary Markdown audit, and audit-only deduplicated
CSV were independently inspected. Source SHA-256 values were recorded before and
after processing and remained unchanged. A reproducible read-only auditor was added
at `scripts/ml/audit_colmap_csv_archive.py`; its outputs are under
`evaluation/ml/colmap_csv_archive_real_audit/` and the findings are documented in
`docs/real_colmap_csv_dataset_audit.md`.

Verified totals: 675 unique experiment tags, 659 numeric labels, and 49 source-group
identifiers. All selected rows declare 60 images and their numeric counts, rounded
ratios, and rounded percentages agree. There are 647 numeric sparse-model rows, 12
numeric mapper-failure zeros, 5 unlabeled mapper failures, 9 budget pauses, and 2
matching exceptions. The ZIP's 1,173 label rows include 373 duplicated tags and 498
extra export rows; semantic normalization found zero contradictory measurements.
Batch G was not present.

Current compatibility is deliberately 0 accepted, 664 recoverable, and 11 excluded.
The ZIP has no per-record result JSON or exact input manifest/hash evidence. Numeric
mapper zeros remain unaccepted until failure stage/command evidence is available.
Only 15 source identifiers can be crosschecked against source-list assignments; A/B's
34 identifiers need the referenced candidate manifest. The five held-out category
names are absent from development rows, but their exact physical source IDs are not
in the archive.

The importer now supports the real `registration_proxy_60f_v1` schema, exact integer
counts, declared rounded ratios/percentages, experiment tags, source tags, and label
versions. Canonical percentages come from integer counts. It refuses generic mapper
errors without verified mapper-stage evidence, refuses audit-only group maps, and
does not regenerate transformed variants from base CO3D frames without an implemented
and evidenced transform contract. A dry run on the actual 52-row Batch C CSV detected
the correct schema, preserved its hash, accepted 0, and rejected all 52 specifically
because their per-record reconstruction JSON files were absent.

The generated small-file retrieval manifests request compact result JSON, ordered
60-frame manifests/hashes, executed transformation configs, and compact mapper-failure
summaries where relevant. They explicitly exclude COLMAP databases, sparse/dense
models, point clouds, and complete run folders. Archived historical feature CSVs do
not match the production eight-feature contract and will not be reused.

Final commands:

```bash
.venv/bin/python scripts/ml/audit_colmap_csv_archive.py \
  --archive "/Users/sasanda/Downloads/csv files.zip" \
  --audit-inventory-csv /Users/sasanda/Downloads/colmap_deduplicated_AUDIT_ONLY.csv \
  --out-dir evaluation/ml/colmap_csv_archive_real_audit
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
git diff --check
.venv/bin/python -m pytest -q
```

The audit command intentionally exited 2 because `training_ready=false`. Compileall
and diff checks succeeded; the final full test suite reported `70 passed, 3 skipped
in 18.43s`, and an explicit audit-artifact consistency check passed. The skips remain
optional sklearn/PyTorch tests. No COLMAP, feature
extraction, training, or model activation occurred, and heuristic/viewpoint scoring
was unchanged.

## Experimental final registration-model training workflow — implemented, not run (2026-11-19)

Added the CPU Colab workflow at
`notebooks/experimental_registration_training_colab.ipynb`, its reproducible builder,
a non-permissive regeneration-contract template, an executed-code adapter template,
tests, and `docs/experimental_registration_training.md`.

The notebook starts only from the 647 audited numeric sparse-model records. It
deduplicates and rechecks 60-frame/count/percentage invariants, requires an exact
executed sampling/transformation contract, regenerates inputs through a separately
hashed adapter, validates 60-image manifests, and calls the production
`registration_features.py` extractor. Unknown or incompletely evidenced variants
are excluded or marked uncertain; mapper failures and infrastructure/interruption
records are never converted into training zeros.

The evaluation freezes identical physical-source `GroupKFold` splits for Dummy
mean, Ridge, Random Forest, and Gradient Boosting; keeps preprocessing inside each
fold; reports MAE, RMSE, prediction distributions, and degradation-family errors;
and selects only from grouped CV. Exactly five declared physical source IDs are
reserved outside development and may be evaluated once only when compatible
held-out features exist.

Every selected artifact is written as experimental with feature metadata, provenance,
hash inventory, and saved parity inputs. Production packaging defaults off and is
allowed only when provenance is complete, all inputs are original-hash verified, the
selected model is compatible with the existing RF-only loader, the independent
holdout has been evaluated, and the existing feature-parity gate passes. No model was
trained or activated locally because the executed labeling code/configuration, CO3D
tree, and exact five held-out source IDs are not present in this workspace. Therefore
no accepted/excluded runtime counts or model metrics are claimed yet.

Verification commands:

```bash
.venv/bin/python scripts/ml/build_experimental_registration_training_notebook.py
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
git diff --check
.venv/bin/python -m pytest -q
```

Observed: the builder regenerated a clean CPU notebook whose code cells parse,
compileall and `git diff --check` exited 0, and the full suite reported
`73 passed, 3 skipped in 18.53s`. The skips are pre-existing optional scikit-learn
and PyTorch tests because those dependencies are absent from the local core
environment; Colab installs the tabular dependencies before training.

## R1 60-frame regeneration pilot — implemented, real Colab run pending (2026-11-19)

Extracted the executed implementation from `co3d_R1_fourhour_colab.ipynb`. The
executed notebook's full SHA-256 is
`e6d66c8515b72ce6263fcb6f297427200713530b7e1038aa11c5f08135340717`;
its executed constants and regeneration cells match the pinned source hashes in
`r1_regeneration.py`. The implementation preserves the original lexicographic CO3D
ordering, all 16 window/span camera-path formulas, EXIF transpose, RGB conversion,
max-edge 800 LANCZOS resize, and Pillow JPEG quality 90 save.

Added the focused CPU notebook `notebooks/r1_regeneration_pilot_colab.ipynb`, the
R1 adapter, and `scripts/ml/run_r1_feature_pilot.py`. The default pilot selects three
real numeric sparse-model labels across three physical groups: apple span (60/60),
backpack window (29/60), and banana span (2/60). It regenerates 60 inputs, writes
source/output hashes and manifests, invokes the production eight-feature extractor,
and saves feature, decision, diagnostics, and provenance outputs. Original prepared
JPEG identity remains honestly `unverified`; labeling-time hashes and the exact
Pillow/libjpeg runtime were not saved.

An end-to-end local smoke test used synthetic images at the three real CO3D directory
paths and the real R1 labels. It accepted 3/3 records, produced all eight finite
features, preserved groups and canonical count-derived targets, and reported
`colmap_run=false`. These are implementation-test features, not real CO3D feature
values and were kept only under `/private/tmp`. The real three-row dataset must be
generated by running the Colab notebook against the official CO3D sequences. After
that succeeds, the disabled final cell expands the identical path to the 72 numeric
R1 sparse-model rows across the five R1 source groups.

Final verification commands:

```bash
.venv/bin/python scripts/ml/build_r1_regeneration_pilot_notebook.py
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
git diff --check
.venv/bin/python -m pytest -q
```

Observed: notebook generation, compileall, and diff checks exited 0. The full suite
reported `79 passed, 3 skipped in 19.93s`; the skips remain optional scikit-learn
and PyTorch tests unavailable in the local core environment. No production artifact
manifest was created, no model was trained or activated, and the heuristic and
viewpoint code paths were unchanged.

## Fast historical-feature ML experiment — trained offline, not activated (2026-11-19)

Paired the original 45-row `random_forest_experiment/features.csv` with its 60-frame
labels and paired the selected A–F/R1–R3 `new_features_50*.csv` files one-to-one with
the label exports already selected by `record_audit.csv`. Every batch feature tag set
matched its audit batch exactly. This produced 675 unambiguous joins; excluding 28
mapper-failure, interruption, or otherwise invalid label rows left 647 numeric
`sparse_model_created` experiments across 49 physical source groups.

The historical extractor implementation was identical in the available A, B, C,
R1, R2, and R3 notebooks (source hash
`ac727c93fc14ffec7dd8079fb53f876f8a8b71560237ff9e319692d4288a2be0`).
It is not production-compatible: 12 versus 16 frame samples, conditional max-edge
480 default resizing versus fixed 320×240 `INTER_AREA`, ORB 400/defaults versus ORB
1000/explicit parameters and RNG, and three different feature meanings on each side.

All models used the same five physical-source GroupKFold splits. Predictions were
clipped to 0–100 before metrics. Observed out-of-fold results were Ridge MAE 19.6249 /
RMSE 27.0212, Random Forest 20.0599 / 27.3581, Gradient Boosting 21.1340 / 29.0513,
and Dummy mean 34.0572 / 39.4749. Ridge won by grouped-CV MAE. Exact held-out source
IDs and compatible evaluation rows were unavailable, so no independent held-out
metric is claimed.

The versioned output is
`evaluation/ml/historical_registration_experiment/historical_opencv8_v2/`. It
contains the 647-row joined table, frozen splits, OOF predictions, metrics, feature
definition, saved Ridge pipeline, parity rows, hashes, and an explicitly offline-only
manifest. No production `artifact_manifest.json` was created: the selected model is
not Random Forest and historical preprocessing cannot pass production feature parity.

The final experiment used Python 3.9.6 and scikit-learn 1.6.1. A second clean run
produced identical SHA-256 values for the selected model, grouped-CV summary, and all
OOF predictions. Compileall and `git diff --check` exited 0; the full suite reported
`82 passed, 3 skipped in 12.36s`. The skips are optional scikit-learn/PyTorch tests in
the core virtual environment; training itself ran successfully with the system
scikit-learn installation. No CO3D download, frame regeneration, or COLMAP command
was run.

## Historical Ridge live-inference compatibility — blocked honestly (2026-11-19)

Recovered the exact legacy eight-feature image-directory function from the available
labeling notebooks and added a standalone compatibility implementation. Its feature
names/order, 12-of-60 sampling, conditional max-edge-480 default resize, ORB 400,
Lowe-ratio matching, brightness/sharpness variation, and dark-fraction definitions
match the notebook source. Focused prepared-input tests produced zero numerical
difference from the reference function.

Live compatibility could not be established. Historical training began with 60
prepared CO3D JPEGs, not uploaded videos; no executed video adapter exists; the
training-time prepared frames/hashes are unavailable for parity; and OpenCV ORB and
resize defaults were not pinned consistently. A candidate video adapter was kept as
a non-deployable probe only. Streamlit does not import it and no Ridge prediction
field or panel was added.

On the real phone recordings `IMG_3913.MOV` and `IMG_3914.MOV`, reference/recreated
prepared-input features matched exactly. Probe runtimes were respectively 40.244810 s
wall / 205.058385 s process CPU and 46.476229 s wall / 239.832806 s process CPU.
These timings also make the unoptimized 60-frame/JPEG route unsuitable for the
current interactive upload path. No model was retrained and no production RF,
heuristic, viewpoint, or geometric code was changed.

Compileall and `git diff --check` exited 0. The final full suite reported
`85 passed, 3 skipped in 51.63s`; the three skips remain optional scikit-learn and
PyTorch tests in the core virtual environment.

## Streamlit Community Cloud deployment preparation — complete (2026-11-19)

Prepared the stable public path for repository `itsmesasanda/3D-vdieo-sutability-analyzer`,
branch `main`, entry point `app.py`, with Python 3.11 recommended in Community Cloud.
The root `requirements.txt` installs the src-layout package with its UI extra;
`packages.txt` requests Debian `ffmpeg`; and `.streamlit/config.toml` configures a
150 MB upload limit, headless mode, minimal toolbar, and disabled usage telemetry.

The Streamlit layer now hard-disables both Ridge and Random Forest inference and
does not import or call either predictor. Supplying an artifact path cannot activate
ML in the app. The existing CLI Random Forest compatibility gates remain unchanged.
The Evaluation tab separately reports the offline Ridge grouped-CV result (MAE
19.6249 percentage points), 647 historical rows, and the lack of an independent
held-out evaluation without presenting it as reconstruction quality.

Estimated Viewpoint Diversity is opt-in in Streamlit and remains separate from the
unchanged heuristic. Its call is exception-isolated in `analyze_video`; a disabled
or failed run returns `Unknown`, records the reason, and leaves the overall score,
subscores, and baseline unchanged. The adjacent ORB/RANSAC evidence used by the
established heuristic remains intact. Uploads above 150 MB and unsupported suffixes
are rejected before temporary-file creation. Accepted uploads use unique temporary
files removed in `finally`; session caching includes the content hash, object box,
viewpoint option, and the fixed disabled-ML policy identity.

Real upload-path checks, with optional viewpoint disabled and diagnostic rendering
timed separately:

- `IMG_3914.MOV`: 44.04 MB, 2160x3840 HEVC, 401 frames at 60 fps. Upload-to-result
  was 11.011878 s wall / 38.593701 s process CPU; deferred rendering of three
  diagnostic frames was 2.120431 s wall. Heuristic score: 60.0.
- `copy_179FD8BF-1F7D-4549-9E3E-75A648BE491E.mp4`: 28.61 MB, 1160x718 HEVC,
  1,331 frames at 30 fps. Upload-to-result was 3.913708 s wall / 10.394277 s process
  CPU; deferred rendering of three frames was 0.236658 s wall. Heuristic score: 55.0.
- An unsupported `.txt` upload raised a concise `ValueError`. Corrupt MP4 bytes
  raised `VideoValidationError` with a codec/corruption explanation. Neither left a
  temporary file through the app's cleanup path.

The local OpenCV 4.14.0 build reports `FFMPEG: YES`; NumPy is 2.0.2 and Streamlit is
1.50.0. `pip check` reported no broken requirements. A localhost Streamlit launch
on port 8502 succeeded and `/_stcore/health` returned `ok`; the process was then
stopped normally. The initial sandboxed bind was denied by local permissions, and
the approved localhost-only retry succeeded.

Final commands and observed results:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/check_streamlit_deployment.py
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-deploy-pycache \
  .venv/bin/python -m compileall -q app.py src scripts tests
git diff --check
.venv/bin/python -m pip check
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

The preflight printed `PASS`; compileall and `git diff --check` exited 0; `pip
check` reported no broken requirements; and the full suite reported `89 passed, 3
skipped in 68.15s`. The skips remain optional scikit-learn/PyTorch tests. No model
was retrained or activated, and no CO3D download, COLMAP command, heuristic scoring
change, or user-media modification occurred.

One preliminary focused pytest command referenced an outdated/nonexistent viewpoint
test node and therefore collected no tests. It was corrected to the actual node;
the focused deployment selection then reported `6 passed in 3.42s`.

## Viewpoint animation rejection and rigid-consensus revision — complete (2026-11-19)

Updated only the isolated experimental viewpoint path; the deployed heuristic
scoring formula, weights, caps, Random Forest/Ridge policies, and default opt-in UI
behavior are unchanged. No COLMAP or ML work ran.

`viewpoint.py` version 1.1 now requires stronger, spatially distributed fundamental-
matrix consensus before an uncalibrated pair can count as parallax-like support. It
also diagnoses long-gap motion populations directly from the shared ORB descriptors.
When multiple evaluable pairs contain both a broad near-static match population and
a spatially supported strongly moving population, the result is `Unknown` with an
explicit object-animation/non-rigid-scene reason. High overlap and optical flow do
not contribute to the viewpoint category by themselves.

Object-focused tracking now records forward/backward box-cycle checks. A successful
reverse track must overlap the source box; a reverse failure rejects weak forward
tracks but retains a forward track with at least 0.65 RANSAC inlier ratio because
image-boundary clipping can make reverse box validation impossible. Object and
background long-gap geometry now records normalized motion separately. Repeated
object motion of at least 0.012 image diagonal against a verified background motion
of at most 0.004 is treated as animation evidence, never camera-orbit evidence.
Tracking below 70% remains `Unknown`.

Added the deterministic `animated_object` negative control: a textured object moves
and rotates in front of a fixed feature-rich background. Its substantial-change
fraction is 0.923077, rigid-consensus parallax-proxy fraction is 0.615385, and
cumulative displacement is 0.238466—enough to satisfy the prior broad-change High
rule—but its mixed static/dynamic pair fraction is 0.307692, so version 1.1 returns
`Unknown`. At 16 samples / 400 px its wall/process CPU time was 0.271830 / 0.673067
seconds. The main heuristic remains 89.19, demonstrating score-path isolation rather
than silently changing the deployed score.

Controlled scenario results at the same configuration:

- front-only: Low, 0.338502 s wall / 0.650023 s CPU;
- animated object: Unknown, 0.271830 / 0.673067 s;
- quarter orbit: Moderate, 0.260104 / 0.644274 s;
- full orbit: High, 0.264197 / 0.665676 s;
- pure camera rotation: Unknown, 0.255172 / 0.645208 s.

Object-focused synthetic results were conservative: animated object Unknown with
43.75% tracking, quarter orbit Moderate with 100% tracking, and full orbit Moderate
with 93.75% tracking. The latter two had zero static-background/object-motion flags.
Cycle verification succeeded for 100% and 71.43% of attempted steps respectively.

Actual-video whole-frame before/after classification remained conservative. The
previous `evaluation/viewpoint_real_validation/validation.csv` is the before source;
new runs used the same 32-sample / 640 px configuration:

- `IMG_3913.MOV`: Unknown -> Unknown; rigid parallax fraction 0.307692 ->
  0.153846; new analysis 13.281200 s wall / 67.295626 s CPU.
- `IMG_3914.MOV`: Unknown -> Unknown; 0.230769 -> 0.076923; 6.689200 /
  32.968301 s.
- `IMG_3915.MOV`: Unknown -> Unknown; 0.230769 -> 0.000000; 9.025252 /
  35.306465 s.
- `orbit_good.mp4`: Moderate -> Moderate; 1.000000 -> 1.000000; 0.662721 /
  1.215327 s.

The 65.5-second real `sampleVideo.mp4` shoe orbit remained Unknown because only
7.69% of long-gap pairs had eight verified correspondences; its new run took
9.603585 s wall / 46.136064 s CPU. The user-reported waving-hand source file was not
present locally, so no result for that exact upload is claimed. It should be rerun
after redeployment; the controlled animated-object case exercises the failure mode
without being presented as real-world accuracy.

Existing real object-box recordings remained Unknown because tracking covered only
6.25% of samples, with explicit reasons. New wall/CPU times were 13.405938/62.477871
seconds (`IMG_3913.MOV`), 6.248531/33.920956 (`IMG_3914.MOV`), and
6.179708/34.413005 (`IMG_3915.MOV`). Their main scores matched the preserved prior
comparison exactly: 60.75, 51.26, and 49.94.

Sharpness investigation did not justify a scoring change. Current smooth synthetic
orbit frames measured median Laplacian variance 681.339770 and visual quality 97.14;
the deliberately blurred counterpart measured 4.781712 and 14.61. Raw pre-encoding
smooth frames measured median 750.899019. Extremely smooth rendered images can
still score low because Laplacian variance measures high-frequency edge energy and
therefore confounds blur with low texture and depends on analysis scale/resampling.
No threshold was changed without a comparative real-capture regression set.

Verification commands:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-viewpoint-v11 \
  .venv/bin/python -m compileall -q app.py src scripts tests
git diff --check
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

Compileall and `git diff --check` exited 0. The final full suite reported `89 passed,
3 skipped in 62.69s`; skips remain the optional scikit-learn/PyTorch tests unavailable
in the core deployment environment.

## Streamlit presentation compatibility fix and UI redesign — complete (2026-11-19)

Fixed the deployed viewpoint presentation crash without modifying analyzer geometry
or any score path. The revised analyzer emits
`viewpoint_diversity.indicators.temporally_separated_substantial_change_fraction`;
the presenter converts it into the display-only `substantial_change_percent`. The
crash was consistent with `app.py` and `presentation.py` being deployed from
different schema revisions. The renderer no longer uses direct indexing for optional
viewpoint fields, and the presentation model now returns `None` for missing,
non-finite, malformed, or unavailable evidence instead of inventing `0%`. The UI
renders these values as `Not available`. Malformed/older partial presentation
payloads and optional presentation exceptions cannot break the heuristic report.

Redesigned the app as a responsive light analytics dashboard with a restrained coral
accent, white cards, dark typography, higher-contrast tabs/widgets, consistent
progress bars, and desktop/mobile breakpoints at 780 px and 460 px. The upload area
now includes concise capture guidance and a stronger drop zone. Advanced viewpoint
analysis is isolated in an expander. Object-region selection uses a persistent
middle-frame preview, coral bounding box, two-axis controls, and explicit guidance to
cover the complete object with motion margin while excluding unnecessary background.

The result hierarchy now presents the overall score, five components, up to three
priority improvements, experimental viewpoint evidence, disabled live ML status,
and uncertainty in that order. Technical warnings, performance data, full JSON, and
research limitations remain available in expanders. The Evaluation tab uses metric
cards for the offline Ridge result while explicitly stating that live Ridge/Random
Forest inference is disabled and no independent held-out evaluation exists. Upload
errors are mapped to concise user guidance; ordinary users do not receive raw
tracebacks (`showErrorDetails = false` remains configured).

Real upload-path verification with experimental viewpoint enabled:

- `WhatsApp Video 2026-11-19 at 5.43.01 PM.mp4` (the waving rendered-hand negative
  control): score 56.60; unchanged visual/lighting/tracking/overlap/path components
  1.66/100.00/88.53/77.80/70.80; viewpoint Unknown because most samples are too
  blurry for trustworthy geometry; display fields 31% substantial change, 38%
  rigid-consensus proxy, 0% mixed motion, one repeated pair; 0.745154 s end-to-end.
- `IMG_3913.MOV`: score 78.54 unchanged; Unknown displayed successfully; 8.406053 s.
- `IMG_3914.MOV`: score 60.00 unchanged; Unknown displayed successfully; 4.311065 s.
- `IMG_3915.MOV`: score 48.51 unchanged; Unknown displayed successfully; 4.574217 s.

All four reports retained `registration_prediction.status = unavailable`. The three
smartphone scores exactly match the pre-UI geometry-revision results, and the waving
hand reproduces the reported high feature tracking, overlap, and camera-path
components without any scoring modification.

Regression coverage renders Low, Moderate, High, pure-rotation Unknown,
animated-object Unknown, completely missing evidence, malformed results, and an
older partial presentation payload. Missing evidence renders four explicit `Not
available` values. Streamlit `AppTest` covers the complete desktop app structure;
deployment tests assert the light theme, high-contrast selectors, upload drop zone,
and both responsive breakpoints. Local Streamlit startup on port 8503 succeeded and
`/_stcore/health` returned `ok`. No browser screenshot automation was available, so
mobile validation is structural/responsive-CSS validation rather than a pixel-level
device screenshot claim.

Final checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/video-suitability-ui \
  .venv/bin/python -m compileall -q app.py src tests
git diff --check
PYTHONPATH=src:. .venv/bin/python scripts/check_streamlit_deployment.py
PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

Compileall and `git diff --check` exited 0, deployment preflight printed `PASS`, and
the final full suite reported `98 passed, 3 skipped in 102.59s`. The skips remain optional
scikit-learn/PyTorch tests. No scoring, ML feature, artifact, model, geometry-decision,
COLMAP, timestamp-diagnostic, or reconstruction code changed.

## Opt-in experimental Ridge live inference — implemented (2026-11-19)

Added a deterministic uploaded-video policy for the exact historical eight-feature
implementation and a separate integrity-gated Ridge result. The adapter defines 60
uniform ordered positions but decodes/JPEG-round-trips only the 12 positions the
legacy extractor consumes. Direct regression testing against a fully materialized
60-JPEG adapter produced exactly equal feature dictionaries. This is a runtime
optimization, not a claim of parity with the historical CO3D input generator.

The loader pins the model, feature definition and prediction-parity hashes; validates
the eight names/order, Ridge estimator and three saved outputs; and deserializes only
after the model bytes match the trusted SHA-256. The `.joblib` stays ignored. Cloud
deployment accepts its base64 bytes through `EXPERIMENTAL_RIDGE_ARTIFACT_B64` and
fails closed if absent or altered. Random Forest gates, heuristic score and geometric
analysis were not changed.

Observed local Ridge-path results were 83.508634% in 10.785149 s for `IMG_3913.MOV`,
78.422076% in 9.729399 s for `IMG_3914.MOV`, and 63.557192% in 0.823165 s for the
waving rendered-hand control. All are explicitly unvalidated. Two repeat hand runs
had identical features/prediction; its moderate output demonstrates domain/model
limitations rather than instability. No COLMAP, CO3D download or retraining ran.
The complete suite reported `106 passed, 2 skipped in 128.71s`; both skips are the
unchanged optional PyTorch tests. Deployment preflight passed, Streamlit started on
port 8504, and `/_stcore/health` returned `ok`.

## Minimal independent Ridge/COLMAP validation workflow — implemented (2026-11-19)

Added a reusable three-video offline runner that materializes the frozen live
adapter's exact 60 ordered JPEGs once and feeds the same files to the historical
Ridge extractor and CPU COLMAP. The adapter contract is pinned as
`eac0ba4e28938adaa03a866c77ed7f2393fe5f45985a345b8f82ac4ac25dd9ef`;
each preparation manifest records source positions and image hashes.

The COLMAP pilot matches the Batch C/R1 CPU settings where available: max image size
800, max 2048 SIFT features, two threads, one camera, sequential overlap 6 and no
exhaustive retry. It explicitly freezes `SIMPLE_RADIAL`; the historical notebooks
relied on their installed default. Uploaded-video JPEG inputs and the current COLMAP
build are also documented differences from historical CO3D labeling.

Only exported sparse-model counts and a completed zero-return-code mapper producing
no model become ground truth. Missing infrastructure, command errors, mapper crashes,
interruptions and export failures remain target-less and are excluded from MAE.
The CSV/report generator was tested with successful, completed-mapper-failure and
infrastructure-failure cases.

COLMAP is not installed locally. A real adapter/Ridge smoke run on `good_orbit.mp4`
prepared 60 hashed images, predicted 80.468357%, classified COLMAP as
`infrastructure_failure`, left actual registration/error blank, and correctly did
not calculate MAE. Total smoke runtime was 45.749002 s, including 44.109144 s for
full 60-frame materialization. This is a workflow smoke test, not independent ground
truth. The Colab CPU runbook is in
`docs/independent_ridge_colmap_validation.md`.
The final full suite reported `112 passed, 2 skipped in 112.05s`; the two skips are
the unchanged optional PyTorch tests. Compileall and `git diff --check` exited 0.

## Experimental object capture completeness — implemented (2026-11-19)

Added an isolated capture-guidance result that reuses object-focused tracking,
long-gap ORB/RANSAC evidence, parallax proxies, repeated views, and the existing
object/background comparison. It runs no new feature matching and cannot change the
heuristic score, Ridge estimate, Random Forest result, or viewpoint category.

The panel reports `Limited viewpoint variation`, `Multiple object viewpoints
detected`, or `Capture coverage not verified`. Higher/lower viewing angles remain
`Not verified`: current image-space evidence cannot establish a reliable
object-relative elevation path. It never names front/back/top/bottom, estimates an
orbit angle, claims 360-degree completion, or reports a surface-coverage percentage.

The animated-hand control remained score 56.60 and viewpoint Unknown; 50% object
tracking produced `Capture coverage not verified` in 1.259257 s total. Existing
`IMG_3913.MOV` remained score 78.54 and viewpoint Unknown; 3% tracking conservatively
produced `Capture coverage not verified` in 14.670735 s. The requested rice-cooker
and matching front-only real files were not present in the workspace or Downloads,
so no result is claimed for those recordings.

Synthetic object-focused results were: front-only `Limited`; quarter-orbit and
full-orbit `Multiple object viewpoints detected`; pure rotation, animated object,
and blurry `Capture coverage not verified`. Full pipeline times were 0.29–0.72 s.
The pure summary step averaged about 0.0036 ms over 10,000 calls, so the panel adds
negligible CPU work compared with existing decoding and geometry.

Final verification commands were `compileall`, `git diff --check`, the Streamlit
deployment preflight, and the full pytest suite. Compilation/diff checks exited 0,
deployment preflight reported PASS, and the final pytest run reported `119 passed,
2 skipped in 145.96s`; the skips are the unchanged optional PyTorch tests.

## Hybrid real-video object tracking — implemented (2026-11-20)

Replaced first-failure global-ORB box propagation with a conservative hybrid of
object-masked periodic ORB re-detection, pyramidal KLT, point-level and box-level
forward/backward checks, affine RANSAC, damped axis-aligned scale updates, and one
bounded post-occlusion reacquisition. A one-frame foreground prior separates object
and background feature caches; background pixels inside a loose selection rectangle
no longer become object-view evidence. The 70% evidence requirement is unchanged.

At production settings, rice-cooker tracking improved from 17/32 (53.1%) to 32/32
(100%), cycle verification from 94.1% to 100%, median verified object matches from
126 to 144, and object valid geometry from 51.6% to 100%. Wall time increased from
1.755 s to 3.887 s. Its score stayed 83.14. Viewpoint/completeness stayed
Unknown/not verified because 77.8% of evaluated long-gap pairs were background
dominated and separated object descriptors remained ambiguous.

The animated hand improved from 50.0% to 71.9% tracking but remained Unknown/not
verified because blur and low texture invalidate coverage inference. The three real
smartphone MOVs remained at 3.1% tracking: sparse production samples contain too
much inter-view appearance replacement for reliable KLT/ORB cycles. Their Unknown
results were preserved. The synthetic front-only fixture remained Low with 100%
tracking. Synthetic quarter/full 2D proxies became Unknown after foreground masking
removed the background-layer parallax that previously inflated object evidence.

Final `compileall` and `git diff --check` exited 0. The Streamlit deployment
preflight reported PASS. The final complete regression suite reported `123 passed,
2 skipped in 157.93s`; the two skips are the unchanged optional PyTorch tests.

