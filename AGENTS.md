# Codex project instructions — 3D Reconstruction Video Suitability MVP

## Mission and non-negotiable scope

Build a complete, locally runnable MVP in 3 days. Given a video of a **mostly static physical object filmed from multiple viewpoints**, output a clearly identified **heuristic suitability score (0–100)**, sub-scores, detected video problems with timestamps, actionable capture recommendations, and optionally an ML prediction if and only if a properly trained model exists. This **does not** reconstruct a 3D model. Target a CPU-runnable MVP; GPU is optional for research/training. Prefer a reliably tested end-to-end app over a speculative advanced model.

Treat `co3d_video_quality_assessment.ipynb` as an **unverified reference implementation**. Audit and extract reusable modules; do not blindly run cells or claim it was successfully trained. Current notebook execution outputs show CO3D data unavailable, no COLMAP-generated labels, no features, no trained model, no evaluation results. The notebook uses Colab-specific `/content/...` paths, and one downloader cell attempted a large `apple` category download that did not yield a usable dataset in the later cells.

## Coding rules

- You own software implementation, refactoring, environment setup, testing, instrumentation, integration, simple UI, docs and reproducible commands. Make reasonable technical choices autonomously.
- Work in small, tested milestones, save and explain progress in `STATUS.md`, and report executed commands and actual results. Never invent metrics or say tests passed without running them.
- Keep user decisions limited to approving large downloads, money, secrets/credentials, destructive actions, hardware choices and providing real test videos.
- No large CO3D downloads, paid APIs, remote data transmission or GPU rentals without approval. Never check data archives, model weights, secrets or large media into Git.
- Maintain `requirements.txt` or `pyproject.toml`, `.gitignore`, tests, and an accessible CLI. Run without dataset or GPU for the heuristic path.
- If missing a dependency/data file, degrade gracefully with explicit status; never silently fabricate measurements.

## Suggested architecture

```
app.py                         # Streamlit UI, only after CLI works
src/video_suitability/
  video.py                     # validation, timestamps, sampling, memory bounds
  quality.py                   # per-frame exposure, blur, contrast, texture, lighting stability
  geometry.py                  # ORB matches, geometric verification, overlap, motion heuristics
  scoring.py                   # explainable heuristic, sub-scores, uncertainty flags
  infer.py                     # video -> JSON result / optional trained model
  datasets/co3d.py            # optional dataset handling
  labels/colmap.py            # optional COLMAP experiments, cached/resumable
  ml/features.py              # optional feature-extraction for training
  ml/train.py                 # optional baseline first; CNN-LSTM second
  ml/evaluate.py
scripts/
tests/
data/samples/                 # ignored videos, 2-4 short local samples
models/                       # ignored saved trained models
reference/co3d_video_quality_assessment.ipynb
AGENTS.md
README.md
STATUS.md
```

You may improve the structure, but preserve clear module boundaries and a stable API. Proposed CLI: `python -m video_suitability.infer --video path/to/video.mp4 --out report.json` (working syntax documented in README). Result should include `analysis_method`, `overall_score`, `confidence_or_uncertainty`, `subscores`, `raw_metrics`, `problem_segments`, `recommendations`, and `warnings`. A trained score and heuristic score must never be mislabeled or conflated.

## Verified notebook features to extract/reuse

- CO3D `.jgz` annotations and category/sequence EDA (cells 5–12).
- COLMAP reconstruction/quality label driver, caching and composite score (cells 14–17), only after auditing and actually running experiments.
- Grouped train/validation/test split (cells 20–21), with stronger identity leakage checks.
- Frame sampling and PyTorch dataset (cells 23–24).
- ImageNet ResNet-18 encoder, twelve RGB/temporal features, feature caching (cells 26–28).
- ResNet-18 + LSTM regression candidate model/training and baseline comparisons (cells 30–38).
- Video reader/prediction function (cells 45–46) AFTER training creates artifacts, not before.

## Known issues / audit checkpoints in uploaded notebook

1. **No dataset or trained artifacts.** Do not claim pretrained CNN-LSTM weights exist merely because the ResNet-18 ImageNet weights were downloaded. Actual sequence quality model was not trained.
2. **Training/inference feature preprocessing mismatch:** training builds raw handcrafted temporal vectors; inference optionally loads and applies `temporal_scaler.npz`, which is fitted/saved separately after training. Fit the scaler on training data only, apply identical transformations throughout train/val/test/inference, and test parity.
3. **Unbounded regression output:** `CNNLSTMQualityModel.forward` comments that the head is sigmoid-bounded but returns `score*100` directly without a sigmoid. Decide on a robust bounded output or consistently bounded inference; test ranges and loss.
4. **Incomplete video handling:** `read_video_frames(..., max_frames=512)` reads the first frames, risking coverage of only the beginning of long videos; implement uniform whole-video temporal sampling and consistent train/inference sampling. Preserve/propagate actual frame validity masks for padded sequences.
5. **Coverage calculation:** COLMAP reconstructed point cloud and CO3D reference cloud may use different coordinate systems. Verify alignment/scales before using nearest-neighbor coverage. If unverified, mark coverage unavailable and renormalize score rather than reporting false coverage.
6. **Viewpoint diversity proxy:** ResNet feature dissimilarity or pixel difference is not proof of camera translation/3D parallax. Include robust geometric verification, cautious wording, and warnings for rotation-only/low-texture scenes.
7. **Label threshold calibration:** existing notebook appears to optimize F1 against labels derived from the same composite score at threshold 50, which is circular. Keep a fixed, declared threshold unless calibrated against genuinely independent ground truth / downstream reconstruction criteria.
8. **Sparse labels / leakage:** current notebook samples only eight sequences per category by default. This cannot justify claims of generalization. Ensure identity/scene/video grouping, separate validation and test sets, and explicitly show confidence/limitations. Prefer tabular Random Forest/LightGBM regression as a baseline before attempting CNN-LSTM.
9. **Bad or unavailable COLMAP runs:** distinguish valid *failed reconstruction* results from *COLMAP unavailable* or missing data. Don't treat missing infrastructure as a legitimate bad-video label.
10. **Computational feasibility:** CUDA/T4 in a past notebook output does not establish a GPU in the user's current environment. Detect hardware at runtime, use CPU fallback, and avoid assuming Colab paths.

## Execution plan with acceptance gates

### MILESTONE 1 — working MVP without dataset/GPU (start now)

- Inspect notebook and document reusable functions, dependencies, runtime assumptions and confirmed issues.
- Create package, dependency setup and test fixtures. Generate synthetic video clips with deterministically varied brightness/blur/motion. If the user provides actual videos, use them for manual validation.
- Implement timestamped frame extraction with whole-video sampling and resource bounds.
- Implement lighting/exposure, contrast, blur, texture, ORB-based feature correspondence, RANSAC matching, overlap and cautious motion metrics.
- Implement reproducible, configurable heuristic suitability scoring, uncertainty handling and a CLI outputting JSON + recommendations with timestamps.
- Implement meaningful unit/integration tests and RUN them; report observed test outputs. No trained model required.
- Keep the UI minimal until CLI is demonstrably functional.

### MILESTONE 2 — dataset audit + optional label generation

- Establish whether user supplies a local CO3D subset. For a fresh download, present the dataset choice and approximate storage/network need for approval. Official CO3Dv2 `--single_sequence_subset` is ~8.9 GB compressed; allow alternate small hand-selected sequences.
- Verify category layout, annotation files, images, masks/point-cloud metadata and sample frame order. Log missing/corrupt inputs.
- If COLMAP is available and a small subset is approved, run a 2-sequence smoke test before launching batches. Capture actual runtime, failure status, per-sequence reconstruction metrics, logs, and cached labels.
- Guard against label artifacts from mismatched coordinate frames or missing reference coverage.

### MILESTONE 3 — ML only if sufficient genuine labels

- Train a cheap tabular baseline first with extracted video-level features and properly independent target labels. Group by object/scene/video, report MAE and a valid binary metric only where both classes exist, and compare to the heuristic.
- Train/fine-tune the CNN-LSTM only if dataset size, labels, hardware and remaining time justify it. Use the same normalization, sampling, masks and architecture in training and inference. Record reproducible configs/checkpoints.
- If insufficient labels, skip supervised ML and clearly report why. Never fit ML to scores invented from the same features and claim independent predictive validity.

### MILESTONE 4 — UI and final testing

- Build Streamlit video upload, individual metrics, score and caveats, a frame-quality timeline and a downloadable JSON report. Clearly indicate method and whether any model is trained/validated.
- Test a moving-orbit, stationary-camera, blurry and poorly illuminated video. Provide a short runbook and demo scripts.

## What to report at each milestone

1. What files changed and why.
2. Exact terminal commands that ran.
3. Exact observed results (including failures); tests run / passed / skipped.
4. Whether code operates without CO3D and without GPU.
5. Current blockers and the next independently executable step.

**Start immediately on Milestone 1, continue through running tests and debugging, and only ask for approval if one of the explicitly user-owned decisions is necessary.**
