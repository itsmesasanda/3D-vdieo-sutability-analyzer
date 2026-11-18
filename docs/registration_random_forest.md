# Random Forest COLMAP-registration integration

## Current audit result

The repository does not contain the requested saved Random Forest artifact. A search
of the working tree, ignored files, Git branches, and history found no `.joblib`,
`.pkl`, preprocessing export, eight-feature list, or saved training parity example;
`models/` contains only `.gitkeep`. The reference notebook defines a different
12-feature handcrafted/CNN baseline, while the newer ML infrastructure exports a
variable-width `cv_numeric_features/2.1` model. Neither can safely stand in for the
specified eight-feature model that predicts COLMAP registered-image percentage.

Consequently, the application currently reports the learned prediction as
**unavailable**. It does not guess feature definitions or expose a numerically
incompatible prediction. No model was retrained on the existing 45 labels.

## Separate result contract

When a compatible artifact is installed, Streamlit displays:

- `Predicted COLMAP image registration: N%`;
- model version and training-example parity status; and
- wording that this is the estimated percentage of sampled COLMAP input images
  expected to register—not the heuristic score, reconstruction-success probability,
  exact surface coverage, or object completeness.

The report field is `registration_prediction`. It records
`affects_suitability_score: false`, `affects_viewpoint_diversity: false`, and
`colmap_run: false`. The original suitability score and experimental viewpoint
category remain independent.

## Exact eight-feature OpenCV contract for final retraining

`registration_features.py` implements one fully specified, versioned CPU extractor.
The final Colab retraining must import this function or export metadata proving it
used the identical contract. Its ordered features are:

1. mean grayscale brightness;
2. mean grayscale contrast;
3. mean variance of `cv2.Laplacian(gray, CV_64F)`;
4. mean ORB keypoint count;
5. mean consecutive-frame grayscale absolute difference;
6. mean Lowe-ratio-passing ORB match count;
7. mean good-match count divided by the smaller descriptor count; and
8. mean homography-RANSAC inlier count divided by good-match count.

The artifact's `feature_extractor.json` must also record the exact:

- whole-video uniform rounding/sample count;
- fixed resize dimensions and `INTER_AREA` interpolation;
- BGR-to-gray conversion;
- every ORB constructor parameter and RNG seed;
- Hamming KNN matcher, `k=2`, Lowe ratio and denominator;
- homography RANSAC threshold;
- zero-match behavior and arithmetic-mean aggregation; and
- feature names and formula strings in order.

The checked template is
`evaluation/ml/registration_rf_artifact_template/feature_extractor.json`. It is a
proposed final-retraining contract, not evidence that the missing historical Colab
run used these settings.

## Mandatory training-example parity gate

A model is loaded only after at least one saved training video is decoded and its
eight features match the vector saved by Colab within declared tolerances. Each
parity video has a SHA-256. A mismatch returns `status: incompatible` and joblib is
not loaded. This detects sampling, codec seeking, orientation, resize, OpenCV/ORB,
matching, formula, and feature-order drift.

The artifact directory must contain:

```text
artifact_manifest.json
model.joblib
feature_extractor.json
parity_examples.json
parity/
  one_or_more_small_training_examples.mp4
```

All files must be SHA-256 protected by the existing artifact manifest. Required
identity fields are:

```text
model_type: random_forest_colmap_registration_percentage
training_scope: genuine_colmap_registration_labels
metadata.target_name: colmap_registered_image_percentage
metadata.target_units: percent_0_to_100
metadata.feature_count: 8
```

Joblib is executable serialization and must only come from the trusted training
pipeline.

After final retraining writes the model, feature spec, parity JSON, and parity
videos into a new versioned directory, validate and hash it without retraining:

```bash
python scripts/ml/package_registration_rf.py \
  --artifact-dir /path/to/registration_rf_final_v2 \
  --model-version final-labels-v2 \
  --record-count 120 \
  --training-run-id colab-final-2026-11
```

## Deployment and artifact replacement

Install the UI and model runtime dependencies:

```bash
python -m pip install -e ".[app-ml]"
```

Deploy the versioned artifact outside Git, then either atomically replace
`models/registration_rf_current` or set:

```bash
export VIDEO_SUITABILITY_RF_ARTIFACT_DIR=/absolute/path/to/versioned_artifact
streamlit run app.py
```

Replacing that directory is sufficient after final labels are merged and the model
is retrained, provided the artifact retains the accepted extractor/parity contract.
No COLMAP binary, dataset, GPU, or network access is required during upload-time
inference.

The CLI can attach the same separate result:

```bash
python -m video_suitability.infer \
  --video capture.mp4 --out report.json \
  --registration-artifact /absolute/path/to/versioned_artifact
```

## Failure behavior

- Missing artifact: `unavailable`; heuristic and viewpoint results continue.
- Missing scikit-learn/joblib: `unavailable` with installation guidance.
- Bad hashes, wrong target, wrong feature count/order, missing parity videos, or
  parity mismatch: `incompatible`; no prediction is shown.
- Decode failure or invalid model output: `unavailable`; no score is fabricated.
- Synthetic/smoke or generic reconstruction-quality artifacts are rejected.

## CPU benchmark

Until the genuine artifact is supplied, only exact feature extraction can be
benchmarked:

```bash
python scripts/benchmark_registration_rf.py \
  --video data/samples/real_tests/orbit_good.mp4 \
  --feature-spec evaluation/ml/registration_rf_artifact_template/feature_extractor.json \
  --out benchmarks/registration_rf_cpu_untrained.json \
  --runs 3
```

On the recorded Apple Silicon run, 16 frames resized to 320×240 and 15 ORB pair
comparisons took a mean **0.260569 s wall** and **0.347153 s process CPU**. Individual
wall times were 0.509813, 0.135002, and 0.136893 seconds. All three vectors were
identical and the source SHA-256 was unchanged. This is an extractor benchmark—not
a model benchmark or accuracy result. Re-run the same script with `--artifact-dir`
after the genuine artifact arrives to measure parity, extraction, and prediction
together.

The same three-run feature-only benchmark on the repository's real 4K HEVC
`IMG_3913.MOV` averaged **12.873245 s wall** and **59.008896 s process CPU**. The
individual wall times were 13.098215, 12.436417, and 13.085103 seconds; all three
feature vectors were identical and the source hash was unchanged. OpenCV/FFmpeg
multithreading makes process CPU exceed wall time. Sparse source-frame decoding is
the dominant deployment cost for this codec. It cannot be optimized by silently
changing sampling or resize behavior after training; the final artifact's parity
contract must be preserved and the complete artifact benchmark rerun first.
