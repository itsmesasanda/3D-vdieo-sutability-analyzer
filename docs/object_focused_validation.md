# Object-focused viewpoint validation

## Scope and safety boundary

The optional object-focused path accepts one normalized bounding box
`x,y,width,height` on the representative middle frame. It affects only the separate
Estimated Viewpoint Diversity result. The whole-frame suitability heuristic,
baseline v1, optional Random Forest adapter, ML feature schema, and score thresholds
are unchanged. Live inference remains CPU-only and does not run COLMAP or train a
model.

The result is still image-evidence, not a measurement of orbit angle, camera
translation scale, complete 360-degree capture, or object-surface coverage.
Appearance change by itself never satisfies a diversity category.

## Implementation

1. Uniformly sample and resize the video once.
2. Extract one shared ORB cache. Adjacent whole-frame geometry still feeds the
   unchanged suitability score.
3. Starting at the middle sample, propagate the selected region forward and
   backward using object-box query features, ratio-test matches, and partial-affine
   RANSAC. Reject low inlier support, implausible scale/size changes, and broken
   tracks.
4. Split the shared ORB features into tracked object and background caches. Run the
   viewpoint classifier on object-only features, with at most nine long-gap pairs
   in the live path. Background matches are reported independently and never
   increase the object category.
5. Return `Unknown` if fewer than 70% of samples retain a reliable region, object
   texture/matches are inadequate, descriptors are ambiguous, motion is
   homography/rotation dominated, or background matches dominate most evaluable
   pairs.

When tracking fails before both boxes in a pair are available, background dominance
is reported as `n/a`, not zero. The report explicitly records the unevaluable pairs.

## Explanations

`decision_explanation` provides one primary reason, a named classification rule,
supporting evidence, limiting evidence, and all insufficient-evidence reasons. The
gates distinguish:

- too few verified matches or valid long-gap pairs;
- repetitive/symmetric descriptor ambiguity;
- pure rotation or planar motion explained by a global homography;
- valid matches but inadequate parallax/path extent;
- background-dominated evaluable pairs;
- object-region tracking failure; and
- blur or low texture.

These thresholds are declared engineering defaults. They were not changed to make
the three available MOV files pass.

## Streamlit and CLI

In Streamlit, enable **Focus viewpoint analysis on a selected object region**, set
horizontal and vertical bounds on the middle-frame preview, then click **Analyze
video**. Analysis is button-triggered and cached by source hash plus bounding box.
Diagnostic still rendering is deferred to a separate button.

The equivalent CLI is:

```bash
python -m video_suitability.infer \
  --video capture.mov --out report.json \
  --object-bbox 0.25,0.10,0.50,0.80 \
  --profile
```

## Controlled comparison

Run:

```bash
python scripts/benchmark_object_focused.py \
  --out benchmarks/object_focused_synthetic_comparison.json \
  --max-samples 16 --max-side 400
```

Observed on the recorded local Apple Silicon run:

| Scenario | Whole frame | Object-focused | Object tracking | Median verified long-gap matches whole/object | Parallax proxy whole/object | Wall seconds whole/object |
|---|---|---|---:|---:|---:|---:|
| Front-only | Low | Low | 1.000 | 1268 / 1143 | 0.000 / 0.000 | 3.932 / 5.028 |
| Quarter-orbit proxy | Moderate | Moderate | 1.000 | 561 / 524 | 0.923 / 1.000 | 3.648 / 5.142 |
| Periodic full-orbit proxy | High | Moderate | 0.938 | 426 / 432 | 0.923 / 0.333 | 2.752 / 4.215 |
| Rotation-only | Unknown | Unknown | 1.000 | 827 / 769 | 0.000 / 0.000 | 5.851 / 3.526 |
| Blurry | Unknown | Unknown | 1.000 | 29 / 29 | 1.000 / 1.000 | 0.339 / 0.585 |

The full-orbit proxy becomes `Moderate`: object-only evidence retains just 0.333
parallax-proxy support and tracking misses one endpoint, so it does not satisfy the
High rule. This conservative change is retained rather than weakening the rule to
fit the expected name. Blur remains `Unknown` even though its surviving matches
show a high parallax proxy; the blur gate takes precedence. All overall scores,
subscores, and v1 baselines were identical between modes.

These layered 2D fixtures test plumbing and failure gates. They are not recordings
of one physical object and do not establish real-world accuracy.

## Prepared physical-object protocol

Copy
`evaluation/object_focused_validation/physical_object_manifest_template.csv`, then
record the same static object and scene as front-only, quarter-orbit, full-orbit,
rotation-only, and blurred videos. Keep one `object_id`, enter the normalized
middle-frame box for each recording, and run:

```bash
python scripts/compare_object_focused_validation.py \
  --input-dir /path/to/five_recordings \
  --manifest /path/to/manifest.csv \
  --output-dir evaluation/object_focused_validation/physical_run \
  --require-complete-scenario-set
```

The flag refuses an incomplete scenario set or mixed object IDs. The output CSV
contains categories, primary reasons, match/parallax/repeat evidence, tracking,
background contamination, separate wall/CPU timings, and a score-invariance check.

## Existing real MOV comparison

The repository has three unlabeled phone captures of the same bottle, not the five
controlled protocol recordings. With 16 samples and a 480-pixel maximum side:

| Video | Whole | Object-focused | Tracking | Median verified long-gap matches whole/object | CPU seconds whole/object | Explanation |
|---|---|---|---:|---:|---:|---|
| IMG_3913.MOV | Unknown | Unknown | 0.0625 | 11 / 0 | 65.596 / 65.093 | Whole descriptors ambiguous; object region too low-texture to track. |
| IMG_3914.MOV | Unknown | Unknown | 0.0625 | 5 / 0 | 36.172 / 36.406 | Insufficient whole matches; cropped/blurred object region could not track. |
| IMG_3915.MOV | Unknown | Unknown | 0.0625 | 11 / 0 | 36.364 / 35.444 | Whole descriptors ambiguous; motion-blurred object region could not track. |

Because no long-gap pair retained two tracked object boxes, background dominance is
`n/a`; there is no valid basis for claiming either object or background dominance.
All three suitability-score paths were unchanged within this comparison. These
videos lack independent diversity labels and downstream reconstruction results, so
there are no defensible misclassification or accuracy statistics. The raw table is
`evaluation/object_focused_validation/comparison.csv`.

The `IMG_3915` whole-frame wall time includes one bounded retry after OpenCV/FFmpeg
timed out while reopening the MOV. The CSV records retry counts separately; decoder
failure is never converted into a viewpoint category.

## Exact upload-path benchmark

`scripts/benchmark_streamlit_upload.py` calls the same upload payload function as
the Streamlit app, then times diagnostic rendering separately. On `IMG_3914.MOV`
with production defaults:

| Mode | Upload-to-result wall / CPU | Decode wall / CPU | Feature extraction wall | Matching/viewpoint wall | Deferred six-frame diagnostics wall / CPU |
|---|---:|---:|---:|---:|---:|
| Whole | 8.779 / 36.836 s | 6.860 / 33.834 s | 0.369 s | 1.335 s | 4.405 / 23.829 s |
| Object-focused | 9.152 / 38.104 s | 7.127 / 34.107 s | 0.360 s | 1.480 s | 5.634 / 25.984 s |

The source SHA-256 was unchanged. Decoding dominates; ORB is extracted once and
reused, object-focused live inference does not duplicate whole-frame long-gap
classification, and diagnostics are not part of upload-to-result. OpenCV uses
multiple CPU threads, so process CPU can exceed wall time. These are single local
runs, not cross-platform guarantees.

## Remaining risks

- A loose box can include table/background texture and make the tracked region look
  reliable even when the object itself is not.
- Symmetric or repetitive object texture can yield consistent but wrong matches.
- Camera rotation or a planar object can mimic image change without translation.
- Incorrect focal length, principal point, crop, stabilization, or digital zoom can
  bias optional intrinsic-calibrated indicators.
- A static background and moving object violate the method's camera-motion
  assumption; a mostly static physical object is required.
- Occlusion, specularity, motion blur, and a viewpoint that replaces most visible
  object texture can break affine box propagation even during a useful orbit.
