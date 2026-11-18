# Estimated Viewpoint Diversity v1

## Scope

This module reports `Low`, `Moderate`, `High`, or `Unknown` as a separate result.
It does not change heuristic v1/v2.1, an optional Random Forest output, or any ML
feature schema. It does not run COLMAP. The result is not an estimate of camera
travel angle, 360-degree coverage, or percentage of object surface captured.

## Evidence path

1. Reuse the uniformly sampled, downscaled frames from `video.py`.
2. Reuse one ORB feature cache with the existing adjacent-pair geometry path.
3. Reuse the ratio test, fundamental-matrix RANSAC, homography RANSAC, spatial
   spread, displacement, and residual calculations for at most 15 additional
   quarter-gap, half-gap, and endpoint pairs.
4. Mark likely repeated views only when a separated pair has at least 15 verified
   matches, a verified-match ratio of at least 0.05, normalized displacement no
   greater than 0.006, and homography support of at least 0.80.
5. Treat a pair as a cautious parallax proxy only when fundamental geometry has at
   least 12 verified, spatially distributed matches, visible displacement, and the
   evidence is not strongly explained by a low-residual global homography.
6. If calibrated intrinsics are supplied, scale them to analysis resolution, run
   essential-matrix RANSAC and `recoverPose`, and report relative rotation,
   translation direction without scale, and rotation-compensated bearing change.

The category thresholds are declared engineering defaults tested against synthetic
regression fixtures. They are not calibrated probabilities or real-world accuracy
statistics.

## Insufficient evidence

The result becomes `Unknown` when evidence is dominated by blur, low texture, too
few RANSAC-valid pairs, highly ambiguous/repetitive descriptors, or changed pairs
that are mostly explained by one global homography. The latter protects against
claiming useful translation from an in-place rotation or planar motion. A symmetric
object can still be ambiguous even when feature counts are high, so the raw pair
evidence and warning remain part of the report.

## Optional object focus

A normalized bounding box on the representative middle frame enables the
object-focused path in `object_focus.py`. The box is propagated in both temporal
directions with in-box ORB matches and affine RANSAC. The shared feature cache is
then split into object and background evidence. Object-only matches feed the
category; background matches remain diagnostic and never increase it. Tracking
below 70%, background dominance on at least 60% of evaluable long-gap pairs, or the
usual blur/texture/ambiguity/rotation gates returns `Unknown`.

This is rectangle tracking, not semantic segmentation. It does not turn appearance
change into a surface-coverage claim. See `docs/object_focused_validation.md` for
the UI/CLI workflow, prepared physical-object protocol, comparisons, and measured
production-path timing.

Real-video diagnostic inspection added a conservative cross-view guard: if the
median strict descriptor-ambiguity fraction across separated pairs is at least
0.65, the result is `Unknown`. This strict metric treats near-identical second-best
ORB descriptors as ambiguous, including the exact-tie edge case that an ordinary
ratio calculation can miss. It does not change the preserved descriptor metric used
by heuristic v2.1 or the Random Forest feature schema.

## Controlled scenario audit

Run:

```bash
python scripts/benchmark_viewpoint_diversity.py \
  --out benchmarks/viewpoint_diversity_synthetic.json
```

On the recorded local Apple Silicon CPU run (Python 3.9.6, OpenCV 4.14.0, 16 frames
at a maximum 400-pixel side), the observed categories matched the declared fixture
expectations:

| Synthetic video | Expected | Observed |
|---|---|---|
| Front-only | Low | Low |
| Quarter-orbit proxy | Moderate | Moderate |
| Periodic full-orbit proxy | High | High |
| Rotation-only | Unknown | Unknown |
| Blurry orbit | Unknown | Unknown |

The five complete end-to-end runs took 1.893 seconds of wall time and 3.514 seconds
of aggregate process CPU time in that single run. With decoded frames, quality,
shared ORB features, and adjacent geometry already available, the new module itself
took 0.350 seconds wall and 0.833 seconds process CPU across all five videos. OpenCV
used multiple CPU threads, so CPU time exceeds wall time. These deterministic 2D layered videos validate
plumbing, gates, and category separation; they are not physically rendered scenes
or evidence of real-video accuracy. Raw timings are in
`benchmarks/viewpoint_diversity_synthetic.json`.

## Real-video audit

`scripts/validate_viewpoint_real_videos.py` recursively processes local MP4/MOV and
other supported videos through the unchanged main pipeline, writes a flat CSV, and
renders selected long-gap match/motion diagnostics. It records process CPU and wall
time separately. An optional filename-keyed intrinsics manifest is supported, but
intrinsics are never guessed.

The accessible historical folder contains three real phone MOV captures plus one
synthetic MP4 regression fixture. The three phone videos return `Unknown` because
their median separated-pair ambiguity is 0.786–0.805 despite visible motion; selected
long-gap pairs contain as few as 4–5 verified matches in the fast/shaky videos. The
synthetic fixture returns `Moderate`. This cohort has no independent viewpoint-band
ground truth and cannot establish accuracy. Detailed measurements, risks, and local
diagnostic links are in `evaluation/viewpoint_real_validation/REPORT.md` and
`validation.csv`.

Object boxes were also reviewed for the three MOVs. Each bottle region retained a
reliable track on only 1 of 16 validation samples, so object-focused analysis stayed
`Unknown` with zero verified object-only long-gap matches. This is tracking failure,
not proof of low viewpoint diversity. Full results are in
`evaluation/object_focused_validation/comparison.csv`.

