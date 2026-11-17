# Heuristic scoring audit and v2 evaluation

Audit date: 2026-11-17. V2 was introduced after independent human review identified v1 overestimation. The four reviewed files were used to identify failure modes, not as regression targets: no score was hardcoded, thresholds were not fitted, and three reviews contain no numerical human score.

## Preserved evaluation evidence

- `evaluation/real_video_human_review.csv` stores the feedback. Missing human scores are empty with `missing_not_provided`; they are never imputed.
- `evaluation/baseline_v1_manifest.csv` fixes the four source/report SHA-256 values and original v1 scores.
- Original reports remain in `reports/real_tests/`; v2 reports are separate under `reports/real_tests/v2/`.
- The accessible `orbit_good.mp4` was fully decoded: 24 frames at 12.0 FPS, first timestamp 0.0 s, last timestamp 1.916667 s, nominal duration 2.0 s. The human recollection of roughly four seconds is retained separately.

## Why v1 overestimated

V1 used an additive weighted mean:

```text
0.20 visual + 0.15 exposure + 0.20 feature trackability
+ 0.30 geometric consistency + 0.15 viewpoint motion
```

This allowed saturated appearance/feature scores to compensate for uncertain viewpoint coverage. For example:

| File | Visual contribution | Exposure | Features | Geometry | Motion | V1 total |
|---|---:|---:|---:|---:|---:|---:|
| IMG_3913.MOV | 20.00 | 15.00 | 20.00 | 22.94 | 14.88 | 92.82 |
| IMG_3914.MOV | 19.74 | 15.00 | 19.01 | 16.94 | 14.76 | 85.44 |
| IMG_3915.MOV | 17.12 | 15.00 | 15.89 | 12.16 | 13.67 | 73.85 |
| orbit_good.mp4 | 19.42 | 14.69 | 20.00 | 30.00 | 4.46 | 88.57 |

The last case is the clearest defect: viewpoint motion contributed only 15% of the total, so perfect feature/geometry matches from homography-dominant planar motion overwhelmed weak depth-bearing evidence.

V1 also measured displacement per sampled pair rather than per second, did not quantify short-gap track survival, and treated every robust match as equally useful even when motion was repetitive, rotation-dominant, or background-driven.

## V2 measurements

V2 preserves all v1 raw fields and adds:

- forward/backward pyramidal Lucas–Kanade optical-flow retention;
- displacement and velocity normalized by image diagonal and timestamp delta;
- sudden speed ratios, direction changes, and rotation-rate changes;
- descriptor ambiguity as a repetitive-feature warning;
- raw homography dominance independent of whether displacement exceeds the old threshold;
- compressed bitrate, bits/pixel/frame, source megapixels, and analysis scaling;
- explicit short-gap evaluability: shake/LK retention require pair gaps ≤0.30 s. Longer gaps are reported as unresolved and use ORB continuity as a fallback.

The 0.30-second rule matters for `IMG_3913.MOV`: 32 samples over 16.62 seconds create roughly 0.53-second gaps. An initial v2 run incorrectly interpreted sparse LK direction changes as shake; the final implementation marks shake unresolved for those gaps instead of penalizing it.

## V2 formula and classification

V2 uses a weighted geometric mean so one weak required evidence family cannot be fully hidden by saturated appearance metrics:

| Subscore | V2 weight |
|---|---:|
| Visual quality | 12% |
| Exposure/lighting | 8% |
| Feature trackability | 15% |
| Geometric consistency | 35% |
| Viewpoint motion/reliability | 30% |

Feature trackability now includes short-gap LK retention (or an explicit ORB fallback), overlap survival, and descriptor ambiguity. Geometric consistency directly includes overlap and track continuity. Viewpoint motion includes timestamp-normalized speed, regularity, retention, homography residual/non-homography evidence, and cumulative observed motion.

Classification thresholds remain unchanged and explicitly uncalibrated:

- `likely_suitable`: score ≥70;
- `review_recommended`: 45–69.99;
- `unlikely_suitable`: below 45.

## Documented physical gates

Every report contains all gate evaluations, trigger state, evidence, and physical basis. Triggered gates cap a score but never raise it.

| Gate | Maximum | Physical condition |
|---|---:|---|
| Insufficient temporal evidence | 25 | Fewer than six samples, fewer than three pairs, or duration below 0.75 s |
| Geometry unavailable | 35 | Robust geometry on less than 10% of pairs |
| Stationary view | 35 | At least 80% nearly stationary pairs |
| Rotation/planar coverage uncertain | 44 | Reliable, distributed features but ≥80% homography-only cues with low residual, plus short duration or weak motion score |
| Persistent low overlap | 60 | At least half of pairs lose overlap and geometry/track retention is weak |
| Fast motion with track loss | 60 | High timestamp-normalized speed on at least half of pairs plus blur, overlap loss, or track loss |
| Irregular motion with overlap loss | 60 | Frequent short-gap direction/speed discontinuity plus overlap loss |
| Severe irregular motion | 60 | Irregularity on at least 70% of evaluable pairs at material speed |
| Severe exposure failure | 55 | At least 90% severely under- or overexposed frames |

The rotation gate requires at least 500 median keypoints, 30% spatial coverage, and 50% track retention. This avoids treating a low-resolution-derived homography cue as reliable proof of failure. No gate uses low source resolution alone.

## Before/after on reviewed videos

| File | Human evidence | V1 | V2 | Change | V2 label | Key v2 evidence |
|---|---|---:|---:|---:|---|---|
| IMG_3913.MOV | Approx. 80; good but overrated | 92.82 | 78.54 | -14.28 | likely suitable | Long-gap shake unresolved; ORB fallback; low overlap 12.90% |
| IMG_3914.MOV | Too fast; no numeric score | 85.44 | 60.00 | -25.44 | review recommended | Median speed 0.2638 diagonals/s; p90 0.5842; irregularity 64.52%; low overlap 38.71% |
| IMG_3915.MOV | Significant shake; no numeric score | 73.85 | 48.51 | -25.34 | review recommended | Median speed 0.3754; retention 5.2%; irregularity 67.74%; low overlap 54.84% |
| orbit_good.mp4 | Very poor; no numeric score | 88.57 | 44.00 | -44.57 | unlikely suitable | 100% rotation/planar cue, 2.0 s duration, rotation/planar gate |

These changes are qualitatively consistent with the supplied comments, but this is **not evidence of improved accuracy**: only one review has a numerical estimate, all four were involved in the failure-mode audit, and no downstream reconstruction outcomes exist.

## Source resolution, compression, and sharpness

| File group | Resolution/codec | Estimated bitrate | Approx. compressed bits/pixel/frame | Analysis sharpness |
|---|---|---:|---:|---|
| IMG_3913–3915 | 2160×3840 HEVC | 55.23–55.57 Mbps | 0.1110–0.1117 | Median Laplacian variance 153.88–498.27 at 360×640 analysis size |
| orbit_good.mp4 | 480×360 FMP4 | 1.06 Mbps | 0.5119 | Median Laplacian variance 417.43 at native size |

Bitrate and bits/pixel/frame are container-level approximations, not codec-quality ground truth. The MP4 has far fewer source pixels but strong measured sharpness and trackability; V2 lowers it for viewpoint/coverage evidence, not for resolution.

## Controlled synthetic regression suite

| Fixture | V1 | V2 | V2 result/gate |
|---|---:|---:|---|
| Smooth multi-layer orbit | 95.17 | 93.89 | likely suitable; no cap |
| Stationary view | 44.00 | 35.00 | stationary gate |
| Pure rotation | 87.80 | 44.00 | rotation/planar gate |
| Shaky orbit | 97.84 | 44.00 | rotation/planar plus severe-irregularity evidence |
| Fast motion | 88.86 | 60.00 | fast-motion/track-loss gate |
| Low resolution, otherwise smooth | 90.84 | 90.51 | no resolution or rotation cap |
| Blurred orbit | 61.69 | 67.75 | review recommended; no cap |
| Severe darkness | 55.00 | 55.00 | severe-exposure gate |
| Insufficient coverage | 89.94 | 25.00 | temporal-evidence gate |

The blurred fixture increasing by 6.06 points is a regression risk: the geometric mean places more weight on still-strong correspondence evidence. It remains below the smooth orbit and is flagged as blurred, but needs independent real blurred videos before its absolute score can be trusted.

## False positives, possible regressions, and unresolved uncertainty

- V1 produced four reviewed high-score false-positive concerns. V2 makes the corresponding measured failures visible and changes three classifications, but the reviewed set cannot validate generalization.
- There are no additional real videos in the accessible workspace. Therefore the required held-out real-video evaluation is blocked by data availability. Synthetic fixtures are regression tests, not a substitute.
- No real downstream reconstruction success/completeness measurements are available.
- `IMG_3913.MOV` shake is unresolved at the current uniform sampling cadence; increasing only total uniform samples would increase HEVC decoding cost. A future two-rate sampler should use sparse whole-video coverage plus short local bursts.
- Flow and ORB may follow a textured background rather than the object. Object segmentation or track ownership remains unresolved.
- The rotation/planar gate can be conservative for genuinely planar targets or short clips that still meet a limited reconstruction goal.
- Repetitive-feature ambiguity is only a warning/proxy and is not independently calibrated.
- Compression diagnostics do not measure motion-compensation artifacts or rolling shutter directly.

No ML model was trained, and no heuristic-generated score was used as a supervised target.

## Revision 2.1 — blurred-orbit regression repair (2026-11-17)

The competition-MVP audit reproduced the documented blurred-orbit regression without
using the four reviewed videos as score targets. With the controlled 400×300,
24-frame synthetic fixture and 16 uniform samples, v1 scored 61.69 while v2 scored
67.75 even though all frames were flagged blurred and median Laplacian variance was
only 4.78. Coarse feature correspondences were strong enough to compensate for a
visual-quality subscore near the floor.

Revision 2.1 retains every weight and classification threshold. It adds one visible
physical gate:

- `persistent_severe_blur`, maximum 55, requires both blur on at least 80% of sampled
  frames and median Laplacian variance below 15 at analysis resolution.

The conjunction is deliberately stricter than the ordinary blur flag. It does not
use source resolution and does not penalize a single blurred interval as a complete
capture failure. Physical basis: nearly universal severe blur removes repeatable
local detail even when coarse background/shape correspondences survive.

A second sampling-invariance defect was found while testing the default 32-sample
path. Dense adjacent views of the synthetic multilayer parallax orbit were locally
homography-like, which could trigger the rotation/planar gate. Reliable homography
dominance now additionally requires a median homography inlier ratio of at least
0.90. This keeps the controlled pure-rotation failure while rejecting the multilayer
case where a material fraction of matches does not fit one plane.

Observed controlled results:

| Fixture/configuration | v1 | Revision 2.1 | Triggered gate |
|---|---:|---:|---|
| Smooth multilayer orbit, 16 samples | 95.17 | 93.89 | none |
| Pure rotation, 16 samples | 87.80 | 44.00 | rotation/planar uncertainty |
| Persistently blurred orbit, 16 samples | 61.69 | 55.00 | persistent severe blur |
| Sharp low-resolution orbit, 16 samples | 90.84 | 90.51 | none |
| Smooth multilayer orbit, default dense sampling | 90.12 | 92.71 | none |
| Pure rotation, default dense sampling | 44.00 | 35.00 | stationary + rotation/planar |
| Persistently blurred orbit, default dense sampling | 58.26 | 55.00 | persistent severe blur |

The top-level `scoring_version` remains `heuristic_v2` for JSON compatibility and an
additive `scoring_revision: "2.1"` field makes the repair explicit. Preserved v2.0
real-video reports and evaluation CSVs were not regenerated or overwritten. This is
a controlled regression repair, not evidence of improved real-world accuracy.

As a non-fitting regression check, the four preserved real-report summaries were
replayed through revision 2.1 without decoding or writing new reports. Their scores
were unchanged: 78.54, 60.00, 48.51, and 44.00 (all deltas 0.00). This stability is
not an accuracy claim; it only shows that the synthetic repair did not silently
retarget those four known outputs.
