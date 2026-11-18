# Real-world Estimated Viewpoint Diversity validation

This is a small, non-held-out diagnostic audit. The videos have independent capture-quality comments, but no verified Low/Moderate/High viewpoint-diversity ground truth. The results therefore do not establish real-world accuracy. The folder contains three phone MOV captures and `orbit_good.mp4`, a deterministic synthetic regression fixture retained in the historical real-test folder; only the three MOV files count as real-world evidence.

Input directory: `data/samples/real_tests`

## Results

| Video | Viewpoint | Evidence | Valid geometry | Median verified nearby / separated | Parallax proxy | Repeated pairs | Planar/rotation | CPU seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IMG_3913.MOV | Unknown | insufficient | 1.000 | 76.0 / 11.0 | 0.308 | 0 | 0.000 | 154.905 |
| IMG_3914.MOV | Unknown | insufficient | 0.932 | 42.0 / 11.0 | 0.231 | 0 | 0.000 | 70.396 |
| IMG_3915.MOV | Unknown | insufficient | 0.886 | 17.0 / 10.0 | 0.231 | 0 | 0.000 | 73.194 |
| orbit_good.mp4 | Moderate | sufficient | 1.000 | 917.0 / 456.0 | 1.000 | 0 | 0.371 | 1.588 |

The parallax column is the temporally separated-pair fraction, not the more
optimistic adjacent-plus-separated fraction. The first pass classified all three
phone videos as `High`. Diagnostic inspection showed that the median long-gap pair
had only 10–11 verified matches and that median strict descriptor ambiguity was
0.786–0.805. Matches visibly span the patterned tablecloth, chair slats, background,
and a dark approximately symmetric object. A conservative `>=0.65` cross-view
ambiguity gate was therefore added to the viewpoint module, without changing either
suitability score. The controlled synthetic expectations still pass.

## Potentially misleading cases and misclassification review

A viewpoint category is not the overall suitability score. Fast motion, shake, blur, or low overlap can coexist with broad viewpoint change. Such a case is not automatically a diversity-category error, but it is a likely user-interpretation error if `High` is read as 'good capture'.

### IMG_3913.MOV

- Predicted `Unknown` with `insufficient` heuristic evidence.
- Independent capture-quality context (not viewpoint ground truth): good but overrated | Human estimates approximately 80/100; good capture but baseline_v1 is overrated.
- `object_motion_unresolved`: The method assumes a static object and cannot reliably separate object motion from camera motion.
- `background_feature_dominance_unresolved`: ORB matches are not segmented to the object, so a textured static background can dominate geometry and inflate apparent evidence.
- `symmetry_or_repetitive_appearance`: Repeated or symmetric descriptors can support geometrically plausible but incorrect correspondences.
- `intrinsics_unavailable_for_pose_disambiguation`: No calibrated intrinsics were supplied, so essential-matrix pose and rotation-compensated parallax are unavailable.
- Misclassification assessment: this may be a conservative false negative because the independent review calls the capture good and the images visibly change around the object. `Unknown` is used because only 11 long-gap matches survive at the median and ambiguity is 0.805; it does not assert that diversity is low.
- Diagnostic images:
  - [parallax_proxy](diagnostics/IMG_3913_MOV/01_parallax_proxy_000000_000257.jpg) — 16 verified matches.
  - [largest_view_change](diagnostics/IMG_3913_MOV/02_largest_view_change_000450_000707.jpg) — 10 verified matches.
  - [weakest_geometry](diagnostics/IMG_3913_MOV/03_weakest_geometry_000289_000803.jpg) — 10 verified matches.

### IMG_3914.MOV

- Predicted `Unknown` with `insufficient` heuristic evidence.
- Independent capture-quality context (not viewpoint ground truth): score too high | too much camera speed | Human states camera speed is too high and baseline_v1 score is too high; no numerical human score was assigned.
- `object_motion_unresolved`: The method assumes a static object and cannot reliably separate object motion from camera motion.
- `background_feature_dominance_unresolved`: ORB matches are not segmented to the object, so a textured static background can dominate geometry and inflate apparent evidence.
- `symmetry_or_repetitive_appearance`: Repeated or symmetric descriptors can support geometrically plausible but incorrect correspondences.
- `intrinsics_unavailable_for_pose_disambiguation`: No calibrated intrinsics were supplied, so essential-matrix pose and rotation-compensated parallax are unavailable.
- Misclassification assessment: `Unknown` is appropriate for the available evidence. The independent review reports excessive camera speed, and selected long-gap pairs fall as low as 4–5 verified matches. A broad camera path may still exist, but it cannot be separated confidently from background/repetitive-pattern support.
- Diagnostic images:
  - [parallax_proxy](diagnostics/IMG_3914_MOV/01_parallax_proxy_000194_000400.jpg) — 16 verified matches.
  - [largest_view_change](diagnostics/IMG_3914_MOV/02_largest_view_change_000116_000323.jpg) — 5 verified matches.
  - [weakest_geometry](diagnostics/IMG_3914_MOV/03_weakest_geometry_000000_000103.jpg) — 4 verified matches.

### IMG_3915.MOV

- Predicted `Unknown` with `insufficient` heuristic evidence.
- Independent capture-quality context (not viewpoint ground truth): score too high | significant camera shake | Human states there is significant camera shake and baseline_v1 score is too high; no numerical human score was assigned.
- `object_motion_unresolved`: The method assumes a static object and cannot reliably separate object motion from camera motion.
- `background_feature_dominance_unresolved`: ORB matches are not segmented to the object, so a textured static background can dominate geometry and inflate apparent evidence.
- `symmetry_or_repetitive_appearance`: Repeated or symmetric descriptors can support geometrically plausible but incorrect correspondences.
- `intrinsics_unavailable_for_pose_disambiguation`: No calibrated intrinsics were supplied, so essential-matrix pose and rotation-compensated parallax are unavailable.
- Misclassification assessment: `Unknown` is appropriate for the available evidence. The independent review reports significant shake, and selected long-gap pairs fall to 4 verified matches. The result deliberately withholds a category rather than interpreting large image displacement as translation.
- Diagnostic images:
  - [parallax_proxy](diagnostics/IMG_3915_MOV/01_parallax_proxy_000240_000347.jpg) — 13 verified matches.
  - [largest_view_change](diagnostics/IMG_3915_MOV/02_largest_view_change_000307_000414.jpg) — 4 verified matches.
  - [weakest_geometry](diagnostics/IMG_3915_MOV/03_weakest_geometry_000120_000334.jpg) — 4 verified matches.

### orbit_good.mp4

- Predicted `Moderate` with `sufficient` heuristic evidence.
- Independent capture-quality context (not viewpoint ground truth): very poor | Human judges quality very poor and far overrated. Human recalls roughly four seconds; complete decode of the accessible file verifies 24 frames at 12 FPS, nominal duration 2.0 seconds, last frame timestamp 1.916667 seconds.
- `object_motion_unresolved`: The method assumes a static object and cannot reliably separate object motion from camera motion.
- `background_feature_dominance_unresolved`: ORB matches are not segmented to the object, so a textured static background can dominate geometry and inflate apparent evidence.
- `intrinsics_unavailable_for_pose_disambiguation`: No calibrated intrinsics were supplied, so essential-matrix pose and rotation-compensated parallax are unavailable.
- Misclassification assessment: `Moderate` should not be read as contradicting the “very poor” capture-quality review. This file is a synthetic fixture with a highly textured static background, and the separate unchanged suitability score is 44 (`unlikely_suitable`). Its 456 median verified long-gap matches can be dominated by synthetic background structure, so it is not real-world validation evidence.
- Diagnostic images:
  - [parallax_proxy](diagnostics/orbit_good_mp4/01_parallax_proxy_000011_000023.jpg) — 311 verified matches.
  - [largest_view_change](diagnostics/orbit_good_mp4/02_largest_view_change_000000_000023.jpg) — 475 verified matches.
  - [weakest_geometry](diagnostics/orbit_good_mp4/03_weakest_geometry_000000_000012.jpg) — 317 verified matches.

## Interpretation limits

- ORB/RANSAC can lock onto the background rather than the object; the current pipeline has no object mask.
- Symmetric and repeated surfaces can create internally consistent false matches.
- Rotation-only and planar motion may fit a homography and do not prove camera translation.
- Intrinsics are optional and never guessed. If provided, crop, stabilization, digital zoom, or inaccurate calibration can bias essential-matrix results.
- Repeated endpoints may be loop closure or a repeated pass; they do not prove a full orbit.
- No surface-coverage percentage or exact orbit angle is inferred.

Failures: 0. Successful videos: 4.
