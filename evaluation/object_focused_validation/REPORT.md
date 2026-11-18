# Object-focused real-video comparison

This comparison is diagnostic, not a real-world accuracy estimate. Scenario labels are capture-protocol metadata only when a human has supplied them; synthetic expectations are not ground truth for these recordings.

Configuration: `16` samples, `480px` maximum analysis side.

| Video | Scenario | Whole | Object | Tracking | Background-dominated | Verified matches whole/object | CPU whole/object (s) | Decode retries whole/object | Score unchanged |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| IMG_3913.MOV | unlabeled_real_capture | Unknown | Unknown | 0.062 | n/a | 11.0 / 0.0 | 65.596 / 65.093 | 0 / 0 | True |
| IMG_3914.MOV | unlabeled_real_capture | Unknown | Unknown | 0.062 | n/a | 5.0 / 0.0 | 36.172 / 36.406 | 0 / 0 | True |
| IMG_3915.MOV | unlabeled_real_capture | Unknown | Unknown | 0.062 | n/a | 11.0 / 0.0 | 36.364 / 35.444 | 1 / 0 | True |

## Interpretation

- Unknown is intentional when tracking, verified object-only matches, descriptor uniqueness, or non-planar parallax evidence is inadequate.
- Background-only verified matches are retained as a contamination diagnostic and do not increase the object-focused category.
- Appearance change is not interpreted as exact camera travel, complete orbit, or object-surface coverage.
- Pure rotation, object symmetry, repeated texture, inaccurate intrinsics, crop/stabilization, and motion blur remain important failure modes.
- The suitability score and optional Random Forest path are not inputs to this comparison and are not modified.
