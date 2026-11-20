# Experimental object capture completeness

This panel is a conservative summary of evidence already produced by the optional
object-focused viewpoint analysis. It runs no additional feature extraction and is
separate from the heuristic score, Ridge estimate, Random Forest gate, and existing
viewpoint category.

## What it can report

- Whether the selected object was tracked across enough of the sampled timeline.
- Whether reliable long-gap object evidence is limited or contains multiple views.
- A cautious camera-motion pattern when both object and background move with
  parallax-like support.
- A cautious object-motion/static-camera pattern when the object moves against a
  verified nearly static background. This can be a turntable, but object animation
  cannot be excluded.
- A recommendation to add lateral or deliberately elevated views when the available
  evidence cannot verify them.

## What it cannot report

The current measurements do not establish an object-relative vertical camera
trajectory. Consequently, higher/lower views are always displayed as **Not
verified** in this MVP. The panel does not identify front, back, top, bottom, a
missing named side, exact orbit angle, 360-degree completion, or surface-coverage
percentage.

Whole-frame viewpoint analysis alone is insufficient for object capture guidance.
If object-focused tracking covers less than 70% of samples, the result is **Capture
coverage not verified** and includes the existing tracking or geometry reason.
Passing 70% tracking is necessary but not sufficient: background dominance, blur,
ambiguous descriptors, planar/rotation evidence, or inadequate long-gap object
geometry still keeps the result unverified.

## Interpretation

`Multiple object viewpoints detected` means only that reliable tracked-object
evidence contains substantial long-gap change and parallax-like support under the
existing viewpoint rules. It is not a completeness claim. `Limited viewpoint
variation` means the reliable tracked interval predominantly repeats similar views
or lacks adequate long-gap evidence.

