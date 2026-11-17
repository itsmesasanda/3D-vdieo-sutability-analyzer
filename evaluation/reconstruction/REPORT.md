# Milestone 3A — reconstruction-grounded validation status

Date: 2026-11-17 (Asia/Colombo)

## Outcome

The local pilot reached the COLMAP execution boundary but could not run sparse
mapping because COLMAP is not installed. Installing it locally was not attempted:
the host had only 1.820 GiB free during the final audit and Homebrew's COLMAP formula
requires a large dependency tree. This is an **infrastructure-unavailable result**,
not a failed reconstruction and not evidence for or against the v2 score.

The pilot extraction itself succeeded. `IMG_3913.MOV` produced 35 timestamped,
upright JPEGs at 2 FPS and 900×1600, spanning 0.000 to 16.602 seconds. The extractor
reported all requested frames decoded, 1.0 temporal coverage, and a measured runtime
of 11.517968 seconds. Its source SHA-256 was identical before and after extraction.
The files remain separated under `IMG_3913/`; the other sequences were not extracted
or mapped because the requested pilot-success gate was not met.

No registered-image count, registration ratio, reconstructed point count,
reprojection error, track length, camera pose, PLY, sparse preview, or human
reconstruction rating is available. Those fields are blank in `results.csv`; none
has been inferred from the filenames or heuristic.

## Host audit

| Capability | Measured status |
|---|---|
| Host | MacBook Air, Apple M3 arm64 |
| CPU | 8 cores (4 performance, 4 efficiency) |
| Memory | 8 GiB |
| GPU | Integrated Apple M3, 8 cores, Metal; no CUDA; unused by CPU workflow |
| Free disk | 1.820 GiB of 228.274 GiB total |
| COLMAP | Not found |
| ffmpeg / ffprobe | Not found / not found |
| pycolmap | Not importable |
| OpenCV | 4.14.0 |
| Homebrew | 7.0.9 |

The machine is computationally capable of a small CPU sparse experiment, but the
missing executable and storage margin prevent a responsible local installation and
artifact run. `environment_audit.json` contains the raw audit.

## Preserved predictions versus reconstruction evidence

| Video | Baseline v1 | Unchanged v2 | V2 label | Reconstruction status | Objective outcome |
|---|---:|---:|---|---|---|
| IMG_3913.MOV | 92.82 | 78.54 | likely suitable | infrastructure unavailable | no COLMAP measurements |
| IMG_3914.MOV | 85.44 | 60.00 | review recommended | not run: pilot gate | no COLMAP measurements |
| IMG_3915.MOV | 73.85 | 48.51 | review recommended | not run: pilot gate | no COLMAP measurements |
| orbit_good.mp4 | 88.57 | 44.00 | unlikely suitable | not run: pilot gate | no COLMAP measurements |

The score-report hashes and source-video hashes in `results.csv` establish that the
existing evidence was read, not regenerated or overwritten. These predictions
cannot yet be validated against reconstruction outcomes. No v2 weight or threshold
was changed, and no ML model was trained.

## Experiment design and interpretation

- Frame timestamps and per-frame hashes are stored in `IMG_3913/frames_manifest.csv`.
- OpenCV orientation auto-handling was enabled; an extracted pilot frame was
  visually checked upright. Encoded and decoded dimensions/orientation metadata are
  retained in `extraction_metadata.json`.
- The prepared COLMAP path shares one `OPENCV` camera across a sequence, disables GPU
  SIFT when the installed version exposes that option, uses sequential matching,
  and can retry exhaustive matching only after sequential mapping yields no model.
- The model analyzer reports registration, sparse points, reprojection error, track
  length, and arbitrary-scale camera-center extent. It explicitly does not convert
  these into surface completeness.
- `human_review_form.csv` keeps object completeness, geometric plausibility, missing
  regions, artifacts, and practical usability separate from objective COLMAP data.
- Sparse output is never described as dense reconstruction.

## Reproducible commands and observed results

Environment audit (first execution failed because sandbox-denied `sysctl` output was
not yet distinguished from successful output; the script was fixed and rerun):

```bash
.venv/bin/python scripts/reconstruction/audit_environment.py \
  --out evaluation/reconstruction/environment_audit.json
```

Final observed result: success; values are shown in the host table.

Pilot extraction:

```bash
.venv/bin/python scripts/reconstruction/extract_frames.py \
  --video data/samples/real_tests/IMG_3913.MOV \
  --output-dir evaluation/reconstruction/IMG_3913 \
  --sample-fps 2 --max-side 1600 --jpeg-quality 95
```

Observed: exit 0, 35/35 frames, 11.517968 s, full declared timeline, unchanged source.

COLMAP pilot gate:

```bash
.venv/bin/python scripts/reconstruction/run_colmap.py \
  --images-dir evaluation/reconstruction/IMG_3913/images \
  --workspace-dir evaluation/reconstruction/IMG_3913/colmap \
  --camera-model OPENCV --sequential-overlap 10 --retry-exhaustive
```

Observed: exit 3, `infrastructure_unavailable`, no command launched, no objective
metrics, mapper runtime null.

Evaluation consolidation:

```bash
.venv/bin/python scripts/reconstruction/build_evaluation.py
```

Observed: four rows written to `results.csv`; blank human-review form created.

Initial complete test run after adding reconstruction utilities:

```text
16 passed in 2.90s
```

Final syntax, test, evaluation-build, and checksum verification completed with
`16 passed in 4.55s`. The checksum command emitted only a macOS locale fallback
warning; all 12 hashes matched the preserved evidence.

## Next required step

Run the documented CPU workflow on a Linux/Colab environment with COLMAP already
available (free-tier only unless separately approved), or free at least several
additional GiB and install COLMAP locally with explicit approval. Return the result
directories for consolidation and human inspection. The pilot must succeed before
the remaining three videos are launched.

