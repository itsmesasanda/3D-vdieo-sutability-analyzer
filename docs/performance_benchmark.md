# 4K MOV analysis optimization benchmark

Benchmark date: 2026-11-17. Full raw values are in
`benchmarks/4k_mov_before_after.json`.

## Input and host

- `IMG_3913.MOV`, unchanged SHA-256
  `0c9ca3a9b4ebc77ce0f2bf4a5f496855c877a6beeb3b80e3b59957cd2b27a42a`
- HEVC, 2160×3840, 997 frames, 59.994 FPS, 16.618 seconds
- 32 uniform full-timeline samples, maximum analysis side 640
- MacBook Air, Apple M3, 8 CPU cores, 8 GiB RAM, macOS 14.5

## Profile finding

The old decoder made 32 independent random seeks into a long-GOP HEVC stream.
Isolated sampling took 27.264 and 26.194 seconds (mean 26.729), while quality and
geometry together were only about 1.29 seconds. Decoding—not scoring—was the clear
bottleneck.

## Change

When uniform targets average no more than 64 source frames apart, the sampler now
makes one forward `grab()` pass and retrieves/resizes only the 32 requested frames.
Memory remains bounded by the requested sample count. Videos with wider spacing keep
the sparse-seek strategy, so a very long source is not always decoded end to end.
If a forward pass misses any requested position, the code retries with the original
sparse-seek implementation rather than accepting reduced coverage.

## Measured before/after

| Measurement | Before | After |
|---|---:|---:|
| End-to-end run 1 | 29.478 s | 15.103 s |
| End-to-end run 2 | 25.558 s | 13.782 s |
| Mean | 27.518 s | 14.443 s |
| Speedup | — | 1.905× |
| Mean runtime reduction | — | 47.516% |

A direct same-process decoder comparison measured 22.268 seconds for random seeking
and 12.833 seconds for one forward pass. All 32 requested indices and all resized
frame pixels were identical.

## Reproducibility check

The optimized report was compared field-for-field with the preserved pre-optimization
v2 report. Overall score (78.54), all five subscores, quality summary, geometry
summary, every sampled-frame quality record, every sampled-pair geometry record, and
all problem segments were identical. The only analysis metadata addition is the
explicit `decode_strategy` field. Scoring revision 2.1 adds a separate documented
blur regression fix, but that gate does not trigger for this video.

## Limitations

This is a two-run local engineering benchmark, not a universal performance claim.
OS cache state, OpenCV/codec builds, CPU, and video GOP structure affect timing. The
new `scripts/benchmark_video_analysis.py` command records source hashes, timing,
coverage, decode strategy, score, and optional reference-report semantic equality
for repeatable future measurements.

