# Audit of `co3d_video_quality_assessment.ipynb`

Audit date: 2026-11-17. The notebook was read as notebook JSON in full: all 50 cells, their source, execution counts, metadata, and every stored text output were inspected. No cell was executed because the notebook contains environment mutations, large downloads, and unverified training/labeling code.

## Verified saved execution state

- Cells 2–3 were executed later (`41`, `43`) than cells 5–49 (`9`–`40`), so the saved run is not a clean top-to-bottom execution.
- Cell 2 points to Colab-only `/content/...` paths. Its saved environment had a Tesla T4, PyTorch, OpenCV, scikit-learn, matplotlib, PIL, tqdm, and a `colmap` executable. That says nothing about the current machine.
- Cell 3 attempted a full `apple` category download. The output shows multiple 15.8–18.7 GB archives downloading concurrently, only 1 of 7 archives completed, and a `KeyboardInterrupt` after about 13 minutes 47 seconds. This was not a usable small subset.
- Cell 5 reports that `/content/co3d_data` did not contain a usable requested category. Cells 7–12 consequently report no annotations/data.
- Cell 17 produced no COLMAP labels. Cell 20 reports zero usable labeled sequences. Cell 21 has an empty split.
- Cell 26 downloaded only ImageNet ResNet-18 weights (44.7 MB). These are encoder weights, not a trained video-quality model.
- Cells 28 and 33–35 show no extracted sequence features, training, training history, or fine-tuning.
- Cells 38 and 40 show no evaluation or error analysis.
- Cells 42–43 saved no model and no temporal scaler.
- Cell 47 explicitly says inference is not executable without a saved model.
- Cell 49 reports zero label, feature, model, figure, and COLMAP-workspace files and no test results.

The notebook therefore provides design ideas and partial code only. It does not provide empirical model performance, a trained predictor, or a successfully validated label pipeline.

## Cell-by-cell functional inventory

| Cells | Content | Audit disposition |
|---|---|---|
| 0–2 | Objective, Colab configuration, dependency checks | Method framing retained; hard-coded paths and runtime assumptions rejected. |
| 3–5 | CO3D download and layout verification | Download cell rejected; defensive layout verification is reusable for a future approved dataset audit. |
| 6–12 | `.jgz` annotation loading and EDA | Raw gzip/JSON loading, frame-number ordering, and missing-file checks are reusable in Milestone 2. |
| 13–17 | COLMAP driver, metric parsing, coverage, label scoring/cache | Resumable concept is reusable only after repairs and real smoke tests; not used by the heuristic MVP. |
| 18–21 | Limitations, quality gate, grouped split, threshold logic | Exclusion logging and group-level splitting are reusable; current calibration and leakage-check code are not valid. |
| 22–24 | Uniform sequence sampling, padding/mask, dataset | Whole-sequence sampling principle retained; implementation requires mask/padding fixes. |
| 25–28 | ResNet and 12 handcrafted features/cache | Explicit image/temporal metric idea retained; current implementation has runtime and preprocessing defects. |
| 29–35 | CNN-LSTM and training loop | Deferred. Architecture has an output-range defect and no training data/artifact. |
| 36–40 | Metrics, baselines, evaluation/error analysis | Baseline-first and independent-test principles retained for a future labeled phase; no saved results exist. |
| 41–47 | Artifact saving, scaler, video reader, model inference | Deferred. Training/inference parity and video coverage are broken, and no artifacts exist. |
| 48–49 | Conclusions and artifact summary | Limitations are useful, but claims that a complete trained/evaluated pipeline “was built” contradict the zero-artifact output. |

## Confirmed defects and repairs

| Defect | Evidence in notebook | Milestone 1 disposition |
|---|---|---|
| Large unapproved category download | Cell 3 starts six multi-GB archives and ends in `KeyboardInterrupt`. | No dataset download exists in the MVP. Synthetic clips are generated locally. |
| Colab coupling | `/content/co3d_data`, `/content/work`, shell magics, and a saved T4 runtime. | Package paths are local/argument-driven; CPU is the only required execution path. |
| Dependency check mislabel | `DEPS["pycolmap"]` checks only whether the `colmap` executable exists. | Not carried into Milestone 1; future code must distinguish executable and Python module explicitly. |
| COLMAP model-directory precedence | `d.is_dir() and images.bin.exists() or images.txt.exists()` can accept a non-directory due to operator precedence. | Documented for future label-pipeline repair. |
| Binary COLMAP output not parsed | The mapper normally emits binary models, while `parse_colmap_model` reads text and no model-converter step is run. | Deferred until a real two-sequence smoke test. |
| Unaligned coverage | Reconstructed SfM points are compared directly to CO3D reference points without estimating coordinate alignment or scale. | Coverage is not reported by the MVP. It remains unavailable until alignment is verified. |
| Infrastructure failure conflated with poor video | `compute_quality_score` assigns score 0 to both `failed` and `unavailable`. | The MVP has no COLMAP label path. Future labels must retain separate unavailable/failed states. |
| Zero reprojection error mishandled | `err <= 0` maps to the worst normalized reprojection component. | Documented for future COLMAP repair; zero finite error should map to best quality. |
| Circular threshold calibration | Validation truth is defined as score ≥50, then F1 is optimized against thresholds of the same score. | MVP uses fixed, explicitly uncalibrated heuristic bands and never reports validation metrics. |
| Weak object grouping | `object_group_key` falls back to sequence name and does not actually read sequence annotation video/object identity. | Deferred; future splits must derive and test physical-object grouping before training. |
| Broken frame overlap loop | Cell 21 indexes the split dictionary with integers and contains `SPLITS[SPLITS[i]]`. | Not reused. Future split tests must explicitly compare named split sets. |
| Padding/mask holes | Failed image loads can leave non-prefix valid mask positions, while packed LSTM lengths assume a valid prefix; tail copying can overwrite later valid slots. | No padded LSTM path exists in the MVP. Video sampling stores only successfully decoded samples. |
| Missing import/runtime error | Cell 27 uses `Image` in `compute_temporal_features` without a module-level import. | Replaced with OpenCV-native deterministic metrics. |
| Invalid Laplacian fallback | Arrays with `(H-1,W-1)` and `(H-2,W-2)` shapes are combined. | Replaced by OpenCV Laplacian at a bounded analysis resolution. |
| Temporal feature layout bug | A 12-element video vector is treated as `n=12` time rows and broadcast into a `[T,12]` tensor. | No learned temporal tensor exists in Milestone 1. Raw video-level and pair-level metrics are stored directly. |
| Training/inference scaling mismatch | Training feeds raw temporal values; the scaler is fit only after training and applied only at inference. | No model score is exposed. A future model must fit the scaler on train only and apply it identically everywhere. |
| Unbounded model head | Cell 30 returns `score * 100` without the documented sigmoid. Saved demo values happen to be small but are not bounded. | No CNN-LSTM is shipped. Future model tests must assert output bounds. |
| Optional temporal input shape failure | A model created with the temporal side channel expects 576 LSTM inputs but receives 512 when `temporal=None`. | Deferred with the model. |
| Short-video inference failure | Cell 46 reshapes to `n_want` even when sampling returns fewer frames; it does not pad and sets every mask element true. | The heuristic accepts any decoded count, reports it, and applies evidence caps for insufficient multi-view data. |
| Beginning-only long-video coverage | Cell 45 reads only the first 512 frames before uniform subsampling. | `video.py` seeks uniformly over the complete declared frame range and reports temporal coverage/fallback limitations. |
| Viewpoint proxy overclaim | ResNet/pixel differences are described as directly proxying baseline. | MVP uses verified feature motion plus homography residuals but explicitly says they do not prove translation/parallax. |
| Automatic remote weight download | `resnet18(weights=...)` downloads weights during execution. | Milestone 1 has no weight or network dependency. |
| Conclusions exceed evidence | Cell 48 describes trained/evaluated/deployed outcomes while cell 49 reports zero artifacts/results. | README and reports distinguish implemented heuristic behavior from unperformed ML/COLMAP work. |

## Reusable foundation extracted

- Uniform temporal coverage became `video_suitability.video`, with explicit resource bounds and actual timestamps.
- The notebook’s sharpness/exposure/temporal-feature intent became `quality.py`, using raw interpretable metrics rather than an untrained representation.
- Its stated geometric blind spot motivated `geometry.py`: ORB matching, fundamental/homography RANSAC, spatial match coverage, normalized displacement, and cautious homography-residual evidence.
- Its composite-score transparency became `scoring.py`, with declared weights, fixed thresholds, evidence caps, raw subscore evidence, timestamped problems, and recommendations.
- Its inference API intent became `infer.py`, but the output is correctly labeled heuristic and works without any model artifact.

CO3D readers, COLMAP labels, split logic, tabular baselines, and neural models remain intentionally outside Milestone 1. They should be introduced only with approved local data and actual reproducible experiments.

