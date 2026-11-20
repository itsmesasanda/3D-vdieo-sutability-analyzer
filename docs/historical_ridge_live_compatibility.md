# Historical Ridge experimental live-inference decision

Decision after independent testing: **live numeric inference is disabled; the
artifact is retained only for offline research and reproducibility**.

The exact historical eight-feature calculation remains in
`historical_registration_features.py`. The new live policy first defines 60 unique,
ordered positions with `np.linspace(0, frame_count - 1, 60, dtype=int)`. For every
position consumed by the historical 12-frame selector it performs codec BGR decode,
BGR-to-RGB conversion, proportional max-edge-800 Pillow LANCZOS resize, Pillow JPEG
quality 90 encoding, OpenCV JPEG decoding, then the unchanged historical grayscale,
max-edge-480 default OpenCV resize, sharpness, darkness and ORB calculations.

The historical extractor consumes only logical positions
`[0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 59]`. The live adapter therefore
materializes only these 12 JPEG round-trips. A regression test confirms its eight
values are exactly equal to creating all 60 adapter JPEG files and running the
directory extractor on the identical video. The other 48 frames cannot influence a
feature. This optimization does not imply equivalence to historical CO3D preparation.

Before any joblib deserialization, `experimental_ridge.py` verifies pinned SHA-256
digests for the model, historical feature definition and prediction-parity CSV. It
then verifies the exact feature order, eight-feature count, Ridge estimator type and
all three saved predictions. The original artifact remains truthfully marked
`offline_only`, `activation_allowed: false`, and `production_feature_compatible:
false`; the Streamlit result explicitly reports this is a new adapter policy.

Local CPU observations (Apple Silicon, OpenCV package 4.14.0.94,
scikit-learn 1.6.1) are
unvalidated predictions, not accuracy measurements:

| Video | Experimental estimate | Ridge path wall time |
|---|---:|---:|
| `IMG_3913.MOV` | 83.508634% | 10.785149 s |
| `IMG_3914.MOV` | 78.422076% | 9.729399 s |
| waving rendered-hand control | 63.557192% | 0.823165 s |

The waving-hand output is an important limitation: high historical ORB match counts
can yield a moderate estimate even though animation is not camera-orbit evidence.
The Ridge result must not replace the geometric analyzer. Repeated waving-hand runs
produced identical feature vectors and prediction, so no numerical instability was
observed; the concern is model/domain validity. Historical out-of-fold Ridge raw
predictions ranged from -3.078 to 126.387 and 19/647 were outside 0–100, which is why
live display applies the training artifact's declared 0–100 clipping policy.

The original model selection used five-fold GroupKFold MAE of 19.6249 percentage
points and no independent held-out set. A later four-video independent CPU COLMAP
diagnostic of the live adapter found MAE 58.743399 percentage points: Ridge/COLMAP
were 100/6.67, 100/100, 84.750987/3.33 and 63.555943/3.33 percent. Three errors
exceeded 60 percentage points. The 60/60 front-only registration result did not
verify whether sparse geometry belonged to the object rather than the background or
a degenerate planar/rotation configuration. See
`evaluation/ml/independent_ridge_validation/FINAL_EVALUATION_REPORT.md`.

This four-video study is too small for a general accuracy estimate, but it directly
demonstrates poor live-domain calibration. Random Forest gates, heuristic scoring,
geometric decisions and model artifacts are unchanged. A training-only
out-of-distribution audit flagged three of the four live probes but missed the
largest 93.33-point error, so selective unfamiliarity rejection was insufficient.
The diagnosis and domain-level abstention decision are documented in
`evaluation/ml/historical_registration_experiment/repair_diagnosis/DIAGNOSIS_REPORT.md`.
