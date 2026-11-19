# Historical Ridge live-compatibility decision

Decision: **do not enable live Ridge prediction**.

The legacy eight-feature function was recovered from the labeling notebooks and
implemented in `historical_registration_features.py`. Its source function hash is
`ac727c93fc14ffec7dd8079fb53f876f8a8b71560237ff9e319692d4288a2be0`.
On two repository phone recordings, the recreated function and the notebook function
produced the same feature order and numerically identical values when both consumed
the same prepared 60-JPEG directory.

That proves the image-directory feature calculation, but not live input parity. The
647 training rows began with prepared CO3D JPEG directories. No executed
upload-video-to-60-JPEG adapter exists, training-time prepared images/hashes are not
available for parity, and ORB/resize runtime defaults were not version-pinned across
all labeling batches. The candidate video adapter is therefore retained only as a
compatibility benchmark and is never called by Streamlit or the model loader.

Observed CPU benchmark using OpenCV 4.14.0, NumPy 2.0.2, and Pillow 11.3.0:

| Video | Wall time | Process CPU | Reference parity |
|---|---:|---:|---|
| `IMG_3913.MOV` | 40.244810 s | 205.058385 s | exact |
| `IMG_3914.MOV` | 46.476229 s | 239.832806 s | exact |

The cost includes 60 sparse 4K frame decodes, max-edge-800 JPEG preparation, and
both reference/recreated feature calculations. The benchmark report is
`evaluation/ml/historical_registration_experiment/historical_video_compatibility_benchmark.json`.

No Streamlit deployment changes were made. Enabling the requested panel later would
require a historically evidenced video adapter or saved training-frame parity cases,
a version-pinned legacy feature artifact, and a separate Ridge loader/result field.
The existing production Random Forest gates and result remain unchanged.
