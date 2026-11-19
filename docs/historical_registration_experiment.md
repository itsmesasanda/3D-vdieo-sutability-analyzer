# Fast historical-feature registration experiment

The completed offline experiment is in
`evaluation/ml/historical_registration_experiment/historical_opencv8_v2/`.
It joined 647 valid sparse-model labels to historical feature rows across 49 physical
CO3D source groups. Twenty-eight mapper-failure, interruption, or otherwise invalid
records were excluded. The five reserved categories do not occur in training, but
their exact held-out sequence IDs and compatible evaluation rows were unavailable;
no independent held-out result is claimed.

Five-fold physical-source GroupKFold results, with predictions clipped to the
target's 0–100 range, are:

| Model | MAE | RMSE |
|---|---:|---:|
| Ridge | 19.6249 | 27.0212 |
| Random Forest | 20.0599 | 27.3581 |
| Gradient Boosting | 21.1340 | 29.0513 |
| Dummy mean | 34.0572 | 39.4749 |

Ridge was selected using grouped-CV MAE only. The serialized model and three saved
prediction-parity rows are experimental and offline-only.

The historical extractor is not production-compatible. It samples 12 prepared
frames, conditionally resizes to a 480-pixel edge with OpenCV defaults, uses ORB 400
with runtime defaults, and calculates brightness/sharpness variation and dark
fraction. Production samples 16 frames, uses fixed 320×240 `INTER_AREA`, explicitly
configures ORB 1000 and instead calculates frame difference, match ratio, and
homography-RANSAC inlier ratio.

Consequently, no production `artifact_manifest.json` was created. The winner is also
Ridge while the current loader legitimately accepts only a parity-verified Random
Forest. Live integration requires production-compatible re-extraction and retraining;
if Random Forest then wins, run `scripts/ml/package_registration_rf.py` with saved
frame parity examples. Do not point `VIDEO_SUITABILITY_RF_ARTIFACT_DIR` at this
offline artifact.

Reproduce the experiment with:

```bash
python3 scripts/ml/train_historical_registration_experiment.py \
  --record-audit evaluation/ml/colmap_csv_archive_real_audit/record_audit.csv \
  --archive-dir "/Users/sasanda/Downloads/csv files" \
  --original-features /Users/sasanda/Downloads/co3d_colmap_two_seq_pilot/labeling_60f_v1/random_forest_experiment/features.csv \
  --original-labels /Users/sasanda/Downloads/co3d_colmap_two_seq_pilot/labeling_60f_v1/labels_60f_combined.csv \
  --output-dir /path/to/new/versioned/output
```
