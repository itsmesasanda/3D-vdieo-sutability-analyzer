# Registration Random Forest artifact template

This is a deployment contract, not a trained model. The repository currently has
no saved Random Forest, eight-feature preprocessing export, or training parity
examples, so production prediction remains unavailable.

The final retraining run must use `extract_registration_features` from this exact
repository revision, or export a feature specification that this revision accepts.
Copy at least one exact 60-frame training manifest and its checksum-protected images
into `parity/`, then store the manifest SHA-256 plus exact eight-feature vector in
`parity_examples.json`. Legacy video parity cases remain accepted, but the final
COLMAP-registration workflow uses the immutable image inputs that generated labels.

A deployable directory contains:

- `model.joblib` — fitted eight-feature scikit-learn Random Forest;
- `feature_extractor.json` — exact sampling, resize, OpenCV, ORB, matching, RANSAC,
  aggregation, feature order, and formula contract;
- `parity_examples.json` and every referenced parity manifest/image;
- `artifact_manifest.json` using `video_suitability_model_artifact/1.0`, with hashes
  for all files, `model_type=random_forest_colmap_registration_percentage`,
  `training_scope=genuine_colmap_registration_labels`, and metadata containing
  `target_name=colmap_registered_image_percentage`,
  `target_units=percent_0_to_100`, and `feature_count=8`.

Do not copy the placeholder parity file into production unchanged. Joblib artifacts
are executable serialization and must only be installed from the trusted training
pipeline. Deploy by replacing `models/registration_rf_current` atomically or setting
`VIDEO_SUITABILITY_RF_ARTIFACT_DIR` to a versioned directory. No COLMAP executable
is used during inference.
