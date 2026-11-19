# R1 three-record regeneration pilot

Open `notebooks/r1_regeneration_pilot_colab.ipynb` in a Google Colab CPU runtime.
Edit the five paths/options in cell 1, then run all cells. The Drive project folder
must contain this repository revision and the executed
`co3d_R1_fourhour_colab(1).ipynb` file. `CO3D_ROOT` must use the executed notebook's
layout: `<root>/<category>/<sequence_id>/images/*`. The optional download cell fetches
only the official apple, backpack, and banana single-sequence subsets.

The default records are:

- `apple_110_13051_23361__span_p03_p84` — 60/60 registered;
- `backpack_537_78249_152122__window_offset_p10_60` — 29/60 registered;
- `banana_610_96665_193716__span_p25_p94` — 2/60 registered.

The main output is `r1_pilot_features.csv`. Keep the adjacent input manifests,
regeneration provenance, diagnostics, and report in Drive; the CSV alone does not
prove input fidelity. Rows are marked `executed_contract_reproduced` and
`original_hashes_verified=false` because the original R1 prepared JPEG hashes and
exact Pillow/libjpeg version were not saved.

Equivalent command inside Colab:

```bash
python scripts/ml/run_r1_feature_pilot.py \
  --record-audit evaluation/ml/colmap_csv_archive_real_audit/record_audit.csv \
  --co3d-root /content/co3d_data \
  --feature-spec evaluation/ml/registration_rf_artifact_template/feature_extractor.json \
  --executed-r1-notebook "/content/drive/MyDrive/co3d_R1_fourhour_colab(1).ipynb" \
  --output-dir "/content/drive/MyDrive/r1_registration_feature_pilot/r1_pilot_RUN_ID"
```

After all three rows are accepted, use the notebook's final disabled-by-default
cell or append `--all-r1-sparse` with a new output directory. That expands only to
the 72 numeric R1 sparse-model records and does not run COLMAP. Full expansion needs
the same five R1 sources used during labeling: apple `110_13051_23361`, backpack
`537_78249_152122`, ball `123_14363_28981`, banana `610_96665_193716`, and book
`119_13962_28926`. The final notebook cell checks these exact paths and can download
their official single-sequence categories when `DOWNLOAD_MISSING_CATEGORIES` is
enabled.
