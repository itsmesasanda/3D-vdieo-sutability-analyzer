# Colab COLMAP label compatibility audit

## Scope and evidence available

This audit compares the repository's production registration-training contract with
the Colab labeling code that is actually available in this checkout. It does not
train a model or rerun COLMAP.

Inspected sources:

- `reference/co3d_video_quality_assessment.ipynb` in full;
- `notebooks/colmap_reconstruction_colab.ipynb` and its source generator,
  `scripts/reconstruction/build_colab_notebook.py`;
- `src/video_suitability/ml/registration_training.py` and the production eight-feature
  implementation in `registration_features.py`;
- the locally accessible `evaluation/reconstruction` artifacts;
- the connected Google Drive account, searched for COLMAP, CO3D, label, batch, and
  reconstruction outputs.

During the initial audit, the connected Drive account did not expose the original
45 labels or batches A–G and R1–R3. A later local archive supplied the real 60-frame
CSV exports, which are independently audited in
[`real_colmap_csv_dataset_audit.md`](real_colmap_csv_dataset_audit.md). The CSVs
confirm the 60-frame label schema but still contain no per-run result JSON or frame
evidence, so no real record is yet certified for training.

## Compatibility findings

| Requirement | Reference CO3D notebook | Persistent 35-frame reconstruction notebook | Final training contract |
|---|---|---|---|
| Exact labeled images | Copied into ephemeral `/content/work/colmap/.../images`; `sampled_frame_paths` points at source files. Not guaranteed to persist in Drive. | Source input directory remains separate; COLMAP workspace and result are copied to Drive. | Exactly 60 ordered, decodable files, each SHA-256 verified. |
| Frame count | `COLMAP_MAX_IMAGES=16` in the checked-in notebook. | `EXPECTED_PILOT_FRAME_COUNT=35`. | Exactly 60. |
| Sampling | Frames ordered by CO3D `frame_number`; `np.linspace(...).astype(int)` (floor/truncate); byte-for-byte copies, no resize/re-encode. | Pre-extracted video frames; the notebook verifies rather than resamples them. | The exact 60 COLMAP inputs are the training input; production then selects its configured 16 positions from that manifest. |
| Manifest and hashes | No frame manifest or stored hashes in the label CSV/cache. | Verifies `frames_manifest.csv` when present and writes `input_integrity.json` with per-file hashes. | Hashed 60-row manifest plus hashed extraction metadata. |
| Reconstruction evidence | Per-record cache JSON plus `sequence_labels.csv` if the labels directory was retained. The checked-in execution produced zero labels. | Persistent `reconstruction_result_persistent.json`, logs, database/sparse workspace, and `reconstruction_summary.csv`. | CSV metric must agree with a hashed per-record result JSON. CSV-only metrics are rejected. |
| Target | `registration_ratio` is registered/sample count in `[0,1]`; `quality_score` is a separate weighted composite. | `registered_image_ratio` is registered/input count in `[0,1]`. | `registration_percentage = ratio * 100`; never `quality_score`, heuristic score, or viewpoint score. |
| Status semantics | `success`, `partial`, `failed`, `unavailable`. A broad exception path defaults to `failed`, so failure evidence is ambiguous. | Structured status/failure stage and recorded commands/logs. | Success/partial needs measured metrics. Zero is accepted only for an explicit, evidenced mapper/no-sparse-model reconstruction failure. Infrastructure failures are excluded. |
| Physical-source grouping | Not saved in the label CSV. The later split code falls back weakly to sequence identity. | Not saved in `reconstruction_summary.csv`. | `source_group` and `source_group_provenance` are mandatory; all variants of one physical source stay together. |
| Variant and holdout role | Not saved. | Sequence/attempt is saved, but no standardized variant or independent-holdout declaration. | `variant` and a predeclared `evaluation_role` are mandatory. |
| Label/protocol version | Not saved. | Result schema version exists, but there is no final label-protocol version. | Batch, label protocol, input fidelity, hash provenance, and regeneration provenance are retained. |
| Production eight features | Not saved. The notebook's later learned/handcrafted feature path is a different contract. | Not saved. | Always extracted fresh with the production implementation after input verification. |

### Consequences

- Any label truly generated from 16 or 35 images is incompatible with the exact
  60-image target definition. Regenerating 60 images cannot change the denominator
  or evidence of an already completed 16/35-image COLMAP run. Such labels remain
  historical-only unless COLMAP is deliberately rerun under the new protocol.
- `quality_score` and `label` from the reference notebook are not the Random Forest
  target. The only compatible target is registered images divided by the exact 60
  inputs, expressed as a percentage.
- A row saying `failed` is not sufficient evidence for a 0% target. Missing COLMAP,
  install/runtime exceptions, Drive errors, interrupted sessions, and absent data
  are not negative reconstruction labels.
- Existing notebook feature caches are not accepted as production features. The
  importer preserves evidence; the final bundle builder computes the eight features
  afresh from verified inputs.

## Import utility

`scripts/ml/import_colab_registration_labels.py` is a read-only compatibility
adapter. It recognizes the reference `sequence_labels.csv` schema, the persistent
`reconstruction_summary.csv` schema, and a narrow generic registration schema.

For each row it:

1. finds and hashes a per-record reconstruction JSON;
2. checks status, counts, and ratio against that JSON;
3. requires an exact total of 60 and rejects infrastructure/unavailable rows;
4. resolves physical-source grouping, grouping provenance, variant, and evaluation
   role from a reviewed mapping CSV when the batch did not store them;
5. copies exact saved labeled inputs to a new output directory, preserving ordered
   `sampled_frame_paths` when present;
6. recognizes original notebook hashes from a frame manifest or
   `input_integrity.json`; otherwise marks saved files `import_time_hash_only`;
7. optionally regenerates byte-identical CO3D selections only when the original
   inputs are absent, a supported sampling strategy is explicitly declared, and a
   notebook/config file proving that strategy is supplied and hashed;
8. writes a canonical final-contract CSV plus a row-level audit report.

It does not alter source CSVs, reconstruction directories, or source images. It
does not run COLMAP, invent missing hashes/metrics, extract a target from a composite
score, train a model, or activate inference.

Input provenance values are deliberately distinct:

- `original_notebook_verified`: saved files match hashes recorded during labeling;
- `import_time_hash_only`: exact saved input files exist, but their first known
  hashes were recorded by this importer;
- `faithfully_regenerated_from_co3d`: the original saved inputs are missing; files
  were deterministically reselected from hashed CO3D annotations using a hashed
  sampling declaration. This is never described as original hash verification.

## Can missing CO3D frames be regenerated safely?

Only under all of these conditions:

1. the label itself says the COLMAP run used 60 inputs;
2. the exact CO3D category/sequence and annotation ordering are known;
3. the original notebook/config proves one supported selection algorithm;
4. the source annotations and selected image files still exist;
5. the original transformation was byte-for-byte copy with no unrecorded resize,
   crop, orientation change, color conversion, or re-encoding.

The checked-in reference notebook meets conditions 2, 3, and 5 for its algorithm,
but its configured labels use 16 images, so those particular labels still fail
condition 1. A Batch A–G notebook configured for 60 may be regenerable after its
actual code/config and result JSON are inspected. If any transformation is unknown,
the importer rejects rather than guesses.

## Representative completed-batch fixture result

The initial importer was tested against a local fixture matching the reference
notebook's completed `sequence_labels.csv` and per-record JSON structure. The three
rows intentionally covered a valid 60-image success, a legacy 16-image success, and
an unavailable infrastructure result. It is retained as a legacy-schema contract
test.

Observed result:

| Fixture row | Result | Reason |
|---|---|---|
| 45/60 successful reconstruction | Accepted | CSV and result JSON agreed; exact 60 inputs were deterministically regenerated from hashed CO3D annotations using hashed notebook evidence; target became 75.0%. |
| 12/16 successful reconstruction | Rejected incompatible | Label used 16 inputs, not 60. |
| unavailable, 0/60 | Excluded | Infrastructure/unavailable status has no training target. |

The source CSV SHA-256 was identical before and after import. The accepted canonical
row then passed the production bundle's manifest/hash/sampling/eight-feature checks.
It was not training-ready by itself because a one-record fixture has neither enough
development groups for cross-validation nor an independent holdout. This is a
contract test, not evidence for a real Batch C row. A subsequent automated test uses
representative real Batch C `registration_proxy_60f_v1` fields, including rounded
2/60 metrics and a generic mapper error, while retaining the strict evidence gate.

## Google Drive collection and import instructions

1. Preserve every original batch folder unchanged. In Colab, mount Drive and use a
   new output folder for imports.
2. For each batch collect the label/summary CSV, every per-record result/cache JSON,
   persistent reconstruction folder and logs, the exact input image directory, any
   `frames_manifest.csv`, `input_integrity.json`, extraction metadata, and the exact
   notebook/config revision used for that batch.
3. Copy and fill
   `evaluation/ml/colab_import_group_map_template.csv`. `source_group` must identify
   the physical source object/sequence, not a processed variant. Record how that
   identity was established in `source_group_provenance`. Declare the independent
   holdout before model selection.
4. First import a single batch into a new empty directory:

```bash
python scripts/ml/import_colab_registration_labels.py \
  --labels-csv /content/drive/MyDrive/COLMAP/batch_C/sequence_labels.csv \
  --batch-root /content/drive/MyDrive/COLMAP/batch_C \
  --group-map-csv /content/drive/MyDrive/COLMAP/group_map.csv \
  --out-dir /content/drive/MyDrive/COLMAP/imported/batch_C_v1
```

5. If exact input files are missing and the audited batch notebook proves the
   reference floor-linspace CO3D selection, add all three regeneration arguments:

```bash
  --co3d-root /content/drive/MyDrive/co3d \
  --regeneration-sampling-strategy legacy_floor_linspace \
  --regeneration-evidence /content/drive/MyDrive/COLMAP/batch_C/label_batch_C.ipynb
```

Use `production_round_linspace` only if the actual batch code uses rounded linspace.
Do not select a strategy by trial and error. The evidence file is hashed into every
regenerated record.

6. Review `import_report.json` and `import_audit.csv`. Exit code 2 means one or more
   rows were rejected/excluded; it does not mean the accepted rows were discarded.
   Resolve each row from original evidence—never fill a missing value by assumption.
7. Repeat for the original 45, A–G, and R1–R3. Multiple `--labels-csv` and
   `--batch-root` arguments may also be passed to one invocation when their group
   map is complete.
8. Feed the canonical imports, not the historical CSVs, into the final preparation
   workflow:

```bash
python scripts/ml/build_registration_training_data.py \
  --labels-csv /content/drive/MyDrive/COLMAP/imported/batch_A_v1/imported_registration_labels.csv \
  --labels-csv /content/drive/MyDrive/COLMAP/imported/batch_C_v1/imported_registration_labels.csv \
  --labels-csv /content/drive/MyDrive/COLMAP/imported/batch_R1_v1/imported_registration_labels.csv \
  --out-dir /content/drive/MyDrive/COLMAP/training/registration_final_v1 \
  --bundle-version registration-final-v1 \
  --fold-count 5 \
  --parity-case-count 2
```

Do not start training unless the final `compatibility_report.json` says
`training_ready=true`, every accepted record has reviewed grouping provenance, and
the independent holdout remains untouched.
