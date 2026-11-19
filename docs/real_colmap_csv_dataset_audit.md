# Real 60-frame COLMAP CSV dataset audit

Audit date: 2026-11-19. No COLMAP run, feature extraction, model training, or model
activation was performed.

## Inputs and preservation

The supplied files were read from `/Users/sasanda/Downloads` without modification:

| File | SHA-256 |
|---|---|
| `csv files.zip` | `f746dcc016527c44857f59ce3f9dd8d151c49de5ca2fcd894058d8d4e2b088fb` |
| `colmap_csv_archive_audit.md` | `77f98c0c8f6e4b50468900963f8b3746af0c0155238854ca881d0477da7d4b5a` |
| `colmap_deduplicated_AUDIT_ONLY.csv` | `bb81ab90b22d2366aa60fc8d7f46f021e0e7ffb079921ea718472043284bbc7a` |

The Markdown audit was treated as preliminary. Counts below were independently
computed from ZIP members. The supplied audit-only inventory was compared afterward:
all 675 tags and selected fields matched, with no missing, extra, or field-mismatched
rows. It remains an inventory, not a training CSV.

## Independently verified inventory

| Batch | Selected export | Experiments | Numeric labels | Source identifiers |
|---|---|---:|---:|---:|
| Original | 45 rows selected from `labels_45_plus_batches_A_B.csv` using Batch C's original-source assignment | 45 | 45 | 15 |
| A | `new_labels_50 (1).csv` | 51 | 51 | 17 |
| B | `new_labels_50 (2).csv` | 51 | 51 | 17 |
| C | `new_labels_50-2.csv` | 52 | 50 | 13 |
| D | `new_labels_50 (8).csv` | 52 | 52 | 13 |
| E | `new_labels_50 (7).csv` | 52 | 52 | 13 |
| F | `new_labels_50 (6).csv` | 52 | 52 | 13 |
| R1 | `new_labels_50 (5).csv` | 80 | 72 | 5 |
| R2 | `new_labels_50.csv` | 144 | 138 | 12 |
| R3 | `new_labels_50-4.csv` | 96 | 96 | 8 |

The batch tag sets are disjoint. Totals are:

- 675 unique experiment tags;
- 659 numeric labels;
- 49 distinct `source_group` identifiers;
- 647 numeric `sparse_model_created` rows;
- 12 numeric `map_failed` rows, all declaring `no_initial_image_pair` and 0/60;
- 5 unlabeled `map_failed` rows;
- 9 unlabeled `session_budget_pause` rows;
- 2 unlabeled `match_exception` rows.

Every selected row declares 60 sampled images and
`registration_proxy_60f_v1`. Every numeric row's integer registered count agrees
with its five-decimal ratio and two-decimal percentage at the declared rounding
precision. There are zero arithmetic mismatches.

No Batch G-specific export or assignment was found.

## Duplicate exports

The ZIP contains 15 label CSV exports with 1,173 total rows. They collapse to the
same 675 unique experiment tags:

- 373 tags occur in more than one export;
- those repeated exports contribute 498 extra rows;
- zero semantic conflicts remain after normalizing harmless numeric formatting such
  as `60` versus `60.0`.

Repeated cumulative/downloaded CSVs are therefore duplicate exports, not independent
experiments. The selected lineage is recorded in `summary.json` and every row in
`record_audit.csv` records its chosen source export.

## Batch C details

Batch C has 52 unique experiments across 13 source identifiers and four variants:

- `gaussian_blur_r2`;
- `gaussian_blur_r7`;
- `high_exposure_x175`;
- `low_exposure_x050`.

It contains 49 `sparse_model_created` rows and 3 `map_failed` rows. Fifty rows have
numeric targets. The two blank targets are mapper failures; the one numeric mapper
failure declares 0/60 and `no_initial_image_pair`. None can yet be accepted solely
from the CSV because per-run evidence is absent.

The updated importer was run against the actual 52-row Batch C CSV with an
audit-derived temporary group map. It detected `registration_proxy_60f_v1`, kept the
source CSV hash unchanged, accepted 0, and rejected all 52 with the precise reason
`no per-record reconstruction JSON was found; CSV-only metrics are not marked
verified`. This is the intended conservative result, not an importer failure.

## Compatibility classification

Accepted now: **0**. This is expected and conservative: the ZIP has no per-record
result JSON, ordered input manifest, input hashes, or exact labeled frames.

| Classification | Count | Required action |
|---|---:|---|
| Recoverable measured success | 647 | Retrieve compact per-record result metrics and exact-input manifest/hash evidence. |
| Recoverable numeric zero, failure evidence required | 12 | Additionally retrieve mapper stage, exact command/return code, and compact failure log summary. A CSV `map_failed` string is insufficient. |
| Potentially recoverable unlabeled mapper failure | 5 | Retrieve the same evidence; assign 0 only if it proves an actual mapper/no-sparse-model failure on all 60 inputs. |
| Excluded infrastructure/interruption | 11 | Keep excluded. Session budget pauses and match exceptions are not negative labels. |

Thus 659 existing numeric labels are recoverable pending evidence, five additional
mapper rows may become verified failures, and 11 rows are currently ineligible.

## Source groups and holdout isolation

The 49 `source_group` values are stable across variants. Fifteen Original/C–F/R
identifiers can be crosschecked against assignment JSON source lists. A and B
assignment JSON files contain categories and a candidate-manifest hash but not their
source lists, so the other 34 identifiers still need the referenced candidate
manifest before all 49 can be called independently verified physical sequences.

The five reserved holdout names are `bicycle`, `bowl`, `cake`, `plant`, and
`toyplane`. None appears in selected development rows. However, the assignments
provide category names only—not exact physical CO3D sequence IDs—and no held-out
label rows are in the ZIP. The exact five source sequence IDs must be retrieved and
kept outside development/CV before the final split can be certified.

## Frame regeneration assessment

The CSVs identify transformations but do not define their executable preprocessing:
blur, exposure, contrast, noise, JPEG, gamma, resolution, motion blur, temporal
windows, view spans, and offsets all require the exact executed code/config.

- Base `original` rows may be regenerable from CO3D after source ordering, sampling,
  annotation version, and encoding/copy behavior are evidenced.
- Transformed rows are not safely regenerable from variant names alone.
- Temporal window/span variants need their precise source-index selection.
- Re-encoded variants need the implementation/library version and original output
  hashes; otherwise byte parity cannot be claimed.

The importer therefore refuses to regenerate a real transformed variant from base
CO3D frames. Exact saved inputs can be used when their hashes are verified. A
regenerated path can be added later only for transformations whose contract is
fully evidenced and implemented.

## Feature compatibility

The ZIP's 15 feature exports contain these historical features:

`brightness_mean`, `brightness_variation`, `contrast_mean`, `sharpness_mean`,
`sharpness_variation`, `dark_fraction`, `orb_features_mean`, and `orb_matches_mean`.

They do not match the production eight-feature names or formulas, notably lacking
the production frame-difference, match-ratio, and RANSAC-inlier features. They are
not accepted for training. Production features must be freshly extracted from
verified exact 60-frame inputs using `registration_features.py`.

Production-compatible feature extraction cannot begin from this ZIP alone.

## Minimal additional evidence to retrieve

The generated
`evaluation/ml/colmap_csv_archive_real_audit/additional_evidence_manifest.csv`
contains one row for each of the 664 potentially recoverable experiments. Retrieve
only:

1. A compact per-experiment result JSON containing status, 60-input count,
   registered count/ratio, sparse metrics, failure stage, and recorded commands.
2. An ordered 60-row input manifest containing source frame identity/path, output
   filename, dimensions, and SHA-256.
3. One batch-level executed notebook/script revision and transformation/sampling
   configuration per protocol.
4. For mapper failures only, a compact summary with failure stage, command, return
   code, and at most the final 200 mapper log lines.
5. CO3D annotation/version hashes, the A/B candidate manifest named by its stored
   SHA-256, the original 45-row label export if available, and exact held-out source
   IDs as listed in `batch_level_evidence_manifest.csv`.

Do not retrieve `database.db`, sparse/dense model directories, point clouds, or
complete `runs/` folders.

`import_group_map_AUDIT_ONLY.csv` is a starting inventory only. Its A/B provenance
rows explicitly say that the candidate manifest is missing. It must not be treated
as training-approved until reviewed.

## Commands and next steps

Reproduce this audit:

```bash
python scripts/ml/audit_colmap_csv_archive.py \
  --archive "/Users/sasanda/Downloads/csv files.zip" \
  --audit-inventory-csv /Users/sasanda/Downloads/colmap_deduplicated_AUDIT_ONLY.csv \
  --out-dir evaluation/ml/colmap_csv_archive_real_audit
```

Exit code 2 is intentional because evidence is incomplete and `training_ready` is
false.

After retrieving Batch C's compact evidence into a local/Drive-mounted batch root,
run the evidence-gated importer first:

```bash
python scripts/ml/import_colab_registration_labels.py \
  --labels-csv "/path/to/new_labels_50-2.csv" \
  --batch-root /path/to/batch_C_evidence \
  --group-map-csv /path/to/reviewed_group_map.csv \
  --out-dir /path/to/imported_batch_C_v1
```

Review `import_audit.csv`; do not proceed merely because some rows were accepted.
Repeat per batch, then pass only canonical `imported_registration_labels.csv` files
to `build_registration_training_data.py`. That builder must verify all frames and
features and report `training_ready=true` before any later training step.
