#!/usr/bin/env python3
"""Regenerate real R1 inputs and extract the production eight-feature vectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from video_suitability.ml.artifacts import sha256_file
from video_suitability.ml.r1_regeneration import (
    DEFAULT_R1_PILOT_TAGS,
    regenerate_r1_experiment,
    select_r1_label_records,
    verify_executed_r1_notebook,
)
from video_suitability.ml.registration_features import (
    REGISTRATION_FEATURE_NAMES,
    extract_registration_features_from_frame_manifest,
    load_feature_spec,
)


FEATURE_COLUMNS = [
    "record_id",
    "batch_id",
    "source_label_export",
    "source_group",
    "source_group_provenance",
    "category",
    "variant",
    "sampled_images",
    "registered_images",
    "declared_target_registration_0_100",
    "target_registration_0_100",
    "label_version",
    "label_provenance",
    "label_status",
    "reconstruction_status",
    "input_fidelity_tier",
    "original_hashes_verified",
    "frames_manifest_relative_path",
    "frames_manifest_sha256",
    "regeneration_provenance_relative_path",
    "regeneration_provenance_sha256",
    "regeneration_implementation_sha256",
    "executed_notebook_sha256",
    "feature_spec_sha256",
    *REGISTRATION_FEATURE_NAMES,
]


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_pilot(
    *,
    record_audit_csv: Path,
    co3d_root: Path,
    feature_spec_path: Path,
    executed_r1_notebook: Path,
    output_dir: Path,
    tags: List[str],
    all_r1_sparse: bool = False,
) -> Dict[str, Any]:
    """Run the pilot or expansion extraction and persist successes and failures."""

    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing pilot output: {root}")
    root.mkdir(parents=True)
    inputs_root = root / "inputs"
    diagnostics_root = root / "feature_diagnostics"
    inputs_root.mkdir()
    diagnostics_root.mkdir()

    notebook_evidence = verify_executed_r1_notebook(executed_r1_notebook)
    feature_spec = load_feature_spec(feature_spec_path)
    records = select_r1_label_records(
        record_audit_csv,
        tags=None if all_r1_sparse else tags,
        all_sparse_records=all_r1_sparse,
    )
    feature_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    for record in records:
        tag = str(record["tag"])
        experiment_root = inputs_root / tag
        item_started_wall = time.perf_counter()
        item_started_cpu = time.process_time()
        try:
            regenerated = regenerate_r1_experiment(
                record=record,
                co3d_root=co3d_root,
                output_dir=experiment_root,
                executed_notebook_evidence=notebook_evidence,
            )
            features, diagnostics = extract_registration_features_from_frame_manifest(
                regenerated["frames_manifest"],
                feature_spec,
                images_dir=regenerated["images_directory"],
                expected_input_frame_count=60,
            )
            row = {
                "record_id": tag,
                "batch_id": record["batch_id"],
                "source_label_export": record["selected_export"],
                "source_group": record["source_group"],
                "source_group_provenance": record["source_group_provenance"],
                "category": record["category"],
                "variant": record["variant"],
                "sampled_images": record["sampled_images"],
                "registered_images": record["registered_images"],
                "declared_target_registration_0_100": record[
                    "declared_target_registration_0_100"
                ],
                "target_registration_0_100": record["target_registration_0_100"],
                "label_version": record["label_version"],
                "label_provenance": record["label_provenance"],
                "label_status": record["label_status"],
                "reconstruction_status": record["reconstruction_status"],
                "input_fidelity_tier": regenerated["fidelity_tier"],
                "original_hashes_verified": False,
                "frames_manifest_relative_path": str(
                    Path(regenerated["frames_manifest"]).relative_to(root)
                ),
                "frames_manifest_sha256": sha256_file(regenerated["frames_manifest"]),
                "regeneration_provenance_relative_path": str(
                    Path(regenerated["regeneration_provenance"]).relative_to(root)
                ),
                "regeneration_provenance_sha256": sha256_file(
                    regenerated["regeneration_provenance"]
                ),
                "regeneration_implementation_sha256": regenerated[
                    "implementation_sha256"
                ],
                "executed_notebook_sha256": notebook_evidence["sha256"],
                "feature_spec_sha256": sha256_file(feature_spec_path),
                **features,
            }
            feature_rows.append(row)
            diagnostic_path = diagnostics_root / f"{tag}.json"
            diagnostic_path.write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            decisions.append(
                {
                    "record_id": tag,
                    "decision": "accepted_experimental",
                    "reason": (
                        "Executed R1 implementation reproduced; original prepared-image "
                        "byte hashes unavailable"
                    ),
                    "wall_seconds": round(time.perf_counter() - item_started_wall, 6),
                    "cpu_seconds": round(time.process_time() - item_started_cpu, 6),
                }
            )
        except Exception as exc:
            decisions.append(
                {
                    "record_id": tag,
                    "decision": "excluded",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": round(time.perf_counter() - item_started_wall, 6),
                    "cpu_seconds": round(time.process_time() - item_started_cpu, 6),
                }
            )

    _write_csv(root / "r1_pilot_features.csv", feature_rows, FEATURE_COLUMNS)
    _write_csv(
        root / "r1_pilot_decisions.csv",
        decisions,
        ["record_id", "decision", "reason", "wall_seconds", "cpu_seconds"],
    )
    accepted = sum(item["decision"] == "accepted_experimental" for item in decisions)
    report = {
        "schema_version": "r1_feature_pilot/1.0",
        "requested_record_count": len(records),
        "accepted_record_count": accepted,
        "excluded_record_count": len(records) - accepted,
        "selected_tags": [str(item["tag"]) for item in records],
        "source_groups": sorted({str(item["source_group"]) for item in records}),
        "feature_names_in_order": list(REGISTRATION_FEATURE_NAMES),
        "feature_spec_sha256": sha256_file(feature_spec_path),
        "executed_notebook_evidence": notebook_evidence,
        "input_fidelity_tier": "executed_contract_reproduced",
        "original_hashes_verified": False,
        "feature_compatible_for_experimental_pipeline": accepted == len(records),
        "sufficient_as_standalone_training_dataset": False,
        "standalone_training_reason": (
            "The three-row pilot validates regeneration/feature compatibility only; "
            "it is not sufficient for model fitting or evaluation."
        ),
        "colmap_run": False,
        "wall_seconds": round(time.perf_counter() - started_wall, 6),
        "cpu_seconds": round(time.process_time() - started_cpu, 6),
    }
    (root / "r1_pilot_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-audit", type=Path, required=True)
    parser.add_argument("--co3d-root", type=Path, required=True)
    parser.add_argument("--feature-spec", type=Path, required=True)
    parser.add_argument("--executed-r1-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--all-r1-sparse",
        action="store_true",
        help="After the pilot, process every numeric R1 sparse-model record",
    )
    args = parser.parse_args()
    if args.tag and args.all_r1_sparse:
        parser.error("--tag and --all-r1-sparse are mutually exclusive")
    tags = args.tag or list(DEFAULT_R1_PILOT_TAGS)
    report = run_pilot(
        record_audit_csv=args.record_audit,
        co3d_root=args.co3d_root,
        feature_spec_path=args.feature_spec,
        executed_r1_notebook=args.executed_r1_notebook,
        output_dir=args.output_dir,
        tags=tags,
        all_r1_sparse=args.all_r1_sparse,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted_record_count"] == report["requested_record_count"] else 2


if __name__ == "__main__":
    sys.exit(main())
