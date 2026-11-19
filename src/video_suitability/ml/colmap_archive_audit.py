"""Read-only audit of the real 60-frame COLMAP CSV export archive."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


AUDIT_SCHEMA_VERSION = "registration_proxy_archive_audit/1.0"
LABEL_COLUMNS = {
    "tag",
    "source_tag",
    "source_group",
    "category",
    "variant",
    "status",
    "sampled_images",
    "registered_images",
    "registered_ratio",
    "target_registration_0_100",
    "label_version",
}
FEATURE_COLUMNS = {
    "tag",
    "source_group",
    "target",
    "brightness_mean",
    "brightness_variation",
    "contrast_mean",
    "sharpness_mean",
    "sharpness_variation",
    "dark_fraction",
    "orb_features_mean",
    "orb_matches_mean",
}
PRODUCTION_FEATURE_NAMES = {
    "mean_brightness",
    "mean_contrast",
    "mean_sharpness",
    "mean_orb_keypoints",
    "mean_frame_difference",
    "mean_orb_good_matches",
    "mean_orb_match_ratio",
    "mean_orb_ransac_inlier_ratio",
}
SELECTED_BATCH_FILES = {
    "A": "new_labels_50 (1).csv",
    "B": "new_labels_50 (2).csv",
    "C": "new_labels_50-2.csv",
    "D": "new_labels_50 (8).csv",
    "E": "new_labels_50 (7).csv",
    "F": "new_labels_50 (6).csv",
    "R1": "new_labels_50 (5).csv",
    "R2": "new_labels_50.csv",
    "R3": "new_labels_50-4.csv",
}
ORIGINAL_CUMULATIVE_FILE = "labels_45_plus_batches_A_B.csv"
C_ASSIGNMENT_FILE = "batch_assignment-2.json"
ASSIGNMENT_FILES = {
    "A": "batch_assignment (1).json",
    "B": "batch_assignment (3).json",
    "C": C_ASSIGNMENT_FILE,
    "D": "batch_assignment (9).json",
    "E": "batch_assignment (8).json",
    "F": "batch_assignment (7).json",
    "R1": "batch_assignment (5).json",
    "R2": "batch_assignment.json",
    "R3": "batch_assignment-4.json",
}
INFRASTRUCTURE_STATUSES = {"session_budget_pause", "match_exception"}
NUMERIC_FIELDS = {
    "sampled_images",
    "registered_images",
    "registered_ratio",
    "points3d",
    "mean_reprojection_error",
    "target_registration_0_100",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _archive_members(archive: Path) -> Dict[str, bytes]:
    with zipfile.ZipFile(archive) as handle:
        members: Dict[str, bytes] = {}
        for info in handle.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            name = Path(info.filename).name
            if name in members:
                raise ValueError(f"Archive contains duplicate basename: {name}")
            members[name] = handle.read(info)
    return members


def _csv_rows(value: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    text = value.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader.fieldnames or []), [dict(row) for row in reader]


def _normalized_number(value: str) -> str:
    if not value.strip():
        return ""
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        return f"invalid:{value}"
    return format(number.normalize(), "f")


def _semantic_label(row: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    keys = [
        "source_tag",
        "source_group",
        "category",
        "variant",
        "status",
        "sampled_images",
        "registered_images",
        "registered_ratio",
        "points3d",
        "mean_reprojection_error",
        "target_registration_0_100",
        "label_version",
        "label_source",
        "geometry_review",
        "label_status",
        "failure_reason",
    ]
    return tuple(
        (key, _normalized_number(row.get(key, "")) if key in NUMERIC_FIELDS else row.get(key, "").strip())
        for key in keys
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _classify_record(row: Mapping[str, str]) -> Tuple[str, str]:
    status = row["status"].strip()
    target = row["target_registration_0_100"].strip()
    if status == "sparse_model_created" and target:
        return (
            "recoverable_success_missing_evidence",
            "needs per-record reconstruction metrics plus exact-input manifest/hash evidence",
        )
    if status == "map_failed" and target:
        return (
            "recoverable_zero_requires_failure_evidence",
            "needs mapper stage/command evidence and compact mapper failure summary; CSV zero is not accepted alone",
        )
    if status == "map_failed":
        return (
            "recoverable_unlabeled_mapper_failure",
            "needs mapper evidence before deciding whether a verified 0/60 target exists",
        )
    if status in INFRASTRUCTURE_STATUSES:
        return (
            "excluded_infrastructure_no_target",
            "interruption or matching exception is not a reconstruction label",
        )
    return "ineligible_unknown_status", f"unsupported status: {status}"


def _verify_row_arithmetic(row: Mapping[str, str]) -> List[str]:
    issues: List[str] = []
    try:
        total_value = float(row["sampled_images"])
        registered_value = float(row["registered_images"])
        if not total_value.is_integer() or not registered_value.is_integer():
            return ["image counts are not integers"]
        total = int(total_value)
        registered = int(registered_value)
        ratio = float(row["registered_ratio"])
        target = float(row["target_registration_0_100"])
    except (KeyError, TypeError, ValueError):
        return ["numeric target fields are missing or invalid"]
    if total != 60:
        issues.append(f"sampled_images={total}, expected 60")
    if not 0 <= registered <= total:
        issues.append("registered_images is outside the input count")
    if abs(ratio - registered / total) > 5.0001e-6:
        issues.append("registered_ratio disagrees with counts")
    if abs(target - 100.0 * registered / total) > 0.005001:
        issues.append("target percentage disagrees with counts")
    return issues


def audit_colmap_csv_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    audit_inventory_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit the supplied archive without treating any CSV row as verified evidence."""

    archive = Path(archive_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    members = _archive_members(archive)
    required = {
        *SELECTED_BATCH_FILES.values(),
        *ASSIGNMENT_FILES.values(),
        ORIGINAL_CUMULATIVE_FILE,
        C_ASSIGNMENT_FILE,
    }
    missing = required - set(members)
    if missing:
        raise ValueError(f"Archive is missing required exports: {sorted(missing)}")

    file_inventory: List[Dict[str, Any]] = []
    label_exports: Dict[str, List[Dict[str, str]]] = {}
    feature_exports: Dict[str, List[Dict[str, str]]] = {}
    for name, value in sorted(members.items()):
        record: Dict[str, Any] = {
            "filename": name,
            "sha256": _sha256_bytes(value),
            "size_bytes": len(value),
            "kind": "other",
            "row_count": "",
            "schema": "",
        }
        if name.endswith(".csv"):
            columns, rows = _csv_rows(value)
            record["row_count"] = len(rows)
            if LABEL_COLUMNS <= set(columns):
                record["kind"] = "label_csv"
                record["schema"] = "registration_proxy_60f_v1"
                label_exports[name] = rows
            elif FEATURE_COLUMNS <= set(columns):
                record["kind"] = "feature_csv"
                record["schema"] = "legacy_eight_features_incompatible_with_production"
                feature_exports[name] = rows
            else:
                record["kind"] = "unknown_csv"
                record["schema"] = _canonical_json(columns)
        elif name.endswith(".json"):
            record["kind"] = "batch_assignment_json"
        file_inventory.append(record)

    c_assignment = json.loads(members[C_ASSIGNMENT_FILE].decode("utf-8"))
    original_groups = set(c_assignment.get("source_groups") or [])
    if len(original_groups) != 15:
        raise ValueError("Batch C assignment does not identify the expected 15 original groups")

    selected: List[Dict[str, str]] = []
    for row in label_exports[ORIGINAL_CUMULATIVE_FILE]:
        if row["source_group"] in original_groups:
            selected.append({**row, "batch_id": "Original", "selected_export": ORIGINAL_CUMULATIVE_FILE})
    for batch, filename in SELECTED_BATCH_FILES.items():
        selected.extend(
            {**row, "batch_id": batch, "selected_export": filename}
            for row in label_exports[filename]
        )

    assignment_validation = []
    for batch, filename in ASSIGNMENT_FILES.items():
        assignment = json.loads(members[filename].decode("utf-8"))
        rows = [row for row in selected if row["batch_id"] == batch]
        declared_categories = set(assignment.get("categories") or [])
        observed_categories = {row["category"] for row in rows}
        declared_groups = set(assignment.get("source_groups") or [])
        observed_groups = {row["source_group"] for row in rows}
        valid = (
            assignment.get("batch") == batch
            and observed_categories <= declared_categories
            and (not declared_groups or observed_groups <= declared_groups)
        )
        if not valid:
            raise ValueError(f"Selected export does not match assignment evidence for Batch {batch}")
        assignment_validation.append(
            {
                "batch_id": batch,
                "assignment_file": filename,
                "assignment_batch_matches": True,
                "observed_categories_are_declared": True,
                "observed_source_groups_are_declared": bool(declared_groups),
            }
        )

    tags = [row["tag"] for row in selected]
    duplicate_selected = sorted(tag for tag, count in Counter(tags).items() if count > 1)
    if duplicate_selected:
        raise ValueError(f"Selected batch lineage contains duplicate experiments: {duplicate_selected[:5]}")

    row_audit: List[Dict[str, Any]] = []
    arithmetic_issue_count = 0
    for row in selected:
        classification, reason = _classify_record(row)
        issues = _verify_row_arithmetic(row) if row["target_registration_0_100"].strip() else []
        arithmetic_issue_count += bool(issues)
        row_audit.append(
            {
                "batch_id": row["batch_id"],
                "selected_export": row["selected_export"],
                "tag": row["tag"],
                "source_tag": row["source_tag"],
                "source_group": row["source_group"],
                "category": row["category"],
                "variant": row["variant"],
                "status": row["status"],
                "sampled_images": row["sampled_images"],
                "registered_images": row["registered_images"],
                "registered_ratio": row["registered_ratio"],
                "target_registration_0_100": row["target_registration_0_100"],
                "classification": "ineligible_arithmetic_mismatch" if issues else classification,
                "reason": "; ".join(issues) if issues else reason,
            }
        )

    occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for filename, rows in label_exports.items():
        for line_number, row in enumerate(rows, start=2):
            occurrences[row["tag"]].append(
                {
                    "filename": filename,
                    "line_number": line_number,
                    "semantic": _semantic_label(row),
                }
            )
    duplicate_rows: List[Dict[str, Any]] = []
    semantic_conflict_count = 0
    for tag, values in sorted(occurrences.items()):
        if len(values) < 2:
            continue
        conflict = len({value["semantic"] for value in values}) > 1
        semantic_conflict_count += conflict
        duplicate_rows.append(
            {
                "tag": tag,
                "export_occurrence_count": len(values),
                "semantic_conflict": str(conflict).lower(),
                "locations": _canonical_json(
                    [
                        {"filename": value["filename"], "line_number": value["line_number"]}
                        for value in values
                    ]
                ),
            }
        )

    evidence_rows: List[Dict[str, Any]] = []
    for item in row_audit:
        if item["classification"] == "excluded_infrastructure_no_target":
            continue
        failure_evidence = ""
        if item["status"] == "map_failed":
            failure_evidence = (
                "compact mapper_summary.json with failure_stage, exact command, return code, "
                "and last 200 mapper log lines"
            )
        evidence_rows.append(
            {
                "batch_id": item["batch_id"],
                "tag": item["tag"],
                "source_group": item["source_group"],
                "variant": item["variant"],
                "required_result_file": (
                    "per-experiment result.json containing status, sampled/registered counts, "
                    "ratio, points, reprojection error, failure_stage, and recorded commands"
                ),
                "required_input_manifest": (
                    "ordered 60-row input_manifest.csv/json with source frame identity/path, "
                    "output filename, dimensions, and SHA-256"
                ),
                "required_transform_evidence": (
                    "batch transformation/sampling config and exact notebook/script revision hash"
                ),
                "required_failure_evidence": failure_evidence,
                "do_not_retrieve": "database.db; sparse/; dense/; point clouds; complete runs/ folder",
            }
        )

    group_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["source_group"]].append(row)
    for group, rows in sorted(grouped.items()):
        batches = sorted({row["batch_id"] for row in rows})
        categories = sorted({row["category"] for row in rows})
        source_tags = sorted({row["source_tag"] for row in rows})
        group_rows.append(
            {
                "source_group": group,
                "source_tags": ";".join(source_tags),
                "categories": ";".join(categories),
                "batches": ";".join(batches),
                "experiment_count": len(rows),
                "physical_identity_status": (
                    "crosschecked_batch_assignment"
                    if set(batches) & {"Original", "C", "D", "E", "F", "R1", "R2", "R3"}
                    else "csv_identifier_needs_candidate_manifest_crosscheck"
                ),
                "evaluation_role": "development",
            }
        )

    assignment_groups: Dict[str, set] = {"Original": original_groups}
    for batch, filename in ASSIGNMENT_FILES.items():
        assignment = json.loads(members[filename].decode("utf-8"))
        assignment_groups[batch] = set(assignment.get("source_groups") or [])
    import_group_rows = []
    for row in selected:
        crosschecked = row["source_group"] in assignment_groups.get(row["batch_id"], set())
        import_group_rows.append(
            {
                "category": row["category"],
                "sequence_name": row["tag"],
                "source_group": row["source_group"],
                "source_group_provenance": (
                    "csv_source_group_crosschecked_batch_assignment"
                    if crosschecked
                    else "audit_only_csv_identifier_candidate_manifest_missing"
                ),
                "variant": row["variant"],
                "evaluation_role": "development",
                "audit_only_not_yet_training_approved": "true",
            }
        )

    batch_evidence_rows = [
        {
            "scope": "global",
            "required_small_file": "CO3D dataset/version manifest with frame_annotations.jgz and sequence_annotations.jgz SHA-256 values",
            "reason": "pins source ordering and source identity without downloading COLMAP workspaces",
        },
        {
            "scope": "Original",
            "required_small_file": "original labels_60f_combined.csv plus original source-selection/transformation manifest",
            "reason": "the archive contains these 45 rows only through cumulative exports",
        },
        {
            "scope": "A,B",
            "required_small_file": "candidate manifest whose SHA-256 is 4a8793f16327406b318e5d6f8013fa879b533a67ab4c221f1c5a13b17289ea3d",
            "reason": "assignment JSON names only categories; source identifiers need crosscheck",
        },
        {
            "scope": "Original,A,B,C,D,E,F,R1,R2,R3",
            "required_small_file": "exact executed notebook/script revision and batch sampling/transformation config",
            "reason": "transformed and temporal-window variants cannot be regenerated from names alone",
        },
        {
            "scope": "independent_holdout",
            "required_small_file": "holdout assignment manifest with exact physical CO3D source sequence IDs for bicycle,bowl,cake,plant,toyplane",
            "reason": "the ZIP reserves only category names; exact holdout source IDs are absent",
        },
    ]

    inventory_comparison: Dict[str, Any] = {"provided": False}
    if audit_inventory_csv is not None:
        inventory_path = Path(audit_inventory_csv).expanduser().resolve()
        with inventory_path.open(newline="", encoding="utf-8-sig") as handle:
            inventory = list(csv.DictReader(handle))
        selected_by_tag = {row["tag"]: row for row in selected}
        inventory_by_tag = {row["tag"]: row for row in inventory}
        mismatches = []
        for tag in sorted(set(selected_by_tag) & set(inventory_by_tag)):
            actual = selected_by_tag[tag]
            supplied = inventory_by_tag[tag]
            for key in (
                "source_group",
                "category",
                "variant",
                "status",
                "sampled_images",
                "registered_images",
                "registered_ratio",
                "target_registration_0_100",
            ):
                left = _normalized_number(actual[key]) if key in NUMERIC_FIELDS else actual[key].strip()
                right = _normalized_number(supplied[key]) if key in NUMERIC_FIELDS else supplied[key].strip()
                if left != right:
                    mismatches.append({"tag": tag, "field": key, "archive": left, "inventory": right})
        inventory_comparison = {
            "provided": True,
            "path": str(inventory_path),
            "sha256": _sha256_bytes(inventory_path.read_bytes()),
            "row_count": len(inventory),
            "missing_from_inventory": sorted(set(selected_by_tag) - set(inventory_by_tag)),
            "extra_in_inventory": sorted(set(inventory_by_tag) - set(selected_by_tag)),
            "field_mismatch_count": len(mismatches),
            "field_mismatches": mismatches,
        }

    classification_counts = Counter(item["classification"] for item in row_audit)
    status_counts = Counter(
        (row["status"], "numeric" if row["target_registration_0_100"].strip() else "blank")
        for row in selected
    )
    batch_summary = []
    for batch in ["Original", *SELECTED_BATCH_FILES]:
        rows = [row for row in selected if row["batch_id"] == batch]
        batch_summary.append(
            {
                "batch_id": batch,
                "selected_export": rows[0]["selected_export"],
                "experiment_count": len(rows),
                "numeric_label_count": sum(bool(row["target_registration_0_100"].strip()) for row in rows),
                "source_group_count": len({row["source_group"] for row in rows}),
                "variants": sorted({row["variant"] for row in rows}),
            }
        )
    holdouts = list(c_assignment.get("holdout_excluded") or [])
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "archive": str(archive),
        "archive_sha256": _sha256_bytes(archive.read_bytes()),
        "original_archive_modified": False,
        "model_trained": False,
        "colmap_run": False,
        "batch_summary": batch_summary,
        "assignment_validation": assignment_validation,
        "unique_experiment_count": len(selected),
        "numeric_label_count": sum(bool(row["target_registration_0_100"].strip()) for row in selected),
        "source_identifier_count": len(grouped),
        "all_selected_rows_declare_60_inputs": all(row["sampled_images"] == "60" for row in selected),
        "arithmetic_issue_count": arithmetic_issue_count,
        "classification_counts": dict(sorted(classification_counts.items())),
        "accepted_record_count": 0,
        "recoverable_record_count": sum(
            count
            for classification, count in classification_counts.items()
            if classification.startswith("recoverable_")
        ),
        "excluded_record_count": classification_counts.get(
            "excluded_infrastructure_no_target", 0
        ),
        "status_target_counts": {
            f"{status}:{target_state}": count
            for (status, target_state), count in sorted(status_counts.items())
        },
        "raw_label_export_count": len(label_exports),
        "raw_label_row_count": sum(len(rows) for rows in label_exports.values()),
        "raw_unique_tag_count": len(occurrences),
        "duplicated_tag_count": len(duplicate_rows),
        "duplicate_extra_row_count": sum(len(values) - 1 for values in occurrences.values()),
        "semantic_conflict_count": semantic_conflict_count,
        "feature_export_count": len(feature_exports),
        "archived_feature_contract_matches_production": False,
        "archived_feature_names": sorted(FEATURE_COLUMNS - {"tag", "source_group", "target"}),
        "production_feature_names": sorted(PRODUCTION_FEATURE_NAMES),
        "holdout": {
            "reserved_identifiers_from_assignment": holdouts,
            "identifier_kind": "category_names_only",
            "present_in_selected_training_rows": sorted(
                set(holdouts) & {row["category"] for row in selected}
            ),
            "exact_physical_source_ids_available": False,
        },
        "inventory_comparison": inventory_comparison,
        "production_compatible_feature_extraction_can_begin": False,
        "training_ready": False,
    }

    _write_csv(
        destination / "archive_files.csv",
        file_inventory,
        ["filename", "sha256", "size_bytes", "kind", "row_count", "schema"],
    )
    _write_csv(
        destination / "record_audit.csv",
        row_audit,
        [
            "batch_id",
            "selected_export",
            "tag",
            "source_tag",
            "source_group",
            "category",
            "variant",
            "status",
            "sampled_images",
            "registered_images",
            "registered_ratio",
            "target_registration_0_100",
            "classification",
            "reason",
        ],
    )
    _write_csv(
        destination / "duplicate_exports.csv",
        duplicate_rows,
        ["tag", "export_occurrence_count", "semantic_conflict", "locations"],
    )
    _write_csv(
        destination / "additional_evidence_manifest.csv",
        evidence_rows,
        [
            "batch_id",
            "tag",
            "source_group",
            "variant",
            "required_result_file",
            "required_input_manifest",
            "required_transform_evidence",
            "required_failure_evidence",
            "do_not_retrieve",
        ],
    )
    _write_csv(
        destination / "source_group_audit.csv",
        group_rows,
        [
            "source_group",
            "source_tags",
            "categories",
            "batches",
            "experiment_count",
            "physical_identity_status",
            "evaluation_role",
        ],
    )
    _write_csv(
        destination / "import_group_map_AUDIT_ONLY.csv",
        import_group_rows,
        [
            "category",
            "sequence_name",
            "source_group",
            "source_group_provenance",
            "variant",
            "evaluation_role",
            "audit_only_not_yet_training_approved",
        ],
    )
    _write_csv(
        destination / "batch_level_evidence_manifest.csv",
        batch_evidence_rows,
        ["scope", "required_small_file", "reason"],
    )
    (destination / "holdout_reservation.json").write_text(
        json.dumps(summary["holdout"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
