#!/usr/bin/env python3
"""Build the guarded Colab CPU workflow for experimental registration training."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "experimental_registration_training_colab.ipynb"


def _source(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str) -> dict:
    normalized = textwrap.dedent(value).strip("\n") + "\n"
    ast.parse(normalized)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
        # Experimental COLMAP-registration model training (CPU, evidence-gated)

        This notebook prepares and evaluates an **experimental** regressor for the
        percentage of 60 COLMAP input images that register. It starts only from the
        647 audited `sparse_model_created` rows. It excludes mapper-failure zeros,
        missing targets, budget pauses, and matching/infrastructure exceptions.

        It does not run COLMAP, alter heuristic/viewpoint scoring, or claim production
        accuracy. A record is used only when an executed regeneration adapter and a
        hashed contract fully specify its source-frame selection and image transform.
        Unknown settings are exclusions, never inferred defaults.

        The existing production feature implementation is imported directly. All four
        models use the same source-group folds. The independent holdout is evaluated
        once, after grouped-CV selection, and only when compatible held-out features
        exist. Deployment packaging is disabled by default and remains blocked while
        provenance is incomplete.
        """
    ),
    code(
        """
        #@title 1. Configuration — edit paths, then Run all { display-mode: "form" }
        from datetime import datetime, timezone
        from pathlib import Path

        DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/3D-vdieo-sutability-analyzer")  #@param {type:"string"}
        CO3D_ROOT = Path("/content/drive/MyDrive/co3d_data")  #@param {type:"string"}
        RECORD_AUDIT_CSV = DRIVE_PROJECT_ROOT / "evaluation/ml/colmap_csv_archive_real_audit/record_audit.csv"
        REGENERATION_CONTRACT_JSON = Path("/content/drive/MyDrive/colmap_evidence/regeneration_contract.json")  #@param {type:"string"}
        REGENERATION_ADAPTER_PY = Path("/content/drive/MyDrive/colmap_evidence/executed_regeneration_adapter.py")  #@param {type:"string"}
        HOLDOUT_SOURCE_GROUPS_JSON = Path("/content/drive/MyDrive/colmap_evidence/holdout_source_groups.json")  #@param {type:"string"}
        HELDOUT_RECORDS_CSV = None  # set to Path("/content/drive/...csv") when available
        OUTPUT_PARENT = Path("/content/drive/MyDrive/registration_experimental_training")  #@param {type:"string"}

        EXPECTED_PROVISIONAL_CANDIDATES = 647
        GROUP_FOLD_COUNT = 5
        RANDOM_SEED = 20261119
        PARITY_CASE_COUNT = 3
        ALLOW_EXECUTED_CONTRACT_WITHOUT_ORIGINAL_HASH_PARITY = True
        REQUEST_DEPLOYMENT_PACKAGING = False
        PROVENANCE_COMPLETE = False
        MODEL_VERSION = "registration-experimental-v1"

        RUN_ID = datetime.now(timezone.utc).strftime("registration_experimental_%Y%m%dT%H%M%SZ")
        RUN_ROOT = OUTPUT_PARENT / RUN_ID
        print({"run_id": RUN_ID, "output": str(RUN_ROOT)})
        print("Deployment packaging requested:", REQUEST_DEPLOYMENT_PACKAGING)
        """
    ),
    code(
        """
        #@title 2. Mount Drive, install the repository, and freeze the environment
        import hashlib
        import importlib.util
        import json
        import os
        import platform
        import shutil
        import subprocess
        import sys

        from google.colab import drive
        drive.mount("/content/drive")

        required_paths = [
            DRIVE_PROJECT_ROOT / "src/video_suitability/ml/registration_features.py",
            RECORD_AUDIT_CSV,
            REGENERATION_CONTRACT_JSON,
            REGENERATION_ADAPTER_PY,
            HOLDOUT_SOURCE_GROUPS_JSON,
        ]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required training evidence is missing: " + "; ".join(missing))
        if not CO3D_ROOT.is_dir():
            raise FileNotFoundError(f"CO3D root is missing: {CO3D_ROOT}")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(DRIVE_PROJECT_ROOT), "--no-deps"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn>=1.3,<2", "joblib>=1.3,<2", "pandas>=2,<3"])

        import cv2
        import joblib
        import numpy as np
        import pandas as pd
        import sklearn

        def sha256_file(path, chunk_size=1024 * 1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        environment = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "project_root": str(DRIVE_PROJECT_ROOT),
            "project_git_commit": subprocess.run(
                ["git", "-C", str(DRIVE_PROJECT_ROOT), "rev-parse", "HEAD"],
                text=True, capture_output=True, check=False,
            ).stdout.strip() or None,
        }
        (RUN_ROOT / "runtime_environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        print(json.dumps(environment, indent=2))
        """
    ),
    code(
        """
        #@title 3. Deduplicate candidates and enforce label/group/holdout gates
        import csv

        audit = pd.read_csv(RECORD_AUDIT_CSV, dtype=str, keep_default_na=False)
        if audit["tag"].duplicated().any():
            duplicates = audit.loc[audit["tag"].duplicated(False), "tag"].tolist()
            raise ValueError(f"Audit input contains duplicate experiment tags: {duplicates[:10]}")

        candidates = audit.loc[
            (audit["classification"] == "recoverable_success_missing_evidence")
            & (audit["status"] == "sparse_model_created")
            & (audit["target_registration_0_100"] != "")
        ].copy()
        if len(candidates) != EXPECTED_PROVISIONAL_CANDIDATES:
            raise ValueError(
                f"Expected {EXPECTED_PROVISIONAL_CANDIDATES} sparse-model candidates, found {len(candidates)}"
            )
        sampled = pd.to_numeric(candidates["sampled_images"], errors="raise")
        registered = pd.to_numeric(candidates["registered_images"], errors="raise")
        target = pd.to_numeric(candidates["target_registration_0_100"], errors="raise")
        if not (sampled == 60).all():
            raise ValueError("Every provisional candidate must declare exactly 60 images")
        if not ((registered >= 0) & (registered <= 60)).all():
            raise ValueError("Registered-image count is outside 0..60")
        expected_target = registered * 100.0 / 60.0
        if not np.allclose(target, expected_target, atol=0.005001, rtol=0.0):
            raise ValueError("At least one target disagrees with registered_images / 60")
        candidates["target_registration_0_100"] = expected_target

        holdout_payload = json.loads(HOLDOUT_SOURCE_GROUPS_JSON.read_text(encoding="utf-8"))
        HOLDOUT_SOURCE_GROUPS = [str(value) for value in holdout_payload.get("source_groups", [])]
        if len(HOLDOUT_SOURCE_GROUPS) != 5 or len(set(HOLDOUT_SOURCE_GROUPS)) != 5:
            raise ValueError("holdout_source_groups.json must declare exactly five unique physical source IDs")
        development = candidates.loc[~candidates["source_group"].isin(HOLDOUT_SOURCE_GROUPS)].copy()
        accidental_holdout = candidates.loc[candidates["source_group"].isin(HOLDOUT_SOURCE_GROUPS)].copy()
        if not accidental_holdout.empty:
            print(f"Reserved {len(accidental_holdout)} candidate rows from the five held-out sources")
        if set(development["source_group"]) & set(HOLDOUT_SOURCE_GROUPS):
            raise RuntimeError("Holdout leakage detected before feature extraction")

        heldout = pd.DataFrame(columns=candidates.columns)
        if HELDOUT_RECORDS_CSV is not None:
            HELDOUT_RECORDS_CSV = Path(HELDOUT_RECORDS_CSV)
            if not HELDOUT_RECORDS_CSV.is_file():
                raise FileNotFoundError(HELDOUT_RECORDS_CSV)
            heldout = pd.read_csv(HELDOUT_RECORDS_CSV, dtype=str, keep_default_na=False)
            required = set(candidates.columns) - {"classification", "reason", "selected_export"}
            if not required <= set(heldout.columns):
                raise ValueError("Held-out inventory is missing required label columns")
            if not set(heldout["source_group"]) <= set(HOLDOUT_SOURCE_GROUPS):
                raise ValueError("Held-out rows contain undeclared source groups")
            if set(heldout["source_group"]) & set(development["source_group"]):
                raise ValueError("A physical source appears in development and holdout")
            if heldout["tag"].duplicated().any():
                raise ValueError("Held-out inventory contains duplicate experiment tags")
            if not (pd.to_numeric(heldout["sampled_images"], errors="raise") == 60).all():
                raise ValueError("Every held-out record must declare exactly 60 images")
            if not (heldout["status"] == "sparse_model_created").all():
                raise ValueError("Held-out inventory contains a non-measured reconstruction status")
            heldout_registered = pd.to_numeric(heldout["registered_images"], errors="raise")
            heldout_target = pd.to_numeric(
                heldout["target_registration_0_100"], errors="raise"
            )
            heldout_expected = heldout_registered * 100.0 / 60.0
            if not np.allclose(heldout_target, heldout_expected, atol=0.005001, rtol=0.0):
                raise ValueError("A held-out target disagrees with registered_images / 60")
            heldout["target_registration_0_100"] = heldout_expected

        development["dataset_role"] = "development"
        heldout["dataset_role"] = "independent_holdout"
        extraction_rows = pd.concat([development, heldout], ignore_index=True)
        print({
            "audited_rows": len(audit),
            "provisional_sparse_candidates": len(candidates),
            "development_candidates": len(development),
            "heldout_candidates": len(heldout),
            "development_source_groups": development["source_group"].nunique(),
            "reserved_holdout_groups": HOLDOUT_SOURCE_GROUPS,
        })
        """
    ),
    code(
        """
        #@title 4. Verify executed-code and regeneration contracts; import adapter
        contract_bytes = REGENERATION_CONTRACT_JSON.read_bytes()
        contract = json.loads(contract_bytes)
        contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
        if contract.get("schema_version") != "registration_regeneration_contract/1.0":
            raise ValueError("Unsupported regeneration contract schema")
        if contract.get("adapter_sha256") != sha256_file(REGENERATION_ADAPTER_PY):
            raise ValueError("Regeneration adapter hash differs from the contract")

        executed_code_checks = []
        for item in contract.get("executed_code_evidence", []):
            path = Path(item["path"])
            actual = sha256_file(path) if path.is_file() else None
            executed_code_checks.append({"path": str(path), "expected": item["sha256"], "actual": actual})
            if actual != item["sha256"]:
                raise ValueError(f"Executed notebook/code evidence hash mismatch: {path}")
        if not executed_code_checks:
            raise ValueError("Contract has no hashed executed notebook/code evidence")

        inspection_patterns = [
            "linspace", "frame_indices", "sample", "window", "span", "offset",
            "gaussianblur", "resize", "exposure", "contrast", "gamma", "noise",
            "jpeg", "imwrite", "motion_blur", "random", "seed", "frame_number",
        ]
        implementation_inspection = []
        for evidence in contract.get("executed_code_evidence", []):
            path = Path(evidence["path"])
            if path.suffix.lower() == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                source_text = "\\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                )
            else:
                source_text = path.read_text(encoding="utf-8", errors="replace")
            lines = source_text.splitlines()
            matches = []
            for line_number, line in enumerate(lines, start=1):
                lowered = line.lower()
                found = sorted({pattern for pattern in inspection_patterns if pattern in lowered})
                if found:
                    matches.append({
                        "line_number": line_number,
                        "patterns": found,
                        "source": line[:500],
                    })
            implementation_inspection.append({
                "path": str(path),
                "sha256": sha256_file(path),
                "matched_line_count": len(matches),
                "matches": matches,
            })
        (RUN_ROOT / "executed_implementation_inspection.json").write_text(
            json.dumps(implementation_inspection, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )

        required_versions = contract.get("required_runtime_versions", {})
        observed_versions = {"opencv": cv2.__version__, "numpy": np.__version__}
        version_mismatches = {
            key: {"required": value, "observed": observed_versions.get(key)}
            for key, value in required_versions.items()
            if observed_versions.get(key) != value
        }
        if version_mismatches:
            raise RuntimeError(f"Regeneration runtime version mismatch: {version_mismatches}")

        adapter_spec = importlib.util.spec_from_file_location(
            "executed_regeneration_adapter", REGENERATION_ADAPTER_PY
        )
        adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(adapter)
        if not callable(getattr(adapter, "generate_experiment", None)):
            raise TypeError(
                "Adapter must define generate_experiment(record, co3d_root, output_dir, variant_contract)"
            )

        variant_contracts = contract.get("variants", {})
        contract_audit = []
        for variant in sorted(extraction_rows["variant"].unique()):
            item = variant_contracts.get(variant)
            reasons = []
            if not isinstance(item, dict):
                reasons.append("variant absent from regeneration contract")
            else:
                for key in ("sampling_fully_specified", "transformation_fully_specified", "deterministic"):
                    if item.get(key) is not True:
                        reasons.append(f"{key} is not true")
                tier = item.get("fidelity_tier")
                if tier not in {"original_hash_verified", "executed_contract_reproduced"}:
                    reasons.append("unsupported fidelity_tier")
                if tier == "executed_contract_reproduced" and not ALLOW_EXECUTED_CONTRACT_WITHOUT_ORIGINAL_HASH_PARITY:
                    reasons.append("original output hashes unavailable and provisional regeneration disabled")
            contract_audit.append({"variant": variant, "eligible": not reasons, "reasons": reasons})
        (RUN_ROOT / "regeneration_contract_audit.json").write_text(
            json.dumps(
                {
                    "contract_sha256": contract_sha256,
                    "adapter_sha256": sha256_file(REGENERATION_ADAPTER_PY),
                    "executed_code_checks": executed_code_checks,
                    "variants": contract_audit,
                },
                indent=2,
                sort_keys=True,
            ) + "\\n",
            encoding="utf-8",
        )
        print(pd.DataFrame(contract_audit).to_string(index=False))
        """
    ),
    code(
        """
        #@title 5. Regenerate eligible inputs and extract production features (CPU, resumable)
        from video_suitability.ml.registration_features import (
            REGISTRATION_FEATURE_NAMES,
            extract_registration_features_from_frame_manifest,
            load_feature_spec,
            verify_registration_frame_manifest,
        )

        feature_spec_path = (
            DRIVE_PROJECT_ROOT
            / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
        )
        feature_spec = load_feature_spec(feature_spec_path)
        feature_spec_sha256 = sha256_file(feature_spec_path)
        cache_path = RUN_ROOT / "feature_records.csv"
        decisions_path = RUN_ROOT / "input_reproduction_decisions.csv"
        generated_root = Path("/content/registration_generated_inputs") / RUN_ID
        generated_root.mkdir(parents=True, exist_ok=False)
        parity_root = RUN_ROOT / "parity_inputs"
        parity_root.mkdir()

        feature_columns = [
            "record_id", "source_group", "category", "variant", "dataset_role",
            "target_registration_0_100", "registered_images", "fidelity_tier",
            "original_hashes_verified", "contract_sha256", "feature_spec_sha256",
            *REGISTRATION_FEATURE_NAMES,
        ]
        feature_rows = []
        decisions = []
        parity_cases = []
        parity_groups = set()

        def safe_tag(value):
            cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
            if not cleaned:
                raise ValueError("Empty safe experiment tag")
            return cleaned

        def remove_generated(path):
            resolved = Path(path).resolve()
            root = generated_root.resolve()
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to remove path outside generated root: {resolved}")
            shutil.rmtree(resolved)

        for position, row in extraction_rows.iterrows():
            record = row.to_dict()
            tag = record["tag"]
            variant = record["variant"]
            item = variant_contracts.get(variant)
            contract_decision = next(value for value in contract_audit if value["variant"] == variant)
            base_decision = {
                "tag": tag,
                "source_group": record["source_group"],
                "variant": variant,
                "dataset_role": record["dataset_role"],
                "contract_sha256": contract_sha256,
            }
            if not contract_decision["eligible"]:
                decisions.append({
                    **base_decision,
                    "decision": "uncertain_excluded",
                    "reason": "; ".join(contract_decision["reasons"]),
                    "fidelity_tier": item.get("fidelity_tier", "") if isinstance(item, dict) else "",
                })
                continue
            work = generated_root / safe_tag(tag)
            work.mkdir()
            try:
                generated = adapter.generate_experiment(
                    record=record,
                    co3d_root=CO3D_ROOT,
                    output_dir=work,
                    variant_contract=item,
                )
                if not isinstance(generated, dict):
                    raise TypeError("Adapter result must be a dictionary")
                manifest = Path(generated["frames_manifest"])
                images = Path(generated["images_directory"])
                verify_registration_frame_manifest(
                    manifest, images_dir=images, expected_input_frame_count=60
                )
                if generated.get("variant") != variant:
                    raise ValueError("Adapter output variant differs from the label")
                if generated.get("source_group") != record["source_group"]:
                    raise ValueError("Adapter output source_group differs from the label")
                fidelity_tier = item["fidelity_tier"]
                if fidelity_tier == "original_hash_verified" and generated.get("original_hashes_verified") is not True:
                    raise ValueError("Contract requires original hash parity but adapter did not verify it")
                features, diagnostics = extract_registration_features_from_frame_manifest(
                    manifest,
                    feature_spec,
                    images_dir=images,
                    expected_input_frame_count=60,
                )
                feature_row = {
                    "record_id": tag,
                    "source_group": record["source_group"],
                    "category": record["category"],
                    "variant": variant,
                    "dataset_role": record["dataset_role"],
                    "target_registration_0_100": float(record["target_registration_0_100"]),
                    "registered_images": int(float(record["registered_images"])),
                    "fidelity_tier": fidelity_tier,
                    "original_hashes_verified": bool(generated.get("original_hashes_verified")),
                    "contract_sha256": contract_sha256,
                    "feature_spec_sha256": feature_spec_sha256,
                    **features,
                }
                feature_rows.append(feature_row)
                decisions.append({
                    **base_decision,
                    "decision": "accepted_experimental",
                    "reason": (
                        "original frame hashes verified"
                        if generated.get("original_hashes_verified")
                        else "executed contract reproduced; original byte identity unverified"
                    ),
                    "fidelity_tier": fidelity_tier,
                })
                if (
                    record["dataset_role"] == "development"
                    and record["source_group"] not in parity_groups
                    and len(parity_cases) < PARITY_CASE_COUNT
                ):
                    case_root = parity_root / safe_tag(tag)
                    shutil.copytree(images, case_root / "images")
                    shutil.copy2(manifest, case_root / "frames_manifest.csv")
                    parity_cases.append({
                        "case_id": tag,
                        "source_group": record["source_group"],
                        "variant": variant,
                        "case_root": str(case_root),
                        "expected_features": features,
                    })
                    parity_groups.add(record["source_group"])
            except Exception as exc:
                decisions.append({
                    **base_decision,
                    "decision": "excluded_generation_or_feature_failure",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "fidelity_tier": item.get("fidelity_tier", ""),
                })
            finally:
                if work.exists():
                    remove_generated(work)

            if (position + 1) % 20 == 0:
                pd.DataFrame(feature_rows, columns=feature_columns).to_csv(cache_path, index=False)
                pd.DataFrame(decisions).to_csv(decisions_path, index=False)
                print(f"Processed {position + 1}/{len(extraction_rows)}; accepted={len(feature_rows)}")

        pd.DataFrame(feature_rows, columns=feature_columns).to_csv(cache_path, index=False)
        pd.DataFrame(decisions).to_csv(decisions_path, index=False)
        decision_counts = pd.Series(
            [item["decision"] for item in decisions], dtype="object"
        ).value_counts().to_dict()
        (RUN_ROOT / "input_reproduction_summary.json").write_text(
            json.dumps(
                {
                    "candidate_count": len(extraction_rows),
                    "accepted_count": len(feature_rows),
                    "decision_counts": decision_counts,
                    "contract_sha256": contract_sha256,
                    "feature_spec_sha256": feature_spec_sha256,
                    "experimental": True,
                },
                indent=2,
                sort_keys=True,
            ) + "\\n",
            encoding="utf-8",
        )
        (RUN_ROOT / "parity_case_index.json").write_text(
            json.dumps(parity_cases, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        print(pd.DataFrame(decisions)["decision"].value_counts(dropna=False))
        """
    ),
    code(
        """
        #@title 6. Freeze accepted data and one identical GroupKFold split plan
        features_df = pd.read_csv(cache_path)
        decisions_df = pd.read_csv(decisions_path)
        if features_df.empty:
            raise RuntimeError("No reproducible inputs passed the feature gate; training cannot start")
        if features_df["record_id"].duplicated().any():
            raise ValueError("Duplicate accepted feature records")
        if not np.isfinite(features_df[list(REGISTRATION_FEATURE_NAMES)].to_numpy(dtype=float)).all():
            raise ValueError("Feature matrix contains non-finite values")

        development_features = features_df.loc[features_df["dataset_role"] == "development"].copy()
        holdout_features = features_df.loc[features_df["dataset_role"] == "independent_holdout"].copy()
        if set(development_features["source_group"]) & set(holdout_features["source_group"]):
            raise RuntimeError("Accepted feature records leak a physical group into holdout")
        if set(development_features["source_group"]) & set(HOLDOUT_SOURCE_GROUPS):
            raise RuntimeError("A reserved holdout group entered development features")
        group_count = development_features["source_group"].nunique()
        if group_count < GROUP_FOLD_COUNT:
            raise ValueError(f"Need at least {GROUP_FOLD_COUNT} accepted development source groups; found {group_count}")

        from sklearn.model_selection import GroupKFold

        X = development_features[list(REGISTRATION_FEATURE_NAMES)].copy()
        y = development_features["target_registration_0_100"].to_numpy(dtype=float)
        groups = development_features["source_group"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=GROUP_FOLD_COUNT)
        split_indices = list(splitter.split(X, y, groups))
        fold_manifest = []
        for fold, (train_index, validation_index) in enumerate(split_indices):
            train_groups = sorted(set(groups[train_index]))
            validation_groups = sorted(set(groups[validation_index]))
            if set(train_groups) & set(validation_groups):
                raise RuntimeError("GroupKFold leakage detected")
            fold_manifest.append({
                "fold": fold,
                "train_record_ids": development_features.iloc[train_index]["record_id"].tolist(),
                "validation_record_ids": development_features.iloc[validation_index]["record_id"].tolist(),
                "train_source_groups": train_groups,
                "validation_source_groups": validation_groups,
            })
        (RUN_ROOT / "grouped_cv_splits.json").write_text(
            json.dumps(
                {
                    "schema_version": "experimental_registration_groupkfold/1.0",
                    "fold_count": GROUP_FOLD_COUNT,
                    "group_key": "source_group",
                    "folds": fold_manifest,
                    "holdout_source_groups": HOLDOUT_SOURCE_GROUPS,
                },
                indent=2,
                sort_keys=True,
            ) + "\\n",
            encoding="utf-8",
        )
        print({"accepted_development": len(development_features), "groups": group_count, "heldout": len(holdout_features)})
        """
    ),
    code(
        """
        #@title 7. Train four regressors on identical folds; preprocessing stays inside folds
        from sklearn.base import clone
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model_factories = {
            "dummy_mean": lambda: DummyRegressor(strategy="mean"),
            "ridge": lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=1.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=3,
                loss="squared_error",
                random_state=RANDOM_SEED,
            ),
        }

        prediction_rows = []
        fold_metric_rows = []
        for model_name, factory in model_factories.items():
            for fold, (train_index, validation_index) in enumerate(split_indices):
                model = factory()
                model.fit(X.iloc[train_index], y[train_index])
                predicted = np.asarray(model.predict(X.iloc[validation_index]), dtype=float)
                actual = y[validation_index]
                error = predicted - actual
                fold_metric_rows.append({
                    "model": model_name,
                    "fold": fold,
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "train_records": len(train_index),
                    "validation_records": len(validation_index),
                })
                for local_position, row_index in enumerate(validation_index):
                    source = development_features.iloc[row_index]
                    prediction_rows.append({
                        "model": model_name,
                        "fold": fold,
                        "record_id": source["record_id"],
                        "source_group": source["source_group"],
                        "variant": source["variant"],
                        "actual": float(actual[local_position]),
                        "predicted": float(predicted[local_position]),
                        "error": float(error[local_position]),
                        "absolute_error": float(abs(error[local_position])),
                    })

        cv_predictions = pd.DataFrame(prediction_rows)
        fold_metrics = pd.DataFrame(fold_metric_rows)
        cv_predictions.to_csv(RUN_ROOT / "grouped_cv_predictions.csv", index=False)
        fold_metrics.to_csv(RUN_ROOT / "grouped_cv_fold_metrics.csv", index=False)
        print(fold_metrics.groupby("model")[["mae", "rmse"]].mean().sort_values("mae"))
        """
    ),
    code(
        """
        #@title 8. Summarize distributions/errors and select by grouped CV only
        def degradation_type(variant):
            value = str(variant).lower()
            if value == "original": return "original"
            if "motion_blur" in value: return "motion_blur"
            if "gaussian_blur" in value or value.startswith("blur"): return "gaussian_blur"
            if "exposure" in value: return "exposure"
            if "contrast" in value: return "contrast"
            if "noise" in value: return "noise"
            if "jpeg" in value: return "jpeg"
            if "gamma" in value: return "gamma"
            if "resolution" in value: return "resolution"
            if "window" in value or "span" in value or "offset" in value: return "temporal_selection"
            return "other"

        cv_predictions["degradation_type"] = cv_predictions["variant"].map(degradation_type)
        overall_rows = []
        distribution_rows = []
        for model_name, rows in cv_predictions.groupby("model"):
            error = rows["predicted"] - rows["actual"]
            overall_rows.append({
                "model": model_name,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "record_count": len(rows),
            })
            values = rows["predicted"].to_numpy(dtype=float)
            distribution_rows.append({
                "model": model_name,
                "minimum": float(np.min(values)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
                "standard_deviation": float(np.std(values)),
                "below_zero_count": int(np.sum(values < 0)),
                "above_100_count": int(np.sum(values > 100)),
            })
        overall = pd.DataFrame(overall_rows).sort_values(["mae", "rmse", "model"])
        distributions = pd.DataFrame(distribution_rows)
        errors_by_type = (
            cv_predictions.groupby(["model", "degradation_type"])
            .agg(
                record_count=("record_id", "count"),
                mae=("absolute_error", "mean"),
                rmse=("error", lambda value: float(np.sqrt(np.mean(np.asarray(value) ** 2)))),
                mean_error=("error", "mean"),
            )
            .reset_index()
        )
        selected_model_name = str(overall.iloc[0]["model"])
        overall.to_csv(RUN_ROOT / "grouped_cv_model_summary.csv", index=False)
        distributions.to_csv(RUN_ROOT / "prediction_distributions.csv", index=False)
        errors_by_type.to_csv(RUN_ROOT / "errors_by_degradation_type.csv", index=False)
        selection = {
            "selection_source": "grouped_cv_only",
            "primary_metric": "mae",
            "tie_breaker": "rmse_then_model_name",
            "selected_model": selected_model_name,
            "holdout_used_for_selection": False,
            "models": overall.to_dict(orient="records"),
        }
        (RUN_ROOT / "model_selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        print(overall.to_string(index=False))
        print("Selected using grouped CV only:", selected_model_name)
        """
    ),
    code(
        """
        #@title 9. Fit selected development model; evaluate holdout exactly once if available
        selected_model = model_factories[selected_model_name]()
        selected_model.fit(X, y)
        holdout_report_path = RUN_ROOT / "independent_holdout_evaluation.json"
        if holdout_report_path.exists():
            raise RuntimeError(
                "Independent holdout output already exists for this immutable run; "
                "refusing to evaluate it a second time"
            )
        holdout_report = {
            "status": "unavailable",
            "reason": "No compatible held-out features were reproduced",
            "used_for_model_selection": False,
            "source_groups": HOLDOUT_SOURCE_GROUPS,
        }
        if not holdout_features.empty:
            X_holdout = holdout_features[list(REGISTRATION_FEATURE_NAMES)].copy()
            y_holdout = holdout_features["target_registration_0_100"].to_numpy(dtype=float)
            holdout_prediction = np.asarray(selected_model.predict(X_holdout), dtype=float)
            holdout_error = holdout_prediction - y_holdout
            holdout_output = holdout_features[["record_id", "source_group", "variant"]].copy()
            holdout_output["actual"] = y_holdout
            holdout_output["predicted"] = holdout_prediction
            holdout_output["error"] = holdout_error
            holdout_output["absolute_error"] = np.abs(holdout_error)
            holdout_output.to_csv(RUN_ROOT / "independent_holdout_predictions.csv", index=False)
            holdout_report = {
                "status": "completed_once",
                "model": selected_model_name,
                "record_count": len(holdout_output),
                "source_groups": sorted(holdout_output["source_group"].unique()),
                "mae": float(np.mean(np.abs(holdout_error))),
                "rmse": float(np.sqrt(np.mean(holdout_error ** 2))),
                "used_for_model_selection": False,
            }
        holdout_report_path.write_text(
            json.dumps(holdout_report, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        print(json.dumps(holdout_report, indent=2))
        """
    ),
    code(
        """
        #@title 10. Save candidate models, selected experimental artifact, and parity cases
        candidate_model_root = RUN_ROOT / "candidate_models"
        candidate_model_root.mkdir()
        fitted_candidates = {}
        for model_name, factory in model_factories.items():
            fitted = factory()
            fitted.fit(X, y)
            fitted_candidates[model_name] = fitted
            joblib.dump(fitted, candidate_model_root / f"{model_name}.joblib")

        artifact_root = RUN_ROOT / f"artifact_{MODEL_VERSION}"
        artifact_root.mkdir()
        joblib.dump(selected_model, artifact_root / "model.joblib")
        shutil.copy2(feature_spec_path, artifact_root / "feature_extractor.json")
        artifact_parity_cases = []
        for case in parity_cases:
            source_root = Path(case["case_root"])
            destination = artifact_root / "parity" / safe_tag(case["case_id"])
            shutil.copytree(source_root, destination)
            relative_manifest = destination.relative_to(artifact_root) / "frames_manifest.csv"
            relative_images = destination.relative_to(artifact_root) / "images"
            artifact_parity_cases.append({
                "case_id": case["case_id"],
                "input_type": "colmap_frame_manifest",
                "frames_manifest": str(relative_manifest),
                "frames_manifest_sha256": sha256_file(destination / "frames_manifest.csv"),
                "images_directory": str(relative_images),
                "expected_input_frame_count": 60,
                "source_group": case["source_group"],
                "variant": case["variant"],
                "expected_features": case["expected_features"],
            })
        parity_document = {
            "schema_version": "registration_feature_parity/1.0",
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 1e-6,
            "cases": artifact_parity_cases,
        }
        (artifact_root / "parity_examples.json").write_text(
            json.dumps(parity_document, indent=2, sort_keys=False) + "\\n", encoding="utf-8"
        )

        decision_counts = decisions_df["decision"].value_counts().to_dict()
        fidelity_counts = features_df["fidelity_tier"].value_counts().to_dict()
        experimental_manifest = {
            "schema_version": "experimental_registration_training/1.0",
            "run_id": RUN_ID,
            "model_version": MODEL_VERSION,
            "selected_model": selected_model_name,
            "experimental": True,
            "label_provenance_complete": PROVENANCE_COMPLETE,
            "activation_allowed": False,
            "reason": "Provisional COLMAP CSV labels and/or regenerated inputs retain incomplete provenance",
            "candidate_count_expected": EXPECTED_PROVISIONAL_CANDIDATES,
            "accepted_feature_record_count": len(features_df),
            "development_record_count": len(development_features),
            "holdout_record_count": len(holdout_features),
            "decision_counts": decision_counts,
            "fidelity_counts": fidelity_counts,
            "feature_names_in_order": list(REGISTRATION_FEATURE_NAMES),
            "feature_spec_sha256": feature_spec_sha256,
            "regeneration_contract_sha256": contract_sha256,
            "grouped_cv_selection": selection,
            "independent_holdout": holdout_report,
            "colmap_run": False,
            "heuristic_or_viewpoint_scores_modified": False,
        }
        (artifact_root / "experimental_training_manifest.json").write_text(
            json.dumps(experimental_manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        artifact_hashes = {
            str(path.relative_to(artifact_root)): sha256_file(path)
            for path in sorted(artifact_root.rglob("*"))
            if path.is_file()
        }
        (artifact_root / "experimental_files_sha256.json").write_text(
            json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        (RUN_ROOT / "evaluation_report.json").write_text(
            json.dumps(experimental_manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        print(json.dumps(experimental_manifest, indent=2))
        """
    ),
    code(
        """
        #@title 11. Optional deployment packaging — all legitimate gates must pass
        from video_suitability.ml.registration_features import load_feature_spec
        from video_suitability.ml.registration_rf import verify_training_feature_parity

        deployment_gates = {
            "explicitly_requested": REQUEST_DEPLOYMENT_PACKAGING,
            "provenance_complete": PROVENANCE_COMPLETE,
            "selected_model_is_random_forest": selected_model_name == "random_forest",
            "holdout_completed": holdout_report.get("status") == "completed_once",
            "all_inputs_original_hash_verified": set(features_df["fidelity_tier"]) == {"original_hash_verified"},
            "parity_cases_present": bool(artifact_parity_cases),
        }
        deployment_allowed = all(deployment_gates.values())
        deployment_status = {
            "allowed": deployment_allowed,
            "gates": deployment_gates,
            "artifact_manifest_created": False,
        }
        if deployment_allowed:
            spec = load_feature_spec(artifact_root / "feature_extractor.json")
            parity_result = verify_training_feature_parity(artifact_root, spec, parity_document)
            subprocess.check_call([
                sys.executable,
                str(DRIVE_PROJECT_ROOT / "scripts/ml/package_registration_rf.py"),
                "--artifact-dir", str(artifact_root),
                "--model-version", MODEL_VERSION,
                "--record-count", str(len(development_features)),
                "--training-run-id", RUN_ID,
            ])
            deployment_status.update({
                "artifact_manifest_created": True,
                "feature_parity": parity_result,
            })
        else:
            print("Deployment is blocked; no production artifact_manifest.json was created.")
            print(json.dumps(deployment_gates, indent=2))
        (RUN_ROOT / "deployment_status.json").write_text(
            json.dumps(deployment_status, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        """
    ),
    markdown(
        """
        ## Outputs and handoff

        The versioned `RUN_ROOT` contains:

        - input reproduction decisions with accepted/excluded/uncertain reasons;
        - exact production eight-feature records;
        - one frozen GroupKFold manifest shared by all models;
        - fold metrics, out-of-fold predictions, distributions, and degradation errors;
        - model selection based only on grouped-CV MAE/RMSE;
        - a once-only holdout report, or an explicit unavailable reason;
        - all four development-fitted candidate models;
        - a selected experimental artifact with hashed parity inputs;
        - `deployment_status.json` explaining every activation gate.

        Download or preserve the entire `RUN_ROOT`. Do not copy the experimental
        artifact into `models/registration_rf_current`. Only an artifact for which the
        final packaging cell creates `artifact_manifest.json` has passed the existing
        loader's hash, target, model-type, and feature-parity requirements.

        To reuse the workflow after evidence improves, create a new run ID and a new
        immutable regeneration contract. Never overwrite a prior run or edit its
        evaluation files after inspecting the independent holdout.
        """
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
