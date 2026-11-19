#!/usr/bin/env python3
"""Build the focused Colab CPU notebook for the three-record R1 pilot."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks" / "r1_regeneration_pilot_colab.ipynb"


def _source(value: str):
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str):
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
        # R1 exact-regeneration pilot (three real labels, CPU)

        This focused notebook regenerates three ordered 60-frame R1 camera-path
        inputs and runs the production eight-feature extractor. It does **not** run
        COLMAP or train/activate a model. The R1 JPEG bytes remain explicitly
        unverified because labeling-time output hashes and the exact Pillow/libjpeg
        runtime were not saved.

        Run all cells after editing the paths below. The default pilot covers three
        physical source groups and high, middle, and low registration targets.
        """
    ),
    code(
        """
        #@title 1. Paths and options { display-mode: "form" }
        from datetime import datetime, timezone
        from pathlib import Path

        PROJECT_ROOT = Path("/content/drive/MyDrive/3D-vdieo-sutability-analyzer")  #@param {type:"string"}
        CO3D_ROOT = Path("/content/co3d_data")  #@param {type:"string"}
        EXECUTED_R1_NOTEBOOK = Path("/content/drive/MyDrive/co3d_R1_fourhour_colab(1).ipynb")  #@param {type:"string"}
        OUTPUT_PARENT = Path("/content/drive/MyDrive/r1_registration_feature_pilot")  #@param {type:"string"}
        DOWNLOAD_MISSING_CATEGORIES = False  #@param {type:"boolean"}

        RECORD_AUDIT = PROJECT_ROOT / "evaluation/ml/colmap_csv_archive_real_audit/record_audit.csv"
        FEATURE_SPEC = PROJECT_ROOT / "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
        RUN_ID = datetime.now(timezone.utc).strftime("r1_pilot_%Y%m%dT%H%M%SZ")
        PILOT_OUTPUT = OUTPUT_PARENT / RUN_ID
        print({"project": str(PROJECT_ROOT), "co3d": str(CO3D_ROOT), "output": str(PILOT_OUTPUT)})
        """
    ),
    code(
        """
        #@title 2. Mount Drive and install CPU dependencies
        import importlib.util
        import subprocess
        import sys

        from google.colab import drive
        drive.mount("/content/drive")

        required = [
            PROJECT_ROOT / "scripts/ml/run_r1_feature_pilot.py",
            RECORD_AUDIT,
            FEATURE_SPEC,
            EXECUTED_R1_NOTEBOOK,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Required pilot inputs are missing: " + "; ".join(missing))
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT),
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "Pillow>=9,<12",
        ])
        print("CPU pilot dependencies installed; COLMAP is not installed or invoked.")
        """
    ),
    code(
        """
        #@title 3. Optionally obtain only the three required CO3D categories
        import shutil

        def download_co3d_categories(categories):
            co3d_source = Path("/content/co3d-source")
            if not (co3d_source / "co3d/download_dataset.py").is_file():
                if co3d_source.exists():
                    raise RuntimeError(f"Incomplete CO3D source folder: {co3d_source}")
                subprocess.check_call([
                    "git", "clone", "--depth", "1",
                    "https://github.com/facebookresearch/co3d.git", str(co3d_source),
                ])
            for category in categories:
                subprocess.check_call([
                    sys.executable, str(co3d_source / "co3d/download_dataset.py"),
                    "--download_folder", str(CO3D_ROOT),
                    "--single_sequence_subset",
                    "--download_categories", category,
                    "--n_download_workers", "1", "--n_extract_workers", "1",
                    "--clear_archives_after_unpacking",
                ])

        required_sequences = [
            CO3D_ROOT / "apple/110_13051_23361/images",
            CO3D_ROOT / "backpack/537_78249_152122/images",
            CO3D_ROOT / "banana/610_96665_193716/images",
        ]
        missing_sequences = [path for path in required_sequences if not path.is_dir()]
        if missing_sequences and DOWNLOAD_MISSING_CATEGORIES:
            download_co3d_categories(["apple", "backpack", "banana"])
            missing_sequences = [path for path in required_sequences if not path.is_dir()]
        if missing_sequences:
            raise FileNotFoundError(
                "The pilot CO3D sequences are missing. Point CO3D_ROOT at an existing "
                "single-sequence subset or enable DOWNLOAD_MISSING_CATEGORIES: "
                + "; ".join(str(path) for path in missing_sequences)
            )
        for path in required_sequences:
            count = len([item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            print(path.parent.parent.name, path.parent.name, count, "source images")
        """
    ),
    code(
        """
        #@title 4. Regenerate three real R1 experiments and extract production features
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/ml/run_r1_feature_pilot.py"),
            "--record-audit", str(RECORD_AUDIT),
            "--co3d-root", str(CO3D_ROOT),
            "--feature-spec", str(FEATURE_SPEC),
            "--executed-r1-notebook", str(EXECUTED_R1_NOTEBOOK),
            "--output-dir", str(PILOT_OUTPUT),
        ]
        print("Running:", " ".join(command))
        completed = subprocess.run(command, text=True, capture_output=True)
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"R1 pilot did not accept all three records (exit {completed.returncode}). "
                f"Inspect {PILOT_OUTPUT / 'r1_pilot_decisions.csv'}"
            )
        """
    ),
    code(
        """
        #@title 5. Inspect and download the three-row result
        import json
        import pandas as pd
        from google.colab import files

        features = pd.read_csv(PILOT_OUTPUT / "r1_pilot_features.csv")
        decisions = pd.read_csv(PILOT_OUTPUT / "r1_pilot_decisions.csv")
        report = json.loads((PILOT_OUTPUT / "r1_pilot_report.json").read_text())
        display(features)
        display(decisions)
        print(json.dumps(report, indent=2))
        assert len(features) == 3
        assert report["feature_compatible_for_experimental_pipeline"] is True

        # The complete output remains in Drive. This CSV is the small training-pipeline handoff.
        files.download(str(PILOT_OUTPUT / "r1_pilot_features.csv"))
        """
    ),
    markdown(
        """
        ## Expand only after the pilot succeeds

        Set `EXPAND_ALL_R1 = True` below and run the final cell once. It selects every
        numeric R1 `sparse_model_created` label, reuses this same implementation, and
        writes a new immutable output directory. Missing CO3D source sequences are
        reported as exclusions; no COLMAP work is performed.
        """
    ),
    code(
        """
        #@title 6. Optional expansion to all reproducible R1 sparse-model labels
        EXPAND_ALL_R1 = False  #@param {type:"boolean"}
        if EXPAND_ALL_R1:
            all_r1_sequences = [
                CO3D_ROOT / "apple/110_13051_23361/images",
                CO3D_ROOT / "backpack/537_78249_152122/images",
                CO3D_ROOT / "ball/123_14363_28981/images",
                CO3D_ROOT / "banana/610_96665_193716/images",
                CO3D_ROOT / "book/119_13962_28926/images",
            ]
            missing_r1 = [path for path in all_r1_sequences if not path.is_dir()]
            if missing_r1 and DOWNLOAD_MISSING_CATEGORIES:
                missing_categories = sorted({path.parent.parent.name for path in missing_r1})
                download_co3d_categories(missing_categories)
                missing_r1 = [path for path in all_r1_sequences if not path.is_dir()]
            if missing_r1:
                raise FileNotFoundError(
                    "Full R1 expansion needs all five source sequences: "
                    + "; ".join(str(path) for path in missing_r1)
                )
            expansion_id = datetime.now(timezone.utc).strftime("r1_all_%Y%m%dT%H%M%SZ")
            expansion_output = OUTPUT_PARENT / expansion_id
            expansion_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/ml/run_r1_feature_pilot.py"),
                "--record-audit", str(RECORD_AUDIT),
                "--co3d-root", str(CO3D_ROOT),
                "--feature-spec", str(FEATURE_SPEC),
                "--executed-r1-notebook", str(EXECUTED_R1_NOTEBOOK),
                "--output-dir", str(expansion_output),
                "--all-r1-sparse",
            ]
            print("Running:", " ".join(expansion_command))
            expansion = subprocess.run(expansion_command, text=True)
            print("Exit:", expansion.returncode, "Output:", expansion_output)
            if expansion.returncode != 0:
                print("Some R1 rows were excluded; inspect r1_pilot_decisions.csv before training.")
        else:
            print("Expansion disabled. Complete and inspect the three-record pilot first.")
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
