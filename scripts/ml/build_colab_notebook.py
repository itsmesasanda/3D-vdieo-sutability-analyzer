#!/usr/bin/env python3
"""Generate the checked-in Colab Free ML preparation/training notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "notebooks" / "ml_training_colab.ipynb"


def _clean_source(source: str) -> str:
    """Normalize the readable quoted-line format used below."""

    cleaned = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('"'):
            line = stripped[1:]
        if line.endswith('"'):
            line = line[:-1]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip() + "\n"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _clean_source(source).splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _clean_source(source).splitlines(True),
    }


CELLS = [
    markdown(
        """# 3D Video Suitability — reconstruction-grounded ML training\n\n"
        "This Colab Free notebook trains only from a versioned JSONL dataset whose targets "
        "come from a documented reconstruction/human-review protocol. It rejects heuristic "
        "scores as targets and rejects synthetic targets by default. The Random Forest is the "
        "first baseline; frozen ResNet-18 embeddings and the temporal LSTM are optional.\n\n"
        "Outputs, split manifests, caches, checkpoints, metrics, and hashes are written to "
        "Google Drive. Synthetic smoke-test metrics are never real accuracy."
        """
    ),
    code(
        """# Configuration — edit these paths before running later cells.\n"
        "from pathlib import Path\n"
        "DRIVE_ROOT = Path('/content/drive/MyDrive')\n"
        "PROJECT_DIR = DRIVE_ROOT / '3D-video-suitability-analyzer'\n"
        "CO3D_ROOT = DRIVE_ROOT / 'co3d_data'\n"
        "DATASET_JSONL = DRIVE_ROOT / 'video_suitability_ml/dataset_v1.jsonl'\n"
        "OUTPUT_ROOT = DRIVE_ROOT / 'video_suitability_ml/runs'\n"
        "RUN_ID = 'genuine_labels_run_001'  # use a new ID for a new experiment\n"
        "SPLIT_SEED = 20261117\n"
        "CLASSIFICATION_THRESHOLD = 70.0  # fixed by label protocol; do not tune on test\n"
        "RUN_RANDOM_FOREST = True\n"
        "RUN_EMBEDDING_EXTRACTION = False  # enable after the tabular baseline\n"
        "RUN_TEMPORAL_LSTM = False\n"
        "ALLOW_SYNTHETIC_SMOKE = False  # only for explicit infrastructure smoke tests\n"
        "EMBEDDING_BATCH_SIZE = 32\n"
        "LSTM_BATCH_SIZE = 16\n"
        "LSTM_MAX_EPOCHS = 30\n"
        "RUN_DIR = OUTPUT_ROOT / RUN_ID\n"
        "print({'project': str(PROJECT_DIR), 'dataset': str(DATASET_JSONL), 'run': str(RUN_DIR)})\n"
        """
    ),
    code(
        """# Mount persistent Drive and verify paths before installing anything.\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "if not PROJECT_DIR.is_dir():\n"
        "    raise FileNotFoundError(f'Copy/clone this repository to {PROJECT_DIR}')\n"
        "if not DATASET_JSONL.is_file():\n"
        "    raise FileNotFoundError(f'Create/copy the versioned labeled JSONL to {DATASET_JSONL}')\n"
        "RUN_DIR.mkdir(parents=True, exist_ok=True)\n"
        "print('Persistent run directory:', RUN_DIR)\n"
        """
    ),
    code(
        """# Install only lightweight missing packages; Colab normally provides torch/torchvision.\n"
        "import json, os, platform, shutil, subprocess, sys, time\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_DIR), '--no-deps'])\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'scikit-learn>=1.3,<2', 'joblib>=1.3,<2'])\n"
        "import joblib, numpy as np, sklearn, torch\n"
        "try:\n"
        "    import torchvision\n"
        "    torchvision_status = torchvision.__version__\n"
        "except Exception as exc:\n"
        "    torchvision_status = f'unavailable: {exc}'\n"
        "disk = shutil.disk_usage('/content')\n"
        "memory_gib = None\n"
        "try:\n"
        "    memory_gib = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30\n"
        "except (ValueError, OSError, AttributeError):\n"
        "    pass\n"
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        "runtime = {\n"
        "    'utc_epoch_seconds': time.time(), 'python': platform.python_version(),\n"
        "    'torch': torch.__version__, 'torchvision': torchvision_status,\n"
        "    'scikit_learn': sklearn.__version__, 'device': DEVICE,\n"
        "    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "    'memory_gib': memory_gib, 'disk_free_gib': disk.free / 2**30,\n"
        "}\n"
        "print(runtime)\n"
        "(RUN_DIR / 'runtime_environment.json').write_text(json.dumps(runtime, indent=2) + '\\n')\n"
        "if (RUN_EMBEDDING_EXTRACTION or RUN_TEMPORAL_LSTM) and torchvision_status.startswith('unavailable'):\n"
        "    raise RuntimeError('torchvision is required for frozen ResNet-18 extraction')\n"
        """
    ),
    code(
        """# Validate schema and label provenance. Missing targets stop genuine training.\n"
        "from video_suitability.ml.schema import read_jsonl\n"
        "records = read_jsonl(\n"
        "    DATASET_JSONL, require_targets=True,\n"
        "    allow_synthetic_targets=ALLOW_SYNTHETIC_SMOKE,\n"
        ")\n"
        "scopes = {record.target.provenance.is_synthetic for record in records}\n"
        "if len(scopes) != 1:\n"
        "    raise ValueError('Synthetic and genuine targets cannot be mixed')\n"
        "if scopes == {True} and not ALLOW_SYNTHETIC_SMOKE:\n"
        "    raise ValueError('Synthetic smoke data is disabled')\n"
        "print({\n"
        "    'records': len(records), 'categories': len({r.category for r in records}),\n"
        "    'split_groups': len({r.split_group_id for r in records}),\n"
        "    'target_kinds': sorted({r.target.provenance.kind for r in records}),\n"
        "    'training_scope': 'synthetic_smoke_only' if scopes == {True} else 'genuine_labels',\n"
        "})\n"
        """
    ),
    code(
        """# Create once, then freeze and validate the physical-object/scene group split.\n"
        "from video_suitability.ml.splits import SplitManifest, assert_no_group_leakage, grouped_split\n"
        "split_path = RUN_DIR / 'split_manifest.json'\n"
        "if split_path.exists():\n"
        "    split_manifest = SplitManifest.read(split_path)\n"
        "else:\n"
        "    split_manifest = grouped_split(records, seed=SPLIT_SEED)\n"
        "    split_manifest.write(split_path)\n"
        "assert_no_group_leakage(records, split_manifest)\n"
        "print({name: len(getattr(split_manifest, f'{name}_sequence_ids')) for name in ['train','validation','test']})\n"
        """
    ),
    code(
        """# First baseline: video-level numerical features, train-only median imputation.\n"
        "from video_suitability.ml.artifacts import load_and_validate_manifest\n"
        "from video_suitability.ml.tabular import train_random_forest\n"
        "rf_root = RUN_DIR / 'random_forest'\n"
        "rf_artifact = rf_root / 'random_forest_numeric_cv__1.0'\n"
        "if RUN_RANDOM_FOREST:\n"
        "    if (rf_artifact / 'artifact_manifest.json').is_file():\n"
        "        print('Using completed RF artifact:', load_and_validate_manifest(rf_artifact))\n"
        "    elif rf_artifact.exists():\n"
        "        raise RuntimeError('Incomplete RF artifact exists; preserve it for diagnosis and choose a new RUN_ID')\n"
        "    else:\n"
        "        rf_result = train_random_forest(\n"
        "            records, split_manifest, rf_root,\n"
        "            allow_synthetic_targets=ALLOW_SYNTHETIC_SMOKE,\n"
        "            random_state=SPLIT_SEED,\n"
        "            classification_threshold=CLASSIFICATION_THRESHOLD,\n"
        "        )\n"
        "        print(rf_result)\n"
        "else:\n"
        "    print('Random Forest disabled by configuration')\n"
        """
    ),
    code(
        """# Optional frozen ImageNet ResNet-18 embeddings; cache keys include files/config.\n"
        "# Re-running this cell loads completed sequence caches instead of recomputing them.\n"
        "from video_suitability.ml.features import EmbeddingCache, ResNet18FrameEmbedder\n"
        "from video_suitability.ml.schema import write_jsonl\n"
        "embeddings_by_sequence = {}\n"
        "if RUN_EMBEDDING_EXTRACTION or RUN_TEMPORAL_LSTM:\n"
        "    cache = EmbeddingCache(OUTPUT_ROOT / 'embedding_cache_resnet18_v1')\n"
        "    embedder = ResNet18FrameEmbedder(device=DEVICE, batch_size=EMBEDDING_BATCH_SIZE)\n"
        "    cache_hits = 0\n"
        "    for index, record in enumerate(records, 1):\n"
        "        paths = [Path(p) if Path(p).is_absolute() else CO3D_ROOT / p for p in record.sampled_frame_paths]\n"
        "        missing = [str(path) for path in paths if not path.is_file()]\n"
        "        if missing:\n"
        "            raise FileNotFoundError(f'{record.sequence_id} missing frame: {missing[0]}')\n"
        "        key, values, hit = embedder.extract_cached(record.sequence_id, paths, cache)\n"
        "        cache_hits += int(hit)\n"
        "        embeddings_by_sequence[record.sequence_id] = values\n"
        "        record.embedding_cache_key = key\n"
        "        record.embedding_dimension = values.shape[1]\n"
        "        if index % 25 == 0 or index == len(records):\n"
        "            print(f'embeddings {index}/{len(records)}; cache hits={cache_hits}')\n"
        "    write_jsonl(records, RUN_DIR / 'dataset_with_embedding_keys.jsonl', allow_synthetic_targets=ALLOW_SYNTHETIC_SMOKE)\n"
        "else:\n"
        "    print('Embedding extraction disabled; no ImageNet weights are downloaded by this notebook')\n"
        """
    ),
    code(
        """# Optional bounded temporal LSTM head. Test metrics are computed only after selection.\n"
        "from video_suitability.ml.temporal import train_temporal_lstm\n"
        "lstm_root = RUN_DIR / 'temporal_lstm'\n"
        "lstm_artifact = lstm_root / 'frozen_resnet18_temporal_lstm__1.0'\n"
        "if RUN_TEMPORAL_LSTM:\n"
        "    if not embeddings_by_sequence:\n"
        "        raise RuntimeError('Enable embedding extraction before LSTM training')\n"
        "    if (lstm_artifact / 'artifact_manifest.json').is_file():\n"
        "        print('Using completed LSTM artifact:', load_and_validate_manifest(lstm_artifact))\n"
        "    elif lstm_artifact.exists():\n"
        "        raise RuntimeError('Incomplete LSTM artifact/checkpoints preserved; diagnose them and choose a new RUN_ID')\n"
        "    else:\n"
        "        lstm_result = train_temporal_lstm(\n"
        "            records, split_manifest, embeddings_by_sequence, lstm_root,\n"
        "            allow_synthetic_targets=ALLOW_SYNTHETIC_SMOKE,\n"
        "            batch_size=LSTM_BATCH_SIZE, max_epochs=LSTM_MAX_EPOCHS,\n"
        "            random_state=SPLIT_SEED, classification_threshold=CLASSIFICATION_THRESHOLD,\n"
        "            device=DEVICE,\n"
        "        )\n"
        "        print(lstm_result)\n"
        "else:\n"
        "    print('Temporal LSTM disabled by configuration')\n"
        """
    ),
    code(
        """# Final inventory. Interpret metrics only according to the recorded training scope.\n"
        "import json\n"
        "manifests = sorted(RUN_DIR.rglob('artifact_manifest.json'))\n"
        "inventory = []\n"
        "for path in manifests:\n"
        "    manifest = json.loads(path.read_text())\n"
        "    inventory.append({\n"
        "        'path': str(path), 'model_type': manifest['model_type'],\n"
        "        'model_version': manifest['model_version'],\n"
        "        'training_scope': manifest['training_scope'],\n"
        "    })\n"
        "print(json.dumps({'run_dir': str(RUN_DIR), 'artifacts': inventory}, indent=2))\n"
        "print('All listed files are already in persistent Google Drive storage.')\n"
        "print('Sparse registration alone does not establish object completeness; retain human review fields.')\n"
        """
    ),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "ml_training_colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {DESTINATION} with {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

