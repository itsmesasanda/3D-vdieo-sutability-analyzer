#!/usr/bin/env python3
"""Audit/import historical Colab COLMAP labels without rerunning COLMAP or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from video_suitability.ml.colab_label_import import import_colab_registration_labels


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, action="append", required=True)
    parser.add_argument("--batch-root", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--co3d-root", type=Path)
    parser.add_argument("--group-map-csv", type=Path)
    parser.add_argument(
        "--regeneration-sampling-strategy",
        choices=["legacy_floor_linspace", "production_round_linspace"],
    )
    parser.add_argument(
        "--regeneration-evidence",
        type=Path,
        help="Notebook/config file proving the declared regeneration sampling strategy",
    )
    parser.add_argument("--batch-id-prefix", default="colab-import")
    args = parser.parse_args(argv)
    report = import_colab_registration_labels(
        args.labels_csv,
        args.out_dir,
        batch_roots=args.batch_root,
        co3d_root=args.co3d_root,
        group_map_csv=args.group_map_csv,
        regeneration_sampling_strategy=args.regeneration_sampling_strategy,
        regeneration_evidence_path=args.regeneration_evidence,
        batch_id_prefix=args.batch_id_prefix,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["rejected_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

