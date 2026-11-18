#!/usr/bin/env python3
"""Build a compatibility-gated registration training bundle; never fit a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from video_suitability.ml.registration_training import (
    prepare_registration_training_bundle,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-csv",
        type=Path,
        action="append",
        required=True,
        help="Merged COLMAP label batch CSV; repeat for each batch",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument(
        "--feature-spec",
        type=Path,
        default=Path(
            "evaluation/ml/registration_rf_artifact_template/feature_extractor.json"
        ),
    )
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--parity-case-count", type=int, default=1)
    args = parser.parse_args(argv)
    report = prepare_registration_training_bundle(
        args.labels_csv,
        args.out_dir,
        args.feature_spec,
        bundle_version=args.bundle_version,
        fold_count=args.fold_count,
        seed=args.seed,
        parity_case_count=args.parity_case_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
