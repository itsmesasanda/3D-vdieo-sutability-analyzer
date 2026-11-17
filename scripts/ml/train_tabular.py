#!/usr/bin/env python3
"""Train the genuine-label Random Forest baseline with a frozen group split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_suitability.ml.schema import read_jsonl
from video_suitability.ml.splits import (
    SplitManifest,
    assert_no_group_leakage,
    grouped_split,
)
from video_suitability.ml.tabular import train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--classification-threshold", type=float, default=70.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # This production entry point intentionally has no synthetic-target switch.
    records = read_jsonl(args.dataset, require_targets=True)
    if args.split_manifest.exists():
        split = SplitManifest.read(args.split_manifest)
        assert_no_group_leakage(records, split)
    else:
        split = grouped_split(records, seed=args.seed)
        split.write(args.split_manifest)
    result = train_random_forest(
        records,
        split,
        args.artifacts,
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        classification_threshold=args.classification_threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

