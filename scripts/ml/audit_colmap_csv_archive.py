#!/usr/bin/env python3
"""Audit the real 60-frame COLMAP CSV archive without training or running COLMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from video_suitability.ml.colmap_archive_audit import audit_colmap_csv_archive


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit-inventory-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_colmap_csv_archive(
        args.archive,
        args.out_dir,
        audit_inventory_csv=args.audit_inventory_csv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
