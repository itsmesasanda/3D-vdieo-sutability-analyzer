"""Reproduce the executed R1 camera-path inputs without running COLMAP.

This module is a direct extraction of the frame selection and Pillow processing in
code cells 6 and 8 of the executed ``co3d_R1_fourhour_colab`` notebook.  It creates
new hashes for regenerated files, but never describes them as the original bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import sha256_file


R1_FRAME_COUNT = 60
R1_MAX_EDGE = 800
R1_LABEL_VERSION = "registration_proxy_60f_v1"
R1_VARIANTS: Tuple[str, ...] = (
    "window_offset_p10_60",
    "window_offset_p24_60",
    "window_offset_p62_60",
    "window_offset_p90_60",
    "span_p00_p68",
    "span_p08_p77",
    "span_p25_p94",
    "span_p32_p100",
    "window_offset_p18_60",
    "window_offset_p40_60",
    "window_offset_p55_60",
    "window_offset_p80_60",
    "span_p03_p84",
    "span_p15_p95",
    "span_p22_p82",
    "span_p40_p98",
)

# The executed download inspected during implementation. Full-file identity is
# recorded when available; the two relevant code-cell hashes are the hard gate.
KNOWN_EXECUTED_NOTEBOOK_SHA256 = (
    "e6d66c8515b72ce6263fcb6f297427200713530b7e1038aa11c5f08135340717"
)
R1_CONSTANTS_CELL_SHA256 = (
    "e40c72338374fcbf6f94a0069f8fa20a8a35b810a38c233a6ec052249fa69475"
)
R1_REGENERATION_CELL_SHA256 = (
    "bdf5e240d7b4ec961cd9be564e05ee2ddc1bc3a39e5ebbae9a04d1c6f08ad83c"
)

DEFAULT_R1_PILOT_TAGS: Tuple[str, ...] = (
    "apple_110_13051_23361__span_p03_p84",
    "backpack_537_78249_152122__window_offset_p10_60",
    "banana_610_96665_193716__span_p25_p94",
)


class R1RegenerationError(ValueError):
    """Raised when an R1 input cannot be reproduced from explicit evidence."""


def _cell_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def verify_executed_r1_notebook(path: Path) -> Dict[str, Any]:
    """Verify the exact executed cells from which regeneration was extracted."""

    notebook_path = Path(path).expanduser().resolve()
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R1RegenerationError(f"Cannot read executed R1 notebook: {exc}") from exc

    constants_matches: List[Dict[str, Any]] = []
    regeneration_matches: List[Dict[str, Any]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        item = {
            "cell_index": index,
            "execution_count": cell.get("execution_count"),
            "sha256": _cell_sha256(source),
        }
        if "FRAME_COUNT=60" in source and "VARIANTS=(" in source:
            constants_matches.append(item)
        if "def _indices_for_view_variant" in source and "def make_variant" in source:
            regeneration_matches.append(item)

    if len(constants_matches) != 1 or len(regeneration_matches) != 1:
        raise R1RegenerationError(
            "Executed R1 notebook must contain exactly one constants cell and one "
            "frame-regeneration cell"
        )
    constants = constants_matches[0]
    regeneration = regeneration_matches[0]
    if constants["sha256"] != R1_CONSTANTS_CELL_SHA256:
        raise R1RegenerationError("R1 constants/variant cell differs from the inspected code")
    if regeneration["sha256"] != R1_REGENERATION_CELL_SHA256:
        raise R1RegenerationError("R1 frame-regeneration cell differs from the inspected code")
    if constants["execution_count"] is None or regeneration["execution_count"] is None:
        raise R1RegenerationError(
            "The supplied notebook has matching source but no execution evidence for the R1 cells"
        )

    notebook_hash = sha256_file(notebook_path)
    return {
        "path": str(notebook_path),
        "sha256": notebook_hash,
        "known_full_file_match": notebook_hash == KNOWN_EXECUTED_NOTEBOOK_SHA256,
        "constants_cell": constants,
        "regeneration_cell": regeneration,
        "implementation_verified": True,
    }


def r1_frame_indices(total_images: int, variant: str) -> List[int]:
    """Return the exact 60 positions selected by the executed R1 notebook."""

    n = int(total_images)
    if variant not in R1_VARIANTS:
        raise R1RegenerationError(f"Unknown R1 camera-path variant: {variant}")
    if variant.startswith("window_offset_p"):
        if n < 90:
            raise R1RegenerationError(
                f"Need >=90 frames for 60-frame window variants, got {n}"
            )
        fraction = int(variant.split("_")[2][1:]) / 100.0
        start = round((n - R1_FRAME_COUNT) * fraction)
        positions = list(range(start, start + R1_FRAME_COUNT))
    elif variant.startswith("span_p"):
        if n < 110:
            raise R1RegenerationError(
                f"Need >=110 frames for 60-frame span variants, got {n}"
            )
        pieces = variant.split("_")
        lower = int(pieces[1][1:]) / 100.0
        upper = int(pieces[2][1:]) / 100.0
        first = round((n - 1) * lower)
        last = round((n - 1) * upper)
        positions = np.linspace(first, last, R1_FRAME_COUNT, dtype=int).tolist()
    else:  # pragma: no cover - protected by the declared variant tuple
        raise R1RegenerationError(f"Unknown R1 selection family: {variant}")

    if not (
        len(positions) == R1_FRAME_COUNT
        and len(set(positions)) == R1_FRAME_COUNT
        and positions == sorted(positions)
        and 0 <= positions[0]
        and positions[-1] < n
    ):
        raise R1RegenerationError(
            f"R1 variant {variant} does not yield 60 unique chronological frames for n={n}"
        )

    # Preserve the executed notebook's same-batch duplicate check and variant order.
    for previous in R1_VARIANTS:
        if previous == variant:
            break
        earlier = r1_frame_indices(n, previous)
        if earlier == positions:
            raise R1RegenerationError(f"{variant} duplicates earlier variant {previous}")
    return positions


def _source_sequence(
    record: Mapping[str, Any], co3d_root: Path
) -> Tuple[str, str, str, Path, List[Path]]:
    category = str(record.get("category", "")).strip()
    source_group = str(
        record.get("source_group") or record.get("source_tag") or ""
    ).strip()
    variant = str(record.get("variant", "")).strip()
    if not category or not source_group or not variant:
        raise R1RegenerationError("R1 record requires category, source_group, and variant")
    prefix = category + "_"
    if not source_group.startswith(prefix) or len(source_group) <= len(prefix):
        raise R1RegenerationError(
            f"source_group {source_group!r} is inconsistent with category {category!r}"
        )
    sequence_id = source_group[len(prefix) :]
    images_directory = Path(co3d_root).expanduser().resolve() / category / sequence_id / "images"
    if not images_directory.is_dir():
        raise FileNotFoundError(
            f"CO3D sequence images are missing for {source_group}: {images_directory}"
        )
    originals = sorted(
        path
        for path in images_directory.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(originals) < R1_FRAME_COUNT:
        raise R1RegenerationError(
            f"CO3D sequence {source_group} has only {len(originals)} supported images"
        )
    return category, sequence_id, source_group, images_directory, originals


def regenerate_r1_experiment(
    *,
    record: Mapping[str, Any],
    co3d_root: Path,
    output_dir: Path,
    executed_notebook_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the ordered 60 JPEGs and a production-compatible frame manifest."""

    try:
        from PIL import Image, ImageOps, __version__ as pillow_version
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for exact R1 input regeneration") from exc

    category, sequence_id, source_group, _, originals = _source_sequence(
        record, co3d_root
    )
    variant = str(record["variant"])
    positions = r1_frame_indices(len(originals), variant)
    tag = str(record.get("tag") or f"{source_group}__{variant}")
    expected_tag = f"{source_group}__{variant}"
    if tag != expected_tag:
        raise R1RegenerationError(
            f"R1 tag {tag!r} does not equal source_group + variant ({expected_tag!r})"
        )

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"R1 output directory must be empty: {root}")
    image_root = root / "images"
    image_root.mkdir()
    manifest_rows: List[Dict[str, Any]] = []

    for sequence_number, source_position in enumerate(positions):
        original = originals[source_position]
        destination = image_root / f"{sequence_number:04d}.jpg"
        with Image.open(original) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            factor = min(1.0, R1_MAX_EDGE / max(image.size))
            size = tuple(max(1, round(side * factor)) for side in image.size)
            if size != image.size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            image.save(destination, "JPEG", quality=90)
        with Image.open(destination) as prepared:
            width, height = prepared.size
        manifest_rows.append(
            {
                "sequence_number": sequence_number,
                "filename": destination.name,
                "actual_source_frame_index": source_position,
                "sha256": sha256_file(destination),
                "width": width,
                "height": height,
                "source_relative_path": str(original.relative_to(Path(co3d_root).resolve())),
                "source_sha256": sha256_file(original),
            }
        )

    manifest_path = root / "frames_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    provenance = {
        "schema_version": "r1_regenerated_input/1.0",
        "implementation": "video_suitability.ml.r1_regeneration/v1",
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "tag": tag,
        "source_group": source_group,
        "category": category,
        "sequence_id": sequence_id,
        "variant": variant,
        "dataset": "CO3Dv2 single-sequence subset",
        "source_image_count": len(originals),
        "source_ordering": "Python sorted(Path) over immediate images directory",
        "selected_source_positions": positions,
        "frame_count": R1_FRAME_COUNT,
        "transformation": {
            "exif": "PIL.ImageOps.exif_transpose",
            "color": "PIL.Image.convert('RGB')",
            "max_edge": R1_MAX_EDGE,
            "resize": "proportional_PIL.Image.Resampling.LANCZOS_if_needed",
            "size_rounding": "Python round per side",
            "encoding": "PIL JPEG quality=90 with other save parameters at defaults",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "executed_notebook_evidence": dict(executed_notebook_evidence or {}),
        "fidelity_tier": "executed_contract_reproduced",
        "original_hashes_verified": False,
        "byte_fidelity_status": "unverified",
        "byte_fidelity_reason": (
            "The R1 run did not preserve original prepared-image hashes or its exact "
            "Pillow/libjpeg version; regenerated source/output hashes are newly measured."
        ),
        "frames_manifest_sha256": sha256_file(manifest_path),
    }
    provenance_path = root / "regeneration_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "tag": tag,
        "variant": variant,
        "source_group": source_group,
        "frames_manifest": manifest_path,
        "images_directory": image_root,
        "regeneration_provenance": provenance_path,
        "original_hashes_verified": False,
        "fidelity_tier": "executed_contract_reproduced",
        "implementation_sha256": provenance["implementation_sha256"],
        "selected_source_positions": positions,
    }


def select_r1_label_records(
    record_audit_csv: Path,
    *,
    tags: Optional[Sequence[str]] = None,
    all_sparse_records: bool = False,
) -> List[Dict[str, Any]]:
    """Select real numeric R1 sparse-model labels without re-auditing the archive."""

    if bool(tags) == bool(all_sparse_records):
        raise ValueError("Specify either explicit tags or all_sparse_records=True")
    with Path(record_audit_csv).open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    selected_tags = set(str(value) for value in tags or [])
    selected = [
        row
        for row in rows
        if row.get("batch_id") == "R1"
        and (
            all_sparse_records
            or row.get("tag") in selected_tags
        )
    ]
    if not all_sparse_records:
        found = {row.get("tag") for row in selected}
        missing = selected_tags - found
        if missing:
            raise R1RegenerationError(f"Requested R1 labels are missing: {sorted(missing)}")
        selected.sort(key=lambda row: list(tags or []).index(str(row["tag"])))
    else:
        selected.sort(key=lambda row: str(row.get("tag", "")))

    valid: List[Dict[str, Any]] = []
    for row in selected:
        if row.get("status") != "sparse_model_created":
            if all_sparse_records:
                continue
            raise R1RegenerationError(f"R1 record is not a sparse-model result: {row['tag']}")
        if row.get("classification") != "recoverable_success_missing_evidence":
            raise R1RegenerationError(f"R1 record is not a numeric measured result: {row['tag']}")
        try:
            sampled = int(float(row["sampled_images"]))
            registered = int(float(row["registered_images"]))
            declared_target = float(row["target_registration_0_100"])
        except (KeyError, TypeError, ValueError) as exc:
            raise R1RegenerationError(f"R1 record has invalid numeric fields: {row.get('tag')}") from exc
        if sampled != R1_FRAME_COUNT or not 0 <= registered <= sampled:
            raise R1RegenerationError(f"R1 record has invalid image counts: {row['tag']}")
        canonical_target = registered * 100.0 / sampled
        if not np.isclose(declared_target, canonical_target, atol=0.005001, rtol=0.0):
            raise R1RegenerationError(f"R1 target/count mismatch: {row['tag']}")
        normalized = dict(row)
        normalized.update(
            {
                "sampled_images": sampled,
                "registered_images": registered,
                "declared_target_registration_0_100": declared_target,
                "target_registration_0_100": canonical_target,
                "label_version": R1_LABEL_VERSION,
                "label_provenance": "COLMAP_sparse_reconstruction_R1_csv",
                "label_status": "provisional",
                "reconstruction_status": row["status"],
                "source_group_provenance": "R1 CSV source_tag/source_group",
            }
        )
        valid.append(normalized)
    return valid
