"""Validated, dependency-light CO3D annotation and sequence loading."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..video import uniform_sample_indices


class CO3DValidationError(RuntimeError):
    """Raised when CO3D annotations or paths violate the expected schema."""


def load_jgz(path: Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise CO3DValidationError(f"Missing CO3D annotation file: {source}")
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CO3DValidationError(f"Invalid gzipped JSON annotation: {source}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise CO3DValidationError(f"CO3D annotation must be a list of objects: {source}")
    return value


def _first_nonempty(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


@dataclass(frozen=True)
class CO3DFrame:
    sequence_id: str
    frame_number: int
    timestamp_seconds: Optional[float]
    image_path: Path
    image_size_hw: Optional[Tuple[int, int]]
    has_mask: bool
    has_depth: bool
    has_viewpoint: bool
    raw: Dict[str, Any]


@dataclass(frozen=True)
class CO3DSequence:
    category: str
    sequence_name: str
    sequence_id: str
    object_id: Optional[str]
    scene_id: Optional[str]
    split_group_id: str
    split_group_provenance: str
    frames: Tuple[CO3DFrame, ...]
    sequence_metadata: Dict[str, Any]

    def uniform_sample(self, count: int) -> Tuple[CO3DFrame, ...]:
        indices = uniform_sample_indices(len(self.frames), count)
        return tuple(self.frames[index] for index in indices)

    def metadata(self) -> Dict[str, Any]:
        timestamps = [
            frame.timestamp_seconds
            for frame in self.frames
            if frame.timestamp_seconds is not None
        ]
        sizes = [frame.image_size_hw for frame in self.frames if frame.image_size_hw]
        return {
            "category": self.category,
            "sequence_name": self.sequence_name,
            "sequence_id": self.sequence_id,
            "object_id": self.object_id,
            "scene_id": self.scene_id,
            "split_group_id": self.split_group_id,
            "split_group_provenance": self.split_group_provenance,
            "frame_count": len(self.frames),
            "first_frame_number": self.frames[0].frame_number if self.frames else None,
            "last_frame_number": self.frames[-1].frame_number if self.frames else None,
            "first_timestamp_seconds": min(timestamps) if timestamps else None,
            "last_timestamp_seconds": max(timestamps) if timestamps else None,
            "image_sizes_hw": sorted({tuple(size) for size in sizes}),
            "mask_frame_count": sum(frame.has_mask for frame in self.frames),
            "depth_frame_count": sum(frame.has_depth for frame in self.frames),
            "viewpoint_frame_count": sum(frame.has_viewpoint for frame in self.frames),
        }


class CO3DSequenceLoader:
    """Load categories and sequences from an existing local CO3D tree.

    No data is downloaded. Image paths are constrained to the configured root, frame
    identities must be unique, and temporal order follows authoritative frame numbers.
    """

    def __init__(self, root: Path, *, verify_images: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.verify_images = verify_images
        if not self.root.is_dir():
            raise CO3DValidationError(f"CO3D root is not a directory: {self.root}")

    def categories(self) -> List[str]:
        return sorted(
            directory.name
            for directory in self.root.iterdir()
            if directory.is_dir()
            and (directory / "frame_annotations.jgz").is_file()
            and (directory / "sequence_annotations.jgz").is_file()
        )

    def audit_layout(self, categories: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        requested = list(categories) if categories is not None else self.categories()
        per_category: Dict[str, Any] = {}
        for category in requested:
            directory = self.root / category
            per_category[category] = {
                "exists": directory.is_dir(),
                "frame_annotations": (directory / "frame_annotations.jgz").is_file(),
                "sequence_annotations": (directory / "sequence_annotations.jgz").is_file(),
                "sequence_directories_with_images": len(
                    [
                        child
                        for child in directory.iterdir()
                        if child.is_dir() and (child / "images").is_dir()
                    ]
                )
                if directory.is_dir()
                else 0,
            }
        return {
            "root": str(self.root),
            "categories": requested,
            "per_category": per_category,
        }

    def _resolve_image_path(
        self, category: str, sequence_name: str, raw_path: str
    ) -> Path:
        annotation_path = Path(raw_path)
        candidates = []
        if annotation_path.is_absolute():
            candidates.append(annotation_path)
        else:
            candidates.extend(
                [
                    self.root / annotation_path,
                    self.root / category / annotation_path,
                    self.root / category / sequence_name / annotation_path,
                ]
            )
        selected = next((path.resolve() for path in candidates if path.is_file()), None)
        if selected is None:
            selected = candidates[0].resolve() if candidates else self.root
        try:
            selected.relative_to(self.root)
        except ValueError as exc:
            raise CO3DValidationError(f"Image path escapes CO3D root: {selected}") from exc
        if self.verify_images and not selected.is_file():
            raise CO3DValidationError(f"Annotated image is missing: {selected}")
        return selected

    def load_category(self, category: str) -> List[CO3DSequence]:
        category_dir = self.root / category
        frames_raw = load_jgz(category_dir / "frame_annotations.jgz")
        sequences_raw = load_jgz(category_dir / "sequence_annotations.jgz")
        sequence_meta: Dict[str, Dict[str, Any]] = {}
        for item in sequences_raw:
            name = str(item.get("sequence_name") or "").strip()
            if not name:
                raise CO3DValidationError(f"Sequence annotation without sequence_name in {category}")
            if name in sequence_meta:
                raise CO3DValidationError(f"Duplicate sequence annotation: {category}/{name}")
            sequence_meta[name] = item

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        identities = set()
        for item in frames_raw:
            name = str(item.get("sequence_name") or "").strip()
            if not name:
                raise CO3DValidationError(f"Frame annotation without sequence_name in {category}")
            try:
                frame_number = int(item["frame_number"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CO3DValidationError(
                    f"Invalid frame_number for {category}/{name}: {item.get('frame_number')}"
                ) from exc
            identity = (name, frame_number)
            if identity in identities:
                raise CO3DValidationError(f"Duplicate frame identity: {category}/{name}/{frame_number}")
            identities.add(identity)
            grouped.setdefault(name, []).append(item)

        unknown_sequence_names = sorted(set(grouped) - set(sequence_meta))
        if unknown_sequence_names:
            raise CO3DValidationError(
                f"Frames reference sequences missing from sequence annotations in "
                f"{category}: {unknown_sequence_names[:10]}"
            )
        sequences_without_frames = sorted(set(sequence_meta) - set(grouped))
        if sequences_without_frames:
            raise CO3DValidationError(
                f"Sequence annotations have no frames in {category}: "
                f"{sequences_without_frames[:10]}"
            )

        output: List[CO3DSequence] = []
        for name, raw_frames in sorted(grouped.items()):
            raw_frames.sort(key=lambda item: int(item["frame_number"]))
            sequence_id = f"co3d:{category}:{name}"
            metadata = sequence_meta.get(name, {})
            meta_block = metadata.get("meta") if isinstance(metadata.get("meta"), dict) else {}
            object_id = _first_nonempty(metadata, ["object_id", "object_name", "object_uid"])
            object_id = object_id or _first_nonempty(meta_block, ["object_id", "object_name", "object_uid"])
            scene_id = _first_nonempty(metadata, ["scene_id", "capture_id", "scene_name"])
            scene_id = scene_id or _first_nonempty(meta_block, ["scene_id", "capture_id", "scene_name"])
            if object_id:
                group_id = f"object:{category}:{object_id}"
                provenance = "verified_annotation_object_id"
            elif scene_id:
                group_id = f"scene:{category}:{scene_id}"
                provenance = "verified_annotation_scene_id"
            else:
                group_id = sequence_id
                provenance = "sequence_fallback_object_identity_unavailable"

            frames: List[CO3DFrame] = []
            for item in raw_frames:
                image = item.get("image") if isinstance(item.get("image"), dict) else {}
                raw_path = image.get("path")
                if not raw_path:
                    raise CO3DValidationError(
                        f"Frame has no image path: {category}/{name}/{item['frame_number']}"
                    )
                raw_size = image.get("size")
                size = None
                if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
                    size = (int(raw_size[0]), int(raw_size[1]))
                timestamp = item.get("frame_timestamp")
                frames.append(
                    CO3DFrame(
                        sequence_id=sequence_id,
                        frame_number=int(item["frame_number"]),
                        timestamp_seconds=float(timestamp) if timestamp is not None else None,
                        image_path=self._resolve_image_path(category, name, str(raw_path)),
                        image_size_hw=size,
                        has_mask=item.get("mask") is not None,
                        has_depth=item.get("depth") is not None,
                        has_viewpoint=item.get("viewpoint") is not None,
                        raw=item,
                    )
                )
            output.append(
                CO3DSequence(
                    category=category,
                    sequence_name=name,
                    sequence_id=sequence_id,
                    object_id=object_id,
                    scene_id=scene_id,
                    split_group_id=group_id,
                    split_group_provenance=provenance,
                    frames=tuple(frames),
                    sequence_metadata=metadata,
                )
            )
        return output

    def iter_sequences(
        self, categories: Optional[Sequence[str]] = None
    ) -> Iterator[CO3DSequence]:
        for category in categories or self.categories():
            yield from self.load_category(category)

