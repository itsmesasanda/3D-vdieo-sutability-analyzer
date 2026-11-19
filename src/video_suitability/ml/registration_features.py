"""Strict OpenCV feature contract for COLMAP-registration RF inference.

This extractor is intentionally versioned separately from the heuristic feature
pipeline. A model artifact must name this exact implementation and carry the full
configuration used during training; otherwise inference is refused.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..video import VideoValidationError, uniform_sample_indices
from .artifacts import sha256_file


FEATURE_EXTRACTOR_SCHEMA_VERSION = "colmap_registration_opencv8/1.0"
FEATURE_EXTRACTOR_IMPLEMENTATION = (
    "video_suitability.ml.registration_features.extract/v1"
)
REGISTRATION_FEATURE_NAMES: Tuple[str, ...] = (
    "mean_brightness",
    "mean_contrast",
    "mean_sharpness",
    "mean_orb_keypoints",
    "mean_frame_difference",
    "mean_orb_good_matches",
    "mean_orb_match_ratio",
    "mean_orb_ransac_inlier_ratio",
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for registration features") from exc
    return cv2


def load_feature_spec(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_feature_spec(payload)
    return payload


def validate_feature_spec(spec: Mapping[str, Any]) -> None:
    """Reject incomplete or semantically different preprocessing metadata."""

    if spec.get("schema_version") != FEATURE_EXTRACTOR_SCHEMA_VERSION:
        raise ValueError("Unsupported registration feature schema")
    if spec.get("implementation") != FEATURE_EXTRACTOR_IMPLEMENTATION:
        raise ValueError("Unsupported registration feature implementation")
    if tuple(spec.get("feature_names_in_order", [])) != REGISTRATION_FEATURE_NAMES:
        raise ValueError("Registration model must use the exact eight-feature order")
    sampling = spec.get("frame_sampling", {})
    if sampling.get("strategy") != "uniform_round_linspace_full_video":
        raise ValueError("Unsupported registration frame-sampling strategy")
    sample_count = sampling.get("sample_count")
    if not isinstance(sample_count, int) or not 2 <= sample_count <= 120:
        raise ValueError("frame_sampling.sample_count must be an integer in [2, 120]")
    resize = spec.get("resize", {})
    if resize.get("mode") != "fixed_size":
        raise ValueError("Only exact fixed-size registration preprocessing is supported")
    if resize.get("interpolation") != "INTER_AREA":
        raise ValueError("Registration resizing must explicitly use INTER_AREA")
    for key in ("width", "height"):
        value = resize.get(key)
        if not isinstance(value, int) or not 64 <= value <= 1920:
            raise ValueError(f"resize.{key} must be an integer in [64, 1920]")
    if spec.get("grayscale_conversion") != "cv2.COLOR_BGR2GRAY":
        raise ValueError("Unsupported grayscale conversion")
    if spec.get("aggregation") != "arithmetic_mean_including_zero_match_pairs":
        raise ValueError("Unsupported registration-feature aggregation")
    orb = spec.get("orb", {})
    required_orb = {
        "nfeatures",
        "scaleFactor",
        "nlevels",
        "edgeThreshold",
        "firstLevel",
        "WTA_K",
        "scoreType",
        "patchSize",
        "fastThreshold",
        "rng_seed",
    }
    if required_orb - set(orb):
        raise ValueError("ORB configuration is incomplete")
    if orb.get("scoreType") not in {"HARRIS_SCORE", "FAST_SCORE"}:
        raise ValueError("Unsupported ORB score type")
    matching = spec.get("matching", {})
    if matching.get("matcher") != "BFMatcher_HAMMING_KNN":
        raise ValueError("Unsupported ORB matcher")
    if matching.get("k") != 2:
        raise ValueError("ORB KNN matching requires k=2")
    ratio = matching.get("lowe_ratio")
    if not isinstance(ratio, (int, float)) or not 0.5 <= float(ratio) < 1.0:
        raise ValueError("matching.lowe_ratio must be in [0.5, 1.0)")
    if matching.get("match_ratio_denominator") != "min_descriptor_count":
        raise ValueError("Unsupported match-ratio denominator")
    verification = spec.get("geometric_verification", {})
    if verification.get("model") != "homography_ransac":
        raise ValueError("Unsupported registration geometric verification")
    threshold = verification.get("reprojection_threshold_px")
    if not isinstance(threshold, (int, float)) or float(threshold) <= 0:
        raise ValueError("RANSAC reprojection threshold must be positive")
    formula = spec.get("formula_contract", {})
    expected_formula = {
        "brightness": "mean(gray)",
        "contrast": "std(gray)",
        "sharpness": "var(Laplacian(gray,CV_64F))",
        "orb_keypoints": "len(ORB.detectAndCompute.keypoints)",
        "frame_difference": "mean(abs(float32(gray_t1)-float32(gray_t0)))",
        "good_matches": "count(m.distance < lowe_ratio*n.distance)",
        "match_ratio": "good_matches/min(len(desc_t0),len(desc_t1))",
        "ransac_inlier_ratio": "homography_ransac_inliers/good_matches",
    }
    if formula != expected_formula:
        raise ValueError("Registration feature formulas do not match this implementation")


def _decode_sampled_frames(
    video_path: Path, spec: Mapping[str, Any]
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    cv2 = _cv2()
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(f"Could not open registration-feature video: {path}")
    frame_count_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(frame_count_raw) or frame_count_raw < 2:
        cap.release()
        raise VideoValidationError(
            "Registration feature parity requires a reliable codec frame count"
        )
    frame_count = int(round(frame_count_raw))
    requested = uniform_sample_indices(
        frame_count, int(spec["frame_sampling"]["sample_count"])
    )
    width = int(spec["resize"]["width"])
    height = int(spec["resize"]["height"])
    frames: List[np.ndarray] = []
    actual_indices: List[int] = []
    for index in requested:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise VideoValidationError(
                f"Could not decode required registration sample at frame {index}"
            )
        reported_next = cap.get(cv2.CAP_PROP_POS_FRAMES)
        actual = int(round(reported_next)) - 1 if reported_next > 0 else index
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(resized)
        actual_indices.append(max(0, actual))
    cap.release()
    if actual_indices != requested:
        raise VideoValidationError(
            "Codec seeking did not return the exact requested registration samples"
        )
    return frames, actual_indices, {
        "frame_count": frame_count,
        "requested_frame_indices": requested,
        "decoded_frame_indices": actual_indices,
        "resized_width": width,
        "resized_height": height,
    }


def verify_registration_frame_manifest(
    manifest_path: Path,
    *,
    images_dir: Optional[Path] = None,
    expected_input_frame_count: int = 60,
) -> Tuple[List[Dict[str, str]], Path]:
    """Verify the immutable COLMAP input manifest without decoding replacements."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise VideoValidationError(f"Frame manifest is missing: {manifest}")
    image_root = (
        Path(images_dir).expanduser().resolve()
        if images_dir is not None
        else (manifest.parent / "images").resolve()
    )
    if not image_root.is_dir():
        raise VideoValidationError(f"Frame image directory is missing: {image_root}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sequence_number",
            "filename",
            "actual_source_frame_index",
            "sha256",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise VideoValidationError(
                f"Frame manifest is missing columns: {sorted(missing)}"
            )
        rows = [dict(row) for row in reader]
    if len(rows) != expected_input_frame_count:
        raise VideoValidationError(
            f"Expected exactly {expected_input_frame_count} labeled input frames; "
            f"manifest contains {len(rows)}"
        )
    expected_sequence = list(range(expected_input_frame_count))
    try:
        actual_sequence = [int(row["sequence_number"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise VideoValidationError(
            "Frame manifest sequence_number values must be integers"
        ) from exc
    if actual_sequence != expected_sequence:
        raise VideoValidationError(
            "Frame manifest sequence_number values must be ordered and contiguous"
        )
    raw_source_indices = [row["actual_source_frame_index"].strip() for row in rows]
    if all(raw_source_indices):
        try:
            actual_source_indices = [int(value) for value in raw_source_indices]
        except ValueError as exc:
            raise VideoValidationError(
                "Frame manifest actual_source_frame_index values must be integers"
            ) from exc
        if actual_source_indices != sorted(set(actual_source_indices)):
            raise VideoValidationError(
                "Frame manifest source-frame indices must be unique and chronological"
            )
    elif not any(raw_source_indices):
        statuses = {
            (row.get("source_frame_index_status") or "").strip() for row in rows
        }
        if statuses != {"unavailable_preserved_colmap_input_order"}:
            raise VideoValidationError(
                "Blank source-frame indices require explicit preserved-COLMAP-order provenance"
            )
    else:
        raise VideoValidationError(
            "Frame manifest cannot mix known and unknown source-frame indices"
        )
    filenames = [row["filename"].strip() for row in rows]
    if any(not value for value in filenames) or len(set(filenames)) != len(filenames):
        raise VideoValidationError("Frame manifest filenames must be nonblank and unique")
    for row, filename in zip(rows, filenames):
        frame_path = (image_root / filename).resolve()
        try:
            frame_path.relative_to(image_root)
        except ValueError as exc:
            raise VideoValidationError(
                f"Frame manifest path escapes the image directory: {filename}"
            ) from exc
        if not frame_path.is_file():
            raise VideoValidationError(f"Labeled input frame is missing: {filename}")
        expected_hash = row["sha256"].strip().lower()
        if len(expected_hash) != 64 or sha256_file(frame_path) != expected_hash:
            raise VideoValidationError(
                f"Labeled input frame checksum mismatch: {filename}"
            )
    return rows, image_root


def _extract_registration_features_from_prepared_frames(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Run the one production feature computation shared by all input adapters."""

    cv2 = _cv2()
    orb_config = spec["orb"]
    score_type = (
        cv2.ORB_HARRIS_SCORE
        if orb_config["scoreType"] == "HARRIS_SCORE"
        else cv2.ORB_FAST_SCORE
    )
    cv2.setRNGSeed(int(orb_config["rng_seed"]))
    orb = cv2.ORB_create(
        nfeatures=int(orb_config["nfeatures"]),
        scaleFactor=float(orb_config["scaleFactor"]),
        nlevels=int(orb_config["nlevels"]),
        edgeThreshold=int(orb_config["edgeThreshold"]),
        firstLevel=int(orb_config["firstLevel"]),
        WTA_K=int(orb_config["WTA_K"]),
        scoreType=score_type,
        patchSize=int(orb_config["patchSize"]),
        fastThreshold=int(orb_config["fastThreshold"]),
    )
    grays: List[np.ndarray] = []
    keypoints: List[Sequence[Any]] = []
    descriptors: List[Any] = []
    brightness: List[float] = []
    contrast: List[float] = []
    sharpness: List[float] = []
    keypoint_counts: List[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, descriptor = orb.detectAndCompute(gray, None)
        points = points or []
        grays.append(gray)
        keypoints.append(points)
        descriptors.append(descriptor)
        brightness.append(float(np.mean(gray)))
        contrast.append(float(np.std(gray)))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        keypoint_counts.append(float(len(points)))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    lowe_ratio = float(spec["matching"]["lowe_ratio"])
    ransac_threshold = float(
        spec["geometric_verification"]["reprojection_threshold_px"]
    )
    frame_differences: List[float] = []
    good_counts: List[float] = []
    match_ratios: List[float] = []
    inlier_ratios: List[float] = []
    pair_diagnostics: List[Dict[str, Any]] = []
    for pair_index in range(len(frames) - 1):
        gray_a, gray_b = grays[pair_index], grays[pair_index + 1]
        desc_a, desc_b = descriptors[pair_index], descriptors[pair_index + 1]
        frame_difference = float(
            np.mean(np.abs(gray_b.astype(np.float32) - gray_a.astype(np.float32)))
        )
        good = []
        if desc_a is not None and desc_b is not None and len(desc_a) >= 2 and len(desc_b) >= 2:
            candidates = matcher.knnMatch(desc_a, desc_b, k=2)
            good = [
                first
                for candidate in candidates
                if len(candidate) == 2
                for first, second in [candidate]
                if first.distance < lowe_ratio * second.distance
            ]
        denominator = (
            min(len(desc_a), len(desc_b))
            if desc_a is not None and desc_b is not None
            else 0
        )
        match_ratio = len(good) / max(1, denominator)
        inlier_ratio = 0.0
        if len(good) >= 4:
            source = np.float32(
                [keypoints[pair_index][match.queryIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            target = np.float32(
                [keypoints[pair_index + 1][match.trainIdx].pt for match in good]
            ).reshape(-1, 1, 2)
            try:
                _, mask = cv2.findHomography(
                    source, target, cv2.RANSAC, ransac_threshold
                )
            except cv2.error:
                mask = None
            if mask is not None:
                inlier_ratio = float(mask.reshape(-1).astype(bool).mean())
        frame_differences.append(frame_difference)
        good_counts.append(float(len(good)))
        match_ratios.append(float(match_ratio))
        inlier_ratios.append(float(inlier_ratio))
        pair_diagnostics.append(
            {
                "start_frame_index": indices[pair_index],
                "end_frame_index": indices[pair_index + 1],
                "frame_difference": round(frame_difference, 9),
                "good_matches": len(good),
                "match_ratio": round(float(match_ratio), 9),
                "ransac_inlier_ratio": round(float(inlier_ratio), 9),
            }
        )
    vector = {
        "mean_brightness": float(np.mean(brightness)),
        "mean_contrast": float(np.mean(contrast)),
        "mean_sharpness": float(np.mean(sharpness)),
        "mean_orb_keypoints": float(np.mean(keypoint_counts)),
        "mean_frame_difference": float(np.mean(frame_differences)),
        "mean_orb_good_matches": float(np.mean(good_counts)),
        "mean_orb_match_ratio": float(np.mean(match_ratios)),
        "mean_orb_ransac_inlier_ratio": float(np.mean(inlier_ratios)),
    }
    if tuple(vector) != REGISTRATION_FEATURE_NAMES:
        raise RuntimeError("Registration feature order changed unexpectedly")
    if not np.isfinite(np.asarray(list(vector.values()), dtype=float)).all():
        raise RuntimeError("Registration features contain non-finite values")
    diagnostics = {
        **metadata,
        "feature_schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
        "feature_implementation": FEATURE_EXTRACTOR_IMPLEMENTATION,
        "sample_count": len(frames),
        "pair_count": len(pair_diagnostics),
        "pairs": pair_diagnostics,
    }
    return vector, diagnostics


def extract_registration_features(
    video_path: Path, spec: Mapping[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract the exact versioned eight-feature vector from one video."""

    validate_feature_spec(spec)
    frames, indices, metadata = _decode_sampled_frames(Path(video_path), spec)
    metadata["input_adapter"] = "video"
    return _extract_registration_features_from_prepared_frames(
        frames, indices, spec, metadata
    )


def extract_registration_features_from_frame_manifest(
    manifest_path: Path,
    spec: Mapping[str, Any],
    *,
    images_dir: Optional[Path] = None,
    expected_input_frame_count: int = 60,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract production features from the exact immutable COLMAP input frames."""

    validate_feature_spec(spec)
    rows, image_root = verify_registration_frame_manifest(
        manifest_path,
        images_dir=images_dir,
        expected_input_frame_count=expected_input_frame_count,
    )
    cv2 = _cv2()
    selected = uniform_sample_indices(
        len(rows), int(spec["frame_sampling"]["sample_count"])
    )
    width = int(spec["resize"]["width"])
    height = int(spec["resize"]["height"])
    frames: List[np.ndarray] = []
    selected_filenames: List[str] = []
    for position in selected:
        row = rows[position]
        filename = row["filename"].strip()
        frame = cv2.imread(str(image_root / filename), cv2.IMREAD_COLOR)
        if frame is None:
            raise VideoValidationError(
                f"Labeled input frame cannot be decoded: {filename}"
            )
        if row.get("width") and int(row["width"]) != frame.shape[1]:
            raise VideoValidationError(
                f"Labeled input frame width differs from manifest: {filename}"
            )
        if row.get("height") and int(row["height"]) != frame.shape[0]:
            raise VideoValidationError(
                f"Labeled input frame height differs from manifest: {filename}"
            )
        frames.append(
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        )
        selected_filenames.append(filename)
    metadata = {
        "input_adapter": "colmap_frame_manifest",
        "frame_count": len(rows),
        "requested_frame_indices": selected,
        "decoded_frame_indices": selected,
        "resized_width": width,
        "resized_height": height,
        "frames_manifest": str(Path(manifest_path).expanduser().resolve()),
        "images_directory": str(image_root),
        "expected_input_frame_count": expected_input_frame_count,
        "all_input_hashes_verified": True,
        "selected_filenames": selected_filenames,
    }
    return _extract_registration_features_from_prepared_frames(
        frames, selected, spec, metadata
    )


def feature_vector(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(features[name]) for name in REGISTRATION_FEATURE_NAMES])
