from pathlib import Path

import cv2
import numpy as np

from video_suitability.ml.historical_registration_features import (
    HISTORICAL_FEATURE_NAMES,
    LIVE_VIDEO_ADAPTER_CONTRACT_SHA256,
    extract_historical_features_from_image_directory,
    extract_historical_features_from_uploaded_video,
    extract_historical_features_from_video_probe,
    prepare_video_for_historical_probe,
    prepare_video_to_historical_60_jpegs,
)


ROOT = Path(__file__).resolve().parents[1]


def _reference_extract_features(image_dir):
    paths = sorted(image_dir.glob("*.jpg"))
    assert len(paths) == 60
    indices = np.linspace(0, len(paths) - 1, 12, dtype=int)
    brightness, contrast, blur, darkness = [], [], [], []
    orb_counts, match_counts = [], []
    previous_desc = None
    orb = cv2.ORB_create(nfeatures=400)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for i in indices:
        frame = cv2.imread(str(paths[i]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) > 480:
            scale = 480 / max(h, w)
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)))
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        blur.append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())))
        darkness.append(float((gray < 32).mean()))
        keypoints, desc = orb.detectAndCompute(gray, None)
        orb_counts.append(len(keypoints))
        if previous_desc is not None:
            if desc is not None and len(desc) >= 2:
                matches = matcher.knnMatch(previous_desc, desc, k=2)
                good = sum(
                    1
                    for pair in matches
                    if len(pair) == 2
                    and pair[0].distance < 0.75 * pair[1].distance
                )
                match_counts.append(good)
            else:
                match_counts.append(0)
        previous_desc = desc
    return {
        "brightness_mean": np.mean(brightness),
        "brightness_variation": np.std(brightness),
        "contrast_mean": np.mean(contrast),
        "sharpness_mean": np.mean(blur),
        "sharpness_variation": np.std(blur),
        "dark_fraction": np.mean(darkness),
        "orb_features_mean": np.mean(orb_counts),
        "orb_matches_mean": np.mean(match_counts),
    }


def _write_images(directory: Path):
    directory.mkdir()
    for index in range(60):
        yy, xx = np.indices((96, 128))
        frame = np.stack(
            (
                (xx * 3 + index * 11) % 256,
                (yy * 5 + index * 7) % 256,
                ((xx + yy) * 9 + index * 13) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        assert cv2.imwrite(str(directory / f"{index:04d}.jpg"), frame)


def test_recreated_historical_features_are_numerically_identical(tmp_path):
    images = tmp_path / "images"
    _write_images(images)
    expected = _reference_extract_features(images)
    actual, diagnostics = extract_historical_features_from_image_directory(images)
    assert tuple(expected) == tuple(actual) == HISTORICAL_FEATURE_NAMES
    assert actual == {name: float(expected[name]) for name in HISTORICAL_FEATURE_NAMES}
    assert diagnostics["sampled_positions_within_60"] == np.linspace(
        0, 59, 12, dtype=int
    ).tolist()


def test_video_adapter_remains_probe_only():
    features, diagnostics = extract_historical_features_from_video_probe(
        ROOT / "data/samples/good_orbit.mp4"
    )
    assert tuple(features) == HISTORICAL_FEATURE_NAMES
    assert diagnostics["video_adapter"]["adapter_status"] == (
        "probe_only_not_historically_verified"
    )
    assert diagnostics["live_inference_authorized"] is False


def test_live_adapter_is_exactly_equivalent_to_materialized_adapter_jpegs(tmp_path):
    video = ROOT / "data/samples/good_orbit.mp4"
    prepared = tmp_path / "prepared"
    preparation = prepare_video_to_historical_60_jpegs(video, prepared)
    materialized, _ = extract_historical_features_from_image_directory(prepared)
    optimized, diagnostics = extract_historical_features_from_uploaded_video(video)

    assert optimized == materialized
    assert tuple(optimized) == HISTORICAL_FEATURE_NAMES
    assert diagnostics["logical_prepared_frame_count"] == 60
    assert diagnostics["materialized_feature_frame_count"] == 12
    assert diagnostics["adapter_status"] == (
        "new_live_policy_training_input_equivalence_not_established"
    )
    assert diagnostics["requested_source_positions"] == diagnostics[
        "decoded_source_positions"
    ]
    assert preparation["prepared_frame_count"] == 60
    assert len(preparation["prepared_images"]) == 60
    assert preparation["decoded_positions"] == preparation["requested_positions"]
    assert preparation["adapter_contract_sha256"] == (
        "eac0ba4e28938adaa03a866c77ed7f2393fe5f45985a345b8f82ac4ac25dd9ef"
    )
    assert LIVE_VIDEO_ADAPTER_CONTRACT_SHA256 == preparation[
        "adapter_contract_sha256"
    ]


def test_streamlit_labels_historical_ridge_as_experimental():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Retired ML Experiment — Registration Predictor" in app_source
    assert "58.743399 percentage points" in app_source
    assert "No live prediction is run" in app_source
    assert "Converting the same inaccurate percentages into categories" in app_source
