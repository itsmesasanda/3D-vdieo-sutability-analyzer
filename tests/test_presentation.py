import hashlib
from pathlib import Path

import app
import pytest

from video_suitability.presentation import (
    component_rows,
    extract_problem_frames,
    load_evaluation_rows,
    recommendation_cards,
    score_style,
    timeline_rows,
    viewpoint_diversity_view,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_report_to_timeline_recommendations_and_representative_frames(
    analysis_reports, synthetic_videos
):
    report = analysis_reports["blurry"]
    rows = timeline_rows(report)
    blur = [row for row in rows if row["problem_type"] == "blur"]
    assert blur
    assert blur[0]["label"] == "Possible blur / low edge detail"
    assert blur[0]["diagnostic_status"] == "suspected"
    assert "repeatable local details" in blur[0]["why"]
    assert 0 <= blur[0]["start_percent"] <= blur[0]["end_percent"] <= 100

    cards = recommendation_cards(rows)
    blur_card = next(card for card in cards if card["problem_type"] == "blur")
    assert blur_card["where"]
    assert blur_card["what"] and blur_card["why"] and blur_card["action"]

    video = synthetic_videos["blurry"]
    before = _sha256(video)
    frames = extract_problem_frames(video, rows, max_frames=4, max_side=320)
    assert frames
    assert any(frame["problem_type"] == "blur" for frame in frames)
    assert all(max(frame["image_rgb"].shape[:2]) <= 320 for frame in frames)
    assert _sha256(video) == before


def test_timeline_csv_preserves_diagnostic_status_and_timestamps(analysis_reports):
    rows = timeline_rows(analysis_reports["blurry"])
    payload = app._timeline_csv(rows)
    assert "diagnostic_status" in payload.splitlines()[0]
    assert "start_time_seconds" in payload.splitlines()[0]
    assert "suspected" in payload


def test_component_and_score_view_models_are_bounded(analysis_reports):
    rows = component_rows(analysis_reports["smooth_orbit"])
    assert len(rows) == 5
    assert sum(row["weight_percent"] for row in rows) == 100
    assert all(0 <= row["score"] <= 100 for row in rows)
    assert score_style(80)["label"] == "Likely suitable"
    assert score_style(55)["label"] == "Review recommended"
    assert score_style(20)["label"] == "Unlikely suitable"


def test_human_evaluation_preserves_missing_numerical_scores():
    root = Path(__file__).resolve().parents[1]
    rows = load_evaluation_rows(
        root / "evaluation/real_video_human_review.csv",
        root / "evaluation/reconstruction/results.csv",
    )
    assert len(rows) == 4
    assert sum(row["human_score"] is not None for row in rows) == 1
    assert next(row for row in rows if row["filename"] == "IMG_3913.MOV")[
        "human_score"
    ] == 80.0
    assert next(row for row in rows if row["filename"] == "IMG_3914.MOV")[
        "human_score"
    ] is None


def test_score_panel_separates_technical_score_from_geometric_readiness(
    analysis_reports,
):
    class FakeStreamlit:
        def __init__(self):
            self.markup = ""

        def markdown(self, value, **_kwargs):
            self.markup += value

    fake = FakeStreamlit()
    report = analysis_reports["smooth_orbit"]
    app._score_panel(fake, report)
    assert f"{report['overall_score']:.0f}" in fake.markup
    assert "Technical evidence score" in fake.markup
    assert "Geometric readiness" in fake.markup
    assert "Not verified" in fake.markup
    assert "reconstruction-success probability" in fake.markup
    assert "Likely suitable" not in fake.markup


def test_streamlit_app_renders_all_competition_views_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    rendered = AppTest.from_file(str(Path(app.__file__))).run(timeout=20)
    assert len(rendered.exception) == 0
    assert [tab.label for tab in rendered.tabs] == [
        "Analyzer",
        "Research archive",
        "Method & limitations",
    ]
    assert len(rendered.get("file_uploader")) == 1
    assert len(rendered.get("metric")) >= 3


def test_experimental_ml_panel_is_minimal_unreliable_and_separate():
    class FakeStreamlit:
        def __init__(self):
            self.markup = ""

        def markdown(self, value, **_kwargs):
            self.markup += str(value)

        def expander(self, *_args, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def dataframe(self, *_args, **_kwargs):
            return None

        def caption(self, value):
            self.markup += str(value)

    fake = FakeStreamlit()
    app._experimental_ridge_prediction(
        fake,
        {
            "experimental_ridge_prediction": {
                "status": "available",
                "predicted_registration_percentage": 99.9,
            }
        },
    )
    assert "Experimental ML Prediction" in fake.markup
    assert "Unreliable" in fake.markup
    assert "99.9%" in fake.markup
    assert "58.74 percentage points across four videos" in fake.markup
    assert "separate from the main heuristic score" in fake.markup


class _ViewpointRendererStub:
    def __init__(self):
        self.markup = ""
        self.messages = []

    def markdown(self, value, **_kwargs):
        self.markup += str(value)

    def expander(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def write(self, value):
        self.messages.append(str(value))

    def caption(self, value):
        self.messages.append(str(value))

    def warning(self, value):
        self.messages.append(str(value))


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("front_only", "Low"),
        ("quarter_orbit", "Moderate"),
        ("full_orbit", "High"),
        ("pure_rotation", "Unknown"),
        ("animated_object", "Unknown"),
    ],
)
def test_viewpoint_panel_renders_every_category_without_key_errors(
    analysis_reports, scenario, expected
):
    fake = _ViewpointRendererStub()
    app._viewpoint_diversity(fake, analysis_reports[scenario])
    assert expected in fake.markup
    assert "Substantial long-gap change" in fake.markup
    assert "Not available" not in fake.markup


def test_viewpoint_panel_marks_missing_optional_evidence_not_available():
    report = {
        "viewpoint_diversity": {
            "label": "Unknown",
            "summary": "Experimental analysis was not run.",
            "evidence_status": "insufficient",
            "method": {"affects_overall_score": False},
            "indicators": {},
            "decision_explanation": {},
            "recommendations": [],
            "warnings": [],
        }
    }
    view = viewpoint_diversity_view(report)
    assert view["substantial_change_percent"] is None
    assert view["parallax_proxy_percent"] is None
    assert view["mixed_motion_percent"] is None
    assert view["repeated_view_count"] is None

    fake = _ViewpointRendererStub()
    app._viewpoint_diversity(fake, report)
    assert fake.markup.count("Not available") == 7
    assert "Physical camera/object motion" in fake.markup
    assert "Not verified" in fake.markup


def test_viewpoint_view_model_tolerates_missing_or_malformed_result():
    for report in ({}, {"viewpoint_diversity": None}):
        view = viewpoint_diversity_view(report)
        assert view["label"] == "Unknown"
        assert view["substantial_change_percent"] is None


def test_viewpoint_panel_tolerates_older_partial_presentation_schema(monkeypatch):
    monkeypatch.setattr(
        app,
        "viewpoint_diversity_view",
        lambda _report: {
            "label": "Unknown",
            "summary": "Partial legacy presentation payload.",
            "style": {"color": "#6f7f7c"},
        },
    )
    fake = _ViewpointRendererStub()
    app._viewpoint_diversity(fake, {"viewpoint_diversity": {}})
    assert "Partial legacy presentation payload" in fake.markup
    assert fake.markup.count("Not available") == 7

    monkeypatch.setattr(app, "viewpoint_diversity_view", lambda _report: None)
    fallback = _ViewpointRendererStub()
    app._viewpoint_diversity(fallback, {})
    assert any("heuristic suitability score" in item for item in fallback.messages)

