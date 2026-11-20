"""Polished Streamlit competition MVP for the stable heuristic JSON contract."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from video_suitability.config import AnalysisConfig
from video_suitability.infer import analyze_video
from video_suitability.ml.experimental_ridge import (
    MODEL_SHA256 as EXPERIMENTAL_RIDGE_MODEL_SHA256,
    attach_experimental_ridge_prediction,
)
from video_suitability.presentation import (
    capture_completeness_view,
    component_rows,
    extract_problem_frames,
    load_evaluation_rows,
    recommendation_cards,
    registration_prediction_view,
    score_style,
    timeline_rows,
    viewpoint_diversity_view,
)
from video_suitability.object_focus import TRACKER_IMPLEMENTATION_VERSION
from video_suitability.runtime_provenance import runtime_provenance
from video_suitability.video import uniform_sample_indices


MAX_UPLOAD_MB = 150
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}
STREAMLIT_ML_IDENTITY = ("random_forest_disabled", "deployment_2026_11")
OBJECT_SELECTION_PREVIEW_POLICY = "middle_decoded_uniform_sample_v2"


def _analysis_cache_identity() -> Any:
    """Invalidate session results when the analyzer/tracker implementation changes."""

    provenance = runtime_provenance()
    return (
        "streamlit_analysis_v2",
        TRACKER_IMPLEMENTATION_VERSION,
        provenance.get("object_tracker_source_sha256"),
        provenance.get("inference_source_sha256"),
        OBJECT_SELECTION_PREVIEW_POLICY,
    )


def _disabled_registration_prediction() -> Dict[str, Any]:
    """Return the unchanged Random Forest cloud policy."""

    return {
        "status": "unavailable",
        "reason": (
            "Live Random Forest inference remains disabled because no saved artifact "
            "has passed its production input-feature parity gates."
        ),
        "trained_model_used": False,
        "colmap_run": False,
        "affects_suitability_score": False,
        "affects_viewpoint_diversity": False,
    }


def _not_requested_experimental_ridge() -> Dict[str, Any]:
    return {
        "status": "not_requested",
        "reason": "Enable the opt-in experimental ML panel to run this estimate.",
        "trained_model_used": False,
        "colmap_run": False,
        "affects_suitability_score": False,
        "affects_viewpoint_diversity": False,
        "experimental": True,
        "independently_validated": False,
    }


def _streamlit_ridge_model_bytes(st: Any) -> Any:
    """Decode the optional cloud secret; integrity is checked before joblib loads it."""

    try:
        encoded = st.secrets.get("EXPERIMENTAL_RIDGE_ARTIFACT_B64")
    except Exception:
        encoded = None
    if not encoded:
        return None
    try:
        return base64.b64decode(str(encoded), validate=True)
    except Exception:
        # The prediction module will safely report the absent/invalid artifact. Do not
        # expose secret contents or decoding internals to ordinary users.
        return b"invalid-secret"


def _ridge_cache_identity(enabled: bool, model_bytes: Any) -> Any:
    if not enabled:
        return ("experimental_ridge", "not_requested")
    if model_bytes is not None:
        return ("experimental_ridge", hashlib.sha256(model_bytes).hexdigest())
    return ("experimental_ridge", "local", EXPERIMENTAL_RIDGE_MODEL_SHA256)


APP_CSS = """
<style>
  :root {
    --ink:#18212f; --muted:#5d6978; --line:#e4e8ee; --surface:#ffffff;
    --canvas:#f5f7fa; --coral:#e85f57; --coral-dark:#c84641; --coral-soft:#fff2f0;
    --green:#237a62; --amber:#a86816; --shadow:0 10px 30px rgba(24,33,47,.07);
  }
  html, body, [class*="css"] { color:var(--ink); }
  .stApp { background:var(--canvas); color:var(--ink); }
  .block-container { max-width:1200px; padding-top:1.4rem; padding-bottom:4rem; }
  [data-testid="stHeader"] { background:rgba(245,247,250,.94); }
  h1, h2, h3, h4, p, label, .stMarkdown { color:var(--ink); }
  .mvp-hero { position:relative; overflow:hidden; padding:2rem 2.1rem; border:1px solid var(--line); border-radius:22px; background:var(--surface); box-shadow:var(--shadow); margin-bottom:1.25rem; }
  .mvp-hero:before { content:""; position:absolute; inset:0 auto 0 0; width:7px; background:linear-gradient(180deg,var(--coral),#f3a071); }
  .mvp-kicker { color:var(--coral-dark); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; }
  .mvp-hero h1 { color:var(--ink); margin:.4rem 0 .55rem; font-size:clamp(2rem,4vw,3.1rem); line-height:1.04; letter-spacing:-.035em; }
  .mvp-hero p { color:var(--muted); max-width:780px; margin:0; font-size:1.02rem; line-height:1.6; }
  .method-chip { display:inline-flex; align-items:center; margin-top:1rem; padding:.38rem .72rem; border-radius:999px; background:var(--coral-soft); color:#913c38; border:1px solid #ffd9d5; font-size:.76rem; font-weight:700; }
  .section-heading { margin:1.65rem 0 .75rem; }
  .section-heading .eyebrow { margin-bottom:.2rem; }
  .section-heading h2 { margin:0; font-size:1.45rem; letter-spacing:-.02em; }
  .section-heading p { margin:.3rem 0 0; color:var(--muted); }
  .upload-intro { padding:1rem 1.1rem; margin:.2rem 0 .75rem; border-radius:14px; background:var(--surface); border:1px solid var(--line); color:var(--muted); }
  .upload-intro strong { color:var(--ink); }
  .score-panel { display:flex; align-items:center; gap:1.6rem; padding:1.5rem; border-radius:20px; background:var(--surface); border:1px solid var(--line); box-shadow:var(--shadow); }
  .score-ring { width:142px; height:142px; border-radius:50%; display:grid; place-items:center; position:relative; flex:0 0 auto; }
  .score-ring:before { content:""; position:absolute; inset:11px; background:white; border-radius:50%; }
  .score-number { position:relative; z-index:2; font-size:2.45rem; font-weight:850; color:var(--ink); line-height:.9; }
  .score-number small { display:block; font-size:.72rem; color:var(--muted); font-weight:700; text-align:center; margin-top:.4rem; }
  .score-copy h2 { margin:.18rem 0 .35rem; color:var(--ink); font-size:1.65rem; }
  .score-copy p { margin:.2rem 0; color:var(--muted); line-height:1.5; }
  .component-card { min-height:118px; padding:1rem; border-radius:15px; background:var(--surface); border:1px solid var(--line); box-shadow:0 5px 16px rgba(24,33,47,.04); margin-bottom:.45rem; }
  .component-card .label { font-size:.78rem; color:var(--muted); min-height:2.35rem; font-weight:650; }
  .component-card .value { font-size:1.7rem; font-weight:800; color:var(--ink); }
  .attention-card { height:100%; padding:1rem; border-radius:14px; background:#fff9f8; border:1px solid #f4d5d2; }
  .attention-card h4 { margin:0 0 .35rem; color:#7e302d; font-size:.95rem; }
  .attention-card p { margin:0; color:#674c4a; font-size:.86rem; line-height:1.45; }
  .viewpoint-panel { margin-top:.35rem; padding:1.25rem 1.35rem; border-radius:18px; background:var(--surface); border:1px solid #ead8d6; border-left:5px solid var(--coral); box-shadow:0 7px 22px rgba(24,33,47,.05); }
  .viewpoint-label { display:inline-block; padding:.28rem .68rem; border-radius:999px; color:white; font-size:.75rem; font-weight:800; letter-spacing:.03em; vertical-align:middle; }
  .viewpoint-panel h3 { margin:.5rem 0 .3rem; color:var(--ink); }
  .viewpoint-panel p { margin:.3rem 0; color:var(--muted); line-height:1.5; }
  .evidence-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; margin-top:.9rem; }
  .evidence-item { padding:.7rem; border-radius:11px; background:#f8f9fb; border:1px solid #eceff3; }
  .evidence-item span { display:block; color:var(--muted); font-size:.7rem; line-height:1.25; }
  .evidence-item strong { display:block; color:var(--ink); font-size:1.05rem; margin-top:.18rem; }
  .registration-panel { margin-top:1rem; padding:1.05rem 1.2rem; border-radius:15px; background:#f8f9fb; border:1px solid var(--line); }
  .registration-panel h3 { margin:.35rem 0; color:var(--ink); }
  .registration-panel p { margin:.25rem 0; color:var(--muted); }
  .registration-value { font-size:2rem; font-weight:800; color:#294b7a; }
  .timeline-row { position:relative; padding:.9rem 1rem .82rem; background:var(--surface); border:1px solid var(--line); border-radius:13px; margin:.5rem 0; overflow:hidden; }
  .timeline-row strong { color:var(--ink); }
  .timeline-meta { color:var(--muted); font-size:.82rem; }
  .timeline-track { height:6px; background:#edf0f4; border-radius:9px; margin-top:.65rem; position:relative; overflow:hidden; }
  .timeline-span { position:absolute; height:100%; min-width:5px; border-radius:9px; background:linear-gradient(90deg,#f09a68,var(--coral)); }
  .recommendation { border-left:4px solid var(--coral); padding:.9rem 1rem; background:var(--surface); border-radius:0 13px 13px 0; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); margin:.6rem 0; }
  .recommendation h4 { margin:0 0 .35rem; color:var(--ink); }
  .recommendation p { margin:.24rem 0; color:var(--muted); font-size:.9rem; }
  .eyebrow { color:var(--coral-dark); font-weight:800; letter-spacing:.09em; text-transform:uppercase; font-size:.7rem; }
  .disclaimer { padding:.9rem 1rem; background:#fff8e9; border:1px solid #ead7a7; border-radius:12px; color:#634f20; }
  .step-card { min-height:112px; padding:1rem; border-radius:14px; background:var(--surface); border:1px solid var(--line); }
  .step-number { color:var(--coral-dark); font-size:.72rem; font-weight:850; letter-spacing:.08em; }
  .step-card strong { display:block; margin:.3rem 0; color:var(--ink); }
  .step-card p { margin:0; color:var(--muted); font-size:.86rem; }
  div[data-testid="stFileUploader"] { background:var(--surface); border-radius:18px; padding:.45rem; border:1px solid var(--line); box-shadow:var(--shadow); }
  [data-testid="stFileUploaderDropzone"] { min-height:150px; background:#fffafa; border:2px dashed #e7aaa6; border-radius:14px; }
  [data-testid="stFileUploaderDropzone"] * { color:var(--ink) !important; }
  [data-testid="stFileUploaderDropzone"] small { color:var(--muted) !important; }
  [data-testid="stCheckbox"] label p, [data-testid="stSlider"] label p, [data-testid="stWidgetLabel"] p { color:var(--ink) !important; font-weight:600; }
  [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:13px; overflow:hidden; }
  [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:var(--ink) !important; font-weight:700; }
  .stTabs [data-baseweb="tab-list"] { gap:.35rem; background:#eef1f5; padding:.32rem; border-radius:12px; }
  .stTabs [data-baseweb="tab"] { height:42px; border-radius:9px; color:#475466; font-weight:700; }
  .stTabs [aria-selected="true"] { background:white !important; color:var(--ink) !important; box-shadow:0 2px 8px rgba(24,33,47,.08); }
  .stTabs [data-baseweb="tab-highlight"] { background:var(--coral); }
  .stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:750; min-height:44px; }
  .stButton > button[kind="primary"] { background:var(--coral); border-color:var(--coral); color:white; }
  .stButton > button[kind="primary"]:hover { background:var(--coral-dark); border-color:var(--coral-dark); }
  [data-testid="stProgressBar"] > div > div { background:var(--coral); }
  [data-testid="stAlert"] { border-radius:12px; color:var(--ink); }
  [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:13px; padding:.8rem; }
  @media (max-width: 780px) {
    .block-container { padding:1rem .75rem 3rem; }
    .mvp-hero { padding:1.35rem 1.2rem; border-radius:17px; }
    .mvp-hero h1 { font-size:2rem; }
    .score-panel { align-items:flex-start; gap:1rem; padding:1.1rem; }
    .score-ring { width:108px; height:108px; }
    .score-number { font-size:1.85rem; }
    .score-copy h2 { font-size:1.3rem; }
    .evidence-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .component-card { min-height:auto; }
    .stTabs [data-baseweb="tab"] { font-size:.78rem; padding-left:.55rem; padding-right:.55rem; }
  }
  @media (max-width: 460px) {
    .score-panel { flex-direction:column; }
    .evidence-grid { grid-template-columns:1fr; }
    .method-chip { line-height:1.35; }
  }
</style>
"""


def _hero(st: Any) -> None:
    st.markdown(
        """
        <section class="mvp-hero">
          <div class="mvp-kicker">3D capture readiness</div>
          <h1>Check your video before reconstruction</h1>
          <p>Review image quality, feature continuity, view overlap and camera motion using a transparent, CPU-based analysis.</p>
          <span class="method-chip">Explainable heuristic · temporary upload processing · no reconstruction claim</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _section_heading(st: Any, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _display_metric(value: Any, suffix: str = "%") -> str:
    """Format optional presentation evidence without implying a measured zero."""

    if value is None:
        return "Not available"
    return f"{value}{suffix}"


def _friendly_analysis_error(exc: Exception) -> str:
    """Return a user-facing upload error without exposing implementation details."""

    message = str(exc).lower()
    if "deployment limit" in message or "unsupported video extension" in message:
        return str(exc)
    if "could not open video" in message or "codec" in message or "corrupt" in message:
        return (
            "We could not decode this video. Try exporting it as MP4 (H.264 or HEVC) "
            "or MOV, then upload the new file."
        )
    return (
        "We could not complete this analysis. Your main data was not retained. "
        "Try a shorter MP4/MOV file or reload the app and try again."
    )


def _score_panel(st: Any, report: Dict[str, Any]) -> None:
    score = float(report["overall_score"])
    style = score_style(score)
    baseline = float(report["baseline_v1"]["overall_score"])
    revision = report.get("scoring_revision", "2.0")
    st.markdown(
        f"""
        <div class="score-panel">
          <div class="score-ring" style="background:conic-gradient({style['color']} {score:.2f}%,#e8edec 0)">
            <div class="score-number">{score:.0f}<small>out of 100</small></div>
          </div>
          <div class="score-copy">
            <div class="eyebrow">Heuristic v2 · revision {revision}</div>
            <h2>{style['label']}</h2>
            <p>This is an evidence score, not a success probability.</p>
            <p>Preserved v1 baseline: <strong>{baseline:.2f}</strong> · change: <strong>{score-baseline:+.2f}</strong></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _components(st: Any, report: Dict[str, Any]) -> None:
    _section_heading(
        st,
        "Score breakdown",
        "Five signals behind the result",
        "Each metric is a measured heuristic component, not a success probability.",
    )
    columns = st.columns(5)
    for column, row in zip(columns, component_rows(report)):
        with column:
            weight = f" · {row['weight_percent']}% weight" if row["weight_percent"] else ""
            st.markdown(
                f"<div class='component-card'><div class='label'>{row['label']}{weight}</div>"
                f"<div class='value'>{row['score']:.0f}</div></div>",
                unsafe_allow_html=True,
            )
            st.progress(int(round(row["score"])))
            st.caption(row["explanation"])


def _attention_summary(st: Any, rows: List[Dict[str, Any]]) -> None:
    _section_heading(
        st,
        "Priority review",
        "What to improve first",
        "The most important sampled issues are summarized here; open the timeline for exact intervals.",
    )
    cards = recommendation_cards(rows)[:3]
    if not cards:
        st.success(
            "No high-priority sampled problem was detected. Still verify that every required side of the object is visible."
        )
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="attention-card">
                  <h4>{html.escape(card['label'])} · {html.escape(card['where'])}</h4>
                  <p>{html.escape(card['action'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _uncertainty(st: Any, report: Dict[str, Any]) -> None:
    uncertainty = report["confidence_or_uncertainty"]
    level = uncertainty["evidence_level"]
    title = {
        "higher_for_this_heuristic": "Stronger heuristic evidence",
        "moderate": "Moderate evidence — review cautions",
        "low": "Low evidence — manual review is important",
    }.get(level, "Evidence limitations")
    body = "\n".join(f"- {note}" for note in uncertainty.get("notes", []))
    if level == "low":
        st.warning(f"**{title}**\n\n{body}")
    elif level == "moderate":
        st.info(f"**{title}**\n\n{body}")
    else:
        st.success(f"**{title}**\n\n{body}")


def _viewpoint_diversity(st: Any, report: Dict[str, Any]) -> None:
    try:
        view = viewpoint_diversity_view(report)
    except Exception:
        st.warning(
            "Experimental viewpoint evidence could not be prepared for display. "
            "The heuristic suitability score above is still complete and unchanged."
        )
        return
    if not isinstance(view, dict):
        st.warning(
            "Experimental viewpoint evidence is unavailable in this report version. "
            "The heuristic suitability score above is still complete and unchanged."
        )
        return
    style = view.get("style") if isinstance(view.get("style"), dict) else {}
    substantial = _display_metric(view.get("substantial_change_percent"))
    parallax = _display_metric(view.get("parallax_proxy_percent"))
    mixed_motion = _display_metric(view.get("mixed_motion_percent"))
    repeated = _display_metric(view.get("repeated_view_count"), suffix="")
    primary_reason = view.get("primary_reason") or "No additional decision detail is available."
    st.markdown(
        f"""
        <div class="viewpoint-panel">
          <div class="eyebrow">Experimental · separate from the suitability score</div>
          <h3>Estimated Viewpoint Diversity <span class="viewpoint-label" style="background:{style.get('color', '#6f7f7c')}">{html.escape(str(view.get('label', 'Unknown')))}</span></h3>
          <p>{html.escape(str(view.get('summary') or 'Viewpoint evidence is not available.'))}</p>
          <p><strong>Decision:</strong> {html.escape(str(primary_reason))}</p>
          <div class="evidence-grid">
            <div class="evidence-item"><span>Substantial long-gap change</span><strong>{substantial}</strong></div>
            <div class="evidence-item"><span>Rigid-consensus parallax proxy</span><strong>{parallax}</strong></div>
            <div class="evidence-item"><span>Mixed static/moving evidence</span><strong>{mixed_motion}</strong></div>
            <div class="evidence-item"><span>Repeated-view pairs</span><strong>{repeated}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(
        "Viewpoint guidance and research limitations",
        expanded=view.get("label") in {"Low", "Unknown"},
    ):
        for recommendation in view.get("recommendations", []):
            st.write(f"- {recommendation}")
        if not view.get("intrinsics_used", False):
            st.caption(
                "No camera intrinsics were supplied. Calibrated relative pose and rotation-compensated parallax remain unavailable."
            )
        if view.get("analysis_mode") == "object_focused":
            background_text = (
                _display_metric(view.get("background_dominated_percent"))
            )
            st.write(
                f"Object tracking coverage: **{_display_metric(view.get('tracking_percent'))}** · "
                f"cycle-verified tracking steps: **{_display_metric(view.get('tracking_cycle_verified_percent'))}** · "
                f"background-dominated long-gap pairs: **{background_text}**"
            )
        for warning in view.get("warnings", []):
            st.caption(f"• {warning}")
        st.caption(
            "This category does not measure exact orbit angle or the percentage of object surface captured."
        )


def _capture_completeness(st: Any, report: Dict[str, Any]) -> None:
    """Render isolated capture guidance without exposing optional failures."""

    try:
        view = capture_completeness_view(report)
    except Exception:
        st.warning(
            "Experimental capture guidance is unavailable. The heuristic, Ridge, "
            "and viewpoint results remain unchanged."
        )
        return
    style = view.get("style") if isinstance(view.get("style"), dict) else {}
    st.markdown(
        f"""
        <div class="viewpoint-panel">
          <div class="eyebrow">Experimental guidance · does not change any score</div>
          <h3>Object Capture Completeness <span class="viewpoint-label" style="background:{style.get('color', '#6f7f7c')}">{html.escape(str(view.get('label', 'Capture coverage not verified')))}</span></h3>
          <p>{html.escape(str(view.get('summary') or 'Capture guidance is unavailable.'))}</p>
          <div class="evidence-grid">
            <div class="evidence-item"><span>Tracked object timeline</span><strong>{_display_metric(view.get('tracking_percent'))}</strong></div>
            <div class="evidence-item"><span>Object view variation</span><strong>{html.escape(str(view.get('view_variation_label', 'Not verified')))}</strong></div>
            <div class="evidence-item"><span>Higher / lower views</span><strong>{html.escape(str(view.get('elevation_label', 'Not verified')))}</strong></div>
            <div class="evidence-item"><span>Capture motion pattern</span><strong>{html.escape(str(view.get('capture_mode_label', 'Uncertain')))}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(
        "Capture guidance and limitations",
        expanded=view.get("status") in {"not_verified", "limited_view_evidence"},
    ):
        st.write(f"**Higher/lower view evidence:** {view.get('elevation_reason')}")
        st.write(f"**Capture-mode evidence:** {view.get('capture_mode_reason')}")
        for suggestion in view.get("suggestions", []):
            st.write(f"- {suggestion}")
        for limitation in view.get("limitations", []):
            st.caption(f"• {limitation}")


def _registration_prediction(st: Any, report: Dict[str, Any]) -> None:
    view = registration_prediction_view(report)
    if view["available"]:
        st.markdown(
            f"""
            <div class="registration-panel">
              <div class="eyebrow">Separate learned estimate · does not change either heuristic result</div>
              <h3>Predicted COLMAP image registration</h3>
              <div class="registration-value">{float(view['percentage']):.1f}%</div>
              <p>{view['interpretation']}</p>
              <p><strong>Artifact:</strong> {view['model_version']} · feature parity verified on {view['parity_case_count']} saved training example(s).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="registration-panel">
              <div class="eyebrow">Live ML prediction unavailable</div>
              <h3>COLMAP registration estimate is not active</h3>
              <p>{html.escape(str(view.get('reason') or 'No compatible model artifact is available.'))}</p>
              <p>The heuristic suitability result remains available and does not depend on this model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _experimental_ridge_prediction(st: Any, report: Dict[str, Any]) -> None:
    prediction = report.get("experimental_ridge_prediction") or {}
    status = prediction.get("status", "not_requested")
    if status == "available":
        percentage = float(prediction["predicted_registration_percentage"])
        clipped_note = (
            " The raw regression output was clipped to the 0–100% display range."
            if prediction.get("prediction_was_clipped")
            else ""
        )
        st.markdown(
            f"""
            <div class="registration-panel">
              <div class="eyebrow">Opt-in research preview · does not change either analyzer</div>
              <h3>Experimental ML Registration Estimate</h3>
              <div class="registration-value">{percentage:.1f}%</div>
              <p>Estimated COLMAP image-registration percentage from the historical Ridge model.{clipped_note}</p>
              <p><strong>Unvalidated live adapter:</strong> uploaded-video preprocessing has not been proven equivalent to the historical CO3D training inputs. No independent held-out evaluation was completed.</p>
              <p>This is not a guarantee of reconstruction quality, object coverage or a successful 3D model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "not_requested":
        st.caption(
            "Experimental ML Registration Estimate was not requested. The heuristic result does not depend on it."
        )
    else:
        st.markdown(
            f"""
            <div class="registration-panel">
              <div class="eyebrow">Experimental ML unavailable</div>
              <h3>Experimental ML Registration Estimate</h3>
              <p>{html.escape(str(prediction.get('reason') or 'The optional estimate could not be produced.'))}</p>
              <p>The heuristic and geometric results remain valid and unchanged.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _timeline(st: Any, rows: List[Dict[str, Any]]) -> None:
    st.subheader("Problem timeline")
    if not rows:
        st.success("No sampled problem interval was detected. Continue to review object coverage manually.")
        return
    types = list(dict.fromkeys(row["problem_type"] for row in rows))
    labels = {row["problem_type"]: row["label"] for row in rows}
    selected = st.multiselect(
        "Filter detected problems", options=types, default=types, format_func=lambda value: labels[value]
    )
    for row in [item for item in rows if item["problem_type"] in selected]:
        width = max(0.6, row["end_percent"] - row["start_percent"])
        st.markdown(
            f"""
            <div class="timeline-row">
              <strong>{row['label']}</strong><span class="timeline-meta"> · {row['time_range']} · {row['severity'].title()}</span>
              <div class="timeline-meta">{row['what']}</div>
              <div class="timeline-track"><span class="timeline-span" style="left:{row['start_percent']}%;width:{width}%"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _frame_gallery(st: Any, frames: List[Dict[str, Any]]) -> None:
    st.subheader("Representative frames to inspect")
    if not frames:
        st.caption("No representative problem frame was available for this report.")
        return
    st.caption(
        "A still can show blur or exposure directly. Motion and overlap labels describe the surrounding frame pair, not the still alone."
    )
    for offset in range(0, len(frames), 3):
        columns = st.columns(3)
        for column, frame in zip(columns, frames[offset : offset + 3]):
            with column:
                st.image(frame["image_rgb"], use_column_width=True)
                st.markdown(f"**{frame['label']}** · {frame['time_range']}")
                st.caption(frame["why"])


def _recommendations(st: Any, rows: List[Dict[str, Any]]) -> None:
    st.subheader("How to improve the next recording")
    cards = recommendation_cards(rows)
    if not cards:
        st.success("Keep a slow orbit, diffuse lighting, sharp focus, and 60–80% overlap.")
        return
    for card in cards:
        st.markdown(
            f"""
            <div class="recommendation">
              <h4>{card['label']} · {card['where']}</h4>
              <p><strong>What:</strong> {card['what']}</p>
              <p><strong>Why it matters:</strong> {card['why']}</p>
              <p><strong>Record differently:</strong> {card['action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _analyze_upload_payload(
    buffer: Any,
    filename: str,
    object_bbox: Any = None,
    registration_artifact_dir: Any = None,
    include_viewpoint_diversity: bool = True,
    include_experimental_ridge: bool = False,
    experimental_ridge_artifact_dir: Any = None,
    experimental_ridge_model_bytes: Any = None,
) -> Dict[str, Any]:
    """Pure upload-to-result path shared by Streamlit and performance benchmarks."""

    upload_wall_started = time.perf_counter()
    upload_cpu_started = time.process_time()
    del registration_artifact_dir  # Retained for compatibility with existing callers/tests.
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video extension: {suffix}")
    if len(buffer) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Upload exceeds the {MAX_UPLOAD_MB} MB deployment limit. "
            "Trim or recompress the video before analysis."
        )
    temporary_path = ""
    try:
        write_wall_started = time.perf_counter()
        write_cpu_started = time.process_time()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(buffer)
            temporary_path = handle.name
        write_wall = time.perf_counter() - write_wall_started
        write_cpu = time.process_time() - write_cpu_started
        report = analyze_video(
            temporary_path,
            object_bbox=object_bbox,
            include_performance_profile=True,
            include_viewpoint_diversity=include_viewpoint_diversity,
        )
        video_digest = hashlib.sha256(buffer).hexdigest()
        tracking = report.get("viewpoint_diversity", {}).get(
            "object_tracking", {}
        )
        report["streamlit_input_provenance"] = {
            "uploaded_filename": filename,
            "video_sha256": video_digest,
            "video_byte_count": len(buffer),
            "selected_object_bbox_normalized": (
                list(object_bbox) if object_bbox is not None else None
            ),
            "selection_preview_policy": OBJECT_SELECTION_PREVIEW_POLICY,
            "selection_source_frame_index": tracking.get(
                "representative_source_frame_index"
            ),
            "analysis_frame_width": tracking.get("analysis_frame_width"),
            "analysis_frame_height": tracking.get("analysis_frame_height"),
            "tracker_version": tracking.get(
                "tracker_version", TRACKER_IMPLEMENTATION_VERSION
            ),
            "analysis_cache_identity": list(_analysis_cache_identity()),
        }
        report["registration_prediction"] = _disabled_registration_prediction()
        report["experimental_ridge_prediction"] = _not_requested_experimental_ridge()
        if include_experimental_ridge:
            try:
                report = attach_experimental_ridge_prediction(
                    report,
                    Path(temporary_path),
                    artifact_directory=experimental_ridge_artifact_dir,
                    model_bytes=experimental_ridge_model_bytes,
                )
            except Exception:
                # Optional research inference must never invalidate the main report.
                report["experimental_ridge_prediction"] = {
                    **_not_requested_experimental_ridge(),
                    "status": "unavailable",
                    "reason": "The optional Ridge inference path failed safely.",
                }
        rows = timeline_rows(report)
        return {
            "filename": filename,
            "report": report,
            "timeline": rows,
            "frames": [],
            "upload_performance": {
                "temporary_file_write_wall_seconds": round(write_wall, 6),
                "temporary_file_write_cpu_seconds": round(write_cpu, 6),
                "upload_to_result_wall_seconds": round(
                    time.perf_counter() - upload_wall_started, 6
                ),
                "upload_to_result_cpu_seconds": round(
                    time.process_time() - upload_cpu_started, 6
                ),
                "diagnostic_rendering_included": False,
            },
        }
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _run_uploaded_analysis(
    st: Any,
    upload: Any,
    object_bbox: Any = None,
    include_viewpoint_diversity: bool = True,
    include_experimental_ridge: bool = False,
    experimental_ridge_model_bytes: Any = None,
) -> Dict[str, Any]:
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    normalized_bbox = tuple(object_bbox) if object_bbox is not None else None
    ridge_identity = _ridge_cache_identity(
        include_experimental_ridge, experimental_ridge_model_bytes
    )
    analysis_identity = _analysis_cache_identity()
    cached = st.session_state.get("video_analysis_cache")
    if (
        cached
        and cached.get("digest") == digest
        and cached.get("object_bbox") == normalized_bbox
        and cached.get("include_viewpoint_diversity")
        == include_viewpoint_diversity
        and cached.get("registration_artifact_identity") == STREAMLIT_ML_IDENTITY
        and cached.get("experimental_ridge_identity") == ridge_identity
        and cached.get("analysis_identity") == analysis_identity
    ):
        return cached
    with st.spinner("Sampling the full timeline and checking quality, overlap and motion…"):
        result = _analyze_upload_payload(
            buffer,
            upload.name,
            object_bbox=object_bbox,
            include_viewpoint_diversity=include_viewpoint_diversity,
            include_experimental_ridge=include_experimental_ridge,
            experimental_ridge_model_bytes=experimental_ridge_model_bytes,
        )
    cached = {
        **result,
        "digest": digest,
        "object_bbox": normalized_bbox,
        "include_viewpoint_diversity": include_viewpoint_diversity,
        "registration_artifact_identity": STREAMLIT_ML_IDENTITY,
        "experimental_ridge_identity": ridge_identity,
        "analysis_identity": analysis_identity,
    }
    st.session_state["video_analysis_cache"] = cached
    return cached


def _representative_upload_frame(st: Any, upload: Any) -> Any:
    """Decode the exact uniform sample used to initialize object tracking."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for object selection") from exc
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    cached = st.session_state.get("object_selection_preview")
    if (
        cached
        and cached.get("digest") == digest
        and cached.get("preview_policy") == OBJECT_SELECTION_PREVIEW_POLICY
    ):
        return cached["image_rgb"]
    temporary_path = ""
    try:
        suffix = Path(upload.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(buffer)
            temporary_path = handle.name
        cap = cv2.VideoCapture(temporary_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not decode a representative frame")
        frame_count = max(1, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        config = AnalysisConfig()
        requested = uniform_sample_indices(frame_count, config.max_samples)
        representative_position = len(requested) // 2
        representative_index = requested[representative_position]
        frame = None
        for frame_index in range(representative_index + 1):
            if not cap.grab():
                break
            if frame_index == representative_index:
                ok, frame = cap.retrieve()
                if not ok:
                    frame = None
        cap.release()
        if frame is None:
            raise RuntimeError("Could not decode the object-selection sample frame")
        source_height, source_width = frame.shape[:2]
        height, width = source_height, source_width
        if max(height, width) > config.max_analysis_side:
            scale = config.max_analysis_side / max(height, width)
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state["object_selection_preview"] = {
            "digest": digest,
            "preview_policy": OBJECT_SELECTION_PREVIEW_POLICY,
            "image_rgb": image_rgb,
            "source_frame_index": representative_index,
            "representative_sample_position": representative_position,
            "requested_sample_count": len(requested),
            "source_width": source_width,
            "source_height": source_height,
            "analysis_width": int(frame.shape[1]),
            "analysis_height": int(frame.shape[0]),
        }
        return image_rgb
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _run_uploaded_diagnostics(st: Any, upload: Any, rows: List[Dict[str, Any]]) -> Any:
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    cached = st.session_state.get("problem_frame_cache")
    if cached and cached.get("digest") == digest:
        return cached["frames"]
    temporary_path = ""
    try:
        suffix = Path(upload.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(buffer)
            temporary_path = handle.name
        started = time.perf_counter()
        frames = extract_problem_frames(
            Path(temporary_path), rows, max_frames=6, max_side=720
        )
        st.session_state["problem_frame_cache"] = {
            "digest": digest,
            "frames": frames,
            "render_wall_seconds": round(time.perf_counter() - started, 6),
        }
        return frames
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _analysis_view(st: Any) -> None:
    _section_heading(
        st,
        "New analysis",
        "Upload an object capture",
        "Use a short MP4 or MOV showing one mostly static object. The standard heuristic is always available; advanced geometry is optional.",
    )
    st.markdown(
        f"""
        <div class="upload-intro">
          <strong>Recommended:</strong> one slow continuous orbit, diffuse lighting, sharp focus and 60–80% overlap. Maximum upload size: {MAX_UPLOAD_MB} MB.
        </div>
        """,
        unsafe_allow_html=True,
    )
    upload = st.file_uploader(
        "Upload or drop a video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg"],
        help=(
            f"For best evidence, record a slow orbit around one mostly static object. "
            f"Cloud uploads are limited to {MAX_UPLOAD_MB} MB."
        ),
    )
    if upload is None:
        left, middle, right = st.columns(3)
        for column, number, title, body in (
            (left, "01", "Record steadily", "Keep the object still and move the camera in a slow lateral orbit."),
            (middle, "02", "Analyze the capture", "Uniform samples check sharpness, lighting, feature continuity and motion."),
            (right, "03", "Retake with confidence", "Use the prioritized warnings and filming recommendations to improve the next pass."),
        ):
            column.markdown(
                f"<div class='step-card'><div class='step-number'>{number}</div><strong>{title}</strong><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
        return
    upload_size_mb = len(upload.getbuffer()) / (1024 * 1024)
    st.caption(f"Selected: {upload.name} · {upload_size_mb:.1f} MB")
    include_viewpoint_diversity = False
    include_experimental_ridge = False
    use_object_box = False
    object_bbox = None
    with st.expander("Advanced viewpoint analysis", expanded=False):
        st.caption(
            "Experimental and separate from the suitability score. It adds long-gap geometric checks and may return Unknown when evidence is insufficient."
        )
        st.caption(
            "Experimental capture-completeness guidance reuses this evidence. Object-specific guidance requires a reliably tracked object region."
        )
        include_viewpoint_diversity = st.checkbox(
            "Run experimental viewpoint-diversity analysis",
            value=False,
            key="advanced_viewpoint_enabled",
            help="Any failure is isolated and cannot invalidate the main heuristic report.",
        )
        if include_viewpoint_diversity:
            use_object_box = st.checkbox(
                "Focus geometry on a selected object region",
                value=False,
                key="advanced_object_region_enabled",
                help="The main score remains whole-frame and unchanged.",
            )
            if use_object_box:
                st.info(
                    "Draw a generous region around the complete object. Include enough margin for its motion, but avoid large background areas. The tracker must follow this region across the full video."
                )
                try:
                    preview = _representative_upload_frame(st, upload)
                except Exception:
                    st.warning(
                        "The representative frame could not be decoded. Continue without object-focused analysis or export the video as MP4/MOV."
                    )
                    preview = None
                preview_column, control_column = st.columns([1.45, 1.0])
                with control_column:
                    horizontal = st.slider(
                        "Horizontal bounds (%)", 0, 100, (25, 75), step=1
                    )
                    vertical = st.slider(
                        "Vertical bounds (%)", 0, 100, (10, 90), step=1
                    )
                    preview_metadata = st.session_state.get(
                        "object_selection_preview", {}
                    )
                    preview_index = preview_metadata.get("source_frame_index")
                    preview_size = (
                        preview_metadata.get("analysis_width"),
                        preview_metadata.get("analysis_height"),
                    )
                    if preview_index is not None and all(preview_size):
                        st.caption(
                            f"This is source frame {preview_index}, resized to "
                            f"{preview_size[0]}×{preview_size[1]}—the exact uniform "
                            "sample used to initialize tracking. Verify that the box "
                            "encloses the complete object, including moving parts."
                        )
                    else:
                        st.caption(
                            "Verify that the box encloses the complete object, including moving parts."
                        )
                if horizontal[1] - horizontal[0] < 5 or vertical[1] - vertical[0] < 5:
                    st.warning("The selected box must cover at least 5% in both dimensions.")
                    return
                object_bbox = [
                    horizontal[0] / 100.0,
                    vertical[0] / 100.0,
                    (horizontal[1] - horizontal[0]) / 100.0,
                    (vertical[1] - vertical[0]) / 100.0,
                ]
                with preview_column:
                    if preview is not None:
                        try:
                            import cv2

                            annotated = preview.copy()
                            height, width = annotated.shape[:2]
                            start = (
                                round(object_bbox[0] * width),
                                round(object_bbox[1] * height),
                            )
                            end = (
                                round((object_bbox[0] + object_bbox[2]) * width),
                                round((object_bbox[1] + object_bbox[3]) * height),
                            )
                            cv2.rectangle(annotated, start, end, (232, 95, 87), 4)
                            st.image(
                                annotated,
                                caption="Exact object-tracker initialization frame and selected region.",
                                use_container_width=True,
                            )
                        except Exception:
                            st.image(preview, use_container_width=True)
    with st.expander("Experimental ML Registration Estimate", expanded=False):
        st.warning(
            "Research/demo mode only. The Ridge model achieved grouped-CV MAE 19.62 percentage points on historical experiments, but it had no independent held-out evaluation. The new uploaded-video adapter is not proven equivalent to the CO3D training-input pipeline."
        )
        include_experimental_ridge = st.checkbox(
            "Run the unvalidated experimental Ridge estimate",
            value=False,
            key="experimental_ridge_enabled",
            help=(
                "Adds separate CPU processing. It cannot change the heuristic score or "
                "experimental viewpoint category and does not run COLMAP."
            ),
        )
        st.caption(
            "The result estimates historical COLMAP image registration, not guaranteed reconstruction quality or surface coverage."
        )
    experimental_ridge_model_bytes = (
        _streamlit_ridge_model_bytes(st) if include_experimental_ridge else None
    )
    ridge_identity = _ridge_cache_identity(
        include_experimental_ridge, experimental_ridge_model_bytes
    )
    analysis_identity = _analysis_cache_identity()
    buffer_digest = hashlib.sha256(upload.getbuffer()).hexdigest()
    bbox_key = tuple(object_bbox) if object_bbox is not None else None
    existing = st.session_state.get("video_analysis_cache")
    matching_cache = bool(
        existing
        and existing.get("digest") == buffer_digest
        and existing.get("object_bbox") == bbox_key
        and existing.get("include_viewpoint_diversity")
        == include_viewpoint_diversity
        and existing.get("registration_artifact_identity")
        == STREAMLIT_ML_IDENTITY
        and existing.get("experimental_ridge_identity") == ridge_identity
        and existing.get("analysis_identity") == analysis_identity
    )
    analyze_clicked = st.button(
        "Analyze video",
        type="primary",
        use_container_width=True,
    )
    if not analyze_clicked and not matching_cache:
        st.info("Your video is ready. Configure optional advanced analysis, then select Analyze video.")
        return
    try:
        result = _run_uploaded_analysis(
            st,
            upload,
            object_bbox=object_bbox,
            include_viewpoint_diversity=include_viewpoint_diversity,
            include_experimental_ridge=include_experimental_ridge,
            experimental_ridge_model_bytes=experimental_ridge_model_bytes,
        )
    except Exception as exc:
        st.error(_friendly_analysis_error(exc))
        return
    report = result["report"]
    overview, problems, guidance, technical = st.tabs(
        ["Overview", "Timeline & frames", "Recording guidance", "Technical report"]
    )
    with overview:
        _score_panel(st, report)
        _components(st, report)
        _attention_summary(st, result["timeline"])
        _section_heading(
            st,
            "Independent geometry",
            "Experimental viewpoint evidence",
            "This category is intentionally separate and cannot change the suitability score.",
        )
        _viewpoint_diversity(st, report)
        _capture_completeness(st, report)
        _experimental_ridge_prediction(st, report)
        _registration_prediction(st, report)
        _uncertainty(st, report)
    with problems:
        _timeline(st, result["timeline"])
        if st.button("Render representative problem frames"):
            with st.spinner("Rendering diagnostic frames separately from analysis…"):
                result["frames"] = _run_uploaded_diagnostics(
                    st, upload, result["timeline"]
                )
        if result.get("frames"):
            _frame_gallery(st, result["frames"])
        else:
            st.caption(
                "Diagnostic frame rendering is deferred so the upload-to-result path stays responsive."
            )
    with guidance:
        _recommendations(st, result["timeline"])
    with technical:
        triggered = [gate for gate in report["critical_failure_gate_evaluations"] if gate["triggered"]]
        if triggered:
            st.warning("Documented physical reliability gates affected the final score.")
            st.dataframe(triggered, use_container_width=True, hide_index=True)
        with st.expander("Method warnings and research limitations", expanded=False):
            for warning in report["warnings"]:
                st.write(f"- {warning}")
        with st.expander("Full JSON report"):
            st.json(report, expanded=False)
        if report.get("performance_profile"):
            with st.expander("CPU performance profile"):
                st.json(
                    {
                        "upload_path": result.get("upload_performance", {}),
                        "analysis_pipeline": report["performance_profile"],
                        "registration_prediction": report.get(
                            "registration_prediction", {}
                        ).get("performance_profile"),
                        "experimental_ridge_prediction": report.get(
                            "experimental_ridge_prediction", {}
                        ).get("performance_profile"),
                        "diagnostic_rendering": st.session_state.get(
                            "problem_frame_cache", {}
                        ).get("render_wall_seconds"),
                    }
                )
        payload = json.dumps(report, indent=2, allow_nan=False)
        st.download_button(
            "Download JSON report",
            payload,
            file_name=f"{Path(result['filename']).stem}_suitability.json",
            mime="application/json",
            use_container_width=True,
        )


def _evaluation_view(st: Any) -> None:
    _section_heading(
        st,
        "Research evidence",
        "Offline experimental ML comparison",
        "Historical experiments are shown for transparency. Live Ridge inference is available only as an explicitly unvalidated opt-in preview.",
    )
    first, second, third = st.columns(3)
    first.metric("Best offline model", "Ridge")
    second.metric("Grouped-CV MAE", "19.62 pp")
    third.metric("Historical records", "647")
    st.markdown(
        """
        <div class="registration-panel">
          <div class="eyebrow">Experimental live adapter available</div>
          <h3>Evidence remains limited</h3>
          <p>No independent held-out evaluation was completed, and uploaded-video preprocessing has not been proven equivalent to the historical CO3D training-input path. Ridge is therefore opt-in research/demo output only; Random Forest remains disabled.</p>
          <p>Predicted image-registration percentage would not guarantee successful 3D reconstruction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Offline grouped-CV results"):
        st.dataframe(
            [
                {"Model": "Ridge", "MAE (pp)": 19.6249, "RMSE (pp)": 27.0212},
                {
                    "Model": "Random Forest",
                    "MAE (pp)": 20.0599,
                    "RMSE (pp)": 27.3581,
                },
                {
                    "Model": "Gradient Boosting",
                    "MAE (pp)": 21.1340,
                    "RMSE (pp)": 29.0513,
                },
                {
                    "Model": "Dummy mean",
                    "MAE (pp)": 34.0572,
                    "RMSE (pp)": 39.4749,
                },
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.divider()
    _section_heading(
        st,
        "Human review",
        "Preserved evaluation evidence",
        "These observations document known failure modes; they are not an accuracy benchmark.",
    )
    st.markdown(
        "<div class='disclaimer'><strong>Not an accuracy benchmark.</strong> These four videos informed the v2 failure-mode audit, only one has a numerical human estimate, and none has completed downstream reconstruction ground truth.</div>",
        unsafe_allow_html=True,
    )
    root = Path(__file__).resolve().parent
    try:
        rows = load_evaluation_rows(
            root / "evaluation/real_video_human_review.csv",
            root / "evaluation/reconstruction/results.csv",
        )
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Local evaluation evidence is unavailable: {exc}")
        return
    first, second, third = st.columns(3)
    first.metric("Reviewed videos", len(rows))
    second.metric("Numerical human scores", sum(row["human_score"] is not None for row in rows))
    third.metric("Reconstruction outcomes", "0 available")
    for row in rows:
        with st.expander(row["filename"], expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline v1", f"{row['baseline_v1_score']:.2f}")
            c2.metric("Preserved v2", f"{row['v2_score']:.2f}" if row["v2_score"] is not None else "Unavailable")
            c3.metric("Human estimate", f"≈{row['human_score']:.0f}" if row["human_score"] is not None else "Not provided")
            feedback = [
                value
                for value in [row["human_assessment"], row["camera_speed_feedback"], row["camera_shake_feedback"]]
                if value
            ]
            st.write("**Human feedback:** " + (" · ".join(feedback) or "No qualitative label"))
            st.caption(row["notes"])
    st.caption(
        "Preserved v2 values are historical evidence. Current revision 2.1 adds a synthetic-regression fix for pervasive severe blur; it was not fitted to these four scores."
    )


def _method_view(st: Any) -> None:
    st.markdown("### What this MVP measures")
    st.write(
        "Uniform whole-timeline samples feed visual quality, exposure, ORB/RANSAC geometry, optical-flow retention, overlap and camera-motion evidence. A separate multi-gap module estimates viewpoint diversity without changing either score path."
    )
    c1, c2, c3 = st.columns(3)
    c1.success("**Local & CPU-runnable**\n\nNo video is sent to a paid API by this app.")
    c2.info("**Explainable**\n\nComponents, raw evidence, gates and timestamps stay visible.")
    c3.warning("**Not reconstruction**\n\nImage-space evidence cannot prove completeness or true camera translation.")
    st.markdown("### A strong capture usually has")
    st.write("- a slow, steady lateral orbit around a mostly static object;")
    st.write("- sharp focus, diffuse lighting and controlled reflections;")
    st.write("- 60–80% overlap between neighboring views;")
    st.write("- several elevations and complete coverage of all required sides.")
    st.caption("COLMAP validation remains a separate workflow and is not presented as an app prediction.")


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install the app extra with `python -m pip install -e '.[app-ml]'`."
        ) from exc
    st.set_page_config(
        page_title="3D Capture Check", page_icon="◈", layout="wide", initial_sidebar_state="collapsed"
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    _hero(st)
    analysis, evaluation, method = st.tabs(["Analyze a video", "Evaluation evidence", "How it works"])
    with analysis:
        _analysis_view(st)
    with evaluation:
        _evaluation_view(st)
    with method:
        _method_view(st)


if __name__ == "__main__":
    main()
