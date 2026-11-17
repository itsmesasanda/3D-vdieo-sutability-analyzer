"""Polished Streamlit competition MVP for the stable heuristic JSON contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from video_suitability.infer import analyze_video
from video_suitability.presentation import (
    component_rows,
    extract_problem_frames,
    load_evaluation_rows,
    recommendation_cards,
    score_style,
    timeline_rows,
)


APP_CSS = """
<style>
  .stApp { background: linear-gradient(150deg, #f6f8fb 0%, #eef3f2 55%, #f8f4ee 100%); }
  .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
  [data-testid="stHeader"] { background: transparent; }
  .mvp-hero { padding: 1.65rem 1.8rem; border: 1px solid rgba(20,55,50,.12); border-radius: 24px; background: linear-gradient(120deg,#123d38 0%,#1f6158 62%,#c77b3d 150%); color:white; box-shadow:0 18px 50px rgba(23,61,55,.16); margin-bottom:1.2rem; }
  .mvp-kicker { font-size:.73rem; letter-spacing:.14em; text-transform:uppercase; opacity:.72; font-weight:700; }
  .mvp-hero h1 { color:white; margin:.35rem 0 .5rem; font-size:clamp(2rem,4vw,3.2rem); line-height:1.02; }
  .mvp-hero p { color:rgba(255,255,255,.82); max-width:760px; margin:0; font-size:1rem; }
  .method-chip { display:inline-block; margin-top:1rem; padding:.35rem .7rem; border-radius:99px; background:rgba(255,255,255,.12); font-size:.76rem; }
  .score-panel { display:flex; align-items:center; gap:1.4rem; padding:1.3rem; border-radius:22px; background:white; border:1px solid #dce6e3; box-shadow:0 10px 30px rgba(33,54,50,.07); }
  .score-ring { width:138px; height:138px; border-radius:50%; display:grid; place-items:center; position:relative; flex:0 0 auto; }
  .score-ring:before { content:""; position:absolute; inset:11px; background:white; border-radius:50%; }
  .score-number { position:relative; z-index:2; font-size:2.35rem; font-weight:800; color:#173d38; line-height:.9; }
  .score-number small { display:block; font-size:.75rem; color:#6f7f7c; font-weight:600; text-align:center; margin-top:.35rem; }
  .score-copy h2 { margin:0 0 .25rem; color:#173d38; }
  .score-copy p { margin:.15rem 0; color:#60706d; }
  .component-card { min-height:112px; padding:.9rem; border-radius:16px; background:white; border:1px solid #e0e8e6; margin-bottom:.5rem; }
  .component-card .label { font-size:.78rem; color:#657672; min-height:2.2rem; }
  .component-card .value { font-size:1.55rem; font-weight:750; color:#173d38; }
  .timeline-row { position:relative; padding:.8rem .9rem .75rem; background:white; border:1px solid #e0e8e6; border-radius:14px; margin:.45rem 0; overflow:hidden; }
  .timeline-row strong { color:#173d38; }
  .timeline-meta { color:#71807d; font-size:.8rem; }
  .timeline-track { height:6px; background:#edf1f0; border-radius:9px; margin-top:.55rem; position:relative; overflow:hidden; }
  .timeline-span { position:absolute; height:100%; min-width:5px; border-radius:9px; background:linear-gradient(90deg,#e6a23c,#df5b5b); }
  .recommendation { border-left:4px solid #d88c38; padding:.8rem 1rem; background:#fff; border-radius:0 14px 14px 0; margin:.55rem 0; }
  .recommendation h4 { margin:0 0 .35rem; color:#173d38; }
  .recommendation p { margin:.2rem 0; color:#53625f; font-size:.9rem; }
  .eyebrow { color:#3f766f; font-weight:750; letter-spacing:.08em; text-transform:uppercase; font-size:.72rem; }
  .disclaimer { padding:.8rem 1rem; background:#fff8e8; border:1px solid #ecd8a6; border-radius:13px; color:#6c5725; }
  div[data-testid="stFileUploader"] { background:rgba(255,255,255,.72); border-radius:18px; padding:.35rem .7rem; border:1px dashed #99b2ad; }
</style>
"""


def _hero(st: Any) -> None:
    st.markdown(
        """
        <section class="mvp-hero">
          <div class="mvp-kicker">Capture intelligence · CPU heuristic</div>
          <h1>Will this video reconstruct well?</h1>
          <p>Inspect sharpness, lighting, view overlap and camera motion before spending time on a 3D reconstruction run.</p>
          <span class="method-chip">No remote upload · no trained model · no reconstruction claim</span>
        </section>
        """,
        unsafe_allow_html=True,
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
    st.subheader("Five signals behind the score")
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


def _run_uploaded_analysis(st: Any, upload: Any) -> Dict[str, Any]:
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    cached = st.session_state.get("video_analysis_cache")
    if cached and cached.get("digest") == digest:
        return cached
    suffix = Path(upload.name).suffix or ".mp4"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(buffer)
            temporary_path = handle.name
        with st.spinner("Sampling the full timeline and checking quality, overlap and motion…"):
            report = analyze_video(temporary_path)
            rows = timeline_rows(report)
            frames = extract_problem_frames(Path(temporary_path), rows, max_frames=6, max_side=720)
        cached = {
            "digest": digest,
            "filename": upload.name,
            "report": report,
            "timeline": rows,
            "frames": frames,
        }
        st.session_state["video_analysis_cache"] = cached
        return cached
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _analysis_view(st: Any) -> None:
    upload = st.file_uploader(
        "Drop in an object-orbit video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg"],
        help="For best evidence, record a slow orbit around one mostly static object.",
    )
    if upload is None:
        left, middle, right = st.columns(3)
        left.info("**1 · Record**\n\nOne static object, diffuse light, slow continuous orbit.")
        middle.info("**2 · Analyze**\n\nUniform samples cover the complete declared timeline.")
        right.info("**3 · Improve**\n\nInspect exact intervals and retake what needs fixing.")
        return
    try:
        result = _run_uploaded_analysis(st, upload)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return
    report = result["report"]
    overview, problems, guidance, technical = st.tabs(
        ["Overview", "Timeline & frames", "Recording guidance", "Technical report"]
    )
    with overview:
        _score_panel(st, report)
        st.write("")
        _components(st, report)
        _uncertainty(st, report)
    with problems:
        _timeline(st, result["timeline"])
        _frame_gallery(st, result["frames"])
    with guidance:
        _recommendations(st, result["timeline"])
    with technical:
        triggered = [gate for gate in report["critical_failure_gate_evaluations"] if gate["triggered"]]
        if triggered:
            st.warning("Documented physical reliability gates affected the final score.")
            st.dataframe(triggered, use_container_width=True, hide_index=True)
        st.markdown("**Warnings**")
        for warning in report["warnings"]:
            st.write(f"- {warning}")
        with st.expander("Full JSON report"):
            st.json(report, expanded=False)
        payload = json.dumps(report, indent=2, allow_nan=False)
        st.download_button(
            "Download JSON report",
            payload,
            file_name=f"{Path(result['filename']).stem}_suitability.json",
            mime="application/json",
            use_container_width=True,
        )


def _evaluation_view(st: Any) -> None:
    st.markdown("### Preserved human-feedback evaluation")
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
        "Uniform whole-timeline samples feed visual quality, exposure, ORB/RANSAC geometry, optical-flow retention, overlap and camera-motion evidence."
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
        raise RuntimeError("Install the UI extra with `python -m pip install -e '.[ui]'`.") from exc
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
