"""Pure presentation helpers for the Streamlit competition MVP.

These functions consume the stable JSON report and never alter analysis metrics or
scores. Keeping view-model construction outside Streamlit makes UI behavior testable
without installing the optional UI dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SUBSCORE_PRESENTATION = {
    "visual_quality": (
        "Image sharpness",
        "Whether frames preserve crisp local detail instead of blur or weak contrast.",
    ),
    "exposure_and_lighting": (
        "Lighting & exposure",
        "Whether surface detail remains visible without crushed shadows or clipped highlights.",
    ),
    "feature_trackability": (
        "Feature tracking",
        "Whether distinctive points can be followed reliably between adjacent views.",
    ),
    "geometric_consistency": (
        "View overlap",
        "Whether matched features agree geometrically across the sampled sequence.",
    ),
    "viewpoint_motion": (
        "Camera path",
        "Whether motion is steady, depth-bearing, and broad enough for multi-view reconstruction.",
    ),
}


PROBLEM_GUIDANCE: Dict[str, Dict[str, str]] = {
    "blur": {
        "label": "Blurred frame",
        "what": "Fine edges and surface texture are smeared.",
        "why": "Reconstruction needs repeatable local details to match the same surface across views.",
        "action": "Move more slowly, stabilize the phone, add light, and lock focus on the object.",
    },
    "underexposure": {
        "label": "Too dark",
        "what": "Large image regions lose detail in shadow.",
        "why": "Feature detectors cannot recover texture that was clipped to near-black.",
        "action": "Add broad diffuse light or raise exposure while keeping highlights controlled.",
    },
    "overexposure": {
        "label": "Too bright",
        "what": "Highlights are clipped and surface detail is missing.",
        "why": "Uniform white regions provide few stable correspondences between views.",
        "action": "Reduce exposure and soften direct reflections with larger diffuse lights.",
    },
    "low_contrast": {
        "label": "Low contrast",
        "what": "Object edges and texture have weak tonal separation.",
        "why": "Weak gradients make repeatable feature localization less reliable.",
        "action": "Use a contrasting background and soft directional light to reveal shape.",
    },
    "low_texture": {
        "label": "Low surface texture",
        "what": "The sampled view contains few distinctive local patterns.",
        "why": "Smooth, reflective, or transparent areas are difficult to match consistently.",
        "action": "Improve side lighting or add removable non-repeating markers nearby when appropriate.",
    },
    "low_overlap": {
        "label": "Low view overlap",
        "what": "Too little of the previous view remains visible in the next sampled view.",
        "why": "Feature tracks break before they can triangulate stable 3D points.",
        "action": "Take smaller camera steps and keep roughly 60–80% of the previous view visible.",
    },
    "low_track_retention": {
        "label": "Tracks are being lost",
        "what": "Optical-flow points do not survive between nearby frames.",
        "why": "Short tracks weaken camera-pose estimation and multi-view triangulation.",
        "action": "Slow down, keep the object centered, and preserve more overlap.",
    },
    "irregular_camera_motion": {
        "label": "Camera shake",
        "what": "Direction or speed changes abruptly between nearby views.",
        "why": "Shake increases blur and makes correspondence continuity less predictable.",
        "action": "Use two hands or stabilization and follow one smooth, continuous orbit.",
    },
    "excessive_camera_speed": {
        "label": "Camera moving too fast",
        "what": "Timestamp-normalized image motion is unusually high.",
        "why": "Fast motion causes blur and moves features too far for reliable tracking.",
        "action": "Slow the orbit and pause briefly if autofocus or exposure needs to settle.",
    },
    "excessive_viewpoint_jump": {
        "label": "Abrupt viewpoint jump",
        "what": "Adjacent sampled views are separated by a large image-space jump.",
        "why": "Large jumps reduce shared content and can break geometric initialization.",
        "action": "Record a continuous path with smaller angular steps around the object.",
    },
    "insufficient_correspondence": {
        "label": "Too few reliable matches",
        "what": "The pair has too few geometrically verified feature correspondences.",
        "why": "Camera poses cannot be estimated reliably without enough consistent matches.",
        "action": "Keep the object larger, sharper, and well lit; include distinctive context.",
    },
    "little_viewpoint_change": {
        "label": "Viewpoint barely changes",
        "what": "The camera remains almost stationary between views.",
        "why": "Triangulation needs a physical baseline, not repeated images from one position.",
        "action": "Move laterally around the static object instead of rotating in place.",
    },
    "homography_dominant_motion": {
        "label": "Depth coverage uncertain",
        "what": "Most motion can be explained by a single planar transform.",
        "why": "This can indicate rotation-only capture or a mostly planar view with weak parallax.",
        "action": "Include lateral translation and views around depth-bearing sides of the object.",
    },
    "ambiguous_repetitive_features": {
        "label": "Repetitive texture",
        "what": "Many local descriptors look similar to one another.",
        "why": "Repeated patterns can match the wrong surface locations and corrupt geometry.",
        "action": "Include distinctive context or temporary non-repeating markers when acceptable.",
    },
    "motion_sampling_gap_too_wide": {
        "label": "High-frequency shake unresolved",
        "what": "Uniform analysis samples are too far apart for short-gap shake measurement.",
        "why": "The score can assess ORB continuity, but cannot confirm frame-to-frame steadiness here.",
        "action": "Review the original interval manually; record a steady path at a consistent speed.",
    },
}


PRIORITY = {
    "blur": 0,
    "underexposure": 1,
    "overexposure": 1,
    "low_overlap": 2,
    "low_track_retention": 3,
    "irregular_camera_motion": 4,
    "excessive_camera_speed": 5,
    "excessive_viewpoint_jump": 5,
}


def score_style(score: float) -> Dict[str, str]:
    if score >= 70:
        return {"label": "Likely suitable", "color": "#16a674", "tone": "positive"}
    if score >= 45:
        return {"label": "Review recommended", "color": "#e49a22", "tone": "warning"}
    return {"label": "Unlikely suitable", "color": "#df5b5b", "tone": "negative"}


def component_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    details = report.get("subscore_details", {})
    rows = []
    for key, score in report["subscores"].items():
        label, explanation = SUBSCORE_PRESENTATION.get(
            key, (key.replace("_", " ").title(), "Measured heuristic component.")
        )
        weight = details.get(key, {}).get("weight")
        rows.append(
            {
                "key": key,
                "label": label,
                "score": float(score),
                "weight_percent": round(float(weight) * 100.0) if weight is not None else None,
                "explanation": explanation,
                "style": score_style(float(score)),
            }
        )
    return rows


def _time_range(start: float, end: float) -> str:
    if abs(end - start) < 0.005:
        return f"{start:.2f}s"
    return f"{start:.2f}–{end:.2f}s"


def timeline_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    duration = report.get("raw_metrics", {}).get("video", {}).get("duration_seconds")
    duration_value = float(duration) if duration else 0.0
    rows: List[Dict[str, Any]] = []
    for segment in report.get("problem_segments", []):
        problem = segment["problem_type"]
        guidance = PROBLEM_GUIDANCE.get(
            problem,
            {
                "label": problem.replace("_", " ").title(),
                "what": "The heuristic detected an unusual condition.",
                "why": "This evidence may reduce reconstruction reliability.",
                "action": "Inspect this interval and repeat it with a slow, stable, well-lit orbit.",
            },
        )
        start = float(segment["start_time_seconds"])
        end = float(segment["end_time_seconds"])
        start_index = int(segment["start_frame_index"])
        end_index = int(segment["end_frame_index"])
        representative_index = int(round((start_index + end_index) / 2.0))
        representative_time = (start + end) / 2.0
        rows.append(
            {
                **segment,
                **guidance,
                "time_range": _time_range(start, end),
                "representative_frame_index": representative_index,
                "representative_time_seconds": round(representative_time, 6),
                "start_percent": round(100.0 * start / duration_value, 3)
                if duration_value > 0
                else 0.0,
                "end_percent": round(100.0 * end / duration_value, 3)
                if duration_value > 0
                else 0.0,
            }
        )
    return rows


def representative_problem_rows(
    rows: Sequence[Dict[str, Any]], limit: int = 8
) -> List[Dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            PRIORITY.get(row["problem_type"], 20),
            0 if row.get("severity") == "high" else 1,
            -(float(row["end_time_seconds"]) - float(row["start_time_seconds"])),
            float(row["start_time_seconds"]),
        ),
    )
    selected: List[Dict[str, Any]] = []
    used_indices = set()
    used_types = set()
    for row in ordered:
        index = row["representative_frame_index"]
        problem = row["problem_type"]
        if index in used_indices or problem in used_types:
            continue
        selected.append(row)
        used_indices.add(index)
        used_types.add(problem)
        if len(selected) >= limit:
            break
    return selected


def extract_problem_frames(
    video_path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    max_frames: int = 8,
    max_side: int = 720,
) -> List[Dict[str, Any]]:
    """Read representative source frames for UI inspection without modifying input."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for representative-frame inspection") from exc

    selected = representative_problem_rows(rows, limit=max_frames)
    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video for frame inspection: {video_path}")
    output: List[Dict[str, Any]] = []
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(row["representative_frame_index"]))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        height, width = frame.shape[:2]
        if max(height, width) > max_side:
            scale = max_side / float(max(height, width))
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        output.append(
            {
                "problem_type": row["problem_type"],
                "label": row["label"],
                "time_range": row["time_range"],
                "timestamp_seconds": row["representative_time_seconds"],
                "frame_index": row["representative_frame_index"],
                "why": row["why"],
                "action": row["action"],
                "image_rgb": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            }
        )
    cap.release()
    return output


def recommendation_cards(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        problem = row["problem_type"]
        if problem not in grouped:
            grouped[problem] = {
                "problem_type": problem,
                "label": row["label"],
                "what": row["what"],
                "why": row["why"],
                "action": row["action"],
                "severity": row["severity"],
                "time_ranges": [],
            }
        grouped[problem]["time_ranges"].append(row["time_range"])
        if row["severity"] == "high":
            grouped[problem]["severity"] = "high"
    cards = list(grouped.values())
    for card in cards:
        card["where"] = ", ".join(card.pop("time_ranges")[:5])
    return sorted(
        cards,
        key=lambda card: (
            PRIORITY.get(card["problem_type"], 20),
            0 if card["severity"] == "high" else 1,
        ),
    )


def load_evaluation_rows(
    human_review_path: Path, results_path: Path
) -> List[Dict[str, Any]]:
    with Path(human_review_path).open(newline="", encoding="utf-8") as handle:
        reviews = list(csv.DictReader(handle))
    with Path(results_path).open(newline="", encoding="utf-8") as handle:
        results = {row["filename"]: row for row in csv.DictReader(handle)}
    rows: List[Dict[str, Any]] = []
    for review in reviews:
        result = results.get(review["filename"], {})
        raw_human_score = review.get("human_score_numeric", "").strip()
        human_score: Optional[float] = float(raw_human_score) if raw_human_score else None
        rows.append(
            {
                "filename": review["filename"],
                "baseline_v1_score": float(review["baseline_score"]),
                "v2_score": float(result["v2_score"]) if result.get("v2_score") else None,
                "human_score": human_score,
                "human_score_status": review["human_score_numeric_status"],
                "human_assessment": review["human_qualitative_assessment"].replace("_", " "),
                "camera_speed_feedback": review["camera_speed_feedback"].replace("_", " "),
                "camera_shake_feedback": review["camera_shake_feedback"].replace("_", " "),
                "notes": review["review_notes"],
            }
        )
    return rows

