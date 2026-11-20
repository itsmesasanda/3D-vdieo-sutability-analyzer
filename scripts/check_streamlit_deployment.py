#!/usr/bin/env python3
"""Fast, network-free preflight for Streamlit Community Cloud deployment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    packages = (ROOT / "packages.txt").read_text(encoding="utf-8").splitlines()

    require((ROOT / "app.py").is_file(), "app.py is missing")
    require("-e .[app-ml]" in requirements, "requirements.txt must install app ML dependencies")
    require("maxUploadSize = 150" in config, "150 MB upload limit is not configured")
    require("ffmpeg" in packages, "packages.txt must request ffmpeg")
    require(importlib.util.find_spec("streamlit") is not None, "streamlit is not installed")
    require(importlib.util.find_spec("cv2") is not None, "headless OpenCV is not installed")
    require(importlib.util.find_spec("sklearn") is not None, "scikit-learn is not installed")
    require(importlib.util.find_spec("joblib") is not None, "joblib is not installed")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import app

    prediction = app._disabled_registration_prediction()
    require(prediction["status"] == "unavailable", "Random Forest must remain disabled")
    require(prediction["trained_model_used"] is False, "Random Forest policy is inconsistent")
    ridge = app._not_requested_experimental_ridge()
    require(ridge["status"] == "not_requested", "Ridge default status changed")
    require("disabled" in ridge["reason"].lower(), "Live numeric Ridge must remain disabled")
    require(ridge["affects_suitability_score"] is False, "Ridge must stay separate")
    print("PASS: entry point, dependencies, upload policy, ffmpeg, and disabled live ML policies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
