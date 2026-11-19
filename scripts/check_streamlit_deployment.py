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
    require("-e .[ui]" in requirements, "requirements.txt must install the UI package")
    require("maxUploadSize = 150" in config, "150 MB upload limit is not configured")
    require("ffmpeg" in packages, "packages.txt must request ffmpeg")
    require(importlib.util.find_spec("streamlit") is not None, "streamlit is not installed")
    require(importlib.util.find_spec("cv2") is not None, "headless OpenCV is not installed")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import app

    prediction = app._disabled_registration_prediction()
    require(prediction["status"] == "unavailable", "live ML is not disabled")
    require(prediction["trained_model_used"] is False, "live ML policy is inconsistent")
    print("PASS: entry point, dependencies, upload policy, ffmpeg, and disabled ML policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
