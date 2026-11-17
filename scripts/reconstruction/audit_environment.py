#!/usr/bin/env python3
"""Record a read-only host audit for the sparse-reconstruction experiment."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def command_output(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or "").strip()
    return output or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    disk = shutil.disk_usage(root)
    colmap = shutil.which("colmap")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    brew = shutil.which("brew")
    python_executable = shutil.which("python3")
    try:
        import cv2

        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = None
    pycolmap_status = command_output(
        [python_executable, "-c", "import pycolmap; print(pycolmap.__version__)"]
    ) if python_executable else None
    report = {
        "schema_version": "1.0",
        "working_directory": str(root),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": __import__("os").cpu_count(),
        "memory_bytes": None,
        "gpu": {
            "cuda_available": False,
            "description": "Apple integrated GPU is not used by the CPU COLMAP workflow",
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_gib": round(disk.free / 1024**3, 3),
        },
        "tools": {
            "colmap": colmap,
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "homebrew": brew,
            "pycolmap_import": pycolmap_status,
            "opencv_version": opencv_version,
        },
        "tool_versions": {
            "colmap": command_output([colmap, "-h"]) if colmap else None,
            "ffmpeg": command_output([ffmpeg, "-version"]) if ffmpeg else None,
            "homebrew": command_output([brew, "--version"]) if brew else None,
        },
        "local_colmap_feasible": False if not colmap else disk.free >= 5 * 1024**3,
        "limitation": (
            "COLMAP is not installed and free disk is below the conservative 5 GiB "
            "minimum for installing/building dependencies plus experiment artifacts."
            if not colmap and disk.free < 5 * 1024**3
            else None
        ),
    }
    # macOS-specific read-only details; unavailable commands remain explicit nulls.
    report["host_details"] = {
        "hardware_model": command_output(["sysctl", "-n", "hw.model"]),
        "physical_cpu_count": command_output(["sysctl", "-n", "hw.physicalcpu"]),
        "memory_bytes": command_output(["sysctl", "-n", "hw.memsize"]),
        "system_profiler_hardware": command_output(
            ["system_profiler", "SPHardwareDataType", "-detailLevel", "mini"]
        ),
        "system_profiler_graphics": command_output(
            ["system_profiler", "SPDisplaysDataType", "-detailLevel", "mini"]
        ),
    }
    if report["host_details"]["memory_bytes"]:
        report["memory_bytes"] = int(report["host_details"]["memory_bytes"])
    elif report["host_details"]["system_profiler_hardware"]:
        memory_match = re.search(
            r"Memory:\s*([0-9.]+)\s*(GB|MB)",
            report["host_details"]["system_profiler_hardware"],
            flags=re.IGNORECASE,
        )
        if memory_match:
            multiplier = 1024**3 if memory_match.group(2).upper() == "GB" else 1024**2
            report["memory_bytes"] = int(float(memory_match.group(1)) * multiplier)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

