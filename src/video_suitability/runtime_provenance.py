"""Small, read-only runtime identity helpers for reproducible diagnostics."""

from __future__ import annotations

import hashlib
import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _repository_commit(repository_root: Path) -> Optional[str]:
    """Resolve a deployment-provided or local Git commit without running Git."""

    for name in (
        "STREAMLIT_GIT_COMMIT",
        "GIT_COMMIT",
        "COMMIT_SHA",
        "SOURCE_COMMIT",
        "RENDER_GIT_COMMIT",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    head_path = repository_root / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        reference = head[5:]
        loose_ref = repository_root / ".git" / reference
        if loose_ref.exists():
            return loose_ref.read_text(encoding="utf-8").strip() or None
        packed_refs = repository_root / ".git" / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, ref = line.split(" ", 1)
                    if ref == reference:
                        return commit
    except (OSError, ValueError):
        return None
    return None


@lru_cache(maxsize=1)
def runtime_provenance() -> Dict[str, Any]:
    """Return stable code/runtime identity fields safe to include in a report."""

    import cv2

    package_dir = Path(__file__).resolve().parent
    repository_root = package_dir.parents[1]
    return {
        "package_version": __version__,
        "git_commit": _repository_commit(repository_root),
        "git_commit_available": _repository_commit(repository_root) is not None,
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "object_tracker_source_sha256": _sha256(package_dir / "object_focus.py"),
        "inference_source_sha256": _sha256(package_dir / "infer.py"),
    }
