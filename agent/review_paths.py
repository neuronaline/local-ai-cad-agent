"""Path scheme for per-model review artifacts.

Centralises the ``<project>/.cad-agent/reviews/<model_sha256>/`` layout so
callers no longer reconstruct the path inline. Tests, the screenshot tool,
the review tool, the build tool, and the Flask endpoints all delegate
here; a single change to the layout now requires editing one function
instead of grep-and-replace.
"""
from __future__ import annotations

from pathlib import Path


def review_dir(project_dir: Path, model_sha256: str) -> Path:
    """Return the per-model review directory.

    ``model_sha256`` is the content-addressed digest of ``model.py``. The
    directory is content-addressed too: a different model always lands
    under a different leaf and never overwrites a prior review.
    """
    return project_dir / ".cad-agent" / "reviews" / model_sha256
