"""Deterministic verifiers for parsed :class:`DesignSpec` requirements.

Phase 2 of the CAD quality plan. The registry maps requirement kinds to a
small callable that returns a :class:`ValidationResult` while the build123d
shape is still available inside the sandboxed runner. Evidence is persisted
next to ``metrics.json`` for the API/tests, but is **not** returned to the
LLM-facing tool envelope.
"""

from __future__ import annotations

from agent.quality.verifiers.registry import (
    registered_kinds,
    register,
    run_verifiers,
)

__all__ = ["register", "registered_kinds", "run_verifiers"]
