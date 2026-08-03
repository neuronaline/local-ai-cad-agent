"""Stable JSON envelopes for model-facing tool results."""

from __future__ import annotations

import json
from typing import Any

from agent.constraints import ConstraintError
from agent.revisions import RevisionIntegrityError


def success(tool: str, data: Any) -> str:
    return json.dumps({"ok": True, "tool": tool, "data": data}, ensure_ascii=False)


def failure(tool: str, error: Exception) -> str:
    message = str(error) or type(error).__name__
    code, phase, retryable, hint = _classify(tool, error, message)
    detail: dict[str, Any] = {
        "code": code,
        "phase": phase,
        "message": message,
        "retryable": retryable,
    }
    if hint:
        detail["hint"] = hint
    return json.dumps({"ok": False, "tool": tool, "error": detail}, ensure_ascii=False)


def is_failure(value: str) -> bool:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value.startswith("ERROR:")
    return isinstance(payload, dict) and payload.get("ok") is False


def _classify(tool: str, error: Exception, message: str) -> tuple[str, str, bool, str]:
    lower = message.lower()
    if isinstance(error, json.JSONDecodeError):
        return (
            "INVALID_TOOL_ARGUMENTS",
            "arguments",
            True,
            "Send one valid JSON object matching the tool schema.",
        )
    if isinstance(error, ConstraintError):
        return (
            "CONSTRAINT_VIOLATION",
            "validation",
            True,
            "Preserve every protected parameter and feature named in the message.",
        )
    if isinstance(error, RevisionIntegrityError):
        return (
            "REVISION_INTEGRITY_ERROR",
            "persistence",
            False,
            "Do not retry the same edit; revision history needs user attention.",
        )
    if isinstance(error, AttributeError):
        return (
            "UNKNOWN_TOOL",
            "arguments",
            False,
            "Use one of the tool names provided in the current tool schema.",
        )
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT", "execution", True, "Simplify the operation before retrying."
    if tool == "cad_build_and_verify":
        code = (
            "MODEL_MISSING"
            if "model.py does not exist" in lower
            else "CAD_BUILD_FAILED"
        )
        hint = (
            "Create model.py first."
            if code == "MODEL_MISSING"
            else "Fix model.py using the reported location and cause, then rebuild."
        )
        return code, "build", True, hint
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return (
            "VALIDATION_ERROR",
            "validation",
            True,
            "Correct the arguments or source named in the message.",
        )
    if isinstance(error, FileNotFoundError):
        return (
            "FILE_NOT_FOUND",
            "execution",
            True,
            "Create the required project file first.",
        )
    return (
        "TOOL_EXECUTION_FAILED",
        "execution",
        True,
        "Use the message to correct the request before retrying.",
    )
