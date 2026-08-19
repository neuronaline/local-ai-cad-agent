"""Stable JSON envelopes for model-facing tool results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.images import as_chat_image
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


def compact_for_context(tool: str, result: str) -> str:
    """Return a context-bounded copy of ``result`` for LLM history.

    The returned string keeps the same envelope shape so the model can still
    parse ``ok``/``tool``/``data``, but drops fields that are repeated across
    every turn (which otherwise inflate the prompt and crowd out new turns).

    Only the in-conversation copy is compacted: the raw result still flows to
    the UI via ``tool_status`` events and to ``.cad-agent/`` artifacts on
    disk, so operators and reviewers see the full payload.
    """
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result
    if not isinstance(payload, dict):
        return result
    if tool == "cad_build_and_verify" and payload.get("ok") is True:
        data = payload.get("data")
        if isinstance(data, dict):
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                fs = metrics.get("feature_summary")
                if isinstance(fs, dict):
                    # Keep summary fields (counts); drop the per-feature
                    # cylinder table — the agent only needs totals to decide
                    # whether to rebuild.
                    metrics["feature_summary"] = _summarize_feature_summary(fs)
            data.pop("feature_summary", None)
            # ``preview`` and ``render`` are file-path markers used by the UI
            # SSE events and the on-disk artifact paths. The agent never
            # reads them back; dropping them shrinks the prompt without
            # information loss.
            data.pop("preview", None)
            data.pop("render", None)
            rm = data.get("review_manifest")
            if isinstance(rm, dict):
                data["review_manifest"] = _compacted_review_manifest(rm)
        return json.dumps(payload, ensure_ascii=False)
    return result


def build_cad_build_multimodal_content(
    text_result: str, project_dir: Path
) -> dict[str, Any] | None:
    """Build a multimodal tool-result payload for a successful ``cad_build_and_verify``.

    Returns a dict shaped like an OpenAI Chat Completions content-part list::

        {"content": [{"type": "text", "text": "<compacted json>"},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}],
         "image_paths": [<host-relative paths>]}

    Returns ``None`` when the build did not produce a render (no point attaching
    images), when the JSON envelope is malformed, or when on-disk artifacts are
    missing. The caller is responsible for recording the host image paths so
    the prose-conversation log can redact them on subsequent loads.
    """
    try:
        payload = json.loads(text_result)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    render_rel = data.get("render")
    if not isinstance(render_rel, str) or not render_rel:
        return None
    preferred = [
        project_dir / "render.png",
        project_dir / "preview.stl",
    ]
    image_paths: list[Path] = []
    for candidate in preferred:
        if candidate.is_file() and candidate.stat().st_size > 0:
            image_paths.append(candidate)
    if not image_paths:
        return None
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": text_result},
    ]
    for image_path in image_paths:
        try:
            parts.append(as_chat_image(image_path))
        except OSError:
            continue
    if len(parts) == 1:
        # No image attachment survived the read; degrade to plain text.
        return None
    return {"content": parts, "image_paths": image_paths}


def _summarize_feature_summary(fs: dict[str, Any]) -> dict[str, Any]:
    """Strip per-feature tables from a feature summary dict."""
    counts: dict[str, Any] = {}
    for key in (
        "disconnected_solid_count",
        "through_hole_count",
        "blind_hole_count",
        "fillet_count",
        "chamfer_count",
    ):
        if key in fs:
            counts[key] = fs[key]
    return counts


def _compacted_review_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reduce the review manifest to the few fields the agent needs.

    The agent never consumes image bytes — it sees renders via the UI — so
    we keep the model/preview/view hashes plus the timing fields and drop
    everything else. ``_review_manifest`` SHA coverage remains unchanged:
    the on-disk artifact under ``.cad-agent/reviews/<sha>/`` is still the
    full manifest consumed by :mod:`agent.cad_review`.
    """
    compacted: dict[str, Any] = {
        "model_sha256": manifest.get("model_sha256"),
        "preview_sha256": manifest.get("preview_sha256"),
        "view_count": manifest.get("view_count"),
        "views": [
            {
                "view_id": v.get("view_id"),
                "image_sha256": v.get("image_sha256"),
            }
            for v in manifest.get("views", [])
            if isinstance(v, dict)
        ],
        "contact_sheet_sha256": (
            (manifest.get("contact_sheet") or {}).get("image_sha256")
            if isinstance(manifest.get("contact_sheet"), dict)
            else None
        ),
    }
    return {k: v for k, v in compacted.items() if v is not None}


def _classify(tool: str, error: Exception, message: str) -> tuple[str, str, bool, str]:
    lower = message.lower()
    if isinstance(error, json.JSONDecodeError):
        return (
            "INVALID_TOOL_ARGUMENTS",
            "arguments",
            True,
            "Send one valid JSON object matching the tool schema.",
        )
    if isinstance(error, RevisionIntegrityError):
        return (
            "REVISION_INTEGRITY_ERROR",
            "persistence",
            False,
            "Do not retry the same edit; revision history needs user attention.",
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
