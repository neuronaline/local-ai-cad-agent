"""Stable JSON envelopes for model-facing tool results."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from agent.images import as_chat_image
from agent.review_paths import review_dir as review_dir_path
from agent.revisions import RevisionIntegrityError

_LOG = logging.getLogger(__name__)


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
    """Return ``result`` as-is without lossy compaction.

    Preserves all metrics, feature summaries, review details, and artifact
    paths so the model's memory is 100% complete, uncompressed, and intact
    across all turns.
    """
    return result


def build_cad_build_multimodal_content(
    raw_result: str,
    project_dir: Path,
    *,
    context_result: str | None = None,
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
        payload = json.loads(raw_result)
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
    # Prefer the canonical contact sheet because it contains every rendered
    # view. Fall back to the single isometric PNG. Never pass preview.stl to
    # ``as_chat_image``: an STL byte stream labelled ``image/png`` is invalid.
    candidates: list[tuple[Path, str | None]] = []
    review_dir = data.get("review")
    if isinstance(review_dir, str) and review_dir and Path(review_dir).name == review_dir:
        review_root = review_dir_path(project_dir, review_dir)
        candidates.append(
            (
                review_root / "review-sheet.png",
                _manifest_hash(review_root / "manifest.json", "contact_sheet"),
            )
        )
    render_path = Path(render_rel)
    if render_path.name == render_rel:
        candidates.append(
            (
                project_dir / render_path,
                _metrics_render_hash(project_dir / ".cad_metrics.json"),
            )
        )
    image_path = next(
        (path for path, expected in candidates if _is_png(path, expected)), None
    )
    if image_path is None:
        return None
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": context_result or compact_for_context("cad_build_and_verify", raw_result),
        },
    ]
    try:
        parts.append(as_chat_image(image_path))
    except OSError as error:
        # ``as_chat_image`` reads the file; an ``OSError`` here is almost
        # always a transient I/O race (e.g. another process truncating
        # the file mid-build). Fall back to text-only and log at DEBUG
        # so debug sessions can correlate with the activity log.
        _LOG.debug(
            "multimodal content failed for %s: %s", image_path, error, exc_info=True
        )
        return None
    return {"content": parts, "image_paths": [image_path]}


def build_view_image_multimodal_content(
    raw_result: str,
    project_dir: Path,
    *,
    context_result: str | None = None,
) -> dict[str, Any] | None:
    """Build a multimodal tool-result for a successful ``get_view_images``.

    Returns the same shape as :func:`build_cad_build_multimodal_content`::

        {"content": [{"type": "text", "text": "<json envelope>"},
                     *[as_chat_image(p) for p in image_paths]],
         "image_paths": [host-relative paths]}

    Returns ``None`` when the envelope is malformed, the success flag is
    missing, the ``images`` list is empty, or every on-disk artifact
    fails its manifest hash check. ``relocate_tool_images`` (see
    :mod:`agent.llm_base`) reuses the same tool->user image handoff as
    ``cad_build_and_verify`` so no extra wiring is needed for the
    OpenAI / Gemini vision-less fallback path.
    """
    try:
        payload = json.loads(raw_result)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    raw_items = data.get("images")
    if not isinstance(raw_items, list) or not raw_items:
        return None

    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": context_result or compact_for_context("get_view_images", raw_result),
        }
    ]
    image_paths: list[Path] = []
    review_dir_name = data.get("review_sha256")
    review_root = (
        project_dir / ".cad-agent" / "reviews" / review_dir_name
        if isinstance(review_dir_name, str) and review_dir_name
        else None
    )
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("path")
        expected_sha = item.get("sha256")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            continue
        if review_root is None:
            continue
        # Refuse paths that try to escape the review directory.
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            continue
        abs_path = review_root / rel_path
        if not _is_png(abs_path, expected_sha):
            continue
        try:
            parts.append(as_chat_image(abs_path))
        except OSError:
            _LOG.debug(
                "view image multimodal content failed for %s", abs_path, exc_info=True
            )
            continue
        image_paths.append(abs_path)

    if not image_paths:
        return None
    return {"content": parts, "image_paths": image_paths}


def _is_png(path: Path, expected_sha256: str | None = None) -> bool:
    """Fully decode PNG evidence and verify its manifest hash when available."""
    try:
        raw = path.read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
            return False
        with Image.open(path, formats=["PNG"]) as image:
            image.verify()
            return image.format == "PNG" and image.width > 0 and image.height > 0
    except (OSError, UnidentifiedImageError, SyntaxError):
        return False


def _manifest_hash(path: Path, artifact: str) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = payload.get(artifact) if isinstance(payload, dict) else None
    value = entry.get("image_sha256") if isinstance(entry, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _view_hash(manifest: dict, view_id: str) -> str | None:
    """Return ``image_sha256`` for ``view_id`` from a review manifest dict."""
    entries = manifest.get("views") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("view_id") == view_id:
            value = entry.get("image_sha256")
            if isinstance(value, str) and len(value) == 64:
                return value
            return None
    return None


def _metrics_render_hash(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest = payload.get("review_manifest") if isinstance(payload, dict) else None
    entry = manifest.get("single_render") if isinstance(manifest, dict) else None
    value = entry.get("image_sha256") if isinstance(entry, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


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
            if "model.scad does not exist" in lower
            else "CAD_BUILD_FAILED"
        )
        hint = (
            "Create model.scad first."
            if code == "MODEL_MISSING"
            else "Fix model.scad using the reported location and cause, then rebuild."
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
