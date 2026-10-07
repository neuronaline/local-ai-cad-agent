"""Stable JSON envelopes for model-facing tool results."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from agent.images import as_chat_image
from agent.prompt import TOOL_HINTS
from agent.review_paths import review_dir
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
    """Compact verbose tool envelopes for LLM context while keeping essential metrics."""
    if tool not in {"cad_build", "cad_build_and_verify"}:
        return result
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return result
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return result
    data = payload.get("data")
    if not isinstance(data, dict):
        return result

    metrics = data.get("metrics") or {}
    compact_data: dict[str, Any] = {
        "is_valid": metrics.get("is_valid", False),
        "solid_count": metrics.get("solid_count", 1),
        "dimensions_mm": metrics.get("dimensions_mm", {}),
        "volume_mm3": metrics.get("volume_mm3", 0.0),
        "summary": data.get("summary", ""),
    }
    features = data.get("feature_summary")
    if isinstance(features, dict) and features:
        compact_data["feature_summary"] = features
    verif = data.get("verification")
    if isinstance(verif, dict) and verif:
        compact_data["verification"] = {
            "status": verif.get("status"),
            "risk_score": verif.get("risk_score"),
            "risk_level": verif.get("risk_level"),
            "findings": verif.get("findings", []),
            "watertight": (verif.get("mesh_integrity") or {}).get("watertight"),
        }
    raw_images = data.get("images")
    if isinstance(raw_images, list) and raw_images:
        compact_data["images"] = [
            {"view": img.get("view"), "cropped": bool(img.get("cropped"))}
            for img in raw_images
            if isinstance(img, dict) and "view" in img
        ]

    return json.dumps({"ok": True, "tool": tool, "data": compact_data}, ensure_ascii=False)


def build_view_image_multimodal_content(
    raw_result: str,
    project_dir: Path,
    *,
    context_result: str | None = None,
) -> dict[str, Any] | None:
    """Build a multimodal tool-result for a successful ``get_view_images``.

    Returns a dict shaped like an OpenAI Chat Completions content-part list::

        {"content": [{"type": "text", "text": "<json envelope>"},
                     *[as_chat_image(p) for p in image_paths]],
         "image_paths": [host-relative paths]}

    Returns ``None`` when the envelope is malformed, the success flag is
    missing, the ``images`` list is empty, or every on-disk artifact
    fails its manifest hash check. ``relocate_tool_images`` (see
    :mod:`agent.llm_base`) handles the tool->user image handoff so no extra
    wiring is needed for the OpenAI / Gemini vision-less fallback path.
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
        review_dir(project_dir, review_dir_name)
        if isinstance(review_dir_name, str) and re.fullmatch(r"[0-9a-fA-F]{64}", review_dir_name)
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


def _classify(tool: str, error: Exception, message: str) -> tuple[str, str, bool, str]:
    lower = message.lower()
    if isinstance(error, json.JSONDecodeError):
        return (
            "INVALID_TOOL_ARGUMENTS",
            "arguments",
            True,
            TOOL_HINTS["INVALID_TOOL_ARGUMENTS"],
        )
    if isinstance(error, RevisionIntegrityError):
        return (
            "REVISION_INTEGRITY_ERROR",
            "persistence",
            False,
            TOOL_HINTS["REVISION_INTEGRITY"],
        )
    if "timed out" in lower or "timeout" in lower:
        return "TIMEOUT", "execution", True, TOOL_HINTS["TIMEOUT"]
    if tool in {"cad_build", "cad_build_and_verify"}:
        code = (
            "MODEL_MISSING"
            if "model.scad does not exist" in lower
            else "CAD_BUILD_FAILED"
        )
        hint = (
            TOOL_HINTS["MODEL_MISSING"].format(filename="model.scad")
            if code == "MODEL_MISSING"
            else TOOL_HINTS["CAD_BUILD_FAILED"].format(filename="model.scad")
        )
        return code, "build", True, hint
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return (
            "VALIDATION_ERROR",
            "validation",
            True,
            TOOL_HINTS["VALIDATION_ERROR"],
        )
    if isinstance(error, FileNotFoundError):
        return (
            "FILE_NOT_FOUND",
            "execution",
            True,
            TOOL_HINTS["FILE_NOT_FOUND"],
        )
    return (
        "TOOL_EXECUTION_FAILED",
        "execution",
        True,
        TOOL_HINTS["TOOL_EXECUTION_FAILED"],
    )
