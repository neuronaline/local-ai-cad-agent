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


def build_cad_screenshot_multimodal_content(
    raw_result: str,
    project_dir: Path,
    *,
    context_result: str | None = None,
    max_images: int = 4,
) -> dict[str, Any] | None:
    """Multimodal payload for a successful ``cad_screenshot`` call.

    The orchestrator publishes ``inline_images`` in the tool data; this helper
    reads them and attaches the requested views (plus the contact sheet when
    present) directly to the tool message so the model can inspect the
    rendered output in-band instead of chasing file paths.

    Returns ``None`` when no inline image is available, when the JSON envelope
    is malformed, or when every attached image fails to decode.
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
    inline = data.get("inline_images")
    if not isinstance(inline, list) or not inline:
        return None
    if max_images <= 0:
        return None
    image_paths: list[Path] = []
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": context_result or compact_for_context("cad_screenshot", raw_result),
        }
    ]
    # The orchestrator publishes the requested views first then the contact
    # sheet; ``max_images`` would otherwise drop the contact sheet for a
    # default call (eight views + contact sheet → only four views attached).
    # Prioritise the contact sheet so the canonical compact evidence is
    # always inline, then fill the remaining budget with views in the
    # orchestrator's declared order.
    inline_paths: list[Path] = []
    inline_view_ids: list[str] = []
    for entry in inline:
        if not isinstance(entry, dict):
            continue
        rel = entry.get("path")
        view_id = entry.get("view_id") or ""
        expected_sha = entry.get("sha256")
        if not isinstance(rel, str) or not rel:
            continue
        candidate = project_dir / rel
        if not _is_png(candidate, expected_sha if isinstance(expected_sha, str) else None):
            continue
        inline_paths.append(candidate)
        inline_view_ids.append(view_id)
    contact_sheet_idx = next(
        (idx for idx, view_id in enumerate(inline_view_ids) if view_id == "contact_sheet"),
        None,
    )
    selected: list[tuple[Path, str]] = []
    if contact_sheet_idx is not None:
        selected.append((inline_paths[contact_sheet_idx], inline_view_ids[contact_sheet_idx]))
    remaining_budget = max(0, max_images - len(selected))
    for path, view_id in zip(inline_paths, inline_view_ids):
        if view_id == "contact_sheet":
            continue
        if remaining_budget <= 0:
            break
        selected.append((path, view_id))
        remaining_budget -= 1
    for candidate, _view_id in selected:
        try:
            parts.append(as_chat_image(candidate))
            image_paths.append(candidate)
        except OSError as error:
            _LOG.debug(
                "screenshot multimodal failed for %s: %s",
                candidate,
                error,
                exc_info=True,
            )
            continue
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


def _metrics_render_hash(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest = payload.get("review_manifest") if isinstance(payload, dict) else None
    entry = manifest.get("single_render") if isinstance(manifest, dict) else None
    value = entry.get("image_sha256") if isinstance(entry, dict) else None
    return value if isinstance(value, str) and len(value) == 64 else None


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
