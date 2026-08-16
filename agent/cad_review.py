"""Structured multimodal review of a CAD build.

The reviewer receives the active user request, a single isometric render, the
multi-view contact sheet, the feature summary, and the latest ``model.py``
source. It must respond via the structured ``submit_review`` tool so we never
have to parse free-form JSON.

The contract:

* ``status`` — one of ``"pass" | "fail" | "inconclusive"``.
* ``findings`` — bounded list of issues with severity, category, view id,
  message, and optional ``repair_hint``.
* ``summary`` — one-line headline suitable for a status pill.

Review-specific LLM calls disable image stripping (``require_images=True``).
Missing/incompatible images, malformed output, or hash mismatches produce
``inconclusive`` instead of ``pass``. The base64 image payload is never
persisted; only the bounded review result is written to the project tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from agent.images import as_chat_image


def _default_create_client(settings: Any) -> Any:
    """Resolve the configured LLM client factory.

    Importing inside the helper avoids a circular import between
    ``agent.core`` and this module, while still letting tests patch
    ``agent.core.create_llm_client`` and have it take effect here.
    """
    from agent.core import create_llm_client

    return create_llm_client(settings)

MAX_FINDINGS = 12
MAX_MESSAGE_CHARS = 240
MAX_HINT_CHARS = 240
MAX_SUMMARY_CHARS = 160

ALLOWED_STATUSES = ("pass", "fail", "inconclusive")
ALLOWED_SEVERITIES = ("blocking", "major", "minor")
ALLOWED_CATEGORIES = (
    "missing_feature",
    "dimensions",
    "alignment",
    "geometry",
    "manufacturability",
)


REVIEW_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": (
            "Submit the structured review verdict for the rendered CAD model. "
            "Call exactly once with the bounded Finding list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": list(ALLOWED_STATUSES),
                    "description": "Overall verdict for the active revision.",
                },
                "summary": {
                    "type": "string",
                    "maxLength": MAX_SUMMARY_CHARS,
                    "description": "One-line verdict headline.",
                },
                "findings": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": MAX_FINDINGS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {
                                "type": "string",
                                "enum": list(ALLOWED_SEVERITIES),
                            },
                            "category": {
                                "type": "string",
                                "enum": list(ALLOWED_CATEGORIES),
                            },
                            "view": {
                                "type": "string",
                                "minLength": 1,
                                "description": (
                                    "View id from the manifest that best shows the issue "
                                    "(e.g. ``x_positive``, ``isometric_positive``)."
                                ),
                            },
                            "message": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_MESSAGE_CHARS,
                            },
                            "repair_hint": {
                                "type": "string",
                                "maxLength": MAX_HINT_CHARS,
                            },
                        },
                        "required": ["severity", "category", "message"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["status", "summary", "findings"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Finding:
    severity: str
    category: str
    message: str
    view: str | None = None
    repair_hint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
        }
        if self.view:
            payload["view"] = self.view
        if self.repair_hint:
            payload["repair_hint"] = self.repair_hint
        return payload


@dataclass(frozen=True)
class ReviewResult:
    status: str
    summary: str
    findings: list[Finding] = field(default_factory=list)
    model_sha256: str | None = None
    preview_sha256: str | None = None

    @property
    def is_pass(self) -> bool:
        return self.status == "pass"

    @property
    def is_blocking(self) -> bool:
        return any(finding.severity == "blocking" for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "summary": self.summary,
            "findings": [finding.as_dict() for finding in self.findings],
        }
        if self.model_sha256:
            payload["model_sha256"] = self.model_sha256
        if self.preview_sha256:
            payload["preview_sha256"] = self.preview_sha256
        return payload


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _coerce_finding(raw: Any, view_ids: Iterable[str]) -> Finding | None:
    if not isinstance(raw, dict):
        return None
    severity = raw.get("severity")
    category = raw.get("category")
    message = raw.get("message")
    if severity not in ALLOWED_SEVERITIES:
        return None
    if category not in ALLOWED_CATEGORIES:
        return None
    if not isinstance(message, str) or not message.strip():
        return None
    view_id = raw.get("view")
    if not isinstance(view_id, str):
        view_id = None
    elif view_id and view_id not in view_ids:
        view_id = None
    hint = raw.get("repair_hint")
    return Finding(
        severity=severity,
        category=category,
        message=_truncate(message, MAX_MESSAGE_CHARS),
        view=view_id,
        repair_hint=_truncate(hint, MAX_HINT_CHARS) if isinstance(hint, str) else None,
    )


def parse_review_response(
    raw_arguments: Any,
    *,
    model_sha256: str,
    preview_sha256: str,
    allowed_view_ids: Iterable[str] = (),
) -> ReviewResult:
    """Parse a structured ``submit_review`` tool-call payload into a result.

    Any malformed input collapses to ``inconclusive`` so callers never have to
    re-validate before raising; the verification pipeline uses ``status`` only.
    """
    view_ids = set(allowed_view_ids)
    try:
        payload = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        return ReviewResult(
            status="inconclusive",
            summary="Review model produced malformed output.",
            model_sha256=model_sha256,
            preview_sha256=preview_sha256,
        )
    status = payload.get("status")
    if status not in ALLOWED_STATUSES:
        return ReviewResult(
            status="inconclusive",
            summary="Review model produced an unknown status.",
            model_sha256=model_sha256,
            preview_sha256=preview_sha256,
        )
    summary = _truncate(payload.get("summary"), MAX_SUMMARY_CHARS) or (
        "All checks passed." if status == "pass" else "Review failed."
    )
    findings: list[Finding] = []
    raw_findings = payload.get("findings")
    if isinstance(raw_findings, list):
        for entry in raw_findings[:MAX_FINDINGS]:
            coerced = _coerce_finding(entry, view_ids)
            if coerced is not None:
                findings.append(coerced)
    # ``pass`` with any blocking finding is incoherent: reclassify.
    if status == "pass" and any(f.severity == "blocking" for f in findings):
        status = "fail"
        if not summary or summary == "All checks passed.":
            summary = "Blocking finding reported."
    return ReviewResult(
        status=status,
        summary=summary,
        findings=findings,
        model_sha256=model_sha256,
        preview_sha256=preview_sha256,
    )


def _extract_tool_call(response: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if name != "submit_review":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None
    return None


def _inconclusive(reason: str, model_sha: str, preview_sha: str) -> ReviewResult:
    return ReviewResult(
        status="inconclusive",
        summary=reason,
        model_sha256=model_sha,
        preview_sha256=preview_sha,
    )


def _verify_image_artifact(
    path: Path,
    artifact: Any,
    label: str,
) -> str | None:
    """Return an evidence error, or ``None`` when a hashed image is usable."""
    if not path.is_file() or path.stat().st_size == 0:
        return f"Review {label} is missing or empty."
    if not isinstance(artifact, dict):
        return f"Review manifest is missing {label} metadata."
    expected_sha = artifact.get("image_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return f"Review manifest has no valid {label} hash."
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        return f"Review {label} hash does not match the manifest."
    return None


def run_review(
    *,
    settings: Any,
    request_text: str,
    model_source: str,
    metrics: dict[str, Any],
    feature_summary: dict[str, Any],
    review_manifest: dict[str, Any],
    sheet_path: Path,
    single_render_path: Path,
    stop_event: Any = None,
    stop_instruction: str | None = None,
    create_client: Any = None,
) -> ReviewResult:
    """Invoke the structured reviewer and return a parsed ``ReviewResult``.

    The LLM is forced to use the ``submit_review`` tool. Missing tool calls,
    rejected image inputs, and parse failures collapse to ``inconclusive``.
    ``create_client`` defaults to ``agent.core.create_llm_client`` so test
    monkeypatches that target the core module take effect here.
    """
    model_sha = str(review_manifest.get("model_sha256") or "")
    preview_sha = str(review_manifest.get("preview_sha256") or "")
    views = review_manifest.get("views") if isinstance(review_manifest, dict) else None
    view_ids: list[str] = []
    if isinstance(views, list):
        for entry in views:
            if isinstance(entry, dict) and isinstance(entry.get("view_id"), str):
                view_ids.append(entry["view_id"])
    sheet_error = _verify_image_artifact(
        sheet_path, review_manifest.get("contact_sheet"), "contact sheet"
    )
    if sheet_error:
        return _inconclusive(sheet_error, model_sha, preview_sha)
    render_error = _verify_image_artifact(
        single_render_path, review_manifest.get("single_render"), "single render"
    )
    if render_error:
        return _inconclusive(render_error, model_sha, preview_sha)
    if not view_ids:
        return _inconclusive(
            "Review manifest has no rendered views.", model_sha, preview_sha
        )
    factory = create_client or _default_create_client
    client = factory(settings)
    if stop_event is not None:
        client.stop_event = stop_event
    client.require_images = True
    prompt = _build_prompt(
        request_text=request_text,
        model_source=model_source,
        metrics=metrics,
        feature_summary=feature_summary,
        review_manifest=review_manifest,
        stop_instruction=stop_instruction,
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "text",
                    "text": "Visual evidence 1 of 2: single isometric render.",
                },
                as_chat_image(single_render_path),
                {
                    "type": "text",
                    "text": "Visual evidence 2 of 2: multi-view contact sheet.",
                },
                as_chat_image(sheet_path),
            ],
        }
    ]
    try:
        response = client.chat(messages, [REVIEW_TOOL_SCHEMA])
    except Exception as error:  # noqa: BLE001 - surface as inconclusive review
        return _inconclusive(
            f"Review model call failed: {type(error).__name__}.",
            model_sha,
            preview_sha,
        )
    arguments = _extract_tool_call(response)
    if arguments is None:
        return _inconclusive(
            "Review model did not call submit_review.", model_sha, preview_sha
        )
    return parse_review_response(
        arguments,
        model_sha256=model_sha,
        preview_sha256=preview_sha,
        allowed_view_ids=view_ids,
    )


def _build_prompt(
    *,
    request_text: str,
    model_source: str,
    metrics: dict[str, Any],
    feature_summary: dict[str, Any],
    review_manifest: dict[str, Any],
    stop_instruction: str | None,
) -> str:
    manifest_lines = [
        f"- view_id: {entry['view_id']} ({entry.get('label', '')})"
        for entry in review_manifest.get("views", [])
        if isinstance(entry, dict)
    ]
    contact = review_manifest.get("contact_sheet") or {}
    view_order = contact.get("view_order") if isinstance(contact, dict) else None
    sections = [
        "You are reviewing a CAD build against the user's request.",
        "",
        "Use the ``submit_review`` tool. Do not respond with prose.",
        "",
        "## User request",
        (request_text or "(empty)").strip(),
        "",
        "## Geometry metrics",
        json.dumps(metrics or {}, indent=2, ensure_ascii=False),
        "",
        "## Feature summary",
        json.dumps(feature_summary or {}, indent=2, ensure_ascii=False),
        "",
        "## Available views",
        "\n".join(manifest_lines) or "(none)",
        "",
        "## Contact sheet",
        (
            f"path: views/{contact.get('path', 'review-sheet.png')}\n"
            f"size: {contact.get('width', '?')}×{contact.get('height', '?')}\n"
            f"image_sha256: {contact.get('image_sha256', '?')}"
        ),
        "The contact-sheet tiles are labelled and ordered left-to-right, top-to-bottom: "
        + (", ".join(view_order) if isinstance(view_order, list) else "see tile labels"),
        "",
        "## Single isometric render",
        (
            f"path: {review_manifest.get('single_render', {}).get('path', 'render.png')}\n"
            f"image_sha256: {review_manifest.get('single_render', {}).get('image_sha256', '?')}"
        ),
        "",
        "## Model source",
        "```python",
        (model_source or "").strip(),
        "```",
        "",
        "## Verdict policy",
        (
            "- ``status: pass`` only when every visible feature in the sheet "
            "and the single render matches the request and no blocking finding is reported.\n"
            "- Cross-check the single render against the multi-view sheet; use the "
            "labelled contact-sheet tile when assigning a finding's ``view``.\n"
            "- ``status: fail`` when at least one blocking finding is reported.\n"
            "- ``status: inconclusive`` when visual evidence is missing or "
            "ambiguous.\n"
            "- Each finding's `category` must be one of the documented enum "
            "values; ``view`` is the view id that best shows the issue.\n"
            "- Quote observations in ``message``; put the concrete fix in "
            "``repair_hint``."
        ),
    ]
    if stop_instruction:
        sections.extend(["", "## Stop", stop_instruction.strip()])
    return "\n".join(sections)


def write_review_result(review_dir: Path, result: ReviewResult) -> Path:
    """Persist the bounded review verdict into ``review_dir/result.json``."""
    target = review_dir / "result.json"
    payload = asdict(result)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target
