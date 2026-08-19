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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.images import as_chat_image
from agent.prompt import get_system_prompt


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
    source: str = "visual"  # "deterministic" or "visual"

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
        if self.source:
            payload["source"] = self.source
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
    if not isinstance(view_id, str) or view_id and view_id not in view_ids:
        view_id = None
    hint = raw.get("repair_hint")
    return Finding(
        severity=severity,
        category=category,
        message=_truncate(message, MAX_MESSAGE_CHARS),
        view=view_id,
        repair_hint=_truncate(hint, MAX_HINT_CHARS) if isinstance(hint, str) else None,
        source="visual",  # findings produced by the LLM reviewer are always "visual"
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
    # Behavior is always strict: a blocking or major finding reclassifies a
    # model-reported pass to fail. ``parse_review_response`` only sees the
    # visual layer's tool call; the deterministic layer adds findings with
    # ``source="deterministic"`` directly. ``ReviewResult.status`` is the
    # final verdict assembled by ``review_cad`` once both layers have run.
    if status == "pass" and any(
        f.severity in {"blocking", "major"} for f in findings
    ):
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


def _visual_review(
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
    """Visual-only layer.

    Pulls the view IDs from the manifest, verifies the contact-sheet and
    single render hashes against the manifest, builds the multimodal
    message, invokes the LLM, and parses the structured ``submit_review``
    tool-call. The visual layer is skipped (returns ``inconclusive``) when:

    - any image fails its hash check, or
    - the manifest has no rendered views, or
    - ``settings`` is ``None`` (so the caller — usually a test with no
      LLM configured — only wants the deterministic layer).

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
    if settings is None:
        return _inconclusive(
            "Review skipped: no LLM settings were provided.", model_sha, preview_sha
        )
    factory = create_client or _default_create_client
    client = factory(settings)
    if stop_event is not None:
        client.stop_event = stop_event
    # Tag the subordinate call so provider-side telemetry distinguishes the
    # reviewer from the parent agent loop. The shared base system prompt keeps
    # the prompt prefix cache key stable across both call sites.
    client.agent_role = "reviewer"
    client.require_images = True
    prompt = _build_prompt(
        request_text=request_text,
        model_source=model_source,
        metrics=metrics,
        review_manifest=review_manifest,
        stop_instruction=stop_instruction,
    )
    # Reuse the parent agent's system prompt so the provider-side prefix cache
    # key stays stable across both call sites. The base prompt is shared with
    # the parent agent loop; the review-specific instructions are folded into
    # the user message below.
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": get_system_prompt()
            + "\n\n<role>\nYou are the visual reviewer. Focus on the geometry evidence; produce a structured verdict via submit_review.\n</role>",
        },
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
        },
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


# ---------------------------------------------------------------------------
# Deterministic layer (Python-only checks; no LLM, no network).
# ---------------------------------------------------------------------------


def _deterministic_review(
    *,
    metrics: dict[str, Any] | None,
    feature_summary: dict[str, Any] | None,
    validation_results: list[dict[str, Any]] | None = None,
) -> list[Finding]:
    """Run the deterministic verification suite and return bounded findings.

    These checks always run — they are cheap, parallel-safe, and source
    the canvas the visual layer later confirms. The findings carry
    ``source="deterministic"`` so the agent can tell where a result came
    from when assembling the final verdict.
    """
    findings: list[Finding] = []
    metrics = metrics if isinstance(metrics, dict) else {}
    feature_summary = feature_summary if isinstance(feature_summary, dict) else {}

    # Geometry sanity — mirror ``cad_tool._enforce_basic_geometry`` so a
    # deterministic finding fires before the LLM sees the image.
    try:
        solid_count = int(metrics.get("solid_count", 0) or 0)
        is_valid = bool(metrics.get("is_valid"))
        dimensions = metrics.get("dimensions_mm") or {}
        volume = float(metrics.get("volume_mm3", 0.0) or 0.0)
    except (TypeError, ValueError):
        solid_count = 0
        is_valid = False
        dimensions = {}
        volume = 0.0
        findings.append(
            Finding(
                severity="major",
                category="dimensions",
                message="CAD geometry metrics are malformed.",
                source="deterministic",
            )
        )
    if solid_count < 1:
        findings.append(
            Finding(
                severity="blocking",
                category="geometry",
                message="Build did not produce any solids.",
                source="deterministic",
            )
        )
    if not is_valid:
        findings.append(
            Finding(
                severity="blocking",
                category="geometry",
                message="CAD geometry is reported as invalid by the runner.",
                source="deterministic",
            )
        )
    for axis in ("x", "y", "z"):
        try:
            value = float(dimensions.get(axis, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            findings.append(
                Finding(
                    severity="major",
                    category="dimensions",
                    message=f"CAD dimension {axis} is non-positive ({value} mm).",
                    source="deterministic",
                )
            )
    if volume <= 0:
        findings.append(
            Finding(
                severity="major",
                category="geometry",
                message="CAD volume is non-positive.",
                source="deterministic",
            )
        )

    # Spec verifiers — propagate the structured ``.cad_validation.json`` so
    # any spec-driven requirement failures become findings.
    for entry in validation_results or []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        severity = entry.get("severity")
        if status == "pass":
            continue
        if severity not in {"blocking", "major", "minor"}:
            severity = "major" if status == "fail" else "minor"
        requirement_id = str(entry.get("requirement_id") or "") or "spec"
        verifier = str(entry.get("verifier") or "spec")
        message = str(entry.get("message") or f"{verifier} reported {status}.")
        findings.append(
            Finding(
                severity=severity,
                category="manufacturability"
                if verifier.startswith("manufactur")
                else "dimensions",
                message=f"[{requirement_id}] {message}"[:MAX_MESSAGE_CHARS],
                source="deterministic",
            )
        )

    return findings


def review_cad(
    *,
    settings: Any,
    request_text: str,
    model_source: str,
    metrics: dict[str, Any],
    feature_summary: dict[str, Any],
    review_manifest: dict[str, Any] | None,
    sheet_path: Path | None,
    single_render_path: Path | None,
    validation_results: list[dict[str, Any]] | None = None,
    stop_event: Any = None,
    stop_instruction: str | None = None,
    create_client: Any = None,
) -> ReviewResult:
    """Compose deterministic + visual findings into the final verdict.

    Deterministic checks always run. The visual layer is skipped only when
    no visual evidence is available (manifest/paths missing or empty); in
    that case the verdict collapses to ``inconclusive``. Verdict rules are
    always strict: any ``blocking`` or ``major`` finding forces ``fail``;
    only ``minor`` (or no) findings permit ``pass``.
    """
    model_sha = (
        str(review_manifest.get("model_sha256") or "")
        if isinstance(review_manifest, dict)
        else ""
    )
    preview_sha = (
        str(review_manifest.get("preview_sha256") or "")
        if isinstance(review_manifest, dict)
        else ""
    )
    deterministic_findings = _deterministic_review(
        metrics=metrics if isinstance(metrics, dict) else None,
        feature_summary=feature_summary if isinstance(feature_summary, dict) else None,
        validation_results=validation_results,
    )
    # Short-circuit: a blocking deterministic finding means the visual
    # layer cannot rescue the verdict.
    if any(f.severity == "blocking" for f in deterministic_findings):
        return ReviewResult(
            status="fail",
            summary="Deterministic verification reported blocking failures.",
            findings=deterministic_findings,
            model_sha256=model_sha,
            preview_sha256=preview_sha,
        )
    visual: ReviewResult | None = None
    if (
        isinstance(review_manifest, dict)
        and sheet_path is not None
        and single_render_path is not None
        and sheet_path.is_file()
        and single_render_path.is_file()
    ):
        visual = _visual_review(
            settings=settings,
            request_text=request_text,
            model_source=model_source,
            metrics=metrics if isinstance(metrics, dict) else {},
            feature_summary=feature_summary if isinstance(feature_summary, dict) else {},
            review_manifest=review_manifest,
            sheet_path=sheet_path,
            single_render_path=single_render_path,
            stop_event=stop_event,
            stop_instruction=stop_instruction,
            create_client=create_client,
        )
    combined: list[Finding] = list(deterministic_findings)
    visual_findings: list[Finding] = []
    visual_status = "skipped"
    if visual is not None:
        visual_findings = list(visual.findings)
        combined.extend(visual_findings)
        visual_status = visual.status
    if not combined and visual_status == "skipped":
        return ReviewResult(
            status="inconclusive",
            summary="No visual evidence available for review.",
            findings=[],
            model_sha256=model_sha,
            preview_sha256=preview_sha,
        )
    # Strict verdict: any blocking or major forces fail.
    has_blocking = any(f.severity == "blocking" for f in combined)
    has_major = any(f.severity == "major" for f in combined)
    if has_blocking or has_major:
        status = "fail"
    elif visual is None:
        status = "inconclusive"
    else:
        status = visual.status if visual.status in {"pass", "fail", "inconclusive"} else "inconclusive"
    summary = visual.summary if visual is not None and visual.summary else (
        "Deterministic checks passed; visual layer skipped."
        if not has_blocking and not has_major
        else "Review failed."
    )
    if has_blocking or has_major:
        # Override the summary when the strict rule forced fail so the UI
        # surfaces the source of the failure.
        summary = summary or "Review failed."
    return ReviewResult(
        status=status,
        summary=_truncate(summary, MAX_SUMMARY_CHARS),
        findings=combined,
        model_sha256=model_sha,
        preview_sha256=preview_sha,
    )


def _build_prompt(
    *,
    request_text: str,
    model_source: str,
    metrics: dict[str, Any],
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
    # The single ``## Geometry metrics`` block carries both the geometry
    # metrics and the ``feature_summary`` sub-object (the same dictionary the
    # host surfaces via ``.cad_metrics.json``). Re-iterating the same payload
    # under a separate ``## Feature summary`` header used to double the prompt
    # size and look like a header mismatch in the reviewer's tool outputs.
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
