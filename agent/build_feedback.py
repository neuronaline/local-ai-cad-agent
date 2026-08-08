"""Compact, spec-driven build feedback for the generator.

Phase 2 of the CAD quality plan. The generator receives a single, bounded
envelope on every successful CAD build: ``{build, requirements, omitted}``.

* ``build`` carries only the bounding-box summary needed to orient the next
  edit; it never includes volume, face data, or duplicate raw metrics.
* ``requirements`` contains only actionable non-passing items for the active
  spec. Each item carries ID, expected, observed, tolerance, severity, and a
  short repair hint. Evidence pointers are intentionally API/UI-only.
* ``omitted`` is the count of passing or already-correct items that were
  elided, so the generator still knows it has a complete picture.

When there is no spec yet, the envelope falls back to a minimal build summary
plus a note that no requirements are recorded.
"""

from __future__ import annotations

from typing import Any

# Hard envelope limits (per the roadmap). Exceeding them triggers omission of
# the lowest-priority items rather than truncating mid-finding.
ENVELOPE_MAX_BYTES = 4 * 1024
ENVELOPE_MAX_FINDINGS = 20

# Repair hints by status, ordered from most to least actionable.
_REPAIR_HINTS: dict[str, str] = {
    "failed": "Adjust model.py to satisfy this requirement, then rebuild.",
    "unclear": "Add a clarification question or a comment in model.py explaining the intent.",
    "not_implemented": "Reduce the requirement to what build123d can express today, or accept it manually.",
}


def _coerce(value: Any) -> Any:
    if isinstance(value, float):
        # Keep numbers terse; verifiers already round to mm precision.
        return round(value, 4)
    return value


def _finding_priority(item: dict[str, Any]) -> tuple[int, int]:
    severity = item.get("severity", "minor")
    severity_rank = {"blocking": 0, "major": 1, "minor": 2}.get(severity, 3)
    status_rank = {"failed": 0, "not_implemented": 1, "unclear": 2, "not_applicable": 3, "passed": 4}.get(
        item.get("status", ""), 5
    )
    return (severity_rank, status_rank)


def _project_for_fix(item: dict[str, Any]) -> str:
    """Return a short repair hint tailored to the requirement kind."""
    base = _REPAIR_HINTS.get(item.get("status", ""), "Inspect the requirement and adjust model.py.")
    verifier = item.get("verifier", "")
    if verifier.startswith("kernel.dimension"):
        return f"{base} Check the bounding box against the requested axis."
    if verifier.startswith("kernel.hole"):
        return f"{base} Check the cylindrical cut against the requested diameter/axis."
    if verifier.startswith("kernel.solid_count") or verifier.startswith("kernel.body_count"):
        return f"{base} Check that the model contains exactly the requested number of bodies."
    return base


def _format_observed(item: dict[str, Any]) -> dict[str, Any]:
    observed = item.get("observed") or {}
    if not isinstance(observed, dict):
        return {}
    # Truncate big observation lists (e.g. cylinder diameters) to one short summary.
    diameters = observed.get("diameters_mm")
    if isinstance(diameters, list) and len(diameters) > 6:
        observed = dict(observed)
        observed["diameters_mm"] = diameters[:6] + ["…"]
    return {key: _coerce(value) for key, value in observed.items()}


def _format_expected(item: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected") or {}
    if not isinstance(expected, dict):
        return {}
    return {key: _coerce(value) for key, value in expected.items()}


def _is_actionable(item: dict[str, Any]) -> bool:
    status = item.get("status", "")
    if status in {"failed", "unclear", "not_implemented"}:
        # `not_applicable` means the requirement does not apply — never actionable.
        return True
    return False


def sanitize_build_result(
    raw_result: Any,
    *,
    max_bytes: int = ENVELOPE_MAX_BYTES,
    max_findings: int = ENVELOPE_MAX_FINDINGS,
) -> dict[str, Any]:
    """Return the compact, LLM-facing envelope for one ``cad_build_and_verify`` call."""
    payload = raw_result if isinstance(raw_result, dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    dimensions = metrics.get("dimensions_mm") if isinstance(metrics.get("dimensions_mm"), dict) else {}
    build_summary = {
        "ok": bool(metrics.get("is_valid")),
        "solid_count": metrics.get("solid_count"),
        "dimensions_mm": {
            key: _coerce(dimensions.get(key)) for key in ("x", "y", "z") if key in dimensions
        },
        "preview": payload.get("preview", "preview.stl"),
        "render": payload.get("render", "render.png"),
    }
    validation = payload.get("validation")
    items = validation if isinstance(validation, list) else []

    actionable = [item for item in items if isinstance(item, dict) and _is_actionable(item)]
    actionable.sort(key=_finding_priority)

    omitted = max(0, len(items) - len(actionable))
    findings: list[dict[str, Any]] = []
    for item in actionable:
        finding = {
            "id": str(item.get("requirement_id", "")),
            "status": str(item.get("status", "")),
            "severity": str(item.get("severity", "blocking")),
            "expected": _format_expected(item),
            "observed": _format_observed(item),
            "message": str(item.get("message", ""))[:400],
            "repair_hint": _project_for_fix(item),
        }
        tolerance = item.get("tolerance")
        if isinstance(tolerance, (int, float)):
            finding["tolerance"] = round(float(tolerance), 4)
        findings.append(finding)

    if len(findings) > max_findings:
        omitted += len(findings) - max_findings
        findings = findings[:max_findings]

    envelope = {
        "build": build_summary,
        "requirements": findings,
        "omitted": omitted,
    }

    serialized = json_dumps(envelope)
    if len(serialized.encode("utf-8")) > max_bytes:
        # Drop findings until we are under the byte cap; track omission count.
        while (
            len(serialized.encode("utf-8")) > max_bytes
            and envelope["requirements"]
        ):
            envelope["requirements"].pop()
            envelope["omitted"] += 1
            serialized = json_dumps(envelope)
    return envelope


def json_dumps(envelope: dict[str, Any]) -> str:
    import json

    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
