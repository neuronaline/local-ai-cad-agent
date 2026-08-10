"""Parse user requests into a stable, typed :class:`DesignSpec`.

Phase 2 of the CAD quality plan. The parser is intentionally code-only — no
LLM call, no eval harness, no regression set — because every requirement we
emit must be **stable** across runs and **inspectable** on disk. The spec is
the contract between the user request and the deterministic verifiers.

The parser is deliberately conservative:

* Anything it cannot classify becomes a ``manual_review`` requirement, never a
  blocking failure. That keeps the generator from being forced to fix what it
  cannot verify and never invented out of thin air.
* Only kinds we have a deterministic verifier for (see ``verifiers/``) are
  emitted as blocking. Anything else is kept as ``manual_review`` with the
  user's source text preserved.
* Numbers are extracted with a small bounded grammar; ambiguous phrases
  ("about 50 mm") keep the source text and a ``tolerance`` default so the LLM
  can still act on them.

The parser does not parse the whole conversation history. It only looks at the
initial request and any explicit answers to recorded clarification questions
supplied via :func:`parse_request`. Failed agent prose, alternative attempts,
and tool errors are deliberately ignored.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

from agent.quality.models import (
    DesignSpec,
    Requirement,
    SPEC_KINDS,
    _slug_requirement_id,
    _utc_now,
)

# Verifier coverage in Phase 2. Any other kind is downgraded to manual_review.
# (See ``agent/quality/verifiers/registry.py`` for the actual callable set.)
_VERIFIED_KINDS = frozenset({"solid_count", "body_count", "dimension", "hole"})


# --- Number / dimension extraction ------------------------------------------------

_NUM_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in|deg|degrees|°|rad)?",
    re.IGNORECASE,
)
_AXIS_PATTERN = re.compile(r"\b(?P<axis>[xyz])\s*(?:-?\s*axis|length|width|height|depth|size)", re.IGNORECASE)
# Single source of truth for axis-prefixed numeric dimensions (audit_044).
_AXIS_DIMENSION_PATTERN = re.compile(
    r"(?P<axis>[xyz])\s*(?:length|width|height|depth|size)\s*[:=]?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?",
    re.IGNORECASE,
)
_HOLE_KEYWORDS = re.compile(r"\b(holes?|bores?|cutouts?|cut-outs?|through[ -]holes?)\b", re.IGNORECASE)
_HOLE_PATTERN_KEYWORDS = re.compile(
    r"\b(\d+\s*(?:x|×)\s*[^\s,]+|\d+\s*holes?|hole\s+pattern|pair\s+of\s+holes)\b",
    re.IGNORECASE,
)
_DIAMETER_PATTERN = re.compile(
    r"(?:diameter|Ø|dia\.?|radius)\s*(?:=|of|is|:)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?",
    re.IGNORECASE,
)
_COUNT_PATTERN = re.compile(
    r"\b(?:(?P<count>\d+)\s+(?:solid|body|solids|bodies|parts?)|a\s+(?:single|one)\s+(?:solid|body|part))\b",
    re.IGNORECASE,
)


def _to_mm(value: float, unit: str | None) -> float | None:
    if value is None:
        return None
    unit = (unit or "mm").lower()
    if unit in {"mm", ""}:
        return float(value)
    if unit == "cm":
        return float(value) * 10.0
    if unit == "m":
        return float(value) * 1000.0
    if unit in {"inch", "in"}:
        return float(value) * 25.4
    # angles / unknown — keep the raw value, caller decides
    return float(value)


def _axis_for(text: str) -> str | None:
    match = _AXIS_PATTERN.search(text or "")
    if match:
        return match.group("axis").upper()
    lowered = (text or "").lower()
    for token, axis in (
        ("length", "X"),
        ("width", "Y"),
        ("depth", "Z"),
        ("height", "Z"),
        ("long", "X"),
        ("thick", "Z"),
    ):
        if token in lowered:
            return axis
    return None


def _strip_axis_words(text: str) -> str:
    """Remove the axis token after we recorded it, so the next regex sees cleaner input."""
    cleaned = _AXIS_PATTERN.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_blocking(kind: str) -> bool:
    return kind in _VERIFIED_KINDS


def _make_requirement(
    kind: str,
    source_text: str,
    *,
    index: int,
    target: float | None = None,
    extras: dict[str, Any] | None = None,
    selector: dict[str, Any] | None = None,
    tolerance: float = 0.05,
    units: str = "mm",
    required: bool | None = None,
) -> Requirement:
    blocking = _is_blocking(kind)
    if required is None:
        required = blocking
    rid = _slug_requirement_id("REQ", source_text or kind, index)
    return Requirement(
        id=rid,
        kind=kind,
        required=required,
        source_text=source_text.strip(),
        target=target,
        tolerance=tolerance,
        units=units,
        selector=dict(selector or {}),
        extras=dict(extras or {}),
    )


# --- Public API -----------------------------------------------------------------


def parse_request(
    *,
    run_id: str,
    request_text: str,
    answers: Iterable[dict[str, Any]] | None = None,
    version: int = 1,
    extra_texts: Iterable[str] | None = None,
) -> DesignSpec:
    """Return a :class:`DesignSpec` for one user request.

    ``answers`` may carry explicit values from clarification questions. The
    parser only uses well-known keys (``dimension_mm``, ``hole_diameter_mm``,
    ``hole_count``, ``solid_count``) to refine blocking requirements; everything
    else becomes ``manual_review``.
    """
    texts: list[str] = [request_text or ""]
    if extra_texts:
        texts.extend(text for text in extra_texts if text)
    if answers:
        for entry in answers:
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            if value is None or value == "":
                continue
            label = str(entry.get("label") or entry.get("id") or "answer").strip()
            texts.append(f"{label}: {value}")
    corpus = "\n".join(texts).strip()

    requirements: list[Requirement] = []
    answer_index = _answer_index(answers)

    requirements.extend(_extract_dimensions(corpus))
    requirements.extend(_extract_holes(corpus, answer_index))
    requirements.extend(_extract_solid_count(corpus, answer_index))
    requirements.extend(_extract_forbidden(corpus))

    if not requirements:
        # Keep the whole request visible as a manual-review item so the UI is
        # never empty even when the parser cannot classify anything.
        requirements.append(
            _make_requirement(
                "manual_review",
                corpus or "User request did not contain classifiable requirements.",
                index=1,
            )
        )

    # Deduplicate by (kind, target, selector) keeping the first occurrence.
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Requirement] = []
    for req in requirements:
        key = (req.kind, req.target, tuple(sorted(req.selector.items())), tuple(sorted(req.extras.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(req)

    return DesignSpec(
        run_id=run_id,
        version=version,
        requirements=tuple(deduped),
        units="mm",
        request_text=request_text or "",
        created_at=_utc_now(),
    )


def _answer_index(answers: Iterable[dict[str, Any]] | None) -> dict[str, Any]:
    """Map known clarification keys to their numeric/string values.

    Performs a single canonicalization pass via ``_canonical_answer_key``
    (audit_045); the previous two-step normalise-then-lookup cost a redundant
    string scan on every entry.
    """
    out: dict[str, Any] = {}
    if not answers:
        return out
    for entry in answers:
        if not isinstance(entry, dict):
            continue
        raw_key = entry.get("id") or entry.get("key") or ""
        canonical = _canonical_answer_key(str(raw_key))
        if not canonical:
            continue
        value = entry.get("value")
        if value is None or value == "":
            continue
        out[canonical] = value
    return out


_CANONICAL_KEYS = {
    "dimension_mm": "dimension_mm",
    "length_mm": "dimension_mm",
    "width_mm": "dimension_mm",
    "height_mm": "dimension_mm",
    "hole_diameter_mm": "hole_diameter_mm",
    "diameter_mm": "hole_diameter_mm",
    "hole_count": "hole_count",
    "solid_count": "solid_count",
    "body_count": "body_count",
}


def _canonical_answer_key(key: str) -> str:
    lowered = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
    return _CANONICAL_KEYS.get(lowered, lowered)


# --- Extractors -----------------------------------------------------------------


def _extract_dimensions(text: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    lowered = text or ""
    # Axis-named dimensions first ("X length: 50 mm").
    for index, match in enumerate(_AXIS_DIMENSION_PATTERN.finditer(lowered)):
        value = _to_mm(float(match.group("value")), match.group("unit"))
        axis = match.group("axis").upper()
        requirements.append(
            _make_requirement(
                "dimension",
                match.group(0).strip(),
                index=index + 1,
                target=value,
                selector={"axis": axis, "measure": "bounding_box"},
                units="mm",
            )
        )
    # Bare numbers like "50 mm" attached to dimension words.
    if not any(req.selector.get("axis") for req in requirements):
        for index, match in enumerate(re.finditer(
            r"(?P<lead>width|length|height|depth|thickness|diameter|size|radius)\s*[:=]?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?",
            lowered,
            re.IGNORECASE,
        )):
            value = _to_mm(float(match.group("value")), match.group("unit"))
            axis = _axis_for(match.group(0))
            if axis is None:
                continue
            requirements.append(
                _make_requirement(
                    "dimension",
                    match.group(0).strip(),
                    index=index + 1,
                    target=value,
                    selector={"axis": axis, "measure": "bounding_box"},
                    units="mm",
                )
            )
    return requirements


def _extract_holes(text: str, answers: dict[str, Any]) -> list[Requirement]:
    if not _HOLE_KEYWORDS.search(text or ""):
        return []
    requirements: list[Requirement] = []
    # Explicit "diameter 3.6 mm hole/bore" pattern wins. Both orderings.
    for index, match in enumerate(re.finditer(
        r"(?:holes?|bores?|cutouts?|cut-outs?|through[ -]holes?)\b[^.\n]{0,40}?(?:diameter|Ø|dia\.?|radius)\s*(?:=|of|is|:)?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?",
        text,
        re.IGNORECASE,
    )):
        value = _to_mm(float(match.group("value")), match.group("unit"))
        if re.search(r"(?:radius)\s*(?:=|of|is|:)?\s*$", match.group(0)[:match.start("value") - match.start()], re.IGNORECASE):
            value *= 2.0
        requirements.append(
            _make_requirement(
                "hole",
                match.group(0).strip(),
                index=index + 1,
                target=value,
                selector={"axis": _hole_axis(text, match.start()), "feature": "hole"},
                units="mm",
            )
        )
    if not requirements:
        # Reverse orderings: "6 mm diameter hole" or "6 mm hole diameter".
        for index, match in enumerate(re.finditer(
            r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?\s*(?:(?:diameter|Ø|dia\.?|radius)\s*)?(?:holes?|bores?|cutouts?|cut-outs?|through[ -]holes?)",
            text,
            re.IGNORECASE,
        )):
            value = _to_mm(float(match.group("value")), match.group("unit"))
            if re.search(r"(?:diameter|Ø|dia\.?)\s*$", match.group(0)[match.end("unit") if match.group("unit") else match.end("value"):], re.IGNORECASE) is None and "radius" in match.group(0).lower():
                value *= 2.0
            requirements.append(
                _make_requirement(
                    "hole",
                    match.group(0).strip(),
                    index=index + 1,
                    target=value,
                    selector={"axis": _hole_axis(text, match.start()), "feature": "hole"},
                    units="mm",
                )
            )
    # Formatted answer lines from the agent (``- <question>: <value>``). The
    # question text mentions "hole"/"diameter" but the value sits in a
    # different segment, so we lift it explicitly here.
    if not requirements:
        for index, match in enumerate(re.finditer(
            r"^[\s\-]*(?P<q>[^:\n]*?(?:hole|bore|cutout|through[ -]hole)[^:\n]*):\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|inch|in)?",
            text,
            re.IGNORECASE | re.MULTILINE,
        )):
            value = _to_mm(float(match.group("value")), match.group("unit"))
            requirements.append(
                _make_requirement(
                    "hole",
                    match.group(0).strip(),
                    index=index + 1,
                    target=value,
                    selector={"axis": "Z", "feature": "hole"},
                    units="mm",
                )
            )
    # Clarification answers refine the requirement if no diameter was explicit.
    diameter = answers.get("hole_diameter_mm")
    if diameter not in (None, "") and not requirements:
        try:
            diameter_value = float(diameter)
        except (TypeError, ValueError):
            diameter_value = None
        if diameter_value is not None:
            requirements.append(
                _make_requirement(
                    "hole",
                    f"hole diameter {diameter_value:g} mm (clarified)",
                    index=1,
                    target=diameter_value,
                    selector={"axis": "Z", "feature": "hole"},
                    units="mm",
                )
            )
    return requirements


def _hole_axis(text: str, offset: int) -> str:
    """Return the axis named in the hole's sentence, defaulting to Z."""
    start = max(text.rfind(".", 0, offset), text.rfind("\n", 0, offset)) + 1
    end_candidates = [point for point in (text.find(".", offset), text.find("\n", offset)) if point >= 0]
    end = min(end_candidates) if end_candidates else len(text)
    return _axis_for(text[start:end]) or "Z"


def _extract_solid_count(text: str, answers: dict[str, Any]) -> list[Requirement]:
    requirements: list[Requirement] = []
    for index, match in enumerate(re.finditer(
        r"\b(?P<count>\d+)\s+(?:solid|solids|bodies|body|parts?)\b",
        text or "",
        re.IGNORECASE,
    )):
        count = int(match.group("count"))
        requirements.append(
            _make_requirement(
                "solid_count",
                match.group(0).strip(),
                index=index + 1,
                target=float(count),
                selector={"measure": "solid_count"},
                units="count",
            )
        )
    clarified = answers.get("solid_count") or answers.get("body_count")
    if clarified not in (None, "") and not requirements:
        try:
            count = int(clarified)
        except (TypeError, ValueError):
            count = None
        if count is not None:
            requirements.append(
                _make_requirement(
                    "solid_count" if "solid_count" in answers else "body_count",
                    f"{count} solids (clarified)",
                    index=1,
                    target=float(count),
                    selector={"measure": "solid_count"},
                    units="count",
                )
            )
    return requirements


def _extract_forbidden(text: str) -> list[Requirement]:
    """Capture explicit "no X" / "without X" features as manual_review."""
    requirements: list[Requirement] = []
    if not text:
        return requirements
    for index, match in enumerate(re.finditer(
        r"\b(?:no|without|never|avoid)\s+(?P<feature>[a-z][a-z0-9_\- ]{1,40})\b",
        text,
        re.IGNORECASE,
    )):
        feature = match.group("feature").strip().lower().rstrip(".,;:")
        if not feature or feature in {"a", "an", "the", "any", "some"}:
            continue
        requirements.append(
            _make_requirement(
                "manual_review",
                match.group(0).strip(),
                index=index + 1,
                extras={"forbidden_feature": feature},
                required=False,
            )
        )
    return requirements


# --- Spec persistence helpers ----------------------------------------------------


def to_dict(spec: DesignSpec) -> dict[str, Any]:
    """Convenience wrapper around :meth:`DesignSpec.to_dict`."""
    return spec.to_dict()


def from_dict(data: dict[str, Any]) -> DesignSpec:
    """Convenience wrapper around :meth:`DesignSpec.from_dict`."""
    return DesignSpec.from_dict(data)


def with_requirements(spec: DesignSpec, requirements: Iterable[Requirement]) -> DesignSpec:
    """Return a new spec with ``requirements`` replaced."""
    return replace(spec, requirements=tuple(requirements))


def spec_kind_is_blocking(kind: str) -> bool:
    """Whether the parser would emit this kind as a blocking requirement."""
    return kind in _VERIFIED_KINDS and kind in SPEC_KINDS


__all__ = [
    "parse_request",
    "to_dict",
    "from_dict",
    "with_requirements",
    "spec_kind_is_blocking",
]
