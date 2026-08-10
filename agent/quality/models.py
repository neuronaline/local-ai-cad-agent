"""Typed domain models for CAD quality runs and attempts.

Phase 1 of the CAD quality plan. Each record is serialized as a JSON manifest
and stored immutable inside the project. ``from_dict`` tolerates unknown keys
and absent optional fields so newer readers do not break older records.

Phase 2 adds :class:`DesignSpec` (parsed user requirement list) and
:class:`ValidationResult` (deterministic verifier outcomes). They share the same
``SCHEMA_VERSION`` and the same forward-compatibility rules.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _as_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


def _is_finite_non_negative(value: float) -> bool:
    """Return True when ``value`` is a real, non-NaN, non-negative number (audit_077)."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0


@dataclass(frozen=True)
class ModelInfo:
    """The model that generated the run's source and reasoning."""

    provider: str | None = None
    name: str | None = None
    reasoning_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "name": self.name,
            "reasoning_effort": self.reasoning_effort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelInfo | None:
        if not isinstance(data, dict):
            return None
        return cls(
            provider=_as_str(data.get("provider")) or None,
            name=_as_str(data.get("name")) or None,
            reasoning_effort=_as_str(data.get("reasoning_effort")) or None,
        )


@dataclass(frozen=True)
class EnvironmentInfo:
    """Runtime environment captured at run creation time."""

    python_version: str = ""
    build123d_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "build123d_version": self.build123d_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EnvironmentInfo | None:
        if not isinstance(data, dict):
            return None
        return cls(
            python_version=_as_str(data.get("python_version")),
            build123d_version=_as_str(data.get("build123d_version")) or None,
        )


@dataclass(frozen=True)
class TaskRun:
    """One user request and every attempt made to satisfy it."""

    run_id: str
    project: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    status: str = "running"
    request_message_id: str = ""
    request_sha256: str = ""
    system_prompt_sha256: str = ""
    model: ModelInfo | None = None
    environment: EnvironmentInfo | None = None
    accepted_revision_id: str | None = None
    attempt_count: int = 0
    ended_at: str | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "project": self.project,
            "created_at": self.created_at,
            "status": self.status,
            "request_message_id": self.request_message_id,
            "request_sha256": self.request_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "model": self.model.to_dict() if self.model else None,
            "environment": self.environment.to_dict() if self.environment else None,
            "accepted_revision_id": self.accepted_revision_id,
            "attempt_count": self.attempt_count,
            "ended_at": self.ended_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRun:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            run_id=_as_str(data.get("run_id")),
            project=_as_str(data.get("project")),
            created_at=_as_str(data.get("created_at"), _utc_now()),
            status=_as_str(data.get("status"), "running"),
            request_message_id=_as_str(data.get("request_message_id")),
            request_sha256=_as_str(data.get("request_sha256")),
            system_prompt_sha256=_as_str(data.get("system_prompt_sha256")),
            model=ModelInfo.from_dict(data.get("model")),
            environment=EnvironmentInfo.from_dict(data.get("environment")),
            accepted_revision_id=_as_str(data.get("accepted_revision_id")) or None,
            attempt_count=int(data.get("attempt_count", 0)),
            ended_at=_as_str(data.get("ended_at")) or None,
            error=data.get("error") if isinstance(data.get("error"), dict) else None,
        )


@dataclass(frozen=True)
class Attempt:
    """One immutable execution of one source revision."""

    attempt_id: str
    run_id: str
    started_at: str
    schema_version: int = SCHEMA_VERSION
    revision_id: str | None = None
    tool_call_id: str = ""
    source_sha256: str | None = None
    status: str = "running"
    phase: str = "execution"
    completed_at: str | None = None
    error: dict[str, Any] | None = None
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "revision_id": self.revision_id,
            "tool_call_id": self.tool_call_id,
            "source_sha256": self.source_sha256,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "phase": self.phase,
            "error": self.error,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attempt:
        artifacts = data.get("artifacts")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            attempt_id=_as_str(data.get("attempt_id")),
            run_id=_as_str(data.get("run_id")),
            revision_id=_as_str(data.get("revision_id")) or None,
            tool_call_id=_as_str(data.get("tool_call_id")),
            source_sha256=_as_str(data.get("source_sha256")) or None,
            started_at=_as_str(data.get("started_at"), _utc_now()),
            completed_at=_as_str(data.get("completed_at")) or None,
            status=_as_str(data.get("status"), "running"),
            phase=_as_str(data.get("phase"), "execution"),
            error=data.get("error") if isinstance(data.get("error"), dict) else None,
            artifacts=artifacts if isinstance(artifacts, dict) else {},
            metrics=data.get("metrics") if isinstance(data.get("metrics"), dict) else None,
        )


@dataclass(frozen=True)
class UserDecision:
    """An explicit user decision about one revision/attempt.

    ``preview_displayed`` is a transport status; only an explicit decision is
    evidence of acceptance or rejection.
    """

    decision_id: str
    run_id: str
    revision_id: str
    attempt_id: str
    decision: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    categories: tuple[str, ...] = ()
    comment: str = ""
    camera: dict[str, Any] | None = None
    supersedes_decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "revision_id": self.revision_id,
            "attempt_id": self.attempt_id,
            "decision": self.decision,
            "categories": list(self.categories),
            "comment": self.comment,
            "camera": self.camera,
            "supersedes_decision_id": self.supersedes_decision_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserDecision:
        categories = data.get("categories")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            decision_id=_as_str(data.get("decision_id")),
            run_id=_as_str(data.get("run_id")),
            revision_id=_as_str(data.get("revision_id")),
            attempt_id=_as_str(data.get("attempt_id")),
            decision=_as_str(data.get("decision")),
            categories=tuple(
                str(item) for item in categories if isinstance(item, str)
            )
            if isinstance(categories, list)
            else (),
            comment=_as_str(data.get("comment")),
            camera=data.get("camera") if isinstance(data.get("camera"), dict) else None,
            supersedes_decision_id=_as_str(data.get("supersedes_decision_id")) or None,
            created_at=_as_str(data.get("created_at"), _utc_now()),
        )


@dataclass(frozen=True)
class Issue:
    """An actionable defect tied to one revision/attempt."""

    issue_id: str
    run_id: str
    attempt_id: str
    revision_id: str
    category: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    source: str = "user"
    severity: str = "blocking"
    message: str = ""
    requirement_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    status: str = "open"
    resolved_by_revision_id: str | None = None
    resolved_at: str | None = None
    confirmed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issue_id": self.issue_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "revision_id": self.revision_id,
            "source": self.source,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "requirement_ids": list(self.requirement_ids),
            "evidence": list(self.evidence),
            "status": self.status,
            "resolved_by_revision_id": self.resolved_by_revision_id,
            "resolved_at": self.resolved_at,
            "confirmed_by": self.confirmed_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Issue:
        def _tuple_of(value: Any) -> tuple[str, ...]:
            return (
                tuple(str(item) for item in value if isinstance(item, str))
                if isinstance(value, list)
                else ()
            )

        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            issue_id=_as_str(data.get("issue_id")),
            run_id=_as_str(data.get("run_id")),
            attempt_id=_as_str(data.get("attempt_id")),
            revision_id=_as_str(data.get("revision_id")),
            source=_as_str(data.get("source"), "user"),
            category=_as_str(data.get("category")),
            severity=_as_str(data.get("severity"), "blocking"),
            message=_as_str(data.get("message")),
            requirement_ids=_tuple_of(data.get("requirement_ids")),
            evidence=_tuple_of(data.get("evidence")),
            status=_as_str(data.get("status"), "open"),
            resolved_by_revision_id=_as_str(data.get("resolved_by_revision_id")) or None,
            resolved_at=_as_str(data.get("resolved_at")) or None,
            confirmed_by=_as_str(data.get("confirmed_by")) or None,
            created_at=_as_str(data.get("created_at"), _utc_now()),
        )


# --- Phase 2: DesignSpec + ValidationResult -------------------------------

# Stable kinds implemented by the Phase 2 verifier registry. Unknown kinds are
# kept in the spec as ``manual_review`` so the generator does not silently lose
# the user's request — see ``agent/quality/specification.py`` for the parser.
SPEC_KINDS = frozenset(
    {
        "solid_count",
        "body_count",
        "dimension",
        "hole",
        "hole_pattern",
        "axis_alignment",
        "position",
        "symmetry",
        "clearance",
        "clearance_path",
        "connectivity",
        "non_intersection",
        "minimum_thickness",
        "bed_contact",
        "forbidden_feature",
        "visual_shape",
        "manual_review",
    }
)


def _slug_requirement_id(prefix: str, source_text: str, index: int) -> str:
    """Stable, human-readable requirement ID (e.g. ``REQ-BASE-X``)."""
    text = source_text or ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    if not slug:
        slug = f"ITEM{index:03d}"
    slug = slug[:32]
    return f"{prefix}-{slug}"


@dataclass(frozen=True)
class Requirement:
    """One parsed requirement from a :class:`DesignSpec`."""

    id: str
    kind: str
    required: bool = True
    source_text: str = ""
    target: float | None = None
    tolerance: float = 0.05
    units: str = "mm"
    selector: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "required": self.required,
            "source_text": self.source_text,
            "tolerance": self.tolerance,
            "units": self.units,
        }
        if self.target is not None:
            data["target"] = self.target
        if self.selector:
            data["selector"] = dict(self.selector)
        if self.extras:
            data["extras"] = dict(self.extras)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        selector = data.get("selector")
        extras = data.get("extras")
        # Preserve an explicit ``tolerance=0`` (audit_077) instead of replacing
        # it with the default. ``0 or 0.05`` would coerce zero to the default.
        raw_tolerance = data.get("tolerance", 0.05)
        tolerance = float(raw_tolerance) if isinstance(raw_tolerance, (int, float)) else 0.05
        if not _is_finite_non_negative(tolerance):
            tolerance = 0.05
        return cls(
            id=_as_str(data.get("id")),
            kind=_as_str(data.get("kind"), "manual_review"),
            required=bool(data.get("required", True)),
            source_text=_as_str(data.get("source_text")),
            target=float(data["target"]) if isinstance(data.get("target"), (int, float)) else None,
            tolerance=tolerance,
            units=_as_str(data.get("units"), "mm") or "mm",
            selector=dict(selector) if isinstance(selector, dict) else {},
            extras=dict(extras) if isinstance(extras, dict) else {},
        )


@dataclass(frozen=True)
class DesignSpec:
    """A parsed, persisted list of requirements for one task run.

    Persisted next to the run it belongs to at
    ``runs/<run_id>/specs/<version>.json``. ``spec_version`` is referenced from
    every :class:`Attempt` so later attempts can be interpreted against the
    exact spec that drove them.
    """

    run_id: str
    version: int
    requirements: tuple[Requirement, ...]
    forbidden: tuple[dict[str, Any], ...] = ()
    units: str = "mm"
    schema_version: int = SCHEMA_VERSION
    request_text: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "version": self.version,
            "units": self.units,
            "request_text": self.request_text,
            "created_at": self.created_at,
            "requirements": [req.to_dict() for req in self.requirements],
            "forbidden": [dict(item) for item in self.forbidden],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignSpec:
        requirements_raw = data.get("requirements") or []
        requirements = tuple(
            Requirement.from_dict(item) for item in requirements_raw
            if isinstance(item, dict)
        )
        forbidden_raw = data.get("forbidden") or []
        forbidden = tuple(
            dict(item) for item in forbidden_raw if isinstance(item, dict)
        )
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            run_id=_as_str(data.get("run_id")),
            version=int(data.get("version", 1) or 1),
            requirements=requirements,
            forbidden=forbidden,
            units=_as_str(data.get("units"), "mm") or "mm",
            request_text=_as_str(data.get("request_text")),
            created_at=_as_str(data.get("created_at"), _utc_now()),
        )


@dataclass(frozen=True)
class ValidationResult:
    """One deterministic verifier outcome against one requirement."""

    validation_id: str
    attempt_id: str
    requirement_id: str
    verifier: str
    status: str
    created_at: str
    schema_version: int = SCHEMA_VERSION
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)
    tolerance: float | None = None
    confidence: float = 1.0
    severity: str = "blocking"
    evidence: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "validation_id": self.validation_id,
            "attempt_id": self.attempt_id,
            "requirement_id": self.requirement_id,
            "verifier": self.verifier,
            "status": self.status,
            "severity": self.severity,
            "confidence": self.confidence,
            "expected": dict(self.expected),
            "observed": dict(self.observed),
            "evidence": list(self.evidence),
            "message": self.message,
            "created_at": self.created_at,
        }
        if self.tolerance is not None:
            data["tolerance"] = self.tolerance
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        evidence = data.get("evidence")
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            validation_id=_as_str(data.get("validation_id")),
            attempt_id=_as_str(data.get("attempt_id")),
            requirement_id=_as_str(data.get("requirement_id")),
            verifier=_as_str(data.get("verifier")),
            status=_as_str(data.get("status"), "unclear"),
            severity=_as_str(data.get("severity"), "blocking"),
            confidence=float(data.get("confidence", 1.0) or 1.0),
            expected=dict(data.get("expected") or {}) if isinstance(data.get("expected"), dict) else {},
            observed=dict(data.get("observed") or {}) if isinstance(data.get("observed"), dict) else {},
            tolerance=float(data["tolerance"]) if isinstance(data.get("tolerance"), (int, float)) else None,
            evidence=tuple(str(item) for item in evidence if isinstance(item, str))
            if isinstance(evidence, list)
            else (),
            message=_as_str(data.get("message")),
            created_at=_as_str(data.get("created_at"), _utc_now()),
        )


def new_validation_id() -> str:
    """UUID4 hex identifier for a :class:`ValidationResult` record."""
    return uuid.uuid4().hex
