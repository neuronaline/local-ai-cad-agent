"""CAD quality observability: taxonomy, typed models, and project-local store.

Phase 0 + Phase 1 of the CAD quality plan: a frozen error/status vocabulary and
append-only run/attempt persistence that every later validation gate builds on.

Phase 2 adds a code-only spec parser plus a registry of deterministic verifiers
that produce inspectable validation records next to each successful attempt.
"""

from agent.quality.errors import (
    ACCEPTED_DECISION_TYPES,
    DECISION_TYPES,
    ISSUE_SEVERITIES,
    RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    VALIDATION_STATUSES,
)
from agent.quality.models import (
    DesignSpec,
    Issue,
    Requirement,
    SCHEMA_VERSION,
    TaskRun,
    UserDecision,
    ValidationResult,
)
from agent.quality.specification import parse_request
from agent.quality.store import (
    QualityError,
    QualityIntegrityError,
    QualityLimitError,
    QualityStore,
)
from agent.quality.verifiers import run_verifiers

__all__ = [
    "ACCEPTED_DECISION_TYPES",
    "DECISION_TYPES",
    "DesignSpec",
    "ISSUE_SEVERITIES",
    "Issue",
    "QualityError",
    "QualityIntegrityError",
    "QualityLimitError",
    "QualityStore",
    "Requirement",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "TERMINAL_RUN_STATUSES",
    "TaskRun",
    "UserDecision",
    "VALIDATION_STATUSES",
    "ValidationResult",
    "parse_request",
    "run_verifiers",
]
