"""Frozen error taxonomy and status vocabulary for CAD quality events.

Phase 0 of the CAD quality plan: stable machine-readable codes grouped by
category, the validation status vocabulary, and the run/attempt lifecycle
statuses. Categories and codes are append-only; adding a new code never changes
an existing one, so older readers keep working.
"""

from __future__ import annotations

from typing import Any

# --- Validation statuses (Gate outcomes). ---
VALIDATION_STATUSES = frozenset(
    {"passed", "failed", "unclear", "not_applicable", "not_implemented"}
)

# `not_implemented` must fail a required requirement unless the user explicitly
# accepts manual review (enforced by later phases; documented here).
REQUIRED_FAIL_STATUSES = frozenset({"failed", "not_implemented"})

# --- Run lifecycle statuses. ---
RUN_STATUSES = frozenset({"running", "completed", "failed", "stopped", "interrupted"})
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "stopped", "interrupted"})

# --- Attempt lifecycle statuses. ---
ATTEMPT_STATUSES = frozenset({"running", "succeeded", "failed"})
TERMINAL_ATTEMPT_STATUSES = frozenset({"succeeded", "failed"})

# --- User decisions (explicit, never inferred from preview loads). ---
DECISION_TYPES = frozenset({"accepted", "accepted_with_limitations", "rejected"})
DECISION_ACCEPTED = "accepted"
DECISION_ACCEPTED_WITH_LIMITATIONS = "accepted_with_limitations"
DECISION_REJECTED = "rejected"
ACCEPTED_DECISION_TYPES = frozenset({DECISION_ACCEPTED, DECISION_ACCEPTED_WITH_LIMITATIONS})

# --- Issue lifecycle. ---
ISSUE_STATUSES = frozenset({"open", "resolved"})
ISSUE_SEVERITIES = frozenset({"blocking", "major", "minor"})

# --- Attempt phases. ---
PHASE_SOURCE = "source"
PHASE_EXECUTION = "execution"
PHASE_KERNEL = "kernel"
PHASE_REQUIREMENTS = "requirements"
PHASE_VISUAL = "visual"
PHASE_DELIVERY = "delivery"
PHASE_ACCEPTANCE = "acceptance"
PHASE_MODEL_RESPONSE = "model_response"
ATTEMPT_PHASES = frozenset(
    {
        PHASE_SOURCE,
        PHASE_EXECUTION,
        PHASE_KERNEL,
        PHASE_REQUIREMENTS,
        PHASE_VISUAL,
        PHASE_DELIVERY,
        PHASE_ACCEPTANCE,
        PHASE_MODEL_RESPONSE,
    }
)

# --- Categories and their stable codes (from the plan's error taxonomy). ---
CATEGORY_SOURCE = "Source"
CATEGORY_API = "API"
CATEGORY_EXECUTION = "Execution"
CATEGORY_KERNEL = "Kernel"
CATEGORY_TOPOLOGY = "Topology"
CATEGORY_DIMENSION = "Dimension"
CATEGORY_POSITION = "Position"
CATEGORY_FEATURE = "Feature"
CATEGORY_FUNCTION = "Function"
CATEGORY_PRINTABILITY = "Printability"
CATEGORY_VISUAL = "Visual"
CATEGORY_DELIVERY = "Delivery"
CATEGORY_USER = "User"
# Operational extension: internal/persistence/coordination failures that are not
# caused by the CAD script or the geometry itself.
CATEGORY_SYSTEM = "System"

ERROR_TAXONOMY: dict[str, str] = {
    # Source
    "SYNTAX_ERROR": CATEGORY_SOURCE,
    "BLOCKED_IMPORT": CATEGORY_SOURCE,
    "UNSAFE_CALL": CATEGORY_SOURCE,
    "CONSTRAINT_VIOLATION": CATEGORY_SOURCE,
    # API
    "UNKNOWN_SYMBOL": CATEGORY_API,
    "INVALID_ARGUMENT": CATEGORY_API,
    "VERSION_MISMATCH": CATEGORY_API,
    "EMPTY_COMPLETION": CATEGORY_API,
    # Execution
    "TIMEOUT": CATEGORY_EXECUTION,
    "OUT_OF_MEMORY": CATEGORY_EXECUTION,
    "SANDBOX_FAILURE": CATEGORY_EXECUTION,
    "EXPORT_FAILURE": CATEGORY_EXECUTION,
    # Kernel
    "EMPTY_SHAPE": CATEGORY_KERNEL,
    "INVALID_BREP": CATEGORY_KERNEL,
    "ZERO_VOLUME": CATEGORY_KERNEL,
    "NON_FINITE_METRIC": CATEGORY_KERNEL,
    # Topology
    "WRONG_SOLID_COUNT": CATEGORY_TOPOLOGY,
    "DISCONNECTED_BODY": CATEGORY_TOPOLOGY,
    "TANGENT_ONLY_CONTACT": CATEGORY_TOPOLOGY,
    # Dimension
    "OUT_OF_TOLERANCE": CATEGORY_DIMENSION,
    "WRONG_DIAMETER": CATEGORY_DIMENSION,
    "WRONG_THICKNESS": CATEGORY_DIMENSION,
    # Position
    "MISALIGNED_AXIS": CATEGORY_POSITION,
    "WRONG_CENTER": CATEGORY_POSITION,
    "ASYMMETRIC_PATTERN": CATEGORY_POSITION,
    # Feature
    "MISSING_FEATURE": CATEGORY_FEATURE,
    "EXTRA_FEATURE": CATEGORY_FEATURE,
    "FORBIDDEN_FEATURE": CATEGORY_FEATURE,
    # Function
    "BLOCKED_CLEARANCE_PATH": CATEGORY_FUNCTION,
    "INTERFERENCE": CATEGORY_FUNCTION,
    "INSUFFICIENT_CLEARANCE": CATEGORY_FUNCTION,
    # Printability
    "FLOATING_BODY": CATEGORY_PRINTABILITY,
    "INSUFFICIENT_BED_CONTACT": CATEGORY_PRINTABILITY,
    "THIN_WALL": CATEGORY_PRINTABILITY,
    "SUPPORT_REQUIRED": CATEGORY_PRINTABILITY,
    # Visual
    "INTENT_MISMATCH": CATEGORY_VISUAL,
    "OCCLUDED_EVIDENCE": CATEGORY_VISUAL,
    "JUDGE_UNCLEAR": CATEGORY_VISUAL,
    # Delivery
    "STALE_PREVIEW": CATEGORY_DELIVERY,
    "PREVIEW_LOAD_FAILED": CATEGORY_DELIVERY,
    "ARTIFACT_DIGEST_MISMATCH": CATEGORY_DELIVERY,
    "ACCEPTANCE_REQUIRED": CATEGORY_DELIVERY,
    # User
    "USER_REJECTED": CATEGORY_USER,
    "USER_CORRECTION": CATEGORY_USER,
    "USER_ACCEPTED_WITH_LIMITATION": CATEGORY_USER,
    # System (operational extension)
    "REVISION_INTEGRITY_ERROR": CATEGORY_SYSTEM,
    "TOOL_EXECUTION_FAILED": CATEGORY_SYSTEM,
    "TOOL_CALL_LIMIT": CATEGORY_SYSTEM,
}

# Legacy codes already emitted by the current tool layer, mapped into the
# frozen taxonomy so attempt records stay categorizable without retraining.
_LEGACY_MAPPING: dict[str, tuple[str, str]] = {
    "INVALID_TOOL_ARGUMENTS": (CATEGORY_API, PHASE_SOURCE),
    "VALIDATION_ERROR": (CATEGORY_API, PHASE_SOURCE),
    "UNKNOWN_TOOL": (CATEGORY_API, PHASE_SOURCE),
    "MODEL_MISSING": (CATEGORY_SOURCE, PHASE_SOURCE),
    "CAD_BUILD_FAILED": (CATEGORY_EXECUTION, PHASE_EXECUTION),
    "FILE_NOT_FOUND": (CATEGORY_SOURCE, PHASE_SOURCE),
    "TIMEOUT": (CATEGORY_EXECUTION, PHASE_EXECUTION),
    "REVISION_INTEGRITY_ERROR": (CATEGORY_SYSTEM, PHASE_SOURCE),
    "TOOL_EXECUTION_FAILED": (CATEGORY_SYSTEM, PHASE_EXECUTION),
}

# Older phase labels used by the tool layer, mapped to the canonical phases.
_PHASE_ALIASES: dict[str, str] = {
    "build": PHASE_EXECUTION,
    "arguments": PHASE_SOURCE,
    "validation": PHASE_SOURCE,
    "persistence": PHASE_SOURCE,
    "model-response": PHASE_MODEL_RESPONSE,
}


def category_of(code: str | None) -> str:
    """Return the stable category for a code, defaulting to System."""
    if not code:
        return CATEGORY_SYSTEM
    if code in ERROR_TAXONOMY:
        return ERROR_TAXONOMY[code]
    legacy = _LEGACY_MAPPING.get(code)
    return legacy[0] if legacy else CATEGORY_SYSTEM


def phase_of(code: str | None) -> str | None:
    """Return the canonical attempt phase for a legacy code, if known."""
    if not code:
        return None
    mapping = _LEGACY_MAPPING.get(code)
    return mapping[1] if mapping else None


def normalize_error(error: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a tool-layer error dict into the frozen taxonomy.

    Accepts the ``{code, phase, message, retryable, hint}`` envelope produced by
    :func:`agent.tool_results.failure` and returns a stable subset used by
    attempt records: ``{code, category, phase, message}``.
    """
    error = error if isinstance(error, dict) else {}
    code = error.get("code")
    code = str(code) if code is not None else None
    message = error.get("message")
    message = str(message) if message is not None else None
    category = error.get("category") or category_of(code)
    phase = phase_of(code) or error.get("phase") or PHASE_EXECUTION
    phase = _PHASE_ALIASES.get(str(phase), str(phase))
    normalized: dict[str, Any] = {
        "code": code or "UNKNOWN_ERROR",
        "category": category,
        "phase": phase,
        "message": message,
    }
    if error.get("retryable") is not None:
        normalized["retryable"] = bool(error["retryable"])
    if error.get("hint"):
        normalized["hint"] = str(error["hint"])
    return normalized
