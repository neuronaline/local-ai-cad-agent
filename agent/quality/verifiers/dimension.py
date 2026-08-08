"""``dimension`` verifier: bounding-box axis comparison.

Phase 2 only supports global bounding-box comparisons. Named-feature
measurement waits for stable feature selectors in a later phase.
"""

from __future__ import annotations

from typing import Any

from agent.quality.models import (
    Requirement,
    ValidationResult,
    _utc_now,
    new_validation_id,
)
from agent.quality.verifiers._common import bounding_box_dimensions, is_finite
from agent.quality.verifiers.registry import register

_AXIS_KEYS = {"X": "x", "Y": "y", "Z": "z"}


@register("dimension", "kernel.dimension.v1")
def verify_dimension(requirement: Requirement, shape: Any) -> ValidationResult:
    """Compare one bounding-box axis against the requirement's target value."""
    sizes = bounding_box_dimensions(shape)
    if not is_finite(*sizes):
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.dimension.v1",
            status="unclear",
            expected={},
            observed={},
            severity="blocking" if requirement.required else "major",
            message="Bounding box has non-finite dimensions; cannot validate.",
            created_at=_utc_now(),
        )
    selector = requirement.selector or {}
    axis = selector.get("axis") if isinstance(selector, dict) else None
    if not isinstance(axis, str) or axis.upper() not in _AXIS_KEYS:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.dimension.v1",
            status="unclear",
            expected={},
            observed={},
            severity="blocking" if requirement.required else "major",
            message=(
                f"dimension requirement {requirement.id!r} is missing a "
                "selector.axis value."
            ),
            created_at=_utc_now(),
        )
    axis_upper = axis.upper()
    observed_value = sizes["XYZ".index(axis_upper)]
    target = requirement.target
    if target is None:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.dimension.v1",
            status="unclear",
            expected={},
            observed={"axis": axis_upper, "value_mm": round(observed_value, 4)},
            severity="blocking" if requirement.required else "major",
            message=f"dimension requirement {requirement.id!r} has no target value.",
            created_at=_utc_now(),
        )
    tolerance = float(requirement.tolerance or 0.05)
    diff = observed_value - float(target)
    within = abs(diff) <= tolerance
    status = "passed" if within else "failed"
    observed = {
        "axis": axis_upper,
        "value_mm": round(observed_value, 4),
    }
    expected = {
        "axis": axis_upper,
        "value_mm": round(float(target), 4),
        "tolerance_mm": tolerance,
    }
    if within:
        message = (
            f"{axis_upper} dimension {observed_value:.3f} mm matches the "
            f"target {float(target):.3f} mm (tolerance {tolerance:.3f})."
        )
    else:
        message = (
            f"{axis_upper} dimension {observed_value:.3f} mm differs from the "
            f"target {float(target):.3f} mm by {diff:+.3f} mm "
            f"(tolerance {tolerance:.3f})."
        )
    return ValidationResult(
        validation_id=new_validation_id(),
        attempt_id="",
        requirement_id=requirement.id,
        verifier="kernel.dimension.v1",
        status=status,
        expected=expected,
        observed=observed,
        tolerance=tolerance,
        severity="blocking" if requirement.required else "major",
        message=message,
        created_at=_utc_now(),
    )
