"""``solid_count`` verifier: exact solid-count comparison."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent.quality.models import (
    Requirement,
    ValidationResult,
    _utc_now,
    new_validation_id,
)
from agent.quality.verifiers.registry import register


@register("solid_count", "kernel.solid_count.v1")
def verify_solid_count(requirement: Requirement, shape: Any) -> ValidationResult:
    """Compare the live shape's solid count with the requested target."""
    observed = int(len(list(shape.solids())))
    target = requirement.target
    if target is None:
        status = "unclear"
        expected_value: dict[str, Any] = {}
        message = "solid_count requirement is missing a numeric target."
    else:
        expected_count = int(round(target))
        status = "passed" if observed == expected_count else "failed"
        expected_value = {"solid_count": expected_count}
        if status == "passed":
            message = f"solid_count={observed} matches the requested {expected_count}."
        else:
            message = (
                f"solid_count={observed} does not match the requested "
                f"{expected_count}."
            )
    return ValidationResult(
        validation_id=new_validation_id(),
        attempt_id="",  # filled by run_verifiers
        requirement_id=requirement.id,
        verifier="kernel.solid_count.v1",
        status=status,
        expected=expected_value,
        observed={"solid_count": observed},
        tolerance=0.0,
        severity="blocking" if requirement.required else "major",
        message=message,
        created_at=_utc_now(),
    )


# ``body_count`` shares the same BREP primitive count.
@register("body_count", "kernel.body_count.v1")
def verify_body_count(requirement: Requirement, shape: Any) -> ValidationResult:
    result = verify_solid_count(requirement, shape)
    return replace(result, verifier="kernel.body_count.v1")
