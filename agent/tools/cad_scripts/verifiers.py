"""Self-contained verifiers bundled into the sandboxed CAD runner.

This module is intentionally dependency-free: it must run inside the
bubblewrap sandbox which does not bind the project source tree, so importing
the agent package from the runner would fail. The CadTool copies this file
into the runner workspace alongside ``runner.py`` before launching.

The implementations mirror :mod:`agent.quality.verifiers` but use a tiny
in-process dataclass instead of importing :class:`ValidationResult` from the
host package. Results are serialized to JSON and consumed by the runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

VALIDATION_STATUSES = frozenset(
    {"passed", "failed", "unclear", "not_applicable", "not_implemented"}
)


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_validation_id() -> str:
    import uuid

    return uuid.uuid4().hex


@dataclass
class Requirement:
    id: str
    kind: str
    required: bool = True
    source_text: str = ""
    target: float | None = None
    tolerance: float = 0.05
    units: str = "mm"
    selector: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    validation_id: str
    attempt_id: str
    requirement_id: str
    verifier: str
    status: str
    created_at: str
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)
    tolerance: float | None = None
    confidence: float = 1.0
    severity: str = "blocking"
    evidence: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
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


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def bounding_box_dimensions(shape: Any) -> tuple[float, float, float]:
    box = shape.bounding_box()
    size = box.size
    return (float(size.X), float(size.Y), float(size.Z))


def is_finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


# --- solid_count -------------------------------------------------------------


def verify_solid_count(requirement: Requirement, shape: Any) -> ValidationResult:
    observed = int(len(list(shape.solids())))
    target = requirement.target
    if target is None:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.solid_count.v1",
            status="unclear",
            expected={},
            observed={"solid_count": observed},
            tolerance=0.0,
            severity="blocking" if requirement.required else "major",
            message="solid_count requirement is missing a numeric target.",
            created_at=_utc_now(),
        )
    expected_count = int(round(target))
    status = "passed" if observed == expected_count else "failed"
    message = (
        f"solid_count={observed} matches the requested {expected_count}."
        if status == "passed"
        else f"solid_count={observed} does not match the requested {expected_count}."
    )
    return ValidationResult(
        validation_id=new_validation_id(),
        attempt_id="",
        requirement_id=requirement.id,
        verifier="kernel.solid_count.v1",
        status=status,
        expected={"solid_count": expected_count},
        observed={"solid_count": observed},
        tolerance=0.0,
        severity="blocking" if requirement.required else "major",
        message=message,
        created_at=_utc_now(),
    )


def verify_body_count(requirement: Requirement, shape: Any) -> ValidationResult:
    # Single underlying call + replace() matches the host registry; avoid
    # re-running the solid-count verifier (which is deterministic but slow).
    return replace(verify_solid_count(requirement, shape), verifier="kernel.body_count.v1")


# --- dimension ---------------------------------------------------------------


def verify_dimension(requirement: Requirement, shape: Any) -> ValidationResult:
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
    axis = (requirement.selector or {}).get("axis")
    if not isinstance(axis, str) or axis.upper() not in _AXIS_INDEX:
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
    observed_value = sizes[_AXIS_INDEX[axis_upper]]
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
    observed = {"axis": axis_upper, "value_mm": round(observed_value, 4)}
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


# --- hole --------------------------------------------------------------------


_AXIS_VECTORS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def _build123d_geom_type():
    from build123d import GeomType

    return GeomType


def _cylinder_faces(shape: Any):
    geom_type = _build123d_geom_type()
    cylinder_marker = getattr(geom_type, "CYLINDER", None)
    out = []
    for face in shape.faces():
        if cylinder_marker is not None and face.geom_type == cylinder_marker:
            out.append(face)
            continue
        raw = getattr(face, "geom_type", "")
        raw_name = getattr(raw, "name", str(raw))
        if "CYLINDER" in str(raw_name).upper():
            out.append(face)
    return out


def _is_hole_face(face: Any) -> bool:
    """Boolean cuts reverse the cylindrical face orientation; bosses do not."""
    orientation = getattr(getattr(face, "wrapped", None), "Orientation", lambda: None)()
    return "REVERSED" in str(orientation).upper()


def _cylinder_diameter_mm(face: Any) -> float:
    radius = getattr(face, "radius", None)
    if radius is None:
        return 0.0
    return float(radius) * 2.0


def _cylinder_axis_unit(face: Any) -> tuple[float, float, float]:
    direction = getattr(getattr(face, "axis_of_rotation", None), "direction", None)
    if direction is None:
        return (0.0, 0.0, 0.0)
    values = (float(direction.X), float(direction.Y), float(direction.Z))
    magnitude = sum(component * component for component in values) ** 0.5
    if magnitude <= 0.0:
        return (0.0, 0.0, 0.0)
    return (values[0] / magnitude, values[1] / magnitude, values[2] / magnitude)


def _aligned_axis(unit: tuple[float, float, float]) -> str | None:
    best_axis = None
    best_dot = 0.0
    for axis, vector in _AXIS_VECTORS.items():
        dot = abs(
            unit[0] * vector[0] + unit[1] * vector[1] + unit[2] * vector[2]
        )
        if dot > best_dot:
            best_dot = dot
            best_axis = axis
    return best_axis if best_dot >= 0.99 else None


def verify_hole(requirement: Requirement, shape: Any) -> ValidationResult:
    target = requirement.target
    if target is None:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.hole.v1",
            status="unclear",
            expected={},
            observed={"cylinder_face_count": len(_cylinder_faces(shape))},
            severity="blocking" if requirement.required else "major",
            message="hole requirement has no diameter target.",
            created_at=_utc_now(),
        )
    tolerance = float(requirement.tolerance or 0.1)
    candidates = [face for face in _cylinder_faces(shape) if _is_hole_face(face)]
    observed_count = len(candidates)
    if observed_count == 0:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.hole.v1",
            status="failed",
            expected={"diameter_mm": round(float(target), 4)},
            observed={"cylinder_face_count": 0},
            tolerance=tolerance,
            severity="blocking" if requirement.required else "major",
            message="No cylindrical face was found on the live shape.",
            created_at=_utc_now(),
        )
    matches: list[dict[str, Any]] = []
    for face in candidates:
        diameter = _cylinder_diameter_mm(face)
        axis_unit = _cylinder_axis_unit(face)
        axis = _aligned_axis(axis_unit)
        expected_axis = str((requirement.selector or {}).get("axis") or "Z").upper()
        if abs(diameter - float(target)) <= tolerance and axis == expected_axis:
            center = face.center()
            matches.append(
                {
                    "diameter_mm": round(diameter, 4),
                    "axis": axis or "unknown",
                    "center_mm": [
                        round(float(center.X), 4),
                        round(float(center.Y), 4),
                        round(float(center.Z), 4),
                    ],
                }
            )
    if matches:
        return ValidationResult(
            validation_id=new_validation_id(),
            attempt_id="",
            requirement_id=requirement.id,
            verifier="kernel.hole.v1",
            status="passed",
            expected={"diameter_mm": round(float(target), 4), "axis": expected_axis, "tolerance_mm": tolerance},
            observed={"matches": matches, "cylinder_face_count": observed_count},
            tolerance=tolerance,
            severity="blocking" if requirement.required else "major",
            message=(
                f"Found {len(matches)} cylindrical face(s) matching the "
                f"requested {float(target):.3f} mm diameter within "
                f"{tolerance:.3f} mm."
            ),
            created_at=_utc_now(),
        )
    diameters = sorted(round(_cylinder_diameter_mm(f), 4) for f in candidates)
    return ValidationResult(
        validation_id=new_validation_id(),
        attempt_id="",
        requirement_id=requirement.id,
        verifier="kernel.hole.v1",
        status="failed",
        expected={"diameter_mm": round(float(target), 4), "axis": str((requirement.selector or {}).get("axis") or "Z").upper(), "tolerance_mm": tolerance},
        observed={"cylinder_face_count": observed_count, "diameters_mm": diameters},
        tolerance=tolerance,
        severity="blocking" if requirement.required else "major",
        message=(
            f"No cylindrical face matches the requested "
            f"{float(target):.3f} mm diameter within {tolerance:.3f} mm. "
            f"Observed diameters: {diameters}."
        ),
        created_at=_utc_now(),
    )


# --- Registry ----------------------------------------------------------------

VERIFIERS: dict[str, Callable[[Requirement, Any], ValidationResult]] = {
    "solid_count": verify_solid_count,
    "body_count": verify_body_count,
    "dimension": verify_dimension,
    "hole": verify_hole,
}


def _requirement_from_dict(data: dict[str, Any]) -> Requirement:
    selector = data.get("selector") or {}
    extras = data.get("extras") or {}
    return Requirement(
        id=str(data.get("id") or ""),
        kind=str(data.get("kind") or "manual_review"),
        required=bool(data.get("required", True)),
        source_text=str(data.get("source_text") or ""),
        target=float(data["target"]) if isinstance(data.get("target"), (int, float)) else None,
        tolerance=float(data.get("tolerance", 0.05) or 0.05),
        units=str(data.get("units") or "mm"),
        selector=dict(selector) if isinstance(selector, dict) else {},
        extras=dict(extras) if isinstance(extras, dict) else {},
    )


def run(requirements_payload: Iterable[dict[str, Any]], shape: Any, *, attempt_id: str) -> list[dict[str, Any]]:
    results: list[ValidationResult] = []
    for raw in requirements_payload:
        if not isinstance(raw, dict):
            continue
        requirement = _requirement_from_dict(raw)
        verifier = VERIFIERS.get(requirement.kind)
        if verifier is None:
            results.append(
                ValidationResult(
                    validation_id=new_validation_id(),
                    attempt_id=attempt_id,
                    requirement_id=requirement.id,
                    verifier=f"unknown.{requirement.kind}",
                    status="not_implemented",
                    expected={},
                    observed={},
                    severity="blocking" if requirement.required else "minor",
                    message=(
                        f"No verifier is registered for kind "
                        f"{requirement.kind!r} in this build."
                    ),
                    created_at=_utc_now(),
                )
            )
            continue
        try:
            result = verifier(requirement, shape)
        except Exception as error:  # noqa: BLE001 - verifier errors are findings
            results.append(
                ValidationResult(
                    validation_id=new_validation_id(),
                    attempt_id=attempt_id,
                    requirement_id=requirement.id,
                    verifier=f"kernel.{requirement.kind}.v1",
                    status="unclear",
                    expected={},
                    observed={},
                    severity="blocking" if requirement.required else "minor",
                    message=f"Verifier raised {type(error).__name__}: {error}",
                    created_at=_utc_now(),
                )
            )
            continue
        results.append(replace(result, attempt_id=attempt_id))
    return [r.to_dict() for r in results]
