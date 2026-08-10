"""``hole`` verifier: concave cylindrical-face diameter and axis check."""

from __future__ import annotations

from typing import Any

from agent.quality.models import (
    Requirement,
    ValidationResult,
    _utc_now,
    new_validation_id,
)
from agent.quality.verifiers.registry import register

# Lazy import inside the verifier to keep ``import build123d`` out of the
# module-level path (the package is optional at test time).
def _build123d_geom_type():
    from build123d import GeomType  # type: ignore

    return GeomType


_AXIS_VECTORS = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def _cylinder_faces(shape: Any):
    geom_type = _build123d_geom_type()
    cylinder_marker = getattr(geom_type, "CYLINDER", None)
    out = []
    for face in shape.faces():
        if cylinder_marker is not None and face.geom_type == cylinder_marker:
            out.append(face)
            continue
        # Fallback for any build123d version that exposes ``geom_type`` as a
        # raw string rather than an enum.
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
        dot = abs(unit[0] * vector[0] + unit[1] * vector[1] + unit[2] * vector[2])
        if dot > best_dot:
            best_dot = dot
            best_axis = axis
    return best_axis if best_dot >= 0.99 else None


@register("hole", "kernel.hole.v1")
def verify_hole(requirement: Requirement, shape: Any) -> ValidationResult:
    """Find a cylindrical face matching the requested diameter within tolerance."""
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
    # Preserve an explicit ``tolerance=0`` (audit_077); ``requirement.tolerance``
    # is already validated to be a finite, non-negative number by the model.
    tolerance = float(requirement.tolerance)
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
