"""Shared helpers for deterministic verifiers."""

from __future__ import annotations

import math
from typing import Any

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def bounding_box_dimensions(shape: Any) -> tuple[float, float, float]:
    """Return the (X, Y, Z) bounding-box sizes of a build123d shape."""
    box = shape.bounding_box()
    size = box.size
    return (float(size.X), float(size.Y), float(size.Z))


def axis_value(requirement, dimensions: tuple[float, float, float]) -> float | None:
    selector = getattr(requirement, "selector", None) or {}
    axis = selector.get("axis") if isinstance(selector, dict) else None
    if isinstance(axis, str) and axis.upper() in _AXIS_INDEX:
        return dimensions[_AXIS_INDEX[axis.upper()]]
    return None


def is_finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)
