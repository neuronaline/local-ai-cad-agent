"""Shared helpers for deterministic verifiers."""

from __future__ import annotations

import math
from typing import Any


def bounding_box_dimensions(shape: Any) -> tuple[float, float, float]:
    """Return the (X, Y, Z) bounding-box sizes of a build123d shape."""
    box = shape.bounding_box()
    size = box.size
    return (float(size.X), float(size.Y), float(size.Z))


def is_finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)