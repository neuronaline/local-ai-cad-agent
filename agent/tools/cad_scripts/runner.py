"""Sandbox-side runner: compile, validate, and render OpenSCAD CAD models.

Executed by ``CadTool._execute`` after the host copies ``runner.py`` and
its ``renderer.py`` sibling into the bubblewrap workspace.

Produces:
- ``preview.stl``: triangulated binary STL mesh exported by OpenSCAD.
- ``render.png``: canonical isometric preview image.
- ``.cad_metrics.json``: structured geometry + feature metrics used by reviewer.
- ``.review-views/<view_id>.png`` and ``.review-sheet.png``: 8-view visual evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from renderer import (
        _HEIGHT,
        _WIDTH,
        _write_isometric_artifact,
        build_contact_sheet,
        load_stl,
        render_views,
    )
except ModuleNotFoundError:
    from .renderer import (
        _HEIGHT,
        _WIDTH,
        _write_isometric_artifact,
        build_contact_sheet,
        load_stl,
        render_views,
    )


# ``runner.py`` is copied into the bubblewrap workspace at execution time
# (see :meth:`agent.tools.cad_tool.CadTool._execute`), so it cannot import
# from the host package tree — including
# ``agent.tools.process_runner.MAX_SANDBOX_TIMEOUT_SECONDS``. Mirror the
# value here as a self-contained constant; the host-side cap MUST stay
# in lock-step with this literal so the two failure modes (outer bubblewrap
# kill vs. inner OpenSCAD timeout) remain back-to-back.
_OPENSCAD_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Feature extraction & mesh geometry
# ---------------------------------------------------------------------------


def _declared_parameters(model_code: str) -> list[dict[str, Any]]:
    """Extract uppercase numeric constants from the parameter block of model.scad."""
    parameters: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^[ \t]*([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)[ \t]*;"
    )
    in_block_comment = False
    for lineno, line in enumerate(model_code.splitlines(), start=1):
        stripped = line.strip()
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue
        if not stripped or stripped.startswith("//"):
            continue
        match = pattern.match(line)
        if match:
            name, raw_val = match.group(1), match.group(2)
            try:
                val = int(raw_val) if "." not in raw_val and "e" not in raw_val.lower() else float(raw_val)
                parameters.append({"name": name, "value": val, "line": lineno})
            except ValueError:
                continue
    return parameters


# Tolerance ratio used by :func:`_extract_scad_features` to classify
# a cylinder cutout as a "through" hole versus a "blind" pocket. A
# cutout spanning at least this ratio of the shortest bounding-box
# dimension is treated as through; ``0.9`` keeps ``h=HEIGHT + EPS``
# (which overhangs by a single epsilon) classified as through while
# obviously shallow pockets become blind.
_THROUGH_HOLE_TOLERANCE_RATIO = 0.9


class ExpressionEvaluationError(ValueError):
    """Raised when an OpenSCAD expression cannot be evaluated.

    Replaces the silent ``0.0`` fallback that previously manufactured
    zero-diameter cylinders and bypassed the geometry validator.
    """


def _eval_expr(expr_str: str, params: dict[str, float]) -> float:
    """Evaluate a numeric / parameter expression.

    Raises :class:`ExpressionEvaluationError` for unknown identifiers
    or non-numeric input; callers must surface the failure instead of
    accepting a sentinel value.
    """
    expr_str = expr_str.strip()
    if expr_str in params:
        return float(params[expr_str])
    try:
        return float(expr_str)
    except ValueError:
        pass
    # Identify unresolved identifiers up-front. The pre-refactor path
    # silently substituted ``0.0`` and shipped the result as a
    # "valid" number — the source of the zero-diameter cylinder bug.
    identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr_str))
    unresolved = sorted(identifiers - params.keys())
    if unresolved:
        raise ExpressionEvaluationError(
            f"Unknown identifier(s) in expression {expr_str!r}: {unresolved}"
        )
    substituted = re.sub(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        lambda m: str(params[m.group(0)]),
        expr_str,
    )
    if not re.match(r"^[\d\.\+\-\*\/\(\)\s]+$", substituted):
        raise ExpressionEvaluationError(
            f"Expression {expr_str!r} contains non-numeric symbols after substitution."
        )
    try:
        return float(eval(substituted, {"__builtins__": {}}, {}))
    except (ArithmeticError, ValueError, SyntaxError, TypeError) as error:
        raise ExpressionEvaluationError(
            f"Failed to evaluate {expr_str!r} after substitution: {error}"
        ) from error


def _count_disconnected_solids(triangles: np.ndarray) -> int:
    """Return number of disconnected mesh components beyond the first."""
    if len(triangles) == 0:
        return 0
    # Map edges to triangle indices
    edges_to_tris: dict[tuple[int, int], list[int]] = {}
    for idx, (v0, v1, v2) in enumerate(triangles):
        for e in ((min(v0, v1), max(v0, v1)), (min(v1, v2), max(v1, v2)), (min(v2, v0), max(v2, v0))):
            edges_to_tris.setdefault(e, []).append(idx)

    parent = list(range(len(triangles)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for tri_list in edges_to_tris.values():
        if len(tri_list) > 1:
            first = tri_list[0]
            for other in tri_list[1:]:
                union(first, other)

    roots = {find(i) for i in range(len(triangles))}
    return max(0, len(roots) - 1)


def _mesh_metrics(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Calculate exact bounding box, volume, solid count, and manifold validity."""
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("The generated CAD mesh contains no vertices or triangles.")

    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    size = max_pt - min_pt
    dim_x, dim_y, dim_z = float(size[0]), float(size[1]), float(size[2])

    if not all(math.isfinite(d) for d in (dim_x, dim_y, dim_z)):
        raise ValueError("The generated CAD shape has non-finite geometry.")

    # Exact signed volume of polyhedron via tetrahedron sum
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v0, v1)
    volume = float(abs(np.sum(cross * v2)) / 6.0)

    disconnected_count = _count_disconnected_solids(triangles)
    solid_count = max(1, disconnected_count + 1)
    is_valid = volume > 0 and dim_x > 0 and dim_y > 0 and dim_z > 0

    return {
        "solid_count": solid_count,
        "is_valid": is_valid,
        "volume_mm3": round(volume, 3),
        "dimensions_mm": {
            "x": round(dim_x, 3),
            "y": round(dim_y, 3),
            "z": round(dim_z, 3),
        },
        "disconnected_solid_count": disconnected_count,
    }


def _extract_scad_features(
    model_code: str, dims: dict[str, float], params: dict[str, float] | None = None
) -> dict[str, Any]:
    """Detect cylinder cutout features in model.scad difference() blocks.

    Cylinders whose arguments cannot be evaluated — typically because
    they reference an unknown parameter — are skipped rather than
    silently substituted with a default value. The previous heuristic
    produced zero-diameter geometries that bypassed the dimension
    validator entirely.
    """
    if params is None:
        params = {p["name"]: float(p["value"]) for p in _declared_parameters(model_code)}
    candidates: list[dict[str, Any]] = []
    cyl_pattern = re.compile(
        r"cylinder\s*\(([^)]+)\)",
        re.IGNORECASE,
    )
    if "difference" in model_code:
        for match in cyl_pattern.finditer(model_code):
            args_str = match.group(1)
            h_val = None
            d_val = None
            r_val = None
            unresolved_args: list[str] = []
            for part in args_str.split(","):
                k, v = part.split("=", 1) if "=" in part else (None, part)
                k = k.strip().lower() if k else None
                # Skip non-dimension keywords (``center=true``,
                # ``$fn=60``); routing them through ``_eval_expr``
                # would surface the right error on the wrong operand.
                if k not in (None, "h", "d", "r"):
                    continue
                try:
                    val = _eval_expr(v, params)
                except ExpressionEvaluationError as error:
                    unresolved_args.append(str(error))
                    continue
                if k == "h":
                    h_val = val
                elif k == "d":
                    d_val = val
                elif k == "r":
                    r_val = val
                elif k is None:
                    if h_val is None:
                        h_val = val
                    elif r_val is None and d_val is None:
                        r_val = val
            if unresolved_args:
                # Skip cylinders with any unresolvable dimension so
                # the reviewer does not see a partial feature.
                continue
            diameter = d_val if d_val is not None else (r_val * 2.0 if r_val is not None else None)
            if diameter is not None and h_val is not None:
                is_through = (
                    h_val
                    >= (min(dims.values()) if dims else 0)
                    * _THROUGH_HOLE_TOLERANCE_RATIO
                )
                candidates.append({
                    "diameter_mm": round(diameter, 3),
                    "axis": [0.0, 0.0, 1.0],
                    "area_mm2": round(math.pi * (diameter / 2.0) ** 2, 3),
                    "is_through_hole": bool(is_through),
                })

    through_holes = sum(1 for c in candidates if c.get("is_through_hole"))
    blind_holes = len(candidates) - through_holes
    cutouts = [
        {
            "radius": round(c["diameter_mm"] / 2.0, 3),
            "diameter": c["diameter_mm"],
            "is_through": c["is_through_hole"],
        }
        for c in candidates
    ]
    return {
        "cutouts": cutouts,
        "cylindrical_cut_candidates": candidates,
        "through_hole_count": through_holes,
        "blind_hole_count": blind_holes,
    }


# ---------------------------------------------------------------------------
# Main model runner
# ---------------------------------------------------------------------------


def _run_model(
    model_code: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile model.scad with OpenSCAD, validate geometry, and render evidence."""
    settings = settings or {}
    model_path = Path(str(settings.get("model_path", "model.scad")))
    model_path.write_text(model_code, encoding="utf-8")

    should_render = bool(settings.get("render_views", True))
    should_write_iso = bool(settings.get("write_isometric", False))
    render_workers = int(settings.get("render_workers", 4) or 4)
    required_views = int(settings.get("required_views", 8) or 8)

    # 1. Compile model.scad to binary STL (headless CGAL/CSG evaluation)
    preview_stl = Path("preview.stl")
    compile_cmd = [
        "openscad",
        "-o",
        str(preview_stl),
        "--export-format",
        "binstl",
        str(model_path),
    ]
    try:
        proc = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=_OPENSCAD_TIMEOUT_SECONDS,
            check=False,
        )
        ret, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except (subprocess.SubprocessError, OSError) as exc:
        ret, stdout, stderr = 1, "", str(exc)

    if ret != 0 or not preview_stl.is_file() or preview_stl.stat().st_size == 0:
        err_detail = (stderr or stdout).strip()
        lines = [
            line
            for line in err_detail.splitlines()
            if any(k in line.lower() for k in ("error", "warning", "syntax", "can't"))
        ]
        summary = "\n".join(lines) if lines else err_detail[-1000:]
        raise RuntimeError(f"OpenSCAD execution failed:\n{summary or 'OpenSCAD produced no output or an empty STL.'}")

    preview_bytes = preview_stl.read_bytes()
    preview_sha256 = hashlib.sha256(preview_bytes).hexdigest()

    # 2. Parse STL mesh and compute geometry metrics
    vertices, triangles = load_stl(preview_stl)
    metrics = _mesh_metrics(vertices, triangles)
    dims = metrics["dimensions_mm"]

    if not metrics["is_valid"] or metrics["solid_count"] < 1 or metrics["volume_mm3"] <= 0:
        raise ValueError("The generated OpenSCAD shape is empty or invalid.")
    if any(v <= 0 for v in dims.values()):
        raise ValueError("The generated OpenSCAD shape has no renderable 3D dimensions.")

    # 3. Parameters & features
    declared_parameters = _declared_parameters(model_code)
    features = _extract_scad_features(model_code, dims)
    features["disconnected_solid_count"] = metrics["disconnected_solid_count"]
    metrics["feature_summary"] = features

    # 4. Verification payload (spec compatibility)
    evidence_path = Path(".cad_validation.json")
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "spec_version": 0,
                "declared_parameters": declared_parameters,
                "results": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 5. Software Rendering + Multi-View Contact Sheet
    review_dir = Path(".review-views")
    review_manifest_payload: dict[str, Any] = {}
    single_render_payload: dict[str, Any] = {}

    if should_write_iso:
        render_path = Path("render.png")
        _write_isometric_artifact(vertices, triangles)
        single_render_payload = {
            "path": "render.png",
            "width": _WIDTH,
            "height": _HEIGHT,
            "image_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
            "image_bytes": render_path.stat().st_size,
        }

    if should_render:
        review_manifest = render_views(
            source_shape=None,
            output_dir=review_dir,
            max_workers=render_workers,
            required_views=required_views,
            vertices=vertices,
            triangles=triangles,
        )
        review_sheet_path = Path(".review-sheet.png")
        sheet_info = build_contact_sheet(review_dir, review_sheet_path)

        views_list = review_manifest.get("views", [])
        review_manifest_payload = {
            "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
            "preview_sha256": preview_sha256,
            "rendered_at": evidence_path.stat().st_mtime_ns,
            "workers": review_manifest.get("workers", render_workers),
            "duration_seconds": review_manifest.get("duration_seconds", 1.0),
            "tessellated_triangles": len(triangles),
            "view_count": review_manifest.get("view_count", len(views_list)),
            "views": views_list,
            "contact_sheet": sheet_info,
            "single_render": single_render_payload,
        }

    # 6. Save .cad_metrics.json
    cache = {
        "schema_version": 2,
        "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
        "preview_sha256": preview_sha256,
        "metrics": metrics,
        "feature_summary": features,
        "spec_version": 0,
        "declared_parameters": declared_parameters,
        "validation_count": 0,
        "validation_results": [],
        "review_manifest": review_manifest_payload,
    }
    metrics_path = Path(".cad_metrics.json")
    metrics_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _parse_settings(argv: list[str]) -> dict[str, Any]:
    raw = argv[1] if len(argv) > 1 else "{}"
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise TypeError("argv[1] must be a JSON object.")
    return parsed


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = list(sys.argv if argv is None else argv)
    settings = _parse_settings(args)
    model_path = Path(str(settings.get("model_path", "model.scad")))
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model_code = model_path.read_text(encoding="utf-8")
    return _run_model(model_code, settings=settings)


if __name__ == "__main__":
    main()
