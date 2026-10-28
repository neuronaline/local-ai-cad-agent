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


def _analyze_connected_components(
    vertices: np.ndarray, triangles: np.ndarray
) -> dict[str, Any]:
    """Analyze connected triangle components and return structured spatial/topological metrics."""
    if len(triangles) == 0:
        return {
            "disconnected_count": 0,
            "solids_min_z": [],
            "floating_solid_count": 0,
            "floating_solids": [],
            "solid_count": 0,
            "micro_debris_count": 0,
            "trapped_cavity_count": 0,
            "trapped_cavities": [],
            "min_part_gap_mm": None,
            "overlapping_part_count": 0,
        }
    edges_to_tris: dict[tuple[int, int], list[int]] = {}
    for idx, (v0, v1, v2) in enumerate(triangles):
        for e in (
            (min(int(v0), int(v1)), max(int(v0), int(v1))),
            (min(int(v1), int(v2)), max(int(v1), int(v2))),
            (min(int(v2), int(v0)), max(int(v2), int(v0))),
        ):
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

    comp_tris: dict[int, list[int]] = {}
    for i in range(len(triangles)):
        comp_tris.setdefault(find(i), []).append(i)

    if len(vertices) == 0:
        disconnected_count = max(0, len(comp_tris) - 1)
        return {
            "disconnected_count": disconnected_count,
            "solids_min_z": [],
            "floating_solid_count": 0,
            "floating_solids": [],
            "solid_count": max(1, disconnected_count + 1),
            "micro_debris_count": 0,
            "trapped_cavity_count": 0,
            "trapped_cavities": [],
            "min_part_gap_mm": None,
            "overlapping_part_count": 0,
        }

    raw_components: list[dict[str, Any]] = []
    micro_debris_count = 0

    for tri_indices in comp_tris.values():
        c_tri_arr = triangles[tri_indices]
        c_vert_indices = np.unique(c_tri_arr)
        c_verts = vertices[c_vert_indices]

        min_pt = np.min(c_verts, axis=0)
        max_pt = np.max(c_verts, axis=0)
        dx = float(max_pt[0] - min_pt[0])
        dy = float(max_pt[1] - min_pt[1])
        dz = float(max_pt[2] - min_pt[2])
        cx = float((min_pt[0] + max_pt[0]) / 2.0)
        cy = float((min_pt[1] + max_pt[1]) / 2.0)
        cz = float((min_pt[2] + max_pt[2]) / 2.0)

        ref_pt = 0.5 * (min_pt + max_pt)
        cv0 = vertices[c_tri_arr[:, 0]] - ref_pt
        cv1 = vertices[c_tri_arr[:, 1]] - ref_pt
        cv2 = vertices[c_tri_arr[:, 2]] - ref_pt
        c_vol = float(np.sum(np.cross(cv0, cv1) * cv2) / 6.0)

        # Microscopic CSG debris: detached slivers with <0.01 mm³ volume and <=8 triangles
        if abs(c_vol) < 0.01 and len(tri_indices) <= 8:
            micro_debris_count += 1
            continue

        raw_components.append({
            "min_z": float(min_pt[2]),
            "max_z": float(max_pt[2]),
            "min_x": float(min_pt[0]),
            "max_x": float(max_pt[0]),
            "min_y": float(min_pt[1]),
            "max_y": float(max_pt[1]),
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "volume": abs(c_vol),
            "raw_volume": c_vol,
        })

    solid_components: list[dict[str, Any]] = []
    trapped_cavities: list[dict[str, Any]] = []

    # Identify positive solid bodies
    positive_solids = [c for c in raw_components if c["raw_volume"] > 0.1]

    for c in raw_components:
        # A component is a trapped internal cavity if it has negative signed volume
        # (inward normals) AND is geometrically enclosed within a positive solid shell.
        # If the entire mesh has inverted normals, raw_volume < 0 across all solids,
        # so these must be treated as solid components (handled by INVERTED_NORMALS).
        if c["raw_volume"] < -0.1 and positive_solids:
            is_enclosed = any(
                p["min_x"] <= c["min_x"] and c["max_x"] <= p["max_x"]
                and p["min_y"] <= c["min_y"] and c["max_y"] <= p["max_y"]
                and p["min_z"] <= c["min_z"] and c["max_z"] <= p["max_z"]
                for p in positive_solids
            )
            if is_enclosed:
                trapped_cavities.append({
                    "min_z": round(c["min_z"], 3),
                    "max_z": round(c["max_z"], 3),
                    "volume_mm3": round(c["volume"], 3),
                    "bbox_mm": [round(c["dx"], 3), round(c["dy"], 3), round(c["dz"], 3)],
                })
                continue

        solid_components.append(c)

    if not solid_components and comp_tris:
        first_indices = next(iter(comp_tris.values()))
        c_verts = vertices[np.unique(triangles[first_indices])]
        min_pt = np.min(c_verts, axis=0)
        max_pt = np.max(c_verts, axis=0)
        solid_components.append({
            "min_z": float(min_pt[2]),
            "max_z": float(max_pt[2]),
            "min_x": float(min_pt[0]),
            "max_x": float(max_pt[0]),
            "min_y": float(min_pt[1]),
            "max_y": float(max_pt[1]),
            "dx": float(max_pt[0] - min_pt[0]),
            "dy": float(max_pt[1] - min_pt[1]),
            "dz": float(max_pt[2] - min_pt[2]),
            "cx": float((min_pt[0] + max_pt[0]) / 2.0),
            "cy": float((min_pt[1] + max_pt[1]) / 2.0),
            "cz": float((min_pt[2] + max_pt[2]) / 2.0),
            "volume": 0.0,
            "raw_volume": 0.0,
        })

    solid_components.sort(key=lambda c: c["min_z"])
    solid_count = len(solid_components)
    disconnected_count = max(0, solid_count - 1)
    solids_min_z = [round(c["min_z"], 3) for c in solid_components]

    base_z = solids_min_z[0] if solids_min_z else 0.0
    floating_solids: list[dict[str, Any]] = []
    if solid_count > 1:
        for c in solid_components:
            if c["min_z"] > base_z + 0.08:
                floating_solids.append({
                    "min_z": round(c["min_z"], 3),
                    "max_z": round(c["max_z"], 3),
                    "bbox_mm": [round(c["dx"], 3), round(c["dy"], 3), round(c["dz"], 3)],
                    "center_xy": [round(c["cx"], 3), round(c["cy"], 3)],
                    "volume_mm3": round(c["volume"], 3),
                })
    floating_solid_count = len(floating_solids)

    grounded = [c for c in solid_components if c["min_z"] <= base_z + 0.08]
    overlapping_part_count = 0
    min_part_gap_mm: float | None = None
    if len(grounded) >= 2:
        for i in range(len(grounded)):
            for j in range(i + 1, len(grounded)):
                a = grounded[i]
                b = grounded[j]
                overlap_x = (a["min_x"] < b["max_x"]) and (b["min_x"] < a["max_x"])
                overlap_y = (a["min_y"] < b["max_y"]) and (b["min_y"] < a["max_y"])
                if overlap_x and overlap_y:
                    overlapping_part_count += 1
                    gap = 0.0
                else:
                    dx_gap = max(0.0, max(a["min_x"] - b["max_x"], b["min_x"] - a["max_x"]))
                    dy_gap = max(0.0, max(a["min_y"] - b["max_y"], b["min_y"] - a["max_y"]))
                    gap = math.hypot(dx_gap, dy_gap)
                if min_part_gap_mm is None or gap < min_part_gap_mm:
                    min_part_gap_mm = gap

    if min_part_gap_mm is not None:
        min_part_gap_mm = round(min_part_gap_mm, 3)

    return {
        "disconnected_count": disconnected_count,
        "solids_min_z": solids_min_z,
        "floating_solid_count": floating_solid_count,
        "floating_solids": floating_solids,
        "solid_count": solid_count,
        "micro_debris_count": micro_debris_count,
        "trapped_cavity_count": len(trapped_cavities),
        "trapped_cavities": trapped_cavities,
        "min_part_gap_mm": min_part_gap_mm,
        "overlapping_part_count": overlapping_part_count,
    }


def _count_disconnected_solids(triangles: np.ndarray) -> int:
    """Return number of disconnected mesh components beyond the first."""
    analysis = _analyze_connected_components(
        np.empty((0, 3), dtype=np.float64), triangles
    )
    return analysis["disconnected_count"]


def _compute_static_stability(
    com_xy: np.ndarray,
    grounded_xy: np.ndarray,
) -> tuple[bool, float, dict[str, float] | None]:
    """Compute 2D convex support hull stability for the projected XY Center of Mass.

    Returns ``(is_statically_stable, stability_margin_mm, support_footprint_bounds_mm)``
    where ``stability_margin_mm`` is positive when ``com_xy`` lies inside the Z=0 convex
    support hull (distance to nearest tipping edge) and negative when outside.
    """
    if len(grounded_xy) == 0:
        return False, 0.0, None

    s_min = np.min(grounded_xy, axis=0)
    s_max = np.max(grounded_xy, axis=0)
    bounds = {
        "min_x": round(float(s_min[0]), 3),
        "max_x": round(float(s_max[0]), 3),
        "min_y": round(float(s_min[1]), 3),
        "max_y": round(float(s_max[1]), 3),
    }

    pts = np.unique(np.round(grounded_xy, 4), axis=0)
    if len(pts) < 3:
        dist = float(np.min(np.linalg.norm(pts - com_xy[None, :], axis=1)))
        return False, round(-dist, 3), bounds

    # Andrew's monotone chain 2D convex hull (CCW order)
    pts_list = sorted((float(p[0]), float(p[1])) for p in pts)

    def _cross_2d(
        o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts_list:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull_list = lower[:-1] + upper[:-1]
    if len(hull_list) < 3:
        dist = float(np.min(np.linalg.norm(pts - com_xy[None, :], axis=1)))
        return False, round(-dist, 3), bounds

    hull = np.asarray(hull_list, dtype=np.float64)
    # Check polygon area to guard against near-collinear knife-edge contact
    hull_next = np.roll(hull, -1, axis=0)
    poly_area = 0.5 * abs(
        float(np.sum(hull[:, 0] * hull_next[:, 1] - hull_next[:, 0] * hull[:, 1]))
    )
    if poly_area <= 1e-4:
        dist = float(np.min(np.linalg.norm(hull - com_xy[None, :], axis=1)))
        return False, round(-dist, 3), bounds

    edges = hull_next - hull
    edge_lengths = np.linalg.norm(edges, axis=1)
    valid_edges = edge_lengths > 1e-9
    if not np.any(valid_edges):
        return False, 0.0, bounds

    h_start = hull[valid_edges]
    e_vec = edges[valid_edges]
    e_len = edge_lengths[valid_edges]

    rel_q = com_xy[None, :] - h_start
    # Signed inward distance to each directed CCW hull edge: (e_x * rel_y - e_y * rel_x) / |e|
    signed_line_dists = (e_vec[:, 0] * rel_q[:, 1] - e_vec[:, 1] * rel_q[:, 0]) / e_len
    min_inward_dist = float(np.min(signed_line_dists))

    if min_inward_dist >= -1e-5:
        return True, round(max(0.0, min_inward_dist), 3), bounds

    # Point is outside the convex hull: compute exact Euclidean distance to nearest segment
    proj_t = np.clip(
        np.sum(rel_q * e_vec, axis=1) / (e_len * e_len),
        0.0,
        1.0,
    )
    closest_pts = h_start + proj_t[:, None] * e_vec
    min_seg_dist = float(np.min(np.linalg.norm(com_xy[None, :] - closest_pts, axis=1)))
    return False, round(-min_seg_dist, 3), bounds


def mesh_metrics(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Calculate exact bounding box, volume, solid count, and manifold validity."""
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("The generated CAD mesh contains no vertices or triangles.")

    min_pt = np.min(vertices, axis=0)
    max_pt = np.max(vertices, axis=0)
    size = max_pt - min_pt
    dim_x, dim_y, dim_z = float(size[0]), float(size[1]), float(size[2])

    if not all(math.isfinite(d) for d in (dim_x, dim_y, dim_z)):
        raise ValueError("The generated CAD shape has non-finite geometry.")

    # 1. Filter index-degenerate and zero-area collinear triangles
    idx_valid = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    v0_all = vertices[triangles[:, 0]]
    v1_all = vertices[triangles[:, 1]]
    v2_all = vertices[triangles[:, 2]]
    raw_cross = np.cross(v1_all - v0_all, v2_all - v0_all)
    norms_all = np.linalg.norm(raw_cross, axis=1)
    area_valid = norms_all > 1e-12
    valid_mask = idx_valid & area_valid

    non_deg_triangles = triangles[valid_mask]
    degenerate_triangle_count = int(len(triangles) - len(non_deg_triangles))

    if len(non_deg_triangles) == 0:
        raise ValueError("The generated CAD mesh contains no non-degenerate triangles.")

    # Exact signed volume and volumetric Center of Mass via tetrahedron sum
    v0 = vertices[non_deg_triangles[:, 0]]
    v1 = vertices[non_deg_triangles[:, 1]]
    v2 = vertices[non_deg_triangles[:, 2]]
    ref_pt = 0.5 * (min_pt + max_pt)
    u0 = v0 - ref_pt
    u1 = v1 - ref_pt
    u2 = v2 - ref_pt
    tet_vols = np.sum(np.cross(u0, u1) * u2, axis=1) / 6.0
    raw_volume = float(np.sum(tet_vols))
    volume = abs(raw_volume)
    if abs(raw_volume) > 1e-9:
        tet_centroids_rel = (u0 + u1 + u2) * 0.25
        com_vec = ref_pt + np.sum(tet_vols[:, None] * tet_centroids_rel, axis=0) / raw_volume
    else:
        com_vec = ref_pt
    center_of_mass_mm = [
        round(float(com_vec[0]), 3),
        round(float(com_vec[1]), 3),
        round(float(com_vec[2]), 3),
    ]

    # Watertight Manifold Edge-sharing Check & Directed Half-Edge Winding
    edges: dict[tuple[int, int], int] = {}
    directed_edges: dict[tuple[int, int], int] = {}
    for tri in non_deg_triangles:
        t0, t1, t2 = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((t0, t1), (t1, t2), (t2, t0)):
            directed_edges[(u, v)] = directed_edges.get((u, v), 0) + 1
            undirected = (min(u, v), max(u, v))
            edges[undirected] = edges.get(undirected, 0) + 1

    boundary_edges = sum(1 for count in edges.values() if count == 1)
    multi_face_edges = sum(1 for count in edges.values() if count > 2)

    inconsistent_winding_edges = 0
    for (u, v), count in edges.items():
        if count == 2:
            fwd = directed_edges.get((u, v), 0)
            rev = directed_edges.get((v, u), 0)
            if fwd != 1 or rev != 1:
                inconsistent_winding_edges += 1

    comp_analysis = _analyze_connected_components(vertices, non_deg_triangles)
    solid_count = comp_analysis["solid_count"]
    disconnected_count = comp_analysis["disconnected_count"]
    floating_solid_count = comp_analysis["floating_solid_count"]
    floating_solids = comp_analysis["floating_solids"]
    solids_min_z = comp_analysis["solids_min_z"]
    micro_debris_count = comp_analysis["micro_debris_count"]
    trapped_cavity_count = comp_analysis["trapped_cavity_count"]
    trapped_cavities = comp_analysis["trapped_cavities"]
    min_part_gap_mm = comp_analysis["min_part_gap_mm"]
    overlapping_part_count = comp_analysis["overlapping_part_count"]

    is_valid = volume > 0 and dim_x > 0 and dim_y > 0 and dim_z > 0

    # Printability telemetry (surface area, bed contact, overhangs, stability)
    e1 = v1 - v0
    e2 = v2 - v0
    tri_cross = np.cross(e1, e2)
    norms = np.linalg.norm(tri_cross, axis=1)
    tri_areas = 0.5 * norms
    surface_area = float(np.sum(tri_areas))

    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    nz = tri_cross[:, 2] / safe_norms
    max_z_tri = np.maximum.reduce([v0[:, 2], v1[:, 2], v2[:, 2]])

    bed_mask = (max_z_tri <= min_pt[2] + 0.20) & (nz <= -0.95)
    bed_contact_area = float(np.sum(tri_areas[bed_mask]))
    footprint_area = max(dim_x * dim_y, 1e-6)
    bed_contact_ratio = float(min(1.0, bed_contact_area / footprint_area))

    elevated_mask = max_z_tri > (min_pt[2] + 0.40)
    overhang_45_mask = elevated_mask & (nz < -0.70710678)
    overhang_60_mask = elevated_mask & (nz < -0.86602540)
    overhang_area_45 = float(np.sum(tri_areas[overhang_45_mask]))
    overhang_area_60 = float(np.sum(tri_areas[overhang_60_mask]))
    overhang_ratio = float(min(1.0, overhang_area_45 / max(surface_area, 1e-6)))

    min_horizontal_dim = max(min(dim_x, dim_y), 0.1)
    slenderness_ratio = float(dim_z / min_horizontal_dim)

    active_vert_indices = np.unique(non_deg_triangles)
    active_verts = vertices[active_vert_indices]
    grounded_xy = active_verts[active_verts[:, 2] <= min_pt[2] + 0.20, :2]
    is_statically_stable, stability_margin_mm, support_footprint_bounds_mm = (
        _compute_static_stability(com_vec[:2], grounded_xy)
    )

    needs_support = bool(overhang_area_45 >= 15.0 or overhang_ratio >= 0.03)
    needs_brim = bool(
        slenderness_ratio >= 2.5
        or not is_statically_stable
        or (bed_contact_area < 150.0 and dim_z >= 20.0)
        or (bed_contact_ratio < 0.08 and dim_z >= 10.0)
    )

    printability = {
        "surface_area_mm2": round(surface_area, 2),
        "bed_contact_area_mm2": round(bed_contact_area, 2),
        "bed_contact_ratio": round(bed_contact_ratio, 4),
        "overhang_area_45deg_mm2": round(overhang_area_45, 2),
        "overhang_area_60deg_mm2": round(overhang_area_60, 2),
        "overhang_ratio": round(overhang_ratio, 4),
        "slenderness_ratio": round(slenderness_ratio, 2),
        "center_of_mass_mm": center_of_mass_mm,
        "is_statically_stable": is_statically_stable,
        "stability_margin_mm": stability_margin_mm,
        "support_footprint_bounds_mm": support_footprint_bounds_mm,
        "needs_support": needs_support,
        "needs_brim": needs_brim,
    }

    return {
        "solid_count": solid_count,
        "is_valid": is_valid,
        "volume_mm3": round(volume, 3),
        "raw_volume": round(raw_volume, 3),
        "dimensions_mm": {
            "x": round(dim_x, 3),
            "y": round(dim_y, 3),
            "z": round(dim_z, 3),
        },
        "disconnected_solid_count": disconnected_count,
        "floating_solid_count": floating_solid_count,
        "floating_solids": floating_solids,
        "solids_min_z": solids_min_z,
        "micro_debris_count": micro_debris_count,
        "trapped_cavity_count": trapped_cavity_count,
        "trapped_cavities": trapped_cavities,
        "min_part_gap_mm": min_part_gap_mm,
        "overlapping_part_count": overlapping_part_count,
        "boundary_edges": boundary_edges,
        "multi_face_edges": multi_face_edges,
        "inconsistent_winding_edges": inconsistent_winding_edges,
        "degenerate_triangle_count": degenerate_triangle_count,
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "non_degenerate_triangle_count": len(non_deg_triangles),
        "printability": printability,
    }


_mesh_metrics = mesh_metrics


# ---------------------------------------------------------------------------
# Main model runner
# ---------------------------------------------------------------------------


def _run_model(
    model_code: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile model.scad with OpenSCAD, validate geometry, and render evidence."""
    settings = settings or {}
    model_code = model_code.replace("\r\n", "\n").replace("\r", "\n")
    model_path = Path(str(settings.get("model_path", "model.scad")))
    model_path.write_text(model_code, encoding="utf-8")

    should_render = bool(settings.get("render_views", True))
    should_write_iso = bool(settings.get("write_isometric", False))
    render_workers = int(settings.get("render_workers", 4) or 4)
    required_views = int(settings.get("required_views", 8) or 8)
    resolution = int(settings.get("resolution", 512) or 512)
    enable_manifold = bool(settings.get("enable_manifold", False))

    # 1. Compile model.scad to binary STL (headless CGAL/CSG evaluation)
    preview_stl = Path("preview.stl")
    needs_compile = not preview_stl.is_file() or preview_stl.stat().st_size == 0
    if needs_compile:
        compile_cmd = [
            "openscad",
            "-o",
            str(preview_stl),
            "--export-format",
            "binstl",
        ]
        if enable_manifold:
            compile_cmd.append("--enable=manifold")
        compile_cmd.append(str(model_path))
        try:
            proc = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=_OPENSCAD_TIMEOUT_SECONDS,
                check=False,
            )
            ret, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            uncommented = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", model_code)
            culprits: list[str] = []
            if re.search(r"\bminkowski\s*\(", uncommented):
                culprits.append("minkowski()")
            if re.search(r"\blinear_extrude\s*\([^)]*\btwist\s*=", uncommented):
                culprits.append("twisted linear_extrude() (keep $fn <= 36 and slices <= turns*18)")
            if re.search(r"\bhull\s*\(", uncommented) and re.search(r"\bfor\s*\(", uncommented):
                culprits.append("hull() inside for() loops")
            hint_suffix = (
                f" Likely culprits in model.scad: {', '.join(culprits)}."
                if culprits
                else " Reduce $fn (<= 64, or <= 36 for threads) and simplify boolean differences."
            )
            raise RuntimeError(
                f"OpenSCAD compilation timed out after {_OPENSCAD_TIMEOUT_SECONDS} seconds.{hint_suffix}"
            ) from None
        except (subprocess.SubprocessError, OSError) as exc:
            ret, stdout, stderr = 1, "", str(exc)

        if ret != 0 or not preview_stl.is_file() or preview_stl.stat().st_size == 0:
            err_detail = (stderr or stdout).strip()
            lines = [
                line
                for line in err_detail.splitlines()
                if any(k in line.lower() for k in ("error", "warning", "syntax", "can't", "cgal", "expr:"))
            ]
            summary = "\n".join(lines) if lines else err_detail[-1000:]
            raise RuntimeError(f"OpenSCAD execution failed:\n{summary or 'OpenSCAD produced no output or an empty STL.'}")

    compiler_warnings: list[dict[str, Any]] = []
    if needs_compile and stderr:
        for line in stderr.splitlines():
            line_s = line.strip()
            if "warning:" in line_s.lower():
                m_line = re.search(r"line\s+(\d+)", line_s, re.IGNORECASE)
                lineno = int(m_line.group(1)) if m_line else None
                compiler_warnings.append({"message": line_s, "line": lineno})

    preview_bytes = preview_stl.read_bytes()
    preview_sha256 = hashlib.sha256(preview_bytes).hexdigest()

    # 2. Parse STL mesh and compute geometry metrics
    vertices, triangles = load_stl(preview_stl)
    metrics = _mesh_metrics(vertices, triangles)
    if compiler_warnings:
        metrics["compiler_warnings"] = compiler_warnings
    dims = metrics["dimensions_mm"]

    if not metrics["is_valid"] or metrics["solid_count"] < 1 or metrics["volume_mm3"] <= 0:
        raise ValueError("The generated OpenSCAD shape is empty or invalid.")
    if any(v <= 0 for v in dims.values()):
        raise ValueError("The generated OpenSCAD shape has no renderable 3D dimensions.")

    # 3. Parameters
    declared_parameters = _declared_parameters(model_code)
    features = {
        "disconnected_solid_count": metrics["disconnected_solid_count"],
    }

    # 4. Software Rendering + Multi-View Contact Sheet
    review_dir = Path(".review-views")
    review_manifest_payload: dict[str, Any] = {}
    single_render_payload: dict[str, Any] = {}
    requested_views = settings.get("requested_views")

    if should_render:
        review_manifest = render_views(
            source_shape=None,
            output_dir=review_dir,
            max_workers=render_workers,
            required_views=required_views,
            vertices=vertices,
            triangles=triangles,
            requested_views=requested_views,
            resolution=resolution,
        )
        review_sheet_path = Path(".review-sheet.png")
        sheet_info = build_contact_sheet(review_dir, review_sheet_path)

        views_list = review_manifest.get("views", [])
        if should_write_iso:
            render_path = Path("render.png")
            iso_view_path = review_dir / "isometric_positive.png"
            if iso_view_path.is_file() and iso_view_path.stat().st_size > 0:
                shutil.copyfile(iso_view_path, render_path)
            else:
                _write_isometric_artifact(vertices, triangles, resolution=resolution)
            single_render_payload = {
                "path": "render.png",
                "width": resolution,
                "height": resolution,
                "image_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
                "image_bytes": render_path.stat().st_size,
            }

        review_manifest_payload = {
            "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
            "preview_sha256": preview_sha256,
            "rendered_at": preview_stl.stat().st_mtime_ns if preview_stl.exists() else 0,
            "workers": review_manifest.get("workers", render_workers),
            "duration_seconds": review_manifest.get("duration_seconds", 1.0),
            "tessellated_triangles": len(triangles),
            "view_count": review_manifest.get("view_count", len(views_list)),
            "views": views_list,
            "contact_sheet": sheet_info,
            "single_render": single_render_payload,
        }
    elif should_write_iso:
        render_path = Path("render.png")
        _write_isometric_artifact(vertices, triangles, resolution=resolution)
        single_render_payload = {
            "path": "render.png",
            "width": resolution,
            "height": resolution,
            "image_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
            "image_bytes": render_path.stat().st_size,
        }

    # 5. Save .cad_metrics.json
    cache = {
        "schema_version": 2,
        "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
        "preview_sha256": preview_sha256,
        "metrics": metrics,
        "feature_summary": features,
        "spec_version": 0,
        "declared_parameters": declared_parameters,
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
