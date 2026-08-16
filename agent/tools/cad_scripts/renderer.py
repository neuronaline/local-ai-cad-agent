"""Multi-view PNG renderer used inside the bubblewrap sandbox subprocess.

The runner.py script concatenates this module and runs it after a successful
build123d tessellation. Views are rasterised in parallel with a bounded
``ProcessPoolExecutor`` (default four workers) using ``fork`` so the workers
inherit the bubblewrap sandbox. Each view produces a 512x512 PNG that the
runner promotes into the per-review ``views/`` directory.

Public entry points:
- ``VIEWS``: ordered list of canonical view specs used by both the parallel
  rasteriser and the test suite.
- ``render_views(...)``: writes ``views/<view_id>.png`` for every required
  view, returns a manifest payload describing the run.
- ``render_iso(vertices, triangles, output_path)``: legacy isometric helper
  kept for the ``render.png`` artifact so the existing
  ``/api/projects/<name>/render`` endpoint keeps working.

This module is concatenated with ``runner.py`` at execution time, so it
deliberately omits any ``from __future__`` imports — those must be confined
to the head of the concatenated file.
"""
# ruff: noqa: F821 - the bare 'shape' reference is injected by runner.py when concatenated.
import hashlib
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ``shape`` is injected by runner.py when this module is concatenated and run.
# The renderer is also exercised as a standalone module in tests, where the
# shape argument is replaced by ``render_views`` callers.
shape = globals().get("shape")  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Camera / view specifications
# ---------------------------------------------------------------------------

_WIDTH = 512
_HEIGHT = 512
_MARGIN = 36.0
_SHEET_LABEL_HEIGHT = 28
_DEFAULT_VIEW_COUNT = 8
_BACKGROUND = np.array([23, 25, 29], dtype=np.uint8)
_BASE_COLOR = np.array([141.0, 170.0, 255.0])


@dataclass(frozen=True)
class ViewSpec:
    """A single orthographic view used for the structured reviewer."""

    view_id: str
    camera_axis: tuple[float, float, float]
    screen_x_axis: tuple[float, float, float]
    label: str

    def as_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "camera_axis": list(self.camera_axis),
            "screen_x_axis": list(self.screen_x_axis),
            "label": self.label,
        }


def _axis(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.array(tuple(values), dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm <= 0:
        raise ValueError("Camera axis must be non-zero.")
    return tuple(float(v) for v in arr / norm)


def _v(
    view_id: str,
    camera: Iterable[float],
    screen_x: Iterable[float],
    label: str,
) -> ViewSpec:
    return ViewSpec(view_id, _axis(camera), _axis(screen_x), label)


# Canonical eight views: six face-aligned orthographics plus a positive and
# negative isometric for occlusion/contact-sheet purposes.
VIEWS: tuple[ViewSpec, ...] = (
    _v("x_positive", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "+X face"),
    _v("x_negative", (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "-X face"),
    _v("y_positive", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "+Y face"),
    _v("y_negative", (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), "-Y face"),
    _v("z_positive", (0.0, 0.0, 1.0), (1.0, -1.0, 0.0), "+Z face"),
    _v("z_negative", (0.0, 0.0, -1.0), (1.0, 1.0, 0.0), "-Z face"),
    _v(
        "isometric_positive",
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 0.0),
        "iso +",
    ),
    _v(
        "isometric_negative",
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, 0.0),
        "iso -",
    ),
)


# ---------------------------------------------------------------------------
# Rasteriser (pure, immutable inputs, single worker)
# ---------------------------------------------------------------------------


def _project(
    vertices: np.ndarray,
    camera_axis: np.ndarray,
    screen_x_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected (xy_screen, depth) arrays for the orthographic camera."""
    camera_axis = camera_axis / np.linalg.norm(camera_axis)
    screen_x_axis = screen_x_axis / np.linalg.norm(screen_x_axis)
    screen_y_axis = np.cross(camera_axis, screen_x_axis)
    screen_y_axis /= np.linalg.norm(screen_y_axis)
    projected = np.column_stack(
        (
            vertices @ screen_x_axis,
            vertices @ screen_y_axis,
            vertices @ camera_axis,
        )
    )
    return projected[:, :2], projected[:, 2]


def _frame(projected_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    span = np.ptp(projected_xy, axis=0)
    scale = min(
        (_WIDTH - 2 * _MARGIN) / max(span[0], 1e-9),
        (_HEIGHT - 2 * _MARGIN) / max(span[1], 1e-9),
    )
    offset = np.array(
        [
            (_WIDTH - span[0] * scale) / 2 - projected_xy[:, 0].min() * scale,
            (_HEIGHT - span[1] * scale) / 2 - projected_xy[:, 1].min() * scale,
        ]
    )
    return scale, offset, span


def _shade_pixels(
    vertices: np.ndarray,
    triangles: np.ndarray,
    screen_vertices: np.ndarray,
    depths: np.ndarray,
    light: np.ndarray,
) -> np.ndarray:
    pixels = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    pixels[:] = _BACKGROUND
    depth_buffer = np.full((_HEIGHT, _WIDTH), -np.inf)
    for triangle in triangles:
        points = screen_vertices[triangle]
        minimum = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
        maximum = np.minimum(
            np.ceil(points.max(axis=0)).astype(int),
            [_WIDTH - 1, _HEIGHT - 1],
        )
        if np.any(maximum < minimum):
            continue
        x0, y0 = minimum
        x1, y1 = maximum
        grid_y, grid_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        sample_x = grid_x + 0.5
        sample_y = grid_y + 0.5
        p0, p1, p2 = points
        denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (
            p0[1] - p2[1]
        )
        if abs(denominator) < 1e-12:
            continue
        weight0 = (
            (p1[1] - p2[1]) * (sample_x - p2[0])
            + (p2[0] - p1[0]) * (sample_y - p2[1])
        ) / denominator
        weight1 = (
            (p2[1] - p0[1]) * (sample_x - p2[0])
            + (p0[0] - p2[0]) * (sample_y - p2[1])
        ) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= -1e-7) & (weight1 >= -1e-7) & (weight2 >= -1e-7)
        triangle_depths = depths[triangle]
        depth = (
            weight0 * triangle_depths[0]
            + weight1 * triangle_depths[1]
            + weight2 * triangle_depths[2]
        )
        target_depth = depth_buffer[y0 : y1 + 1, x0 : x1 + 1]
        visible = inside & (depth > target_depth)
        if not np.any(visible):
            continue
        world_points = vertices[triangle]
        normal = np.cross(
            world_points[1] - world_points[0], world_points[2] - world_points[0]
        )
        normal_length = np.linalg.norm(normal)
        diffuse = max(
            0.0,
            float(
                np.dot(
                    normal / max(normal_length, 1e-12),
                    light,
                )
            ),
        )
        color = np.clip(_BASE_COLOR * (0.48 + 0.52 * diffuse), 0, 255).astype(
            np.uint8
        )
        target_pixels = pixels[y0 : y1 + 1, x0 : x1 + 1]
        target_depth[visible] = depth[visible]
        target_pixels[visible] = color
    return pixels


def rasterize_view(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    camera_axis: Sequence[float],
    screen_x_axis: Sequence[float],
    light: Sequence[float] | None = None,
) -> np.ndarray:
    """Render one orthographic view of the mesh to an HxWx3 RGB array."""
    light_vec = np.array(light or (0.35, -0.25, 0.9), dtype=np.float64)
    light_vec /= np.linalg.norm(light_vec)
    camera = np.array(camera_axis, dtype=np.float64)
    screen_x = np.array(screen_x_axis, dtype=np.float64)
    projected_xy, depths = _project(vertices, camera, screen_x)
    scale, offset, _span = _frame(projected_xy)
    screen_vertices = projected_xy * scale + offset
    return _shade_pixels(vertices, triangles, screen_vertices, depths, light_vec)


# ---------------------------------------------------------------------------
# Parallel multi-view rasteriser
# ---------------------------------------------------------------------------


def _tessellate(source_shape) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangles) for the given build123d ``shape``."""
    raw_vertices, raw_triangles = source_shape.tessellate(0.1)
    vertices = np.array(
        [[float(p.X), float(p.Y), float(p.Z)] for p in raw_vertices]
    )
    triangles = np.asarray(raw_triangles, dtype=np.int32)
    if not len(vertices) or not len(triangles):
        raise ValueError("Shape tessellation did not produce renderable triangles.")
    return vertices, triangles


def _worker_render(args: tuple[str, dict[str, object]]) -> dict[str, object]:
    """Render a single view in a worker process.

    The worker receives immutable numpy arrays (positions + indices) plus a
    plain-dict view spec. It writes the PNG to a per-worker temp path and
    returns the file's bytes + sha256 + view id so the parent can promote the
    file atomically. Doing the file write here keeps the parent's promotion
    step small and atomic.
    """
    view_id, view_dict = args
    camera_axis = np.array(view_dict["camera_axis"], dtype=np.float64)
    screen_x_axis = np.array(view_dict["screen_x_axis"], dtype=np.float64)
    light = np.array(view_dict.get("light", (0.35, -0.25, 0.9)), dtype=np.float64)
    light /= np.linalg.norm(light)
    vertices_blob = view_dict["vertices"]
    triangles_blob = view_dict["triangles"]
    vertices = np.frombuffer(vertices_blob, dtype=np.float64).reshape(-1, 3).copy()
    triangles = np.frombuffer(triangles_blob, dtype=np.int32).reshape(-1, 3).copy()
    projected_xy, depths = _project(vertices, camera_axis, screen_x_axis)
    scale, offset, _span = _frame(projected_xy)
    screen_vertices = projected_xy * scale + offset
    pixels = _shade_pixels(vertices, triangles, screen_vertices, depths, light)
    image = Image.fromarray(pixels, "RGB")
    buffer = tempfile.NamedTemporaryFile(
        prefix=f"view-{view_id}-", suffix=".png", delete=False
    )
    try:
        buffer.close()
        image.save(buffer.name, "PNG", optimize=True)
        data = Path(buffer.name).read_bytes()
    finally:
        Path(buffer.name).unlink(missing_ok=True)
    return {
        "view_id": view_id,
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _max_workers(requested: int) -> int:
    cpu = max(1, (os.cpu_count() or 1))
    bounded = max(1, int(requested))
    return min(bounded, cpu)


def render_views(
    source_shape=None,
    output_dir: Path | None = None,
    *,
    max_workers: int = 4,
    required_views: int = _DEFAULT_VIEW_COUNT,
    vertices: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
) -> dict[str, object]:
    """Render every required canonical view and persist them under ``output_dir``.

    The caller must supply exactly one of ``source_shape`` (a build123d shape
    to be tessellated) or the pre-tessellated ``vertices``/``triangles`` pair.
    Returns a manifest dictionary describing the rendered views; the caller is
    responsible for atomic promotion of ``output_dir`` into the review tree.
    Raises ``RuntimeError`` if any required view fails to render or its PNG is
    missing/empty — partial output is treated as a build failure.
    """
    if output_dir is None:
        raise ValueError("output_dir is required")
    if (source_shape is None) == (vertices is None or triangles is None):
        raise ValueError(
            "render_views requires either source_shape or pre-tessellated "
            "(vertices, triangles)."
        )
    if vertices is None or triangles is None:
        vertices, triangles = _tessellate(source_shape)
    selected = tuple(VIEWS[: max(1, int(required_views))])
    workers = _max_workers(max_workers)

    # The mesh is small (<= a few MB); pickle it once for every worker.
    view_payloads: list[tuple[str, dict[str, object]]] = [
        (
            spec.view_id,
            {
                "camera_axis": spec.camera_axis,
                "screen_x_axis": spec.screen_x_axis,
                "light": (0.35, -0.25, 0.9),
                "vertices": vertices.tobytes(),
                "triangles": triangles.tobytes(),
            },
        )
        for spec in selected
    ]

    started = time.monotonic()
    results: dict[str, dict[str, object]] = {}
    if workers <= 1 or len(selected) <= 1:
        for payload in view_payloads:
            result = _worker_render(payload)
            results[result["view_id"]] = result
    else:
        # ``fork`` is required here. The renderer runs inside the bubblewrap
        # sandbox (no execve, no writable tmp for spawn's bootstrap); the
        # worker function only depends on numpy + PIL which are already
        # imported in the parent, so the forking cost is bounded. The
        # fork-after-OCP-init safety concern flagged by PEP-687 / CPython
        # issue 84531 is real for general Python 3.12+ code paths, but
        # the renderer is invoked from the sandbox runner that already
        # imports build123d before reaching this block, so any fork-
        # related hazards would already affect the parent process.
        ctx_method = "fork" if "fork" in mp_get_start_methods() else None
        ctx = mp_get_context(ctx_method) if ctx_method else None
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx
        ) if ctx else ProcessPoolExecutor(max_workers=workers)
        with executor as pool:
            futures = [pool.submit(_worker_render, p) for p in view_payloads]
            for future in as_completed(futures):
                result = future.result()
                results[result["view_id"]] = result

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale PNGs from a previous partial render before writing the
    # new views. ``build_contact_sheet`` reads ``sorted(view_dir.glob("*.png"))``
    # so leftover files from an interrupted run would otherwise be included in
    # the next contact sheet — diverging from the manifest's view list. Only
    # files matching ``*.png`` are removed; non-PNG side artifacts (logs,
    # hidden markers) are left untouched.
    if output_dir.exists():
        for stale in output_dir.glob("*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
    view_entries: list[dict[str, object]] = []
    for spec in selected:
        result = results.get(spec.view_id)
        if not result:
            raise RuntimeError(
                f"Review rendering missed required view: {spec.view_id}"
            )
        view_path = output_dir / f"{spec.view_id}.png"
        view_path.write_bytes(result["bytes"])
        if view_path.stat().st_size == 0:
            raise RuntimeError(
                f"Review rendering produced an empty image for {spec.view_id}."
            )
        view_entries.append(
            {
                "view_id": spec.view_id,
                "label": spec.label,
                "camera_axis": list(spec.camera_axis),
                "screen_x_axis": list(spec.screen_x_axis),
                "path": f"views/{spec.view_id}.png",
                "image_sha256": result["sha256"],
                "image_bytes": int(result["size"]),
                "width": _WIDTH,
                "height": _HEIGHT,
                "render_status": "rendered",
            }
        )

    return {
        "view_count": len(selected),
        "workers": workers,
        "duration_seconds": round(time.monotonic() - started, 3),
        "tessellated_triangles": int(len(triangles)),
        "views": view_entries,
    }


def render_iso(vertices: np.ndarray, triangles: np.ndarray, output_path: Path) -> None:
    """Write a single isometric PNG (kept for the legacy ``render.png``)."""
    spec = VIEWS[6]  # isometric_positive
    pixels = rasterize_view(
        vertices,
        triangles,
        camera_axis=spec.camera_axis,
        screen_x_axis=spec.screen_x_axis,
    )
    Image.fromarray(pixels, "RGB").save(output_path, "PNG", optimize=True)


def build_contact_sheet(view_dir: Path, output_path: Path) -> dict[str, object]:
    """Compose a labelled, canonically ordered contact sheet.

    Labels and ``view_order`` let the reviewer reliably associate an observed
    issue with the matching manifest ``view_id`` instead of relying on the
    filesystem's alphabetical order.
    """
    view_dir = Path(view_dir)
    paths_by_id = {path.stem: path for path in view_dir.glob("*.png")}
    ordered_views = [
        (spec, paths_by_id[spec.view_id])
        for spec in VIEWS
        if spec.view_id in paths_by_id
    ]
    if not ordered_views:
        raise RuntimeError("No rendered views available for the contact sheet.")
    columns = 4
    rows = max(1, math.ceil(len(ordered_views) / columns))
    sheet = Image.new(
        "RGB",
        (_WIDTH * columns, (_HEIGHT + _SHEET_LABEL_HEIGHT) * rows),
        tuple(_BACKGROUND.tolist()),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (spec, path) in enumerate(ordered_views):
        x = (index % columns) * _WIDTH
        y = (index // columns) * (_HEIGHT + _SHEET_LABEL_HEIGHT)
        with Image.open(path) as source:
            sheet.paste(source, (x, y))
        draw.text(
            (x + 8, y + _HEIGHT + 6),
            f"{spec.label} ({spec.view_id})",
            fill=(210, 220, 240),
        )
    sheet.save(output_path, "PNG", optimize=True)
    return {
        "path": "review-sheet.png",
        "width": sheet.size[0],
        "height": sheet.size[1],
        "tile_width": _WIDTH,
        "tile_height": _HEIGHT,
        "view_order": [spec.view_id for spec, _path in ordered_views],
        "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "image_bytes": output_path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# multiprocessing context helper (avoid hard import shadowing in sandbox)
# ---------------------------------------------------------------------------


def mp_get_start_methods() -> tuple[str, ...]:
    import multiprocessing

    return tuple(multiprocessing.get_all_start_methods())


def mp_get_context(method: str | None):
    import multiprocessing

    if method is None:
        return multiprocessing.get_context()
    return multiprocessing.get_context(method)


# ---------------------------------------------------------------------------
# Backward-compatible single render.png helper used by runner.py
# ---------------------------------------------------------------------------


def _write_isometric_artifact(vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Render the backward-compatible single isometric PNG."""
    target = Path("render.png")
    render_iso(vertices, triangles, target)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("CAD execution did not produce a render.")
