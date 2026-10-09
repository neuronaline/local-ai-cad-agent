"""Multi-view PNG renderer used inside the bubblewrap sandbox subprocess.

The runner / screenshot modules import this one for the canonical ``VIEWS``
list, ``rasterize_view``, ``build_contact_sheet``, ``render_views``, and the
parallel ``render_subset``. Each caller (build / subset) reads its own
arguments from a JSON payload on ``argv[1]`` and writes a separate artifact
manifest, so this module owns only the rendering primitives.

Public entry points:

- :data:`VIEWS`: ordered list of canonical view specs.
- :func:`rasterize_view`: render one orthographic view to an HxWx3 array.
- :func:`render_views`: parallel multi-view rasteriser writing ``*.png`` per
  required view. Returns a manifest payload describing the run.
- :func:`render_subset`: per-request subset re-tessellator used by
  ``screenshot.py``.
- :func:`render_iso`: single legacy isometric helper kept for ``render.png``.
- :func:`build_contact_sheet`: labelled 4x2 composite for the reviewer.
- :func:`build_contact_sheet_subset`: subset-only composite used by
  ``screenshot.py``.

Real tests import this module directly, so there is no runtime
``from __future__`` hack. ``runner.py`` / ``screenshot.py`` ``import renderer``
when they themselves run inside the sandbox, so the source-file structure
mirrors a normal Python package boundary.
"""

import hashlib
import math
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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


def _frame(
    projected_xy: np.ndarray,
    resolution: int = _WIDTH,
) -> tuple[float, np.ndarray, np.ndarray]:
    span = np.ptp(projected_xy, axis=0)
    # Proportional margin scaling matching default 36px on 512px view
    margin = round(resolution * (_MARGIN / _WIDTH))
    scale = min(
        (resolution - 2 * margin) / max(span[0], 1e-9),
        (resolution - 2 * margin) / max(span[1], 1e-9),
    )
    offset = np.array(
        [
            (resolution - span[0] * scale) / 2 - projected_xy[:, 0].min() * scale,
            (resolution - span[1] * scale) / 2 - projected_xy[:, 1].min() * scale,
        ]
    )
    return scale, offset, span


def _shade_pixels(
    vertices: np.ndarray,
    triangles: np.ndarray,
    screen_vertices: np.ndarray,
    depths: np.ndarray,
    light: np.ndarray,
    resolution: int = _WIDTH,
    camera_axis: np.ndarray | None = None,
) -> np.ndarray:
    width = resolution
    height = resolution
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = _BACKGROUND
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    normal_buffer = np.zeros((height, width, 3), dtype=np.float32)

    if len(triangles) == 0:
        return pixels

    # 1. Vectorized face normal calculation
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norm_lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    valid_mask = (norm_lens[:, 0] > 1e-12)
    unit_normals = np.zeros_like(face_normals)
    unit_normals[valid_mask] = face_normals[valid_mask] / norm_lens[valid_mask]

    # 2. Backface culling in orthographic screen space
    if camera_axis is not None:
        cam_norm = camera_axis / max(np.linalg.norm(camera_axis), 1e-12)
        dot_cam = np.sum(unit_normals * cam_norm, axis=1)
        cand_indices = np.where(dot_cam > 1e-5)[0]
    else:
        cand_indices = np.arange(len(triangles))

    # 3. Vectorized diffuse color calculation
    diffuse = np.maximum(0.0, np.dot(unit_normals, light))
    colors = np.clip(_BASE_COLOR * (0.48 + 0.52 * diffuse[:, None]), 0, 255).astype(np.uint8)

    # 4. Rasterize candidate front-facing triangles
    for idx in cand_indices:
        triangle = triangles[idx]
        points = screen_vertices[triangle]
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        if min_xy[0] >= width or min_xy[1] >= height or max_xy[0] < 0 or max_xy[1] < 0:
            continue
        x0 = max(int(np.floor(min_xy[0])), 0)
        y0 = max(int(np.floor(min_xy[1])), 0)
        x1 = min(int(np.ceil(max_xy[0])), width - 1)
        y1 = min(int(np.ceil(max_xy[1])), height - 1)
        if x1 < x0 or y1 < y0:
            continue

        p0, p1, p2 = points
        denominator = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (
            p0[1] - p2[1]
        )
        if abs(denominator) < 1e-12:
            continue

        grid_y, grid_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        sample_x = grid_x + 0.5
        sample_y = grid_y + 0.5
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

        target_depth[visible] = depth[visible]
        pixels[y0 : y1 + 1, x0 : x1 + 1][visible] = colors[idx]
        normal_buffer[y0 : y1 + 1, x0 : x1 + 1][visible] = unit_normals[idx]

    # 5. Image-space feature crease and depth step enhancement
    fg_mask = depth_buffer > -1e20
    if np.any(fg_mask):
        depth_span = max(float(np.ptp(depth_buffer[fg_mask])), 1e-5)
        d_thresh = max(depth_span * 0.015, 0.05)
        d_clean = np.where(fg_mask, depth_buffer, 0.0)

        edge_mask = np.zeros_like(fg_mask)

        # Horizontal adjacent foreground pairs (depth steps & normal angle creases >44 deg)
        fg_x = fg_mask[:, :-1] & fg_mask[:, 1:]
        diff_d_x = np.abs(d_clean[:, 1:] - d_clean[:, :-1])
        dot_n_x = np.sum(normal_buffer[:, :-1] * normal_buffer[:, 1:], axis=-1)
        edge_x = fg_x & ((diff_d_x > d_thresh) | (dot_n_x < 0.72))
        edge_mask[:, :-1] |= edge_x

        # Vertical adjacent foreground pairs (depth steps & normal angle creases >44 deg)
        fg_y = fg_mask[:-1, :] & fg_mask[1:, :]
        diff_d_y = np.abs(d_clean[1:, :] - d_clean[:-1, :])
        dot_n_y = np.sum(normal_buffer[:-1, :] * normal_buffer[1:, :], axis=-1)
        edge_y = fg_y & ((diff_d_y > d_thresh) | (dot_n_y < 0.72))
        edge_mask[:-1, :] |= edge_y

        # Apply crisp dark CAD line color to internal feature edges
        pixels[edge_mask] = np.array([28, 30, 36], dtype=np.uint8)

    return pixels


def rasterize_view(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    camera_axis: Sequence[float],
    screen_x_axis: Sequence[float],
    light: Sequence[float] | None = None,
    resolution: int = _WIDTH,
) -> np.ndarray:
    """Rasterize a 3D mesh into an RGB image buffer from a specified camera view."""
    camera = np.array(camera_axis, dtype=np.float64)
    screen_x = np.array(screen_x_axis, dtype=np.float64)
    if light is not None:
        light_vec = np.array(light, dtype=np.float64)
    else:
        cam_dir = camera / max(np.linalg.norm(camera), 1e-12)
        sx_dir = screen_x / max(np.linalg.norm(screen_x), 1e-12)
        sy_dir = np.cross(cam_dir, sx_dir)
        light_vec = cam_dir + 0.35 * sx_dir + 0.45 * sy_dir
    light_vec /= np.linalg.norm(light_vec)
    projected_xy, depths = _project(vertices, camera, screen_x)
    scale, offset, _span = _frame(projected_xy, resolution=resolution)
    screen_vertices = projected_xy * scale + offset
    return _shade_pixels(
        vertices,
        triangles,
        screen_vertices,
        depths,
        light_vec,
        resolution=resolution,
        camera_axis=camera,
    )


# ---------------------------------------------------------------------------
# Parallel multi-view rasteriser
# ---------------------------------------------------------------------------


def load_stl(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse binary or ASCII STL into (vertices, triangles)."""
    stl_path = Path(stl_path)
    data = stl_path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file too short.")
    num_triangles = int.from_bytes(data[80:84], byteorder="little")
    expected_bin_size = 84 + num_triangles * 50
    if len(data) == expected_bin_size and num_triangles > 0:
        dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])
        records = np.frombuffer(data[84:], dtype=dtype, count=num_triangles)
        tri_coords = np.stack([records["v0"], records["v1"], records["v2"]], axis=1).reshape(-1, 3)
        tri_coords_snapped = np.round(tri_coords, decimals=5)
        unique_verts, inverse_indices = np.unique(tri_coords_snapped, axis=0, return_inverse=True)
        triangles = inverse_indices.reshape(-1, 3).astype(np.int32)
        return unique_verts.astype(np.float64), triangles

    # Fallback to ASCII STL
    text = data.decode("utf-8", errors="replace")
    coords: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "vertex":
            try:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    if not coords or len(coords) % 3 != 0:
        raise ValueError("Invalid STL: no renderable triangles found.")
    tri_coords = np.array(coords, dtype=np.float32)
    tri_coords_snapped = np.round(tri_coords, decimals=5)
    unique_verts, inverse_indices = np.unique(tri_coords_snapped, axis=0, return_inverse=True)
    triangles = inverse_indices.reshape(-1, 3).astype(np.int32)
    return unique_verts.astype(np.float64), triangles


def _tessellate(source_shape) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangles) for an STL path, shape, or (vertices, triangles) tuple."""
    if isinstance(source_shape, tuple) and len(source_shape) == 2 and isinstance(source_shape[0], np.ndarray):
        return source_shape
    if isinstance(source_shape, (str, Path)):
        return load_stl(Path(source_shape))
    if hasattr(source_shape, "tessellate"):
        raw_vertices, raw_triangles = source_shape.tessellate(0.1)
        vertices = np.array([[float(p.X), float(p.Y), float(p.Z)] for p in raw_vertices])
        triangles = np.asarray(raw_triangles, dtype=np.int32)
        if not len(vertices) or not len(triangles):
            raise ValueError("Shape tessellation did not produce renderable triangles.")
        return vertices, triangles
    raise TypeError(f"Cannot tessellate {type(source_shape)}")


def _worker_render(args: tuple[str, dict[str, object]]) -> dict[str, object]:
    """Render a single view in a worker process.

    The worker receives immutable numpy arrays (positions + indices) plus a
    plain-dict view spec. It writes the PNG to a per-worker temp path and
    returns the file's bytes + sha256 + view id so the parent can promote the
    file atomically. Doing the file write here keeps the parent's promotion
    step small and atomic.
    """
    view_id, view_dict = args
    resolution = int(view_dict.get("resolution", _WIDTH) or _WIDTH)
    camera_axis = np.array(view_dict["camera_axis"], dtype=np.float64)
    screen_x_axis = np.array(view_dict["screen_x_axis"], dtype=np.float64)
    light = np.array(view_dict.get("light", (0.35, -0.25, 0.9)), dtype=np.float64)
    light /= np.linalg.norm(light)
    vertices_blob = view_dict["vertices"]
    triangles_blob = view_dict["triangles"]
    vertices = np.frombuffer(vertices_blob, dtype=np.float64).reshape(-1, 3).copy()
    triangles = np.frombuffer(triangles_blob, dtype=np.int32).reshape(-1, 3).copy()
    projected_xy, depths = _project(vertices, camera_axis, screen_x_axis)
    scale, offset, _span = _frame(projected_xy, resolution=resolution)
    screen_vertices = projected_xy * scale + offset
    pixels = _shade_pixels(
        vertices,
        triangles,
        screen_vertices,
        depths,
        light,
        resolution=resolution,
        camera_axis=camera_axis,
    )
    image = Image.fromarray(pixels, "RGB")
    with tempfile.NamedTemporaryFile(
        prefix=f"view-{view_id}-", suffix=".png", delete=False
    ) as buffer:
        buffer_path = Path(buffer.name)
    try:
        image.save(buffer_path, "PNG", compress_level=1)
        data = buffer_path.read_bytes()
    finally:
        buffer_path.unlink(missing_ok=True)
    return {
        "view_id": view_id,
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "width": resolution,
        "height": resolution,
    }


def _max_workers(requested: int) -> int:
    cpu = max(1, (os.cpu_count() or 1))
    bounded = max(1, int(requested))
    return min(bounded, cpu)


_VIEW_ALIASES: dict[str, str] = {
    "all": "all",
    "sheet": "all",
    "contact": "all",
    "contact_sheet": "all",
    "review-sheet": "all",
    "review_sheet": "all",
    "overview": "all",
    "isometric": "isometric_positive",
    "iso": "isometric_positive",
    "iso+": "isometric_positive",
    "iso_pos": "isometric_positive",
    "iso_positive": "isometric_positive",
    "isometric+": "isometric_positive",
    "isometric_positive": "isometric_positive",
    "iso-": "isometric_negative",
    "iso_neg": "isometric_negative",
    "iso_negative": "isometric_negative",
    "isometric-": "isometric_negative",
    "isometric_negative": "isometric_negative",
    "top": "z_positive",
    "+z": "z_positive",
    "bottom": "z_negative",
    "-z": "z_negative",
    "front": "y_negative",
    "-y": "y_negative",
    "back": "y_positive",
    "+y": "y_positive",
    "right": "x_positive",
    "+x": "x_positive",
    "left": "x_negative",
    "-x": "x_negative",
    "x_positive": "x_positive",
    "x_negative": "x_negative",
    "y_positive": "y_positive",
    "y_negative": "y_negative",
    "z_positive": "z_positive",
    "z_negative": "z_negative",
}


def render_views(
    source_shape=None,
    output_dir: Path | None = None,
    *,
    max_workers: int = 4,
    required_views: int = _DEFAULT_VIEW_COUNT,
    vertices: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
    requested_views: Sequence[str] | None = None,
    resolution: int = _WIDTH,
) -> dict[str, object]:
    """Render required canonical views and persist them under ``output_dir``.

    The caller must supply either ``source_shape`` (which can provide
    vertices/triangles) or the pre-tessellated ``vertices``/``triangles`` pair.
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

    if requested_views:
        resolved_ids: set[str] = set()
        use_all = False
        for raw_view in requested_views:
            alias = _VIEW_ALIASES.get(str(raw_view).lower(), str(raw_view))
            if alias == "all":
                use_all = True
                break
            resolved_ids.add(alias)
        if use_all:
            selected = tuple(VIEWS[: max(1, int(required_views))])
        else:
            selected = tuple(spec for spec in VIEWS if spec.view_id in resolved_ids)
            if not selected:
                selected = tuple(VIEWS[: max(1, int(required_views))])
    else:
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
                "resolution": resolution,
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
        # imported in the parent, so the forking cost is bounded.
        import multiprocessing

        ctx_method = (
            "fork" if "fork" in multiprocessing.get_all_start_methods() else None
        )
        ctx = multiprocessing.get_context(ctx_method) if ctx_method else None
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
                "width": resolution,
                "height": resolution,
                "render_status": "rendered",
            }
        )

    return {
        "view_count": len(selected),
        "workers": workers,
        "duration_seconds": round(time.monotonic() - started, 3),
        "tessellated_triangles": len(triangles),
        "views": view_entries,
    }


def render_iso(
    vertices: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    resolution: int = _WIDTH,
) -> None:
    """Write a single isometric PNG (kept for the legacy ``render.png``)."""
    spec = VIEWS[6]  # isometric_positive
    pixels = rasterize_view(
        vertices,
        triangles,
        camera_axis=spec.camera_axis,
        screen_x_axis=spec.screen_x_axis,
        resolution=resolution,
    )
    Image.fromarray(pixels, "RGB").save(output_path, "PNG", compress_level=1)


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

    first_path = ordered_views[0][1]
    with Image.open(first_path) as first_img:
        tile_w, tile_h = first_img.size

    columns = 4
    rows = max(1, math.ceil(len(ordered_views) / columns))
    sheet = Image.new(
        "RGB",
        (tile_w * columns, (tile_h + _SHEET_LABEL_HEIGHT) * rows),
        tuple(_BACKGROUND.tolist()),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (spec, path) in enumerate(ordered_views):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + _SHEET_LABEL_HEIGHT)
        with Image.open(path) as source:
            sheet.paste(source, (x, y))
        draw.text(
            (x + 8, y + tile_h + 6),
            f"{spec.label} ({spec.view_id})",
            fill=(210, 220, 240),
        )
    sheet.save(output_path, "PNG", compress_level=1)
    return {
        "path": "review-sheet.png",
        "width": sheet.size[0],
        "height": sheet.size[1],
        "tile_width": tile_w,
        "tile_height": tile_h,
        "view_order": [spec.view_id for spec, _path in ordered_views],
        "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "image_bytes": output_path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Backward-compatible single render.png helper used by runner.py
# ---------------------------------------------------------------------------


def _write_isometric_artifact(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = _WIDTH,
) -> None:
    """Render the backward-compatible single isometric PNG."""
    target = Path("render.png")
    render_iso(vertices, triangles, target, resolution=resolution)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("CAD execution did not produce a render.")
