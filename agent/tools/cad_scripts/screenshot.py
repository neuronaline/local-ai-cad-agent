"""Sandbox-side subset rasteriser for ``cad_screenshot``.

Imports ``renderer`` directly for the canonical :data:`VIEWS`,
:func:`rasterize_view`, and :func:`build_contact_sheet` primitives. Re-tessellates
``shape`` once with the requested quality and writes only the requested view
subset to ``output_dir``.

Public entry points:

- ``SUBSET_VIEWS`` — every legal ``view_id`` the orchestrator may request.
- ``QUALITY_TOLERANCES`` — pixel size + tessellation tol per ``quality``.
- ``render_subset(...)`` — writes ``views/<view_id>.png`` for each requested
  view and returns a manifest payload describing the run.
- ``build_contact_sheet_subset(...)`` — composes a sheet from the rendered
  subset in canonical order.
- ``main(argv)`` — JSON-kwarg CLI entry point that the sandbox subprocess invokes.
"""
# ``renderer`` and ``main`` are siblings in the same sandbox workspace; a normal
# absolute-import style keeps the suite testable from the repo root. The
# import is wrapped so the host side (``agent.tools.cad_screenshot_tool``)
# can still pull ``SUBSET_VIEWS`` / ``QUALITY_TOLERANCES`` from this module
# without needing the renderer side of the workspace on its import path.
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from renderer import (  # type: ignore[import-not-found]  # sandbox sibling import
        VIEWS,
        _tessellate,
        rasterize_view,
    )
except ModuleNotFoundError:
    # Package import on the host. Inside the sandbox the sibling import above
    # succeeds because both scripts are copied into one workspace.
    from .renderer import VIEWS, _tessellate, rasterize_view

# ---------------------------------------------------------------------------
# Subset + quality configuration
# ---------------------------------------------------------------------------

# Canonical view ids the orchestrator is allowed to request.
SUBSET_VIEWS: tuple[str, ...] = (
    "x_positive",
    "x_negative",
    "y_positive",
    "y_negative",
    "z_positive",
    "z_negative",
    "isometric_positive",
    "isometric_negative",
)

# Pixel size + tessellation tolerance per quality tier.
# ``low``    — 256x256, coarse tol (fast iteration / overview)
# ``standard`` — 512x512, 0.1 tol (matches cad_build_and_verify default)
# ``high``   — 1024x1024, 0.05 tol (final verification / zoom-in)
QUALITY_TOLERANCES: dict[str, dict[str, float]] = {
    "low": {"pixel": 256, "tol": 0.3},
    "standard": {"pixel": 512, "tol": 0.1},
    "high": {"pixel": 1024, "tol": 0.05},
}


def _pixel_size(quality: str) -> int:
    spec = QUALITY_TOLERANCES.get(quality)
    if spec is None:
        raise ValueError(
            f"Unknown quality tier: {quality!r}. "
            f"Expected one of {sorted(QUALITY_TOLERANCES)}."
        )
    return int(spec["pixel"])


def _tolerance(quality: str) -> float:
    spec = QUALITY_TOLERANCES.get(quality)
    if spec is None:
        raise ValueError(
            f"Unknown quality tier: {quality!r}. "
            f"Expected one of {sorted(QUALITY_TOLERANCES)}."
        )
    return float(spec["tol"])


def _view_spec_by_id(view_id: str):
    """Resolve a canonical view id to its ``ViewSpec``.

    The function looks up ``VIEWS`` (defined by the concatenated renderer
    module) and returns the matching ``ViewSpec``. Raises ``ValueError`` for
    unknown ids so the orchestrator can surface a clean schema violation.
    """
    for spec in VIEWS:
        if spec.view_id == view_id:
            return spec
    raise ValueError(
        f"Unknown view id: {view_id!r}. "
        f"Expected one of {[s.view_id for s in VIEWS]}."
    )


# ---------------------------------------------------------------------------
# Subset renderer
# ---------------------------------------------------------------------------


def render_subset(
    shape,
    output_dir: Path,
    requested_views: tuple[str, ...],
    quality: str = "standard",
) -> dict[str, object]:
    """Render a subset of views and persist them under ``output_dir``.

    Parameters
    ----------
    shape:
        A build123d shape object. Tessellation runs once with the
        quality-tier tolerance, then ``rasterize_view`` rasterises only the
        requested subset.
    output_dir:
        Target directory; cleared of any stale ``*.png`` files before the new
        views are written so the manifest cannot pick up leftovers.
    requested_views:
        Tuple of view ids. Must be non-empty and every id must exist in
        ``SUBSET_VIEWS``. Subset rendering only — no implicit canonical
        eight fallback (the orchestrator expands the empty/None case).

    Returns
    -------
    dict
        ``view_count``, ``pixel_size``, ``tolerance``, ``duration_seconds``,
        ``tessellated_triangles``, ``views`` (manifest entries).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel = _pixel_size(quality)
    tolerance = _tolerance(quality)
    if not requested_views:
        raise ValueError("render_subset requires at least one requested view.")
    selected_specs = []
    for view_id in requested_views:
        if view_id not in SUBSET_VIEWS:
            raise ValueError(
                f"Unknown view id: {view_id!r}. "
                f"Expected one of {list(SUBSET_VIEWS)}."
            )
        selected_specs.append(_view_spec_by_id(view_id))
    started = time.monotonic()
    raw_vertices, raw_triangles = shape.tessellate(tolerance)
    vertices = np.array(
        [[float(p.X), float(p.Y), float(p.Z)] for p in raw_vertices]
    )
    triangles = np.asarray(raw_triangles, dtype=np.int32)
    if not len(vertices) or not len(triangles):
        raise ValueError("Shape tessellation did not produce renderable triangles.")

    # Clear stale PNGs to avoid build_contact_sheet picking them up if the
    # caller asks for a sheet later. The subset manifest will list only the
    # views that were rendered in this call.
    for stale in output_dir.glob("*.png"):
        try:
            stale.unlink()
        except OSError:
            pass

    view_entries: list[dict[str, object]] = []
    for spec in selected_specs:
        pixels = rasterize_view(
            vertices,
            triangles,
            camera_axis=spec.camera_axis,
            screen_x_axis=spec.screen_x_axis,
        )
        # Scale to the requested pixel size while preserving aspect ratio.
        if pixels.shape[0] != pixel or pixels.shape[1] != pixel:
            image = Image.fromarray(pixels, "RGB").resize(
                (pixel, pixel), Image.LANCZOS
            )
            pixels = np.asarray(image)
        view_path = output_dir / f"{spec.view_id}.png"
        image = Image.fromarray(pixels, "RGB")
        image.save(view_path, "PNG", optimize=True)
        data = view_path.read_bytes()
        view_entries.append(
            {
                "view_id": spec.view_id,
                "label": spec.label,
                "camera_axis": list(spec.camera_axis),
                "screen_x_axis": list(spec.screen_x_axis),
                "path": f"views/{spec.view_id}.png",
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "image_bytes": len(data),
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "render_status": "rendered",
            }
        )
    return {
        "view_count": len(selected_specs),
        "pixel_size": pixel,
        "tolerance": tolerance,
        "duration_seconds": round(time.monotonic() - started, 3),
        "tessellated_triangles": len(triangles),
        "views": view_entries,
    }


def build_contact_sheet_subset(
    view_dir: Path,
    output_path: Path,
    requested_views: tuple[str, ...] = SUBSET_VIEWS,
) -> dict[str, object]:
    """Compose a labelled contact sheet from the rendered subset.

    Mirrors ``renderer.build_contact_sheet`` but only includes tiles for
    view_ids in ``requested_views`` (in canonical order) so the reviewer's
    attention matches the orchestrator's narrowed scope.
    """
    view_dir = Path(view_dir)
    paths_by_id = {path.stem: path for path in view_dir.glob("*.png")}
    ordered_views = [
        (spec, paths_by_id[spec.view_id])
        for spec in VIEWS
        if spec.view_id in paths_by_id and spec.view_id in requested_views
    ]
    if not ordered_views:
        raise RuntimeError("No rendered views available for the contact sheet.")
    columns = min(4, len(ordered_views))
    rows = max(1, math.ceil(len(ordered_views) / columns))
    # Pick the per-tile size from the first available PNG; subset renders
    # may differ per call so the sheet must match the latest pixel size.
    first_image_path = ordered_views[0][1]
    with Image.open(first_image_path) as first:
        tile_w, tile_h = first.size
    sheet_label_height = 28
    sheet = Image.new(
        "RGB",
        (tile_w * columns, (tile_h + sheet_label_height) * rows),
        (23, 25, 29),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (spec, path) in enumerate(ordered_views):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + sheet_label_height)
        with Image.open(path) as source:
            sheet.paste(source, (x, y))
        draw.text(
            (x + 8, y + tile_h + 6),
            f"{spec.label} ({spec.view_id})",
            fill=(210, 220, 240),
        )
    sheet.save(output_path, "PNG", optimize=True)
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
# Main entry point — invoked by ``agent.tools.cad_screenshot_tool``.
# ``renderer.py`` is a sibling module in the same sandbox workspace, imported
# once at module load time.
# ---------------------------------------------------------------------------

# ``ImageDraw`` is only needed by ``build_contact_sheet_subset``; defer the
# import to keep the top-of-file PIL surface narrow.
try:
    from PIL import ImageDraw
except ImportError:  # pragma: no cover - Pillow is required by the runner anyway.
    ImageDraw = None  # type: ignore[assignment]


def _resolve_shape(model_path: Path) -> object:
    """Exec ``model.py`` and return the build123d ``result`` shape."""
    model_code = model_path.read_text(encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": "model.py"}
    exec(compile(model_code, "model.py", "exec"), namespace)
    shape = namespace.get("result")
    if shape is None and getattr(namespace.get("part"), "part", None) is not None:
        shape = namespace["part"].part
    if shape is None:
        raise ValueError(
            "model.py must expose the final build123d shape as `result`."
        )
    return shape


def _run_subset_pipeline(
    model_path: Path,
    requested_views: tuple[str, ...],
    output_dir: Path,
    quality: str,
    contact_sheet: bool,
) -> dict[str, object]:
    """Exec ``model.py`` and rasterise only the requested subset.

    Returns a manifest dict that the orchestrator validates + promotes into
    ``.cad-agent/reviews/<sha>/``.
    """
    shape = _resolve_shape(model_path)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    payload = render_subset(shape, views_dir, requested_views, quality=quality)
    sheet_info: dict[str, object] | None = None
    if contact_sheet:
        sheet_path = output_dir / "review-sheet.png"
        sheet_info = build_contact_sheet_subset(
            views_dir, sheet_path, requested_views
        )
    single_render = next(
        (
            entry
            for entry in payload["views"]
            if entry.get("view_id") == "isometric_positive"
        ),
        None,
    )
    preview_path = Path("preview.stl")
    preview_sha256 = (
        hashlib.sha256(preview_path.read_bytes()).hexdigest()
        if preview_path.is_file()
        else ""
    )
    model_code = model_path.read_text(encoding="utf-8")
    return {
        "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
        "preview_sha256": preview_sha256,
        "views": payload["views"],
        "requested_views": list(requested_views),
        "pixel_size": payload["pixel_size"],
        "tolerance": payload["tolerance"],
        "duration_seconds": payload["duration_seconds"],
        "tessellated_triangles": payload["tessellated_triangles"],
        "view_count": payload["view_count"],
        "contact_sheet": sheet_info,
        # A standalone screenshot has no legacy render.png. When it includes
        # the canonical isometric view, that image is equivalent visual
        # evidence and lets cad_review use its auto-screenshot fallback.
        "single_render": single_render,
        "quality": quality,
    }


def _parse_json_argv(argv: list[str]) -> dict[str, object]:
    import json

    raw = argv[1] if len(argv) > 1 else "{}"
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise TypeError("argv[1] must be a JSON object.")
    return parsed


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Sandbox entry point: parse JSON kwargs, render subset, write manifest.

    Called as ``python screenshot.py '<json kwargs>'`` by
    ``agent.tools.cad_screenshot_tool`` inside the bubblewrap subprocess. Also
    unit-testable from the repo root by passing ``argv`` directly.
    """
    import json

    args = list(sys.argv if argv is None else argv)
    kwargs = _parse_json_argv(args)
    model_path = Path(str(kwargs.get("model_path", "model.py")))
    output_dir = Path(str(kwargs.get("output_dir", ".screenshot-staging")))
    requested_views = tuple(kwargs.get("requested_views") or ())
    quality = str(kwargs.get("quality", "standard"))
    contact_sheet = bool(kwargs.get("contact_sheet", True))
    payload = _run_subset_pipeline(
        model_path=model_path,
        requested_views=requested_views,
        output_dir=output_dir,
        quality=quality,
        contact_sheet=contact_sheet,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / ".screenshot_manifest.json"
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(manifest_path)
    return payload


if __name__ == "__main__":
    main()
