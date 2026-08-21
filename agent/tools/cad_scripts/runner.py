"""Sandbox-side runner: tessellate, validate, and render the CAD model.

Executed by ``CadTool._execute`` after the host copies ``runner.py`` and
its ``renderer.py`` sibling into the bubblewrap workspace. ``runner.py``
imports ``renderer`` for the canonical :data:`VIEWS`,
:func:`render_views`, and :func:`build_contact_sheet` helpers rather than
relying on source-file concatenation.

Three artifacts are consumed by the host:

- ``preview.stl``: the triangulated mesh (kept for STL compatibility).
- ``render.png``: a single backward-compatible isometric PNG.
- ``.cad_metrics.json``: compact geometry + feature metrics used by the
  structured reviewer. Includes the multi-view ``review_manifest`` produced
  by ``renderer.render_views`` and a deterministic feature summary.
- ``.review-views/<view_id>.png`` and ``.review-sheet.png``: the multi-view
  rasterisation the host promotes into ``.cad-agent/reviews/<sha>/``.

A single ``main(argv)`` entry point reads its settings from a JSON payload on
``argv[1]``. The host (``CadTool._execute``) writes the JSON next to the
script so the runtime contract is a real Python function call instead of an
injected module-level global.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from build123d import export_step, export_stl

# ``renderer`` is a sibling module in the same sandbox workspace; defer the
# import so this file stays importable from the host side (it is consumed
# indirectly by ``agent.tools.cad_tool`` which only reads path constants).
try:
    from renderer import (  # type: ignore[import-not-found]
        _HEIGHT,
        _WIDTH,
        _tessellate,
        _write_isometric_artifact,
        build_contact_sheet,
        render_views,
    )
except ModuleNotFoundError:
    # Host-side import — the rasteriser is unused. Provide stand-ins so the
    # module can still be imported for ``Path(__file__).parent`` access.
    _WIDTH = 512  # type: ignore[assignment]
    _HEIGHT = 512  # type: ignore[assignment]
    _tessellate = None  # type: ignore[assignment]
    _write_isometric_artifact = None  # type: ignore[assignment]
    build_contact_sheet = None  # type: ignore[assignment]
    render_views = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Feature extraction (deterministic, testable kernels only)
# ---------------------------------------------------------------------------


def _bbox_diag(box) -> float:
    return math.sqrt(
        float(box.size.X) ** 2 + float(box.size.Y) ** 2 + float(box.size.Z) ** 2
    )


def _unit(axis) -> tuple[float, float, float]:
    arr = np.array([float(axis.X), float(axis.Y), float(axis.Z)], dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        raise ValueError("Direction vector has zero magnitude.")
    return tuple(float(v) for v in arr / norm)


def _candidate_cut_axes(shape) -> list[dict[str, object]]:
    """Detect cylindrical features that look like cuts/holes.

    The detection uses build123d's face classification: cylinder/circle faces
    whose area is small relative to the bounding box are good cut candidates.
    For each candidate we emit diameter, axis direction, and centroid.
    """
    bbox = shape.bounding_box()
    bbox_diag = _bbox_diag(bbox)
    bbox_surface = 2.0 * (
        float(bbox.size.X) * float(bbox.size.Y)
        + float(bbox.size.Y) * float(bbox.size.Z)
        + float(bbox.size.X) * float(bbox.size.Z)
    )
    if bbox_diag <= 0 or bbox_surface <= 0:
        return []
    candidates: list[dict[str, object]] = []
    try:
        faces = shape.faces()
    except Exception:  # noqa: BLE001 - keep the runner robust to malformed models.
        return []
    for face in faces:
        # ``geom_type`` is a build123d enum; both ``GeomType.CYLINDER`` and
        # ``GeomType.CIRCLE`` expose a ``name`` attribute that matches the
        # documented string identifier.
        geom_type = getattr(face.geom_type, "name", str(face.geom_type))
        if geom_type not in {"CYLINDER", "CIRCLE"}:
            continue
        try:
            radius = float(face.radius)
        except (AttributeError, ValueError):
            continue
        if radius <= 0:
            continue
        try:
            area = float(face.area)
        except (AttributeError, ValueError):
            continue
        # The lateral area of a cylindrical face scales with the radius and
        # the depth of the cut; a small relative area means the feature is
        # likely a localised cut or hole, not the bulk of the part.
        relative = area / bbox_surface
        if relative > 0.5:
            continue
        try:
            center = face.center()
        except Exception:  # noqa: BLE001
            center = None
        try:
            axis = _unit(face.axis_of_rotation.direction)
        except Exception:  # noqa: BLE001, S112 - malformed faces are skipped
            continue
        closed_circles = sum(
            1
            for edge in face.edges()
            if getattr(edge.geom_type, "name", str(edge.geom_type)) == "CIRCLE"
            and bool(edge.is_closed)
        )
        candidates.append(
            {
                "diameter_mm": round(2.0 * radius, 3),
                "axis": [round(axis[0], 4), round(axis[1], 4), round(axis[2], 4)],
                "area_mm2": round(area, 3),
                "center_mm": (
                    [round(center.X, 3), round(center.Y, 3), round(center.Z, 3)]
                    if center is not None
                    else None
                ),
                "is_through_hole": closed_circles >= 2,
            }
        )
    return candidates


def _through_hole_evidence(shape, candidates: list[dict[str, object]]) -> int:
    """Count cylindrical cut faces bounded by two complete circular edges."""
    return sum(1 for candidate in candidates if candidate.get("is_through_hole"))


def _count_disconnected_solids(shape) -> int:
    """Return only solids beyond the first connected component."""
    return max(0, sum(1 for _ in shape.solids()) - 1)


def _validate_parameters(
    namespace: dict[str, object], checks: object
) -> list[dict[str, object]]:
    """Validate explicit model parameters against user-facing bounds."""
    if not isinstance(checks, list):
        return []
    results: list[dict[str, object]] = []
    for check in checks[:20]:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw = namespace.get(name)
        try:
            actual = float(raw)
        except (TypeError, ValueError):
            results.append(
                {
                    "requirement_id": name,
                    "verifier": "parameter",
                    "status": "fail",
                    "severity": "blocking",
                    "message": f"Parameter {name!r} is missing or non-numeric.",
                }
            )
            continue
        tolerance = max(0.0, float(check.get("tolerance", 1e-6) or 0.0))
        failures: list[str] = []
        if "equals" in check and not math.isclose(
            actual,
            float(check["equals"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            failures.append(f"expected {float(check['equals']):g}±{tolerance:g}")
        if "minimum" in check and actual + tolerance < float(check["minimum"]):
            failures.append(f"minimum {float(check['minimum']):g}")
        if "maximum" in check and actual - tolerance > float(check["maximum"]):
            failures.append(f"maximum {float(check['maximum']):g}")
        passed = not failures and math.isfinite(actual)
        results.append(
            {
                "requirement_id": name,
                "verifier": "parameter",
                "status": "pass" if passed else "fail",
                "severity": "info" if passed else "blocking",
                "message": (
                    f"{name}={actual:g}"
                    if passed
                    else f"{name}={actual:g}; " + ", ".join(failures)
                ),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main runner flow
# ---------------------------------------------------------------------------


def _run_model(
    model_code: str, settings: dict[str, object] | None = None
) -> dict[str, object]:
    """Execute ``model_code`` and emit the structured ``.cad_metrics.json`` payload.

    Kept as a function so the helper routines above can be unit-tested in
    isolation by importing the module without triggering the sandbox side
    effects.

    ``settings`` is an optional dict holding the render-phase flags previously
    encoded as module-level globals (``_RENDER_VIEWS``, ``_WRITE_ISOMETRIC``,
    ``_RENDER_WORKERS``, ``_REQUIRED_VIEWS``). When called from
    :func:`main`, ``settings`` is parsed from the JSON payload on ``argv[1]``
    so the runtime contract is a real function argument rather than an
    injected global.

    Resource limits: the bubblewrap sandbox (``agent.sandbox.command``)
    constrains the subprocess via ``prlimit`` -- ``--cpu=timeout+5`` for
    CPU-seconds, ``--fsize``, ``--nofile``, ``--nproc``, and ``--as``. The
    host (``CadTool._execute``) bounds wall-clock time at 120 s with
    ``_stream_with_limit``; the two timers race benignly because the
    runner's ``exec(compile(...))`` blocks the main thread for the
    duration of the build.
    """
    settings = settings or {}
    should_render = bool(settings.get("render_views", True))
    should_write_iso = bool(settings.get("write_isometric", False))
    render_workers = int(settings.get("render_workers", 4) or 4)
    required_views = int(settings.get("required_views", 8) or 8)

    namespace = {"__name__": "__main__", "__file__": "model.py"}
    exec(compile(model_code, "model.py", "exec"), namespace)
    shape = namespace.get("result")
    if shape is None and getattr(namespace.get("part"), "part", None) is not None:
        shape = namespace["part"].part
    if shape is None:
        raise ValueError("model.py must expose the final build123d shape as `result`.")

    box = shape.bounding_box()
    solids = shape.solids()
    volume = float(shape.volume)
    dim_x = float(box.size.X)
    dim_y = float(box.size.Y)
    dim_z = float(box.size.Z)
    if not math.isfinite(volume) or not all(
        math.isfinite(d) for d in (dim_x, dim_y, dim_z)
    ):
        raise ValueError(
            "The generated CAD shape has non-finite (NaN/Infinity) geometry."
        )
    metrics = {
        "solid_count": len(solids),
        "is_valid": bool(shape.is_valid),
        "volume_mm3": round(volume, 3),
        "dimensions_mm": {
            "x": round(dim_x, 3),
            "y": round(dim_y, 3),
            "z": round(dim_z, 3),
        },
    }
    if not metrics["is_valid"] or metrics["solid_count"] < 1 or metrics["volume_mm3"] <= 0:
        raise ValueError("The generated CAD shape is empty or invalid.")
    if any(value <= 0 for value in metrics["dimensions_mm"].values()):
        raise ValueError("The generated CAD shape has no renderable 3D dimensions.")

    # --- Phase 2: deterministic verifiers (if a spec is on disk) -----------------
    validation_results: list[dict] = _validate_parameters(
        namespace, settings.get("parameter_checks")
    )
    spec_version = 0
    spec_path = Path("spec.json")
    if spec_path.is_file():
        try:
            spec_payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            spec_payload = {}
        try:
            from verifiers import run as _run_verifiers  # type: ignore

            if isinstance(spec_payload, dict):
                spec_version = int(spec_payload.get("version", 0) or 0)
                requirements = spec_payload.get("requirements") or []
                if isinstance(requirements, list) and requirements:
                    # ``extend`` (not ``=``) keeps the
                    # ``_validate_parameters`` results from the prior block;
                    # the previous assignment-form overwrote them whenever a
                    # spec was present, which silently dropped
                    # parameter-check failures once a spec landed on disk.
                    validation_results.extend(_run_verifiers(
                        requirements, shape, attempt_id=""
                    ))
        except Exception as error:  # noqa: BLE001 - verifier errors must not block the build
            validation_results.append(
                {
                    "requirement_id": "",
                    "verifier": "spec.parse",
                    "status": "unclear",
                    "severity": "minor",
                    "message": (
                        f"Spec could not be evaluated: {type(error).__name__}: {error}"
                    ),
                }
            )

    blocking_validation = [
        result
        for result in validation_results
        if result.get("status") == "fail"
        and result.get("severity") in {"blocking", "major"}
    ]
    if blocking_validation:
        raise ValueError(
            "Parameter validation failed: "
            + "; ".join(str(result.get("message", "failed")) for result in blocking_validation)
        )

    # --- Phase 3: deterministic feature evidence --------------------------------
    cut_candidates = _candidate_cut_axes(shape)
    metrics["feature_summary"] = {
        "disconnected_solid_count": _count_disconnected_solids(shape),
        "cylindrical_cut_candidates": cut_candidates,
        "through_hole_count": _through_hole_evidence(shape, cut_candidates),
    }

    # --- Phase 4: STL preview export --------------------------------------------
    preview = Path("preview.stl")
    preview_tmp = Path(".preview.stl.tmp")
    try:
        export_stl(shape, preview_tmp)
    except Exception:
        # ``export_stl`` may write partial bytes to ``preview_tmp`` before
        # raising (e.g. an OOM mid-serialisation). Clean the orphan so the
        # next build starts from a clean slate and the operator does not see
        # a confusing half-written STL on disk.
        preview_tmp.unlink(missing_ok=True)
        raise
    preview_tmp.replace(preview)
    if not preview.is_file() or preview.stat().st_size == 0:
        raise RuntimeError("CAD execution did not save a usable preview.")

    preview_sha256 = hashlib.sha256(preview.read_bytes()).hexdigest()

    # --- Phase 5: bounded verification artifact (used by tests / API) ----------
    evidence_path = Path(".cad_validation.json")
    evidence_tmp = Path(".cad_validation.json.tmp")
    evidence_tmp.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "spec_version": spec_version,
                "results": validation_results,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    evidence_tmp.replace(evidence_path)

    # --- Phase 6: backward-compatible single render.png + multi-view rasterisation
    review_dir = Path(".review-views")
    review_manifest_payload: dict[str, object]
    sheet_info_payload: dict[str, object]
    single_render_payload: dict[str, object] = {}
    if should_write_iso:
        # Backward-compatible single isometric PNG (render.png). The renderer
        # tessellates the shape once more for this single image because the
        # cached vertices/triangles are local to ``render_views``.
        iso_vertices, iso_triangles = _tessellate(shape)
        _write_isometric_artifact(iso_vertices, iso_triangles)
        render_path = Path("render.png")
        single_render_payload = {
            "path": "render.png",
            "width": _WIDTH,
            "height": _HEIGHT,
            "image_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
            "image_bytes": render_path.stat().st_size,
        }
    if should_render:
        # Clear any stale review outputs from a previous partial run so the
        # new render doesn't pick up leftover PNGs (build_contact_sheet
        # composes from ``sorted(view_dir.glob("*.png"))`` which would
        # otherwise include files not present in the new manifest).
        review_dir.mkdir(exist_ok=True)
        for stale in review_dir.glob("*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
        review_sheet_tmp = Path(".review-sheet.png")
        if review_sheet_tmp.exists():
            try:
                review_sheet_tmp.unlink()
            except OSError:
                pass
        # Share the tessellation with the iso path when possible: when both
        # ``_WRITE_ISOMETRIC`` and ``_RENDER_VIEWS`` are enabled we already
        # paid for one ``_tessellate`` call, so pass the result through to
        # ``render_views`` instead of tessellating a second time. Without
        # this, every full build re-runs OCP's expensive triangulation.
        if should_write_iso:
            render_vertices, render_triangles = iso_vertices, iso_triangles
            render_source_shape = None
        else:
            render_vertices = render_triangles = None
            render_source_shape = shape
        review_manifest = render_views(
            source_shape=render_source_shape,
            output_dir=review_dir,
            max_workers=render_workers,
            required_views=required_views,
            vertices=render_vertices,
            triangles=render_triangles,
        )
        # build_contact_sheet composes the labelled 4x2 sheet the reviewer will see.
        review_sheet_path = Path(".review-sheet.png")
        sheet_info_payload = build_contact_sheet(review_dir, review_sheet_path)
        review_manifest_payload = {
            "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
            "preview_sha256": preview_sha256,
            "rendered_at": Path(".cad_validation.json").stat().st_mtime_ns,
            "workers": review_manifest["workers"],
            "duration_seconds": review_manifest["duration_seconds"],
            "tessellated_triangles": review_manifest["tessellated_triangles"],
            "view_count": review_manifest["view_count"],
            "views": review_manifest["views"],
            "contact_sheet": sheet_info_payload,
            "single_render": single_render_payload,
        }
    else:
        review_manifest_payload = {}
        sheet_info_payload = {}

    # --- Phase 7: persist the structured metrics/manifest consumed by the host -
    cache = {
        "schema_version": 2,
        "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
        "preview_sha256": preview_sha256,
        "metrics": metrics,
        "feature_summary": metrics["feature_summary"],
        "spec_version": spec_version,
        "validation_count": len(validation_results),
        "validation_results": validation_results,
        "review_manifest": review_manifest_payload,
    }
    metrics_path = Path(".cad_metrics.json")
    metrics_tmp = Path(".cad_metrics.json.tmp")
    try:
        metrics_tmp.write_text(json.dumps(cache), encoding="utf-8")
        metrics_tmp.replace(metrics_path)
    finally:
        metrics_tmp.unlink(missing_ok=True)
    return cache


def _parse_settings(argv: list[str]) -> dict[str, object]:
    """Decode the JSON kwargs payload on ``argv[1]``."""
    raw = argv[1] if len(argv) > 1 else "{}"
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise TypeError("argv[1] must be a JSON object.")
    return parsed


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Sandbox entry point: parse JSON kwargs, build the model, write metrics.

    Called as ``python runner.py '<json kwargs>'`` by
    ``agent.tools.cad_tool`` inside the bubblewrap subprocess. Also
    unit-testable from the repo root by passing ``argv`` directly. Returns
    the cache dict so callers can inspect the structured payload.
    """
    args = list(sys.argv if argv is None else argv)
    settings = _parse_settings(args)
    model_path = Path(str(settings.get("model_path", "model.py")))
    model_code = model_path.read_text(encoding="utf-8")
    return _run_model(model_code, settings=settings)


if __name__ == "__main__":
    main()
