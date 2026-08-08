import hashlib
import json
import math
from pathlib import Path

from build123d import export_step, export_stl

namespace = {"__name__": "__main__", "__file__": "model.py"}
model_code = Path("model.py").read_text(encoding="utf-8")
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
    raise ValueError("The generated CAD shape has non-finite (NaN/Infinity) geometry.")
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
validation_results: list[dict] = []
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
                # The runner does not know the active attempt id; the host
                # agent stamps it when it consumes ``.cad_validation.json``.
                validation_results = _run_verifiers(
                    requirements, shape, attempt_id=""
                )
    except Exception as error:  # noqa: BLE001 - verifier errors must not block the build
        validation_results = [
            {
                "requirement_id": "",
                "verifier": "spec.parse",
                "status": "unclear",
                "severity": "minor",
                "message": f"Spec could not be evaluated: {type(error).__name__}: {error}",
            }
        ]

preview = Path("preview.stl")
preview_tmp = Path(".preview.stl.tmp")
export_stl(shape, preview_tmp)
preview_tmp.replace(preview)
if not preview.is_file() or preview.stat().st_size == 0:
    raise RuntimeError("CAD execution did not save a usable preview.")

# Persist the bounded validation/evidence artifact (used by the API and tests;
# never returned to the LLM).
evidence_path = Path(".cad_validation.json")
evidence_path.write_text(
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

cache = {
    "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
    "metrics": metrics,
    "spec_version": spec_version,
    "validation_count": len(validation_results),
}
Path(".cad_metrics.json").write_text(json.dumps(cache), encoding="utf-8")
