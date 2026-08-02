import hashlib
import json
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
metrics = {
    "solid_count": len(solids),
    "is_valid": bool(shape.is_valid),
    "volume_mm3": round(float(shape.volume), 3),
    "dimensions_mm": {
        "x": round(float(box.size.X), 3),
        "y": round(float(box.size.Y), 3),
        "z": round(float(box.size.Z), 3),
    },
}
if not metrics["is_valid"] or metrics["solid_count"] < 1 or metrics["volume_mm3"] <= 0:
    raise ValueError("The generated CAD shape is empty or invalid.")
if any(value <= 0 for value in metrics["dimensions_mm"].values()):
    raise ValueError("The generated CAD shape has no renderable 3D dimensions.")
preview = Path("preview.stl")
preview_tmp = Path(".preview.stl.tmp")
export_stl(shape, preview_tmp)
preview_tmp.replace(preview)
if not preview.is_file() or preview.stat().st_size == 0:
    raise RuntimeError("CAD execution did not save a usable preview.")
cache = {
    "model_sha256": hashlib.sha256(model_code.encode("utf-8")).hexdigest(),
    "metrics": metrics,
}
Path(".cad_metrics.json").write_text(json.dumps(cache), encoding="utf-8")
