"""CAD model mesh and format conversion utilities."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil
import subprocess

import tempfile

from agent.tools.cad_scripts.renderer import load_stl

SUPPORTED_EXPORT_FORMATS = frozenset({"stl", "scad", "3mf", "obj", "amf", "off", "csg"})

EXPORT_MIME_TYPES: dict[str, str] = {
    "stl": "model/stl",
    "scad": "text/x-scad",
    "3mf": "model/3mf",
    "obj": "model/obj",
    "amf": "application/x-amf",
    "off": "text/plain",
    "csg": "text/plain",
}


def find_openscad() -> str | None:
    """Locate the openscad executable on PATH or in local .venv."""
    direct = shutil.which("openscad")
    if direct:
        return direct
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (
        repo_root / ".venv" / "bin" / "openscad",
        repo_root / ".venv" / "openscad",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def stl_to_obj(stl_path: Path) -> str:
    """Convert an STL file (binary or ASCII) into a Wavefront OBJ string."""
    vertices, triangles = load_stl(stl_path)
    buf = io.StringIO()
    buf.write("# OBJ export from Local AI CAD Agent\n")
    for v in vertices:
        buf.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for t in triangles:
        # OBJ indices are 1-based
        buf.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return buf.getvalue()


def export_project_model(
    project_dir: Path,
    fmt: str,
    *,
    timeout_seconds: int = 30,
) -> Path:
    """Export the project model to the specified format with caching.

    Returns the path to the exported file on disk.
    Raises ValueError for unsupported formats or empty models.
    Raises FileNotFoundError for missing model files.
    Raises RuntimeError if external conversion fails.
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_EXPORT_FORMATS))
        raise ValueError(f"Unsupported export format '{fmt}'. Supported formats: {supported}")

    model_path = project_dir / "model.scad"
    if not model_path.is_file():
        raise FileNotFoundError("No model source file found.")
    if model_path.stat().st_size == 0:
        raise ValueError("The model source file is empty.")

    if fmt == "scad":
        return model_path

    preview_path = project_dir / "preview.stl"
    if fmt in {"stl", "obj"}:
        if not preview_path.is_file():
            raise FileNotFoundError("No preview has been generated.")
        if preview_path.stat().st_size == 0:
            raise ValueError("The generated preview is empty.")
        if fmt == "stl":
            return preview_path

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    exports_dir = project_dir / ".cad-agent" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    cached_file = exports_dir / f"{model_sha256}.{fmt}"

    if cached_file.is_file() and cached_file.stat().st_size > 0:
        return cached_file

    if fmt == "obj":
        obj_content = stl_to_obj(preview_path)
        with tempfile.NamedTemporaryFile(
            dir=exports_dir,
            prefix=f".tmp_{model_sha256}_",
            suffix=".obj",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp.write(obj_content)
            temp_file = Path(tmp.name)
        try:
            temp_file.replace(cached_file)
        finally:
            if temp_file.is_file():
                temp_file.unlink()
        return cached_file

    # For OpenSCAD-backed formats: 3mf, amf, off, csg
    openscad_bin = find_openscad()
    if not openscad_bin:
        raise RuntimeError("OpenSCAD executable not found on server.")

    with tempfile.NamedTemporaryFile(
        dir=exports_dir,
        prefix=f".tmp_{model_sha256}_",
        suffix=f".{fmt}",
        delete=False,
    ) as tmp:
        temp_file = Path(tmp.name)

    try:
        proc = subprocess.run(
            [openscad_bin, "-o", str(temp_file), str(model_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0 or not temp_file.is_file() or temp_file.stat().st_size == 0:
            err_msg = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(f"OpenSCAD export failed:\n{err_msg or 'No output produced.'}")
        temp_file.replace(cached_file)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"OpenSCAD export to {fmt} timed out.") from exc
    finally:
        if temp_file.is_file():
            temp_file.unlink()

    return cached_file
