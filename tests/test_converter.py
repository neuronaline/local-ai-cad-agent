import subprocess
from pathlib import Path

import pytest

from agent.converter import (
    SUPPORTED_EXPORT_FORMATS,
    export_project_model,
    find_openscad,
    stl_to_obj,
)


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a temporary project folder with a valid model.scad and preview.stl."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir(parents=True)
    scad_file = project_dir / "model.scad"
    scad_file.write_text("cube([10, 20, 30]);", encoding="utf-8")

    openscad = find_openscad()
    assert openscad is not None, "OpenSCAD must be available for tests."

    preview_file = project_dir / "preview.stl"
    subprocess.run(
        [openscad, "-o", str(preview_file), "--export-format", "binstl", str(scad_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert preview_file.is_file() and preview_file.stat().st_size > 0
    return project_dir


def test_stl_to_obj_conversion(sample_project: Path) -> None:
    """Test converting a binary STL into Wavefront OBJ."""
    preview_file = sample_project / "preview.stl"
    obj_str = stl_to_obj(preview_file)
    assert "# OBJ export from Local AI CAD Agent" in obj_str

    v_lines = [line for line in obj_str.splitlines() if line.startswith("v ")]
    f_lines = [line for line in obj_str.splitlines() if line.startswith("f ")]
    # A cube should have 8 vertices and 12 triangular faces
    assert len(v_lines) == 8
    assert len(f_lines) == 12


def test_export_project_model_all_formats(sample_project: Path) -> None:
    """Verify exporting to each supported format succeeds and creates cached files."""
    for fmt in SUPPORTED_EXPORT_FORMATS:
        out_path = export_project_model(sample_project, fmt)
        assert out_path.is_file()
        assert out_path.stat().st_size > 0

        # Subsequent call should hit cache for derived formats
        cached_path = export_project_model(sample_project, fmt)
        assert cached_path == out_path


def test_export_unsupported_format(sample_project: Path) -> None:
    """Invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_project_model(sample_project, "unsupported_xyz")


def test_export_missing_model(tmp_path: Path) -> None:
    """Missing model.scad raises FileNotFoundError."""
    empty_dir = tmp_path / "empty-project"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No model source file found"):
        export_project_model(empty_dir, "stl")


def test_export_empty_model(tmp_path: Path) -> None:
    """Empty model.scad raises ValueError."""
    project_dir = tmp_path / "empty-model"
    project_dir.mkdir()
    (project_dir / "model.scad").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="model source file is empty"):
        export_project_model(project_dir, "stl")


def test_export_missing_preview_for_stl_and_obj(tmp_path: Path) -> None:
    """Missing preview.stl raises FileNotFoundError when requesting stl or obj."""
    project_dir = tmp_path / "no-preview"
    project_dir.mkdir()
    (project_dir / "model.scad").write_text("cube([5, 5, 5]);", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No preview has been generated"):
        export_project_model(project_dir, "stl")

    with pytest.raises(FileNotFoundError, match="No preview has been generated"):
        export_project_model(project_dir, "obj")


def test_export_empty_preview_for_stl_and_obj(tmp_path: Path) -> None:
    """Empty preview.stl raises ValueError when requesting stl or obj."""
    project_dir = tmp_path / "empty-preview"
    project_dir.mkdir()
    (project_dir / "model.scad").write_text("cube([5, 5, 5]);", encoding="utf-8")
    (project_dir / "preview.stl").write_bytes(b"")

    with pytest.raises(ValueError, match="The generated preview is empty"):
        export_project_model(project_dir, "stl")

    with pytest.raises(ValueError, match="The generated preview is empty"):
        export_project_model(project_dir, "obj")


def test_export_obj_requires_preview_even_if_cache_exists(sample_project: Path) -> None:
    """Deleting preview.stl after OBJ export must still fail future requests."""
    out = export_project_model(sample_project, "obj")
    assert out.is_file()

    # Delete preview.stl
    (sample_project / "preview.stl").unlink()
    with pytest.raises(FileNotFoundError, match="No preview has been generated"):
        export_project_model(sample_project, "obj")

