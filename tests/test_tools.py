import base64
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.terminal_tool import TerminalTool


def test_file_tool_rejects_unsafe_model(tmp_path: Path):
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write("model.py", "import subprocess\n")
    assert not (tmp_path / "model.py").exists()

    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write("model.py", "__import__('os').system('id')\n")


def test_file_tool_only_edits_allowlisted_files(tmp_path: Path):
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="Only model.py"):
        tool.write("notes.txt", "nope")


def test_file_tool_supports_limited_regex_patches(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("summary.md", "Width: 10 mm\n")
    result = tool.regex_replace("summary.md", r"\d+", "12")
    assert result == "Updated summary.md with 1 regex replacement(s)."
    assert tool.read("summary.md") == "Width: 12 mm\n"


@pytest.mark.parametrize("keyword", ["center", "start_angle", "end_angle"])
def test_model_preflight_rejects_ellipse_arc_keywords(tmp_path: Path, keyword: str):
    code = (
        "from build123d import *\n"
        "with BuildLine():\n"
        f"    Ellipse(30, 22, {keyword}=0)\n"
    )

    with pytest.raises(ValueError, match="use EllipticalCenterArc"):
        FileTool(tmp_path).write("model.py", code)

    assert not (tmp_path / "model.py").exists()


def test_model_preflight_rejects_radius_arc_below_half_chord(tmp_path: Path):
    code = """from build123d import *
base_radius = 30.0
overall_height = 30.0
top_blend_r = 12.0
top_flat = 6.0
with BuildLine():
    side = Line((base_radius, 0), (base_radius, overall_height - top_blend_r))
    RadiusArc(side @ 1, (top_flat, overall_height), radius=top_blend_r)
"""

    with pytest.raises(ValueError, match=r"minimum radius is 13\.416"):
        FileTool(tmp_path).write("model.py", code)


def test_model_preflight_warns_on_fixed_selector_index_after_fillet(tmp_path: Path):
    code = """from build123d import *
with BuildPart() as model:
    Box(20, 20, 10)
    bottom = model.edges().sort_by(Axis.Z)[0]
    fillet(bottom, radius=1)
    junction = model.edges().filter_by(GeomType.CIRCLE).sort_by(Axis.Z)[1]
result = model.part
"""

    result = FileTool(tmp_path).write("model.py", code)

    assert "PRE-FLIGHT WARNING" in result
    assert "fixed selector index used after a topology-changing operation" in result


def test_model_preflight_accepts_valid_radius_arc_and_elliptical_arc(tmp_path: Path):
    code = """from build123d import *
with BuildLine():
    RadiusArc((30, 8), (8, 30), radius=22)
    EllipticalCenterArc((0, 8), 30, 22, start_angle=0, end_angle=90)
"""

    result = FileTool(tmp_path).write("model.py", code)

    assert result.startswith("Wrote model.py")
    assert "WARNING" not in result


def test_terminal_tool_runs_only_local_python_script(tmp_path: Path):
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")
    tool = TerminalTool(tmp_path)
    result = tool.run(["python", "check.py"])
    assert result["returncode"] == 0
    assert result["stdout"] == "ok\n"
    with pytest.raises(ValueError, match="Only Python commands"):
        tool.run(["bash", "check.py"])


def test_terminal_tool_reports_python_failures_as_errors(tmp_path: Path):
    (tmp_path / "fail.py").write_text("print('before failure')\nraise ValueError('broken')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="before failure") as error:
        TerminalTool(tmp_path).run(["python", "fail.py"])

    assert "ValueError: broken" in str(error.value)


def test_terminal_sandbox_hides_environment_and_blocks_network(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    host_secret = tmp_path.parent / "host-secret.txt"
    host_secret.write_text("private", encoding="utf-8")
    (tmp_path / "probe.py").write_text(
        "import os, socket\n"
        "from pathlib import Path\n"
        "print(os.getenv('OPENROUTER_API_KEY'))\n"
        f"print(Path({str(host_secret)!r}).exists())\n"
        "try:\n"
        "    socket.socket()\n"
        "except OSError as error:\n"
        "    print(error.errno)\n",
        encoding="utf-8",
    )

    result = TerminalTool(tmp_path).run(["python", "probe.py"])

    assert result["stdout"] == "None\nFalse\n1\n"


def test_cad_sandbox_does_not_inherit_api_key(tmp_path: Path, monkeypatch):
    pytest.importorskip("build123d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    (tmp_path / "model.py").write_text(
        "import requests\n"
        "from build123d import Box\n"
        "secret = requests.utils.os.environ.get('OPENROUTER_API_KEY')\n"
        "result = Box(1 if secret is None else 2, 1, 1)\n",
        encoding="utf-8",
    )

    assert CadTool(tmp_path).run()["dimensions_mm"]["x"] == 1


def test_screenshot_requires_matching_one_time_request(tmp_path: Path):
    events = []
    result = {}
    cad = CadTool(tmp_path, lambda kind, data: events.append((kind, data)))

    worker = threading.Thread(
        target=lambda: result.update(cad.screenshot("front")),
        daemon=True,
    )
    worker.start()
    for _ in range(100):
        if events:
            break
        time.sleep(0.01)
    request_id = events[0][1]["request_id"]
    image = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, "PNG")
    encoded = base64.b64encode(image.getvalue()).decode()

    assert not cad.receive_screenshot("wrong", encoded)
    assert cad.receive_screenshot(request_id, encoded)
    assert not cad.receive_screenshot(request_id, encoded)
    worker.join(timeout=1)

    assert result["screenshot"] == "screenshot.png"
    assert (tmp_path / "screenshot.png").is_file()


def test_cad_inspect_invalidates_metrics_after_model_change(tmp_path: Path):
    pytest.importorskip("build123d")
    model = tmp_path / "model.py"
    model.write_text("from build123d import Box\nresult = Box(10, 20, 30)\n", encoding="utf-8")
    cad = CadTool(tmp_path)
    assert cad.run()["dimensions_mm"] == {"x": 10.0, "y": 20.0, "z": 30.0}

    model.write_text("from build123d import Box\nresult = Box(40, 20, 30)\n", encoding="utf-8")

    # inspect returns file info only when cache is stale (no auto re-run)
    info = cad.inspect()
    assert "dimensions_mm" not in info
    assert info["model_lines"] == 2
    assert info["preview_available"] is True

    # fresh run updates metrics
    assert cad.run()["dimensions_mm"] == {"x": 40.0, "y": 20.0, "z": 30.0}


def test_cad_failure_detail_keeps_model_location_and_final_error():
    output = """Traceback (most recent call last):
  File "/tmp/runner.py", line 9, in <module>
  File "model.py", line 35, in <module>
    chamfer(edges, length=1.5)
ValueError: Failed creating a chamfer, try a smaller length value(s)
"""

    detail = CadTool._failure_detail(output)

    assert 'File "model.py", line 35' in detail
    assert "chamfer(edges, length=1.5)" in detail
    assert detail.endswith("try a smaller length value(s)")
    assert "runner.py" not in detail


def test_cad_render_is_deterministic_and_has_no_center_depth_hole_for_solid_box(tmp_path: Path):
    pytest.importorskip("build123d")
    (tmp_path / "model.py").write_text(
        "from build123d import Box\nresult = Box(40, 30, 8)\n",
        encoding="utf-8",
    )
    cad = CadTool(tmp_path)

    cad.render()
    first = (tmp_path / "render.png").read_bytes()
    cad.render()
    second = (tmp_path / "render.png").read_bytes()

    with Image.open(tmp_path / "render.png") as image:
        assert image.size == (512, 512)
        assert image.getpixel((256, 256)) != (23, 25, 29)
    assert first == second
