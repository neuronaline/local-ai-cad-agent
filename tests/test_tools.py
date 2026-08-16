from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.revisions import RevisionStore
from agent.tools.cad_scripts.renderer import _BACKGROUND, _shade_pixels
from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import normalize_questions
from agent.tools.terminal_tool import (
    TerminalTool,
    _validate_check_code,
)


def test_review_renderer_uses_correct_barycentric_coordinates():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    screen_vertices = np.array([[100, 100], [200, 100], [150, 200]], dtype=float)
    pixels = _shade_pixels(
        vertices,
        triangles,
        screen_vertices,
        np.array([0, 0, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
    )

    assert not np.array_equal(pixels[150, 150], _BACKGROUND)


def test_cad_tool_runner_code_reflects_review_settings():
    """Review knobs propagate to the generated runner script. Clamping to 1
    prevents a zero/negative worker or view count from producing an invalid
    subprocess invocation."""
    # Arrange: zero / negative values must be clamped to 1 (the documented
    # minimum for both workers and required views).
    tool_zero = CadTool(Path("/tmp/project"), review_render_workers=0, review_required_views=0)
    code_zero = tool_zero._runner_code(render=True)
    assert "_RENDER_WORKERS = 1" in code_zero
    assert "_REQUIRED_VIEWS = 1" in code_zero

    # Arrange / Act: positive overrides land in the generated code verbatim.
    tool_custom = CadTool(
        Path("/tmp/project"), review_render_workers=2, review_required_views=6
    )
    code_custom = tool_custom._runner_code(render=True)
    assert "_RENDER_WORKERS = 2" in code_custom
    assert "_REQUIRED_VIEWS = 6" in code_custom

    # Act / Assert: render=False must disable the review-rendering block so the
    # subprocess never spawns review workers, regardless of the configured
    # worker/view counts.
    tool_no_render = CadTool(Path("/tmp/project"))
    code_no_render = tool_no_render._runner_code(render=False)
    assert "_RENDER_VIEWS = False" in code_no_render
    assert "_RENDER_WORKERS = " not in code_no_render
    assert "_REQUIRED_VIEWS = " not in code_no_render


@pytest.mark.parametrize(
    ("workers", "views", "expected_workers", "expected_views"),
    [
        pytest.param(0, 0, 1, 1, id="zero-zero-clamps-to-one"),
        pytest.param(-1, -2, 1, 1, id="negative-values-clamps-to-one"),
        pytest.param(1, 1, 1, 1, id="minimum-one-is-unchanged"),
        pytest.param(8, 16, 8, 16, id="typical-values-pass-through"),
        pytest.param("3", "5", 3, 5, id="string-numbers-are-coerced"),
    ],
)
def test_cad_tool_review_settings_are_normalized(
    workers, views, expected_workers, expected_views
):
    """Boundary: input clamping/normalization at construction time is the
    contract that keeps the generated runner script valid."""
    tool = CadTool(
        Path("/tmp/project"),
        review_render_workers=workers,
        review_required_views=views,
    )

    code = tool._runner_code(render=True)

    assert f"_RENDER_WORKERS = {expected_workers}" in code
    assert f"_REQUIRED_VIEWS = {expected_views}" in code


def test_file_tool_rejects_unsafe_model(tmp_path: Path):
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write("model.py", "import subprocess\n")
    assert not (tmp_path / "model.py").exists()

    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write("model.py", "__import__('os').system('id')\n")

    # Attribute-access forms of blocked builtins must be caught too
    # (e.g. __builtins__.eval, builtins.exec). The sandbox is the real
    # boundary; the AST check is the contract development relies on.
    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write("model.py", "__builtins__.eval('1+1')\n")
    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write("model.py", "builtins.exec('print(1)')\n")


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


def test_file_tool_regex_replace_rejects_pathological_pattern(tmp_path: Path):
    """ReDoS-prone patterns must be killed by the safety timeout."""
    tool = FileTool(tmp_path)
    tool.write("summary.md", "a" * 30)
    with pytest.raises(RuntimeError, match="safety timeout"):
        tool.regex_replace("summary.md", "(a+)+b", "X")


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
    (tmp_path / "fail.py").write_text(
        "print('before failure')\nraise ValueError('broken')\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="before failure") as error:
        TerminalTool(tmp_path).run(["python", "fail.py"])

    assert "ValueError: broken" in str(error.value)


def test_terminal_sandbox_hides_environment_and_blocks_network(
    tmp_path: Path, monkeypatch
):
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


def test_cad_build_recording_is_best_effort(tmp_path: Path, monkeypatch):
    cad = CadTool(tmp_path)
    monkeypatch.setattr(
        cad._revisions, "head", lambda: (_ for _ in ()).throw(OSError("disk full"))
    )

    cad._record_build_success({"solid_count": 1})
    cad._record_build_failure("original CAD error")


    cad = CadTool(tmp_path)
    calls = []
    metrics = {"solid_count": 1, "is_valid": True}
    wrapped = {
        "metrics": metrics,
        "feature_summary": {},
        "review_manifest": None,
        "review_views_dir": None,
        "review_sheet_path": None,
    }
    monkeypatch.setattr(
        cad,
        "_execute",
        lambda **kwargs: calls.append(kwargs) or wrapped,
    )

    result = cad.build_and_verify()

    assert calls == [{"render": True}]
    assert result == {
        "metrics": metrics,
        "feature_summary": {},
        "preview": "preview.stl",
        "render": "render.png",
    }


# --------------------------------------------------------------------------- #
#  FileTool revision integration tests
# --------------------------------------------------------------------------- #


def test_file_write_creates_revision_after_preflight(tmp_path: Path):
    tool = FileTool(tmp_path)
    result = tool.write(
        "model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n"
    )

    assert "revision" in result
    store = RevisionStore(tmp_path)
    head = store.head()
    assert head is not None
    assert (
        (tmp_path / "model.py").read_text(encoding="utf-8").startswith("from build123d")
    )


def test_file_replace_creates_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n")
    store = RevisionStore(tmp_path)
    first_head = store.head().id

    tool.replace("model.py", "Box(10, 20, 30)", "Box(40, 20, 30)")

    second_head = store.head()
    assert second_head.id != first_head
    assert second_head.parent_id == first_head
    assert second_head.origin.operation == "replace"


def test_file_regex_replace_creates_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("model.py", "WIDTH = 10\nresult = WIDTH\n")
    store = RevisionStore(tmp_path)
    first_head = store.head().id

    tool.regex_replace("model.py", r"\d+", "20")

    second_head = store.head()
    assert second_head.id != first_head
    assert second_head.origin.operation == "regex_replace"


def test_file_missing_replace_target_creates_no_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id

    with pytest.raises(ValueError, match="not found"):
        tool.replace("model.py", "nonexistent", "whatever")

    # No new revision.
    assert store.head().id == head_id


def test_file_preflight_failure_creates_no_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id

    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write("model.py", "import subprocess\n")

    # No new revision, model.py unchanged.
    assert store.head().id == head_id
    assert (tmp_path / "model.py").read_text(encoding="utf-8") == "result = 1\n"


def test_file_noop_write_creates_no_new_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id

    # Write the same content again.
    tool.write("model.py", "result = 1\n")

    # Same revision (deduplicated).
    assert store.head().id == head_id


def test_file_tool_result_includes_revision_id_without_paths(tmp_path: Path):
    tool = FileTool(tmp_path)
    result = tool.write("model.py", "result = 1\n")

    assert "revision" in result
    # No internal filesystem paths exposed.
    assert ".cad-agent" not in result
    assert "/" not in result.split("revision")[0]


def test_file_tool_with_call_id_stores_origin(tmp_path: Path):
    tool = FileTool(tmp_path, tool_call_id="call_abc")
    tool.write("model.py", "result = 1\n")

    store = RevisionStore(tmp_path)
    head = store.head()
    assert head.origin.tool_call_id == "call_abc"
    assert head.origin.operation == "write"


def test_file_tool_direct_construction_without_revisions_works(tmp_path: Path):
    """Existing direct FileTool(tmp_path) test construction remains supported."""
    tool = FileTool(tmp_path)
    result = tool.write("summary.md", "# Test\n")
    assert result == "Wrote summary.md (7 characters)."


def test_file_summary_write_does_not_create_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write("summary.md", "# Test\n")

    store = RevisionStore(tmp_path)
    assert store.head() is None  # No model.py revision created.


# --------------------------------------------------------------------------- #
#  TerminalTool check -c AST validator regression tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "snippet",
    [
        # Pre-existing banned identifiers.
        "eval('1+1')",
        "exec('print(1)')",
        "open('/etc/passwd')",
        "__import__('os')",
        "compile('x', 'y', 'exec')",
        "input('prompt')",
        "breakpoint()",
        "globals()",
        "locals()",
        "vars()",
        # getattr was in the old allowlist and is now blocked.
        "getattr(__builtins__, 'eval')('1')",
        # Subscript bypass: __builtins__['eval'].
        "__builtins__['eval']('1')",
        # MRO introspection chains that previously slipped past the walker.
        "().__class__.__bases__[0].__subclasses__()",
        "().__class__.__mro__[1].__subclasses__()",
        # Attribute access into a banned root identifier.
        "builtins.eval('1')",
        "builtins.open('/etc/passwd')",
        # Blocked module references.
        "import os\nos.system('id')",
        "from sys import stdin",
    ],
)
def test_validate_check_code_rejects_sandbox_escape_vectors(snippet: str):
    """Every snippet below must fail ``_validate_check_code``.

    These are the canonical sandbox-escape vectors the AST walker was
    hardened against. Regression coverage here ensures that re-introducing
    ``getattr`` to the allowlist (or removing any other guard) is caught by
    CI rather than discovered in production.
    """
    with pytest.raises(ValueError):
        _validate_check_code(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "print(2 + 2)",
        "print(type(1))",
        "print(len('abc'))",
        "print(repr('x'))",
        "from math import sqrt\nprint(sqrt(4))",
        "import numpy as np\nprint(np.zeros(1))",
    ],
)
def test_validate_check_code_accepts_read_only_inspection(snippet: str):
    """Read-only inspection snippets must pass without raising."""
    _validate_check_code(snippet)


def test_validate_check_code_rejects_final_statement_not_in_allowlist():
    """The trailing statement must be a direct ``Name`` call, not an ``Attribute``.

    ``list.append(x, 1)`` is a single-statement ``Attribute`` call whose root
    (``list``) and attr (``append``) are both outside the AST-walker's
    blocked sets, so the walker is silent. The final-statement allowlist
    check then rejects it because the call target is not a direct ``Name``.
    """
    with pytest.raises(ValueError, match="final statement must call one of"):
        _validate_check_code("list.append(x, 1)")


# --------------------------------------------------------------------------- #
#  QuestionTool.normalize_questions regression tests
# --------------------------------------------------------------------------- #


def test_normalize_questions_legacy_format_rejects_missing_text():
    """The legacy flat format must surface a friendly ValueError for empty
    question text rather than letting the underlying KeyError / ``None`` slip
    through to the SSE consumer."""
    with pytest.raises(ValueError, match="'question' missing or empty"):
        normalize_questions({"input_type": "text"})


def test_normalize_questions_legacy_format_rejects_non_string_text():
    with pytest.raises(ValueError, match="'question' missing or empty"):
        normalize_questions({"question": 123, "input_type": "text"})


def test_normalize_questions_legacy_format_rejects_whitespace_only_text():
    with pytest.raises(ValueError, match="'question' missing or empty"):
        normalize_questions({"question": "   ", "input_type": "text"})


def test_normalize_questions_legacy_format_accepts_well_formed_text():
    result = normalize_questions({"question": "Choose width", "input_type": "number"})
    assert result == [
        {
            "id": "q1",
            "question": "Choose width",
            "input_type": "number",
            "options": [],
        }
    ]
