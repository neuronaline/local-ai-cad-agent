import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.revisions import RevisionStore
from agent.tools.cad_scripts.renderer import (
    _BACKGROUND,
    _HEIGHT,
    _SHEET_LABEL_HEIGHT,
    _WIDTH,
    _shade_pixels,
    build_contact_sheet,
)
from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import normalize_questions


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


def test_contact_sheet_uses_canonical_order_and_labels(tmp_path: Path):
    views = tmp_path / "views"
    views.mkdir()
    Image.new("RGB", (_WIDTH, _HEIGHT), (255, 0, 0)).save(views / "x_positive.png")
    Image.new("RGB", (_WIDTH, _HEIGHT), (0, 0, 255)).save(
        views / "isometric_positive.png"
    )

    sheet_path = tmp_path / "review-sheet.png"
    manifest = build_contact_sheet(views, sheet_path)

    with Image.open(sheet_path) as sheet:
        assert sheet.size == (_WIDTH * 4, _HEIGHT + _SHEET_LABEL_HEIGHT)
        assert sheet.getpixel((10, 10)) == (255, 0, 0)
        assert sheet.getpixel((_WIDTH + 10, 10)) == (0, 0, 255)
    assert manifest["view_order"] == ["x_positive", "isometric_positive"]


def test_cad_tool_runner_settings_reflect_review_settings():
    """Review knobs propagate to the JSON kwargs payload. Clamping to 1
    prevents a zero/negative worker or view count from producing an invalid
    subprocess invocation."""
    import json

    # Arrange: zero / negative values must be clamped to 1 (the documented
    # minimum for both workers and required views).
    tool_zero = CadTool(Path("/tmp/project"), review_render_workers=0, review_required_views=0)
    settings_zero = tool_zero._runner_settings(render=True)
    assert settings_zero["render_workers"] == 1
    assert settings_zero["required_views"] == 1

    # Arrange / Act: positive overrides land in the generated code verbatim.
    tool_custom = CadTool(
        Path("/tmp/project"), review_render_workers=2, review_required_views=6
    )
    settings_custom = tool_custom._runner_settings(render=True)
    assert settings_custom["render_workers"] == 2
    assert settings_custom["required_views"] == 6

    # Act / Assert: render=False must disable the review-rendering block so the
    # subprocess never spawns review workers, regardless of the configured
    # worker/view counts.
    tool_no_render = CadTool(Path("/tmp/project"))
    settings_no_render = tool_no_render._runner_settings(render=False)
    assert settings_no_render["render_views"] is False
    # Worker / view counts still surface so the runner can normalise them even
    # in the no-render path (the script currently ignores them).
    payload = json.dumps(settings_no_render)
    assert "render_workers" in payload
    assert "required_views" in payload


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
    contract that keeps the generated kwargs payload valid."""
    tool = CadTool(
        Path("/tmp/project"),
        review_render_workers=workers,
        review_required_views=views,
    )

    settings = tool._runner_settings(render=True)

    assert settings["render_workers"] == expected_workers
    assert settings["required_views"] == expected_views


def test_file_tool_rejects_unsafe_model(tmp_path: Path):
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write_file("model.py", "import subprocess\n")
    assert not (tmp_path / "model.py").exists()

    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write_file("model.py", "__import__('os').system('id')\n")

    # Attribute-access forms of blocked builtins must be caught too
    # (e.g. __builtins__.eval, builtins.exec). The sandbox is the real
    # boundary; the AST check is the contract development relies on.
    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write_file("model.py", "__builtins__.eval('1+1')\n")
    with pytest.raises(ValueError, match="Unsafe function blocked"):
        tool.write_file("model.py", "builtins.exec('print(1)')\n")


def test_file_tool_only_edits_allowlisted_files(tmp_path: Path):
    tool = FileTool(tmp_path)
    with pytest.raises(ValueError, match="Only model.py"):
        tool.write_file("notes.txt", "nope")


def test_write_file_creates_new_file_without_digest(tmp_path: Path):
    tool = FileTool(tmp_path)
    result = tool.write_file("model.py", "# Width: 10 mm\n")
    assert "Wrote model.py" in result
    payload = json.loads(tool.read_file("model.py"))
    assert payload["content"] == "# Width: 10 mm\n"
    assert payload["exists"] is True
    assert len(payload["sha256"]) == 64


def test_read_file_missing_reports_that_it_cannot_be_edited(tmp_path: Path):
    payload = json.loads(FileTool(tmp_path).read_file("model.py"))

    assert payload == {
        "exists": False,
        "content": "",
        "sha256": None,
        "total_lines": 0,
        "offset": 1,
        "returned_lines": 0,
        "next_offset": None,
    }


def test_write_file_allows_unconditional_overwrite(tmp_path: Path):
    """Omitting ``expected_sha256`` is an unconditional overwrite contract.

    The agent loop is allowed to call ``write_file`` without a prior
    ``read_file`` round-trip when it already knows the file content (e.g. it
    just produced the file itself). Conflict detection is opt-in via
    ``expected_sha256``; without it, the new content is written.
    """
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# Width: 10 mm\n")
    tool.write_file("model.py", "# Width: 12 mm\n")
    assert json.loads(tool.read_file("model.py"))["content"] == "# Width: 12 mm\n"


def test_write_file_stale_digest_leaves_content_unchanged(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# Width: 10 mm\n")
    digest = json.loads(tool.read_file("model.py", offset=1, limit=1))["sha256"]
    tool.write_file("model.py", "# Width: 11 mm\n", expected_sha256=digest)

    with pytest.raises(ValueError, match="changed since it was read"):
        tool.write_file("model.py", "# Width: 12 mm\n", expected_sha256=digest)

    assert json.loads(tool.read_file("model.py"))["content"] == "# Width: 11 mm\n"


def test_read_file_range_returns_digest_for_guarded_edit(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# one\n# two\n# three\n")

    payload = json.loads(tool.read_file("model.py", offset=2, limit=1))

    assert payload["content"] == "# two\n"
    assert payload["offset"] == 2
    assert payload["next_offset"] == 3
    with pytest.raises(ValueError, match="changed since it was read"):
        tool.write_file("model.py", "# updated\n", expected_sha256="0" * 64)


def test_read_file_range_digest_can_guard_an_edit(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# one\n# two\n")

    payload = json.loads(tool.read_file("model.py", offset=1, limit=2))
    tool.edit_file("model.py", "# two", "# updated", payload["sha256"])

    assert json.loads(tool.read_file("model.py"))["content"] == "# one\n# updated\n"


def test_edit_file_rejects_ambiguous_target(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# same\n# same\n")
    digest = json.loads(tool.read_file("model.py"))["sha256"]

    with pytest.raises(ValueError, match="found 2"):
        tool.edit_file("model.py", "# same", "# new", digest)

    assert json.loads(tool.read_file("model.py"))["content"] == "# same\n# same\n"


def test_edit_file_rejects_missing_target(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# anchor here\n")
    digest = json.loads(tool.read_file("model.py"))["sha256"]

    with pytest.raises(ValueError, match="one exact match"):
        tool.edit_file("model.py", "# nonexistent", "# replacement", digest)

    assert json.loads(tool.read_file("model.py"))["content"] == "# anchor here\n"


def test_edit_file_rejects_missing_file(tmp_path: Path):
    tool = FileTool(tmp_path)

    with pytest.raises(ValueError, match="use write_file"):
        tool.edit_file("model.py", "# anything", "# replacement", "0" * 64)


def test_edit_file_rejects_stale_digest(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# alpha\n")
    digest = json.loads(tool.read_file("model.py"))["sha256"]
    tool.write_file("model.py", "# beta\n", expected_sha256=digest)

    with pytest.raises(ValueError, match="changed since it was read"):
        tool.edit_file("model.py", "# beta", "# gamma", "0" * 64)

    assert json.loads(tool.read_file("model.py"))["content"] == "# beta\n"


def test_edit_file_requires_non_empty_old_string(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# alpha\n")
    digest = json.loads(tool.read_file("model.py"))["sha256"]

    with pytest.raises(ValueError, match="old_string must not be empty"):
        tool.edit_file("model.py", "", "# new", digest)


def test_edit_file_allows_unconditional_edit(tmp_path: Path):
    """Omitting ``expected_sha256`` is an unconditional edit contract.

    Same rationale as ``test_write_file_allows_unconditional_overwrite``: the
    agent knows the current content from its own previous write, so forcing a
    ``read_file`` round-trip only to capture the SHA would just waste a tool
    call + tokens + latency. Conflict detection remains opt-in.
    """
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# alpha\n")

    tool.edit_file("model.py", "# alpha", "# beta")

    assert (
        json.loads(tool.read_file("model.py"))["content"] == "# beta\n"
    )


def test_edit_file_reports_replaced_line_range(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# alpha\n# beta\n# gamma\n")
    digest = json.loads(tool.read_file("model.py"))["sha256"]

    result = tool.edit_file("model.py", "# beta\n", "# B\n", digest)

    assert "replaced lines 2-2 with lines 2-2" in result
    assert json.loads(tool.read_file("model.py"))["content"] == "# alpha\n# B\n# gamma\n"


@pytest.mark.parametrize("keyword", ["center", "start_angle", "end_angle"])
def test_model_preflight_rejects_ellipse_arc_keywords(tmp_path: Path, keyword: str):
    code = (
        "from build123d import *\n"
        "with BuildLine():\n"
        f"    Ellipse(30, 22, {keyword}=0)\n"
    )

    with pytest.raises(ValueError, match="use EllipticalCenterArc"):
        FileTool(tmp_path).write_file("model.py", code)

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
        FileTool(tmp_path).write_file("model.py", code)


def test_model_preflight_warns_on_fixed_selector_index_after_fillet(tmp_path: Path):
    code = """from build123d import *
with BuildPart() as model:
    Box(20, 20, 10)
    bottom = model.edges().sort_by(Axis.Z)[0]
    fillet(bottom, radius=1)
    junction = model.edges().filter_by(GeomType.CIRCLE).sort_by(Axis.Z)[1]
result = model.part
"""

    result = FileTool(tmp_path).write_file("model.py", code)

    assert "PRE-FLIGHT WARNING" in result
    assert "fixed selector index used after a topology-changing operation" in result


def test_model_preflight_accepts_valid_radius_arc_and_elliptical_arc(tmp_path: Path):
    code = """from build123d import *
with BuildLine():
    RadiusArc((30, 8), (8, 30), radius=22)
    EllipticalCenterArc((0, 8), 30, 22, start_angle=0, end_angle=90)
"""

    result = FileTool(tmp_path).write_file("model.py", code)

    assert result.startswith("Wrote model.py")
    assert "WARNING" not in result


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


def test_cad_build_and_verify_legacy_payload(tmp_path: Path, monkeypatch):
    """render=True keeps the legacy review manifest + artifact-path payload.

    The default flipped to render=False for fast iteration; the legacy
    payload (review_manifest, review_path) only materialises when the caller
    explicitly opts into rendering.
    """
    cad = CadTool(tmp_path)
    calls = []
    metrics = {
        "solid_count": 1,
        "is_valid": True,
        "dimensions_mm": {"x": 1, "y": 1, "z": 1},
    }
    wrapped = {
        "metrics": metrics,
        "feature_summary": {},
        "model_sha256": "a" * 64,
        "preview_sha256": "b" * 64,
        "review_manifest": {"model_sha256": "a" * 64, "preview_sha256": "b" * 64},
        "review_views_dir": "",
        "review_sheet_path": "",
    }
    monkeypatch.setattr(
        cad,
        "_execute",
        lambda **kwargs: calls.append(kwargs) or wrapped,
    )
    monkeypatch.setattr(
        cad,
        "promote_review",
        lambda manifest, *, sandbox_views_dir, sandbox_sheet_path: {"artifact_dir": "/tmp/review"},
    )

    result = cad.build_and_verify(render=True)

    assert calls == [{"render": True}]
    assert result["metrics"] is metrics
    assert result["feature_summary"] == {}
    assert result["preview"] == "preview.stl"
    assert result["render"] == "render.png"
    assert result["review"] == "/tmp/review"
    assert result["model_sha256"] == "a" * 64
    assert result["preview_sha256"] == "b" * 64
    # The review_manifest is dropped from the legacy payload: cad_review is
    # its own tool now and the agent never needs to re-parse it.
    assert "review_manifest" not in result
    assert result["summary"] == (
        "Solid 1 (valid); bbox 1.0×1.0×1.0 mm; 0.0 cm³; 0 features; with render."
    )


def test_cad_build_and_verify_default_is_metrics_only(tmp_path: Path, monkeypatch):
    """The default render=False path skips the review artifacts entirely.

    Early iteration never needs the multi-view rasterisation or the
    contact-sheet promotion; the build returns only ``metrics``,
    ``preview.stl``, and the model/preview SHAs.
    """
    cad = CadTool(tmp_path)
    calls = []
    metrics = {
        "solid_count": 1,
        "is_valid": True,
        "dimensions_mm": {"x": 1, "y": 1, "z": 1},
    }
    wrapped = {
        "metrics": metrics,
        "feature_summary": {},
        "model_sha256": "a" * 64,
        "preview_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        cad,
        "_execute",
        lambda **kwargs: calls.append(kwargs) or wrapped,
    )

    result = cad.build_and_verify()

    assert calls == [{"render": False}]
    assert result["metrics"] is metrics
    assert result["preview"] == "preview.stl"
    assert result["render"] is None
    assert "review" not in result
    assert "review_manifest" not in result
    assert result["model_sha256"] == "a" * 64
    assert result["preview_sha256"] == "b" * 64


def test_cad_build_and_verify_with_render_false_skips_review_artifacts(
    tmp_path: Path, monkeypatch
):
    pytest.importorskip("build123d")
    cad = CadTool(tmp_path)
    (tmp_path / "model.py").write_text(
        "from build123d import Box\nresult = Box(1, 1, 1)\n", encoding="utf-8"
    )
    captured: list[dict] = []

    payload = {
        "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
        "feature_summary": {},
        "review_manifest": {"should_be": "ignored"},
        "review_views_dir": "/tmp/views",
        "review_sheet_path": "/tmp/sheet.png",
    }

    def fake_execute(**kwargs):
        captured.append(kwargs)
        return payload

    monkeypatch.setattr(cad, "_execute", fake_execute)

    result = cad.build_and_verify(render=False)

    assert captured == [{"render": False}]
    # Structural assertion: review keys must not leak into the render-less
    # payload, but unrelated future keys (e.g. additional summary fields) can
    # grow without breaking this regression test.
    assert result["metrics"] is payload["metrics"]
    assert result["feature_summary"] == {}
    assert result["preview"] == "preview.stl"
    assert result["render"] is None
    assert "review" not in result
    assert "review_manifest" not in result
    assert result["summary"] == (
        "Solid 1 (valid); bbox 1.0×1.0×1.0 mm; 0.0 cm³; 0 features; metrics-only."
    )


def test_cad_build_and_verify_with_render_true_keeps_legacy_payload(
    tmp_path: Path, monkeypatch
):
    pytest.importorskip("build123d")
    cad = CadTool(tmp_path)
    (tmp_path / "model.py").write_text(
        "from build123d import Box\nresult = Box(1, 1, 1)\n", encoding="utf-8"
    )
    captured: list[dict] = []

    payload = {
        "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
        "feature_summary": {},
        "model_sha256": "a" * 64,
        "preview_sha256": "b" * 64,
        "review_manifest": {"some": "manifest"},
        "review_views_dir": None,
        "review_sheet_path": None,
    }

    def fake_execute(**kwargs):
        captured.append(kwargs)
        return payload

    monkeypatch.setattr(cad, "_execute", fake_execute)
    monkeypatch.setattr(
        cad, "promote_review", lambda manifest, *, sandbox_views_dir, sandbox_sheet_path: {"artifact_dir": "/tmp/review"}
    )

    result = cad.build_and_verify(render=True)

    assert captured == [{"render": True}]
    assert result["render"] == "render.png"
    # The review manifest is dropped from the public payload: cad_review is
    # its own tool now and the agent never needs to re-parse it.
    assert "review_manifest" not in result
    assert result["review"] == "/tmp/review"
    assert result["model_sha256"] == "a" * 64
    assert result["preview_sha256"] == "b" * 64
    assert "summary" in result
    assert "Solid 1" in result["summary"]


def test_cad_build_and_verify_summary_mentions_features(tmp_path: Path, monkeypatch):
    pytest.importorskip("build123d")
    cad = CadTool(tmp_path)
    (tmp_path / "model.py").write_text(
        "from build123d import Box\nresult = Box(1, 1, 1)\n", encoding="utf-8"
    )
    payload = {
        "metrics": {
            "solid_count": 2,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 20, "z": 30},
            "volume_mm3": 6_000,
        },
        "feature_summary": {
            "through_hole_count": 3,
            "blind_hole_count": 1,
            "fillet_count": 2,
            "chamfer_count": 0,
        },
        "review_manifest": None,
        "review_views_dir": None,
        "review_sheet_path": None,
    }
    monkeypatch.setattr(cad, "_execute", lambda **kwargs: payload)

    result = cad.build_and_verify(render=False)

    assert "6 features" in result["summary"]
    assert "6.0 cm³" in result["summary"]
    assert "10.0×20.0×30.0 mm" in result["summary"]


def test_cad_build_and_verify_summary_handles_missing_metrics(tmp_path: Path, monkeypatch):
    pytest.importorskip("build123d")
    cad = CadTool(tmp_path)
    (tmp_path / "model.py").write_text(
        "from build123d import Box\nresult = Box(1, 1, 1)\n", encoding="utf-8"
    )
    payload = {
        "metrics": None,
        "feature_summary": {},
        "review_manifest": None,
        "review_views_dir": None,
        "review_sheet_path": None,
    }
    monkeypatch.setattr(cad, "_execute", lambda **kwargs: payload)

    result = cad.build_and_verify(render=False)

    assert result["summary"] == "Build produced no metrics."


# --------------------------------------------------------------------------- #
#  FileTool revision integration tests
# --------------------------------------------------------------------------- #


def test_write_file_creates_revision_after_preflight(tmp_path: Path):
    tool = FileTool(tmp_path)
    result = tool.write_file(
        "model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n"
    )

    assert "revision" in result
    store = RevisionStore(tmp_path)
    head = store.head()
    assert head is not None
    assert (
        (tmp_path / "model.py").read_text(encoding="utf-8").startswith("from build123d")
    )


def test_edit_file_creates_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n")
    store = RevisionStore(tmp_path)
    first_head = store.head().id
    digest = hashlib.sha256((tmp_path / "model.py").read_bytes()).hexdigest()

    tool.edit_file("model.py", "Box(10, 20, 30)", "Box(40, 20, 30)", digest)

    second_head = store.head()
    assert second_head.id != first_head
    assert second_head.parent_id == first_head
    assert second_head.origin.operation == "edit_file"


def test_missing_edit_target_creates_no_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id
    digest = hashlib.sha256((tmp_path / "model.py").read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="one exact match"):
        tool.edit_file("model.py", "nonexistent", "whatever", digest)

    # No new revision.
    assert store.head().id == head_id


def test_file_preflight_failure_creates_no_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id
    digest = hashlib.sha256((tmp_path / "model.py").read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write_file("model.py", "import subprocess\n", expected_sha256=digest)

    # No new revision, model.py unchanged.
    assert store.head().id == head_id
    assert (tmp_path / "model.py").read_text(encoding="utf-8") == "result = 1\n"


def test_file_noop_write_creates_no_new_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "result = 1\n")
    store = RevisionStore(tmp_path)
    head_id = store.head().id
    digest = hashlib.sha256((tmp_path / "model.py").read_bytes()).hexdigest()

    # Write the same content again with a matching digest.
    tool.write_file("model.py", "result = 1\n", expected_sha256=digest)

    # Same revision (deduplicated).
    assert store.head().id == head_id


def test_file_tool_result_includes_revision_id_without_paths(tmp_path: Path):
    tool = FileTool(tmp_path)
    result = tool.write_file("model.py", "result = 1\n")

    assert "revision" in result
    # No internal filesystem paths exposed.
    assert ".cad-agent" not in result
    assert "/" not in result.split("revision")[0]


def test_file_tool_with_call_id_stores_origin(tmp_path: Path):
    tool = FileTool(tmp_path, tool_call_id="call_abc")
    tool.write_file("model.py", "result = 1\n")

    store = RevisionStore(tmp_path)
    head = store.head()
    assert head.origin.tool_call_id == "call_abc"
    assert head.origin.operation == "write_file"


def test_file_tool_direct_construction_without_revisions_works(tmp_path: Path):
    """Existing direct FileTool(tmp_path) test construction remains supported."""
    tool = FileTool(tmp_path)
    result = tool.write_file("model.py", "# Test\n")
    assert result.startswith("Wrote model.py")


def test_model_write_creates_revision(tmp_path: Path):
    tool = FileTool(tmp_path)
    tool.write_file("model.py", "# Test\n")

    store = RevisionStore(tmp_path)
    assert store.head() is not None  # model.py edit always creates a revision.


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


def test_cad_screenshot_tool_exposes_required_surface():
    """``CadScreenshotTool`` exposes the constants the schema/agent rely on."""
    from agent.tools.cad_screenshot_tool import CadScreenshotTool

    assert len(CadScreenshotTool.SUBSET_VIEWS) == 8
    assert set(CadScreenshotTool.QUALITY_TIERS) == {"low", "standard", "high"}
    # ``with_call_id`` returns the tool for fluent chaining.
    tool = CadScreenshotTool(Path("/tmp/project"), publish=None)
    assert tool.with_call_id("test") is tool


def test_cad_review_tool_exposes_required_surface():
    """``CadReviewTool`` exposes the execute() / with_call_id() surface."""
    from agent.tools.cad_review_tool import CadReviewTool

    tool = CadReviewTool(Path("/tmp/project"), publish=None)
    assert tool.with_call_id("test") is tool


def test_review_enabled_only_disables_multiview_rasterisation(tmp_path: Path):
    """The setting keeps render.png while skipping the expensive sheet."""
    from agent.tools.cad_tool import CadTool

    settings = CadTool(tmp_path, review_enabled=False)._runner_settings(render=True)

    assert settings["render_views"] is False
    assert settings["write_isometric"] is True


def test_compact_for_context_strips_cad_build_payload_redundancy():
    """``cad_build_and_verify`` tool results must drop the per-feature
    cylinder table, the duplicated top-level ``feature_summary``, and the
    bulky ``review_manifest.views`` details before they enter the LLM
    history. The compact form keeps ``summary`` and ``metrics`` so the
    agent still has what it needs to decide what to do next.
    """
    from agent.tool_results import compact_for_context

    full_payload = {
        "ok": True,
        "tool": "cad_build_and_verify",
        "data": {
            "metrics": {
                "solid_count": 1,
                "is_valid": True,
                "volume_mm3": 100.0,
                "dimensions_mm": {"x": 10, "y": 10, "z": 1},
                "feature_summary": {
                    "disconnected_solid_count": 1,
                    "cylindrical_cut_candidates": [
                        {"diameter_mm": 3.2, "axis": [0, 0, 1], "area_mm2": 8.0,
                         "center_mm": [0, 0, 0]}
                    ] * 20,
                    "through_hole_count": 20,
                },
            },
            "preview": "preview.stl",
            "render": "render.png",
            "feature_summary": {
                "disconnected_solid_count": 1,
                "cylindrical_cut_candidates": ["x"] * 20,
                "through_hole_count": 20,
            },
            "review_manifest": {
                "model_sha256": "a" * 64,
                "preview_sha256": "b" * 64,
                "view_count": 8,
                "views": [
                    {
                        "view_id": f"view-{i}",
                        "label": "x",
                        "camera_axis": [1, 0, 0],
                        "screen_x_axis": [0, 1, 0],
                        "path": f"views/{i}.png",
                        "image_sha256": "c" * 64,
                        "image_bytes": 1500,
                        "width": 512,
                        "height": 512,
                        "render_status": "rendered",
                    }
                    for i in range(8)
                ],
                "contact_sheet": {
                    "path": "review-sheet.png",
                    "width": 2048,
                    "height": 1080,
                    "image_sha256": "d" * 64,
                    "image_bytes": 20000,
                },
                "single_render": {"path": "render.png", "image_sha256": "e" * 64},
            },
            "summary": "Solid 1 (valid); bbox 10x10x1 mm.",
        },
    }
    raw = json.dumps(full_payload, ensure_ascii=False)
    compacted = json.loads(compact_for_context("cad_build_and_verify", raw))

    assert compacted["ok"] is True
    data = compacted["data"]
    # Counts preserved; per-feature tables dropped from metrics.feature_summary
    fs = data["metrics"]["feature_summary"]
    assert fs["through_hole_count"] == 20
    assert fs["disconnected_solid_count"] == 1
    assert "cylindrical_cut_candidates" not in fs
    # Top-level feature_summary removed (was duplicating the metrics one)
    assert "feature_summary" not in data
    # Review manifest trimmed to model/preview hashes and view IDs only
    rm = data["review_manifest"]
    assert rm["model_sha256"] == "a" * 64
    assert rm["preview_sha256"] == "b" * 64
    assert rm["view_count"] == 8
    assert rm["contact_sheet_sha256"] == "d" * 64
    for entry in rm["views"]:
        assert set(entry.keys()) == {"view_id", "image_sha256"}
    # Summary untouched so the agent still has the one-liner.
    assert data["summary"].startswith("Solid 1 (valid)")

    # The compaction should also actually shrink the payload.
    assert len(compact_for_context("cad_build_and_verify", raw)) < len(raw)


def test_compact_for_context_is_passthrough_for_other_tools():
    """Tools other than ``cad_build_and_verify`` must keep their full
    payload; otherwise we'd risk truncating data the agent needs.
    """
    from agent.tool_results import compact_for_context

    raw = json.dumps(
        {"ok": True, "tool": "terminal_run", "data": {"stdout": "hello"}},
        ensure_ascii=False,
    )
    assert compact_for_context("terminal_run", raw) == raw


def test_prune_prior_read_file_drops_duplicate_bodies():
    """``_prune_prior_read_file`` must drop the file body from older
    ``read_file`` tool results while keeping the most recent body intact
    so the agent can still build ``edit_file`` ``old_string`` arguments.
    """
    from agent.dispatcher import _prune_prior_read_file

    new_sha = "b" * 64
    prior_sha_a = "a" * 64
    new_payload = json.dumps(
        {"ok": True, "tool": "read_file", "data": json.dumps(
            {"exists": True, "content": "latest body\n", "sha256": new_sha,
             "total_lines": 12}
        )},
        ensure_ascii=False,
    )
    prior_a = json.dumps(
        {"ok": True, "tool": "read_file", "data": json.dumps(
            {"exists": True, "content": "old body\n", "sha256": prior_sha_a,
             "total_lines": 10}
        )},
        ensure_ascii=False,
    )
    messages = [{"role": "tool", "content": prior_a}]

    _prune_prior_read_file(
        messages, {"filename": "model.py"}, new_payload
    )

    pruned = json.loads(messages[0]["content"])
    pruned_data = json.loads(pruned["data"])
    assert pruned_data.get("unchanged") is True
    assert pruned_data["sha256"] == prior_sha_a
    assert pruned_data["exists"] is True
    assert pruned_data["total_lines"] == 10
    assert "content" not in pruned_data

    # The new payload itself is untouched (caller still appends it).
    fresh = json.loads(new_payload)
    fresh_data = json.loads(fresh["data"])
    assert fresh_data["content"] == "latest body\n"


def test_prune_prior_read_file_skips_already_compact_entries():
    """Already-compacted ``read_file`` entries (``content is None``) must
    be left alone — calling the helper repeatedly must not rewrite them.
    """
    from agent.dispatcher import _prune_prior_read_file

    sha = "a" * 64
    compact_prior = json.dumps(
        {"ok": True, "tool": "read_file", "data": json.dumps(
            {"exists": True, "unchanged": True, "sha256": sha, "total_lines": 5}
        )},
        ensure_ascii=False,
    )
    new_payload = json.dumps(
        {"ok": True, "tool": "read_file", "data": json.dumps(
            {"exists": True, "content": "x\n", "sha256": sha, "total_lines": 1}
        )},
        ensure_ascii=False,
    )
    messages = [{"role": "tool", "content": compact_prior}]

    _prune_prior_read_file(
        messages, {"filename": "model.py"}, new_payload
    )

    assert messages[0]["content"] == compact_prior


# --------------------------------------------------------------------------- #
#  cad_build_and_verify multimodal tool-result tests
# --------------------------------------------------------------------------- #


def test_build_cad_build_multimodal_content_attaches_render(tmp_path: Path):
    """A successful render=true build must attach PNG/STL as inline image
    content so the agent evaluates the build in-band instead of asking the
    subordinate visual reviewer to start a fresh session."""
    from agent.tool_results import build_cad_build_multimodal_content

    render_payload = {
        "ok": True,
        "tool": "cad_build_and_verify",
        "data": {
            "metrics": {"solid_count": 1, "is_valid": True},
            "preview": "preview.stl",
            "render": "render.png",
            "summary": "Solid 1 (valid); with render.",
        },
    }
    (tmp_path / "render.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (tmp_path / "preview.stl").write_bytes(b"solid demo\n")

    raw = json.dumps(render_payload, ensure_ascii=False)
    multimodal = build_cad_build_multimodal_content(raw, tmp_path)

    assert multimodal is not None
    parts = multimodal["content"]
    # text part first, then image_url parts.
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == raw
    image_parts = [p for p in parts if p["type"] == "image_url"]
    assert len(image_parts) >= 1
    assert all(
        p["image_url"]["url"].startswith("data:image/png;base64,")
        for p in image_parts
    )
    saved_paths = [Path(p) for p in multimodal["image_paths"]]
    assert tmp_path / "render.png" in saved_paths


def test_build_cad_build_multimodal_content_skips_render_false(tmp_path: Path):
    """render=false builds return only metrics + preview.stl; no inline
    images should be attached so the agent request stays small."""
    from agent.tool_results import build_cad_build_multimodal_content

    metrics_payload = {
        "ok": True,
        "tool": "cad_build_and_verify",
        "data": {
            "metrics": {"solid_count": 1, "is_valid": True},
            "preview": "preview.stl",
            "render": None,
            "summary": "metrics-only.",
        },
    }
    (tmp_path / "preview.stl").write_bytes(b"solid demo\n")

    raw = json.dumps(metrics_payload, ensure_ascii=False)
    assert build_cad_build_multimodal_content(raw, tmp_path) is None


def test_build_cad_build_multimodal_content_handles_missing_artifacts(
    tmp_path: Path,
):
    """A render=true payload without on-disk artifacts must collapse to a
    plain JSON tool result instead of crashing the agent loop."""
    from agent.tool_results import build_cad_build_multimodal_content

    payload = {
        "ok": True,
        "tool": "cad_build_and_verify",
        "data": {"render": "render.png", "summary": "with render."},
    }
    assert (
        build_cad_build_multimodal_content(
            json.dumps(payload, ensure_ascii=False), tmp_path
        )
        is None
    )


def test_cad_build_tool_message_carries_inline_images(tmp_path: Path):
    """``process_tool_call`` must emit a multimodal ``tool`` message when
    the underlying build produced a render. Regression coverage for the
    single-session review flow (sub-tool re-derivation is no longer needed)."""
    from agent.dispatcher import process_tool_call

    project = tmp_path / "demo"
    project.mkdir(parents=True)
    (project / "render.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")

    class _FakeCad:
        def __init__(self, project_dir):
            self.project_dir = project_dir

        def build_and_verify(self, render=True):
            return {
                "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
                "feature_summary": {},
                "preview": "preview.stl",
                "render": "render.png",
                "model_sha256": "a" * 64,
                "preview_sha256": "b" * 64,
                "summary": "Solid 1 (valid); with render.",
            }

        with_call_id = lambda self, call_id: self
        stop = lambda self: None

    class _FakeTools:
        def __init__(self, project_dir):
            self.cad = _FakeCad(project_dir)
            self.project_dir = project_dir

    captured: list[dict] = []

    def _append(_project_dir, message):
        captured.append(message)

    messages: list[dict] = []
    process_tool_call(
        _FakeTools(project),
        "demo",
        project,
        {"id": "call-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
        publish=lambda *a, **kw: None,
        register_preview=lambda *a, **kw: "preview-1",
        append_message=_append,
    )

    assert captured, "tool message must be persisted to the conversation log"
    persisted = captured[-1]
    assert persisted["role"] == "tool"
    assert persisted["tool_call_id"] == "call-1"
    assert isinstance(persisted["content"], list)
    image_parts = [
        p for p in persisted["content"] if p.get("type") == "image_url"
    ]
    assert image_parts, "render=true must attach inline image evidence"
    # The persisted image paths index lets the history redactor replace
    # the artifacts with a placeholder on subsequent loads.
    index_path = project / ".agent_tool_images.json"
    assert index_path.is_file()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["call-1"] == ["render.png", "preview.stl"]
