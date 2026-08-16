"""Critical coverage for reusable agent experience memory."""

import json
from pathlib import Path

import pytest

from agent.tools.experience_tool import ExperienceTool


@pytest.fixture
def tool(tmp_path: Path) -> ExperienceTool:
    return ExperienceTool(tmp_path, "project-a")


def test_add_persists_normalized_experience_and_searches_all_fields(
    tool: ExperienceTool,
):
    created = tool.add(
        "  Fillet on sharp edges  ",
        "  build123d fillet fails on sharp edges  ",
        "  Apply a chamfer before the fillet.  ",
        [" Build123d ", "GEOMETRY", "build123d"],
    )

    record = tool._read()[0]
    assert created == {"id": record["id"], "updated": False}
    assert record["title"] == "Fillet on sharp edges"
    assert record["problem"] == "build123d fillet fails on sharp edges"
    assert record["solution"] == "Apply a chamfer before the fillet."
    assert record["tags"] == ["build123d", "geometry"]
    assert record["projects"] == ["project-a"]
    assert record["reuse_count"] == 0
    assert tool._memory_file.is_file()
    assert json.loads(tool._memory_file.read_text(encoding="utf-8"))["version"] == 2

    for query in ("FILLET FAILS", "chamfer", "geometry"):
        matches = tool.search(query)["matches"]
        assert [match["id"] for match in matches] == [record["id"]]
    assert tool.search("unrelated zirconium catalyst") == {"matches": []}


def test_duplicate_experience_merges_learning_across_projects(
    tool: ExperienceTool,
):
    first = tool.add(
        "Fillet failure",
        "fillet fails on sharp edges",
        "Use a chamfer first.",
        ["build123d"],
    )
    other_project = ExperienceTool(tool._workspace_root, "project-b")

    merged = other_project.add(
        "Sharp-edge fillet failure",
        "fillet failing on sharp edges",
        "Use a smaller chamfer before filleting.",
        ["geometry"],
    )

    record = tool._read()[0]
    assert merged == {"id": first["id"], "updated": True}
    assert record["title"] == "Sharp-edge fillet failure"
    assert record["solution"] == "Use a smaller chamfer before filleting."
    assert record["tags"] == ["build123d", "geometry"]
    assert record["projects"] == ["project-a", "project-b"]
    assert record["reuse_count"] == 1


def test_projects_sharing_a_memory_file_share_a_lock(tool: ExperienceTool):
    other_project = ExperienceTool(tool._workspace_root, "project-b")

    assert other_project._memory_lock is tool._memory_lock


def test_update_changes_existing_experience_and_tracks_reuse(
    tool: ExperienceTool,
):
    created = tool.add("Export validation", "export fails", "Check the solid.", ["export"])

    result = tool.update(
        created["id"],
        title="Solid export validation",
        solution="Validate the solid before export.",
        tags=["CAD", "export"],
    )

    record = tool._read()[0]
    assert result == {"id": created["id"], "updated": True}
    assert record["title"] == "Solid export validation"
    assert record["solution"] == "Validate the solid before export."
    assert record["tags"] == ["cad", "export"]
    assert record["reuse_count"] == 1


@pytest.mark.parametrize(
    "invocation",
    [
        pytest.param(lambda t: t.search(""), id="search-empty"),
        pytest.param(lambda t: t.search("   \t\n "), id="search-whitespace"),
        pytest.param(lambda t: t.add("", "problem", "solution"), id="add-blank-title"),
        pytest.param(lambda t: t.add("title", "problem", "  "), id="add-blank-solution"),
        pytest.param(lambda t: t.add("\n\t ", "problem", "solution"), id="add-whitespace-title"),
        pytest.param(lambda t: t.update("missing-id", solution="new solution"), id="update-missing-id"),
        pytest.param(lambda t: t.update("   ", solution="x"), id="update-blank-id"),
        pytest.param(lambda t: t.get(""), id="get-blank-id"),
        pytest.param(lambda t: t.get("ghost"), id="get-missing-id"),
    ],
)
def test_invalid_memory_inputs_are_rejected(tool: ExperienceTool, invocation):
    """Boundary: empty/whitespace/missing inputs across search/add/update/get
    must raise ``ValueError`` so a malformed agent tool call fails fast."""
    with pytest.raises(ValueError):
        invocation(tool)


@pytest.mark.parametrize(
    ("title", "problem", "solution", "match"),
    [
        pytest.param(
            "x" * 121,
            "p",
            "s",
            "title exceeds 120 characters",
            id="oversized-title",
        ),
        pytest.param(
            "t",
            "x" * 501,
            "s",
            "problem exceeds 500 characters",
            id="oversized-problem",
        ),
        pytest.param(
            "t",
            "p",
            "x" * 2001,
            "solution exceeds 2000 characters",
            id="oversized-solution",
        ),
    ],
)
def test_add_rejects_oversized_fields(
    tool: ExperienceTool, title: str, problem: str, solution: str, match: str
):
    """Field-length caps are the contract that keeps the database queryable."""
    with pytest.raises(ValueError, match=match):
        tool.add(title, problem, solution)


def test_add_accepts_unicode_and_special_characters(tool: ExperienceTool):
    """Unicode, quotes, and embedded newlines must round-trip through the
    tool without being mangled or rejected."""
    title = "Edge 'fillet' \"bug\" — ünïcödé & <script>alert(1)</script>"
    problem = "Multi\nline\nproblem with\ttabs and \"quotes\""
    solution = "Apply a 0.5 mm chamfer; see docs/notes.md (line 12) — ✓"

    created = tool.add(title, problem, solution)
    record = tool.get(created["id"])

    # Title collapses whitespace via " ".join(value.split()) but preserves
    # the unicode/special-char content. The other fields round-trip verbatim.
    assert "ünïcödé" in record["title"]
    assert "<script>" in record["title"]
    assert record["problem"] == problem
    assert record["solution"] == solution


def test_add_rejects_too_many_tags(tool: ExperienceTool):
    """More than ``_MAX_TAGS`` unique tags must be rejected with a clear error."""
    too_many = [f"tag{i}" for i in range(11)]  # one over the 10-tag cap

    with pytest.raises(ValueError, match="Maximum 10 unique tags"):
        tool.add("Title", "Problem", "Solution", tags=too_many)


def test_add_rejects_oversized_tag(tool: ExperienceTool):
    """A single tag longer than ``_MAX_TAG_LENGTH`` must be rejected so
    queries do not return pathologically large strings."""
    oversized_tag = "a" * 51  # one over the 50-char cap

    with pytest.raises(ValueError, match="exceeds 50 characters"):
        tool.add("Title", "Problem", "Solution", tags=[oversized_tag])


@pytest.mark.parametrize(
    "bad_tags",
    [
        pytest.param("not-a-list", id="string-instead-of-list"),
        pytest.param(42, id="integer-instead-of-list"),
        pytest.param({"tag": "value"}, id="dict-instead-of-list"),
        pytest.param((1, 2), id="tuple-instead-of-list"),
    ],
)
def test_add_rejects_non_list_tags(tool: ExperienceTool, bad_tags):
    """Tags must be a list — passing any non-list, non-None value must raise
    ``ValueError`` rather than silently coerce. ``None`` is allowed (it
    means 'no tags') by ``add``."""
    with pytest.raises(ValueError, match="tags must be an array"):
        tool.add("Title", "Problem", "Solution", tags=bad_tags)


def test_add_accepts_none_tags(tool: ExperienceTool):
    """``None`` tags must be accepted (it is documented as 'no tags')."""
    created = tool.add("Title", "Problem", "Solution", tags=None)

    assert created["id"]
    assert tool.get(created["id"])["tags"] == []


def test_search_handles_unicode_content_in_problem_or_solution(tool: ExperienceTool):
    """Search must be case-insensitive over the indexed fields (problem +
    solution) and handle unicode content without errors."""
    # Arrange: the indexed fields carry unicode.
    tool.add(
        "Unicode chamfer recovery",
        "Chamfer fails on ünïcödé edge geometry",
        "Apply a smaller chamfer; verify with ✓",
    )

    # Act: a unicode-lowercase query matches the unicode content.
    result = tool.search("ünïcödé")

    # Assert: the case-insensitive search surfaces the record.
    assert result["matches"], "case-insensitive unicode query must match"


def test_malformed_memory_is_never_overwritten(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True)
    original = "{not valid json"
    tool._memory_file.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed"):
        tool.add("Title", "problem", "solution")

    assert tool._memory_file.read_text(encoding="utf-8") == original


def test_agent_exposes_operation_specific_experience_tools(tmp_path: Path):
    from agent.core import TOOL_SCHEMAS, ProjectTools

    schemas = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}
    assert {
        "experience_search",
        "experience_get",
        "experience_add",
        "experience_update",
    } <= schemas.keys()
    assert schemas["experience_search"]["function"]["parameters"]["required"] == [
        "query"
    ]
    assert schemas["experience_add"]["function"]["parameters"]["required"] == [
        "title",
        "problem",
        "solution",
    ]
    assert schemas["experience_get"]["function"]["parameters"]["required"] == [
        "id"
    ]

    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    project_tools = ProjectTools(project_dir, lambda *_: None)
    assert project_tools.experience._project_name == "demo"
    assert project_tools.experience._workspace_root == tmp_path


def test_context_index_and_get_expose_only_titles_and_ids(tool: ExperienceTool):
    created = tool.add(
        "Fillet recovery",
        "Fillet fails on sharp edges.",
        "Apply a chamfer first.",
        ["geometry"],
    )

    assert tool.context_index() == {
        "available": True,
        "issues": [{"id": created["id"], "title": "Fillet recovery"}],
    }
    assert tool.get(created["id"])["solution"] == "Apply a chamfer first."
    with pytest.raises(ValueError, match="No record found"):
        tool.get("missing")


def test_context_index_handles_empty_and_malformed_memory(tool: ExperienceTool):
    assert tool.context_index() == {"available": True, "issues": []}

    tool._memory_dir.mkdir(parents=True)
    tool._memory_file.write_text("{not valid json", encoding="utf-8")
    assert tool.context_index() == {"available": False, "issues": []}
