"""Critical coverage for reusable agent experience memory."""

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
        "  build123d fillet fails on sharp edges  ",
        "  Apply a chamfer before the fillet.  ",
        [" Build123d ", "GEOMETRY", "build123d"],
    )

    record = tool._read()[0]
    assert created == {"id": record["id"], "updated": False}
    assert record["problem"] == "build123d fillet fails on sharp edges"
    assert record["solution"] == "Apply a chamfer before the fillet."
    assert record["tags"] == ["build123d", "geometry"]
    assert record["projects"] == ["project-a"]
    assert record["reuse_count"] == 0
    assert tool._memory_file.is_file()

    for query in ("FILLET FAILS", "chamfer", "geometry"):
        matches = tool.search(query)["matches"]
        assert [match["id"] for match in matches] == [record["id"]]
    assert tool.search("unrelated zirconium catalyst") == {"matches": []}


def test_duplicate_experience_merges_learning_across_projects(
    tool: ExperienceTool,
):
    first = tool.add(
        "fillet fails on sharp edges",
        "Use a chamfer first.",
        ["build123d"],
    )
    other_project = ExperienceTool(tool._workspace_root, "project-b")

    merged = other_project.add(
        "fillet failing on sharp edges",
        "Use a smaller chamfer before filleting.",
        ["geometry"],
    )

    record = tool._read()[0]
    assert merged == {"id": first["id"], "updated": True}
    assert record["solution"] == "Use a smaller chamfer before filleting."
    assert record["tags"] == ["build123d", "geometry"]
    assert record["projects"] == ["project-a", "project-b"]
    assert record["reuse_count"] == 1


def test_update_changes_existing_experience_and_tracks_reuse(
    tool: ExperienceTool,
):
    created = tool.add("export fails", "Check the solid.", ["export"])

    result = tool.update(
        created["id"],
        solution="Validate the solid before export.",
        tags=["CAD", "export"],
    )

    record = tool._read()[0]
    assert result == {"id": created["id"], "updated": True}
    assert record["solution"] == "Validate the solid before export."
    assert record["tags"] == ["cad", "export"]
    assert record["reuse_count"] == 1


def test_invalid_memory_inputs_are_rejected(tool: ExperienceTool):
    invalid_calls = (
        lambda: tool.search(""),
        lambda: tool.add("", "solution"),
        lambda: tool.add("problem", ""),
        lambda: tool.update("missing-id", solution="new solution"),
    )

    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()


def test_malformed_memory_is_never_overwritten(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True)
    original = "{not valid json"
    tool._memory_file.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed"):
        tool.add("problem", "solution")

    assert tool._memory_file.read_text(encoding="utf-8") == original


def test_agent_exposes_operation_specific_experience_tools(tmp_path: Path):
    from agent.core import TOOL_SCHEMAS, ProjectTools

    schemas = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}
    assert {
        "experience_search",
        "experience_add",
        "experience_update",
    } <= schemas.keys()
    assert schemas["experience_search"]["function"]["parameters"]["required"] == [
        "query"
    ]
    assert schemas["experience_add"]["function"]["parameters"]["required"] == [
        "problem",
        "solution",
    ]

    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    project_tools = ProjectTools(project_dir, lambda *_: None)
    assert project_tools.experience._project_name == "demo"
    assert project_tools.experience._workspace_root == tmp_path
