import json
from pathlib import Path

import pytest

from agent.tools.experience_tool import ExperienceTool

# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tool(tmp_path: Path) -> ExperienceTool:
    return ExperienceTool(tmp_path, "test-project")


@pytest.fixture
def populated(tool: ExperienceTool) -> ExperienceTool:
    tool.add(
        "build123d fillet fails on sharp edges",
        "Apply a small chamfer first, then fillet the chamfered edges.",
        ["build123d", "geometry", "fillet"],
    )
    tool.add(
        "preview.stl is empty after cad.run",
        "Check that `result` is exported as a solid, not a Compound.",
        ["build123d", "export"],
    )
    return tool


# ── initial directory and file creation ───────────────────────────────


def test_creates_memory_dir_and_file_on_first_write(tool: ExperienceTool, tmp_path: Path):
    memory_dir = tmp_path / ".agent-memory"
    assert not memory_dir.exists()
    tool.add("empty preview", "Ensure result is a Part, not Compound.", ["cad"])
    assert memory_dir.is_dir()
    memory_file = memory_dir / "past_issues.json"
    assert memory_file.is_file()
    data = json.loads(memory_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["issues"]) == 1


def test_read_returns_empty_list_when_file_missing(tool: ExperienceTool):
    assert tool.search("anything") == {"matches": []}


def test_read_returns_empty_list_for_empty_file(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True, exist_ok=True)
    tool._memory_file.write_text("", encoding="utf-8")
    assert tool.search("anything") == {"matches": []}


def test_empty_json_whitespace_file_returns_empty(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True, exist_ok=True)
    tool._memory_file.write_text("\n\n", encoding="utf-8")
    assert tool.search("anything") == {"matches": []}


# ── search ─────────────────────────────────────────────────────────────


def test_search_finds_by_problem(populated: ExperienceTool):
    result = populated.search("fillet fails")
    assert len(result["matches"]) >= 1
    assert any("fillet" in r["problem"] for r in result["matches"])


def test_search_finds_by_solution(populated: ExperienceTool):
    result = populated.search("chamfer first then fillet")
    assert len(result["matches"]) >= 1
    assert any("chamfer" in r["solution"] for r in result["matches"])


def test_search_finds_by_tag(populated: ExperienceTool):
    result = populated.search("export")
    assert len(result["matches"]) >= 1
    assert any("export" in r["tags"] for r in result["matches"])


def test_search_is_case_insensitive(populated: ExperienceTool):
    result = populated.search("FILLET")  # Uppercase query
    assert len(result["matches"]) >= 1


def test_search_returns_empty_for_no_match(populated: ExperienceTool):
    result = populated.search("zirconium catalyst")
    assert result == {"matches": []}


def test_search_rejects_empty_query(tool: ExperienceTool):
    with pytest.raises(ValueError, match="empty"):
        tool.search("")


def test_search_limits_results(tool: ExperienceTool):
    for i in range(10):
        tool.add(f"problem number {i}", f"solution number {i}", ["tag"])
    result = tool.search("problem solution")
    assert len(result["matches"]) <= 5
    # Results should be sorted by score (descending)
    scores = [r["_score"] for r in result["matches"]]
    assert scores == sorted(scores, reverse=True)


# ── add ────────────────────────────────────────────────────────────────


def test_add_creates_record_with_required_fields(tool: ExperienceTool):
    result = tool.add("short problem", "short solution")
    assert "id" in result
    assert not result["updated"]
    issues = tool._read()
    assert len(issues) == 1
    record = issues[0]
    assert record["problem"] == "short problem"
    assert record["solution"] == "short solution"
    assert record["tags"] == []
    assert record["projects"] == ["test-project"]
    assert record["reuse_count"] == 0
    assert "created_at" in record
    assert "last_seen_at" in record


def test_add_detects_duplicate_and_updates(tool: ExperienceTool):
    first = tool.add("fillet fails on sharp edges", "Use chamfer first.", ["build123d"])
    second = tool.add("fillet failing on sharp edges", "Better solution: chamfer first", ["build123d", "geometry"])
    assert second["id"] == first["id"]
    assert second["updated"] is True
    issues = tool._read()
    assert len(issues) == 1
    assert issues[0]["solution"] == "Better solution: chamfer first"
    assert issues[0]["reuse_count"] == 1
    assert set(issues[0]["tags"]) == {"build123d", "geometry"}


def test_add_duplicate_adds_new_project(tool: ExperienceTool):
    first = tool.add("exact same problem text", "sol a")
    tool2 = ExperienceTool(tool._workspace_root, "other-project")
    second = tool2.add("exact same problem text", "sol b")
    assert second["id"] == first["id"]
    assert second["updated"] is True
    issues = tool._read()
    assert len(issues) == 1  # merged
    assert set(issues[0]["projects"]) == {"test-project", "other-project"}


def test_add_rejects_empty_problem(tool: ExperienceTool):
    with pytest.raises(ValueError, match="problem cannot be empty"):
        tool.add("", "solution")


def test_add_rejects_empty_solution(tool: ExperienceTool):
    with pytest.raises(ValueError, match="solution cannot be empty"):
        tool.add("problem", "")


def test_add_rejects_overlong_problem(tool: ExperienceTool):
    long_problem = "x" * 501
    with pytest.raises(ValueError, match="exceeds 500"):
        tool.add(long_problem, "ok")


def test_add_rejects_overlong_solution(tool: ExperienceTool):
    with pytest.raises(ValueError, match="exceeds 2000"):
        tool.add("ok", "x" * 2001)


def test_add_rejects_too_many_tags(tool: ExperienceTool):
    with pytest.raises(ValueError, match="Maximum 10"):
        tool.add("problem", "solution", [f"tag{i}" for i in range(11)])


def test_add_rejects_overlong_tag(tool: ExperienceTool):
    with pytest.raises(ValueError, match="exceeds 50"):
        tool.add("problem", "solution", ["x" * 51])


def test_add_normalizes_tags(tool: ExperienceTool):
    tool.add("p", "s", ["  Build123d  ", "GEOMETRY", "build123d"])
    issues = tool._read()
    assert issues[0]["tags"] == ["build123d", "geometry"]


def test_add_strips_fields(tool: ExperienceTool):
    tool.add("  padded problem  ", "  padded solution  ")
    issues = tool._read()
    assert issues[0]["problem"] == "padded problem"
    assert issues[0]["solution"] == "padded solution"


# ── update ─────────────────────────────────────────────────────────────


def test_update_changes_solution(tool: ExperienceTool):
    result = tool.add("original problem", "original solution")
    tool.update(result["id"], solution="corrected solution")
    issues = tool._read()
    assert issues[0]["solution"] == "corrected solution"
    assert issues[0]["reuse_count"] == 1


def test_update_changes_tags(tool: ExperienceTool):
    result = tool.add("problem", "solution", ["old-tag"])
    tool.update(result["id"], tags=["new-tag"])
    issues = tool._read()
    assert issues[0]["tags"] == ["new-tag"]


def test_update_updates_usage_metadata(tool: ExperienceTool):
    result = tool.add("problem", "solution")
    original_last_seen = tool._read()[0]["last_seen_at"]
    tool.update(result["id"], tags=["tag"])
    updated = tool._read()[0]
    assert updated["reuse_count"] == 1
    assert updated["last_seen_at"] >= original_last_seen


def test_update_adds_project(tool: ExperienceTool):
    result = tool.add("problem", "solution")
    tool2 = ExperienceTool(tool._workspace_root, "other-project")
    tool2.update(result["id"], tags=["tag"])
    issues = tool._read()
    assert set(issues[0]["projects"]) == {"test-project", "other-project"}


def test_update_rejects_missing_id(tool: ExperienceTool):
    with pytest.raises(ValueError, match="Record id is required"):
        tool.update("", solution="x")


def test_update_rejects_nonexistent_id(tool: ExperienceTool):
    with pytest.raises(ValueError, match="No record found"):
        tool.update("nonexistent-id", solution="x")


def test_update_rejects_invalid_fields(tool: ExperienceTool):
    result = tool.add("problem", "solution")
    with pytest.raises(ValueError, match="problem cannot be empty"):
        tool.update(result["id"], problem="")


# ── malformed JSON preservation ────────────────────────────────────────


def test_preserves_malformed_json(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True, exist_ok=True)
    tool._memory_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        tool.search("anything")
    # File should still contain the original malformed content
    assert tool._memory_file.read_text(encoding="utf-8") == "{not valid json"


def test_rejects_non_object_root(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True, exist_ok=True)
    tool._memory_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="object"):
        tool.search("anything")


def test_rejects_non_list_issues(tool: ExperienceTool):
    tool._memory_dir.mkdir(parents=True, exist_ok=True)
    tool._memory_file.write_text('{"version": 1, "issues": "not-a-list"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="array"):
        tool.search("anything")


# ── atomic writes ──────────────────────────────────────────────────────


def test_atomic_write_no_tmp_leftover(tool: ExperienceTool):
    tool.add("problem", "solution")
    tmp_files = list(tool._memory_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


# ── id uniqueness ──────────────────────────────────────────────────────


def test_each_record_has_unique_id(tool: ExperienceTool):
    ids = set()
    for i in range(20):
        result = tool.add(f"problem {i}", f"solution {i}")
        ids.add(result["id"])
    assert len(ids) == 20


# ── tool schema integration ───────────────────────────────────────────


def test_tool_schema_includes_experience():
    from agent.core import TOOL_SCHEMAS
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert "experience" in names


def test_experience_schema_has_three_operations():
    from agent.core import TOOL_SCHEMAS
    schema = next(
        s for s in TOOL_SCHEMAS if s["function"]["name"] == "experience"
    )
    operations = schema["function"]["parameters"]["properties"]["operation"]["enum"]
    assert set(operations) == {"search", "add", "update"}


# ── prompt rules ───────────────────────────────────────────────────────


def test_prompt_includes_experience_memory_rules():
    from agent.prompt import SYSTEM_PROMPT
    assert "<experience_memory>" in SYSTEM_PROMPT
    assert "Search experience memory proactively" in SYSTEM_PROMPT
    assert "failed attempts, or unverified workarounds" in SYSTEM_PROMPT


# ── project listing excludes agent-memory ──────────────────────────────


def test_agent_memory_dir_not_visible_as_project():
    from app import PROJECT_NAME_RE
    assert PROJECT_NAME_RE.fullmatch(".agent-memory") is None


# ── integration: ProjectTools wires experience tool ────────────────────


def test_project_tools_has_experience_attribute(tmp_path: Path):
    from agent.core import ProjectTools

    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    (project_dir / "workspace-root").mkdir()

    def noop(_type: str, _data: dict) -> None:
        pass

    tools = ProjectTools(project_dir, noop)
    assert hasattr(tools, "experience")
    assert tools.experience._project_name == "my-project"
    assert tools.experience._workspace_root == project_dir.parent


# ── search short words are ignored ─────────────────────────────────────


def test_search_ignores_short_words(tool: ExperienceTool):
    tool.add("build123d fillet fails on sharp edges", "Use chamfer first.", ["build123d"])
    # "on" has 2 chars, should be ignored; only "fillet" (6) and "sharp" (5) and "edges" (5) match
    result = tool.search("on")
    assert result == {"matches": []}


def test_search_partial_match_scores(tool: ExperienceTool):
    tool.add("fillet fails on sharp edges", "Use chamfer.", ["build123d"])
    tool.add("export stl fails", "Check solid.", ["export"])
    # "fillet" only matches first record
    result = tool.search("fillet")
    assert len(result["matches"]) == 1
    assert result["matches"][0]["_score"] == 1.0
