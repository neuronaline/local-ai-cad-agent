import json
from pathlib import Path
from typing import ClassVar

import pytest

from agent.core import TOOL_SCHEMAS, AgentRunner, ProjectTools
from agent.settings import Settings


class FakeOpenRouterClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, _messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "call-1", "function": {"name": "file", "arguments": json.dumps({
                    "operation": "write", "filename": "summary.md", "content": "# Confirmed\n"
                })}
            }]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "Model plan is ready."}}]}


class FakeQuestionClient:
    def __init__(self, _settings):
        pass

    def chat(self, _messages, _tools):
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "question-1", "function": {"name": "question", "arguments": json.dumps({
                "question": "What hole diameter should I use?", "input_type": "number"
            })}
        }]}}]}


class FakeFinalClient:
    def __init__(self, _settings):
        pass

    def chat(self, _messages, _tools):
        return {"choices": [{"message": {"role": "assistant", "content": "Thanks, continuing."}}]}


class FailedCadRecoveryClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, _messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "cad-1", "function": {"name": "cad", "arguments": '{"operation":"run"}'}
            }]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "I can correct the failed model."}}]}


class MultiToolCadClient:
    instances: ClassVar[list] = []

    def __init__(self, _settings):
        self.calls = 0
        self.second_messages = None
        self.instances.append(self)

    def chat(self, messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "cad-1", "function": {"name": "cad", "arguments": '{"operation":"run"}'}},
                {
                    "id": "file-1",
                    "function": {
                        "name": "file",
                        "arguments": '{"operation":"read","filename":"summary.md"}',
                    },
                },
            ]}}]}
        self.second_messages = list(messages)
        return {"choices": [{"message": {"role": "assistant", "content": "Reviewed."}}]}


class LiveFinalClient:
    def __init__(self, _settings):
        self.last_usage = None
        self.stream_callback = None

    def chat(self, _messages, _tools):
        self.stream_callback({"type": "content", "delta": "Live response."})
        return {"choices": [{"message": {"role": "assistant", "content": "Live response."}}]}


def test_agent_does_not_complete_without_a_new_drawing(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text('', encoding="utf-8")
    events = []
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeOpenRouterClient)
    runner = AgentRunner(Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000), lambda kind, data: events.append((kind, data)))

    runner._run("demo", "Make a bracket")

    assert (project_dir / "summary.md").read_text(encoding="utf-8") == "# Confirmed\n"
    assert not any(kind == "agent_message" for kind, _data in events)
    assert any(
        kind == "agent_error" and "did not produce a new CAD preview" in data["message"]
        for kind, data in events
    )


def test_question_is_persisted_for_the_next_user_reply(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text('', encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeQuestionClient)
    runner = AgentRunner(Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000), lambda *_: None)

    runner._run("demo", "Make a bracket")

    history = (project_dir / "conversation.jsonl").read_text(encoding="utf-8")
    assert "Question: What hole diameter should I use?" in history
    question = runner.waiting_question("demo")
    assert question is not None
    assert isinstance(question.get("questions"), list)
    assert question["questions"][0]["question"] == "What hole diameter should I use?"
    assert question["questions"][0]["input_type"] == "number"


def test_answer_resumes_a_waiting_agent(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text('', encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeFinalClient)
    runner = AgentRunner(Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000), lambda *_: None)
    runner._waiting_questions["demo"] = {"title": "", "questions": [{"id": "q1", "question": "Hole size?", "input_type": "text"}]}

    assert runner.answer("demo", "6 mm")
    runner._thread.join(timeout=1)
    assert runner.waiting_question("demo") is None
    assert '"content": "6 mm"' in (project_dir / "conversation.jsonl").read_text(encoding="utf-8")


def test_waiting_question_survives_runner_recreation_and_validates_answer(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    settings = Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000)
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeQuestionClient)
    AgentRunner(settings, lambda *_: None)._run("demo", "Make a bracket")

    recreated = AgentRunner(settings, lambda *_: None)
    question = recreated.waiting_question("demo")
    assert question is not None
    assert question["questions"][0]["input_type"] == "number"
    assert not recreated.answer("demo", "not a number")
    assert recreated.waiting_question("demo") is not None


def test_agent_tool_schema_and_dispatch_forbid_export(tmp_path: Path):
    cad_schema = next(tool for tool in TOOL_SCHEMAS if tool["function"]["name"] == "cad")
    assert cad_schema["function"]["parameters"]["properties"]["operation"]["enum"] == [
        "run",
        "inspect",
        "render",
    ]
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    with pytest.raises(ValueError, match="Unsupported CAD operation"):
        runner._execute(ProjectTools(project, lambda *_: None), "demo", "cad", {"operation": "export"})


def test_failed_cad_run_is_not_reported_as_completed(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", FailedCadRecoveryClient)
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda kind, data: events.append((kind, data)),
    )

    runner._run("demo", "Build a part")

    assert not any(kind == "agent_message" for kind, _data in events)
    assert any(
        kind == "agent_error" and "model.py does not exist yet" in data["message"]
        for kind, data in events
    )


def test_self_critique_is_added_after_all_tool_results(tmp_path: Path, monkeypatch):
    pytest.importorskip("build123d")
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    (project / "summary.md").write_text("# Demo\n", encoding="utf-8")
    (project / "model.py").write_text(
        "from build123d import Box\nresult = Box(10, 20, 30)\n",
        encoding="utf-8",
    )
    MultiToolCadClient.instances.clear()
    monkeypatch.setattr(agent.core, "OpenRouterClient", MultiToolCadClient)
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )

    runner._run("demo", "Review the part")

    messages = MultiToolCadClient.instances[0].second_messages
    assistant_index = next(index for index, item in enumerate(messages) if item.get("tool_calls"))
    assert [item["role"] for item in messages[assistant_index + 1 :]] == ["tool", "tool", "user"]


def test_protocol_history_is_append_only_and_preserves_tool_call_content(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeOpenRouterClient)
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )

    runner._run("demo", "Make a bracket")
    initial = (project / "api_messages.jsonl").read_text(encoding="utf-8")
    messages = runner._context(project, "Add a chamfer", [])

    assert messages[0]["role"] == "system"
    assert [message["role"] for message in messages[1:]] == ["user", "assistant", "tool", "assistant", "user"]
    assert (project / "api_messages.jsonl").read_text(encoding="utf-8").startswith(initial)


def test_configured_tool_call_limit_stops_the_agent_loop(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", FakeOpenRouterClient)
    events = []
    runner = AgentRunner(
        Settings(
            project_root,
            "https://example.test",
            "test",
            1,
            "127.0.0.1",
            5000,
            agent_tool_call_limit=1,
        ),
        lambda kind, data: events.append((kind, data)),
    )

    runner._run("demo", "Make a bracket")

    assert not any(kind == "agent_message" for kind, _data in events)
    assert any(
        kind == "agent_error" and "Tool-call limit (1) reached" in data["message"]
        for kind, data in events
    )


def test_content_delta_is_published_before_stream_end(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "OpenRouterClient", LiveFinalClient)
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda kind, data: events.append((kind, data)),
    )

    runner._run("demo", "Hello")

    event_types = [kind for kind, _data in events]
    assert event_types.index("agent_content_delta") < event_types.index("agent_stream_end")


def test_model_edit_invalidates_registered_preview(tmp_path: Path):
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"preview")
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda kind, data: events.append((kind, data)),
    )
    preview_id = runner._register_preview("demo", project)

    (project / "model.py").write_text("result = 2\n", encoding="utf-8")
    runner._await_preview("demo", preview_id, "Done")

    assert not runner.is_awaiting_preview("demo")
    assert events[-1][0] == "agent_error"


def test_failed_rerun_clears_previous_preview_and_blocks_inspect(tmp_path: Path):
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    tools = ProjectTools(project, lambda *_: None)
    tools.cad.run = lambda: (_ for _ in ()).throw(RuntimeError("broken model"))
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    messages = []
    run_call = {
        "id": "run-1",
        "function": {"name": "cad", "arguments": '{"operation":"run"}'},
    }

    preview_id, error, fix_required, _critique, _waiting = runner._process_tool_call(
        tools, "demo", project, run_call, False, "old-preview", None, messages
    )

    assert preview_id is None
    assert error == "broken model"
    assert fix_required

    inspect_call = {
        "id": "inspect-1",
        "function": {"name": "cad", "arguments": '{"operation":"inspect"}'},
    }
    runner._process_tool_call(
        tools, "demo", project, inspect_call, True, None, error, messages
    )
    assert "cad.inspect was skipped" in messages[-1]["content"]
