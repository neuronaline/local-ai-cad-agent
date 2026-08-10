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
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "file_write",
                                        "arguments": json.dumps(
                                            {
                                                "filename": "summary.md",
                                                "content": "# Confirmed\n",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Model plan is ready."}}
            ]
        }


class FakeQuestionClient:
    def __init__(self, _settings):
        pass

    def chat(self, _messages, _tools):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "question-1",
                                "function": {
                                    "name": "question",
                                    "arguments": json.dumps(
                                        {
                                            "question": "What hole diameter should I use?",
                                            "input_type": "number",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }


class FakeFinalClient:
    def __init__(self, _settings):
        pass

    def chat(self, _messages, _tools):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Thanks, continuing."}}
            ]
        }


class FailedCadRecoveryClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, _messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "cad-1",
                                    "function": {
                                        "name": "cad_build_and_verify",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I can correct the failed model.",
                    }
                }
            ]
        }


class MultiToolCadClient:
    instances: ClassVar[list] = []

    def __init__(self, _settings):
        self.calls = 0
        self.second_messages = None
        self.instances.append(self)

    def chat(self, messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "cad-1",
                                    "function": {
                                        "name": "cad_build_and_verify",
                                        "arguments": "{}",
                                    },
                                },
                                {
                                    "id": "file-1",
                                    "function": {
                                        "name": "file_read",
                                        "arguments": '{"filename":"summary.md"}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        self.second_messages = list(messages)
        return {"choices": [{"message": {"role": "assistant", "content": "Reviewed."}}]}


class LiveFinalClient:
    def __init__(self, _settings):
        self.last_usage = None
        self.stream_callback = None

    def chat(self, _messages, _tools):
        self.stream_callback({"type": "content", "delta": "Live response."})
        return {
            "choices": [{"message": {"role": "assistant", "content": "Live response."}}]
        }


def test_agent_does_not_complete_without_a_new_drawing(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeOpenRouterClient(_settings))
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda kind, data: events.append((kind, data)),
    )

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
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeQuestionClient(_settings))
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )

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
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeFinalClient(_settings))
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    runner._waiting_questions["demo"] = {
        "title": "",
        "questions": [{"id": "q1", "question": "Hole size?", "input_type": "text"}],
    }

    assert runner.answer("demo", "6 mm")
    runner._thread.join(timeout=1)
    assert runner.waiting_question("demo") is None
    assert '"content": "6 mm"' in (project_dir / "conversation.jsonl").read_text(
        encoding="utf-8"
    )


def test_waiting_question_survives_runner_recreation_and_validates_answer(
    tmp_path: Path, monkeypatch
):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    settings = Settings(
        project_root, "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeQuestionClient(_settings))
    AgentRunner(settings, lambda *_: None)._run("demo", "Make a bracket")

    recreated = AgentRunner(settings, lambda *_: None)
    question = recreated.waiting_question("demo")
    assert question is not None
    assert question["questions"][0]["input_type"] == "number"
    assert not recreated.answer("demo", "not a number")
    assert recreated.waiting_question("demo") is not None


def test_agent_tool_schema_and_dispatch_forbid_export(tmp_path: Path):
    names = {tool["function"]["name"] for tool in TOOL_SCHEMAS}
    assert "cad_build_and_verify" in names
    assert {
        "file_read",
        "file_write",
        "file_replace",
        "file_regex_replace",
        "terminal_run",
        "terminal_check",
        "experience_search",
        "experience_add",
        "experience_update",
    } <= names
    assert {"cad", "file", "terminal", "experience"}.isdisjoint(names)
    assert all(
        "operation" not in schema["function"]["parameters"]["properties"]
        for schema in TOOL_SCHEMAS
    )
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    with pytest.raises(AttributeError):
        runner._execute(
            ProjectTools(project, lambda *_: None), "demo", "cad_export", {}
        )


def test_successful_tool_result_uses_structured_envelope(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "summary.md").write_text("# Ready\n", encoding="utf-8")
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    messages = []
    call = {
        "id": "read-1",
        "function": {
            "name": "file_read",
            "arguments": '{"filename":"summary.md"}',
        },
    }

    runner._process_tool_call(
        ProjectTools(project, lambda *_: None),
        "demo",
        project,
        call,
        False,
        None,
        None,
        messages,
    )

    payload = json.loads(messages[-1]["content"])
    assert payload == {"ok": True, "tool": "file_read", "data": "# Ready\n"}


def test_missing_tool_argument_returns_structured_validation_error(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    messages = []
    call = {
        "id": "write-1",
        "function": {
            "name": "file_write",
            "arguments": '{"filename":"model.py"}',
        },
    }

    runner._process_tool_call(
        ProjectTools(project, lambda *_: None),
        "demo",
        project,
        call,
        False,
        None,
        None,
        messages,
    )

    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is False
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["phase"] == "validation"


def test_malformed_tool_call_returns_structured_error(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    messages = []

    runner._process_tool_call(
        ProjectTools(project, lambda *_: None),
        "demo",
        project,
        {"function": {}},
        False,
        None,
        None,
        messages,
    )

    assert messages[-1]["tool_call_id"].startswith("invalid-")
    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is False
    assert payload["error"]["code"] == "TOOL_EXECUTION_FAILED"


def test_normalize_tool_calls_repairs_protocol_identifiers():
    normalized = AgentRunner._normalize_tool_calls(
        [{"function": {"name": "file_read", "arguments": "{}"}}, "invalid"]
    )

    assert [call["function"]["name"] for call in normalized] == [
        "file_read",
        "unknown_tool",
    ]
    assert all(call["id"].startswith("invalid-") for call in normalized)


def test_failed_cad_run_is_not_reported_as_completed(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FailedCadRecoveryClient(_settings))
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


def test_protocol_history_is_append_only_and_preserves_tool_call_content(
    tmp_path: Path, monkeypatch
):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeOpenRouterClient(_settings))
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )

    runner._run("demo", "Make a bracket")
    initial = (project / "conversation.jsonl").read_text(encoding="utf-8")
    messages = runner._context(project, "Add a chamfer", [])

    assert messages[0]["role"] == "system"
    assert [message["role"] for message in messages[1:]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert (
        (project / "conversation.jsonl").read_text(encoding="utf-8").startswith(initial)
    )


def test_legacy_history_is_loaded_from_conversation_jsonl(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    transcript = (
        '{"role":"user","content":"Make a bracket"}\n'
        '{"role":"assistant","content":"I will make it."}\n'
    )
    (project / "conversation.jsonl").write_text(transcript, encoding="utf-8")

    history = AgentRunner._load_history(project)

    assert history == [
        {"role": "user", "content": "Make a bracket"},
        {"role": "assistant", "content": "I will make it."},
    ]
    assert (project / "conversation.jsonl").read_text(encoding="utf-8") == transcript


def test_configured_tool_call_limit_stops_the_agent_loop(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FakeOpenRouterClient(_settings))
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
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: LiveFinalClient(_settings))
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda kind, data: events.append((kind, data)),
    )

    runner._run("demo", "Hello")

    event_types = [kind for kind, _data in events]
    assert event_types.index("agent_content_delta") < event_types.index(
        "agent_stream_end"
    )


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


def test_failed_build_clears_preview_and_returns_structured_error(tmp_path: Path):
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    tools = ProjectTools(project, lambda *_: None)
    tools.cad.build_and_verify = lambda: (_ for _ in ()).throw(
        RuntimeError("broken model")
    )
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    messages = []
    run_call = {
        "id": "run-1",
        "function": {"name": "cad_build_and_verify", "arguments": "{}"},
    }

    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools, "demo", project, run_call, False, "old-preview", None, messages
    )

    assert preview_id is None
    assert error == "broken model"
    assert fix_required

    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is False
    assert payload["error"]["code"] == "CAD_BUILD_FAILED"
    assert payload["error"]["phase"] == "build"
