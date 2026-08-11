import json
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest

from agent.core import TOOL_SCHEMAS, AgentRunner, ProjectTools
from agent.settings import Settings
from agent.tool_results import failure as tool_failure


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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeOpenRouterClient(_settings),
    )
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeQuestionClient(_settings),
    )
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeFinalClient(_settings),
    )
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
    log_text = (project_dir / "conversation.jsonl").read_text(encoding="utf-8")
    # The answer is persisted exactly once via the canonical history sink.
    assert log_text.count('"role":"user","content":"6 mm"') == 1
    assert '"content": "6 mm"' not in log_text


def test_answer_persists_user_message_before_thread_starts(tmp_path: Path, monkeypatch):
    """Regression: answer() must write the user message before starting the run.

    Previously the new run thread was started first and the user message was
    appended afterwards. The thread's _context() could read the log before
    the append, miss the answer, and re-append it as a duplicate entry. The
    fix persists the answer first (still under the runner lock) and only
    then starts the thread.
    """
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")

    # Block the new thread at the very start of _run so it cannot read the
    # log until the main thread has had a chance to write.
    proceed = threading.Event()
    original_context = agent.core.AgentRunner._context

    def blocked_context(self, project_dir, message, image_paths):
        proceed.wait(timeout=5)
        return original_context(self, project_dir, message, image_paths)

    monkeypatch.setattr(agent.core.AgentRunner, "_context", blocked_context)

    class _NoopClient:
        def __init__(self, _settings):
            self.stop_event = threading.Event()
            self.session_id = ""
            self.last_usage = None
            self.stream_callback = lambda event: None

        def chat(self, _messages, _tools):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: _NoopClient(_settings),
    )

    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    runner._waiting_questions["demo"] = {
        "title": "",
        "questions": [{"id": "q1", "question": "Size?", "input_type": "text"}],
    }
    assert runner.answer("demo", "6 mm")
    # At this point the new thread is blocked in _context. The user message
    # must already be on disk so the thread, when unblocked, reads it once.
    log_text = (project_dir / "conversation.jsonl").read_text(encoding="utf-8")
    assert log_text.count('"role":"user","content":"6 mm"') == 1

    proceed.set()
    runner._thread.join(timeout=2)
    log_text = (project_dir / "conversation.jsonl").read_text(encoding="utf-8")
    assert log_text.count('"role":"user","content":"6 mm"') == 1


def test_answer_waits_for_previous_run_to_finish(tmp_path: Path, monkeypatch):
    """answer() must block until a prior run's finally has cleared thread state."""
    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")

    class _SlowClient:
        def __init__(self, _settings):
            self.stop_event = threading.Event()
            self.session_id = ""
            self.last_usage = None
            self.stream_callback = lambda event: None

        def chat(self, _messages, _tools):
            # Block long enough that the previous run is still in flight when
            # answer() is invoked. Returning a no-tool-call response lets the
            # loop exit and the finally reset _thread/_run_complete.
            time.sleep(0.5)
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    import agent.core

    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: _SlowClient(_settings),
    )
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_: None,
    )
    # Start a run that will sleep in the LLM client.
    assert runner.start("demo", "build something", [])
    runner._waiting_questions["demo"] = {
        "title": "",
        "questions": [{"id": "q1", "question": "Size?", "input_type": "text"}],
    }

    # Calling answer() while the previous run is still alive must still
    # complete successfully (it waits on _run_complete, not on is_alive()).
    started = time.time()
    assert runner.answer("demo", "6 mm")
    elapsed = time.time() - started
    assert elapsed >= 0.4, (
        f"answer() returned too quickly ({elapsed:.2f}s); did it wait?"
    )
    # And after the run finishes, thread state is fully reset.
    runner._thread.join(timeout=2)
    assert runner._thread is None
    assert runner._run_complete.is_set()


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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeQuestionClient(_settings),
    )
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


def test_tool_schemas_document_side_effects_and_reject_extra_arguments():
    schemas = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}

    for schema in schemas.values():
        assert schema["function"]["parameters"]["additionalProperties"] is False

    write_description = schemas["file_write"]["function"]["description"]
    build_description = schemas["cad_build_and_verify"]["function"]["description"]
    terminal_arguments = schemas["terminal_run"]["function"]["parameters"][
        "properties"
    ]["arguments"]
    question_parameters = schemas["question"]["function"]["parameters"]

    assert "overwrites the whole file" in write_description
    assert "do not repeat unless the source changes" in build_description
    assert terminal_arguments["minItems"] == terminal_arguments["maxItems"] == 2
    assert question_parameters["properties"]["questions"]["minItems"] == 1
    assert (
        question_parameters["properties"]["questions"]["items"]["additionalProperties"]
        is False
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FailedCadRecoveryClient(_settings),
    )
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeOpenRouterClient(_settings),
    )
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeOpenRouterClient(_settings),
    )
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
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: LiveFinalClient(_settings),
    )
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


def test_debug_error_log_records_recoverable_tool_failures(tmp_path: Path):
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    settings = Settings(
        project_root,
        "https://example.test",
        "test",
        1,
        "127.0.0.1",
        5000,
        agent_debug_log_tool_errors=True,
    )
    runner = AgentRunner(settings, lambda *_: None)
    error = ValueError("Invalid Python: unexpected indent (line 12)")

    runner._debug_tool_error(
        project,
        "write-1",
        "file_write",
        error,
        tool_failure("file_write", error),
    )

    entries = (project / "debug-errors.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(entries[0])
    assert record["call_id"] == "write-1"
    assert record["tool"] == "file_write"
    assert record["classification"]["code"] == "VALIDATION_ERROR"
