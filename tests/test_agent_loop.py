import dataclasses
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from agent.core import AgentRunner, ProjectTools
from agent.settings import Settings
from agent.tool_results import failure as tool_failure
from agent.tool_schemas import TOOL_SCHEMAS


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
                                        "name": "write_file",
                                        "arguments": json.dumps(
                                            {
                                                "filename": "model.py",
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


def test_new_task_resets_review_cycle_count(tmp_path: Path, monkeypatch):
    """Auto-review is gone; the agent no longer maintains ``_review_cycles``.

    Keep a smoke test so the migration is intentional: starting a new task
    on a runner that previously tracked review cycles no longer raises
    ``AttributeError``.
    """
    import agent.core

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: FakeFinalClient(_settings),
    )
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_args, **_kwargs: None,
    )
    assert not hasattr(runner, "_review_cycles")
    runner._run("demo", "Start a new task")
    assert not hasattr(runner, "_review_cycles")


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
                                        "name": "read_file",
                                        "arguments": '{"filename":"model.py"}',
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


class CancelledClient:
    def __init__(self, _settings):
        self.stop_event = None
        self.session_id = None
        self.stream_callback = None

    def chat(self, _messages, _tools):
        from agent.llm_base import RequestCancelled

        raise RequestCancelled("OpenRouter request cancelled.")


def test_agent_treats_cancelled_llm_request_as_a_stopped_task(tmp_path: Path, monkeypatch):
    import agent.core

    project_root = tmp_path / "projects"
    project_dir = project_root / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "conversation.jsonl").write_text("", encoding="utf-8")
    events = []
    monkeypatch.setattr(
        agent.core,
        "create_llm_client",
        lambda _settings, _ignored=None: CancelledClient(_settings),
    )
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )

    runner._run("demo", "Make a bracket")

    assert ("agent_status", {
        "project": "demo", "status": "stopped", "message": "Task stopped."
    }) in events
    assert not any(kind == "agent_error" for kind, _data in events)


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
        lambda *args, **_: events.append((args[0], args[1])),
    )

    runner._run("demo", "Make a bracket")

    assert (project_dir / "model.py").read_text(encoding="utf-8") == "# Confirmed\n"
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
        lambda *_, **__: None,
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
        lambda *_, **__: None,
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
        lambda *_, **__: None,
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
        lambda *_, **__: None,
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
    AgentRunner(settings, lambda *_, **__: None)._run("demo", "Make a bracket")

    recreated = AgentRunner(settings, lambda *_, **__: None)
    question = recreated.waiting_question("demo")
    assert question is not None
    assert question["questions"][0]["input_type"] == "number"
    assert not recreated.answer("demo", "not a number")
    assert recreated.waiting_question("demo") is not None


def test_agent_tool_schema_and_dispatch_forbid_export(tmp_path: Path):
    names = {tool["function"]["name"] for tool in TOOL_SCHEMAS}
    assert "cad_build_and_verify" in names
    assert {"cad", "file"}.isdisjoint(names)
    assert all(
        "operation" not in schema["function"]["parameters"]["properties"]
        for schema in TOOL_SCHEMAS
    )
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
    )
    with pytest.raises(AttributeError):
        runner._execute(
            ProjectTools(project, lambda *_, **__: None), "demo", "cad_export", {}
        )


def test_project_state_tells_agent_to_create_a_missing_model(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()

    state = AgentRunner._project_state_message(project)

    assert state["role"] == "user"
    assert "model.py does not exist" in state["content"]
    assert "write_file" in state["content"]
    assert "read_file" in state["content"]


def test_tool_schemas_document_side_effects_and_reject_extra_arguments():
    schemas = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}

    for schema in schemas.values():
        assert schema["function"]["parameters"]["additionalProperties"] is False

    write_description = schemas["write_file"]["function"]["description"]
    build_description = schemas["cad_build_and_verify"]["function"]["description"]
    review_description = schemas["cad_review"]["function"]["description"]
    screenshot_description = schemas["cad_screenshot"]["function"]["description"]
    question_parameters = schemas["question"]["function"]["parameters"]

    assert "deliberately replace its entire contents" in write_description
    assert "call cad_review separately if you want a verdict" in build_description
    assert "never use it as a routine final step for a small local edit" in review_description
    assert "do not use it for routine small edits" in screenshot_description
    assert question_parameters["properties"]["questions"]["minItems"] == 1
    assert (
        question_parameters["properties"]["questions"]["items"]["additionalProperties"]
        is False
    )


def test_tool_schemas_expose_safe_file_tools():
    file_tools = {
        schema["function"]["name"]
        for schema in TOOL_SCHEMAS
        if schema["function"]["name"]
        in {"file_read", "file_write", "file_replace", "file_regex_replace"}
    }
    assert file_tools == set()

    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert {"read_file", "write_file", "edit_file", "insert_file"} <= names

    build = next(
        schema for schema in TOOL_SCHEMAS
        if schema["function"]["name"] == "cad_build_and_verify"
    )
    checks = build["function"]["parameters"]["properties"]["parameter_checks"]
    assert checks["items"]["additionalProperties"] is False


def test_agent_loop_runs_read_file_then_edit_file(tmp_path: Path):
    """A complete read_file → edit_file round-trip on model.py must mutate
    the file, commit a revision, and surface a structured successful
    envelope on both calls.
    """
    project = tmp_path / "demo"
    project.mkdir()
    model = project / "model.py"
    model.write_text("# alpha\n# beta\n# gamma\n", encoding="utf-8")

    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
    )

    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    messages = []

    runner._process_tool_call(
        ProjectTools(project, lambda *_, **__: None),
        "demo",
        project,
        {
            "id": "edit-1",
            "function": {
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "filename": "model.py",
                        "old_string": "# beta",
                        "new_string": "# B",
                        "expected_sha256": digest,
                    }
                ),
            },
        },
        False,
        None,
        None,
        messages,
    )

    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is True
    assert payload["tool"] == "edit_file"
    assert model.read_text(encoding="utf-8") == "# alpha\n# B\n# gamma\n"


def test_agent_loop_runs_read_file_then_write_file(tmp_path: Path):
    """A complete read_file → write_file round-trip on model.py must
    overwrite the file when the digest matches.
    """
    project = tmp_path / "demo"
    project.mkdir()
    model = project / "model.py"
    model.write_text("# alpha\n", encoding="utf-8")

    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
    )

    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    messages = []

    runner._process_tool_call(
        ProjectTools(project, lambda *_, **__: None),
        "demo",
        project,
        {
            "id": "write-1",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {
                        "filename": "model.py",
                        "content": "# replaced\n",
                        "expected_sha256": digest,
                    }
                ),
            },
        },
        False,
        None,
        None,
        messages,
    )

    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is True
    assert payload["tool"] == "write_file"
    assert model.read_text(encoding="utf-8") == "# replaced\n"


def test_successful_tool_result_uses_structured_envelope(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "model.py").write_text("# Ready\n", encoding="utf-8")
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
    )
    messages = []
    call = {
        "id": "read-1",
        "function": {
            "name": "read_file",
            "arguments": '{"filename":"model.py"}',
        },
    }

    runner._process_tool_call(
        ProjectTools(project, lambda *_, **__: None),
        "demo",
        project,
        call,
        False,
        None,
        None,
        messages,
    )

    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is True
    assert payload["tool"] == "read_file"
    read_result = json.loads(payload["data"])
    assert read_result["content"] == "# Ready\n"
    assert read_result["exists"] is True
    assert len(read_result["sha256"]) == 64


def test_missing_tool_argument_returns_structured_validation_error(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    runner = AgentRunner(
        Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
    )
    messages = []
    call = {
        "id": "edit-1",
        "function": {
            "name": "edit_file",
            "arguments": '{"filename":"model.py","old_string":"x"}',
        },
    }

    runner._process_tool_call(
        ProjectTools(project, lambda *_, **__: None),
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
        lambda *_, **__: None,
    )
    messages = []

    runner._process_tool_call(
        ProjectTools(project, lambda *_, **__: None),
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
        [{"function": {"name": "read_file", "arguments": "{}"}}, "invalid"]
    )

    assert [call["function"]["name"] for call in normalized] == [
        "read_file",
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
        lambda *args, **_: events.append((args[0], args[1])),
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
        lambda *_, **__: None,
    )

    runner._run("demo", "Make a bracket")
    initial = (project / "conversation.jsonl").read_text(encoding="utf-8")
    messages = runner._context(project, "Add a chamfer", [])

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "model.py exists" in messages[1]["content"]
    assert [message["role"] for message in messages[2:]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
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
        lambda *args, **_: events.append((args[0], args[1])),
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
        lambda *args, **_: events.append((args[0], args[1])),
    )

    runner._run("demo", "Hello")

    event_types = [kind for kind, _data in events]
    assert event_types.index("agent_content_delta") < event_types.index(
        "agent_stream_end"
    )


def test_model_edit_after_completion_does_not_rewind_published_message(tmp_path: Path):
    """A late model edit cannot retroactively invalidate a published completion.

    With the preview-ACK state machine gone, the agent's canonical
    ``agent_message`` is published the moment the STL is verified; later
    file edits yield a new turn / revision and do not rewind the prior
    state. This test locks in the new contract.
    """
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"preview")
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )
    preview_id = runner._register_preview("demo", project)
    assert preview_id
    # Even if the model changes after the preview is registered, no error
    # event is fired retroactively: the success side is finalised via
    # ``agent_message``/``agent_status:completed`` rather than parked.
    (project / "model.py").write_text("result = 2\n", encoding="utf-8")
    runner._complete("demo", "Done")

    assert not any(kind == "agent_error" for kind, _data in events)
    assert events[-1][0] == "agent_status"
    assert events[-1][1]["status"] == "completed"


def test_failed_build_clears_preview_and_returns_structured_error(tmp_path: Path):
    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: (_ for _ in ()).throw(
        RuntimeError("broken model")
    )
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *_, **__: None,
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
    runner = AgentRunner(settings, lambda *_, **__: None)
    error = ValueError("Invalid Python: unexpected indent (line 12)")

    runner._debug_tool_error(
        project,
        "write-1",
        "write_file",
        error,
        tool_failure("write_file", error),
    )

    entries = (project / "debug-errors.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(entries[0])
    assert record["call_id"] == "write-1"
    assert record["tool"] == "write_file"
    assert record["classification"]["code"] == "VALIDATION_ERROR"


def test_activity_log_records_tool_call_start_and_result(tmp_path: Path):
    """Enabling ``agent_log_tool_activity`` writes tool-call events to the
    per-project activity log when ``_process_tool_call`` runs."""
    from agent.activity_log import ActivityLogger

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
        agent_log_tool_activity=True,
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None, settings)
    logger = ActivityLogger(project)
    runner._active_activity_logger = logger
    runner._active_run_id = "run-test"
    call = {
        "id": "call-1",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"filename": "model.py"}),
        },
    }
    messages: list[dict] = []

    # ``read_file`` may legitimately fail because model.py does not exist
    # yet — the activity logger still has to record *both* start and
    # result events regardless of success/failure, so we discard any
    # exception from the dispatcher.
    runner._process_tool_call(
        tools,
        "demo",
        project,
        call,
        cad_fix_required=True,
        prev_preview_id=None,
        cad_error=None,
        messages=messages,
    )

    entries = (project / ".cad-agent" / "activity.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    events = [json.loads(line) for line in entries]
    kinds = [entry["event"] for entry in events]
    assert "tool_call_start" in kinds
    assert "tool_call_result" in kinds
    for entry in events:
        assert entry["run_id"] == "run-test"
    start = next(e for e in events if e["event"] == "tool_call_start")
    assert start["data"]["tool"] == "read_file"
    assert start["data"]["call_id"] == "call-1"


def test_activity_log_disabled_by_default_does_not_write(tmp_path: Path):
    """Without the flag, no activity.jsonl is created."""
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
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None, settings)
    call = {
        "id": "call-1",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"filename": "model.py"}),
        },
    }
    messages: list[dict] = []

    # Dispatcher may raise because model.py is absent; the contract we test
    # is that the absence of the flag suppresses *all* logging.
    runner._process_tool_call(
        tools,
        "demo",
        project,
        call,
        cad_fix_required=True,
        prev_preview_id=None,
        cad_error=None,
        messages=messages,
    )

    assert not (project / ".cad-agent" / "activity.jsonl").exists()


# --------------------------------------------------------------------------- #
# Review gate integration
# --------------------------------------------------------------------------- #


class _FakeReviewClient:
    """Returns a fixed verdict; counts how many times it was invoked."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.call_count = 0

    def chat(self, messages, tools=None):
        self.call_count += 1
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "function": {"name": "submit_review", "arguments": json.dumps(self._verdict)}
        }]}}]}


def _seed_pass_build(project: Path) -> dict[str, object]:
    """Pretend ``cad_build_and_verify`` produced a passing-ready manifest."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    Image.new("RGB", (2, 2), "white").save(project / "render.png")
    render_bytes = (project / "render.png").read_bytes()
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    review_root = project / ".cad-agent" / "reviews" / ("a" * 64)
    review_root.mkdir(parents=True)
    Image.new("RGB", (4, 2), "white").save(review_root / "review-sheet.png")
    sheet_bytes = (review_root / "review-sheet.png").read_bytes()
    return {
        "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
        "feature_summary": {"through_hole_count": 0},
        "preview": "preview.stl",
        "render": "render.png",
        "review_manifest": {
            "model_sha256": "a" * 64,
            "preview_sha256": "b" * 64,
            "views": [{"view_id": "x_positive"}, {"view_id": "isometric_positive"}],
            "contact_sheet": {
                "path": "review-sheet.png",
                "image_sha256": hashlib.sha256(sheet_bytes).hexdigest(),
            },
            "single_render": {
                "path": "render.png",
                "image_sha256": hashlib.sha256(render_bytes).hexdigest(),
            },
        },
        "review": "a" * 64,
    }


def test_review_gate_passes_for_passing_verdict(tmp_path: Path, monkeypatch):
    """Auto-review is gone: a successful ``cad_build_and_verify`` now
    registers the preview directly and never invokes the reviewer."""
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: payload

    review_client = _FakeReviewClient(
        {"status": "pass", "summary": "All good.", "findings": []}
    )
    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: review_client
    )

    messages: list[dict] = []
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
    )

    assert preview_id is not None, "successful build must register a preview"
    assert error is None
    assert fix_required is False
    # No auto-review: the reviewer is never invoked from cad_build_and_verify.
    assert review_client.call_count == 0


def test_review_updates_the_build_call_in_place(tmp_path: Path, monkeypatch):
    """cad_build_and_verify emits a single ``running`` → ``completed`` transition.

    Auto-review's intermediate ``reviewing`` status is gone; a deliberate
    ``cad_review`` call emits its own status pair when invoked.
    """
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    events = []
    runner = AgentRunner(settings, lambda kind, data, **_: events.append((kind, data)))
    tools = ProjectTools(project, lambda kind, data, **_: events.append((kind, data)))
    tools.cad.build_and_verify = lambda render=True: payload
    monkeypatch.setattr(
        "agent.cad_review._default_create_client",
        lambda _settings: _FakeReviewClient(
            {"status": "pass", "summary": "All good.", "findings": []}
        ),
    )

    runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        [],
    )

    build_events = [
        data for kind, data in events
        if kind == "tool_status" and data.get("call_id") == "run-1"
    ]
    assert [event["status"] for event in build_events] == ["running", "completed"]
    assert not any(
        kind == "tool_status" and data.get("status") == "reviewing"
        for kind, data in events
    )


def test_review_gate_blocks_completion_on_blocking_finding(tmp_path: Path, monkeypatch):
    """Auto-review is gone, so a blocking finding no longer blocks completion.

    The legacy test asserted ``preview_id is None`` and ``fix_required is True``
    because the agent loop used the verdict to force a regenerate. With
    auto-review removed, a successful ``cad_build_and_verify`` returns a
    preview unconditionally; the agent invokes ``cad_review`` itself when it
    wants a verdict.
    """
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: payload

    review_client = _FakeReviewClient(
        {
            "status": "fail",
            "summary": "Missing through hole.",
            "findings": [
                {
                    "severity": "blocking",
                    "category": "missing_feature",
                    "view": "isometric_positive",
                    "message": "Through hole is missing.",
                }
            ],
        }
    )
    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: review_client
    )

    messages: list[dict] = []
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
    )

    assert preview_id is not None, "build path no longer gates on review verdict"
    assert error is None
    assert fix_required is False
    assert review_client.call_count == 0


def test_review_gate_treats_inconclusive_as_blocking(tmp_path: Path, monkeypatch):
    """Auto-review is gone: ``inconclusive`` is no longer surfaced as a build failure.

    The agent now calls ``cad_review`` explicitly; ``inconclusive`` only
    affects that call's return value, not the build loop.
    """
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: payload

    class InconclusiveClient:
        def chat(self, messages, tools=None):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: InconclusiveClient()
    )

    messages: list[dict] = []
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
    )

    assert preview_id is not None
    assert error is None
    assert fix_required is False


def test_review_gate_respects_cycle_limit(tmp_path: Path, monkeypatch):
    """The per-task ``review_max_cycles`` knob is gone.

    Verify ``Settings`` no longer carries the field and the runner does not
    maintain a cycle counter.
    """
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000)
    assert "review_max_cycles" not in settings.__dataclass_fields__
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: payload

    review_client = _FakeReviewClient(
        {
            "status": "fail",
            "summary": "Still wrong.",
            "findings": [
                {
                    "severity": "blocking",
                    "category": "geometry",
                    "message": "Wrong shape.",
                }
            ],
        }
    )
    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: review_client
    )

    messages: list[dict] = []
    preview_id, error, _fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
    )

    assert preview_id is not None
    assert error is None
    assert not hasattr(runner, "_review_cycles")
    # Reviewer is never invoked by cad_build_and_verify — review is opt-in.
    assert review_client.call_count == 0


def test_history_truncation_never_starts_with_an_orphan_tool_message():
    from agent.conversation import ConversationStore

    history = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ] + [{"role": "user", "content": "latest"}] * 99

    truncated = ConversationStore._truncate(history)

    assert len(truncated) < ConversationStore.MAX_HISTORY
    assert truncated[0]["role"] != "tool"


def test_review_gate_disabled_skips_review_call(tmp_path: Path, monkeypatch):
    """``review_enabled=False`` still skips the canonical eight-view render.

    Auto-review is gone in every configuration, but the build-time
    rendering knobs (``review_render_workers``, ``review_required_views``,
    ``render`` argument) still apply. ``review_enabled=False`` is preserved
    as the historical "metrics-only build" switch.
    """
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = dataclasses.replace(
        Settings(tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000),
        review_enabled=False,
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True: payload

    called = {"count": 0}

    def fake_factory(_settings):
        called["count"] += 1
        raise AssertionError("review client should not be created when disabled")

    monkeypatch.setattr("agent.cad_review._default_create_client", fake_factory)

    messages: list[dict] = []
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {"id": "run-1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        False,
        None,
        None,
        messages,
    )

    assert preview_id is not None
    assert error is None
    assert fix_required is False
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# Final-verification gate: cad_fix_required is cleared by a successful
# render=true build whose multimodal content actually attached an image,
# and stays True otherwise.
# ---------------------------------------------------------------------------


def test_final_verification_clears_when_render_attaches_image(tmp_path: Path):
    """Happy path: render=true with a real on-disk PNG clears ``cad_fix_required``."""
    project = tmp_path / "projects" / "demo"
    payload = _seed_pass_build(project)
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True, **__: payload

    messages: list[dict] = []
    # ``cad_fix_required=True`` (as if the model was just edited); a render=true
    # build with an attached image must reset it to False.
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {
            "id": "run-1",
            "function": {
                "name": "cad_build_and_verify",
                "arguments": json.dumps({"render": True}),
            },
        },
        True,
        None,
        None,
        messages,
    )

    assert preview_id is not None
    assert error is None
    assert fix_required is False


def test_final_verification_stays_required_when_render_image_missing(tmp_path: Path):
    """render=true with no decodable PNG must keep ``cad_fix_required=True``.

    The agent must be unable to claim success without inline image evidence.
    A render that produced no readable artifact (cache miss, corrupt PNG,
    contact-sheet build failure) is functionally the same as a render=false
    build from the verification-gate's perspective.
    """
    project = tmp_path / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    # Note: no render.png and no .cad-agent/reviews/* — multimodal content
    # is None, so the verification gate must remain armed.
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, lambda *_, **__: None)
    tools = ProjectTools(project, lambda *_, **__: None)
    tools.cad.build_and_verify = lambda render=True, **__: {
        "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
        "feature_summary": {"through_hole_count": 0},
        "preview": "preview.stl",
        "render": "render.png" if True else None,
        "validation_results": [],
    }

    messages: list[dict] = []
    preview_id, error, fix_required, _waiting = runner._process_tool_call(
        tools,
        "demo",
        project,
        {
            "id": "run-1",
            "function": {
                "name": "cad_build_and_verify",
                "arguments": json.dumps({"render": True}),
            },
        },
        True,
        None,
        None,
        messages,
    )

    assert preview_id is not None
    assert error is None
    assert fix_required is True, (
        "render=true with no attached image must leave cad_fix_required=True"
    )


def test_final_verification_gate_nudges_then_stops(tmp_path: Path, monkeypatch):
    """End-to-end: render=true that never attaches an image triggers one
    nudge, then the agent stops with the documented ``agent_error`` message.

    The agent sees a model.py that needs verification. It calls
    ``cad_build_and_verify(render=true)``, gets a build that fails to attach
    any image, and tries to deliver a final answer. The loop nudges once;
    on the next final-answer attempt, the gate hard-stops.
    """
    import agent.core
    from agent.tools.cad_tool import CadTool

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    # No render.png on disk and no review dir — multimodal content is None.
    (project / "conversation.jsonl").write_text("", encoding="utf-8")

    # Replace CadTool.build_and_verify at the class level so the runner's
    # internal ProjectTools() instance also sees the no-image stub.
    no_image_payload = {
        "metrics": {"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}},
        "feature_summary": {"through_hole_count": 0},
        "preview": "preview.stl",
        "render": "render.png",  # claimed, but the file is absent
        "validation_results": [],
    }

    def no_image_build_and_verify(self, render=True, parameter_checks=None, **__):
        return no_image_payload

    monkeypatch.setattr(CadTool, "build_and_verify", no_image_build_and_verify)

    class AlwaysRenderNoImage:
        """Calls cad_build_and_verify(render=true) twice, then finalises.

        Every cad_build call returns a build that fails to attach an image
        (no render.png on disk), so the verification gate must keep
        flagging the model as unverified.
        """

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
                                            "arguments": json.dumps({"render": True}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            if self.calls == 2:
                # After the nudge, call cad_build_and_verify(render=true)
                # again — still no image, gate stays armed.
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "cad-2",
                                        "function": {
                                            "name": "cad_build_and_verify",
                                            "arguments": json.dumps({"render": True}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            # Third call: the agent finally produces a final message.
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Final answer.",
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        agent.core, "create_llm_client", lambda _settings, _ignored=None: AlwaysRenderNoImage(_settings)
    )
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )
    runner._run("demo", "Build a bracket")

    # No agent_message was published (the agent stopped with an error).
    assert not any(kind == "agent_message" for kind, _data in events)
    # The terminal agent_error must reference the final-verification gate.
    assert any(
        kind == "agent_error"
        and "final visual verification is still missing" in data["message"]
        for kind, data in events
    ), events
    # A nudge user-message was appended to the conversation log.
    history = (project / "conversation.jsonl").read_text(encoding="utf-8")
    assert "has not passed final visual verification" in history


def test_build_failure_stop_after_six_consecutive_failures(
    tmp_path: Path, monkeypatch
):
    """Six consecutive ``cad_build_and_verify`` failures trip the stop-loop.

    The agent should NOT be allowed to loop on a build that is structurally
    broken; the contract is "stop after 6 cumulative failures" so the
    operator sees a clear error and can intervene.
    """
    import agent.core
    from agent.tools.cad_tool import CadTool

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    (project / "conversation.jsonl").write_text("", encoding="utf-8")

    def failing_build(self, render=False, parameter_checks=None, **__):
        raise RuntimeError("NameError: name 'x' is not defined")

    monkeypatch.setattr(CadTool, "build_and_verify", failing_build)

    class AlwaysFailBuild:
        """Calls cad_build_and_verify until the loop stops."""

        def __init__(self, _settings):
            self.calls = 0

        def chat(self, _messages, _tools):
            self.calls += 1
            if self.calls <= 6:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": f"cad-{self.calls}",
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
                    {"message": {"role": "assistant", "content": "Should not reach."}}
                ]
            }

    monkeypatch.setattr(
        agent.core, "create_llm_client", lambda _settings, _ignored=None: AlwaysFailBuild(_settings)
    )
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )
    runner._run("demo", "Build something")

    assert not any(kind == "agent_message" for kind, _data in events)
    assert any(
        kind == "agent_error" and "repeated CAD build failures" in data["message"]
        for kind, data in events
    ), events


def test_build_failure_stop_after_duplicate_signature(tmp_path: Path, monkeypatch):
    """Two identical-signature failures stop the loop early.

    Six distinct failures are a hard limit, but two of the same kind should
    stop the agent faster: there is no point retrying a failure the agent
    cannot distinguish.
    """
    import agent.core
    from agent.tools.cad_tool import CadTool

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    (project / "conversation.jsonl").write_text("", encoding="utf-8")

    # Two identical-signature RuntimeErrors (line numbers differ in the
    # raw message, but the signature normalises them).
    line_counter = {"n": 0}

    def duplicate_failing_build(self, render=False, parameter_checks=None, **__):
        line_counter["n"] += 1
        raise RuntimeError(
            f"NameError: name 'x' is not defined (model.py, line {line_counter['n']})"
        )

    monkeypatch.setattr(CadTool, "build_and_verify", duplicate_failing_build)

    class DuplicateFailBuild:
        def __init__(self, _settings):
            self.calls = 0

        def chat(self, _messages, _tools):
            self.calls += 1
            if self.calls <= 2:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": f"cad-{self.calls}",
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
                    {"message": {"role": "assistant", "content": "Should not reach."}}
                ]
            }

    monkeypatch.setattr(
        agent.core, "create_llm_client", lambda _settings, _ignored=None: DuplicateFailBuild(_settings)
    )
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )
    runner._run("demo", "Build something")

    # Two identical signatures must trip the stop; the 6-count limit
    # is a fallback, not the primary trigger.
    assert not any(kind == "agent_message" for kind, _data in events)
    assert any(
        kind == "agent_error" and "repeated CAD build failures" in data["message"]
        for kind, data in events
    ), events
    assert line_counter["n"] <= 2, (
        f"duplicate-signature stop should fire after 2 attempts, got {line_counter['n']}"
    )


def test_image_fallback_keeps_cad_fix_required_true(tmp_path: Path, monkeypatch):
    """When the provider rejects the trailing inline render, the verification
    gate must keep ``cad_fix_required=True`` so the agent nudges/repairs.

    The test directly exercises the loop's gate-keeping branch
    (``awaiting_tool_render`` AND ``client.last_image_fallback_used``) by
    running the full loop with a fake LLM client that mimics a successful
    cad build followed by a final-answer attempt where the image fallback
    has fired. The loop must NOT publish an ``agent_message`` because the
    verification gate is still armed.
    """
    import agent.core
    from agent.tools.cad_tool import CadTool

    project_root = tmp_path / "projects"
    project = project_root / "demo"
    project.mkdir(parents=True)
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"solid demo\n")
    Image.new("RGB", (2, 2), "white").save(project / "render.png")
    (project / "conversation.jsonl").write_text("", encoding="utf-8")

    # Successful cad build with an attached image so the gate is initially
    # clearable. The trailing render then triggers a fallback on the next
    # LLM call.
    payload = _seed_pass_build(project)

    def passing_build(self, render=True, parameter_checks=None, **__):
        return payload

    monkeypatch.setattr(CadTool, "build_and_verify", passing_build)

    class FallbackClient:
        """Mimics a successful build followed by an image-fallback LLM call."""

        def __init__(self, _settings):
            self.last_image_fallback_used = True  # set by image-fallback path
            self.calls = 0

        def chat(self, _messages, _tools):
            self.calls += 1
            if self.calls == 1:
                # First turn: agent calls cad_build_and_verify(render=true).
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
                                            "arguments": json.dumps({"render": True}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            # Second turn: agent finalises. The loop's image-fallback
            # branch must keep cad_fix_required armed, so the agent
            # cannot deliver an agent_message and must instead nudge or
            # stop with an agent_error.
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Final answer.",
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        agent.core, "create_llm_client", lambda _settings, _ignored=None: FallbackClient(_settings)
    )
    events = []
    runner = AgentRunner(
        Settings(project_root, "https://example.test", "test", 1, "127.0.0.1", 5000),
        lambda *args, **_: events.append((args[0], args[1])),
    )
    runner._run("demo", "Build a bracket")

    # The image-fallback path flips cad_fix_required=True and assigns a
    # cad_error message. The agent delivers a final answer without a
    # verified render → the gate nudges once or stops. Either way, no
    # agent_message is published because the verification is incomplete.
    assert not any(
        kind == "agent_message" for kind, _data in events
    ), "no agent_message should be emitted when verification is incomplete"
