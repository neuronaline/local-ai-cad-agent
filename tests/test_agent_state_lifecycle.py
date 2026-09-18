import threading
from pathlib import Path
from unittest.mock import MagicMock

from agent.core import AgentRunner
from agent.settings import Settings
from app import create_app


def test_is_running_cleans_up_dead_thread(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # Simulate a thread that finished or crashed without finally setting _run_complete
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()

    with runner._lock:
        runner._thread = dead_thread
        runner._run_complete.clear()
        runner._active_project = "my-project"

    assert not runner.is_running()
    assert runner._thread is None
    assert runner._run_complete.is_set()
    assert runner.active_project() is None
    assert not runner.has_active_state_for("my-project")


def test_stop_aborts_active_client_and_clears_state(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    mock_client = MagicMock()
    mock_tools = MagicMock()

    with runner._lock:
        runner._active_project = "proj"
        runner._active_client = mock_client
        runner._active_tools = mock_tools

    project_dir = settings.workspace_root / "proj"
    project_dir.mkdir(parents=True, exist_ok=True)
    state_file = project_dir / ".agent_state.json"
    state_file.write_text('{"waiting_question": {"id": "q1"}}')

    affected = runner.stop("proj")

    assert affected == ["proj"]
    assert runner._stop_event.is_set()
    mock_client.abort.assert_called_once()
    mock_tools.stop.assert_called_once()
    assert not state_file.exists()


def test_delete_and_reset_stop_active_project(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    flask_app = create_app(settings)
    client = flask_app.test_client()

    created = client.post("/api/projects/new", json={"name": "test-box"})
    assert created.status_code == 201

    runner = flask_app.config["AGENT_RUNNER"]

    # Pretend a run is active on test-box
    mock_client = MagicMock()
    with runner._lock:
        runner._active_project = "test-box"
        runner._active_client = mock_client
        # An alive thread that exits when stop_event is set
        stop_evt = runner._stop_event
        def worker():
            stop_evt.wait(timeout=5)
        t = threading.Thread(target=worker)
        t.start()
        runner._thread = t
        runner._run_complete.clear()

    assert runner.has_active_state_for("test-box")

    # Reset project should stop the active task and succeed
    res = client.post("/api/projects/test-box/reset")
    assert res.status_code == 200
    assert res.get_json()["reset"] is True
    assert not runner.has_active_state_for("test-box")

    # Re-simulate active run and test delete
    with runner._lock:
        runner._stop_event.clear()
        runner._active_project = "test-box"
        runner._active_client = mock_client
        stop_evt = runner._stop_event
        def worker2():
            stop_evt.wait(timeout=5)
        t2 = threading.Thread(target=worker2)
        t2.start()
        runner._thread = t2
        runner._run_complete.clear()

    assert runner.has_active_state_for("test-box")

    # Delete project should stop the active task and delete
    del_res = client.delete("/api/projects/test-box")
    assert del_res.status_code == 200
    assert del_res.get_json()["deleted"] is True
    assert not (settings.workspace_root / "test-box").exists()


def test_question_validator_number_accepts_json_int_float_string() -> None:
    """Plan Issue 3 — ``QuestionValidator._validate_multi`` must coerce JSON numbers to strings.

    When the web UI submits a ``question`` form, the multi-question payload is a
    JSON dict of answers. Numeric fields can arrive as JSON ints (``25``), JSON
    floats (``3.14``), or string-encoded numbers with units (``"25mm"``). The
    validator must accept all three; it must also reject ``bool`` (a Python
    ``int`` subclass that would otherwise sneak through) and ``None``.
    """
    from agent.tools.question_validator import QuestionValidator

    question = {
        "questions": [
            {
                "id": "length",
                "question": "Enter the length",
                "input_type": "number",
                "required": True,
            }
        ]
    }

    # JSON-encoded int → coerced to "25" → accepted
    assert QuestionValidator.validate(question, '{"length": 25}')
    # JSON-encoded float → coerced to "3.14" → accepted
    assert QuestionValidator.validate(question, '{"length": 3.14}')
    # JSON-encoded float with trailing zeros
    assert QuestionValidator.validate(question, '{"length": 0.5}')
    # JSON-encoded negative int → coerced to "-12" → accepted
    assert QuestionValidator.validate(question, '{"length": -12}')
    # String-encoded number with unit → accepted directly
    assert QuestionValidator.validate(question, '{"length": "25mm"}')

    # bool is technically an int subclass in Python but is never a legitimate
    # numeric answer — the coercion block excludes it.
    assert not QuestionValidator.validate(question, '{"length": true}')
    assert not QuestionValidator.validate(question, '{"length": false}')

    # None / missing / non-numeric strings → rejected.
    assert not QuestionValidator.validate(question, '{"length": null}')
    assert not QuestionValidator.validate(question, '{"length": "abc"}')

    # Optional questions must accept empty / null without failing the whole batch.
    optional_question = {
        "questions": [
            {
                "id": "depth",
                "question": "Enter depth (optional)",
                "input_type": "number",
                "required": False,
            }
        ]
    }
    assert QuestionValidator.validate(optional_question, '{"depth": null}')
    assert QuestionValidator.validate(optional_question, '{"depth": ""}')


def test_stop_all_projects_clears_all_states(tmp_path: Path) -> None:
    """runner.stop(None) stops the active project and purges state across all workspace projects."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    p1 = settings.workspace_root / "proj-one"
    p2 = settings.workspace_root / "proj-two"
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)

    (p1 / ".agent_state.json").write_text('{"status": "WAITING_FOR_USER"}')
    (p2 / ".agent_state.json").write_text('{"status": "WAITING_FOR_USER"}')

    mock_client = MagicMock()
    with runner._lock:
        runner._active_project = "proj-one"
        runner._active_client = mock_client

    affected = runner.stop(None)

    assert "proj-one" in affected
    assert "proj-two" in affected
    mock_client.abort.assert_called_once()
    assert not (p1 / ".agent_state.json").exists()
    assert not (p2 / ".agent_state.json").exists()
    assert runner.active_project() is None


def test_start_rejects_when_running_or_waiting(tmp_path: Path) -> None:
    """runner.start returns False if another run is active or a question is waiting."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reject when already running
    mock_thread = MagicMock()
    with runner._lock:
        runner._thread = mock_thread
        runner._run_complete.clear()

    assert runner.start("test-box", "hello") is False

    # 2. Reject when question is waiting
    with runner._lock:
        runner._thread = None
        runner._run_complete.set()

    state_file = project_dir / ".agent_state.json"
    state_file.write_text(
        '{"status": "WAITING_FOR_USER", "waiting_question": {"questions": [{"id": "q1", "question": "Diam?", "input_type": "number", "required": true}]}}'
    )
    assert runner.start("test-box", "hello") is False


def test_answer_validation_and_state_transition(tmp_path: Path, monkeypatch) -> None:
    """runner.answer validates the input, rejects bad answers, and starts execution on valid answers."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Calling answer when no question is waiting returns False
    assert runner.answer("test-box", "10mm") is False

    # Seed waiting question
    state_file = project_dir / ".agent_state.json"
    state_file.write_text(
        '{"status": "WAITING_FOR_USER", "waiting_question": {"questions": [{"id": "q1", "question": "Diam?", "input_type": "number", "required": true}]}}'
    )

    # An invalid answer fails validation and preserves waiting state
    assert runner.answer("test-box", "not-a-number") is False
    assert runner.waiting_question("test-box") is not None
    assert state_file.exists()

    # Stub _start_locked so we verify answer transition without running a full LLM thread
    monkeypatch.setattr(runner, "_start_locked", lambda proj, msg, imgs: True)
    # A valid answer succeeds, unlinks state file, and clears waiting question
    assert runner.answer("test-box", "10mm") is True
    assert runner.waiting_question("test-box") is None
    assert not state_file.exists()


