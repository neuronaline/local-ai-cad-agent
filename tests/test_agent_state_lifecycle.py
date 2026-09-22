import threading
from pathlib import Path
from unittest.mock import MagicMock

from agent.core import AgentRunner, _RunState, _TurnOutcome
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
    monkeypatch.setattr(runner, "_start_locked", lambda proj, msg, imgs, run_id: True)
    # A valid answer succeeds, unlinks state file, and clears waiting question
    assert runner.answer("test-box", "10mm") is True
    assert runner.waiting_question("test-box") is None
    assert not state_file.exists()


# ---------------------------------------------------------------------------
# Concurrency isolation for input images
# ---------------------------------------------------------------------------


def test_take_run_inputs_moves_files_under_lock(tmp_path: Path) -> None:
    """``AgentRunner._take_run_inputs`` relocates uploads to a run-scoped dir.

    A split-brain scenario exists when Request B's speculative
    ``unlink`` of files in ``<project>/inputs/`` races against Request
    A's runner still reading them. Giving the runner transactional
    ownership fixes this: as soon as ``start()`` runs, inputs live
    under ``<project>/.cad-agent/runs/<run_id>/inputs/`` and the
    original staging area is empty.
    """
    from agent.core import _CAD_AGENT_DIRNAME, _RUN_INPUTS_DIRNAME, _RUNS_DIRNAME

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)

    staging = project_dir / "inputs"
    staging.mkdir()
    source_a = staging / "first.png"
    source_b = staging / "second.png"
    source_a.write_bytes(b"AAAA")
    source_b.write_bytes(b"BBBB")

    run_id = "test-run-1"
    new_paths = runner._take_run_inputs("test-box", [source_a, source_b], run_id)

    expected_dir = (
        project_dir
        / _CAD_AGENT_DIRNAME
        / _RUNS_DIRNAME
        / run_id
        / _RUN_INPUTS_DIRNAME
    )
    assert expected_dir.is_dir()
    assert len(new_paths) == 2
    # The runner's view of the files lives entirely under the run-scoped
    # directory; the original staging area is gone.
    assert {path.parent for path in new_paths} == {expected_dir}
    assert not source_a.exists()
    assert not source_b.exists()
    # Files were moved, not copied — bytes survive unchanged.
    assert {path.read_bytes() for path in new_paths} == {b"AAAA", b"BBBB"}


def test_take_run_inputs_rolls_back_on_failure(tmp_path: Path) -> None:
    """A missing source must abort cleanly without losing the rest of the batch.

    ``shutil.move`` deletes the source before the destination is
    committed, so an unguarded move loop would silently lose every
    sibling processed before the failing one. The implementation walks
    the inputs first and refuses to start the move when any source is
    gone — the original uploads are preserved so the next request can
    pick them up unchanged.
    """
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)
    staging = project_dir / "inputs"
    staging.mkdir()
    good_source = staging / "first.png"
    good_source.write_bytes(b"AAAA")
    missing_source = staging / "ghost.png"
    # The second source is created in the staging area, then deleted
    # before ``_take_run_inputs`` runs so it is missing on entry. The
    # implementation must abort before any move starts; ``first.png``
    # therefore survives intact and the run directory is never created.
    missing_source.write_bytes(b"BBBB")
    missing_source.unlink()

    run_id = "test-rollback"
    new_paths = runner._take_run_inputs(
        "test-box", [good_source, missing_source], run_id
    )

    from agent.core import _CAD_AGENT_DIRNAME, _RUNS_DIRNAME

    run_dir = project_dir / _CAD_AGENT_DIRNAME / _RUNS_DIRNAME / run_id

    assert new_paths == []
    # No run directory exists because the move never started.
    assert not run_dir.exists()
    # The original upload survives so a retry can pick it up unchanged.
    assert good_source.exists()
    assert good_source.read_bytes() == b"AAAA"


def test_discard_run_directory_removes_only_run_scope(tmp_path: Path) -> None:
    """``_discard_run_directory`` is scoped to a single run.

    The runner must never accidentally remove a sibling run's directory
    or the parent ``.cad-agent`` state directory while cleaning up after
    one run. The fix surfaces when two concurrent runs would otherwise
    collide at the same path — UUID collisions are astronomically
    unlikely, but the cleanup must still be precise.
    """
    from agent.core import _CAD_AGENT_DIRNAME, _RUNS_DIRNAME

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)
    runs_root = project_dir / _CAD_AGENT_DIRNAME / _RUNS_DIRNAME

    keep_dir = runs_root / "keep"
    keep_dir.mkdir(parents=True, exist_ok=True)
    discard_dir = runs_root / "discard"
    discard_dir.mkdir(parents=True, exist_ok=True)
    (discard_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (discard_dir / "inputs" / "stale.png").write_bytes(b"STALE")

    runner._discard_run_directory("test-box", "discard")

    assert not discard_dir.exists()
    assert keep_dir.is_dir(), "Sibling run directory must not be removed"
    assert runs_root.is_dir(), "Runs root must persist across run cleanup"


def test_run_input_ownership_removes_inputs_on_terminal_state(tmp_path: Path) -> None:
    """End-to-end: ``_run``'s ``finally`` removes the run-scoped inputs dir.

    The runner owns cleanup at terminal states — never the HTTP
    handler. This test runs ``_run`` to completion via a stubbed tool
    loop that finishes immediately, then verifies the run directory is
    gone after the thread joins.
    """
    from agent.core import _CAD_AGENT_DIRNAME, _RUNS_DIRNAME

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)
    project_dir = settings.workspace_root / "test-box"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Seed the run-scoped directory as ``_take_run_inputs`` would.
    run_id = "cleanup-test"
    run_dir = (
        project_dir / _CAD_AGENT_DIRNAME / _RUNS_DIRNAME / run_id
    )
    run_inputs_dir = run_dir / "inputs"
    run_inputs_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_inputs_dir / "upload.png"
    artifact.write_bytes(b"PNG-BYTES")

    # Stub the LLM client so the loop produces one final assistant turn
    # and exits. The relevant contract is the ``finally`` block; everything
    # between thread start and the terminal state is incidental and we
    # want this test to focus exclusively on cleanup.
    class _StubClient:
        def __init__(self) -> None:
            self.stop_event = None
            self.session_id = None
            self.activity_logger = None
            self.run_id = None
            self.last_usage = None
            self.last_image_fallback_used = False
            self.stream_callback = None
            self.preserve_reasoning = False
            self.require_images = False

        def chat(self, messages, tools=None):  # type: ignore[no-untyped-def]
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "All done.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        def abort(self) -> None:  # pragma: no cover - never invoked
            return None

    import agent.core as core_module
    real_create_llm_client = core_module.create_llm_client
    core_module.create_llm_client = lambda settings: _StubClient()  # type: ignore[assignment]
    real_append_message = core_module.AgentRunner._append_message
    core_module.AgentRunner._append_message = lambda self, project_dir, msg: None  # type: ignore[assignment]
    # ``publish`` is wired into ``AgentRunner.__init__`` as an instance
    # attribute (the EventBus callback). Override it on the runner we are
    # about to exercise so the stub loop runs without touching real SSE
    # state.
    real_publish = runner.publish
    runner.publish = lambda *a, **kw: None  # type: ignore[assignment]
    real_complete = core_module.AgentRunner._complete
    core_module.AgentRunner._complete = lambda self, project, content: None  # type: ignore[assignment]
    try:
        runner._run("test-box", "build it", [], run_id=run_id)
    finally:
        core_module.create_llm_client = real_create_llm_client  # type: ignore[assignment]
        core_module.AgentRunner._append_message = real_append_message  # type: ignore[assignment]
        runner.publish = real_publish
        core_module.AgentRunner._complete = real_complete  # type: ignore[assignment]

    # The run-scoped directory must be gone after the terminal state.
    assert not run_dir.exists()
    # And the project directory itself must still exist — cleanup is
    # scoped to the run, not the project.
    assert project_dir.is_dir()
    # The artifact must not survive into a subsequent run.
    assert not artifact.exists()


def test_chat_endpoint_does_not_unlink_inputs_on_failure(tmp_path: Path) -> None:
    """``POST /api/chat`` never speculatively unlinks uploaded inputs.

    File management is intentionally stripped out of the ``chat()``
    error cleanup path. The runner owns the inputs the moment
    ``start()`` returns; a failed start leaves the originals in
    ``<project>/inputs/`` and the HTTP handler must not delete them on
    its own initiative. Cleanup, in either path, belongs to the runner.
    """
    from io import BytesIO

    from PIL import Image
    from werkzeug.datastructures import FileStorage

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "box"
    project_dir.mkdir()
    (project_dir / "project.json").write_text("{}", encoding="utf-8")

    settings = Settings(workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()

    # Build a tiny in-memory PNG and upload it.
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(0, 128, 255)).save(buf, format="PNG")
    payload = buf.getvalue()
    upload = FileStorage(
        stream=BytesIO(payload), filename="ref.png", content_type="image/png"
    )

    # Force ``runner.start`` to return False even though no thread is
    # actually running. The handler must NOT unlink the just-uploaded
    # files in that case — the design removed the speculative cleanup.
    runner = app.config["AGENT_RUNNER"]
    original_start = runner.start
    runner.start = lambda project, message, image_paths: False  # type: ignore[assignment]
    try:
        response = client.post(
            "/api/chat",
            data={
                "project": "box",
                "message": "build a flange",
                "attachments": (upload,),
            },
            content_type="multipart/form-data",
        )
    finally:
        runner.start = original_start  # type: ignore[assignment]

    assert response.status_code == 409
    # The uploaded image is still on disk; the HTTP handler does not
    # speculatively unlink. The runner, not the handler, owns the
    # cleanup — verified by the broader suite via
    # ``_run``'s ``finally`` block. ``store_images`` re-encodes the PNG
    # (optimize=True, RGB conversion) so byte-exact comparison is not
    # possible; check the file is non-empty PNG bytes instead.
    inputs_dir = project_dir / "inputs"
    assert inputs_dir.is_dir(), "Upload must persist on a rejected start"
    remaining = list(inputs_dir.iterdir())
    assert len(remaining) == 1
    saved_bytes = remaining[0].read_bytes()
    assert saved_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(saved_bytes) > 0


# ---------------------------------------------------------------------------
# _run_turn outcome coverage
# ---------------------------------------------------------------------------


def _make_run_state(tmp_path: Path, **overrides) -> _RunState:
    """Build a minimal ``_RunState`` for direct ``_handle_no_tool_calls`` tests.

    Only the fields the helper actually reads need to be realistic; the
    rest stay at their dataclass defaults so each test stays focused on
    one branch.
    """
    state = _RunState(
        project_dir=tmp_path,
        messages=[],
        tools=MagicMock(),
        client=MagicMock(),
        activity_logger=None,
        run_id="run-test",
        project="proj",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_handle_no_tool_calls_completes_when_no_tools_used(tmp_path: Path) -> None:
    """A text-only turn with no tool calls and no preview must COMPLETE."""
    publish = MagicMock()
    runner = AgentRunner(
        Settings(
            tmp_path / "projects", "https://x", "m", 1, "127.0.0.1", 5000
        ),
        publish,
    )
    state = _make_run_state(tmp_path)

    outcome = runner._handle_no_tool_calls(
        state, {"role": "assistant", "content": "Done."}
    )

    assert outcome is _TurnOutcome.COMPLETED
    # ``_complete`` publishes ``agent_message`` then a terminal
    # ``agent_status``; both must reach the UI for the success path.
    assert any(
        call.args and call.args[0] == "agent_message"
        for call in publish.call_args_list
    )


def test_handle_no_tool_calls_returns_drawing_not_created_on_cad_error(
    tmp_path: Path,
) -> None:
    """A no-preview turn that carries a CAD error must terminate as ``DRAWING_NOT_CREATED``."""
    publish = MagicMock()
    runner = AgentRunner(
        Settings(
            tmp_path / "projects", "https://x", "m", 1, "127.0.0.1", 5000
        ),
        publish,
    )
    state = _make_run_state(tmp_path, cad_error="syntax error")

    outcome = runner._handle_no_tool_calls(
        state, {"role": "assistant", "content": ""}
    )

    assert outcome is _TurnOutcome.DRAWING_NOT_CREATED
    # The diagnostic must reach the UI before the loop bails out.
    assert any(
        call.args and call.args[0] == "agent_error"
        for call in publish.call_args_list
    )


def test_handle_no_tool_calls_emits_final_verification_missing_after_nudge(
    tmp_path: Path,
) -> None:
    """A verification nudge that never landed must trip the terminal gate."""
    publish = MagicMock()
    runner = AgentRunner(
        Settings(
            tmp_path / "projects", "https://x", "m", 1, "127.0.0.1", 5000
        ),
        publish,
    )
    # preview_id present + cad_fix_required still true + nudge already sent.
    state = _make_run_state(
        tmp_path,
        preview_id="prev-1",
        cad_fix_required=True,
        nudged_final_verification=True,
    )

    outcome = runner._handle_no_tool_calls(
        state, {"role": "assistant", "content": "looks good"}
    )

    assert outcome is _TurnOutcome.FINAL_VERIFICATION_MISSING
    assert any(
        call.args and call.args[0] == "agent_error"
        for call in publish.call_args_list
    )


