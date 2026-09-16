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

