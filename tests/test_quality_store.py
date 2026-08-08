"""Critical coverage for CAD quality tracking."""

from pathlib import Path

import pytest

from agent.quality.store import QualityIntegrityError, QualityStore
from agent.settings import Settings
from app import create_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
    )


def test_run_and_attempt_survive_restart_with_artifacts(tmp_path: Path):
    store = QualityStore(tmp_path)
    run = store.start_run(project="demo", request_sha256="request-sha")
    attempt = store.start_attempt(
        run.run_id,
        revision_id="revision-1",
        source_sha256="source-sha",
    )
    source = tmp_path / "model.py"
    preview = tmp_path / "preview.stl"
    source.write_text("result = 1\n", encoding="utf-8")
    preview.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    store.complete_attempt(
        run.run_id,
        attempt.attempt_id,
        status="succeeded",
        phase="kernel",
        metrics={"solid_count": 1},
        artifact_paths={"source": source, "preview": preview},
    )
    store.complete_run(run.run_id, status="completed")

    reloaded = QualityStore(tmp_path)
    restored_run = reloaded.get_run(run.run_id)
    restored_attempt = reloaded.get_attempt(run.run_id, attempt.attempt_id)
    assert restored_run.status == "completed"
    assert restored_run.request_sha256 == "request-sha"
    assert restored_attempt.status == "succeeded"
    assert restored_attempt.metrics == {"solid_count": 1}
    assert restored_attempt.artifacts["preview"]["path"] == "preview.stl"
    assert restored_attempt.artifacts["source"]["sha256"]
    snapshot = restored_attempt.artifacts["preview"]["snapshot_path"]
    assert (tmp_path / snapshot).read_text(encoding="utf-8") == preview.read_text(encoding="utf-8")
    preview.write_text("changed\n", encoding="utf-8")
    assert (tmp_path / snapshot).read_text(encoding="utf-8") == "solid demo\nendsolid demo\n"
    assert [event["event_type"] for event in reloaded.get_events(run.run_id)] == [
        "run_started",
        "attempt_started",
        "attempt_completed",
        "run_completed",
    ]

    with pytest.raises(QualityIntegrityError):
        reloaded.complete_attempt(
            run.run_id,
            attempt.attempt_id,
            status="failed",
            phase="execution",
        )


def test_restart_marks_abandoned_run_interrupted(tmp_path: Path):
    run = QualityStore(tmp_path).start_run(project="demo")

    restarted = QualityStore(tmp_path)

    assert restarted.reconcile() == 1
    assert restarted.get_run(run.run_id).status == "interrupted"


class FailedCadClient:
    def __init__(self, _settings):
        self.calls = 0

    def chat(self, _messages, _tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "cad-1",
                            "function": {
                                "name": "cad_build_and_verify",
                                "arguments": "{}",
                            },
                        }],
                    }
                }]
            }
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "I can fix it."}
            }]
        }


def test_failed_cad_build_records_actionable_quality_failure(
    tmp_path: Path, monkeypatch
):
    import agent.core

    settings = _settings(tmp_path)
    project = settings.workspace_root / "demo"
    project.mkdir(parents=True)
    (project / "conversation.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(agent.core, "create_llm_client", lambda _settings, _ignored=None: FailedCadClient(_settings))
    events = []
    runner = agent.core.AgentRunner(
        settings,
        lambda kind, data: events.append((kind, data)),
    )

    runner._run("demo", "Build a bracket")

    store = QualityStore(project)
    run = store.list_runs()[0]
    attempt = store.list_attempts(run.run_id)[0]
    assert run.status == "failed"
    assert run.error["code"] == "CAD_BUILD_FAILED"
    assert attempt.status == "failed"
    assert attempt.tool_call_id == "cad-1"
    assert attempt.error == {
        "code": "MODEL_MISSING",
        "category": "Source",
        "phase": "source",
        "message": "model.py does not exist yet.",
        "retryable": True,
        "hint": "Create model.py first.",
    }
    assert {kind for kind, _ in events} >= {
        "quality_run_started",
        "quality_attempt_completed",
        "quality_run_completed",
    }


def test_quality_api_exposes_recorded_failure(tmp_path: Path):
    settings = _settings(tmp_path)
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    store = QualityStore(settings.workspace_root / "demo")
    run = store.start_run(project="demo", request_sha256="request-sha")
    attempt = store.start_attempt(run.run_id, source_sha256="source-sha")
    store.complete_attempt(
        run.run_id,
        attempt.attempt_id,
        status="failed",
        phase="source",
        error={"code": "MODEL_MISSING", "message": "model.py is missing"},
    )
    store.complete_run(run.run_id, status="failed")

    response = client.get(f"/api/projects/demo/quality/runs/{run.run_id}")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["run"]["status"] == "failed"
    assert payload["attempts"][0]["error"]["code"] == "MODEL_MISSING"
    assert payload["events"][-1]["event_type"] == "run_completed"
