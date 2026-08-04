"""Critical user acceptance, issue reporting, and finalization flows."""

import base64
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.quality.store import QualityStore
from agent.revisions import RevisionOrigin, RevisionStore
from agent.settings import Settings
from app import create_app


@dataclass
class QualityScenario:
    client: object
    project_dir: Path
    revision_id: str
    run_id: str
    attempt_id: str


def _scenario(tmp_path: Path, *, require_acceptance: bool = False) -> QualityScenario:
    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=require_acceptance,
    )
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    source = "result = 1\n"
    (project_dir / "model.py").write_text(source, encoding="utf-8")
    revision = RevisionStore(project_dir).commit(
        source,
        RevisionOrigin(kind="agent_edit"),
    )
    store = QualityStore(project_dir)
    run = store.start_run(project="demo")
    attempt = store.start_attempt(
        run.run_id,
        revision_id=revision.id,
        source_sha256="source-sha",
    )
    store.complete_attempt(
        run.run_id,
        attempt.attempt_id,
        status="succeeded",
        phase="kernel",
        metrics={"solid_count": 1},
    )
    store.complete_run(run.run_id, status="completed")
    return QualityScenario(
        client,
        project_dir,
        revision.id,
        run.run_id,
        attempt.attempt_id,
    )


@pytest.fixture
def scenario(tmp_path: Path) -> QualityScenario:
    return _scenario(tmp_path)


def test_accepting_current_preview_updates_revision_and_history(
    scenario: QualityScenario,
):
    response = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "accepted", "comment": "Matches the request"},
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert {
        key: payload[key] for key in ("revision_id", "attempt_id", "decision")
    } == {
        "revision_id": scenario.revision_id,
        "attempt_id": scenario.attempt_id,
        "decision": "accepted",
    }
    revision = scenario.client.get("/api/projects/demo/revisions").get_json()[
        "revisions"
    ][0]
    assert revision["acceptance"]["decision"] == "accepted"
    assert revision["open_issues"] == 0
    history = (scenario.project_dir / "conversation.jsonl").read_text(encoding="utf-8")
    assert "design_accepted" in history
    assert scenario.revision_id in history
    assert QualityStore(scenario.project_dir).get_run(scenario.run_id).accepted_revision_id == scenario.revision_id


def test_revised_decision_supersedes_the_prior_decision(scenario: QualityScenario):
    first = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "accepted"},
    ).get_json()
    second = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "rejected"},
    ).get_json()

    assert second["supersedes_decision_id"] == first["decision_id"]
    metrics = scenario.client.get("/api/projects/demo/quality/metrics").get_json()
    assert metrics["false_pass_rate"] == 1.0
    assert metrics["accepted_revisions"] == 0


def test_reporting_issue_rejects_revision_and_preserves_evidence(
    scenario: QualityScenario,
):
    screenshot = base64.b64encode(b"\x89PNG\r\n\x1a\npayload").decode("ascii")
    response = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/issues",
        json={
            "category": "hole_or_clearance",
            "severity": "blocking",
            "message": "The bore is blocked.",
            "screenshot": screenshot,
        },
    )

    assert response.status_code == 201
    issue = response.get_json()["issue"]
    assert issue["status"] == "open"
    assert issue["category"] == "hole_or_clearance"
    evidence_path = (
        f".cad-agent/quality/runs/{scenario.run_id}/artifacts/"
        f"{scenario.attempt_id}/evidence/issue-{issue['issue_id']}.png"
    )
    assert issue["evidence"] == [evidence_path]
    assert (scenario.project_dir / issue["evidence"][0]).is_file()

    revision = scenario.client.get("/api/projects/demo/revisions").get_json()[
        "revisions"
    ][0]
    assert revision["acceptance"]["decision"] == "rejected"
    assert revision["open_issues"] == 1
    metrics = scenario.client.get("/api/projects/demo/quality/metrics").get_json()
    assert metrics["issues"] == {"open": 1, "resolved": 0}
    assert metrics["false_pass_rate"] == 1.0
    assert "design_rejected" in (
        scenario.project_dir / "conversation.jsonl"
    ).read_text(encoding="utf-8")


def test_decision_requires_a_successful_current_preview(scenario: QualityScenario):
    store = QualityStore(scenario.project_dir)
    failed_run = store.start_run(project="demo")
    failed = store.start_attempt(
        failed_run.run_id,
        revision_id=scenario.revision_id,
        source_sha256="failed-source",
    )
    store.complete_attempt(
        failed_run.run_id,
        failed.attempt_id,
        status="failed",
        phase="execution",
        error={"code": "CAD_BUILD_FAILED", "message": "boom"},
    )
    response = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{failed.attempt_id}/decision",
        json={"decision": "accepted"},
    )
    assert response.status_code == 409
    assert response.get_json()["code"] == "ATTEMPT_NOT_SUCCEEDED"

    response = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{failed.attempt_id}/issues",
        json={"category": "other", "message": "No preview"},
    )
    assert response.status_code == 409
    assert response.get_json()["code"] == "ATTEMPT_NOT_SUCCEEDED"

    source = "result = 2\n"
    (scenario.project_dir / "model.py").write_text(source, encoding="utf-8")
    RevisionStore(scenario.project_dir).commit(
        source,
        RevisionOrigin(kind="agent_edit"),
    )
    response = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "accepted"},
    )
    assert response.status_code == 409
    assert response.get_json()["code"] == "STALE_PREVIEW"


def test_finalize_gate_accepts_explicit_decision_or_audited_bypass(
    tmp_path: Path, monkeypatch
):
    import app as app_module

    scenario = _scenario(tmp_path, require_acceptance=True)
    monkeypatch.setattr(
        app_module,
        "finalize_project",
        lambda _project_dir: {"report_text": "ok"},
    )

    blocked = scenario.client.post("/api/projects/demo/finalize")
    assert blocked.status_code == 409
    assert blocked.get_json()["code"] == "ACCEPTANCE_REQUIRED"

    bypassed = scenario.client.post(
        "/api/projects/demo/finalize",
        json={"force": True},
    )
    assert bypassed.status_code == 200
    events = QualityStore(scenario.project_dir).get_events(scenario.run_id)
    assert events[-1]["event_type"] == "finalize_bypassed"

    scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "accepted"},
    )
    assert scenario.client.post("/api/projects/demo/finalize").status_code == 200


def test_finalize_gate_is_opt_in(scenario: QualityScenario, monkeypatch):
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "finalize_project",
        lambda _project_dir: {"report_text": "ok"},
    )

    assert scenario.client.post("/api/projects/demo/finalize").status_code == 200


def test_forced_finalize_creates_an_audit_run_when_none_exists(tmp_path: Path, monkeypatch):
    import app as app_module

    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=True,
    )
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    source = "result = 1\n"
    (project_dir / "model.py").write_text(source, encoding="utf-8")
    RevisionStore(project_dir).commit(source, RevisionOrigin(kind="agent_edit"))
    monkeypatch.setattr(app_module, "finalize_project", lambda _project_dir: {"report_text": "ok"})

    assert client.post("/api/projects/demo/finalize", json={"force": True}).status_code == 200
    run = QualityStore(project_dir).list_runs()[0]
    assert [event["event_type"] for event in QualityStore(project_dir).get_events(run.run_id)] == [
        "run_started",
        "finalize_bypassed",
        "run_completed",
    ]


def test_resolving_issue_updates_open_count_and_finalize_gate(tmp_path: Path, monkeypatch):
    import app as app_module

    scenario = _scenario(tmp_path, require_acceptance=True)
    monkeypatch.setattr(app_module, "finalize_project", lambda _project_dir: {"report_text": "ok"})
    issue = scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/issues",
        json={"category": "other", "severity": "blocking", "message": "Fix this"},
    ).get_json()["issue"]
    scenario.client.post(
        f"/api/projects/demo/quality/attempts/{scenario.attempt_id}/decision",
        json={"decision": "accepted_with_limitations"},
    )
    assert scenario.client.post("/api/projects/demo/finalize").get_json()["code"] == "BLOCKING_ISSUES_OPEN"

    blocked_resolution = scenario.client.post(
        f"/api/projects/demo/quality/issues/{issue['issue_id']}/resolve"
    )
    assert blocked_resolution.status_code == 409
    assert blocked_resolution.get_json()["code"] == "RESOLUTION_REVIEW_REQUIRED"

    source = "result = 2\n"
    (scenario.project_dir / "model.py").write_text(source, encoding="utf-8")
    revision = RevisionStore(scenario.project_dir).commit(
        source, RevisionOrigin(kind="agent_edit")
    )
    store = QualityStore(scenario.project_dir)
    run = store.start_run(project="demo")
    attempt = store.start_attempt(run.run_id, revision_id=revision.id)
    store.complete_attempt(
        run.run_id, attempt.attempt_id, status="succeeded", phase="kernel"
    )
    store.complete_run(run.run_id, status="completed")

    response = scenario.client.post(
        f"/api/projects/demo/quality/issues/{issue['issue_id']}/resolve"
    )
    assert response.status_code == 200
    assert response.get_json()["issue"]["status"] == "resolved"
    assert scenario.client.post("/api/projects/demo/finalize").status_code == 409
    scenario.client.post(
        f"/api/projects/demo/quality/attempts/{attempt.attempt_id}/decision",
        json={"decision": "accepted"},
    )
    assert scenario.client.post("/api/projects/demo/finalize").status_code == 200
