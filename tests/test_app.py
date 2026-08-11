import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from agent.revisions import RevisionOrigin, RevisionStore
from agent.settings import Settings
from app import SSE_QUEUE_SIZE, EventBus, create_app


def test_project_lifecycle(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    assert client.get("/api/projects").get_json() == {"projects": []}
    response = client.post("/api/projects/new", json={"name": "Mounting Bracket"})
    assert response.status_code == 201
    assert response.get_json() == {"project": "mounting-bracket"}
    assert (settings.workspace_root / "mounting-bracket" / "inputs").is_dir()
    projects_data = client.get("/api/projects").get_json()
    assert len(projects_data["projects"]) == 1
    assert projects_data["projects"][0]["name"] == "mounting-bracket"
    assert projects_data["projects"][0]["model_status"] == "none"


def test_chat_ignores_client_supplied_routing_preferences(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "openai/gpt-4o-mini", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()

    client.post("/api/projects/new", json={"name": "demo"})
    monkeypatch.setattr(app.config["AGENT_RUNNER"], "start", lambda *_args, **_kwargs: True)
    response = client.post(
        "/api/chat",
        data={"project": "demo", "message": "make a bracket", "model": "invalid", "force_provider": "true"},
        headers={"Origin": "http://localhost:5000"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202


def test_cross_origin_mutation_is_rejected(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})

    # Mismatched Origin (a malicious page driving the local service).
    response = client.post(
        "/api/chat",
        json={"project": "demo", "message": "hi"},
        headers={"Origin": "http://attacker.test"},
    )
    assert response.status_code == 403
    assert "Cross-origin" in response.get_json()["error"]

    # Cross-origin Host on a specific bind (still rejected even without Origin).
    response = client.post(
        "/api/chat",
        json={"project": "demo", "message": "hi"},
        headers={"Host": "evil.example.test"},
    )
    assert response.status_code == 403


def test_same_origin_state_change_is_allowed(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    monkeypatch.setattr(app.config["AGENT_RUNNER"], "start", lambda *_args, **_kwargs: True)

    # Same-origin browser request (Origin matches Host) is allowed.
    response = client.post(
        "/api/chat",
        json={"project": "demo", "message": "hi"},
        headers={"Origin": "http://localhost:5000", "Host": "localhost:5000"},
    )
    assert response.status_code == 202


def test_cross_origin_form_post_without_origin_is_rejected(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})

    # Form POST with a non-loopback Host and no Origin header must be rejected.
    # Browsers do not send Origin on simple form POSTs so a malicious page on
    # another origin could otherwise drive the local service; we require an
    # Origin header for form-encoded mutating requests whose Host does not
    # already establish same-origin.
    response = client.post(
        "/api/chat",
        data={"project": "demo", "message": "hi"},
        headers={"Host": "attacker.example.test"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 403
    assert "Cross-origin" in response.get_json()["error"]


def test_same_origin_form_post_without_origin_is_allowed(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    monkeypatch.setattr(app.config["AGENT_RUNNER"], "start", lambda *_args, **_kwargs: True)

    # curl-style form POST with a loopback Host and no Origin header must be
    # allowed: the Host already establishes same-origin even though the
    # browser-only Origin header is absent.
    response = client.post(
        "/api/chat",
        data={"project": "demo", "message": "hi"},
        headers={"Host": "localhost:5000"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202


def test_settings_routes_are_not_exposed(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    assert client.get("/settings").status_code == 404
    assert client.get("/api/agent-settings").status_code == 404


def test_info_messages_are_persisted_and_returned_in_history(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})

    app.config["EVENT_BUS"].publish(
        "tool_status",
        {"project": "demo", "tool": "cad", "status": "completed", "result": "Preview updated."},
    )

    events = client.get("/api/projects/demo/history").get_json()["events"]
    assert events[-1]["type"] == "tool_status"
    assert events[-1]["data"]["result"] == "Preview updated."


def test_info_messages_remain_stored_but_are_hidden_when_disabled(tmp_path: Path):
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000, show_info_messages=False
    )
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    app.config["EVENT_BUS"].publish("agent_status", {"project": "demo", "status": "started"})

    history = client.get("/api/projects/demo/history").get_json()["events"]
    raw_history = (settings.workspace_root / "demo" / "conversation.jsonl").read_text(encoding="utf-8")
    assert history == []
    assert '"type": "agent_status"' in raw_history
    assert b'"showInfoMessages": false' in client.get("/project/demo").data


def test_chat_requires_existing_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    response = client.post("/api/chat", json={"project": "missing", "message": "make a bracket"})
    assert response.status_code == 404


def test_preview_is_served_only_from_requested_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    preview = settings.workspace_root / "demo" / "preview.stl"
    preview.write_bytes(b"solid demo\nendsolid demo\n")

    response = client.get("/api/projects/demo/preview")
    assert response.status_code == 200
    assert response.data == preview.read_bytes()
    assert client.get("/api/projects/../preview").status_code == 404


def test_empty_preview_is_rejected(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    (settings.workspace_root / "demo" / "preview.stl").write_bytes(b"")

    response = client.get("/api/projects/demo/preview")

    assert response.status_code == 422
    assert response.get_json()["error"] == "The generated preview is empty."


def test_preview_metadata_changes_when_stl_is_updated(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    preview = settings.workspace_root / "demo" / "preview.stl"

    assert client.get("/api/projects/demo/preview/meta").get_json() == {"available": False}

    preview.write_bytes(b"first")
    first = client.get("/api/projects/demo/preview/meta").get_json()
    preview.write_bytes(b"second-version")
    second = client.get("/api/projects/demo/preview/meta").get_json()

    assert first["available"] is True
    assert first["model_sha256"] is None
    assert second["available"] is True
    assert first["revision"] != second["revision"]


def test_preview_completion_requires_matching_display_confirmation(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    project = settings.workspace_root / "demo"
    preview = project / "preview.stl"
    preview.write_bytes(b"solid demo\nendsolid demo\n")
    runner = app.config["AGENT_RUNNER"]
    preview_id = runner._register_preview("demo", project)
    runner._await_preview("demo", preview_id, "Task completed.")

    assert client.get("/api/projects/demo/state").get_json() == {
        "status": "rendering",
        "preview_id": preview_id,
    }
    assert client.post(
        "/api/projects/demo/preview/displayed",
        json={"preview_id": "wrong"},
    ).status_code == 409
    assert not any(
        event.get("role") == "assistant"
        for event in client.get("/api/projects/demo/history").get_json()["events"]
    )

    response = client.post(
        "/api/projects/demo/preview/displayed",
        json={"preview_id": preview_id},
    )

    assert response.status_code == 200
    assert client.get("/api/projects/demo/state").get_json() == {"status": "idle"}
    history = client.get("/api/projects/demo/history").get_json()["events"]
    assert history[-1]["content"] == "Task completed."


def test_preview_render_failure_is_reported_as_an_error(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    project = settings.workspace_root / "demo"
    (project / "preview.stl").write_bytes(b"solid demo\nendsolid demo\n")
    runner = app.config["AGENT_RUNNER"]
    preview_id = runner._register_preview("demo", project)
    runner._await_preview("demo", preview_id, "Task completed.")

    response = client.post(
        "/api/projects/demo/preview/failed",
        json={"preview_id": preview_id, "message": "The preview contains no triangles."},
    )

    assert response.status_code == 200
    history = client.get("/api/projects/demo/history").get_json()["events"]
    assert history[-1]["type"] == "agent_error"
    assert "contains no triangles" in history[-1]["data"]["message"]
    assert not any(event.get("content") == "Task completed." for event in history)


def test_chat_stores_normalized_image_attachment(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    source = BytesIO()
    Image.new("RGB", (20, 10), "red").save(source, format="JPEG")
    source.seek(0)

    response = client.post(
        "/api/chat",
        data={"project": "demo", "message": "Use this sketch", "attachments": (source, "sketch.jpg")},
        headers={"Origin": "http://localhost:5000"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    attachment = response.get_json()["attachments"][0]
    assert attachment.startswith("inputs/") and attachment.endswith(".png")
    assert (settings.workspace_root / "demo" / attachment).is_file()


def test_chat_rolls_back_earlier_images_when_later_attachment_is_invalid(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    source = BytesIO()
    Image.new("RGB", (20, 10), "red").save(source, format="PNG")
    source.seek(0)

    response = client.post(
        "/api/chat",
        data={
            "project": "demo",
            "message": "Use these",
            "attachments": [(source, "valid.png"), (BytesIO(b"not an image"), "invalid.txt")],
        },
        headers={"Origin": "http://localhost:5000"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert list((settings.workspace_root / "demo" / "inputs").iterdir()) == []


def test_question_answer_endpoint_resumes_waiting_project(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    runner = app.config["AGENT_RUNNER"]
    runner._waiting_questions["demo"] = {
        "title": "", "questions": [{"id": "q1", "question": "Hole size?", "input_type": "text"}]
    }
    monkeypatch.setattr(runner, "start", lambda *_args, **_kwargs: True)

    response = client.post("/api/questions/answer", json={"project": "demo", "answer": "6 mm"})

    assert response.status_code == 202
    assert runner.waiting_question("demo") is None


def test_question_answer_endpoint_rejects_invalid_numeric_answer(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    runner = app.config["AGENT_RUNNER"]
    runner._waiting_questions["demo"] = {
        "title": "", "questions": [{"id": "q1", "question": "Hole size?", "input_type": "number"}]
    }

    response = client.post("/api/questions/answer", json={"project": "demo", "answer": "large"})

    assert response.status_code == 400
    assert runner.waiting_question("demo") is not None


def test_stop_clears_persisted_waiting_question(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    project = settings.workspace_root / "demo"
    state = json.dumps({
        "status": "WAITING_FOR_USER",
        "waiting_question": {
            "title": "",
            "questions": [{"id": "q1", "question": "Size?", "input_type": "number"}],
        },
    })
    (project / ".agent_state.json").write_text(state, encoding="utf-8")

    assert client.post("/api/stop", json={"project": "demo"}).status_code == 200
    assert app.config["AGENT_RUNNER"].waiting_question("demo") is None
    assert not (project / ".agent_state.json").exists()


def test_project_state_recovers_persisted_question(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    project = settings.workspace_root / "demo"
    state = json.dumps({
        "status": "WAITING_FOR_USER",
        "waiting_question": {
            "title": "",
            "questions": [{"id": "q1", "question": "Hole size?", "input_type": "number", "options": []}],
        },
    })
    (project / ".agent_state.json").write_text(state, encoding="utf-8")

    response = client.get("/api/projects/demo/state")

    assert response.get_json() == {
        "status": "waiting_for_user",
        "question": {
            "title": "",
            "questions": [{"id": "q1", "question": "Hole size?", "input_type": "number", "options": []}],
        },
    }


def test_delete_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    # Create a project first
    client.post("/api/projects/new", json={"name": "to-delete"})
    assert (settings.workspace_root / "to-delete").is_dir()

    # Successful delete
    response = client.delete("/api/projects/to-delete")
    assert response.status_code == 200
    assert response.get_json() == {"deleted": True}
    assert not (settings.workspace_root / "to-delete").exists()

    # Delete nonexistent project → 404
    response = client.delete("/api/projects/to-delete")
    assert response.status_code == 404


def test_rename_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    # Create a project
    client.post("/api/projects/new", json={"name": "old-name"})
    assert (settings.workspace_root / "old-name").is_dir()

    # Successful rename
    response = client.put("/api/projects/old-name/rename", json={"name": "new-name"})
    assert response.status_code == 200
    assert response.get_json() == {"project": "new-name"}
    assert not (settings.workspace_root / "old-name").exists()
    assert (settings.workspace_root / "new-name").is_dir()

    # Rename to the same name → 200 (no-op)
    response = client.put("/api/projects/new-name/rename", json={"name": "new-name"})
    assert response.status_code == 200
    assert response.get_json() == {"project": "new-name"}

    # Create another project to test conflict
    client.post("/api/projects/new", json={"name": "conflicting"})

    # Rename to existing name → 409
    response = client.put("/api/projects/new-name/rename", json={"name": "conflicting"})
    assert response.status_code == 409

    # Rename nonexistent project → 404
    response = client.put("/api/projects/nonexistent/rename", json={"name": "anything"})
    assert response.status_code == 404


def test_project_crud_allows_persisted_waiting_question(tmp_path: Path):
    # A persisted waiting question is recoverable UI state, not an active
    # worker; the project directory can be deleted/renamed because the next
    # load will simply find no agent state to resume.
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    for name in ("delete-me", "rename-me"):
        client.post("/api/projects/new", json={"name": name})
        state = {
            "status": "WAITING_FOR_USER",
            "waiting_question": {
                "title": "",
                "questions": [{"id": "q1", "question": "Size?", "input_type": "text"}],
            },
        }
        (settings.workspace_root / name / ".agent_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

    assert client.delete("/api/projects/delete-me").status_code == 200
    assert client.put(
        "/api/projects/rename-me/rename", json={"name": "renamed"}
    ).status_code == 200


def test_project_modified_at_tracks_file_content_changes(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    first = client.get("/api/projects").get_json()["projects"][0]["modified_at"]
    summary = settings.workspace_root / "demo" / "summary.md"
    summary.write_text("updated", encoding="utf-8")
    future = time.time() + 10
    os.utime(summary, (future, future))

    second = client.get("/api/projects").get_json()["projects"][0]["modified_at"]

    assert second > first


def test_event_bus_disconnects_overflowed_subscriber(tmp_path: Path):
    settings = Settings(tmp_path, "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    bus = EventBus(settings)
    subscriber = bus.subscribe()

    for index in range(SSE_QUEUE_SIZE + 1):
        bus.publish("delta", {"index": index})

    assert subscriber.get_nowait()["type"] == "stream_reset"
    assert subscriber.get_nowait() is None
    assert not bus._subscribers


# ── Setup page ──



# ── Preflight ──

def test_preflight_returns_check_results(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    client = create_app(settings).test_client()

    resp = client.get("/api/preflight")
    data = resp.get_json()
    assert resp.status_code == 200
    assert "api_key" in data
    assert "model_configured" in data
    assert "workspace_writable" in data
    assert "bwrap_installed" in data
    assert "seccomp" in data
    assert "python_packages" in data


# ── Example prompts ──

def test_index_contains_example_prompts(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})

    response = client.get("/project/demo")
    assert response.status_code == 200
    assert b"example-prompt" in response.data
    assert b"mounting plate" in response.data


# ── Local vendor assets ──

def test_local_three_js_import_map_is_used(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})

    response = client.get("/project/demo")
    assert response.status_code == 200
    assert b"vendor/three/three.module.js" in response.data
    assert b"vendor/three/addons/" in response.data
    assert b"vendor/marked/marked.min.js" in response.data
    assert b"vendor/highlight/highlight.min.js" in response.data


def test_frontend_source_has_no_cdn_urls():
    project_root = Path(__file__).resolve().parents[1]
    cdn_url = re.compile(r"https?://[^\"'\s]*(?:cdn|unpkg|jsdelivr|cdnjs|esm\.sh|skypack)[^\"'\s]*", re.IGNORECASE)
    frontend_files = [*project_root.glob("templates/*.html"), *project_root.glob("static/css/*"), *project_root.glob("static/js/*")]

    assert frontend_files
    assert not [path for path in frontend_files if cdn_url.search(path.read_text(encoding="utf-8"))]


# ── Error message mapping ──

from agent.core import AgentRunner


def test_user_error_messages_are_actionable():
    cases = (
        ("HTTP 401 Unauthorized", "HTTPError", "invalid openrouter api key"),
        ("429 Too Many Requests: rate limit exceeded", "HTTPError", "rate limit"),
        ("Connection timed out", "ConnectionError", "timed out"),
        ("bubblewrap sandbox failed: bwrap not found", "RuntimeError", "sandbox"),
        ("STEP export failed: invalid geometry", "ValueError", "export failed"),
        ("Permission denied: cannot write to workspace", "OSError", "permission"),
        ("Some unknown error occurred", "RuntimeError", "some unknown error occurred"),
    )

    for error, error_type, expected in cases:
        message = AgentRunner._user_error_message(error, error_type)
        assert expected in message.lower()


# --------------------------------------------------------------------------- #
#  Revision API tests
# --------------------------------------------------------------------------- #

def _setup_project_with_revisions(tmp_path: Path, count: int = 3):
    """Create a project with the given number of model.py revisions."""
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    project_dir = settings.workspace_root / "demo"
    store = RevisionStore(project_dir)
    revisions = []
    for i in range(count):
        rev = store.commit(f"# model v{i}\nresult = {i}\n", RevisionOrigin(kind="agent_edit"))
        revisions.append(rev)
    return settings, app, client, project_dir, revisions


def test_revision_list_returns_revisions_newest_first(tmp_path: Path):
    _settings, _app, client, _dir, revisions = _setup_project_with_revisions(tmp_path, 3)

    response = client.get("/api/projects/demo/revisions")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["revisions"]) == 3
    # Newest first.
    assert data["revisions"][0]["id"] == revisions[2].id
    assert data["revisions"][0]["is_active"] is True
    assert data["revisions"][1]["is_active"] is False


def test_revision_list_includes_build_status(tmp_path: Path):
    _settings, _app, client, project_dir, revisions = _setup_project_with_revisions(tmp_path, 1)
    store = RevisionStore(project_dir)
    store.record_build_success(revisions[0].id, {"solid_count": 1}, project_dir / "preview.stl")

    response = client.get("/api/projects/demo/revisions")
    data = response.get_json()
    rev0 = next(r for r in data["revisions"] if r["id"] == revisions[0].id)
    assert rev0["build_status"] == "succeeded"
    assert rev0["metrics"]["solid_count"] == 1


def test_revision_detail_returns_source(tmp_path: Path):
    _settings, _app, client, _dir, revisions = _setup_project_with_revisions(tmp_path, 1)

    response = client.get(f"/api/projects/demo/revisions/{revisions[0].id}")
    assert response.status_code == 200
    data = response.get_json()
    assert "source" in data
    assert "result = 0" in data["source"]


def test_revision_detail_404_for_unknown_id(tmp_path: Path):
    _settings, _app, client, _dir, _revisions = _setup_project_with_revisions(tmp_path, 1)

    response = client.get("/api/projects/demo/revisions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_revision_diff_returns_unified_diff(tmp_path: Path):
    _settings, _app, client, _dir, revisions = _setup_project_with_revisions(tmp_path, 2)

    response = client.get(f"/api/projects/demo/revisions/{revisions[1].id}/diff")
    assert response.status_code == 200
    data = response.get_json()
    assert "diff" in data
    assert "result = 0" in data["diff"]
    assert "result = 1" in data["diff"]
    assert data["truncated"] is False


def test_revision_diff_404_for_unknown_revision(tmp_path: Path):
    _settings, _app, client, _dir, _revisions = _setup_project_with_revisions(tmp_path, 1)

    response = client.get("/api/projects/demo/revisions/00000000-0000-0000-0000-000000000000/diff")
    assert response.status_code == 404


def test_revision_diff_handles_pruned_parent(tmp_path: Path):
    _settings, _app, client, project_dir, _revisions = _setup_project_with_revisions(tmp_path, 0)
    store = RevisionStore(project_dir, retention_count=2)
    for i in range(3):
        store.commit(f"result = {i}\n", RevisionOrigin(kind="agent_edit"))

    oldest_retained = store.list(limit=200)[-1]
    response = client.get(
        f"/api/projects/demo/revisions/{oldest_retained.id}/diff"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["against"] is None
    assert "+result = 1" in data["diff"]


def test_revision_list_404_for_unknown_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    response = client.get("/api/projects/nonexistent/revisions")
    assert response.status_code == 404


def test_restore_rejects_when_agent_active(tmp_path: Path):
    _settings, app, client, _project_dir, revisions = _setup_project_with_revisions(tmp_path, 2)
    runner = app.config["AGENT_RUNNER"]
    # Simulate running agent.
    runner._active_project = "demo"
    runner._thread = type("T", (), {"is_alive": lambda self: True})()

    try:
        response = client.post(f"/api/projects/demo/revisions/{revisions[0].id}/restore")
        assert response.status_code == 409
        assert "active" in response.get_json()["error"].lower()
    finally:
        runner._active_project = None
        runner._thread = None


def test_restore_404_for_unknown_project(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()

    response = client.post("/api/projects/nonexistent/revisions/00000000-0000-0000-0000-000000000000/restore")
    assert response.status_code == 404


def test_restore_422_for_corrupt_revision(tmp_path: Path):
    _settings, _app, client, project_dir, revisions = _setup_project_with_revisions(tmp_path, 2)
    # Delete the source blob for the first revision.
    blob = project_dir / ".cad-agent" / "history" / "blobs" / f"{revisions[0].model_sha256}.py"
    blob.unlink()

    response = client.post(f"/api/projects/demo/revisions/{revisions[0].id}/restore")
    assert response.status_code == 422



def test_restore_rebuilds_and_registers_preview(tmp_path: Path, monkeypatch):
    _settings, _app, client, project_dir, _revisions = _setup_project_with_revisions(tmp_path, 0)
    store = RevisionStore(project_dir)
    old = store.commit("result = 'old'\n", RevisionOrigin(kind="agent_edit"))
    store.commit("result = 'new'\n", RevisionOrigin(kind="agent_edit"))

    def fake_run(self):
        (self.project_dir / "preview.stl").write_bytes(b"solid preview\nendsolid preview\n")
        return {"dimensions_mm": {"x": 10.0, "y": 20.0, "z": 30.0}}

    import agent.tools.cad_tool
    monkeypatch.setattr(agent.tools.cad_tool.CadTool, "run", fake_run)

    response = client.post(f"/api/projects/demo/revisions/{old.id}/restore")
    data = response.get_json()

    assert response.status_code == 200
    assert data["build_status"] == "succeeded"
    assert data["metrics"]["dimensions_mm"] == {"x": 10.0, "y": 20.0, "z": 30.0}
    displayed = client.post(
        "/api/projects/demo/preview/displayed",
        json={"preview_id": data["preview_id"]},
    )
    assert displayed.status_code == 200


def test_revision_list_rejects_non_integer_limit(tmp_path: Path):
    _settings, _app, client, _project_dir, _revisions = _setup_project_with_revisions(tmp_path, 1)

    response = client.get("/api/projects/demo/revisions?limit=invalid")

    assert response.status_code == 400


def test_revision_list_reports_corrupt_history(tmp_path: Path):
    _settings, _app, client, project_dir, _revisions = _setup_project_with_revisions(tmp_path, 1)
    revisions_dir = project_dir / ".cad-agent" / "history" / "revisions"
    (revisions_dir / "not-a-revision.json").write_text("{}", encoding="utf-8")

    response = client.get("/api/projects/demo/revisions")

    assert response.status_code == 422
    assert "filename is invalid" in response.get_json()["error"]
