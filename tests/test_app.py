import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

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
    )
    assert response.status_code == 202


def test_index_configures_three_js_import_map(tmp_path: Path):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    client = create_app(settings).test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    response = client.get("/project/demo")

    assert response.status_code == 200
    assert b'type="importmap"' in response.data
    assert b'"three/addons/"' in response.data


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
    assert b"showInfoMessages: false" in client.get("/project/demo").data


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

    response = client.post("/api/chat", data={"project": "demo", "message": "Use this sketch", "attachments": (source, "sketch.jpg")})

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


def test_finalize_is_rejected_while_agent_is_running(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    app = create_app(settings)
    client = app.test_client()
    client.post("/api/projects/new", json={"name": "demo"})
    monkeypatch.setattr(app.config["AGENT_RUNNER"], "is_running", lambda: True)

    response = client.post("/api/projects/demo/finalize")

    assert response.status_code == 409


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


def test_project_crud_rejects_persisted_waiting_question(tmp_path: Path):
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

    assert client.delete("/api/projects/delete-me").status_code == 409
    assert client.put(
        "/api/projects/rename-me/rename", json={"name": "renamed"}
    ).status_code == 409


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

def test_setup_page_serves_when_no_api_key(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = create_app(settings).test_client()

    response = client.get("/setup")
    assert response.status_code == 200
    assert b"OpenRouter API Key" in response.data


def test_setup_page_redirects_when_configured(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    client = create_app(settings).test_client()

    response = client.get("/setup", follow_redirects=False)
    assert response.status_code == 302


def test_index_redirects_to_setup_when_no_api_key(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = create_app(settings).test_client()

    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302


def test_project_view_redirects_to_setup_when_no_api_key(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = create_app(settings).test_client()

    response = client.get("/project/demo", follow_redirects=False)
    assert response.status_code == 302


def test_api_setup_saves_key_and_model(tmp_path: Path, monkeypatch):
    settings = Settings(tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    # Point _project_root to tmp_path so we don't touch the real .env.
    monkeypatch.setattr("app._project_root", lambda: tmp_path)
    (tmp_path / ".env.example").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
    client = create_app(settings).test_client()

    resp = client.post("/api/setup", json={"api_key": "sk-or-v1-mykey", "model": "openai/gpt-4o"})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}

    env_content = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=sk-or-v1-mykey" in env_content


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


def test_user_error_message_maps_unauthorized():
    msg = AgentRunner._user_error_message("HTTP 401 Unauthorized", "HTTPError")
    assert "Invalid OpenRouter API key" in msg


def test_user_error_message_maps_rate_limit():
    msg = AgentRunner._user_error_message("429 Too Many Requests: rate limit exceeded", "HTTPError")
    assert "rate limit" in msg.lower()


def test_user_error_message_maps_timeout():
    msg = AgentRunner._user_error_message("Connection timed out", "ConnectionError")
    assert "timed out" in msg.lower()


def test_user_error_message_maps_sandbox():
    msg = AgentRunner._user_error_message("bubblewrap sandbox failed: bwrap not found", "RuntimeError")
    assert "sandbox" in msg.lower()


def test_user_error_message_maps_export():
    msg = AgentRunner._user_error_message("STEP export failed: invalid geometry", "ValueError")
    assert "Export failed" in msg


def test_user_error_message_maps_permission():
    msg = AgentRunner._user_error_message("Permission denied: cannot write to workspace", "OSError")
    assert "permission" in msg.lower()


def test_user_error_message_fallback():
    msg = AgentRunner._user_error_message("Some unknown error occurred", "RuntimeError")
    assert "Some unknown error occurred" in msg
