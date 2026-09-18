import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image

from agent.conversation import ConversationStore
from agent.revisions import RevisionOrigin, RevisionStore
from agent.settings import Settings
from app import create_app


def test_project_creation_and_cross_origin_chat_guard(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    created = client.post("/api/projects/new", json={"name": "Mounting Bracket"})

    assert created.status_code == 201
    assert created.get_json() == {"project": "mounting-bracket"}
    blocked = client.post(
        "/api/chat",
        json={"project": "mounting-bracket", "message": "make a bracket"},
        headers={"Origin": "https://attacker.test"},
    )
    assert blocked.status_code == 403


def test_synthetic_reminder_persisted_and_filtered_from_ui_history(tmp_path: Path) -> None:
    """Synthetic reminders are persisted in conversation store but filtered from UI history endpoint."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    client.post("/api/projects/new", json={"name": "Widget"})
    project_dir = tmp_path / "projects" / "widget"

    user_msg = {"role": "user", "content": "Initial user prompt"}
    synthetic_msg = {
        "role": "user",
        "content": "model.scad exists but it has not been verified. Call cad_build_and_verify now.",
        "synthetic": True,
    }
    assistant_msg = {"role": "assistant", "content": "Done."}

    ConversationStore.append(project_dir, user_msg)
    ConversationStore.append(project_dir, synthetic_msg)
    ConversationStore.append(project_dir, assistant_msg)

    stored = ConversationStore.load(project_dir)
    assert len(stored) == 3
    assert stored[1]["synthetic"] is True

    res = client.get("/api/projects/widget/history")
    assert res.status_code == 200
    events = res.get_json()["events"]
    assert len(events) == 2
    assert events[0]["content"] == "Initial user prompt"
    assert events[1]["content"] == "Done."


def test_project_crud_and_name_validation(tmp_path: Path) -> None:
    """Project CRUD endpoints enforce strict slug validation and handle lifecycle properly."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    # 1. Invalid names on creation -> 400
    for bad_name in ["", "   ", "Invalid@Chars", "-starts-with-dash", "a" * 64]:
        res = client.post("/api/projects/new", json={"name": bad_name})
        assert res.status_code == 400, f"Expected 400 for {bad_name!r}"
        assert "Use 1-63 lowercase letters" in res.get_json()["error"]

    # 2. Valid creation and duplicate rejection -> 201 then 409
    res = client.post("/api/projects/new", json={"name": "Box Frame"})
    assert res.status_code == 201
    assert res.get_json() == {"project": "box-frame"}

    res_dup = client.post("/api/projects/new", json={"name": "box-frame"})
    assert res_dup.status_code == 409
    assert res_dup.get_json() == {"error": "Project already exists."}

    # 3. List projects -> 200
    list_res = client.get("/api/projects")
    assert list_res.status_code == 200
    projects = list_res.get_json()["projects"]
    assert len(projects) == 1
    assert projects[0]["name"] == "box-frame"
    assert projects[0]["created_at"] is not None

    # 4. Rename project
    # 4a. Bad new name -> 400
    res_bad_rename = client.put("/api/projects/box-frame/rename", json={"name": "$bad$"})
    assert res_bad_rename.status_code == 400

    # 4b. Non-existent project -> 404
    res_missing_rename = client.put("/api/projects/missing/rename", json={"name": "new-name"})
    assert res_missing_rename.status_code == 404

    # 4c. Valid rename -> 200
    res_rename = client.put("/api/projects/box-frame/rename", json={"name": "solid-box"})
    assert res_rename.status_code == 200
    assert res_rename.get_json() == {"project": "solid-box"}
    assert not (settings.workspace_root / "box-frame").exists()
    assert (settings.workspace_root / "solid-box").exists()

    # 4d. Same name is a no-op -> 200
    res_noop = client.put("/api/projects/solid-box/rename", json={"name": "solid-box"})
    assert res_noop.status_code == 200

    # 5. Delete project
    del_missing = client.delete("/api/projects/missing")
    assert del_missing.status_code == 404

    del_ok = client.delete("/api/projects/solid-box")
    assert del_ok.status_code == 200
    assert del_ok.get_json() == {"deleted": True}
    assert not (settings.workspace_root / "solid-box").exists()


def test_stop_endpoint_integration(tmp_path: Path) -> None:
    """POST /api/stop stops the runner and clears state files."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    app = create_app(settings)
    client = app.test_client()

    client.post("/api/projects/new", json={"name": "active-task"})
    p_dir = settings.workspace_root / "active-task"
    (p_dir / ".agent_state.json").write_text('{"status": "WAITING_FOR_USER"}')

    # Non-existent project stop -> 404
    res_404 = client.post("/api/stop", json={"project": "ghost-proj"})
    assert res_404.status_code == 404

    # Stop specific project -> 200
    res = client.post("/api/stop", json={"project": "active-task"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["stopped"] is True
    assert "active-task" in body["affected_projects"]
    assert not (p_dir / ".agent_state.json").exists()

    # Stop all active tasks -> 200
    (p_dir / ".agent_state.json").write_text('{"status": "WAITING_FOR_USER"}')
    res_all = client.post("/api/stop", json={})
    assert res_all.status_code == 200
    assert res_all.get_json()["stopped"] is True
    assert "active-task" in res_all.get_json()["affected_projects"]
    assert not (p_dir / ".agent_state.json").exists()


def test_preflight_and_page_routes(tmp_path: Path) -> None:
    """GET /api/preflight and HTML template routes behave according to spec."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    # Preflight
    preflight = client.get("/api/preflight")
    assert preflight.status_code == 200
    data = preflight.get_json()
    assert "model_configured" in data
    assert "api_key" in data
    assert "bwrap_installed" in data
    assert "openscad_installed" in data
    assert "seccomp" in data

    # Root view
    root = client.get("/")
    assert root.status_code == 200
    assert b"<!DOCTYPE html>" in root.data or b"<html" in root.data

    # Project view (existing vs missing)
    client.post("/api/projects/new", json={"name": "ui-view"})
    proj_view = client.get("/project/ui-view")
    assert proj_view.status_code == 200

    missing_view = client.get("/project/not-found")
    assert missing_view.status_code == 404


def test_revisions_api(tmp_path: Path) -> None:
    """Revisions endpoints handle listing, pagination, detail, diff, and restoration."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    app = create_app(settings)
    client = app.test_client()

    client.post("/api/projects/new", json={"name": "gear"})
    project_dir = settings.workspace_root / "gear"
    store = RevisionStore(project_dir)

    # Commit two revisions
    code_v1 = "cube([10, 10, 10]);\n"
    code_v2 = "cube([20, 20, 20]);\n"
    rev1 = store.commit(source=code_v1, origin=RevisionOrigin(kind="agent_edit"))
    rev2 = store.commit(source=code_v2, origin=RevisionOrigin(kind="agent_edit"))

    # 1. List revisions
    list_res = client.get("/api/projects/gear/revisions")
    assert list_res.status_code == 200
    revs = list_res.get_json()["revisions"]
    assert len(revs) == 2
    assert revs[0]["id"] == rev2.id
    assert revs[0]["is_active"] is True
    assert revs[1]["id"] == rev1.id

    # Pagination with limit
    paged = client.get("/api/projects/gear/revisions?limit=1")
    assert paged.status_code == 200
    paged_json = paged.get_json()
    assert len(paged_json["revisions"]) == 1
    assert paged_json["next_before"] == rev2.id

    # 2. Revision detail
    detail = client.get(f"/api/projects/gear/revisions/{rev1.id}")
    assert detail.status_code == 200
    assert detail.get_json()["id"] == rev1.id
    assert detail.get_json()["source"] == code_v1

    # Missing revision detail -> 404
    missing_detail = client.get("/api/projects/gear/revisions/00000000-0000-0000-0000-000000000000")
    assert missing_detail.status_code == 404

    # 3. Revision diff
    diff_res = client.get(f"/api/projects/gear/revisions/{rev2.id}/diff")
    assert diff_res.status_code == 200
    diff_data = diff_res.get_json()
    assert "-cube([10, 10, 10]);" in diff_data["diff"]
    assert "+cube([20, 20, 20]);" in diff_data["diff"]

    # 4. Restore revision
    # Perform restore call
    restore_res = client.post(f"/api/projects/gear/revisions/{rev1.id}/restore")
    # Restore can be 200 or 207 (Multi-Status if CAD rebuild fails/succeeds)
    assert restore_res.status_code in {200, 207}
    assert (project_dir / "model.scad").read_text(encoding="utf-8") == code_v1
    # Head revision is now the restored one
    head = store.head()
    assert head.origin.kind == "restore"
    assert head.restored_from == rev1.id


def test_preview_render_and_review_endpoints(tmp_path: Path) -> None:
    """Preview metadata, STL, render, and review evidence endpoints return expected content."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    client.post("/api/projects/new", json={"name": "bracket"})
    project_dir = settings.workspace_root / "bracket"

    # 1. When files are absent
    meta = client.get("/api/projects/bracket/preview/meta").get_json()
    assert meta["available"] is False
    assert meta["displayable"] is False

    assert client.get("/api/projects/bracket/preview").status_code == 404
    assert client.get("/api/projects/bracket/render").status_code == 404
    assert client.get("/api/projects/bracket/review/manifest").status_code == 404

    # 2. Create preview and render artifacts
    model_code = "cube([15, 15, 15]);\n"
    model_sha = hashlib.sha256(model_code.encode("utf-8")).hexdigest()
    (project_dir / "model.scad").write_text(model_code, encoding="utf-8")
    (project_dir / "preview.stl").write_bytes(b"\x00" * 84)

    img_buf = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(img_buf, format="PNG")
    render_bytes = img_buf.getvalue()
    render_sha = hashlib.sha256(render_bytes).hexdigest()
    (project_dir / "render.png").write_bytes(render_bytes)

    manifest_data = {
        "model_sha256": model_sha,
        "single_render": {"image_sha256": render_sha},
        "views": ["isometric"],
    }
    metrics = {
        "model_sha256": model_sha,
        "preview_sha256": "preview-sha",
        "metrics": {"solid_count": 1, "is_valid": True},
        "review_manifest": manifest_data,
    }
    (project_dir / ".cad_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    # Stage review evidence under .cad-agent/reviews/<model_sha>/
    rev_dir = project_dir / ".cad-agent" / "reviews" / model_sha
    rev_dir.mkdir(parents=True, exist_ok=True)
    (rev_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    (rev_dir / "review-sheet.png").write_bytes(render_bytes)
    (rev_dir / "views").mkdir(exist_ok=True)
    (rev_dir / "views" / "isometric.png").write_bytes(render_bytes)

    # 3. Test artifacts retrieval
    meta = client.get("/api/projects/bracket/preview/meta").get_json()
    assert meta["available"] is True
    assert meta["displayable"] is True
    assert meta["model_sha256"] == model_sha
    assert meta["review_status"] == "not_required"

    preview_res = client.get("/api/projects/bracket/preview")
    assert preview_res.status_code == 200
    assert len(preview_res.data) == 84

    render_res = client.get("/api/projects/bracket/render")
    assert render_res.status_code == 200
    assert render_res.mimetype == "image/png"

    manifest_res = client.get("/api/projects/bracket/review/manifest")
    assert manifest_res.status_code == 200
    assert manifest_res.get_json()["model_sha256"] == model_sha

    sheet_res = client.get("/api/projects/bracket/review/sheet")
    assert sheet_res.status_code == 200
    assert sheet_res.mimetype == "image/png"

    view_res = client.get("/api/projects/bracket/review/view/isometric")
    assert view_res.status_code == 200
    assert view_res.mimetype == "image/png"


def test_chat_endpoint_request_validation(tmp_path: Path) -> None:
    """POST /api/chat strictly validates payload parameters."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    client.post("/api/projects/new", json={"name": "chat-proj"})

    # 1. Empty message -> 400
    res = client.post("/api/chat", json={"project": "chat-proj", "message": "   "})
    assert res.status_code == 400
    assert "Message is required" in res.get_json()["error"]

    # 2. Non-existent project -> 404
    res_missing = client.post("/api/chat", json={"project": "ghost", "message": "hello"})
    assert res_missing.status_code == 404


def test_security_headers_and_cross_origin_mutations(tmp_path: Path) -> None:
    """Validate CSP header presence and origin checking on mutations."""
    # Wildcard bind settings
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "0.0.0.0", 5000
    )
    client = create_app(settings).test_client()

    # 1. CSP header on responses
    res = client.get("/api/projects")
    assert "Content-Security-Policy" in res.headers
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]

    # 2. Cross-origin mutation blocked when host is not local
    blocked = client.post(
        "/api/projects/new",
        json={"name": "evil-proj"},
        headers={"Host": "malicious.com", "Origin": "https://malicious.com"},
    )
    assert blocked.status_code == 403
    assert "Cross-origin requests are not allowed" in blocked.get_json()["error"]

    # 3. Form POST without Origin when host is non-local -> blocked
    form_blocked = client.post(
        "/api/projects/new",
        data={"name": "evil-form"},
        content_type="application/x-www-form-urlencoded",
        headers={"Host": "external.com"},
    )
    assert form_blocked.status_code == 403


def test_export_endpoints(tmp_path: Path) -> None:
    """Test the /api/projects/<name>/export endpoint across various formats and error conditions."""
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    # 1. Unknown project -> 404
    res_404 = client.get("/api/projects/ghost/export?format=stl")
    assert res_404.status_code == 404

    # Create project
    client.post("/api/projects/new", json={"name": "export-widget"})
    proj_dir = settings.workspace_root / "export-widget"

    # 2. Unsupported format -> 400
    res_400 = client.get("/api/projects/export-widget/export?format=badfmt")
    assert res_400.status_code == 400
    assert "Unsupported format" in res_400.get_json()["error"]

    # 3. Missing model.scad -> 404
    res_no_model = client.get("/api/projects/export-widget/export?format=stl")
    assert res_no_model.status_code == 404

    # Write model.scad
    scad_content = "cube([10, 10, 10]);"
    (proj_dir / "model.scad").write_text(scad_content, encoding="utf-8")

    # 4. SCAD export succeeds even without preview
    res_scad = client.get("/api/projects/export-widget/export?format=scad")
    assert res_scad.status_code == 200
    assert "text/x-scad" in res_scad.headers.get("Content-Type", "")
    disp_scad = res_scad.headers.get("Content-Disposition", "")
    assert "attachment" in disp_scad and "export-widget.scad" in disp_scad
    assert res_scad.data.decode("utf-8") == scad_content

    # 5. STL/OBJ without preview.stl -> 404
    res_no_preview = client.get("/api/projects/export-widget/export?format=stl")
    assert res_no_preview.status_code == 404
    res_no_preview_obj = client.get("/api/projects/export-widget/export?format=obj")
    assert res_no_preview_obj.status_code == 404

    # Generate preview.stl
    preview_file = proj_dir / "preview.stl"
    subprocess.run(
        ["openscad", "-o", str(preview_file), "--export-format", "binstl", str(proj_dir / "model.scad")],
        check=True,
    )

    # 6. STL export (default format, including empty format param) -> 200
    res_default = client.get("/api/projects/export-widget/export")
    assert res_default.status_code == 200
    assert "model/stl" in res_default.headers.get("Content-Type", "")
    disp_default = res_default.headers.get("Content-Disposition", "")
    assert "attachment" in disp_default and "export-widget.stl" in disp_default
    assert len(res_default.data) == preview_file.stat().st_size

    res_empty_fmt = client.get("/api/projects/export-widget/export?format=")
    assert res_empty_fmt.status_code == 200
    assert "model/stl" in res_empty_fmt.headers.get("Content-Type", "")

    # 7. OBJ export -> 200
    res_obj = client.get("/api/projects/export-widget/export?format=obj")
    assert res_obj.status_code == 200
    assert "model/obj" in res_obj.headers.get("Content-Type", "")
    disp_obj = res_obj.headers.get("Content-Disposition", "")
    assert "attachment" in disp_obj and "export-widget.obj" in disp_obj
    assert b"# OBJ export" in res_obj.data

    # 8. 3MF export -> 200
    res_3mf = client.get("/api/projects/export-widget/export?format=3mf")
    assert res_3mf.status_code == 200
    assert "model/3mf" in res_3mf.headers.get("Content-Type", "")
    disp_3mf = res_3mf.headers.get("Content-Disposition", "")
    assert "attachment" in disp_3mf and "export-widget.3mf" in disp_3mf
    assert len(res_3mf.data) > 0

