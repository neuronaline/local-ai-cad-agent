"""Local AI CAD Agent web server."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from agent.core import AgentRunner
from agent.finalize import finalize_project
from agent.images import store_images
from agent.sandbox import _BWRAP, seccomp_filter_fd
from agent.settings import Settings, load_settings

PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
INFO_EVENT_TYPES = frozenset({"agent_status", "tool_status", "agent_usage", "agent_stopped"})
HISTORY_EVENT_TYPES = INFO_EVENT_TYPES | {"agent_error", "finalized"}
SSE_QUEUE_SIZE = 512


class EventBus:
    """In-process fan-out for the single local browser client(s)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type in HISTORY_EVENT_TYPES:
            project = data.get("project")
            if isinstance(project, str) and project:
                project_dir = self.settings.workspace_root / project
                if project_dir.is_dir():
                    with self._history_lock:
                        _append_conversation(
                            project_dir,
                            {"timestamp": _utc_now(), "type": event_type, "data": data},
                        )
        event = {"type": event_type, "data": data}
        with self._lock:
            subscribers = list(self._subscribers)
        stale: list[queue.Queue[dict[str, Any] | None]] = []
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                stale.append(subscriber)
        if stale:
            with self._lock:
                for subscriber in stale:
                    if subscriber in self._subscribers:
                        self._subscribers.remove(subscriber)
                    while True:
                        try:
                            subscriber.get_nowait()
                        except queue.Empty:
                            break
                    subscriber.put_nowait({"type": "stream_reset", "data": {}})
                    subscriber.put_nowait(None)

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        subscriber: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=SSE_QUEUE_SIZE)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any] | None]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(settings: Settings, project_name: str) -> Path:
    if not PROJECT_NAME_RE.fullmatch(project_name):
        raise ValueError("Project name must use lowercase letters, numbers, and hyphens (no uppercase or spaces).")
    path = settings.workspace_root / project_name
    if not path.is_dir():
        raise FileNotFoundError("Project not found.")
    return path


def _append_conversation(project_dir: Path, event: dict[str, Any]) -> None:
    with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, ensure_ascii=False) + "\n")


def _read_project_metadata(project_dir: Path) -> dict[str, Any]:
    metadata_path = project_dir / "project.json"
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_project_metadata(project_dir: Path, metadata: dict[str, Any]) -> None:
    target = project_dir / "project.json"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=project_dir, encoding="utf-8", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(metadata, temporary, ensure_ascii=False, indent=2)
    temporary_path.replace(target)


def _model_status(project_dir: Path) -> str:
    output = project_dir / "output"
    finalized = [output / name for name in ("report.md", "model.step", "model.stl")]
    if all(path.is_file() and path.stat().st_size > 0 for path in finalized):
        return "finalized"
    model = output / "model.stl"
    if model.is_file() and model.stat().st_size > 0:
        return "has_model"
    return "none"


def _api_key_configured(settings: Settings) -> bool:
    """Return True when a non-empty OpenRouter API key is available."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    return bool(key.strip()) and bool(settings.openrouter_model.strip())


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _run_preflight(settings: Settings) -> dict[str, Any]:
    """Return a dict with preflight check results."""
    checks: dict[str, bool | str] = {}

    # OpenRouter API key
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    checks["api_key"] = bool(api_key)

    # Model configured
    checks["model_configured"] = bool(settings.openrouter_model.strip())

    # Workspace writable
    try:
        settings.workspace_root.mkdir(parents=True, exist_ok=True)
        probe = settings.workspace_root / ".preflight-probe"
        probe.write_text("ok")
        probe.unlink()
        checks["workspace_writable"] = True
    except OSError:
        checks["workspace_writable"] = False

    # Bubblewrap installed
    checks["bwrap_installed"] = _BWRAP is not None

    # seccomp functional
    try:
        fd = seccomp_filter_fd()
        os.close(fd)
        checks["seccomp"] = True
    except RuntimeError:
        checks["seccomp"] = False

    # Python packages
    try:
        import build123d  # noqa: F401
        checks["python_packages"] = True
    except ImportError:
        checks["python_packages"] = False

    return checks


def _project_lock(app: Flask, project_name: str) -> threading.Lock:
    """Return a per-project lock to serialize concurrent operations on the same project."""
    locks: dict[str, threading.Lock] = app.config["PROJECT_LOCKS"]
    lock = locks.get(project_name)
    if lock is None:
        with app.config["PROJECT_LOCKS_LOCK"]:
            lock = locks.get(project_name)
            if lock is None:
                lock = threading.Lock()
                locks[project_name] = lock
    return lock


@contextmanager
def _project_locks(app: Flask, *project_names: str):
    locks = [_project_lock(app, name) for name in sorted(set(project_names))]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _project_modified_at(project_dir: Path) -> str:
    modified = project_dir.stat().st_mtime
    for path in project_dir.rglob("*"):
        try:
            if path.is_file():
                modified = max(modified, path.stat().st_mtime)
        except OSError:
            continue
    return datetime.fromtimestamp(modified, tz=timezone.utc).isoformat()


def _save_env_key(env_path: Path, api_key: str) -> None:
    """Write OPENROUTER_API_KEY to .env, preserving other lines."""
    lines: list[str] = []
    found = False
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for line in lines:
        if line.strip().startswith("OPENROUTER_API_KEY"):
            new_lines.append(f"OPENROUTER_API_KEY={api_key}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"OPENROUTER_API_KEY={api_key}\n")
    env_path.write_text("".join(new_lines), encoding="utf-8")


def _save_config_model(config_path: Path, model: str) -> None:
    """Set openrouter.model in config.yaml, preserving the rest."""
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("openrouter", {})
    if not isinstance(data["openrouter"], dict):
        data["openrouter"] = {}
    data["openrouter"]["model"] = model
    config_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def create_app(settings: Settings | None = None) -> Flask:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    settings = settings or load_settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    bus = EventBus(settings)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024
    app.config["SETTINGS"] = settings
    app.config["EVENT_BUS"] = bus
    app.config["AGENT_RUNNER"] = AgentRunner(settings, bus.publish)
    app.config["PROJECT_LOCKS"]: dict[str, threading.Lock] = {}
    app.config["PROJECT_LOCKS_LOCK"] = threading.Lock()

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify({"error": "The request is too large; upload at most five 10 MB images."}), 413

    @app.get("/")
    def index() -> str:
        if not _api_key_configured(settings):
            return redirect(url_for("setup_page"))
        return render_template("projects.html")

    @app.get("/setup")
    def setup_page() -> str:
        if _api_key_configured(settings):
            return redirect(url_for("index"))
        return render_template("setup.html")

    @app.post("/api/setup")
    def setup_save():
        payload = request.get_json(silent=True) or {}
        api_key = str(payload.get("api_key", "")).strip()
        model = str(payload.get("model", "")).strip()
        if not api_key:
            return jsonify({"error": "An API key is required."}), 400
        if not model:
            return jsonify({"error": "A model name is required."}), 400
        root = _project_root()
        env_path = root / ".env"
        config_path = root / "config.yaml"
        try:
            _save_env_key(env_path, api_key)
            _save_config_model(config_path, model)
        except OSError as error:
            return jsonify({"error": f"Could not save settings: {error}"}), 500
        # Reload env so subsequent requests see the key.
        load_dotenv(env_path, override=True)
        # Rebuild settings with new model.
        new_settings = load_settings()
        app.config["SETTINGS"] = new_settings
        app.config["AGENT_RUNNER"].settings = new_settings
        return jsonify({"ok": True})

    @app.get("/api/preflight")
    def preflight():
        current_settings = app.config["SETTINGS"]
        checks = _run_preflight(current_settings)
        return jsonify(checks)

    @app.get("/project/<name>")
    def project_view(name: str) -> str:
        if not _api_key_configured(settings):
            return redirect(url_for("setup_page"))
        try:
            _project_path(settings, name)
        except (ValueError, FileNotFoundError):
            return "Project not found.", 404
        return render_template(
            "index.html",
            project_name=name,
            show_info_messages=settings.show_info_messages,
        )

    @app.get("/api/projects")
    def list_projects():
        projects: list[dict[str, Any]] = []
        for item in sorted(settings.workspace_root.iterdir()):
            if not item.is_dir() or not PROJECT_NAME_RE.fullmatch(item.name):
                continue
            metadata = _read_project_metadata(item)
            projects.append({
                "name": item.name,
                "created_at": metadata.get("created_at"),
                "modified_at": _project_modified_at(item),
                "model_status": _model_status(item),
            })
        return jsonify({"projects": projects})

    @app.post("/api/projects/new")
    def new_project():
        payload = request.get_json(silent=True) or {}
        name = str(payload.get("name", "")).strip().lower().replace(" ", "-")
        if not PROJECT_NAME_RE.fullmatch(name):
            return jsonify({"error": "Use 1-63 lowercase letters, numbers, or hyphens."}), 400
        project_dir = settings.workspace_root / name
        with _project_lock(app, name):
            if project_dir.exists():
                return jsonify({"error": "Project already exists."}), 409
            with tempfile.TemporaryDirectory(prefix=".new-project-", dir=settings.workspace_root) as temporary:
                staging = Path(temporary)
                (staging / "inputs").mkdir()
                (staging / "output").mkdir()
                (staging / "summary.md").write_text(
                    f"# {name}\n\nNo model has been created yet.\n", encoding="utf-8"
                )
                (staging / "conversation.jsonl").write_text("", encoding="utf-8")
                (staging / "api_messages.jsonl").write_text("", encoding="utf-8")
                _write_project_metadata(staging, {"name": name, "created_at": _utc_now()})
                staging.rename(project_dir)
        bus.publish("project_created", {"project": name})
        return jsonify({"project": name}), 201

    @app.delete("/api/projects/<project_name>")
    def delete_project(project_name: str):
        try:
            project_dir = _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            try:
                project_dir = _project_path(settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            runner = app.config["AGENT_RUNNER"]
            if (
                (runner.is_running() and runner.active_project() == project_name)
                or runner.is_awaiting_preview(project_name)
                or runner.waiting_question(project_name)
            ):
                return jsonify({"error": "Cannot delete a project with active agent state."}), 409
            shutil.rmtree(project_dir)
        return jsonify({"deleted": True})

    @app.put("/api/projects/<project_name>/rename")
    def rename_project(project_name: str):
        try:
            project_dir = _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        new_name = str(payload.get("name", "")).strip().lower().replace(" ", "-")
        if not PROJECT_NAME_RE.fullmatch(new_name):
            return jsonify({"error": "Use 1-63 lowercase letters, numbers, or hyphens."}), 400
        if new_name == project_name:
            return jsonify({"project": project_name})
        target = settings.workspace_root / new_name
        with _project_locks(app, project_name, new_name):
            try:
                project_dir = _project_path(settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            if target.exists():
                return jsonify({"error": "A project with that name already exists."}), 409
            runner = app.config["AGENT_RUNNER"]
            if (
                (runner.is_running() and runner.active_project() == project_name)
                or runner.is_awaiting_preview(project_name)
                or runner.waiting_question(project_name)
            ):
                return jsonify({"error": "Cannot rename a project with active agent state."}), 409
            project_dir.rename(target)
            metadata = _read_project_metadata(target)
            metadata["name"] = new_name
            metadata["created_at"] = metadata.get("created_at") or _utc_now()
            _write_project_metadata(target, metadata)
            summary_path = target / "summary.md"
            if summary_path.is_file():
                content = summary_path.read_text(encoding="utf-8")
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("# "):
                        lines[i] = f"# {new_name}"
                        break
                summary_path.write_text("\n".join(lines), encoding="utf-8")
        return jsonify({"project": new_name})

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        project_name = str(payload.get("project", ""))
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "Message is required."}), 400
        try:
            project_dir = _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        runner = app.config["AGENT_RUNNER"]
        if runner.waiting_question(project_name):
            return jsonify({"error": "Answer the pending question before sending another message."}), 409
        if runner.is_running() or runner.is_awaiting_preview():
            return jsonify({"error": "An agent task is already running."}), 409
        with _project_lock(app, project_name):
            try:
                project_dir = _project_path(settings, project_name)
                image_paths = store_images(request.files.getlist("attachments"), project_dir)
            except FileNotFoundError as error:
                return jsonify({"error": str(error)}), 404
            except (ValueError, OSError) as error:
                return jsonify({"error": str(error)}), 400
            if (
                runner.waiting_question(project_name)
                or runner.is_running()
                or runner.is_awaiting_preview()
            ):
                for image_path in image_paths:
                    image_path.unlink(missing_ok=True)
                return jsonify({"error": "An agent task started while the upload was being processed."}), 409
            event = {
                "timestamp": _utc_now(),
                "role": "user",
                "content": message,
                "attachments": [
                    path.relative_to(project_dir).as_posix() for path in image_paths
                ],
            }
            # Start under the lock so a failed start cannot leave a phantom
            # user message in the conversation log.
            started = runner.start(project_name, message, image_paths)
            if started:
                _append_conversation(project_dir, event)
        if not started:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            return jsonify({"error": "Unable to start the agent task."}), 409
        return jsonify({"accepted": True, "attachments": event["attachments"]}), 202

    @app.post("/api/questions/answer")
    def answer_question():
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        project_name = str(payload.get("project", ""))
        # Accept either legacy flat string or structured JSON answers dict.
        raw_answer = payload.get("answer", "")
        answers = payload.get("answers")
        if isinstance(answers, dict):
            raw_answer = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
        else:
            raw_answer = str(raw_answer).strip()
        if not raw_answer:
            return jsonify({"error": "Answer is required."}), 400
        try:
            _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            try:
                _project_path(settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            accepted = app.config["AGENT_RUNNER"].answer(project_name, raw_answer)
        if not accepted:
            if app.config["AGENT_RUNNER"].waiting_question(project_name):
                return jsonify({"error": "The answer does not match the requested input type or options."}), 400
            return jsonify({"error": "No question is awaiting an answer for this project."}), 409
        return jsonify({"accepted": True}), 202

    @app.post("/api/stop")
    def stop():
        payload = request.get_json(silent=True) if request.is_json else request.form
        project_name = str((payload or {}).get("project", "")) or None
        if project_name:
            try:
                _project_path(settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
        runner = app.config["AGENT_RUNNER"]
        event_project = project_name or runner.active_project()
        runner.stop(project_name)
        bus.publish("agent_stopped", {"project": event_project})
        return jsonify({"stopped": True})

    @app.get("/api/projects/<project_name>/state")
    def project_state(project_name: str):
        try:
            _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        runner = app.config["AGENT_RUNNER"]
        question = runner.waiting_question(project_name)
        if question:
            return jsonify({"status": "waiting_for_user", "question": question})
        if runner.is_running() and runner.active_project() == project_name:
            return jsonify({"status": "running"})
        preview_id = runner.pending_preview_id(project_name)
        if preview_id:
            return jsonify({"status": "rendering", "preview_id": preview_id})
        return jsonify({"status": "idle"})

    @app.get("/api/projects/<project_name>/history")
    def project_history(project_name: str):
        try:
            project_dir = _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        log_path = project_dir / "conversation.jsonl"
        if not log_path.is_file():
            return jsonify({"events": []})
        events = []
        with log_path.open("r", encoding="utf-8") as log:
            for line in log:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        continue
                    if settings.show_info_messages or event.get("type") not in INFO_EVENT_TYPES:
                        events.append(event)
                except json.JSONDecodeError:
                    pass
        return jsonify({"events": events})

    @app.post("/api/projects/<project_name>/finalize")
    def finalize(project_name: str):
        try:
            with _project_lock(app, project_name):
                project_dir = _project_path(settings, project_name)
                runner = app.config["AGENT_RUNNER"]
                if (
                    runner.is_running()
                    or runner.is_awaiting_preview()
                    or runner.waiting_question(project_name)
                ):
                    return jsonify({"error": "Stop or finish the active agent task before finalizing."}), 409
                bus.publish("agent_status", {
                    "project": project_name,
                    "status": "finalizing",
                    "message": "Finalization started — running CAD, rendering, and exporting in one pass…",
                })
                result = finalize_project(project_dir)
        except (ValueError, RuntimeError, TypeError, KeyError, OSError) as error:
            bus.publish("agent_error", {"project": project_name, "message": str(error)})
            return jsonify({"error": str(error)}), 400
        bus.publish("agent_status", {
            "project": project_name,
            "status": "finalizing",
            "message": "Report written and output artifacts verified.",
        })
        report_text = result.pop("report_text", "")
        bus.publish("finalized", {"project": project_name, **result, "report_text": report_text})
        bus.publish("preview_updated", {"project": project_name})
        result["report_text"] = report_text
        return jsonify(result)

    @app.get("/api/projects/<project_name>/render")
    def project_render(project_name: str):
        try:
            render_path = _project_path(settings, project_name) / "render.png"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not render_path.is_file() or render_path.stat().st_size == 0:
            return jsonify({"error": "No render has been generated."}), 404
        return send_file(render_path, mimetype="image/png", max_age=0)

    @app.get("/api/projects/<project_name>/output/report")
    def project_report(project_name: str):
        try:
            report_path = _project_path(settings, project_name) / "output" / "report.md"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not report_path.is_file():
            return jsonify({"error": "No report has been generated."}), 404
        return send_file(report_path, mimetype="text/markdown", max_age=0)

    @app.get("/api/projects/<project_name>/output/<path:filename>")
    def project_output_file(project_name: str, filename: str):
        try:
            project_dir = _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        allowed = {"model.step", "model.stl"}
        if filename not in allowed:
            return jsonify({"error": "File not found."}), 404
        file_path = project_dir / "output" / filename
        if not file_path.is_file():
            return jsonify({"error": "File not found."}), 404
        mimetypes = {"model.step": "application/step", "model.stl": "model/stl"}
        return send_file(file_path, mimetype=mimetypes.get(filename, "application/octet-stream"), max_age=0)

    @app.get("/api/projects/<project_name>/preview")
    def preview(project_name: str):
        try:
            preview_path = _project_path(settings, project_name) / "preview.stl"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not preview_path.is_file():
            return jsonify({"error": "No preview has been generated."}), 404
        if preview_path.stat().st_size == 0:
            return jsonify({"error": "The generated preview is empty."}), 422
        return send_file(preview_path, mimetype="model/stl", max_age=0)

    @app.get("/api/projects/<project_name>/preview/meta")
    def preview_meta(project_name: str):
        try:
            preview_path = _project_path(settings, project_name) / "preview.stl"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            return jsonify({"available": False})
        stat = preview_path.stat()
        return jsonify({
            "available": True,
            "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
        })

    @app.post("/api/projects/<project_name>/preview/displayed")
    def preview_displayed(project_name: str):
        try:
            _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        preview_id = str(payload.get("preview_id", ""))
        if not preview_id:
            return jsonify({"error": "Preview id is required."}), 400
        if not app.config["AGENT_RUNNER"].confirm_preview(project_name, preview_id):
            return jsonify({"error": "The preview is stale, missing, or does not match this task."}), 409
        return jsonify({"displayed": True})

    @app.post("/api/projects/<project_name>/preview/failed")
    def preview_failed(project_name: str):
        try:
            _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        preview_id = str(payload.get("preview_id", ""))
        message = str(payload.get("message", "")).strip()
        if not preview_id:
            return jsonify({"error": "Preview id is required."}), 400
        if not app.config["AGENT_RUNNER"].fail_preview(project_name, preview_id, message):
            return jsonify({"error": "The preview does not belong to this task."}), 409
        return jsonify({"failed": True})

    @app.post("/api/projects/<project_name>/screenshot")
    def screenshot_capture(project_name: str):
        try:
            _project_path(settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        request_id = str(payload.get("request_id", "")).strip()
        image_b64 = str(payload.get("image", "")).strip()
        capture_error = str(payload.get("error", "")).strip()[:500]
        if not request_id:
            return jsonify({"error": "Screenshot request id is required."}), 400
        if not image_b64 and not capture_error:
            return jsonify({"error": "A screenshot image or capture error is required."}), 400
        if len(image_b64) > 7 * 1024 * 1024:
            return jsonify({"error": "Screenshot exceeds 5 MB."}), 413
        runner = app.config["AGENT_RUNNER"]
        if not runner.deliver_screenshot(project_name, request_id, image_b64, capture_error):
            return jsonify({"error": "No active screenshot request for this project."}), 409
        return jsonify({"captured": True})

    @app.get("/api/stream")
    def stream():
        subscriber = bus.subscribe()

        @stream_with_context
        def generate():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        event = subscriber.get(timeout=15)
                        if event is None:
                            break
                        yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                bus.unsubscribe(subscriber)

        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    current_settings = load_settings()
    create_app(current_settings).run(host=current_settings.host, port=current_settings.port, debug=False, threaded=True)
