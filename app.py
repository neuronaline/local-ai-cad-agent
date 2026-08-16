"""Local AI CAD Agent web server.

A small Flask app that hosts a chat UI, a Three.js preview viewer, and
the project/revision APIs the agent loop needs.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)
from werkzeug.exceptions import RequestEntityTooLarge

from agent.core import AgentRunner
from agent.core import _history_lock_slot
from agent.images import store_images
from agent.io import atomic_write_text
from agent.revisions import RevisionIntegrityError, RevisionStore
from agent.sandbox import _BWRAP, seccomp_filter_fd
from agent.settings import Settings, load_settings

PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
INFO_EVENT_TYPES = frozenset({"agent_status", "tool_status", "agent_usage", "agent_stopped"})
HISTORY_EVENT_TYPES = INFO_EVENT_TYPES | {"agent_error"}
SSE_QUEUE_SIZE = 512
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::", ""})
_LOCAL_BINDS = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _hostname(header_value: str) -> str:
    """Extract the bare hostname from a Host or Origin header."""
    if not header_value:
        return ""
    # Strip scheme for Origin headers.
    value = header_value.split("://")[-1]
    # Strip path.
    value = value.split("/")[0]
    # Handle IPv6 brackets: [::1]:5000 -> ::1
    if value.startswith("["):
        value = value.lstrip("[").split("]")[0]
    else:
        value = value.split(":")[0]
    return value


class EventBus:
    """In-process fan-out for the single local browser client(s)."""

    def __init__(self, settings: Settings, app: Flask | None = None) -> None:
        self.settings = settings
        self.app = app
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()

    def _workspace_root(self) -> Path:
        if self.app is not None:
            try:
                current = self.app.config.get("SETTINGS")
                if current is not None:
                    return current.workspace_root
            except Exception:  # noqa: BLE001
                pass
        return self.settings.workspace_root

    def publish(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        transient: bool = False,
    ) -> None:
        # Transient events are delivered to live subscribers but never
        # appended to the conversation log. Used for terminal status events
        # (completed/failed) that exist solely to clear the UI's thinking
        # indicator and must not displace the canonical assistant message
        # that just preceded them.
        if not transient and event_type in HISTORY_EVENT_TYPES:
            project = data.get("project")
            if isinstance(project, str) and project:
                project_dir = self._workspace_root() / project
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
        raise ValueError(
            "Project name must use lowercase letters, numbers, and hyphens (no uppercase or spaces)."
        )
    path = settings.workspace_root / project_name
    if not path.is_dir():
        raise FileNotFoundError("Project not found.")
    return path


def _redact_history_event(event: dict[str, Any]) -> dict[str, Any]:
    """Replace inline image data URLs in history responses with a placeholder.

    The persisted ``conversation.jsonl`` stores the full base64 payload on
    user-image attachments so the LLM can still consume it. Returning those
    blobs through the History endpoint makes responses unnecessarily large
    and exposes content the UI does not need; substitute a lightweight
    ``[Reference image]`` marker instead.
    """
    if event.get("role") != "user":
        return event
    content = event.get("content")
    if not isinstance(content, list):
        return event
    redacted = False
    parts: list[Any] = []
    image_index = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            redacted = True
            image_index += 1
            parts.append({"type": "text", "text": f"[Reference image {image_index}]"})
        else:
            parts.append(part)
    if not redacted:
        return event
    cleaned = dict(event)
    cleaned["content"] = parts
    return cleaned


def _append_conversation(project_dir: Path, event: dict[str, Any]) -> None:
    with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, ensure_ascii=False) + "\n")


# Module-level lock guarding the idempotency cache dict. The cache is touched
# by every chat request and modified out-of-band by cleanup, so unsynchronized
# access can produce redundant eviction work or skipped targets under load.
_IDEMPOTENCY_LOCK = threading.Lock()


def _idempotency_check(app: Flask, key: str) -> bool:
    """Return True if the key is already recorded as in-flight."""
    if not key:
        return False
    with _IDEMPOTENCY_LOCK:
        cache = app.config.setdefault("IDEMPOTENCY_CACHE", {})
        return key in cache


def _idempotency_record(app: Flask, key: str) -> None:
    """Record the key as in-flight and evict the oldest half if oversized."""
    if not key:
        return
    with _IDEMPOTENCY_LOCK:
        cache = app.config.setdefault("IDEMPOTENCY_CACHE", {})
        cache[key] = True
        if len(cache) > 1000:
            oldest_keys = list(cache.keys())[:-500]
            for old_key in oldest_keys:
                cache.pop(old_key, None)


def _idempotency_forget(app: Flask, key: str) -> None:
    """Drop a key so a retry can be accepted."""
    if not key:
        return
    with _IDEMPOTENCY_LOCK:
        cache = app.config.setdefault("IDEMPOTENCY_CACHE", {})
        cache.pop(key, None)


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
    try:
        temporary_path.replace(target)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def _model_status(project_dir: Path) -> str:
    preview = project_dir / "preview.stl"
    if preview.is_file() and preview.stat().st_size > 0:
        return "has_model"
    return "none"


def _active_api_key_env(settings: Settings) -> str:
    from agent.llm_base import api_key_env
    return api_key_env(settings.llm_provider)


def _api_key_configured(settings: Settings) -> bool:
    key = os.getenv(_active_api_key_env(settings), "")
    return bool(key.strip()) and bool(settings.llm_model.strip())


def _run_preflight(settings: Settings) -> dict[str, Any]:
    checks: dict[str, bool | str] = {}
    api_key = os.getenv(_active_api_key_env(settings), "").strip()
    checks["api_key"] = bool(api_key)
    checks["provider"] = settings.llm_provider
    checks["model_configured"] = bool(settings.llm_model.strip())
    try:
        settings.workspace_root.mkdir(parents=True, exist_ok=True)
        probe = settings.workspace_root / ".preflight-probe"
        probe.write_text("ok")
        probe.unlink()
        checks["workspace_writable"] = True
    except OSError:
        checks["workspace_writable"] = False
    checks["bwrap_installed"] = _BWRAP is not None
    try:
        fd = seccomp_filter_fd()
        os.close(fd)
        checks["seccomp"] = True
    except RuntimeError:
        checks["seccomp"] = False
    try:
        import build123d  # noqa: F401
        checks["python_packages"] = True
    except ImportError:
        checks["python_packages"] = False
    return checks


def _project_lock(app: Flask, project_name: str) -> threading.Lock:
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


_PROJECT_MTIME_CANDIDATES = (
    "model.py",
    "conversation.jsonl",
    "preview.stl",
    "render.png",
)


def _project_modified_at(project_dir: Path) -> str:
    modified = project_dir.stat().st_mtime
    for name in _PROJECT_MTIME_CANDIDATES:
        candidate = project_dir / name
        try:
            if candidate.is_file():
                modified = max(modified, candidate.stat().st_mtime)
        except OSError:
            continue
    return datetime.fromtimestamp(modified, tz=timezone.utc).isoformat()


def create_app(settings: Settings | None = None) -> Flask:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    settings = settings or load_settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024
    app.config["SETTINGS"] = settings
    bus = EventBus(settings, app)
    app.config["EVENT_BUS"] = bus
    _history_lock_slot[0] = bus._history_lock
    app.config["AGENT_RUNNER"] = AgentRunner(settings, bus.publish)
    app.config["PROJECT_LOCKS"]: dict[str, threading.Lock] = {}
    app.config["PROJECT_LOCKS_LOCK"] = threading.Lock()

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "script-src 'self' 'unsafe-inline'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "form-action 'self'",
        )
        return response

    @app.before_request
    def validate_origin():
        """Reject cross-origin mutation requests when bound to localhost.

        Safe methods (GET/HEAD/OPTIONS) are always allowed. For state-changing
        methods the Origin / Host headers must match the configured bind
        address so a malicious page on another origin cannot drive the local
        service. When the bind address is a wildcard (0.0.0.0, ::) the check
        falls back to a same-origin comparison between the Host and Origin
        headers, and the Host must resolve to a loopback address.

        Form-encoded POSTs (application/x-www-form-urlencoded or
        multipart/form-data) whose Host header does not resolve to a loopback
        address are explicitly required to carry an Origin header. Browsers
        do not send Origin on simple form POSTs to a localhost target, so the
        absence of Origin on a form POST whose Host is non-local is treated
        as a forgery attempt. JSON POSTs from the bundled client include
        Origin and remain allowed.
        """
        if request.method in _ALLOWED_METHODS:
            return None
        origin = request.headers.get("Origin", "")
        host = request.headers.get("Host", "")
        if not origin and not host:
            return None  # No headers to check (e.g., direct curl).
        host_name = _hostname(host)
        origin_name = _hostname(origin)

        content_type = (request.content_type or "").split(";", 1)[0].strip().lower()
        is_form_post = request.method == "POST" and content_type in {
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        }

        bind_host = settings.host
        if bind_host in _WILDCARD_BINDS:
            # Wildcard bind — must be same-origin on loopback.
            if origin and host and origin_name and origin_name != host_name:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            if host_name not in _LOCAL_BINDS:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            if origin_name and origin_name not in _LOCAL_BINDS:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            # Block form POSTs whose Host does not resolve to a loopback
            # address. Browsers do not send Origin on simple form POSTs;
            # when the Host is loopback the request is unambiguously
            # same-origin and is allowed, including curl-style requests
            # that omit Origin.
            if is_form_post and host_name not in _LOCAL_BINDS:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
        else:
            allowed = {bind_host, "localhost", "127.0.0.1", "::1"}
            if host_name and host_name not in allowed:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            if origin_name and origin_name not in allowed:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            # Block form POSTs without an Origin header when the Host is not
            # local. Browsers do not send Origin on simple form POSTs so a
            # malicious page on another origin could otherwise drive the
            # local service; we require an Origin header for form-encoded
            # mutating requests whose Host does not already establish
            # same-origin. The Flask test client opt-in is handled by
            # sending an Origin header.
            if is_form_post and not origin and host_name not in _LOCAL_BINDS:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
        return None

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify({"error": "The request is too large; upload at most five 10 MB images."}), 413

    @app.get("/")
    def index() -> str:
        return render_template("projects.html")

    @app.get("/api/preflight")
    def preflight():
        return jsonify(_run_preflight(app.config["SETTINGS"]))

    @app.get("/project/<name>")
    def project_view(name: str) -> str:
        try:
            _project_path(app.config["SETTINGS"], name)
        except (ValueError, FileNotFoundError):
            return "Project not found.", 404
        return render_template(
            "index.html",
            project_name=name,
            show_info_messages=app.config["SETTINGS"].show_info_messages,
        )

    @app.get("/api/projects")
    def list_projects():
        workspace_root = app.config["SETTINGS"].workspace_root
        projects: list[dict[str, Any]] = []
        for item in sorted(workspace_root.iterdir()):
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
        workspace_root = app.config["SETTINGS"].workspace_root
        project_dir = workspace_root / name
        with _project_lock(app, name):
            if project_dir.exists():
                return jsonify({"error": "Project already exists."}), 409
            with tempfile.TemporaryDirectory(prefix=".new-project-", dir=workspace_root) as temporary:
                staging = Path(temporary)
                (staging / "inputs").mkdir()
                _write_project_metadata(staging, {"name": name, "created_at": _utc_now()})
                staging.rename(project_dir)
        bus.publish("project_created", {"project": name})
        return jsonify({"project": name}), 201

    @app.delete("/api/projects/<project_name>")
    def delete_project(project_name: str):
        current_settings = app.config["SETTINGS"]
        try:
            project_dir = _project_path(current_settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            try:
                project_dir = _project_path(current_settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            runner = app.config["AGENT_RUNNER"]
            if runner.has_active_state_for(project_name):
                return jsonify({"error": "Cannot delete a project with active agent state."}), 409
            shutil.rmtree(project_dir)
        return jsonify({"deleted": True})

    @app.put("/api/projects/<project_name>/rename")
    def rename_project(project_name: str):
        current_settings = app.config["SETTINGS"]
        try:
            project_dir = _project_path(current_settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        new_name = str(payload.get("name", "")).strip().lower().replace(" ", "-")
        if not PROJECT_NAME_RE.fullmatch(new_name):
            return jsonify({"error": "Use 1-63 lowercase letters, numbers, or hyphens."}), 400
        if new_name == project_name:
            return jsonify({"project": project_name})
        target = current_settings.workspace_root / new_name
        with _project_locks(app, project_name, new_name):
            try:
                project_dir = _project_path(current_settings, project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            if target.exists():
                return jsonify({"error": "A project with that name already exists."}), 409
            runner = app.config["AGENT_RUNNER"]
            if runner.has_active_state_for(project_name):
                return jsonify({"error": "Cannot rename a project with active agent state."}), 409
            metadata = _read_project_metadata(project_dir)
            metadata["name"] = new_name
            metadata["created_at"] = metadata.get("created_at") or _utc_now()
            with tempfile.NamedTemporaryFile(
                mode="w", dir=project_dir, encoding="utf-8", delete=False
            ) as temporary:
                metadata_tmp_name = Path(temporary.name).name
                json.dump(metadata, temporary, ensure_ascii=False, indent=2)
            renamed = False
            try:
                project_dir.rename(target)
                renamed = True
                (target / metadata_tmp_name).replace(target / "project.json")
            except Exception:
                (target if renamed else project_dir).joinpath(metadata_tmp_name).unlink(
                    missing_ok=True
                )
                if renamed and target.exists():
                    try:
                        target.rename(project_dir)
                        _write_project_metadata(project_dir, metadata)
                    except OSError:
                        pass
                raise
        return jsonify({"project": new_name})

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        project_name = str(payload.get("project", ""))
        message = str(payload.get("message", "")).strip()
        idempotency_key = str(payload.get("idempotency_key", "")).strip()
        if not message:
            return jsonify({"error": "Message is required."}), 400
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        runner = app.config["AGENT_RUNNER"]
        if runner.waiting_question(project_name):
            return jsonify({"error": "Answer the pending question before sending another message."}), 409
        if idempotency_key:
            if _idempotency_check(app, idempotency_key):
                if runner.is_running() or runner.is_awaiting_preview():
                    return jsonify(
                        {
                            "accepted": True,
                            "duplicate": True,
                            "error": "Your message is already being processed.",
                        }
                    ), 202
                _idempotency_forget(app, idempotency_key)
        if runner.is_running() or runner.is_awaiting_preview():
            return jsonify({"error": "An agent task is already running."}), 409
        with _project_lock(app, project_name):
            try:
                project_dir = _project_path(app.config["SETTINGS"], project_name)
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
            started = runner.start(project_name, message, image_paths)
        if not started:
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            return jsonify({"error": "Unable to start the agent task."}), 409
        if idempotency_key:
            _idempotency_record(app, idempotency_key)
        return jsonify({"accepted": True, "attachments": [
            path.relative_to(project_dir).as_posix() for path in image_paths
        ]}), 202

    @app.post("/api/questions/answer")
    def answer_question():
        payload = request.get_json(silent=True) if request.is_json else request.form
        payload = payload or {}
        project_name = str(payload.get("project", ""))
        raw_answer = payload.get("answer", "")
        answers = payload.get("answers")
        if isinstance(answers, dict):
            raw_answer = json.dumps(answers, ensure_ascii=False, separators=(",", ":"))
        else:
            raw_answer = str(raw_answer).strip()
        if not raw_answer:
            return jsonify({"error": "Answer is required."}), 400
        try:
            _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
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
                _project_path(app.config["SETTINGS"], project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
        runner = app.config["AGENT_RUNNER"]
        affected = runner.stop(project_name)
        event_project = project_name or runner.active_project()
        if event_project is None and affected:
            event_project = affected[0]
        bus.publish(
            "agent_stopped",
            {"project": event_project, "affected_projects": affected},
        )
        return jsonify({"stopped": True, "affected_projects": affected})

    @app.get("/api/projects/<project_name>/state")
    def project_state(project_name: str):
        try:
            _project_path(app.config["SETTINGS"], project_name)
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
            current_settings = app.config["SETTINGS"]
            project_dir = _project_path(current_settings, project_name)
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
                    if current_settings.show_info_messages or event.get("type") not in INFO_EVENT_TYPES:
                        events.append(_redact_history_event(event))
                except json.JSONDecodeError:
                    pass
        return jsonify({"events": events})

    @app.get("/api/projects/<project_name>/render")
    def project_render(project_name: str):
        try:
            render_path = _project_path(app.config["SETTINGS"], project_name) / "render.png"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not render_path.is_file() or render_path.stat().st_size == 0:
            return jsonify({"error": "No render has been generated."}), 404
        return send_file(render_path, mimetype="image/png", max_age=0)

    @app.get("/api/projects/<project_name>/preview")
    def preview(project_name: str):
        try:
            preview_path = _project_path(app.config["SETTINGS"], project_name) / "preview.stl"
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
            project_dir = _project_path(app.config["SETTINGS"], project_name)
            preview_path = project_dir / "preview.stl"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            return jsonify({"available": False, "displayable": False})
        stat = preview_path.stat()
        preview_sha256 = hashlib.sha256(preview_path.read_bytes()).hexdigest()
        model_path = project_dir / "model.py"
        model_sha256 = (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.is_file()
            else None
        )
        review_status = "not_required"
        displayable = not app.config["SETTINGS"].review_enabled
        if not displayable:
            review_status = "pending"
            latest_review = _review_latest(project_dir)
            if latest_review is not None:
                manifest_path = latest_review / _REVIEW_MANIFEST_NAME
                result_path = latest_review / _REVIEW_RESULT_NAME
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = result = None
                if (
                    isinstance(manifest, dict)
                    and manifest.get("preview_sha256") == preview_sha256
                    and manifest.get("model_sha256") == model_sha256
                    and isinstance(result, dict)
                    and result.get("status") in {"pass", "fail", "inconclusive"}
                ):
                    review_status = result["status"]
                    displayable = review_status == "pass"
        return jsonify({
            "available": True,
            "displayable": displayable,
            "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
            "model_sha256": model_sha256,
            "review_status": review_status,
        })

    @app.post("/api/projects/<project_name>/preview/displayed")
    def preview_displayed(project_name: str):
        try:
            _project_path(app.config["SETTINGS"], project_name)
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
            _project_path(app.config["SETTINGS"], project_name)
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

    # ------------------------------------------------------------------ #
    #  Multi-view review APIs (read-only views into .cad-agent/reviews/)
    # ------------------------------------------------------------------ #

    _REVIEW_MANIFEST_NAME = "manifest.json"
    _REVIEW_RESULT_NAME = "result.json"
    _REVIEW_VIEWS_SUBDIR = "views"
    _REVIEW_SHEET_NAME = "review-sheet.png"

    def _review_latest(project_dir: Path) -> Path | None:
        """Return the most recently modified review directory, or ``None``."""
        root = project_dir / ".cad-agent" / "reviews"
        if not root.is_dir():
            return None
        candidates = [path for path in root.iterdir() if path.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @app.get("/api/projects/<project_name>/review/manifest")
    def review_manifest(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        latest = _review_latest(project_dir)
        if latest is None:
            return jsonify({"error": "No review has been generated yet."}), 404
        manifest_path = latest / _REVIEW_MANIFEST_NAME
        if not manifest_path.is_file():
            return jsonify({"error": "Review manifest is missing."}), 404
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return jsonify({"error": "Review manifest is corrupted."}), 500
        result_path = latest / _REVIEW_RESULT_NAME
        if result_path.is_file():
            try:
                payload = dict(payload)
                payload["result"] = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        payload = dict(payload)
        payload.setdefault("artifact_dir", latest.name)
        return jsonify(payload)

    @app.get("/api/projects/<project_name>/review/sheet")
    def review_sheet(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        latest = _review_latest(project_dir)
        if latest is None:
            return jsonify({"error": "No review has been generated yet."}), 404
        sheet_path = latest / _REVIEW_SHEET_NAME
        if not sheet_path.is_file() or sheet_path.stat().st_size == 0:
            return jsonify({"error": "Review contact sheet is missing."}), 404
        return send_file(sheet_path, mimetype="image/png", max_age=0)

    @app.get("/api/projects/<project_name>/review/view/<view_id>")
    def review_view(project_name: str, view_id: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        # ``view_id`` is a filesystem identifier; restrict to the documented
        # canonical names to avoid path traversal via the URL.
        if not view_id or "/" in view_id or "\\" in view_id or view_id.startswith("."):
            return jsonify({"error": "Unknown review view."}), 404
        latest = _review_latest(project_dir)
        if latest is None:
            return jsonify({"error": "No review has been generated yet."}), 404
        view_path = latest / _REVIEW_VIEWS_SUBDIR / f"{view_id}.png"
        if not view_path.is_file() or view_path.stat().st_size == 0:
            return jsonify({"error": "Unknown review view."}), 404
        return send_file(view_path, mimetype="image/png", max_age=0)

    # ------------------------------------------------------------------ #
    #  Revision history APIs
    # ------------------------------------------------------------------ #

    _MAX_DIFF_LINES = 500

    def _revision_summary(
        store: RevisionStore,
        revision,
        active_id: str | None,
        lkg_id: str | None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "id": revision.id,
            "parent_id": revision.parent_id,
            "created_at": revision.created_at,
            "origin": revision.origin.to_dict(),
            "restored_from": revision.restored_from,
            "is_active": revision.id == active_id,
            "is_last_known_good": revision.id == lkg_id,
            "build_status": "not_run",
        }
        try:
            build = store.build_for(revision.id)
        except RevisionIntegrityError:
            build = None
        if build is not None:
            summary["build_status"] = build.status
            if build.metrics:
                summary["metrics"] = build.metrics
            if build.error:
                summary["error"] = build.error
        return summary

    @app.get("/api/projects/<project_name>/revisions")
    def list_revisions(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = RevisionStore(project_dir)
        try:
            store.reconcile()
        except RevisionIntegrityError as error:
            return jsonify({"error": str(error)}), 422
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "Revision limit must be an integer."}), 400
        limit = max(1, min(limit, 200))
        before = request.args.get("before")
        try:
            active = store.head()
            lkg = store.last_known_good()
            revisions = store.list(limit=limit, before=before)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except RevisionIntegrityError as error:
            return jsonify({"error": str(error)}), 422
        next_before = revisions[-1].id if len(revisions) == limit else None
        return jsonify({
            "revisions": [
                _revision_summary(
                    store,
                    r,
                    active.id if active else None,
                    lkg.id if lkg else None,
                )
                for r in revisions
            ],
            "next_before": next_before,
        })

    @app.get("/api/projects/<project_name>/revisions/<revision_id>")
    def revision_detail(project_name: str, revision_id: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = RevisionStore(project_dir)
        try:
            store.reconcile()
            revision = store.get(revision_id)
        except (ValueError, RevisionIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        active = store.head()
        lkg = store.last_known_good()
        summary = _revision_summary(
            store,
            revision,
            active.id if active else None,
            lkg.id if lkg else None,
        )
        try:
            summary["source"] = store.source(revision_id)
        except RevisionIntegrityError as error:
            summary["source_error"] = str(error)
        return jsonify(summary)

    @app.get("/api/projects/<project_name>/revisions/<revision_id>/diff")
    def revision_diff(project_name: str, revision_id: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = RevisionStore(project_dir)
        try:
            store.reconcile()
            revision = store.get(revision_id)
        except (ValueError, RevisionIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        # If ``?against=`` is explicitly provided (including empty string),
        # treat it as the authoritative source. Otherwise fall back to the
        # revision's parent_id (audit_183).
        if "against" in request.args:
            against_id = request.args.get("against")
        else:
            against_id = revision.parent_id
        if against_id is None or against_id == "":
            return jsonify({"diff": "", "truncated": False, "against": None})
        try:
            against = store.get(against_id)
        except ValueError as error:
            return jsonify({"error": str(error)}), 404
        except RevisionIntegrityError as error:
            if against_id != revision.parent_id or store.has_manifest(against_id):
                return jsonify({"error": str(error)}), 404
            against = None
        try:
            source_a = store.source(against_id) if against is not None else ""
            source_b = store.source(revision_id)
        except RevisionIntegrityError as error:
            return jsonify({"error": str(error)}), 422
        diff_lines = list(unified_diff(
            source_a.splitlines(keepends=True),
            source_b.splitlines(keepends=True),
            fromfile=f"revision {against.id[:8]}" if against else "retention boundary",
            tofile=f"revision {revision.id[:8]}",
            n=3,
        ))
        truncated = len(diff_lines) > _MAX_DIFF_LINES
        if truncated:
            diff_lines = diff_lines[:_MAX_DIFF_LINES]
        return jsonify({
            "diff": "".join(diff_lines),
            "truncated": truncated,
            "against": against_id if against else None,
        })

    @app.post("/api/projects/<project_name>/revisions/<revision_id>/restore")
    def restore_revision(project_name: str, revision_id: str):
        current_settings = app.config["SETTINGS"]
        try:
            project_dir = _project_path(current_settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            runner = app.config["AGENT_RUNNER"]
            if runner.has_active_state_for(project_name):
                return jsonify({"error": "Cannot restore while the agent is active."}), 409
            store = RevisionStore(
                project_dir,
                retention_count=current_settings.revision_retention_count,
            )
            try:
                store.reconcile()
                if not store.has_source(revision_id):
                    return jsonify({"error": "Revision source is missing or corrupt."}), 422
                revision = store.restore(revision_id)
            except (ValueError, RevisionIntegrityError) as error:
                return jsonify({"error": str(error)}), 422

            bus.publish("revision_updated", {"project": project_name})
            bus.publish("agent_status", {
                "project": project_name,
                "status": "restoring",
                "message": f"Restoring revision {revision.id[:8]} and rebuilding…",
            })

            from agent.tools.cad_tool import CadTool
            cad = CadTool(project_dir, bus.publish, store)
            try:
                metrics = cad.run()
            except (RuntimeError, ValueError, TypeError) as error:
                bus.publish("agent_error", {
                    "project": project_name,
                    "message": f"Restore succeeded but CAD rebuild failed: {error}",
                })
                return jsonify({
                    "restored": True,
                    "revision_id": revision.id,
                    "build_status": "failed",
                    "error": str(error),
                })

            preview_id = runner._register_preview(project_name, project_dir)
            bus.publish("preview_updated", {
                "project": project_name,
                "preview_id": preview_id,
            })
            bus.publish("agent_status", {
                "project": project_name,
                "status": "rendering",
                "message": "Restored model is being displayed…",
            })
            return jsonify({
                "restored": True,
                "revision_id": revision.id,
                "build_status": "succeeded",
                "metrics": metrics,
                "preview_id": preview_id,
            })

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
