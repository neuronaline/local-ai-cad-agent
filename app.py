"""Local AI CAD Agent web server."""
from __future__ import annotations
import urllib.parse

import hashlib
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import unified_diff
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

from agent.constraints import ConstraintError, ConstraintStore, ModelConstraintValidator
from agent.core import AgentRunner
from agent.core import _history_lock_slot
from agent.finalize import finalize_project
from agent.images import store_images
from agent.quality.errors import (
    ACCEPTED_DECISION_TYPES,
    DECISION_TYPES,
    ISSUE_SEVERITIES,
)
from agent.quality.models import Issue
from agent.quality.store import (
    QualityError,
    QualityIntegrityError,
    QualityLimitError,
    QualityStore,
)
from agent.io import atomic_write_text
from agent.revisions import RevisionIntegrityError, RevisionStore
from agent.sandbox import _BWRAP, seccomp_filter_fd
from agent.settings import Settings, load_settings
from agent.tools.cad_tool import CadTool

PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
INFO_EVENT_TYPES = frozenset({"agent_status", "tool_status", "agent_usage", "agent_stopped"})
HISTORY_EVENT_TYPES = INFO_EVENT_TYPES | {
    "agent_error",
    "finalized",
    "design_accepted",
    "design_rejected",
}
SSE_QUEUE_SIZE = 512


class EventBus:
    """In-process fan-out for the single local browser client(s)."""

    def __init__(self, settings: Settings, app: Flask | None = None) -> None:
        self.settings = settings
        self.app = app
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()

    def _workspace_root(self) -> Path:
        """Return the active workspace root, picking up post-setup reloads.

        The bus is bound at app construction time but settings may be replaced
        by ``/api/setup``; fall back to ``app.config`` whenever a Flask app is
        attached so history events land in the correct project directory
        (audit_078).
        """
        if self.app is not None:
            try:
                current = self.app.config.get("SETTINGS")
                if current is not None:
                    return current.workspace_root
            except Exception:  # noqa: BLE001 - defensive: app teardown path
                pass
        return self.settings.workspace_root

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type in HISTORY_EVENT_TYPES:
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
        raise ValueError("Project name must use lowercase letters, numbers, and hyphens (no uppercase or spaces).")
    path = settings.workspace_root / project_name
    if not path.is_dir():
        raise FileNotFoundError("Project not found.")
    return path


def _valid_uuid(value: str) -> bool:
    """Accept both dashed and compact hex UUID forms used by quality records."""
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


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
        # Check if the finalized output matches the active model source.
        model_path = project_dir / "model.py"
        if not model_path.is_file():
            return "stale"
        active_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        meta_path = output / ".finalize_meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return "stale"
        if not isinstance(meta, dict) or meta.get("model_sha256") != active_digest:
            return "stale"
        return "finalized"
    model = output / "model.stl"
    if model.is_file() and model.stat().st_size > 0:
        return "has_model"
    return "none"


def _active_api_key_env(settings: Settings) -> str:
    """Return the env var name that supplies the API key for the active provider."""
    from agent.llm_base import api_key_env
    return api_key_env(settings.llm_provider)


def _api_key_configured(settings: Settings) -> bool:
    """Return True when a non-empty API key is available for the active provider."""
    key = os.getenv(_active_api_key_env(settings), "")
    return bool(key.strip()) and bool(settings.llm_model.strip())


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _run_preflight(settings: Settings) -> dict[str, Any]:
    """Return a dict with preflight check results."""
    checks: dict[str, bool | str] = {}

    # Active provider's API key (OpenRouter or OpenAI)
    api_key = os.getenv(_active_api_key_env(settings), "").strip()
    checks["api_key"] = bool(api_key)
    checks["provider"] = settings.llm_provider

    # Model configured for the active provider
    checks["model_configured"] = bool(settings.llm_model.strip())

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


# Files whose mtimes actually drive the project-card "modified at" (audit_035).
_PROJECT_MTIME_CANDIDATES = (
    "model.py",
    "conversation.jsonl",
    "summary.md",
    "report.md",
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


def _reject_newlines(value: str, name: str) -> None:
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must not contain line breaks.")


def _save_env_api_key(env_path: Path, env_var: str, api_key: str) -> None:
    """Write ``env_var=api_key`` to ``.env``, preserving other lines.

    The key is matched strictly against ``<env_var>=...`` assignments only
    (never a prefix such as ``OPENAI_API_KEY_FOO``) and the file is replaced
    atomically so a crash cannot leave a truncated .env behind.
    """
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env_var):
        raise ValueError(f"Invalid environment variable name: {env_var!r}")
    _reject_newlines(api_key, "API key")
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: list[str] = []
    pattern = re.compile(rf"^[ \t]*{re.escape(env_var)}[ \t]*=[ \t]*(.*)$")
    found = False
    for line in lines:
        if pattern.match(line):
            new_lines.append(f"{env_var}={api_key}\n")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{env_var}={api_key}\n")
    atomic_write_text(env_path, "".join(new_lines))


def _save_config_setup(config_path: Path, provider: str, model: str) -> None:
    """Persist the active provider and its model into ``config.yaml``.

    Always writes ``llm.provider`` and the model under the matching namespace
    (``openrouter.model`` or ``openai.model``). Other sections are preserved.
    """
    from agent.settings import LLM_PROVIDERS

    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
    _reject_newlines(model, "Model")
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    llm = data.setdefault("llm", {})
    if not isinstance(llm, dict):
        llm = {}
        data["llm"] = llm
    llm["provider"] = provider
    section = data.setdefault(provider, {})
    if not isinstance(section, dict):
        section = {}
        data[provider] = section
    section["model"] = model
    atomic_write_text(
        config_path,
        yaml.dump(data, default_flow_style=False, allow_unicode=True),
    )


def create_app(settings: Settings | None = None) -> Flask:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    settings = settings or load_settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024
    app.config["SETTINGS"] = settings
    bus = EventBus(settings, app)
    app.config["EVENT_BUS"] = bus
    # Share the bus's history lock with the agent loop so the
    # conversation.jsonl mirror written by _append_api_message serializes with
    # the events the bus publishes (audit_078 follow-up).
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

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify({"error": "The request is too large; upload at most five 10 MB images."}), 413

    def _hostname(header_value: str) -> str:
        """Extract the bare hostname from a Host or Origin header.

        Delegates to urllib.parse which handles IPv6 brackets, missing
        schemes, and ports correctly (audit_036).
        """
        if not header_value:
            return ""
        # Add scheme if missing so urlparse returns the right hostname field.
        raw = header_value if "://" in header_value else f"//{header_value}"
        return urllib.parse.urlparse(raw).hostname or ""

    @app.before_request
    def validate_origin():
        """Reject cross-origin mutation requests when bound to localhost.

        GET/HEAD/OPTIONS are always allowed. For state-changing methods the
        Origin / Host headers must match the configured bind address so a
        malicious page on another origin cannot drive the local service.

        When the bind address is a wildcard (0.0.0.0, ::) the check falls back
        to a same-origin comparison between the Host and Origin headers.
        """
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None
        origin = request.headers.get("Origin", "")
        host = request.headers.get("Host", "")
        if not origin and not host:
            return None  # No headers to check (e.g., direct curl).
        host_name = _hostname(host)
        origin_name = _hostname(origin)

        bind_host = app.config["SETTINGS"].host
        _WILDCARD_BINDS = frozenset({"0.0.0.0", "::", ""})
        if bind_host in _WILDCARD_BINDS:
            # Wildcard bind — must be same-origin.
            if origin and host and origin_name and origin_name != host_name:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            if host_name not in ("localhost", "127.0.0.1", "::1") or not host_name:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
        else:
            allowed = {bind_host, "localhost", "127.0.0.1", "::1"}
            if host_name and host_name not in allowed:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
            if origin_name and origin_name not in allowed:
                return jsonify({"error": "Cross-origin requests are not allowed."}), 403
        return None

    @app.get("/")
    def index() -> str:
        current = app.config["SETTINGS"]
        if not _api_key_configured(current):
            return redirect(url_for("setup_page"))
        return render_template("projects.html")

    @app.get("/setup")
    def setup_page() -> str:
        current = app.config["SETTINGS"]
        if _api_key_configured(current):
            return redirect(url_for("index"))
        return render_template("setup.html")

    @app.post("/api/setup")
    def setup_save():
        from agent.settings import LLM_PROVIDERS

        payload = request.get_json(silent=True) or {}
        api_key = str(payload.get("api_key", "")).strip()
        model = str(payload.get("model", "")).strip()
        provider = str(payload.get("provider", "openrouter")).strip().lower() or "openrouter"
        if provider not in LLM_PROVIDERS:
            return jsonify({"error": f"Unknown provider: {provider!r}."}), 400
        if not api_key:
            return jsonify({"error": "An API key is required."}), 400
        if not model:
            return jsonify({"error": "A model name is required."}), 400
        if "\n" in api_key or "\r" in api_key:
            return jsonify({"error": "The API key must not contain line breaks."}), 400
        if "\n" in model or "\r" in model:
            return jsonify({"error": "The model name must not contain line breaks."}), 400
        root = _project_root()
        env_path = root / ".env"
        config_path = root / "config.yaml"
        env_var = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        try:
            _save_env_api_key(env_path, env_var, api_key)
            _save_config_setup(config_path, provider, model)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except OSError as error:
            return jsonify({"error": f"Could not save settings: {error}"}), 500
        # Reload env so subsequent requests see the key.
        load_dotenv(env_path, override=True)
        # Rebuild settings with new provider + model.
        new_settings = load_settings()
        app.config["SETTINGS"] = new_settings
        app.config["AGENT_RUNNER"].settings = new_settings
        app.config["EVENT_BUS"].settings = new_settings
        return jsonify({"ok": True})

    @app.get("/api/preflight")
    def preflight():
        current_settings = app.config["SETTINGS"]
        checks = _run_preflight(current_settings)
        return jsonify(checks)

    @app.get("/project/<name>")
    def project_view(name: str) -> str:
        current = app.config["SETTINGS"]
        if not _api_key_configured(current):
            return redirect(url_for("setup_page"))
        try:
            _project_path(current, name)
        except (ValueError, FileNotFoundError):
            return "Project not found.", 404
        return render_template(
            "index.html",
            project_name=name,
            show_info_messages=current.show_info_messages,
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
            # Prepare all updated file contents before moving anything so a
            # mid-rename failure cannot leave project.json behind the rename.
            metadata = _read_project_metadata(project_dir)
            original_metadata = dict(metadata)
            metadata["name"] = new_name
            metadata["created_at"] = metadata.get("created_at") or _utc_now()
            summary_content: str | None = None
            summary_path = project_dir / "summary.md"
            if summary_path.is_file():
                content = summary_path.read_text(encoding="utf-8")
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("# "):
                        lines[i] = f"# {new_name}"
                        break
                summary_content = "\n".join(lines)

            # Stage the new metadata in a temp file inside the source dir so
            # the write cannot fail after the directory has moved.
            with tempfile.NamedTemporaryFile(
                mode="w", dir=project_dir, encoding="utf-8", delete=False
            ) as temporary:
                metadata_tmp_name = Path(temporary.name).name
                json.dump(metadata, temporary, ensure_ascii=False, indent=2)

            renamed = False
            try:
                project_dir.rename(target)
                renamed = True
                # Rebuild the staged path: the directory moved under it.
                (target / metadata_tmp_name).replace(target / "project.json")
                if summary_content is not None:
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=target, encoding="utf-8", delete=False
                    ) as temporary:
                        summary_tmp = Path(temporary.name)
                        temporary.write(summary_content)
                    summary_tmp.replace(target / "summary.md")
            except Exception:
                (target if renamed else project_dir).joinpath(metadata_tmp_name).unlink(
                    missing_ok=True
                )
                if renamed and target.exists():
                    # Best-effort rollback so a failed update never leaves a
                    # half-renamed project; restore the original metadata.
                    try:
                        target.rename(project_dir)
                        _write_project_metadata(project_dir, original_metadata)
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
        # Idempotency: check before the running guard so the retry of a
        # successfully accepted submission receives a graceful response.
        if idempotency_key:
            cache = app.config.setdefault("IDEMPOTENCY_CACHE", {})
            if idempotency_key in cache:
                if runner.is_running() or runner.is_awaiting_preview():
                    return jsonify(
                        {
                            "accepted": True,
                            "duplicate": True,
                            "error": "Your message is already being processed.",
                        }
                    ), 202
                # Stale entry: runner finished but key was never cleaned.
                cache.pop(idempotency_key, None)
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
                if idempotency_key:
                    cache = app.config.setdefault("IDEMPOTENCY_CACHE", {})
                    cache[idempotency_key] = True
                    if len(cache) > 1000:
                        oldest_keys = list(cache.keys())[:-500]
                        for key in oldest_keys:
                            cache.pop(key, None)
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
            _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            try:
                _project_path(app.config["SETTINGS"], project_name)
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
                        events.append(event)
                except json.JSONDecodeError:
                    pass
        return jsonify({"events": events})

    @app.post("/api/projects/<project_name>/finalize")
    def finalize(project_name: str):
        try:
            with _project_lock(app, project_name):
                current_settings = app.config["SETTINGS"]
                project_dir = _project_path(current_settings, project_name)
                runner = app.config["AGENT_RUNNER"]
                if (
                    runner.is_running()
                    or runner.is_awaiting_preview()
                    or runner.waiting_question(project_name)
                ):
                    return jsonify({"error": "Stop or finish the active agent task before finalizing."}), 409
                finalization_status = "accepted"
                bypassed = False
                if current_settings.quality_require_acceptance_before_finalize:
                    body = request.get_json(silent=True) or {}
                    force_value = body.get("force")
                    force = force_value is True
                    quality = QualityStore(project_dir)
                    revisions = RevisionStore(project_dir)
                    revisions.reconcile()
                    head = revisions.head()
                    decision = (
                        quality.latest_decision_for_revision(head.id)
                        if head is not None
                        else None
                    )
                    accepted = (
                        decision is not None
                        and decision.decision in ACCEPTED_DECISION_TYPES
                    )
                    blocking_issues = (
                        [
                            issue
                            for issue in quality.issues_for_revision(head.id)
                            if issue.status == "open" and issue.severity == "blocking"
                        ]
                        if head is not None
                        else []
                    )
                    if not accepted or blocking_issues:
                        if not force:
                            if blocking_issues:
                                return jsonify({
                                    "error": "Resolve or explicitly bypass open blocking issues before finalizing.",
                                    "code": "BLOCKING_ISSUES_OPEN",
                                    "hint": "Accept this design or record a manual bypass.",
                                }), 409
                            return jsonify({
                                "error": "The current design has not been explicitly accepted. Review the preview and click Accept design first (or force a recorded local bypass).",
                                "code": "ACCEPTANCE_REQUIRED",
                                "hint": "Accept this design or record a manual bypass.",
                            }), 409
                        # Recorded manual bypass: never silent.
                        try:
                            runs = quality.list_runs(limit=1)
                            bypass_run = runs[0] if runs else quality.start_run(
                                project=project_name
                            )
                            quality.append_event(
                                bypass_run.run_id,
                                "finalize_bypassed",
                                {
                                    "revision_id": head.id if head else None,
                                    "decision": decision.decision if decision else None,
                                    "blocking_issue_count": len(blocking_issues),
                                },
                            )
                            if not runs:
                                quality.complete_run(
                                    bypass_run.run_id, status="completed"
                                )
                        except (QualityError, QualityIntegrityError) as error:
                            return jsonify({
                                "error": f"Cannot audit finalization bypass: {error}",
                                "code": "BYPASS_AUDIT_FAILED",
                            }), 422
                        finalization_status = "finalized_with_bypass"
                        bypassed = True
                    elif decision is not None and decision.decision == "accepted_with_limitations":
                        finalization_status = "accepted_with_limitations"
                bus.publish("agent_status", {
                    "project": project_name,
                    "status": "finalizing",
                    "message": "Finalization started — running CAD, rendering, and exporting in one pass…",
                })
                result = finalize_project(
                    project_dir, finalization_status=finalization_status
                )
        except (ValueError, RuntimeError, TypeError, KeyError, OSError) as error:
            bus.publish("agent_error", {"project": project_name, "message": str(error)})
            return jsonify({"error": str(error)}), 400
        bus.publish("agent_status", {
            "project": project_name,
            "status": "finalizing",
            "message": "Report written and output artifacts verified.",
        })
        report_text = result.pop("report_text", "")
        bus.publish(
            "finalized",
            {
                "project": project_name,
                **result,
                "report_text": report_text,
                "bypassed": bypassed,
            },
        )
        bus.publish("preview_updated", {"project": project_name})
        result["report_text"] = report_text
        result["bypassed"] = bypassed
        return jsonify(result)

    @app.get("/api/projects/<project_name>/render")
    def project_render(project_name: str):
        try:
            render_path = _project_path(app.config["SETTINGS"], project_name) / "render.png"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not render_path.is_file() or render_path.stat().st_size == 0:
            return jsonify({"error": "No render has been generated."}), 404
        return send_file(render_path, mimetype="image/png", max_age=0)

    @app.get("/api/projects/<project_name>/output/report")
    def project_report(project_name: str):
        try:
            report_path = _project_path(app.config["SETTINGS"], project_name) / "output" / "report.md"
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        if not report_path.is_file():
            return jsonify({"error": "No report has been generated."}), 404
        return send_file(report_path, mimetype="text/markdown", max_age=0)

    @app.get("/api/projects/<project_name>/output/finalize-meta")
    def project_finalize_meta(project_name: str):
        """Return ``.finalize_meta.json`` (or 404) for the active finalization."""
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        meta_path = project_dir / "output" / ".finalize_meta.json"
        if not meta_path.is_file():
            return jsonify({"error": "No finalization metadata found."}), 404
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return jsonify({"error": "Finalization metadata is unreadable."}), 404
        return jsonify(payload)

    @app.get("/api/projects/<project_name>/output/<path:filename>")
    def project_output_file(project_name: str, filename: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
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
            return jsonify({"available": False})
        stat = preview_path.stat()
        model_path = project_dir / "model.py"
        model_sha256 = (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.is_file()
            else None
        )
        return jsonify({
            "available": True,
            "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
            "model_sha256": model_sha256,
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

    @app.post("/api/projects/<project_name>/screenshot")
    def screenshot_capture(project_name: str):
        try:
            _project_path(app.config["SETTINGS"], project_name)
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

    # ------------------------------------------------------------------ #
    #  Revision history APIs
    # ------------------------------------------------------------------ #

    _MAX_DIFF_LINES = 500

    def _revision_summary(
        store: RevisionStore,
        revision,
        active_id: str | None,
        lkg_id: str | None,
        project_dir: Path,
    ) -> dict[str, Any]:
        build = None
        try:
            build = store.build_for(revision.id)
        except RevisionIntegrityError:
            pass
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
        if build is not None:
            summary["build_status"] = build.status
            if build.metrics:
                summary["metrics"] = build.metrics
            if build.error:
                summary["error"] = build.error
        # User acceptance state + open issues from the quality store (Phase 2).
        try:
            quality = QualityStore(project_dir)
            decision = quality.latest_decision_for_revision(revision.id)
            if decision is not None:
                summary["acceptance"] = {
                    "decision": decision.decision,
                    "comment": decision.comment,
                    "created_at": decision.created_at,
                    "attempt_id": decision.attempt_id,
                }
            summary["open_issues"] = sum(
                1 for issue in quality.issues_for_revision(revision.id) if issue.status == "open"
            )
        except (QualityError, QualityIntegrityError):
            pass
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
        next_before = None
        if revisions and store.list(limit=1, before=revisions[-1].id):
            next_before = revisions[-1].id
        return jsonify({
            "revisions": [
                _revision_summary(
                    store,
                    r,
                    active.id if active else None,
                    lkg.id if lkg else None,
                    project_dir,
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
            project_dir,
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
        against_id = request.args.get("against") or revision.parent_id
        if against_id is None:
            return jsonify({"diff": "", "truncated": False, "against": None})
        try:
            against = store.get(against_id)
        except ValueError as error:
            return jsonify({"error": str(error)}), 404
        except RevisionIntegrityError as error:
            # Retention can remove the parent manifest of the oldest retained
            # revision. Show that boundary revision as an addition from empty.
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
        try:
            current_settings = app.config["SETTINGS"]
            project_dir = _project_path(current_settings, project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            try:
                project_dir = _project_path(app.config["SETTINGS"], project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
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
                source = store.source(revision_id)
                ModelConstraintValidator(ConstraintStore(project_dir)).validate(source)
                revision = store.restore(revision_id)
            except (ConstraintError, ValueError, RevisionIntegrityError) as error:
                return jsonify({"error": str(error)}), 422

            bus.publish("revision_updated", {"project": project_name})

            bus.publish("agent_status", {
                "project": project_name,
                "status": "restoring",
                "message": f"Restoring revision {revision.id[:8]} and rebuilding…",
            })

            # Full sandboxed CAD run on the restored source.
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

            # Register and publish the new preview.
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

    # ------------------------------------------------------------------ #
    #  Constraint APIs
    # ------------------------------------------------------------------ #

    @app.get("/api/projects/<project_name>/constraints")
    def list_constraints(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            store = ConstraintStore(project_dir)
            try:
                constraints = store.list()
                model_path = project_dir / "model.py"
                targets = (
                    store.discover_targets(model_path.read_text(encoding="utf-8"))
                    if model_path.is_file()
                    else {"parameters": [], "features": []}
                )
            except (ConstraintError, OSError) as error:
                return jsonify({"error": str(error)}), 422
        return jsonify({
            "constraints": [constraint.to_dict() for constraint in constraints],
            "targets": targets,
        })

    @app.post("/api/projects/<project_name>/constraints")
    def create_constraint(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        payload = request.get_json(silent=True) or {}
        kind = str(payload.get("kind", ""))
        name = str(payload.get("name", "")).strip()
        if kind not in ("parameter", "source_feature"):
            return jsonify({"error": "Constraint kind must be 'parameter' or 'source_feature'."}), 400
        if not name:
            return jsonify({"error": "Constraint name is required."}), 400
        with _project_lock(app, project_name):
            runner = app.config["AGENT_RUNNER"]
            if runner.has_active_state_for(project_name):
                return jsonify({"error": "Cannot modify constraints while the agent is active."}), 409
            try:
                project_dir = _project_path(app.config["SETTINGS"], project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            model_path = project_dir / "model.py"
            if not model_path.is_file():
                return jsonify({"error": "model.py does not exist."}), 400
            source = model_path.read_text(encoding="utf-8")
            store = ConstraintStore(project_dir)
            try:
                if kind == "parameter":
                    constraint = store.create_parameter_constraint(name, source)
                else:
                    constraint = store.create_source_feature_constraint(name, source)
                store.add(constraint)
            except ConstraintError as error:
                return jsonify({"error": str(error)}), 422
            _append_conversation(project_dir, {
                "timestamp": _utc_now(),
                "type": "constraint_added",
                "data": {"kind": kind, "name": name},
            })
        bus.publish("constraint_added", {
            "project": project_name,
            "kind": kind,
            "name": name,
        })
        return jsonify({"constraint": constraint.to_dict()}), 201

    @app.delete("/api/projects/<project_name>/constraints/<constraint_id>")
    def delete_constraint(project_name: str, constraint_id: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        with _project_lock(app, project_name):
            runner = app.config["AGENT_RUNNER"]
            if runner.has_active_state_for(project_name):
                return jsonify({"error": "Cannot modify constraints while the agent is active."}), 409
            try:
                project_dir = _project_path(app.config["SETTINGS"], project_name)
            except (ValueError, FileNotFoundError) as error:
                return jsonify({"error": str(error)}), 404
            store = ConstraintStore(project_dir)
            try:
                removed = store.remove(constraint_id)
            except ConstraintError as error:
                return jsonify({"error": str(error)}), 422
            if removed is None:
                return jsonify({"error": "Constraint not found."}), 404
            _append_conversation(project_dir, {
                "timestamp": _utc_now(),
                "type": "constraint_removed",
                "data": {"kind": removed.kind, "name": removed.name},
            })
        bus.publish("constraint_removed", {
            "project": project_name,
            "kind": removed.kind,
            "name": removed.name,
        })
        return jsonify({"removed": constraint_id})

    @app.get("/api/projects/<project_name>/quality/runs")
    def quality_runs(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        try:
            limit = int(request.args.get("limit", 50))
        except (TypeError, ValueError):
            return jsonify({"error": "Run limit must be an integer."}), 400
        store = QualityStore(project_dir)
        before = request.args.get("before")
        try:
            runs = store.list_runs(limit=limit, before=before)
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 400
        next_before = None
        if runs:
            try:
                if store.list_runs(limit=1, before=runs[-1].run_id):
                    next_before = runs[-1].run_id
            except QualityError:
                next_before = None
        return jsonify({
            "runs": [run.to_dict() for run in runs],
            "next_before": next_before,
        })

    @app.get("/api/projects/<project_name>/quality/runs/<run_id>")
    def quality_run_detail(project_name: str, run_id: str):
        if not _valid_uuid(run_id):
            return jsonify({"error": "Invalid run id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        try:
            run = store.get_run(run_id)
            attempts = store.list_attempts(run_id)
            events = store.get_events(run_id)
            decisions = store.list_decisions(run_id)
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        return jsonify({
            "run": run.to_dict(),
            "attempts": [attempt.to_dict() for attempt in attempts],
            "events": events,
            "decisions": [decision.to_dict() for decision in decisions],
        })

    @app.get("/api/projects/<project_name>/quality/runs/<run_id>/events")
    def quality_run_events(project_name: str, run_id: str):
        if not _valid_uuid(run_id):
            return jsonify({"error": "Invalid run id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        try:
            limit = int(request.args.get("limit", 200))
        except (TypeError, ValueError):
            return jsonify({"error": "Event limit must be an integer."}), 400
        store = QualityStore(project_dir)
        try:
            events = store.get_events(run_id, limit=limit)
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        return jsonify({"events": events})

    @app.get("/api/projects/<project_name>/quality/spec")
    def quality_latest_spec(project_name: str):
        """Return the most recent DesignSpec for the active run (read-only)."""
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        try:
            run_id = request.args.get("run_id")
            spec = None
            if run_id and _valid_uuid(run_id):
                spec = store.latest_spec(run_id)
            if spec is None:
                # Fall back to the most recent run's spec.
                for run in store.list_runs(limit=20):
                    spec = store.latest_spec(run.run_id)
                    if spec is not None:
                        break
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        if spec is None:
            return jsonify({"spec": None, "requirements": []})
        return jsonify({"spec": spec.to_dict()})

    @app.get("/api/projects/<project_name>/quality/validations/<attempt_id>")
    def quality_attempt_validations(project_name: str, attempt_id: str):
        """Return every ValidationResult for one attempt."""
        if not _valid_uuid(attempt_id):
            return jsonify({"error": "Invalid attempt id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        try:
            results = store.validations_for_attempt(attempt_id)
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 404
        return jsonify(
            {"validations": [r.to_dict() for r in results]}
        )

    @app.get("/api/projects/<project_name>/quality/attempts/<attempt_id>")
    def quality_attempt_detail(project_name: str, attempt_id: str):
        if not _valid_uuid(attempt_id):
            return jsonify({"error": "Invalid attempt id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        # Reuse the shared resolver and the in-memory attempt→run index
        # (audit_019 + audit_018) to avoid walking every run on disk.
        found, error_response = _resolve_quality_attempt(store, attempt_id)
        if error_response is not None:
            return error_response
        run, attempt = found["run"], found["attempt"]
        return jsonify({"run": run.to_dict(), "attempt": attempt.to_dict()})

    def _resolve_quality_attempt(
        store: QualityStore, attempt_id: str
    ) -> tuple[dict[str, Any] | None, Any]:
        """Locate the run+attempt for an attempt id, or return an error response.

        Uses the in-memory attempt->run index so the lookup is O(1) instead of
        scanning every run manifest on disk (audit_019 + audit_018).
        """
        try:
            run_id = store._run_id_for_attempt(attempt_id)  # noqa: SLF001
            run = store.get_run(run_id)
            attempt = store.get_attempt(run_id, attempt_id)
        except QualityIntegrityError:
            return None, (jsonify({"error": "Attempt not found."}), 404)
        except QualityError as error:
            return None, (jsonify({"error": str(error)}), 400)
        return {"run": run, "attempt": attempt}, None

    def _require_current_attempt(
        store: QualityStore, found: dict[str, Any], project_name: str
    ) -> Any:
        """Ensure the attempt targets the current head revision (delivery gate)."""
        try:
            with _project_lock(app, project_name):
                revisions = RevisionStore(_project_path(app.config["SETTINGS"], project_name))
                revisions.reconcile()
                head = revisions.head()
        except (ValueError, RevisionIntegrityError) as error:
            return jsonify({"error": str(error)}), 422
        if head is None or head.id != found["attempt"].revision_id:
            return jsonify(
                {
                    "error": "This attempt belongs to an outdated revision. Refresh the preview before deciding.",
                    "code": "STALE_PREVIEW",
                }
            ), 409
        return None

    @app.post("/api/projects/<project_name>/quality/attempts/<attempt_id>/decision")
    def quality_record_decision(project_name: str, attempt_id: str):
        if not _valid_uuid(attempt_id):
            return jsonify({"error": "Invalid attempt id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        found, error_response = _resolve_quality_attempt(store, attempt_id)
        if error_response is not None:
            return error_response
        if found["attempt"].status != "succeeded":
            return jsonify({
                "error": "Only a successfully built design can be accepted or rejected.",
                "code": "ATTEMPT_NOT_SUCCEEDED",
            }), 409
        stale_response = _require_current_attempt(store, found, project_name)
        if stale_response is not None:
            return stale_response
        body = request.get_json(silent=True) or {}
        decision = body.get("decision")
        if not isinstance(decision, str) or decision not in DECISION_TYPES:
            return jsonify(
                {
                    "error": "decision must be one of: accepted, accepted_with_limitations, rejected."
                }
            ), 400
        categories = body.get("categories")
        categories = (
            [item for item in categories if isinstance(item, str)]
            if isinstance(categories, list)
            else []
        )
        comment = body.get("comment")
        camera = body.get("camera") if isinstance(body.get("camera"), dict) else None
        try:
            record = store.record_decision(
                found["run"].run_id,
                revision_id=found["attempt"].revision_id,
                attempt_id=attempt_id,
                decision=decision,
                categories=categories,
                comment=str(comment) if comment is not None else "",
                camera=camera,
            )
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 422
        accepted = decision in ACCEPTED_DECISION_TYPES
        bus.publish(
            "design_accepted" if accepted else "design_rejected",
            {
                "project": project_name,
                "run_id": found["run"].run_id,
                "attempt_id": attempt_id,
                "revision_id": found["attempt"].revision_id,
                "decision": decision,
            },
        )
        return jsonify(record.to_dict()), 201

    @app.post("/api/projects/<project_name>/quality/attempts/<attempt_id>/issues")
    def quality_create_issue(project_name: str, attempt_id: str):
        if not _valid_uuid(attempt_id):
            return jsonify({"error": "Invalid attempt id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        found, error_response = _resolve_quality_attempt(store, attempt_id)
        if error_response is not None:
            return error_response
        if found["attempt"].status != "succeeded":
            return jsonify({
                "error": "Issues can only be reported for a successfully built design.",
                "code": "ATTEMPT_NOT_SUCCEEDED",
            }), 409
        stale_response = _require_current_attempt(store, found, project_name)
        if stale_response is not None:
            return stale_response
        body = request.get_json(silent=True) or {}
        category = body.get("category")
        if not isinstance(category, str) or not category:
            return jsonify({"error": "category is required."}), 400
        severity = body.get("severity") or "blocking"
        if severity not in ISSUE_SEVERITIES:
            return jsonify({"error": "severity must be one of: blocking, major, minor."}), 400
        requirement_ids = body.get("requirement_ids")
        requirement_ids = (
            [str(item) for item in requirement_ids if isinstance(item, str)]
            if isinstance(requirement_ids, list)
            else []
        )
        message = body.get("message") or ""
        camera = body.get("camera") if isinstance(body.get("camera"), dict) else None
        from agent.quality.models import new_id

        issue_id = new_id()
        screenshot = body.get("screenshot")
        png: bytes | None = None
        if isinstance(screenshot, str) and screenshot:
            try:
                import base64

                png = base64.b64decode(screenshot, validate=True)
            except Exception:  # noqa: BLE001 - Malformed base64 from the browser.
                return jsonify({"error": "screenshot must be a base64 PNG."}), 400
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                return jsonify({"error": "screenshot must be a PNG image."}), 400
        issue: Issue | None = None
        evidence: list[str] = []
        try:
            # Create the issue first so a screenshot/save failure cannot leave
            # orphaned evidence pointing at a non-existent issue (audit_079).
            issue = store.create_issue(
                found["run"].run_id,
                attempt_id=attempt_id,
                revision_id=found["attempt"].revision_id,
                category=category,
                severity=severity,
                message=str(message),
                requirement_ids=requirement_ids,
                evidence=(),
                issue_id=issue_id,
            )
            if png is not None:
                evidence.append(
                    store.save_issue_evidence(
                        found["run"].run_id, attempt_id, issue_id, png
                    )
                )
            if evidence:
                issue = store.update_issue_evidence(
                    found["run"].run_id, issue_id, tuple(evidence)
                )
            # An explicit issue report marks the targeted revision rejected.
            store.record_decision(
                found["run"].run_id,
                revision_id=found["attempt"].revision_id,
                attempt_id=attempt_id,
                decision="rejected",
                categories=[category],
                comment=str(message)[:2000],
                camera=camera,
            )
        except (QualityError, QualityIntegrityError, QualityLimitError, RuntimeError, OSError) as error:
            # Roll back any persisted state so a retry cannot duplicate work
            # or leave a stale decision pointing at a missing issue (audit_079).
            if issue is not None:
                try:
                    store.delete_issue(found["run"].run_id, issue.issue_id)
                except Exception:  # noqa: BLE001 - best-effort rollback.
                    pass
            status = 413 if isinstance(error, QualityLimitError) else 422
            return jsonify({"error": str(error)}), status
        bus.publish(
            "design_rejected",
            {
                "project": project_name,
                "run_id": found["run"].run_id,
                "attempt_id": attempt_id,
                "revision_id": found["attempt"].revision_id,
                "issue_id": issue.issue_id,
            },
        )
        return jsonify({"issue": issue.to_dict()}), 201

    @app.get("/api/projects/<project_name>/quality/issues")
    def quality_issues(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        try:
            limit = int(request.args.get("limit", 200))
        except (TypeError, ValueError):
            return jsonify({"error": "Issue limit must be an integer."}), 400
        open_only = request.args.get("open") in {"1", "true"}
        revision_id = request.args.get("revision_id")
        store = QualityStore(project_dir)
        try:
            issues = store.list_issues(
                revision_id=revision_id,
                open_only=open_only,
                limit=limit,
            )
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"issues": [issue.to_dict() for issue in issues]})

    @app.post("/api/projects/<project_name>/quality/issues/<issue_id>/resolve")
    def quality_resolve_issue(project_name: str, issue_id: str):
        if not _valid_uuid(issue_id):
            return jsonify({"error": "Invalid issue id."}), 404
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        try:
            store = QualityStore(project_dir)
            issue = next(
                (item for item in store.list_issues(limit=500) if item.issue_id == issue_id),
                None,
            )
            head = RevisionStore(project_dir).head()
        except (QualityError, QualityIntegrityError, RevisionIntegrityError) as error:
            return jsonify({"error": str(error)}), 422
        if issue is None:
            return jsonify({"error": "Issue not found."}), 404
        if head is None:
            return jsonify({"error": "No current revision is available."}), 409
        if head.id == issue.revision_id or not store.has_successful_attempt_for_revision(
            head.id
        ):
            return jsonify({
                "error": "Resolve an issue only after a newer revision has a successful build.",
                "code": "RESOLUTION_REVIEW_REQUIRED",
            }), 409
        body = request.get_json(silent=True) or {}
        confirmed_by = str(body.get("confirmed_by") or "user")[:200]
        try:
            resolved = store.resolve_issue(
                issue.run_id,
                issue.issue_id,
                resolved_by_revision_id=head.id,
                confirmed_by=confirmed_by,
            )
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 422
        bus.publish("issue_resolved", {
            "project": project_name,
            "issue_id": resolved.issue_id,
            "revision_id": resolved.revision_id,
            "resolved_by_revision_id": head.id,
        })
        return jsonify({"issue": resolved.to_dict()})

    @app.get("/api/projects/<project_name>/quality/metrics")
    def quality_metrics(project_name: str):
        try:
            project_dir = _project_path(app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
        store = QualityStore(project_dir)
        try:
            return jsonify(store.get_metrics())
        except (QualityError, QualityIntegrityError) as error:
            return jsonify({"error": str(error)}), 400

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
