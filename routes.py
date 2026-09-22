"""HTTP route blueprints for the local AI CAD agent.

Five Flask blueprints mount the project's HTTP surface: ``projects``,
``agent``, ``render``, ``review``, ``revisions``. ``app.create_app`` is
a pure orchestrator that wires them up alongside the global hooks;
unit tests can mount the slices they need at ``Blueprint`` granularity.
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
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from agent.converter import (
    EXPORT_MIME_TYPES,
    SUPPORTED_EXPORT_FORMATS,
    export_project_model,
)
from agent.core import AgentRunner
from agent.images import store_images
from agent.io import utc_now_iso
from agent.review_paths import review_dir
from agent.revisions import (
    MODEL_FILENAME,
    RevisionIntegrityError,
    RevisionStore,
    cached_model_sha256,
)
from agent.settings import Settings

PROJECT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
INFO_EVENT_TYPES = frozenset({"agent_status", "tool_status", "agent_usage", "agent_stopped"})
HISTORY_EVENT_TYPES = INFO_EVENT_TYPES | {"agent_error"}
SSE_QUEUE_SIZE = 512
_MAX_DIFF_LINES = 500


# ---------------------------------------------------------------------------
# Cross-cutting helpers (module-level, app-context agnostic where possible).
# ---------------------------------------------------------------------------


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


def _project_path(settings: Settings, project_name: str) -> Path:
    if not PROJECT_NAME_RE.fullmatch(project_name):
        raise ValueError(
            "Project name must use lowercase letters, numbers, and hyphens (no uppercase or spaces)."
        )
    path = settings.workspace_root / project_name
    if not path.is_dir():
        raise FileNotFoundError("Project not found.")
    return path


def _resolve_project_or_404(
    settings: Settings, project_name: str
) -> Path | tuple[Response, int]:
    """Return the project :class:`Path` or a Flask ``(body, 404)`` response.

    Callers should run this *inside* the project lock so a project that
    is concurrently deleted cannot leave the handler operating on a
    dangling path. The helper packages the resolve+404 dance into one
    call so each handler — previously repeating the try/except outside
    *and* inside the lock — only inspects the result once. The shape is::

        project_dir = _resolve_project_or_404(settings, project_name)
        if not isinstance(project_dir, Path):
            return project_dir
    """
    try:
        return _project_path(settings, project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404


def _redact_history_event(event: dict[str, Any]) -> dict[str, Any]:
    """Replace inline image data URLs in history responses with a placeholder.

    The persisted ``conversation.jsonl`` stores the full base64 payload on
    user-image attachments so the LLM can still consume it. Returning those
    blobs through the History endpoint makes responses unnecessarily large
    and exposes content the UI does not need; substitute a lightweight
    ``[Reference image N]`` marker instead. Tool-role messages produced by
    ``cad_build_and_verify`` may carry inline render evidence; redact those
    with the same shield so the History view never echoes base64.
    """
    role = event.get("role")
    if role not in {"user", "tool"}:
        return event
    content = event.get("content")
    if not isinstance(content, list):
        return event
    redacted = False
    parts: list[Any] = []
    image_index = 0
    placeholder = "[Reference image {n}]" if role == "user" else "[Inline render {n}]"
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            redacted = True
            image_index += 1
            parts.append(
                {"type": "text", "text": placeholder.format(n=image_index)}
            )
        else:
            parts.append(part)
    if not redacted:
        return event
    cleaned = dict(event)
    cleaned["content"] = parts
    return cleaned


# ---------------------------------------------------------------------------
# Idempotency cache: TTL-based with LRU hard-cap fallback.
# ---------------------------------------------------------------------------

_IDEMPOTENCY_LOCK = threading.Lock()
# Per-entry lifetime. Beyond this window the next read drops the key so
# a stale entry cannot accidentally de-dupe a fresh request. The hard
# cap bounds a burst of equal-timestamp inserts; the previous count-
# slicing policy dropped 500 in-flight keys during a burst, exposing
# endpoints to replay attacks.
_IDEMPOTENCY_CACHE_TTL_SECONDS = 600.0
_IDEMPOTENCY_CACHE_HARD_CAP = 1000


def _idempotency_check(key: str) -> bool:
    """Return True if the key is in-flight and still fresh.

    Expired entries are dropped on read so the cache stays small
    without waiting for the next write to find the staleness.
    """
    if not key:
        return False
    with _IDEMPOTENCY_LOCK:
        cache = current_app.config.setdefault("IDEMPOTENCY_CACHE", {})
        timestamp = cache.get(key)
        if timestamp is None:
            return False
        if _idempotency_expired(timestamp):
            cache.pop(key, None)
            return False
        return True


def _idempotency_record(key: str) -> None:
    """Record the key as in-flight; sweep expired entries first.

    If the TTL sweep leaves the cache over :data:`_IDEMPOTENCY_CACHE_HARD_CAP`,
    drop the oldest entries (by timestamp, not insertion order) until the
    cap is reached.
    """
    if not key:
        return
    now = time.monotonic()
    with _IDEMPOTENCY_LOCK:
        cache = current_app.config.setdefault("IDEMPOTENCY_CACHE", {})
        for old_key in [k for k, ts in cache.items() if _idempotency_expired(ts)]:
            cache.pop(old_key, None)
        cache[key] = now
        if len(cache) > _IDEMPOTENCY_CACHE_HARD_CAP:
            by_age = sorted(cache.items(), key=lambda kv: kv[1])
            for old_key, _ in by_age[: len(cache) - _IDEMPOTENCY_CACHE_HARD_CAP]:
                cache.pop(old_key, None)


def _idempotency_forget(key: str) -> None:
    """Drop a key so a retry can be accepted."""
    if not key:
        return
    with _IDEMPOTENCY_LOCK:
        current_app.config.setdefault("IDEMPOTENCY_CACHE", {}).pop(key, None)


def _idempotency_expired(timestamp: float) -> bool:
    """Return True if ``timestamp`` is older than the TTL window."""
    return (time.monotonic() - timestamp) > _IDEMPOTENCY_CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Project / metadata helpers.
# ---------------------------------------------------------------------------


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
    from agent.sandbox import _BWRAP, seccomp_filter_fd

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
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        checks["python_packages"] = True
    except ImportError:
        checks["python_packages"] = False
    checks["openscad_installed"] = shutil.which("openscad") is not None
    return checks


def _project_lock(project_name: str) -> threading.Lock:
    locks: dict[str, threading.Lock] = current_app.config["PROJECT_LOCKS"]
    lock = locks.get(project_name)
    if lock is None:
        with current_app.config["PROJECT_LOCKS_LOCK"]:
            lock = locks.get(project_name)
            if lock is None:
                lock = threading.Lock()
                locks[project_name] = lock
    return lock


@contextmanager
def _project_locks(*project_names: str):
    locks = [_project_lock(name) for name in sorted(set(project_names))]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


_PROJECT_MTIME_CANDIDATES = (
    MODEL_FILENAME,
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


def _review_for_model(project_dir: Path, model_sha256: str) -> Path | None:
    """Return the review directory matching ``model_sha256``, or ``None``.

    Review directories are content-addressed by ``model_sha256``; this
    binding replaces the older mtime-based "latest" lookup so cache
    hits, restores, and file touches cannot accidentally rebind a
    verdict to a different revision.
    """
    if not model_sha256:
        return None
    review_path = review_dir(project_dir, model_sha256)
    if not review_path.is_dir():
        return None
    return review_path


def _revision_summary(
    store: RevisionStore,
    revision: Any,
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


# ---------------------------------------------------------------------------
# Blueprints.
# ---------------------------------------------------------------------------


projects_bp = Blueprint("projects", __name__)
agent_bp = Blueprint("agent", __name__)
render_bp = Blueprint("render", __name__)
review_bp = Blueprint("review", __name__)
revisions_bp = Blueprint("revisions", __name__)


# ---------------------------------------------------------------------------
# Project lifecycle routes.
# ---------------------------------------------------------------------------


@projects_bp.get("/")
def index() -> str:
    return render_template("projects.html")


@projects_bp.get("/api/preflight")
def preflight():
    return jsonify(_run_preflight(current_app.config["SETTINGS"]))


@projects_bp.get("/project/<name>")
def project_view(name: str) -> str:
    try:
        _project_path(current_app.config["SETTINGS"], name)
    except (ValueError, FileNotFoundError):
        return "Project not found.", 404
    return render_template(
        "index.html",
        project_name=name,
        show_info_messages=current_app.config["SETTINGS"].show_info_messages,
        viewer_grid_size=current_app.config["SETTINGS"].viewer_grid_size,
        viewer_grid_divisions=current_app.config["SETTINGS"].viewer_grid_divisions,
    )


@projects_bp.get("/api/projects")
def list_projects():
    settings: Settings = current_app.config["SETTINGS"]
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


@projects_bp.post("/api/projects/new")
def new_project():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip().lower().replace(" ", "-")
    if not PROJECT_NAME_RE.fullmatch(name):
        return jsonify({"error": "Use 1-63 lowercase letters, numbers, or hyphens."}), 400
    settings: Settings = current_app.config["SETTINGS"]
    workspace_root = settings.workspace_root
    project_dir = workspace_root / name
    with _project_lock(name):
        if project_dir.exists():
            return jsonify({"error": "Project already exists."}), 409
        with tempfile.TemporaryDirectory(prefix=".new-project-", dir=workspace_root) as temporary:
            staging = Path(temporary)
            (staging / "inputs").mkdir()
            _write_project_metadata(staging, {"name": name, "created_at": utc_now_iso()})
            staging.rename(project_dir)
    current_app.config["EVENT_BUS"].publish("project_created", {"project": name})
    return jsonify({"project": name}), 201


@projects_bp.delete("/api/projects/<project_name>")
def delete_project(project_name: str):
    settings: Settings = current_app.config["SETTINGS"]
    with _project_lock(project_name):
        project_dir = _resolve_project_or_404(settings, project_name)
        if not isinstance(project_dir, Path):
            return project_dir
        runner: AgentRunner = current_app.config["AGENT_RUNNER"]
        if runner.has_active_state_for(project_name):
            runner.stop(project_name)
        if runner.has_active_state_for(project_name):
            return jsonify({"error": "Cannot delete a project with active agent state."}), 409
        shutil.rmtree(project_dir)
    return jsonify({"deleted": True})


@projects_bp.post("/api/projects/<project_name>/reset")
def reset_project(project_name: str):
    """Clear the agent's conversation memory without touching the model.

    Removes ``conversation.jsonl`` so the next chat turn starts a fresh
    context. ``model.scad``, ``preview.stl``, ``render.png``, revisions, and
    review artifacts are left intact. Refuses to run while an agent task
    or a pending preview is in-flight so a reset cannot race the worker.
    """
    settings: Settings = current_app.config["SETTINGS"]
    with _project_lock(project_name):
        project_dir = _resolve_project_or_404(settings, project_name)
        if not isinstance(project_dir, Path):
            return project_dir
        runner: AgentRunner = current_app.config["AGENT_RUNNER"]
        if runner.has_active_state_for(project_name):
            runner.stop(project_name)
        if runner.has_active_state_for(project_name):
            return jsonify(
                {"error": "Cannot reset a project with active agent state."}
            ), 409
        if runner.is_running():
            return jsonify(
                {"error": "Cannot reset while an agent task is running."}
            ), 409
        removed = runner.clear_history(project_dir)
        (project_dir / ".agent_state.json").unlink(missing_ok=True)
        (project_dir / ".agent_initial_state.json").unlink(missing_ok=True)
    current_app.config["EVENT_BUS"].publish(
        "conversation_reset",
        {"project": project_name, "removed": removed},
    )
    return jsonify({"reset": True, "removed": removed})


@projects_bp.put("/api/projects/<project_name>/rename")
def rename_project(project_name: str):
    settings: Settings = current_app.config["SETTINGS"]
    payload = request.get_json(silent=True) or {}
    new_name = str(payload.get("name", "")).strip().lower().replace(" ", "-")
    if not PROJECT_NAME_RE.fullmatch(new_name):
        return jsonify({"error": "Use 1-63 lowercase letters, numbers, or hyphens."}), 400
    if new_name == project_name:
        return jsonify({"project": project_name})
    target = settings.workspace_root / new_name
    with _project_locks(project_name, new_name):
        project_dir = _resolve_project_or_404(settings, project_name)
        if not isinstance(project_dir, Path):
            return project_dir
        if target.exists():
            return jsonify({"error": "A project with that name already exists."}), 409
        runner: AgentRunner = current_app.config["AGENT_RUNNER"]
        if runner.has_active_state_for(project_name):
            runner.stop(project_name)
        if runner.has_active_state_for(project_name):
            return jsonify({"error": "Cannot rename a project with active agent state."}), 409
        metadata = _read_project_metadata(project_dir)
        metadata["name"] = new_name
        metadata["created_at"] = metadata.get("created_at") or utc_now_iso()
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


# ---------------------------------------------------------------------------
# Agent / chat lifecycle routes.
# ---------------------------------------------------------------------------


@agent_bp.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    project_name = str(payload.get("project", ""))
    message = str(payload.get("message", "")).strip()
    idempotency_key = str(payload.get("idempotency_key", "")).strip()
    if not message:
        return jsonify({"error": "Message is required."}), 400
    runner: AgentRunner = current_app.config["AGENT_RUNNER"]
    if runner.waiting_question(project_name):
        return jsonify({"error": "Answer the pending question before sending another message."}), 409
    if idempotency_key and _idempotency_check(idempotency_key):
        if runner.is_running():
            return jsonify(
                {
                    "accepted": True,
                    "duplicate": True,
                    "error": "Your message is already being processed.",
                }
            ), 202
        _idempotency_forget(idempotency_key)
    if runner.is_running():
        return jsonify({"error": "An agent task is already running."}), 409
    with _project_lock(project_name):
        project_dir = _resolve_project_or_404(
            current_app.config["SETTINGS"], project_name
        )
        if not isinstance(project_dir, Path):
            return project_dir
        try:
            image_paths = store_images(request.files.getlist("attachments"), project_dir)
        except FileNotFoundError as error:
            return jsonify({"error": str(error)}), 404
        except (ValueError, OSError) as error:
            return jsonify({"error": str(error)}), 400
        # ``AgentRunner.start`` takes transactional ownership of the
        # uploaded inputs by moving them into a run-scoped directory under
        # its own lock. Cleanup happens in ``_run``'s ``finally`` block,
        # so the HTTP handler must no longer speculatively unlink here —
        # the runner is the sole owner of the artifacts from the moment
        # ``start`` returns.
        started = runner.start(project_name, message, image_paths)
    if not started:
        return jsonify({"error": "Unable to start the agent task."}), 409
    if idempotency_key:
        _idempotency_record(idempotency_key)
    return jsonify({"accepted": True, "attachments": [
        path.relative_to(project_dir).as_posix() for path in image_paths
    ]}), 202


@agent_bp.post("/api/questions/answer")
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
        _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    with _project_lock(project_name):
        accepted = current_app.config["AGENT_RUNNER"].answer(project_name, raw_answer)
    if not accepted:
        if current_app.config["AGENT_RUNNER"].waiting_question(project_name):
            return jsonify({"error": "The answer does not match the requested input type or options."}), 400
        return jsonify({"error": "No question is awaiting an answer for this project."}), 409
    return jsonify({"accepted": True}), 202


@agent_bp.post("/api/stop")
def stop():
    payload = request.get_json(silent=True) if request.is_json else request.form
    project_name = str((payload or {}).get("project", "")) or None
    if project_name:
        try:
            _project_path(current_app.config["SETTINGS"], project_name)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 404
    runner: AgentRunner = current_app.config["AGENT_RUNNER"]
    affected = runner.stop(project_name)
    event_project = project_name or runner.active_project()
    if event_project is None and affected:
        event_project = affected[0]
    current_app.config["EVENT_BUS"].publish(
        "agent_stopped",
        {"project": event_project, "affected_projects": affected},
    )
    return jsonify({"stopped": True, "affected_projects": affected})


@agent_bp.get("/api/projects/<project_name>/state")
def project_state(project_name: str):
    try:
        _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    runner: AgentRunner = current_app.config["AGENT_RUNNER"]
    question = runner.waiting_question(project_name)
    if question:
        return jsonify({"status": "waiting_for_user", "question": question})
    if runner.is_running() and runner.active_project() == project_name:
        return jsonify({"status": "running"})
    return jsonify({"status": "idle"})


@agent_bp.get("/api/projects/<project_name>/history")
def project_history(project_name: str):
    try:
        settings: Settings = current_app.config["SETTINGS"]
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
                if event.get("synthetic"):
                    continue
                events.append(_redact_history_event(event))
            except json.JSONDecodeError:
                pass
    return jsonify({"events": events})


@agent_bp.get("/api/stream")
def stream():
    bus = current_app.config["EVENT_BUS"]
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


# ---------------------------------------------------------------------------
# Render / preview / export routes.
# ---------------------------------------------------------------------------


@render_bp.get("/api/projects/<project_name>/render")
def project_render(project_name: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    render_path = project_dir / "render.png"
    if not render_path.is_file() or render_path.stat().st_size == 0:
        return jsonify({"error": "No render has been generated."}), 404
    try:
        build = json.loads(
            (project_dir / ".cad_metrics.json").read_text(encoding="utf-8")
        )
        manifest = build["review_manifest"]
        expected_sha = manifest["single_render"]["image_sha256"]
        current_model_sha = cached_model_sha256(project_dir)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return jsonify({"error": "No render exists for the current model."}), 404
    if (
        build.get("model_sha256") != current_model_sha
        or manifest.get("model_sha256") != current_model_sha
        or hashlib.sha256(render_path.read_bytes()).hexdigest() != expected_sha
    ):
        return jsonify({"error": "No render exists for the current model."}), 404
    return send_file(render_path, mimetype="image/png", max_age=0)


@render_bp.get("/api/projects/<project_name>/preview")
def preview(project_name: str):
    try:
        preview_path = _project_path(current_app.config["SETTINGS"], project_name) / "preview.stl"
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    if not preview_path.is_file():
        return jsonify({"error": "No preview has been generated."}), 404
    if preview_path.stat().st_size == 0:
        return jsonify({"error": "The generated preview is empty."}), 422
    return send_file(preview_path, mimetype="model/stl", max_age=0)


@render_bp.get("/api/projects/<project_name>/preview/meta")
def preview_meta(project_name: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
        preview_path = project_dir / "preview.stl"
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        return jsonify({"available": False, "displayable": False})
    stat = preview_path.stat()
    model_sha256 = cached_model_sha256(project_dir)
    # The ``review_status`` field is a backward-compatible stub. The
    # dedicated ``cad_review`` tool was removed from the model-facing
    # schema, so no structured verdict (``result.json``) is ever produced
    # for new projects. We keep the field so the existing frontend
    # (``hideUnapprovedPreview``) can keep reading it without breaking,
    # but it always resolves to ``"not_required"`` because there is no
    # reviewer gate in front of the preview.
    review_status = "not_required"
    return jsonify({
        "available": True,
        "displayable": True,
        "revision": f"{stat.st_mtime_ns}-{stat.st_size}",
        "model_sha256": model_sha256,
        "review_status": review_status,
    })


@render_bp.get("/api/projects/<project_name>/export")
def export_model(project_name: str):
    fmt = (request.args.get("format") or "stl").lower().strip()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_EXPORT_FORMATS))
        return jsonify({"error": f"Unsupported format '{fmt}'. Supported: {supported}"}), 400
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404

    try:
        exported_path = export_project_model(project_dir, fmt)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except ValueError as error:
        return jsonify({"error": str(error)}), 422
    except (RuntimeError, OSError) as error:
        return jsonify({"error": str(error)}), 500

    mimetype = EXPORT_MIME_TYPES.get(fmt, "application/octet-stream")
    download_name = f"{project_name}.{fmt}"
    return send_file(
        exported_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_name,
        max_age=0,
    )


# ---------------------------------------------------------------------------
# Multi-view review routes (read-only views into .cad-agent/reviews/).
# ---------------------------------------------------------------------------


_REVIEW_MANIFEST_NAME = "manifest.json"
_REVIEW_VIEWS_SUBDIR = "views"
_REVIEW_SHEET_NAME = "review-sheet.png"


@review_bp.get("/api/projects/<project_name>/review/manifest")
def review_manifest(project_name: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    model_path = project_dir / MODEL_FILENAME
    if not model_path.is_file():
        return jsonify({"error": f"{MODEL_FILENAME} is missing."}), 404
    model_sha = cached_model_sha256(project_dir)
    latest = _review_for_model(project_dir, model_sha)
    if latest is None:
        return jsonify({"error": "No review has been generated yet."}), 404
    manifest_path = latest / _REVIEW_MANIFEST_NAME
    if not manifest_path.is_file():
        return jsonify({"error": "Review manifest is missing."}), 404
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return jsonify({"error": "Review manifest is corrupted."}), 500
    payload = dict(payload)
    payload.setdefault("artifact_dir", latest.name)
    return jsonify(payload)


@review_bp.get("/api/projects/<project_name>/review/sheet")
def review_sheet(project_name: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    model_path = project_dir / MODEL_FILENAME
    if not model_path.is_file():
        return jsonify({"error": f"{MODEL_FILENAME} is missing."}), 404
    model_sha = cached_model_sha256(project_dir)
    latest = _review_for_model(project_dir, model_sha)
    if latest is None:
        return jsonify({"error": "No review has been generated yet."}), 404
    sheet_path = latest / _REVIEW_SHEET_NAME
    if not sheet_path.is_file() or sheet_path.stat().st_size == 0:
        return jsonify({"error": "Review contact sheet is missing."}), 404
    return send_file(sheet_path, mimetype="image/png", max_age=0)


@review_bp.get("/api/projects/<project_name>/review/view/<view_id>")
def review_view(project_name: str, view_id: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404
    # ``view_id`` is a filesystem identifier; restrict to the documented
    # canonical names to avoid path traversal via the URL.
    if not view_id or "/" in view_id or "\\" in view_id or view_id.startswith("."):
        return jsonify({"error": "Unknown review view."}), 404
    model_path = project_dir / MODEL_FILENAME
    if not model_path.is_file():
        return jsonify({"error": f"{MODEL_FILENAME} is missing."}), 404
    model_sha = cached_model_sha256(project_dir)
    latest = _review_for_model(project_dir, model_sha)
    if latest is None:
        return jsonify({"error": "No review has been generated yet."}), 404
    view_path = latest / _REVIEW_VIEWS_SUBDIR / f"{view_id}.png"
    if not view_path.is_file() or view_path.stat().st_size == 0:
        return jsonify({"error": "Unknown review view."}), 404
    return send_file(view_path, mimetype="image/png", max_age=0)


# ---------------------------------------------------------------------------
# Revision history routes.
# ---------------------------------------------------------------------------


@revisions_bp.get("/api/projects/<project_name>/revisions")
def list_revisions(project_name: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
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


@revisions_bp.get("/api/projects/<project_name>/revisions/<revision_id>")
def revision_detail(project_name: str, revision_id: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
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


@revisions_bp.get("/api/projects/<project_name>/revisions/<revision_id>/diff")
def revision_diff(project_name: str, revision_id: str):
    try:
        project_dir = _project_path(current_app.config["SETTINGS"], project_name)
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
    # revision's parent_id.
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


@revisions_bp.post("/api/projects/<project_name>/revisions/<revision_id>/restore")
def restore_revision(project_name: str, revision_id: str):
    settings: Settings = current_app.config["SETTINGS"]
    with _project_lock(project_name):
        project_dir = _resolve_project_or_404(settings, project_name)
        if not isinstance(project_dir, Path):
            return project_dir
        runner: AgentRunner = current_app.config["AGENT_RUNNER"]
        if runner.has_active_state_for(project_name):
            runner.stop(project_name)
        if runner.has_active_state_for(project_name):
            return jsonify({"error": "Cannot restore while the agent is active."}), 409
        store = RevisionStore(
            project_dir,
            retention_count=settings.revision_retention_count,
        )
        try:
            store.reconcile()
            if not store.has_source(revision_id):
                return jsonify({"error": "Revision source is missing or corrupt."}), 422
            revision = store.restore(revision_id)
        except (ValueError, RevisionIntegrityError) as error:
            return jsonify({"error": str(error)}), 422

        bus = current_app.config["EVENT_BUS"]
        bus.publish("revision_updated", {"project": project_name})
        bus.publish("agent_status", {
            "project": project_name,
            "status": "restoring",
            "message": f"Restoring revision {revision.id[:8]} and rebuilding…",
        })

        from agent.tools.cad_tool import CadTool, RenderMode
        cad = CadTool(project_dir, bus.publish, store)
        try:
            build = cad.build_and_verify(mode=RenderMode.FULL_REVIEW)
            metrics = build.get("metrics") or {}
        except (RuntimeError, ValueError, TypeError) as error:
            bus.publish("agent_error", {
                "project": project_name,
                "message": f"Restore succeeded but CAD rebuild failed: {error}",
            })
            # 207 Multi-Status signals a partial-success: ``model.scad`` was
            # restored, but the rebuild that re-derives ``preview.stl`` /
            # ``render.png`` failed. The explicit ``ok: false`` plus a
            # top-level ``error`` field lets clients detect the failure
            # without inspecting ``build_status``; the 207 keeps the
            # surrounding 2xx semantic that the restore half succeeded.
            return jsonify({
                "ok": False,
                "restored": True,
                "revision_id": revision.id,
                "build_status": "failed",
                "error": str(error),
            }), 207

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
            "ok": True,
            "restored": True,
            "revision_id": revision.id,
            "build_status": "succeeded",
            "metrics": metrics,
            "preview_id": preview_id,
        })