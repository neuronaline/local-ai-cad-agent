"""Local AI CAD Agent web server.

A small Flask app that hosts a chat UI, a Three.js preview viewer, and
the project/revision APIs the agent loop needs.

The HTTP route surface lives in :mod:`routes` as five Flask blueprints
(projects / agent / render / review / revisions). ``create_app`` is a
pure orchestrator: it wires configuration, mounts the blueprints,
attaches the global ``before_request`` (cross-origin guard) /
``after_request`` (security headers) / error-handler hooks, and
returns the assembled :class:`flask.Flask` instance. No route
definitions live here — that lets each blueprint be unit-tested at
``Blueprint`` granularity without standing up the entire application
container.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from agent.conversation import ConversationStore, set_shared_history_lock
from agent.core import AgentRunner
from agent.io import utc_now_iso
from agent.settings import Settings, load_settings

# Re-export the project name pattern, blueprint registry, and the SSE
# queue size so existing callers (``tests`` modules, ``run.sh``) that
# import these symbols from ``app`` keep working. The runtime helpers
# themselves now live in :mod:`routes`.
from routes import (  # noqa: F401  (re-export)
    HISTORY_EVENT_TYPES,
    INFO_EVENT_TYPES,
    PROJECT_NAME_RE,
    SSE_QUEUE_SIZE,
    agent_bp,
    projects_bp,
    render_bp,
    review_bp,
    revisions_bp,
)

# Cross-origin bind classification. The ``before_request`` guard (still
# defined inline here because it must run before any blueprint route)
# needs these sets to decide which mutation requests require an
# explicit ``Origin`` header.
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::", ""})
_LOCAL_BINDS = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class EventBus:
    """In-process fan-out for the single local browser client(s)."""

    def __init__(self, settings: Settings, app: Flask | None = None) -> None:
        self.settings = settings
        self.app = app
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._lock = threading.Lock()
        self._history_lock = threading.RLock()

    def _workspace_root(self) -> Path:
        if self.app is not None:
            try:
                current = self.app.config.get("SETTINGS")
                if current is not None:
                    return current.workspace_root
            except Exception:  # noqa: BLE001, S110 - fall back to startup settings
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
                    # Route through the central conversation writer so the
                    # EventBus status path and ``ConversationStore.append``
                    # share one lock and one append handle. Routing through
                    # a single writer also removes the file-handle schema
                    # drift risk where LLM context and SSE history could
                    # silently fall out of sync.
                    ConversationStore.append_event(
                        project_dir, event_type, data, timestamp=utc_now_iso()
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


def _active_api_key_env(settings: Settings) -> str:
    """Resolve the env var name that carries the active provider's API key."""
    from agent.llm_base import api_key_env
    return api_key_env(settings.llm_provider)


def create_app(settings: Settings | None = None) -> Flask:
    """Build and return a fully wired :class:`flask.Flask` instance.

    Acts as the orchestrator: configures the app, mounts the route
    blueprints from :mod:`routes`, and attaches the global
    ``before_request`` / ``after_request`` / error-handler hooks. No
    route handlers live here; that lets blueprints be tested in
    isolation and keeps this function's responsibility narrow.
    """
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    settings = settings or load_settings()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 52 * 1024 * 1024
    app.config["SETTINGS"] = settings
    bus = EventBus(settings, app)
    app.config["EVENT_BUS"] = bus
    set_shared_history_lock(bus._history_lock)
    app.config["AGENT_RUNNER"] = AgentRunner(settings, bus.publish)
    app.config["PROJECT_LOCKS"]: dict[str, threading.Lock] = {}
    app.config["PROJECT_LOCKS_LOCK"] = threading.Lock()
    app.config["IDEMPOTENCY_CACHE"]: dict[str, float] = {}

    # Mount route blueprints. The order is cosmetic (Flask dispatches by
    # URL pattern, not registration order) but groups the lifecycle
    # blueprint first so unrelated blueprints have access to project
    # locks via the shared ``current_app`` config.
    app.register_blueprint(projects_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(render_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(revisions_bp)

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

    @app.errorhandler(HTTPException)
    def http_exception(error):
        # Werkzeug HTTPException subclasses carry their own status code and
        # description (404 for missing project, 405 for wrong method, etc.).
        # Returning JSON instead of the default HTML page keeps the API
        # contract consistent for the bundled UI/JS, which expects every
        # response — including errors — to be JSON-parseable.
        response = jsonify(
            {
                "error": error.description or error.name,
                "status": error.code,
            }
        )
        return response, error.code or 500

    @app.errorhandler(Exception)
    def unhandled_exception(error):
        # Any other exception (KeyError, AttributeError, lock acquisition
        # failure, serialization bug, …) is rendered as JSON so the
        # JavaScript client never sees an HTML 500 page. The generic body
        # hides internals from the user; the full traceback is logged
        # server-side via Flask's logger for the operator.
        app.logger.exception("Unhandled exception in Flask route")
        return jsonify(
            {
                "error": "Internal server error.",
                "type": type(error).__name__,
            }
        ), 500

    return app


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    current_settings = load_settings()
    create_app(current_settings).run(host=current_settings.host, port=current_settings.port, debug=False, threaded=True)
