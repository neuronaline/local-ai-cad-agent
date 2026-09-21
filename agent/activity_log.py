"""Per-project raw activity log for agent runs.

Records the full tool-loop trace — LLM HTTP wire events, tool-call lifecycle,
subprocess I/O — to ``<project>/.cad-agent/activity.jsonl``. Designed to
support post-mortem debugging when a tool loop fails or returns an unexpected
verdict; complements the narrower ``debug-errors.jsonl`` (recoverable tool
failures only) with a complete ordered record of everything the agent did.

Activated by ``agent.log_tool_activity`` in ``config.yaml``. The log file is
append-only with periodic size-based trimming so long-running projects do
not exhaust disk.

Wire model:

* Every event is a single JSON line containing ``{ts, run_id, event, ...}``.
* All payloads pass through :func:`redact` before serialisation. API keys
  and inline image data URLs are always removed; argument/response metadata
  stays.
* Writes are guarded by a per-project lock so concurrent callers cannot
  interleave partial lines. The activity log is a separate file from the
  canonical ``conversation.jsonl``; the two do not need to share a lock.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent.io import atomic_write_text, utc_now_iso

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Default file size cap. The agent logs include LLM request bodies and
# full tool outputs; 5 MiB keeps roughly the last ~10k events of a typical
# build/review loop without slowing the agent. Adjustable via the
# ``AGENT_ACTIVITY_LOG_MAX_BYTES`` environment variable for one-off debugging.
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024

# Trimming cadence. Running a full read-rewrite on every ``log()`` call
# turned a hot loop into a multi-megabyte I/O storm that could pin the
# event-loop worker. Trim only once every ``_TRIM_EVERY`` appends (or
# sooner if a single oversized event would push us well past the cap).
_TRIM_EVERY = 100

# Keys whose value is always redacted regardless of context. Lowercased for
# case-insensitive matching. Covers the headers and provider-specific payload
# fields the agent never needs in a debug log.
_REDACTED_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "x-api-key",
        "api_key",
        "apikey",
        "openrouter_api_key",
        "openai_api_key",
        "openrouter-key",
        "openai-key",
        "openrouter_api_key_redacted",
        "password",
        "token",
        "secret",
        "access_token",
        "refresh_token",
        "session_token",
        "cookie",
        "set-cookie",
    }
)

# Sentinel substituted in place of redacted values.
_REDACTED_PLACEHOLDER = "[REDACTED]"

# File names. We always write a single rolling file per project; the agent
# does not need per-run splits because every entry carries ``run_id``.
_LOG_FILENAME = "activity.jsonl"
_LOG_DIRNAME = ".cad-agent"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class ActivityLogger:
    """Append-only JSONL logger with size-based trimming and redaction."""

    def __init__(
        self,
        project_dir: Path,
        *,
        max_bytes: int | None = None,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._max_bytes = max_bytes if max_bytes is not None else _DEFAULT_MAX_BYTES
        # Per-process lock so concurrent threads inside one worker do
        # not race. The cross-process guarantee comes from the
        # ``fcntl.flock`` on the sidecar lock file below — a
        # ``threading.Lock`` alone is invisible to other processes.
        self._lock = threading.Lock()
        # Increments on every ``log()`` so trimming happens on a fixed
        # cadence instead of every single append.
        self._writes_since_trim = 0
        # Sidecar advisory lock file. Created lazily; ``flock`` is a
        # POSIX/Linux primitive so this code path requires the same
        # platform assumption as bwrap/libseccomp (see README).
        self._lockfile: Path | None = None

    @property
    def log_path(self) -> Path:
        return self.project_dir / _LOG_DIRNAME / _LOG_FILENAME

    @property
    def lockfile_path(self) -> Path:
        """Path of the sidecar advisory lock used for cross-process safety."""
        return self.log_path.with_suffix(self.log_path.suffix + ".lock")

    @contextmanager
    def _acquire_file_lock(self) -> Iterator[None]:
        """Acquire an advisory ``flock`` on the sidecar lock file.

        Cross-process guarantee: requires POSIX ``fcntl``. Falls back to
        a thread-only guarantee (matching the previous behaviour) when
        ``flock`` is unavailable (e.g. non-POSIX runners).
        """
        path = self.lockfile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8")
        acquired = False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                acquired = True
            except (OSError, AttributeError):
                # Either flock is unavailable or fcntl itself is missing.
                pass
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()

    def log(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Append a single event line. Best-effort; never raises."""
        try:
            entry = {
                "ts": utc_now_iso(),
                "run_id": run_id or "",
                "event": event,
            }
            if payload:
                entry["data"] = redact(payload)
            line = json.dumps(entry, ensure_ascii=False)
            log_dir = self.log_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            # Acquire the cross-process lock first so concurrent workers
            # (CLI + server, multiple gevent spawns, etc.) cannot
            # interleave partial lines.
            with self._lock, self._acquire_file_lock():
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")
                self._writes_since_trim += 1
                # Cheap size probe — a single stat() is enough to detect
                # when trimming is overdue. The actual rewrite happens
                # at most every ``_TRIM_EVERY`` writes (or sooner when
                # the cap is dramatically exceeded by a single event).
                #
                # ``force=True`` when one append alone blew past
                # ``2 * max_bytes`` so the cadence never lets the log
                # grow unboundedly when an unusually large event lands.
                try:
                    size = self.log_path.stat().st_size
                except OSError:
                    size = 0
                force_trim = size > 2 * self._max_bytes
                self._maybe_trim(force=force_trim)
        except (OSError, TypeError, ValueError) as error:
            # Activity logging must never break the agent loop. Surface
            # the failure at DEBUG so persistent issues (e.g. disk full
            # or a non-writable project dir) are visible to operators
            # running with debug logging without spamming the default
            # log level for transient races.
            _LOG.debug(
                "activity log write failed: %s", error, exc_info=True
            )
            return

    def clear(self) -> bool:
        """Remove the log file. Returns True if anything was removed."""
        try:
            self.log_path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    # ------------------------------------------------------------------ internals

    def _maybe_trim(self, *, force: bool = False) -> None:
        """Drop oldest lines when the log exceeds ``max_bytes``.

        Trimming is gated by ``_TRIM_EVERY`` so the previous behaviour of
        a full read-rewrite on every ``log()`` call does not pin the
        worker. ``force=True`` is reserved for the oversized-event case
        (when one append alone blew past ``2 * max_bytes``) so we still
        recover instead of waiting up to ``_TRIM_EVERY`` more appends.
        """
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return
        if size <= self._max_bytes:
            return
        if not force and self._writes_since_trim < _TRIM_EVERY:
            return
        try:
            text = self.log_path.read_text(encoding="utf-8")
        except OSError:
            return
        lines = text.splitlines()
        # Keep roughly the second half — coarse trim keeps subsequent
        # operations cheap and preserves the most recent context.
        keep_from = max(1, len(lines) // 2)
        kept = lines[keep_from:]
        if not kept:
            return
        # Write atomically via a sibling temp so an interrupted trim cannot
        # corrupt the canonical log. The trim is best-effort; only OSError
        # is swallowed because that is the failure mode a size cap is
        # designed to recover from (interrupted trim -> cap exceeded ->
        # next call trims again).
        try:
            atomic_write_text(self.log_path, "\n".join(kept) + "\n")
            self._writes_since_trim = 0
        except OSError:
            return



# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact(value: Any) -> Any:
    """Return a deep-copied payload with secrets and image blobs replaced.

    * Authorization-style keys (any depth) are replaced with a placeholder.
    * ``image_url.url`` and ``data:`` payloads are replaced with a short
      placeholder noting the original size in bytes so a debugger can still
      tell how big the dropped content was.
    * All other values pass through untouched.

    The function never raises so it can be used as a final write filter.
    """
    return _redact_value(value)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _REDACTED_KEYS:
                result[key] = _REDACTED_PLACEHOLDER
            elif _looks_like_image_url_part(key, item):
                result[key] = _redact_image_url_part(key, item)
            else:
                result[key] = _redact_value(item)
        return result
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _looks_like_image_url_part(key: Any, value: Any) -> bool:
    """Heuristic: dict-like OpenAI image_url part with a data: URL."""
    if not isinstance(value, dict):
        return False
    if key != "image_url" and "image_url" not in value:
        return False
    url = value.get("url") if "url" in value else None
    if not isinstance(url, str):
        return False
    return url.startswith("data:")


def _redact_image_url_part(key: Any, value: Any) -> Any:
    """Replace a base64 image_url payload with a size-only placeholder."""
    if not isinstance(value, dict):
        return value
    cloned = deepcopy(value)
    url = cloned.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        comma = url.find(",")
        if comma != -1:
            size = len(url) - (comma + 1)
            cloned["url"] = f"[IMAGE_DATA_URL redacted, {size} base64 chars]"
        else:
            cloned["url"] = "[IMAGE_DATA_URL redacted]"
    return cloned


# ---------------------------------------------------------------------------
# Helpers used by the rest of the agent
# ---------------------------------------------------------------------------


def is_enabled(settings: Any) -> bool:
    """Return True when the agent has activity logging turned on."""
    return bool(getattr(settings, "agent_log_tool_activity", False))


def get_logger(project_dir: Path) -> ActivityLogger:
    """Return a fresh per-project logger.

    Each call returns a new instance with its own append-only file handle
    and per-project write lock. Callers may hold a single instance across
    runs; this helper exists so the runner can wire one logger without
    caring about prior state.
    """
    return ActivityLogger(project_dir)