"""Per-project raw activity log for agent runs.

Records the full tool-loop trace — LLM HTTP wire events, tool-call lifecycle,
subprocess I/O — to ``<project>/.cad-agent/activity.jsonl``. Designed to
support post-mortem debugging when a tool loop fails or returns an unexpected
verdict; complements the narrower ``debug-errors.jsonl`` (recoverable tool
failures only) with a complete ordered record of everything the agent did.

Activated by ``agent.log_mode`` (or legacy ``agent.log_tool_activity``) in
``config.yaml``. Supports rolling debug logging and isolated dataset trajectories.
The log file is append-only with periodic size-based trimming so long-running
projects do not exhaust disk.

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
import queue
import re
import threading
import time
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
        "x-goog-api-key",
        "api_key",
        "apikey",
        "openrouter_api_key",
        "openai_api_key",
        "gemini_api_key",
        "google_api_key",
        "openrouter-key",
        "openai-key",
        "gemini-key",
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
        log_mode: str = "debug",
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.log_mode = log_mode
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
        # Non-blocking async queue for background writes
        self._queue: queue.Queue[tuple[str, str, dict[str, Any] | None, str] | None] = (
            queue.Queue(maxsize=10000)
        )
        self._closed = False
        if self.log_mode != "off":
            self._worker_thread: threading.Thread | None = threading.Thread(
                target=self._worker_loop, daemon=True, name="ActivityLoggerWorker"
            )
            self._worker_thread.start()
        else:
            self._worker_thread = None

    @property
    def log_path(self) -> Path:
        return self.project_dir / _LOG_DIRNAME / _LOG_FILENAME

    @property
    def trajectories_dir(self) -> Path:
        return self.project_dir / _LOG_DIRNAME / "trajectories"

    def trajectory_path(self, run_id: str) -> Path:
        clean_id = Path(re.sub(r"[^\w\-.]", "_", run_id or "")).name
        if not clean_id or clean_id in {".", ".."}:
            clean_id = "unknown"
        return self.trajectories_dir / f"{clean_id}.jsonl"

    @property
    def lockfile_path(self) -> Path:
        """Path of the sidecar advisory lock used for cross-process safety."""
        return self.log_path.with_suffix(self.log_path.suffix + ".lock")

    @contextmanager
    def _acquire_file_lock(self) -> Iterator[None]:
        """Acquire an advisory ``flock`` on the sidecar lock.

        Cross-process guarantee: requires POSIX ``fcntl``. Falls back to
        a thread-only guarantee (matching the previous behaviour) when
        ``flock`` is unavailable (e.g. non-POSIX runners).

        The lock file is opened in append mode (``O_APPEND`` /
        ``"a"``). Opening with ``"w"`` would truncate the file at
        zero bytes, which would silently discard the inode another
        process already has ``flock``'d — ``fcntl`` locks are bound to
        the underlying file (inode + file descriptor), not the path, so
        truncation does not steal an existing lock and instead causes
        subsequent acquisitions on the same path to deadlock against
        a phantom inode the holder no longer references.
        """
        path = self.lockfile_path
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        acquired = False
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                acquired = True
            except (OSError, AttributeError):
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
        sync: bool = False,
    ) -> None:
        """Append a single event line asynchronously (or synchronously when sync=True). Best-effort; never raises."""
        if self._closed or self.log_mode == "off":
            return
        ts = utc_now_iso()
        safe_payload = deepcopy(payload) if payload is not None else None
        if sync:
            self._write_event(event, safe_payload, run_id=run_id or "", ts=ts)
            return
        try:
            self._queue.put_nowait((event, run_id or "", safe_payload, ts))
        except queue.Full:
            self._write_event(event, safe_payload, run_id=run_id or "", ts=ts)
        except Exception as error:  # noqa: BLE001
            _LOG.debug("activity log queue failed: %s", error, exc_info=True)

    def flush(self, timeout: float | None = 5.0) -> None:
        """Wait until all pending entries are written to disk."""
        if self._closed or self.log_mode == "off":
            return
        if timeout is None:
            try:
                self._queue.join()
            except Exception:  # noqa: BLE001, S110
                pass
            return
        end_time = time.monotonic() + timeout
        try:
            with self._queue.all_tasks_done:
                while self._queue.unfinished_tasks:
                    remaining = end_time - time.monotonic()
                    if remaining <= 0:
                        break
                    self._queue.all_tasks_done.wait(timeout=remaining)
        except Exception as error:  # noqa: BLE001
            _LOG.debug("activity log flush error: %s", error, exc_info=True)

    def close(self, timeout: float = 5.0) -> None:
        """Flush remaining events and stop the worker thread."""
        if self._closed:
            return
        self._closed = True
        if self.log_mode == "off":
            return
        try:
            self.flush(timeout=timeout)
            try:
                self._queue.put(None, timeout=max(0.1, timeout))
            except (queue.Full, Exception):  # noqa: BLE001, S110
                pass
            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
        except Exception as error:  # noqa: BLE001
            _LOG.debug("activity log close error: %s", error, exc_info=True)

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                event, run_id, payload, ts = item
                self._write_event(event, payload, run_id=run_id, ts=ts)
            except Exception as error:  # noqa: BLE001
                _LOG.debug("activity log worker error: %s", error, exc_info=True)
            finally:
                self._queue.task_done()

    def _write_event(
        self,
        event: str,
        payload: dict[str, Any] | None,
        *,
        run_id: str,
        ts: str,
    ) -> None:
        try:
            entry = {
                "ts": ts,
                "run_id": run_id,
                "event": event,
            }
            if payload:
                entry["data"] = redact(payload)
            line = json.dumps(entry, ensure_ascii=False) + "\n"

            # 1. Rolling activity.jsonl write
            log_dir = self.log_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            with self._lock, self._acquire_file_lock():
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                self._writes_since_trim += 1
                try:
                    size = self.log_path.stat().st_size
                except OSError:
                    size = 0
                force_trim = size > 2 * self._max_bytes
                self._maybe_trim(force=force_trim)

            # 2. Per-run isolated trajectory (untrimmed for dataset extraction)
            if self.log_mode == "dataset" and run_id:
                traj_path = self.trajectory_path(run_id)
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                with traj_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except (OSError, TypeError, ValueError) as error:
            _LOG.debug("activity log write failed: %s", error, exc_info=True)

    def clear(self) -> bool:
        """Remove the log file. Returns True if anything was removed."""
        self.flush()
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
            elif _looks_like_inline_data_part(key, item):
                result[key] = _redact_inline_data_part(key, item)
            else:
                result[key] = _redact_value(item)
        return result
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _looks_like_image_url_part(key: Any, value: Any) -> bool:
    """Heuristic: dict-like OpenAI image_url part or local image payload."""
    if not isinstance(value, dict):
        return False
    if key != "image_url" and "image_url" not in value and "image_path" not in value:
        return False
    url = value.get("url") if "url" in value else None
    if isinstance(url, str):
        return url.startswith("data:")
    return bool(value.get("image_path") or value.get("path") or value.get("name"))


def _looks_like_inline_data_part(key: Any, value: Any) -> bool:
    """Heuristic: dict-like Gemini inline_data payload with base64 data."""
    if not isinstance(value, dict):
        return False
    if key == "inline_data" or "inline_data" in value:
        return True
    return "mime_type" in value and "data" in value and isinstance(value.get("data"), str)


def _redact_image_url_part(key: Any, value: Any) -> Any:
    """Replace a base64 image_url payload with a size placeholder, preserving local path if present."""
    if not isinstance(value, dict):
        return value
    cloned = deepcopy(value)
    url = cloned.get("url")
    image_path = cloned.get("image_path") or cloned.get("path") or cloned.get("name")
    if isinstance(url, str) and url.startswith("data:"):
        comma = url.find(",")
        if comma != -1:
            size = len(url) - (comma + 1)
            if image_path:
                cloned["url"] = f"[IMAGE_DATA_URL redacted, {size} chars, path={image_path}]"
            else:
                cloned["url"] = f"[IMAGE_DATA_URL redacted, {size} base64 chars]"
        else:
            cloned["url"] = "[IMAGE_DATA_URL redacted]"
    elif image_path:
        cloned["url"] = f"[LOCAL_IMAGE path={image_path}]"
    return cloned


def _redact_inline_data_part(key: Any, value: Any) -> Any:
    """Replace a base64 inline_data payload with a size placeholder."""
    if not isinstance(value, dict):
        return value
    cloned = deepcopy(value)
    if "inline_data" in cloned and isinstance(cloned["inline_data"], dict):
        inner = deepcopy(cloned["inline_data"])
        data = inner.get("data")
        if isinstance(data, str):
            inner["data"] = f"[BASE64_DATA redacted, {len(data)} chars]"
        cloned["inline_data"] = inner
    elif "data" in cloned and isinstance(cloned.get("data"), str):
        cloned["data"] = f"[BASE64_DATA redacted, {len(cloned['data'])} chars]"
    return cloned


# ---------------------------------------------------------------------------
# LLM payload sanitization
# ---------------------------------------------------------------------------

# Truncated text preview attached to redacted ``messages`` entries.
# 200 chars keeps the preview short enough that a single line cannot
# exhaust the rolling 5 MiB cap while still preserving a useful
# operator-readable sample of the prompt.
_LLM_MESSAGE_TEXT_PREVIEW_CHARS = 200

# Conservative credential patterns used by the preview guard. Any text
# matching any of these is suppressed so the activity log never echoes
# a raw secret even when the LLM conversation does. Suppression is the
# safe default — a missing preview is a nuisance; a leaked credential
# is a security incident.
_LLM_PREVIEW_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-or-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"(?i)\bapi[_-]?key\b\s*[:=]"),
    re.compile(r"(?i)\bpassword\b\s*[:=]"),
    re.compile(r"(?i)\bsecret\b\s*[:=]"),
    re.compile(r"(?i)\bauthorization\b\s*:\s*bearer\s+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
)


def _preview_text_is_safe(text: str) -> bool:
    """Return True when ``text`` does not match any credential pattern."""
    return not any(pattern.search(text) for pattern in _LLM_PREVIEW_SECRET_PATTERNS)


def summarize_llm_messages(
    messages: list[Any],
    *,
    include_preview: bool = True,
    lossless: bool = False,
) -> list[dict[str, Any]]:
    """Return a structural or lossless description of an LLM request body.

    In lossless mode (used when ``log_mode == "dataset"``), full message
    content, reasoning, and tool calls are preserved for fine-tuning extraction,
    with credential and image data filtering applied via :func:`redact`.
    In summary mode (``lossless=False``), text is capped at 200 chars.
    """
    if not isinstance(messages, list):
        return []
    if lossless:
        return [redact(msg) if isinstance(msg, dict) else msg for msg in messages]
    summary: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            summary.append(
                {"role": None, "content_bytes": 0, "image_count": 0}
            )
            continue
        role = message.get("role") or message.get("type")
        content = message.get("content")
        image_count = 0
        text_chars = 0
        text_preview_source: str | None = None
        if role == "function_call":
            fn_name = message.get("name") or ""
            raw_args = message.get("arguments") or message.get("args") or {}
            arg_str = json.dumps(raw_args) if isinstance(raw_args, dict) else str(raw_args)
            text_chars = len(arg_str.encode("utf-8"))
            text_preview_source = f"{fn_name}({arg_str[:60]}...)" if len(arg_str) > 60 else f"{fn_name}({arg_str})"
        elif role == "function_result":
            fn_name = message.get("name") or ""
            res = message.get("result")
            res_str = json.dumps(res) if isinstance(res, dict) else str(res)
            text_chars = len(res_str.encode("utf-8"))
            text_preview_source = f"{fn_name} -> {res_str[:60]}..." if len(res_str) > 60 else f"{fn_name} -> {res_str}"
        elif isinstance(content, str):
            text_chars = len(content.encode("utf-8"))
            text_preview_source = content
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        text_chars += len(text_value.encode("utf-8"))
                        if text_preview_source is None:
                            text_preview_source = text_value
                elif part_type in ("image", "image_url"):
                    image_count += 1
        elif content is None:
            text_chars = 0
        else:
            text_chars = len(str(content).encode("utf-8"))
        entry: dict[str, Any] = {
            "role": role,
            "content_bytes": text_chars,
            "image_count": image_count,
        }
        if (
            include_preview
            and text_preview_source is not None
            and text_preview_source
            and _preview_text_is_safe(text_preview_source)
        ):
            truncated = text_preview_source[:_LLM_MESSAGE_TEXT_PREVIEW_CHARS]
            if len(text_preview_source) > _LLM_MESSAGE_TEXT_PREVIEW_CHARS:
                truncated += "..."
            entry["text_preview"] = truncated
        summary.append(entry)
    return summary


def get_logger(project_dir: Path, *, log_mode: str = "debug") -> ActivityLogger:
    """Return a fresh per-project logger.

    Each call returns a new instance with its own append-only file
    handle and per-project write lock.
    """
    return ActivityLogger(project_dir, log_mode=log_mode)


# ---------------------------------------------------------------------------
# Helpers used by the rest of the agent
# ---------------------------------------------------------------------------


def is_enabled(settings: Any) -> bool:
    """Return True when the agent has activity logging turned on."""
    mode = getattr(settings, "agent_log_mode", None)
    if mode is not None:
        return mode != "off"
    return bool(getattr(settings, "agent_log_tool_activity", False))