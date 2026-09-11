"""Conversation persistence extracted from ``agent.core.AgentRunner``.

The store owns the canonical ``conversation.jsonl`` file for a project:
loading (with image redaction + FIFO cache + bounded history), appending,
and clearing. Splitting it out keeps ``AgentRunner`` focused on the
tool-calling lifecycle without dragging in JSONL serialisation rules.
"""
from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path
from typing import Any, ClassVar

_history_lock_slot: list[threading.RLock] = [threading.RLock()]


def shared_history_lock() -> threading.RLock:
    """Return the shared history write lock (replaceable from the host).

    ``RLock`` lets a thread that already holds the lock (e.g. the
    ``EventBus`` thread mid-append) re-enter it when calling ``load`` to
    populate the cache. A plain ``Lock`` would deadlock in that case.
    """
    return _history_lock_slot[0]


# Reclaim the cached file descriptors on interpreter shutdown so the OS
# doesn't report them as leaked when the process exits cleanly. Registered
# at import time so gunicorn / gevent worker reimports each install a
# single closer for this worker's handles.
@atexit.register
def _close_log_handles_on_exit() -> None:
    ConversationStore.close_handles()


class ConversationStore:
    """JSONL-backed conversation log with a FIFO in-memory cache.

    The cache is keyed by the project's absolute path. Cap is small enough
    that a project create/delete cycle on a long-lived process cannot grow
    the cache unbounded; the same cap also makes the test fixture
    deterministic. Inline images are redacted to ``[image: ...]``
    placeholders on load so the LLM context never re-injects image bytes
    from past turns.
    """

    #: Bounded number of cached projects. Older entries evict FIFO when a
    #: new project reads/writes through the store.
    CACHE_MAX: ClassVar[int] = 32

    #: Maximum entries returned by :meth:`load` to keep the LLM context
    #: window bounded.
    MAX_HISTORY: ClassVar[int] = 100

    #: Roles kept when truncating the log on load.
    _KEPT_ROLES: ClassVar[set[str]] = {"user", "assistant", "tool"}

    _cache: ClassVar[dict[str, list[dict[str, Any]]]] = {}

    #: Per-project open file handles kept in append mode. Reusing a handle
    #: avoids repeated ``open(2)``/``close(2)`` syscalls on every message
    #: (hundreds of writes per agent turn). Each entry is closed by
    #: :meth:`close_handles` on process exit via :mod:`atexit` and on
    #: :meth:`clear`. The cap mirrors :attr:`CACHE_MAX` so the cache and
    #: the handle table cannot drift apart on a long-lived process.
    _open_log_handles: ClassVar[dict[str, Any]] = {}

    @classmethod
    def invalidate(cls, project_dir: Path) -> None:
        """Forget the cached entry for ``project_dir`` (used after writes
        and on project deletion)."""
        cls._cache.pop(str(project_dir), None)

    @classmethod
    def load(cls, project_dir: Path) -> list[dict[str, Any]]:
        """Load the truncated, image-redacted history for ``project_dir``."""
        cache_key = str(project_dir)
        # A cache hit short-circuits the file read. The cache is invalidated
        # by ``append``/``clear`` and the cache fill is guarded by the lock
        # below, so hits are consistent with the latest persisted state.
        cached = cls._cache.get(cache_key)
        if cached is not None:
            if not project_dir.exists():
                cls._cache.pop(cache_key, None)
            else:
                return list(cached)
        # Serialise the read+cache-set against appends so a concurrent writer
        # cannot have its line missed by this load (or, conversely, so a
        # partially-written tail cannot be observed). ``RLock`` allows the
        # current thread to re-enter if it already holds the lock (e.g. the
        # EventBus thread populating the cache after an append).
        with shared_history_lock():
            history: list[dict[str, Any]] = []
            log_path = project_dir / "conversation.jsonl"
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(item, dict)
                        and item.get("role") in cls._KEPT_ROLES
                    ):
                        history.append(item)
            history = [
                cls._strip_image_parts(item, project_dir) for item in history
            ]
            history = cls._truncate(history)
            cls._set_cached(cache_key, history)
        return history

    @classmethod
    def append(cls, project_dir: Path, message: dict[str, Any]) -> None:
        """Persist a single message and invalidate the cache."""
        cls.invalidate(project_dir)
        log = cls._get_log_handle(project_dir)
        with shared_history_lock():
            log.write(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            log.flush()

    @classmethod
    def append_event(
        cls,
        project_dir: Path,
        event_type: str,
        data: dict[str, Any],
        *,
        timestamp: str | None = None,
    ) -> None:
        """Persist an ``{timestamp, type, data}`` envelope event.

        Centralises the ``conversation.jsonl`` writer so the SSE history
        filters (``_load_history`` / ``_KEPT_ROLES``) and the EventBus
        status persistence share a single lock and a single lazily-opened
        append handle. Previously the two writers — ``ConversationStore.append``
        and ``app._append_conversation`` — shared the lock by reference but
        each opened its own file handle with an incompatible payload schema,
        which made a future refactor that swapped the lock slot or split the
        file silently break either the LLM context or the SSE history view.
        """
        cls.invalidate(project_dir)
        record = {
            "timestamp": timestamp or cls._now_iso(),
            "type": event_type,
            "data": data,
        }
        log = cls._get_log_handle(project_dir)
        with shared_history_lock():
            log.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            log.flush()

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _get_log_handle(cls, project_dir: Path) -> Any:
        """Return a cached append-mode handle for ``project_dir``.

        On miss the handle is opened lazily and added to
        :attr:`_open_log_handles`. The cache is FIFO-evicted at
        :attr:`CACHE_MAX` to mirror the read-side cache, and stale entries
        are dropped if the project directory has been removed underneath us.
        Concurrent appenders are serialised by :func:`shared_history_lock`
        in :meth:`append`, so a single shared handle is sufficient.
        """
        key = str(project_dir)
        log = cls._open_log_handles.get(key)
        if log is not None:
            return log
        # Close the oldest entries when the handle cache is full so a
        # create/delete cycle on a long-lived process cannot leak FDs.
        while len(cls._open_log_handles) >= cls.CACHE_MAX:
            oldest_key = next(iter(cls._open_log_handles))
            oldest_log = cls._open_log_handles.pop(oldest_key, None)
            if oldest_log is not None:
                try:
                    oldest_log.close()
                except Exception:
                    pass
        project_dir.mkdir(parents=True, exist_ok=True)
        log = (project_dir / "conversation.jsonl").open("a", encoding="utf-8")
        cls._open_log_handles[key] = log
        return log

    @classmethod
    def close_handles(cls) -> None:
        """Close every cached log handle. Safe to call repeatedly."""
        while cls._open_log_handles:
            _key, log = cls._open_log_handles.popitem()
            try:
                log.close()
            except Exception:
                pass

    @classmethod
    def clear(cls, project_dir: Path) -> bool:
        """Remove the log file and cache entry. Returns whether anything
        was removed."""
        cls.invalidate(project_dir)
        log_path = project_dir / "conversation.jsonl"
        # Drop any cached handle so the unlink below cannot race with a
        # buffered writer holding the inode open on Windows / POSIX.
        cached = cls._open_log_handles.pop(str(project_dir), None)
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
        removed = False
        with shared_history_lock():
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
            else:
                removed = True
        return removed

    @classmethod
    def _set_cached(cls, key: str, history: list[dict[str, Any]]) -> None:
        while len(cls._cache) >= cls.CACHE_MAX:
            oldest = next(iter(cls._cache))
            if oldest == key:
                break
            cls._cache.pop(oldest, None)
        cls._cache[key] = list(history)

    @staticmethod
    def _truncate(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim ``history`` to ``MAX_HISTORY`` entries at a coherent boundary.

        OpenAI tool-call messages can interleave ``assistant.tool_calls``
        arrays with their ``tool`` results across turns. A raw ``[-N:]``
        cut can leave the window open in two ways:

        1. The head is a ``tool`` message whose supplying assistant turn
           has been dropped (orphan tool result).
        2. The head is an assistant ``tool_calls`` turn whose first one or
           more results have been dropped (orphan assistant turn).

        Both shapes confuse downstream LLM clients that expect every
        ``tool_call_id`` referenced by an assistant message to have a
        matching ``tool`` response in the same window. This helper walks
        forward from the cut point until the remaining window is
        self-contained: leading ``tool`` messages are dropped until the
        first ``user`` or ``assistant`` message appears, and a leading
        ``assistant`` turn is dropped together with its trailing tool
        results if any of its ``tool_call_id`` references are missing.
        """
        if len(history) <= ConversationStore.MAX_HISTORY:
            return history
        truncated = history[-ConversationStore.MAX_HISTORY:]
        # Drop orphan tool results until we hit a non-tool message.
        while truncated and truncated[0].get("role") == "tool":
            truncated.pop(0)
        # Drop an orphan assistant turn whose tool results were truncated
        # away. The assistant's ``tool_calls`` are paired with subsequent
        # ``role == "tool"`` messages in order; if any reference is missing
        # the head assistant must be discarded along with every tool
        # message we already kept after it.
        while truncated and truncated[0].get("role") == "assistant":
            expected_ids = _assistant_tool_call_ids(truncated[0])
            if not expected_ids:
                break
            supplied_ids = {
                item.get("tool_call_id")
                for item in truncated[1:]
                if item.get("role") == "tool"
            }
            if expected_ids.issubset(supplied_ids):
                break
            truncated.pop(0)
            while truncated and truncated[0].get("role") == "tool":
                truncated.pop(0)
        return truncated

    @staticmethod
    def _strip_image_parts(
        item: dict[str, Any], project_dir: Path | None = None
    ) -> dict[str, Any]:
        """Replace inline image parts with placeholders when loading history.

        The user-role scan matches the original behaviour (user reference
        images become ``[image: <name>]``). Tool-role messages produced by
        ``cad_build_and_verify`` may also carry inline image content; redact
        those too so the agent never re-injects a base64 payload from a
        previous turn (history would otherwise balloon turn-over-turn).

        Tool entries whose ``tool_call_id`` is registered in
        ``.agent_tool_images.json`` (written by ``_remember_inline_tool_images``
        in :mod:`agent.dispatcher`) carry the documented
        ``[Inline render from <relative path>]`` placeholder naming the
        evidence file. ``[Inline render N]`` remains the fallback when no
        index entry is available.
        """
        content = item.get("content")
        role = item.get("role")
        if role not in {"user", "tool"} or not isinstance(content, list):
            return item
        image_index = (
            _load_image_index(item.get("tool_call_id"), project_dir)
            if role == "tool"
            else None
        )
        parts: list[object] = []
        replaced = False
        placeholder_index = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                replaced = True
                placeholder_index += 1
                if role == "user":
                    name = part.get("filename") or f"reference-{placeholder_index}"
                    parts.append({"type": "text", "text": f"[image: {name}]"})
                else:
                    parts.append(
                        {"type": "text", "text": _inline_placeholder(image_index, placeholder_index)}
                    )
            else:
                parts.append(part)
        if not replaced:
            return item
        cleaned = dict(item)
        cleaned["content"] = parts
        return cleaned


def _inline_placeholder(
    image_paths: list[str] | None, ordinal: int
) -> str:
    """Render the documented placeholder for an inline tool image.

    ``image_paths`` is the list of host-relative paths registered against
    the supplying tool call. When exactly one path is registered the
    placeholder names it directly; otherwise the ordinal-based
    ``[Inline render N]`` fallback is used so the model's context remains
    stable regardless of how many renders the previous turn shipped.
    """
    if image_paths and len(image_paths) == 1:
        return f"[Inline render from {image_paths[0]}]"
    return f"[Inline render {ordinal}]"


def _assistant_tool_call_ids(message: dict[str, Any]) -> set[str]:
    """Return the set of ``tool_call_id`` strings referenced by ``message``.

    Returns an empty set when ``message`` is not an assistant turn with a
    ``tool_calls`` array, or when its entries are not shaped as expected.
    The helper is intentionally tolerant of malformed history entries:
    truncation must not raise on a corrupt cache line.
    """
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return set()
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return set()
    ids: set[str] = set()
    for entry in calls:
        if isinstance(entry, dict):
            value = entry.get("id")
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def _load_image_index(
    tool_call_id: object, project_dir: Path | None
) -> list[str] | None:
    """Read the per-call image index from ``.agent_tool_images.json``.

    The index is written by ``_remember_inline_tool_images`` in
    :mod:`agent.dispatcher`. Loading it from the placeholder step keeps the
    data flow self-contained: the index is read-only history metadata that
    survives even if the dispatcher's helper is moved. Returns the
    registered path list for ``tool_call_id`` or ``None`` when no entry
    matches.
    """
    if not isinstance(tool_call_id, str) or not tool_call_id or project_dir is None:
        return None
    index_path = project_dir / ".agent_tool_images.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    entry = payload.get(tool_call_id)
    if not isinstance(entry, list):
        return None
    paths = [value for value in entry if isinstance(value, str)]
    return paths or None
