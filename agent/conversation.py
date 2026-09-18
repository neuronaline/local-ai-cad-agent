"""Conversation persistence extracted from ``agent.core.AgentRunner``.

The store owns the canonical ``conversation.jsonl`` file for a project:
loading (with a FIFO cache) and appending. Splitting it out keeps
``AgentRunner`` focused on the tool-calling lifecycle without dragging
in JSONL serialisation rules.
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

_history_lock: threading.RLock = threading.RLock()


def shared_history_lock() -> threading.RLock:
    """Return the shared history write lock (replaceable from the host).

    ``RLock`` lets a thread that already holds the lock (e.g. the
    ``EventBus`` thread mid-append) re-enter it when calling ``load`` to
    populate the cache. A plain ``Lock`` would deadlock in that case.
    """
    return _history_lock


def set_shared_history_lock(lock: threading.RLock) -> None:
    """Replace the module-level history lock (used by ``app.py`` at startup).

    The Flask ``EventBus`` owns its own ``RLock`` so SSE status writes and
    ``ConversationStore`` appends serialize against one another. The slot
    indirection keeps the agent package free of a Flask import.
    """
    global _history_lock
    _history_lock = lock


class ConversationStore:
    """JSONL-backed conversation log with a FIFO in-memory cache.

    The cache is keyed by the project's absolute path. Cap is small enough
    that a project create/delete cycle on a long-lived process cannot grow
    the cache unbounded; the same cap also makes the test fixture
    deterministic. Multimodal ``image_url`` parts are preserved verbatim
    on every load so the prompt prefix stays byte-stable across turns —
    prompt-cache continuity depends on the cached bytes matching the
    bytes the provider hashed.
    """

    #: Bounded number of cached projects. Older entries evict FIFO when a
    #: new project reads/writes through the store.
    CACHE_MAX: ClassVar[int] = 32

    #: Maximum entries returned by :meth:`load` to keep the LLM context
    #: window bounded. None disables truncation so full conversation history
    #: is permanently preserved in memory without prompt prefix invalidation.
    MAX_HISTORY: ClassVar[int | None] = None

    #: Roles kept when truncating the log on load.
    _KEPT_ROLES: ClassVar[set[str]] = {"user", "assistant", "tool"}

    _cache: ClassVar[dict[str, list[dict[str, Any]]]] = {}

    @classmethod
    def invalidate(cls, project_dir: Path) -> None:
        """Forget the cached entry for ``project_dir`` (used after writes
        and on project deletion).

        **Caller MUST hold :func:`shared_history_lock`.** Performing this
        mutation outside the lock lets a concurrent :meth:`load` refill
        the cache from a pre-write snapshot and re-cache the now-stale
        data after this method returns — a thread can subsequently observe
        the line appended by another writer *forever*. When the lock
        contract is respected the invalidate and the next file write
        happen as a single critical section, so no reader can refill
        stale data between the two steps.
        """
        cls._cache.pop(str(project_dir), None)

    @classmethod
    def load(cls, project_dir: Path) -> list[dict[str, Any]]:
        """Load the conversation history for ``project_dir``.

        Multimodal ``image_url`` parts are returned verbatim so the
        prompt prefix stays byte-stable across turns. Returning the same
        bytes the provider hashed keeps prompt caches warm.
        """
        cache_key = str(project_dir)
        # A cache hit short-circuits the file read. The cache is invalidated
        # by ``append``/``clear`` and the cache fill is guarded by the lock
        # below, so hits are consistent with the latest persisted state.
        cached = cls._cache.get(cache_key)
        if cached is not None:
            if not project_dir.exists():
                # Drop the stale entry under the lock so a concurrent
                # append cannot race with the pop and silently re-cache
                # stale data afterwards.
                with shared_history_lock():
                    cls._cache.pop(cache_key, None)
                return []
            return [deepcopy(item) for item in cached]
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
            cls._set_cached(cache_key, [deepcopy(item) for item in history])
        return [deepcopy(item) for item in history]

    @classmethod
    def append(cls, project_dir: Path, message: dict[str, Any]) -> None:
        """Persist a single message and invalidate the cache.

        The invalidate, the directory check, and the file write all happen
        inside :func:`shared_history_lock` so a concurrent :meth:`load`
        cannot refill the cache from a pre-write snapshot and silently
        overwrite the cache after this method returned — the exact race
        that used to drop appended lines from every subsequent read.
        """
        with shared_history_lock():
            cls.invalidate(project_dir)
            log_path = project_dir / "conversation.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

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
        status persistence share one lock and one plain ``open("a")``
        handle. The previous implementation cached append-mode file
        descriptors across the whole process, which leaked resources
        on Windows / POSIX and made cross-thread invalidation brittle.

        The cache invalidate and the file write all run inside
        :func:`shared_history_lock` to keep the race condition
        documented on :meth:`invalidate` from reappearing here.
        """
        record = {
            "timestamp": timestamp or cls._now_iso(),
            "type": event_type,
            "data": data,
        }
        with shared_history_lock():
            cls.invalidate(project_dir)
            log_path = project_dir / "conversation.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def clear(cls, project_dir: Path) -> bool:
        """Remove the log file and cache entry. Returns whether anything
        was removed.

        The cache invalidate and the unlink both run inside
        :func:`shared_history_lock` so a concurrent append or load
        cannot race with the unlink and resurrect a half-deleted file.
        """
        log_path = project_dir / "conversation.jsonl"
        removed = False
        with shared_history_lock():
            cls.invalidate(project_dir)
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
            else:
                removed = True
        return removed

    @classmethod
    def _set_cached(cls, key: str, history: list[dict[str, Any]]) -> None:
        """Store ``history`` under ``key`` in the read cache, evicting the
        oldest entry first when the cache is full.

        **Caller MUST hold :func:`shared_history_lock`.** The eviction
        loop iterates ``_cache`` and pops during iteration, which raises
        ``RuntimeError: dictionary changed size during iteration`` if a
        concurrent append invalidates an entry between ``len(...)`` and
        the next ``pop``.
        """
        while len(cls._cache) >= cls.CACHE_MAX:
            oldest = next(iter(cls._cache))
            if oldest == key:
                break
            cls._cache.pop(oldest, None)
        cls._cache[key] = list(history)
