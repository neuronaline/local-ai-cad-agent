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

from agent.io import atomic_write_text
from agent.locks import project_history_lock


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
    CACHE_MAX: ClassVar[int] = 64

    #: Roles loaded as active LLM conversation turns (filters out non-chat event records).
    _KEPT_ROLES: ClassVar[set[str]] = {"user", "assistant", "tool"}

    _cache: ClassVar[dict[str, list[dict[str, Any]]]] = {}
    #: Guards ``_cache`` itself. File I/O is serialised per project by
    #: :func:`project_history_lock`; lock order is always history → guard.
    _cache_guard: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def invalidate(cls, project_dir: Path) -> None:
        """Forget the cached entry for ``project_dir`` (used after writes
        and on project deletion).

        Writers call this while holding :func:`project_history_lock`, so a
        concurrent :meth:`load` cannot refill the cache from a pre-write
        snapshot between the invalidate and the file write.
        """
        with cls._cache_guard:
            cls._cache.pop(str(project_dir), None)
            try:
                cls._cache.pop(str(project_dir.resolve()), None)
            except OSError:
                pass

    @classmethod
    def load(cls, project_dir: Path) -> list[dict[str, Any]]:
        """Load the conversation history for ``project_dir``.

        Multimodal ``image_url`` parts are returned verbatim so the
        prompt prefix stays byte-stable across turns. Returning the same
        bytes the provider hashed keeps prompt caches warm.

        Cache hits only take the cache guard. A miss re-checks the cache
        under :func:`project_history_lock` before parsing, so the parse and
        the cache fill cannot interleave with an append for the project.
        """
        cache_key = str(project_dir)
        with cls._cache_guard:
            cached = cls._cache.get(cache_key)
            if cached is not None:
                if not project_dir.exists():
                    cls._cache.pop(cache_key, None)
                    return []
                return [deepcopy(item) for item in cached]

        with project_history_lock(project_dir):
            with cls._cache_guard:
                cached = cls._cache.get(cache_key)
                if cached is not None:
                    if not project_dir.exists():
                        cls._cache.pop(cache_key, None)
                        return []
                    return [deepcopy(item) for item in cached]

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
            with cls._cache_guard:
                cls._set_cached(cache_key, history)
            return [deepcopy(item) for item in history]

    @classmethod
    def read_raw_events(cls, project_dir: Path) -> list[dict[str, Any]]:
        """Return every non-synthetic record of ``conversation.jsonl`` (for the History view)."""
        events: list[dict[str, Any]] = []
        log_path = project_dir / "conversation.jsonl"
        with project_history_lock(project_dir):
            if not log_path.exists():
                return []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and not item.get("synthetic"):
                        events.append(item)
                except json.JSONDecodeError:
                    continue
        return events

    @classmethod
    def append(cls, project_dir: Path, message: dict[str, Any]) -> None:
        """Persist a single message and update the in-memory cache if present.

        The file write and the cache update both happen inside
        :func:`project_history_lock` so a concurrent :meth:`load` always
        observes a consistent snapshot without re-parsing the entire JSONL
        log on every turn.
        """
        with project_history_lock(project_dir):
            log_path = project_dir / "conversation.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if isinstance(message, dict) and message.get("role") in cls._KEPT_ROLES:
                with cls._cache_guard:
                    for cached in cls._cached_histories(project_dir):
                        cached.append(deepcopy(message))

    @classmethod
    def _cached_histories(cls, project_dir: Path) -> list[list[dict[str, Any]]]:
        """Return the cached histories keyed by ``project_dir`` (raw and resolved)."""
        keys = {str(project_dir)}
        try:
            keys.add(str(project_dir.resolve()))
        except OSError:
            pass
        return [cls._cache[key] for key in keys if cls._cache.get(key) is not None]

    @classmethod
    def replace_tool_result(cls, project_dir: Path, message: dict[str, Any]) -> bool:
        """Rewrite the newest ``role: tool`` record whose ``tool_call_id`` matches ``message``.

        Used to attach prefix-safe tail metadata (``step_budget``) to a tool
        result that has been persisted but not yet sent to the LLM, keeping
        ``conversation.jsonl`` and the in-memory cache identical.
        """
        call_id = message.get("tool_call_id")

        def matches(item: Any) -> bool:
            return (
                isinstance(item, dict)
                and item.get("role") == "tool"
                and item.get("tool_call_id") == call_id
            )

        with project_history_lock(project_dir):
            with cls._cache_guard:
                for cached in cls._cached_histories(project_dir):
                    idx = next((i for i in range(len(cached) - 1, -1, -1) if matches(cached[i])), None)
                    if idx is not None:
                        cached[idx] = deepcopy(message)
            log_path = project_dir / "conversation.jsonl"
            if not log_path.exists():
                return False
            lines = log_path.read_text(encoding="utf-8").splitlines()
            for idx in range(len(lines) - 1, -1, -1):
                try:
                    item = json.loads(lines[idx])
                except json.JSONDecodeError:
                    continue
                if matches(item):
                    lines[idx] = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                    atomic_write_text(log_path, "\n".join(lines) + "\n")
                    return True
            return False

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
        status persistence share one per-project lock and one plain
        ``open("a")`` handle. Non-role event envelopes do not alter active
        LLM history (``_KEPT_ROLES``), so the message cache remains warm.
        """
        record = {
            "timestamp": timestamp or cls._now_iso(),
            "type": event_type,
            "data": data,
        }
        with project_history_lock(project_dir):
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
    def rewind_to_user_message(
        cls, project_dir: Path, user_index: int = -1
    ) -> dict[str, Any] | None:
        """Prune conversation.jsonl back to the user turn at ``user_index``
        (0-based index among non-synthetic user messages; -1 denotes the latest).

        Discards any trailing assistant/tool/continuation messages after that
        user turn. Returns the targeted user message dict, or None if no matching
        user turn is found.
        """
        with project_history_lock(project_dir):
            cls.invalidate(project_dir)
            log_path = project_dir / "conversation.jsonl"
            if not log_path.exists():
                return None
            lines = log_path.read_text(encoding="utf-8").splitlines()
            user_turns: list[tuple[int, dict[str, Any]]] = []
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and item.get("role") == "user" and not item.get("synthetic"):
                        user_turns.append((idx, item))
                except json.JSONDecodeError:
                    continue
            if not user_turns:
                return None
            if -len(user_turns) <= user_index < len(user_turns):
                target_idx, target_item = user_turns[user_index]
            else:
                target_idx, target_item = user_turns[-1]
            kept_lines = lines[: target_idx + 1]
            atomic_write_text(log_path, "\n".join(kept_lines) + "\n")
            cls.invalidate(project_dir)
            return deepcopy(target_item)

    @classmethod
    def rewind_to_last_user_message(cls, project_dir: Path) -> dict[str, Any] | None:
        """Prune conversation.jsonl back to the last user turn, discarding any
        uncompleted trailing assistant/tool/continuation messages.

        Returns the last user message dict, or None if no user turn is found.
        """
        return cls.rewind_to_user_message(project_dir, user_index=-1)

    @classmethod
    def clear(cls, project_dir: Path) -> bool:
        """Remove the log file and cache entry. Returns whether anything
        was removed.

        The cache invalidate and the unlink both run inside
        :func:`project_history_lock` so a concurrent append or load
        cannot race with the unlink and resurrect a half-deleted file.
        """
        log_path = project_dir / "conversation.jsonl"
        removed = False
        with project_history_lock(project_dir):
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

        **Caller MUST hold** ``_cache_guard``: the eviction loop pops while
        iterating ``_cache``.
        """
        while len(cls._cache) >= cls.CACHE_MAX:
            oldest = next(iter(cls._cache))
            if oldest == key:
                break
            cls._cache.pop(oldest, None)
        cls._cache[key] = list(history)
