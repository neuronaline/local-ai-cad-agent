"""Conversation persistence extracted from ``agent.core.AgentRunner``.

The store owns the canonical ``conversation.jsonl`` file for a project:
loading (with image redaction + FIFO cache + bounded history), appending,
and clearing. Splitting it out keeps ``AgentRunner`` focused on the
tool-calling lifecycle without dragging in JSONL serialisation rules.
"""
from __future__ import annotations

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
        with (
            shared_history_lock(),
            (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log,
        ):
            log.write(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    @classmethod
    def clear(cls, project_dir: Path) -> bool:
        """Remove the log file and cache entry. Returns whether anything
        was removed."""
        cls.invalidate(project_dir)
        log_path = project_dir / "conversation.jsonl"
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
        if len(history) <= ConversationStore.MAX_HISTORY:
            return history
        truncated = history[-ConversationStore.MAX_HISTORY:]
        # A raw tail cut can start between an assistant tool call and its tool
        # result. Leading tool messages are invalid without their supplying
        # assistant message, so advance to the next coherent boundary.
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
