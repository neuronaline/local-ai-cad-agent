"""Process-wide ref-counted per-key RLock registries with idle LRU eviction.

``project_model_lock`` serialises ``model.scad`` / revision-store mutations and
``project_history_lock`` serialises ``conversation.jsonl`` / debug-log I/O.
The two are independent; avoid nesting them so no lock-order cycle can form.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class _RefLockEntry:
    __slots__ = ("active_count", "lock")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.active_count = 0


class KeyedRLockRegistry:
    """Thread-safe keyed ``RLock`` table that evicts only idle (``active_count == 0``) keys.

    ``active_count`` covers waiters as well as the holder, so an entry can
    never be evicted (and silently replaced) while any thread references it.
    """

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(16, max_entries)
        self._entries: OrderedDict[str, _RefLockEntry] = OrderedDict()
        self._guard = threading.Lock()

    @contextmanager
    def acquire(self, key: str) -> Iterator[threading.RLock]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _RefLockEntry()
                self._entries[key] = entry
            entry.active_count += 1
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_entries:
                for k in list(self._entries):
                    if len(self._entries) <= self._max_entries:
                        break
                    if self._entries[k].active_count == 0:
                        self._entries.pop(k, None)
        entry.lock.acquire()
        try:
            yield entry.lock
        finally:
            entry.lock.release()
            with self._guard:
                entry.active_count = max(0, entry.active_count - 1)

    def invalidate(self, key: str) -> None:
        """Drop ``key`` if idle (e.g. after its project was deleted or renamed)."""
        with self._guard:
            entry = self._entries.get(key)
            if entry is not None and entry.active_count == 0:
                self._entries.pop(key, None)


_MODEL_LOCKS = KeyedRLockRegistry()
_HISTORY_LOCKS = KeyedRLockRegistry()


def _key(project_dir: Path) -> str:
    return str(Path(project_dir).resolve())


@contextmanager
def project_model_lock(project_dir: Path) -> Iterator[threading.RLock]:
    with _MODEL_LOCKS.acquire(_key(project_dir)) as lk:
        yield lk


def invalidate_project_model_lock(project_dir: Path) -> None:
    _MODEL_LOCKS.invalidate(_key(project_dir))


@contextmanager
def project_history_lock(project_dir: Path) -> Iterator[threading.RLock]:
    with _HISTORY_LOCKS.acquire(_key(project_dir)) as lk:
        yield lk


def invalidate_project_history_lock(project_dir: Path) -> None:
    _HISTORY_LOCKS.invalidate(_key(project_dir))
