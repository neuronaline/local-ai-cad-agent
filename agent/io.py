"""Shared low-level I/O utilities.

The helpers here consolidate the "temp sibling + fsync + atomic replace"
pattern that was previously duplicated across :mod:`agent.revisions`,
:mod:`agent.activity_log`, :mod:`agent.revision_archive`, and
``agent.tools.cad_tool``. Centralising the helpers keeps the cleanup-on-
exception behaviour uniform so an interrupted write never leaves a
``.tmp`` file behind.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a file with text content.

    Creates a temp sibling, writes with ``fsync``, then renames over the
    target.  If the write or rename fails the temp file is cleaned up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a file with raw bytes; mirror of :func:`atomic_write_text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: dict) -> None:
    """Atomically replace a file with a JSON document.

    Uses :func:`atomic_write_text` so the temp-cleanup and fsync semantics
    are identical to the canonical text helper.
    """
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone."""
    return datetime.now(timezone.utc).isoformat()
