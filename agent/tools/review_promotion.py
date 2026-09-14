"""Shared atomic-promotion helpers for the review / screenshot pipelines.

Both ``CadTool.promote_review`` (canonical eight-view evidence) and
``CadScreenshotTool._promote_to_cache`` (subset variants) need to
materialise a brand-new directory next to an existing target, verify
the PNGs against their manifest hashes, write a manifest.json, and
finally swap the directory in atomically. The original implementations
hand-rolled the same ~25-line "create ``<target>.tmp``, populate,
``os.replace`` with a ``<target>.previous`` rollback, clean up" dance
twice with subtle drift (one raised on missing views, the other
silently skipped them) — a textbook place for a shared helper.

Centralising the swap means both call sites use one tested atomic
contract: an interrupted promotion leaves either the previous directory
intact or the new one fully promoted, never a half-written mix. The
helper does not own view hashing — that policy still lives with each
caller because the strict-vs-lenient behaviour is different.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path


def atomic_swap_directory(target_dir: Path, staging_dir: Path) -> None:
    """Replace ``target_dir`` with ``staging_dir`` via a single ``os.replace``.

    The staging directory lives at ``target_dir.with_suffix(target_dir.suffix
    + ".tmp")`` — a sibling, not a child — so a failed promotion never
    leaves a half-populated ``target_dir`` visible to readers.

    The contract is intentionally narrow: the caller is responsible for
    ensuring ``staging_dir`` already exists with the final contents and
    that any required manifest / hashes were written there before this
    helper runs. The previous target (if any) is renamed to
    ``<target>.previous`` so a crash between the two ``os.replace`` calls
    leaves both an old and a new copy on disk; once the swap lands the
    ``.previous`` sibling is removed.

    The function never deletes ``staging_dir`` *before* the swap — the
    caller just populated it. A failed swap leaves the staging directory
    in place (best-effort cleanup happens in :func:`_promote_to_cache` /
    :meth:`CadTool.promote_review` callers' ``finally`` blocks).
    """
    target_dir = Path(target_dir)
    staging_dir = Path(staging_dir)
    if not staging_dir.is_dir():
        raise FileNotFoundError(
            f"atomic_swap_directory: staging directory is missing: {staging_dir}"
        )
    backup = target_dir.with_suffix(target_dir.suffix + ".previous")
    try:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if target_dir.exists():
            os.replace(target_dir, backup)
        os.replace(staging_dir, target_dir)
    except OSError:
        # Roll back: the previous (now stale) version lives in
        # ``backup``. Restore it when ``target_dir`` is missing — i.e.
        # we crashed after moving the old target aside but before the
        # new one was renamed into place.
        if backup.exists() and not target_dir.exists():
            os.replace(backup, target_dir)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def iter_view_pngs(views: Iterable[dict[str, object]]) -> Iterable[tuple[str, dict[str, object]]]:
    """Yield ``(view_id, manifest_entry)`` pairs from a manifest's ``views`` list.

    Filters out entries that are not dicts or are missing a string
    ``view_id``; both promotion call sites used to repeat this guard
    inline. Iteration order matches the manifest order so the helper is
    a drop-in replacement for the previous inline list comprehensions.
    """
    for entry in views:
        if not isinstance(entry, dict):
            continue
        view_id = entry.get("view_id")
        if not isinstance(view_id, str):
            continue
        yield view_id, entry
