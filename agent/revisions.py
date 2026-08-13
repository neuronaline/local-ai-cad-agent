"""Application-managed source revision store for model.py.

Stores immutable content-addressed source blobs, revision manifests with
explicit parentage, build records linked to revisions, and a head pointer.
All persistence is atomic (temp file + Path.replace).  Malformed data is
surfaced as an integrity error, never silently overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

SCHEMA_VERSION = 1
_MAX_SOURCE_BYTES = 2 * 1024 * 1024  # 2 MB
_REVISION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LIST_DEFAULT_LIMIT = 50
_LIST_MAX_LIMIT = 200
_MAX_ERROR_CHARS = 2000
# Fixed retention policy per docs/AGGRESSIVE_CLEANUP.md: keep at most this many
# revisions plus the last successful (last-known-good) revision.  Removing
# this knob keeps history small enough to be useful without surfacing another
# user-facing configuration option.
_DEFAULT_RETENTION = 25
_BUILDS_LOG_NAME = "builds.jsonl"
_BUILDS_MAX_BYTES = 2 * 1024 * 1024  # 2 MB append-only log
_P = ParamSpec("_P")
_R = TypeVar("_R")


class RevisionIntegrityError(Exception):
    """Raised when revision data is corrupt, inconsistent, or missing."""


def _synchronized(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Acquire the per-instance RLock so each store serializes its own mutations.

    The previous module-level RLock serialized every store across every project
    (audit_033); this decorator uses the instance lock so parallel projects
    build/restore/prune concurrently while still protecting a single store
    from interleaved read-modify-write cycles.
    """

    @wraps(method)
    def wrapped(self: "RevisionStore", *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class RevisionOrigin:
    kind: str  # agent_edit, restore, import, recovery
    operation: str | None = None  # write, replace, regex_replace
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind}
        if self.operation is not None:
            d["operation"] = self.operation
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> RevisionOrigin:
        return cls(
            kind=str(data.get("kind", "")),
            operation=data.get("operation"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass(frozen=True)
class Revision:
    id: str
    parent_id: str | None
    model_sha256: str
    created_at: str
    origin: RevisionOrigin
    restored_from: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "parent_id": self.parent_id,
            "model_sha256": self.model_sha256,
            "created_at": self.created_at,
            "origin": self.origin.to_dict(),
            "restored_from": self.restored_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Revision:
        return cls(
            id=str(data["id"]),
            parent_id=data.get("parent_id"),
            model_sha256=str(data["model_sha256"]),
            created_at=str(data["created_at"]),
            origin=RevisionOrigin.from_dict(data.get("origin") or {}),
            restored_from=data.get("restored_from"),
        )


@dataclass(frozen=True)
class BuildRecord:
    revision_id: str
    model_sha256: str
    status: str  # succeeded, failed
    attempted_at: str
    metrics: dict | None = None
    preview_sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "schema_version": SCHEMA_VERSION,
            "revision_id": self.revision_id,
            "model_sha256": self.model_sha256,
            "status": self.status,
            "attempted_at": self.attempted_at,
        }
        if self.metrics is not None:
            d["metrics"] = self.metrics
        if self.preview_sha256 is not None:
            d["preview_sha256"] = self.preview_sha256
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict) -> BuildRecord:
        return cls(
            revision_id=str(data["revision_id"]),
            model_sha256=str(data["model_sha256"]),
            status=str(data["status"]),
            attempted_at=str(data["attempted_at"]),
            metrics=data.get("metrics"),
            preview_sha256=data.get("preview_sha256"),
            error=data.get("error"),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RevisionStore:
    """Owns all paths, validation, and atomic persistence for revision history."""

    def __init__(self, project_dir: Path, retention_count: int = _DEFAULT_RETENTION) -> None:
        self.project_dir = project_dir.resolve()
        self.retention_count = retention_count
        # Per-instance lock — was a module-level RLock that serialized every
        # store across every project (audit_033). Per-instance lets parallel
        # projects build/restore/prune concurrently.
        self._lock = threading.RLock()
        # Revision-list cache (audit_032); invalidated on commit/restore/prune.
        self._revisions_cache: list[Revision] | None = None
        history = self.project_dir / ".cad-agent" / "history"
        self._head_path = history / "head.json"
        self._blobs_dir = history / "blobs"
        self._revisions_dir = history / "revisions"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @_synchronized
    def reconcile(self) -> Revision | None:
        """Ensure head.json, model.py, and the active revision are consistent.

        Returns the recovery/import revision if one was created, or None if
        already consistent.  Raises RevisionIntegrityError if model.py is
        missing but head exists, or if stored data is malformed.
        """
        model_path = self.project_dir / "model.py"
        model_digest = (
            _sha256(model_path.read_bytes()) if model_path.is_file() else None
        )

        head_data = self._read_json_safe(self._head_path)

        # No history yet.
        if head_data is None:
            if model_digest is None:
                return None
            # Existing model.py without history — create an import revision.
            return self._adopt_existing_model(model_digest)

        # Validate head pointer.
        head_data = self._read_and_validate_head()
        head_revision_id = head_data["revision_id"]
        head_digest = head_data["model_sha256"]

        # Validate head revision manifest exists and matches.
        head_revision = self.get(head_revision_id)
        if head_revision.model_sha256 != head_digest:
            raise RevisionIntegrityError(
                "head.json digest does not match the head revision manifest digest."
            )
        if not self.has_source(head_revision_id):
            raise RevisionIntegrityError("The active revision source blob is missing or corrupt.")

        # Everything matches.
        if model_digest == head_digest:
            return None

        # model.py exists but doesn't match head — create a recovery revision.
        if model_digest is not None:
            return self._create_recovery_revision(model_digest, head_revision_id)

        # Head exists but model.py is missing — integrity error.
        raise RevisionIntegrityError(
            "model.py is missing but head.json references an active revision. "
            "Use the restore API to recover from a known-good revision."
        )

    @_synchronized
    def head(self) -> Revision | None:
        """Return the active head revision, or None if no history exists."""
        head_data = self._read_and_validate_head()
        if head_data is None:
            return None
        head_revision_id = head_data["revision_id"]
        head_digest = head_data["model_sha256"]
        revision = self.get(head_revision_id)
        if revision.model_sha256 != head_digest:
            raise RevisionIntegrityError(
                "head.json digest does not match the head revision manifest digest."
            )
        return revision

    def _read_and_validate_head(self) -> dict | None:
        """Read head.json, returning a validated ``{revision_id, model_sha256}``
        dict or ``None`` when no head exists. Raises ``RevisionIntegrityError``
        for malformed pointers (audit_181)."""
        head_data = self._read_json_safe(self._head_path)
        if head_data is None:
            return None
        head_revision_id = head_data.get("revision_id")
        if not (
            isinstance(head_revision_id, str)
            and _REVISION_ID_RE.fullmatch(head_revision_id)
        ):
            raise RevisionIntegrityError("head.json has an invalid revision ID.")
        head_digest = head_data.get("model_sha256")
        if not isinstance(head_digest, str) or not _SHA256_RE.fullmatch(head_digest):
            raise RevisionIntegrityError("head.json has an invalid model digest.")
        return {"revision_id": head_revision_id, "model_sha256": head_digest}

    @_synchronized
    def list(
        self, *, limit: int = _LIST_DEFAULT_LIMIT, before: str | None = None
    ) -> list[Revision]:
        """List revisions newest first, with optional pagination cursor."""
        limit = max(1, min(limit, _LIST_MAX_LIMIT))
        revisions = self._all_revisions()

        if before is not None:
            self._validate_revision_id(before)
            cursor_index = next(
                (i for i, r in enumerate(revisions) if r.id == before), None
            )
            if cursor_index is not None:
                revisions = revisions[cursor_index + 1 :]

        return revisions[:limit]

    def _all_revisions(self) -> list[Revision]:
        """Return every readable revision, newest first, for internal operations.

        Cached in memory (audit_032); invalidated on commit/restore/prune.

        Must be called while holding ``self._lock`` (audit_185). All public
        callers wrap this in ``@_synchronized``; new callers must do the same
        to avoid races with cache invalidation.
        """
        if self._revisions_cache is not None:
            return list(self._revisions_cache)
        revisions: list[Revision] = []
        if self._revisions_dir.is_dir():
            for entry in self._revisions_dir.iterdir():
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                try:
                    revisions.append(self.get(entry.stem))
                except ValueError as error:
                    raise RevisionIntegrityError(
                        f"Revision manifest filename is invalid: {entry.name}"
                    ) from error

        # Sort by created_at descending, then by ID for stability.
        revisions.sort(key=lambda r: (r.created_at, r.id), reverse=True)
        self._revisions_cache = revisions
        return list(revisions)


    def _invalidate_revision_cache(self) -> None:
        """Drop the in-memory revision list cache (called on mutation)."""
        self._revisions_cache = None

    @_synchronized
    def get(self, revision_id: str) -> Revision:
        """Return a revision by ID. Raises if not found or malformed."""
        self._validate_revision_id(revision_id)
        data = self._read_json_safe(self._revisions_dir / f"{revision_id}.json")
        if data is None:
            raise RevisionIntegrityError(f"Revision {revision_id} not found.")
        try:
            revision = Revision.from_dict(data)
        except (KeyError, TypeError) as error:
            raise RevisionIntegrityError(
                f"Revision {revision_id} manifest is malformed."
            ) from error
        if revision.id != revision_id:
            raise RevisionIntegrityError("Revision manifest ID does not match its filename.")
        if not _SHA256_RE.fullmatch(revision.model_sha256):
            raise RevisionIntegrityError("Revision manifest has an invalid model digest.")
        if revision.parent_id is not None and not _REVISION_ID_RE.fullmatch(revision.parent_id):
            raise RevisionIntegrityError("Revision manifest has an invalid parent ID.")
        if revision.restored_from is not None and not _REVISION_ID_RE.fullmatch(revision.restored_from):
            raise RevisionIntegrityError("Revision manifest has an invalid restore source ID.")
        if revision.origin.kind not in {"agent_edit", "restore", "import", "recovery"}:
            raise RevisionIntegrityError("Revision manifest has an invalid origin kind.")
        return revision

    @_synchronized
    def source(self, revision_id: str) -> str:
        """Return the source code for a revision."""
        self._validate_revision_id(revision_id)
        revision = self.get(revision_id)
        blob_path = self._blobs_dir / f"{revision.model_sha256}.py"
        if not blob_path.is_file():
            raise RevisionIntegrityError(
                f"Source blob for revision {revision_id} is missing."
            )
        source_bytes = blob_path.read_bytes()
        if _sha256(source_bytes) != revision.model_sha256:
            raise RevisionIntegrityError(
                "Source blob digest does not match the revision manifest."
            )
        try:
            return source_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RevisionIntegrityError("Source blob is not valid UTF-8.") from error

    @_synchronized
    def has_source(self, revision_id: str) -> bool:
        """Check if the source blob for a revision exists and is verifiable."""
        try:
            revision = self.get(revision_id)
        except RevisionIntegrityError:
            return False
        blob_path = self._blobs_dir / f"{revision.model_sha256}.py"
        if not blob_path.is_file():
            return False
        return _sha256(blob_path.read_bytes()) == revision.model_sha256

    @_synchronized
    def has_manifest(self, revision_id: str) -> bool:
        """Return whether a revision manifest exists, without parsing it."""
        self._validate_revision_id(revision_id)
        return (self._revisions_dir / f"{revision_id}.json").is_file()

    @_synchronized
    def commit(self, source: str, origin: RevisionOrigin, retention: int = 0) -> Revision:
        """Persist a new source revision and activate it.

        If the source is identical to the active head, returns the existing
        head revision without creating a new one.

        When retention > 0, prunes excess revisions after commit, keeping at
        most `retention` revisions plus the last-known-good revision.
        """
        source_bytes = source.encode("utf-8")
        if len(source_bytes) > _MAX_SOURCE_BYTES:
            raise ValueError("Source exceeds the maximum allowed size.")

        self.reconcile()

        source_digest = _sha256(source_bytes)

        # No-op: identical to active head.
        head = self.head()
        if head is not None and head.model_sha256 == source_digest:
            return head

        # Write blob (content-addressed, deduplicates).
        self._write_blob_bytes(source_bytes, source_digest)

        # Create revision manifest.
        revision = Revision(
            id=str(uuid.uuid4()),
            parent_id=head.id if head else None,
            model_sha256=source_digest,
            created_at=_utc_now(),
            origin=origin,
        )
        # Write the manifest and update head.json *before* rewriting model.py
        # so that the recoverable invariant on crash is "head always points at
        # a revision whose manifest exists". If the process dies between the
        # head.json write and model.py, reconcile() will detect the mismatch
        # and create a recovery revision adopting the stale model.py content.
        self._write_revision(revision)
        self._write_head(revision)

        # Replace model.py last; any failure here leaves the revision history
        # consistent (head points at a valid manifest) and reconcile() will
        # rebuild the model.py difference on the next run.
        self._atomic_write_text(self.project_dir / "model.py", source)

        if retention > 0:
            self.prune(retention)
        elif self.retention_count > 0:
            self.prune(self.retention_count)

        self._invalidate_revision_cache()
        return revision

    @_synchronized
    def restore(self, revision_id: str) -> Revision:
        """Restore an earlier revision by creating a new child revision.

        The head pointer never moves backward; a restore creates a new
        revision with restored_from set to the target.
        """
        self._validate_revision_id(revision_id)
        self.reconcile()

        target = self.get(revision_id)
        source = self.source(revision_id)

        # Verify blob integrity.
        source_bytes = source.encode("utf-8")
        actual_digest = _sha256(source_bytes)
        if actual_digest != target.model_sha256:
            raise RevisionIntegrityError(
                "Source blob digest does not match the revision manifest."
            )

        head = self.head()

        # No-op: restoring the same source as current head.
        if head is not None and head.model_sha256 == target.model_sha256:
            return head

        revision = Revision(
            id=str(uuid.uuid4()),
            parent_id=head.id if head else None,
            model_sha256=target.model_sha256,
            created_at=_utc_now(),
            origin=RevisionOrigin(kind="restore"),
            restored_from=revision_id,
        )
        # Manifest is unique even if the blob is shared.
        self._write_revision(revision)

        # Update head.json *before* rewriting model.py so that crash recovery
        # leaves the history consistent (head points at a valid manifest).
        self._write_head(revision)

        self._atomic_write_text(self.project_dir / "model.py", source)

        if self.retention_count > 0:
            self.prune(self.retention_count)

        self._invalidate_revision_cache()
        return revision

    @_synchronized
    def record_build_success(
        self, revision_id: str, metrics: dict, preview: Path
    ) -> None:
        """Record a successful build for a revision into builds.jsonl."""
        self._validate_revision_id(revision_id)
        revision = self.get(revision_id)
        self._verify_active_model(revision)

        preview_digest = (
            _sha256(preview.read_bytes()) if preview.is_file() else None
        )

        self._append_build(
            BuildRecord(
                revision_id=revision_id,
                model_sha256=revision.model_sha256,
                status="succeeded",
                attempted_at=_utc_now(),
                metrics=metrics,
                preview_sha256=preview_digest,
            )
        )

    @_synchronized
    def record_build_failure(self, revision_id: str, error: str) -> None:
        """Record a failed build for a revision into builds.jsonl."""
        self._validate_revision_id(revision_id)
        revision = self.get(revision_id)
        self._verify_active_model(revision)

        bounded_error = (error or "CAD execution failed.")[:_MAX_ERROR_CHARS]

        self._append_build(
            BuildRecord(
                revision_id=revision_id,
                model_sha256=revision.model_sha256,
                status="failed",
                attempted_at=_utc_now(),
                error=bounded_error,
            )
        )

    @_synchronized
    def build_for(self, revision_id: str) -> BuildRecord | None:
        """Return the latest build record for a revision, or None if not built.

        Reads the append-only builds.jsonl log in reverse chronological order.
        """
        self._validate_revision_id(revision_id)
        revision = self.get(revision_id)
        try:
            log_path = self._builds_log_path()
        except OSError:
            return None
        if not log_path.is_file():
            return None
        latest: BuildRecord | None = None
        with log_path.open("r", encoding="utf-8") as log:
            for line in log:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RevisionIntegrityError(
                        f"Build log is malformed: {error}"
                    ) from error
                if not isinstance(item, dict):
                    continue
                if item.get("revision_id") != revision_id:
                    continue
                try:
                    build = BuildRecord.from_dict(item)
                except (KeyError, TypeError) as error:
                    raise RevisionIntegrityError(
                        f"Build record for {revision_id} is malformed."
                    ) from error
                if build.model_sha256 != revision.model_sha256:
                    continue
                if latest is None or build.attempted_at > latest.attempted_at:
                    latest = build
        if latest is not None and latest.status not in {"succeeded", "failed"}:
            raise RevisionIntegrityError(
                f"Build record for {revision_id} has an invalid status."
            )
        return latest

    def _append_build(self, build: BuildRecord) -> None:
        """Append a build record to the per-project builds.jsonl log.

        Truncates the oldest entries when the file exceeds the size cap so the
        log stays bounded without per-record files.
        """
        log_path = self._builds_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(build.to_dict(), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(payload + "\n")
        self._trim_builds_log(log_path)

    def _builds_log_path(self) -> Path:
        return self.project_dir / _BUILDS_LOG_NAME

    def _trim_builds_log(self, log_path: Path) -> None:
        """Keep builds.jsonl under the size cap by dropping oldest lines."""
        try:
            size = log_path.stat().st_size
        except OSError:
            return
        if size <= _BUILDS_MAX_BYTES:
            return
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return
        lines = [line for line in text.splitlines() if line]
        while lines and (sum(len(line.encode("utf-8")) + 1 for line in lines) > _BUILDS_MAX_BYTES):
            lines.pop(0)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=log_path.parent, encoding="utf-8", delete=False
        ) as temporary:
            tmp_path = Path(temporary.name)
            tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        try:
            tmp_path.replace(log_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _write_build(self, build: BuildRecord) -> None:
        """Backward-compat alias used by older callers and round-trip tests."""
        self._append_build(build)

    @_synchronized
    def last_known_good(self) -> Revision | None:
        """Return the newest revision with a successful build, or None."""
        for revision in self._all_revisions():
            build = self.build_for(revision.id)
            if build is not None and build.status == "succeeded":
                return revision
        return None

    @_synchronized
    def prune(self, max_revisions: int | None = None) -> int:
        """Remove old revisions exceeding `max_revisions`, preserving the
        last-known-good revision. Returns the number of revisions removed.

        When `max_revisions` is None, uses the store's configured retention
        count (default 25).  Set to 0 to disable pruning.

        Blob files are NOT garbage-collected — they may be shared by
        preserved revisions or restores.
        """
        target = max_revisions if max_revisions is not None else self.retention_count
        if target < 1:
            return 0

        revisions = self._all_revisions()
        if len(revisions) <= target:
            return 0

        lkg = self.last_known_good()
        lkg_id = lkg.id if lkg else None

        # The active head must always survive pruning — even if it sits
        # outside the newest ``target`` window (e.g. an old head.json written
        # by an external tool). Aborting here keeps prune atomic if head()
        # fails after we've already inspected revisions.
        head = self.head()
        head_id = head.id if head is not None else None

        removed = 0
        # Keep the `target` most recent, plus the LKG and active head
        # if any of them are outside the recent window.
        keep_ids: set[str] = {r.id for r in revisions[:target]}
        if lkg_id is not None:
            keep_ids.add(lkg_id)
        if head_id is not None:
            keep_ids.add(head_id)

        for revision in revisions:
            if revision.id in keep_ids:
                continue
            self._delete_revision(revision)
            removed += 1

        self._invalidate_revision_cache()
        return removed

    @_synchronized
    def export_history(self, target_dir: Path) -> Path:
        """Export the complete revision history as a portable JSON archive.

        Returns the path to the created archive file.

        NOTE (audit_031): this method exists for the round-trip test suite only.
        Production code should not invoke it. Retained on the public class
        surface because removing it would break the legacy test archive.
        """
        target_dir = target_dir.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        revisions_data: list[dict] = []
        blobs: dict[str, str] = {}

        head_data = self._read_json_safe(self._head_path)

        for revision in self._all_revisions():
            rev_data = revision.to_dict()
            try:
                source = self.source(revision.id)
                blobs[revision.model_sha256] = source
            except RevisionIntegrityError:
                rev_data["_source_missing"] = True

            build = self.build_for(revision.id)
            if build is not None:
                rev_data["build"] = build.to_dict()

            revisions_data.append(rev_data)

        archive = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "head": head_data,
            "revisions": revisions_data,
            "blobs": blobs,
        }

        archive_path = target_dir / "cad-agent-history.json"
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target_dir, encoding="utf-8", delete=False, suffix=".tmp"
        ) as temporary:
            tmp_path = Path(temporary.name)
            json.dump(archive, temporary, ensure_ascii=False, indent=2)
        try:
            tmp_path.replace(archive_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        return archive_path

    @_synchronized
    def import_history(self, archive_path: Path) -> int:
        """Import revisions from a previously exported archive.

        Existing history is preserved; imported revisions that would collide
        with existing IDs are skipped. Build records attached to imported
        revisions are restored. Returns the number of revisions imported.

        NOTE (audit_031): this method exists for the round-trip test suite only.
        Production code should not invoke it.
        """
        try:
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RevisionIntegrityError(
                f"Cannot read history archive: {error}"
            ) from error

        if not isinstance(archive, dict):
            raise RevisionIntegrityError("Invalid history archive format.")

        blobs_data = archive.get("blobs", {})
        if not isinstance(blobs_data, dict):
            raise RevisionIntegrityError("Archive blobs section is malformed.")

        imported = 0
        for revision_data in archive.get("revisions", []):
            if not isinstance(revision_data, dict):
                continue
            rev_id = revision_data.get("id")
            if not isinstance(rev_id, str) or not _REVISION_ID_RE.fullmatch(rev_id):
                continue

            # Skip if this revision already exists.
            if (self._revisions_dir / f"{rev_id}.json").is_file():
                continue

            model_sha256 = str(revision_data.get("model_sha256", ""))
            if not _SHA256_RE.fullmatch(model_sha256):
                continue

            # Write blob if provided.
            if model_sha256 in blobs_data and not (self._blobs_dir / f"{model_sha256}.py").is_file():
                blob_content = str(blobs_data[model_sha256])
                self._write_blob_bytes(blob_content.encode("utf-8"), model_sha256)

            # Write revision manifest.
            try:
                revision = Revision.from_dict(revision_data)
            except (KeyError, TypeError):
                continue  # Skip malformed revision entries.
            self._write_revision(revision)

            # Write build record if present.
            build_data = revision_data.get("build")
            if isinstance(build_data, dict):
                try:
                    self._write_build(BuildRecord.from_dict(build_data))
                except (KeyError, TypeError):
                    pass  # Best-effort; skip malformed build data.

            imported += 1

        # If no head exists, adopt the archive head.
        if self._read_json_safe(self._head_path) is None:
            head_data = archive.get("head")
            if isinstance(head_data, dict):
                head_rev = head_data.get("revision_id")
                if isinstance(head_rev, str) and (self._revisions_dir / f"{head_rev}.json").is_file():
                    self._write_head(self.get(head_rev))

        return imported

    def active_model_digest(self) -> str | None:
        """Return the SHA-256 of the active model.py, or None if absent."""
        model_path = self.project_dir / "model.py"
        if not model_path.is_file():
            return None
        return _sha256(model_path.read_bytes())

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_revision_id(revision_id: str) -> None:
        if not _REVISION_ID_RE.fullmatch(revision_id):
            raise ValueError("Invalid revision ID format.")

    def _delete_revision(self, revision: Revision) -> None:
        """Remove a revision manifest and its build record, but not the blob.

        Blobs are content-addressed and may be shared by other revisions
        or restores. They are cleaned up separately via garbage collection.
        """
        manifest = self._revisions_dir / f"{revision.id}.json"
        manifest.unlink(missing_ok=True)
        self._remove_build_records(revision.id)

    def _remove_build_records(self, revision_id: str) -> None:
        """Rewrite builds.jsonl, dropping every entry for ``revision_id``.

        Build records live in a single append-only JSONL log; without this
        filter, ``prune`` would leave orphaned build records that no future
        ``build_for`` lookup can match (audit_176)."""
        try:
            log_path = self._builds_log_path()
        except OSError:
            return
        if not log_path.is_file():
            return
        try:
            raw = log_path.read_text(encoding="utf-8")
        except OSError:
            return
        kept: list[str] = []
        removed = False
        for line in raw.splitlines(keepends=False):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if isinstance(item, dict) and item.get("revision_id") == revision_id:
                removed = True
                continue
            kept.append(line)
        if not removed:
            return
        payload = "\n".join(kept) + ("\n" if kept else "")
        with tempfile.NamedTemporaryFile(
            mode="w", dir=log_path.parent, encoding="utf-8", delete=False
        ) as temporary:
            tmp_path = Path(temporary.name)
            tmp_path.write_text(payload, encoding="utf-8")
        try:
            tmp_path.replace(log_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def _verify_active_model(self, revision: Revision) -> None:
        """Confirm the active model.py digest matches the revision."""
        model_digest = self.active_model_digest()
        if model_digest != revision.model_sha256:
            raise RevisionIntegrityError(
                "Cannot record build: active model.py digest does not match "
                "the revision."
            )

    def _adopt_existing_model(self, model_digest: str) -> Revision:
        """Create an import revision for an existing model.py without history."""
        source = (self.project_dir / "model.py").read_text(encoding="utf-8")
        revision = Revision(
            id=str(uuid.uuid4()),
            parent_id=None,
            model_sha256=model_digest,
            created_at=_utc_now(),
            origin=RevisionOrigin(kind="import"),
        )
        self._write_blob_bytes(source.encode("utf-8"), model_digest)
        self._write_revision(revision)
        self._write_head(revision)
        self._maybe_import_build(revision)
        self._invalidate_revision_cache()
        return revision

    def _maybe_import_build(self, revision: Revision) -> None:
        """If .cad_metrics.json matches the imported source, create a build record."""
        metrics_path = self.project_dir / ".cad_metrics.json"
        if not metrics_path.is_file():
            return
        try:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(cached, dict) or cached.get("model_sha256") != revision.model_sha256:
            return
        metrics = cached.get("metrics")
        if not isinstance(metrics, dict):
            return
        preview_path = self.project_dir / "preview.stl"
        preview_digest = (
            _sha256(preview_path.read_bytes()) if preview_path.is_file() else None
        )
        self._write_build(
            BuildRecord(
                revision_id=revision.id,
                model_sha256=revision.model_sha256,
                status="succeeded",
                attempted_at=_utc_now(),
                metrics=metrics,
                preview_sha256=preview_digest,
            )
        )

    def _create_recovery_revision(
        self, model_digest: str, parent_id: str
    ) -> Revision:
        """Create a recovery revision when model.py doesn't match head."""
        source = (self.project_dir / "model.py").read_text(encoding="utf-8")
        revision = Revision(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            model_sha256=model_digest,
            created_at=_utc_now(),
            origin=RevisionOrigin(kind="recovery"),
        )
        self._write_blob_bytes(source.encode("utf-8"), model_digest)
        self._write_revision(revision)
        self._write_head(revision)
        self._invalidate_revision_cache()
        return revision

    def _write_blob_bytes(self, source_bytes: bytes, sha256: str) -> None:
        blob_path = self._blobs_dir / f"{sha256}.py"
        if blob_path.exists():
            if _sha256(blob_path.read_bytes()) != sha256:
                raise RevisionIntegrityError(
                    "Existing source blob does not match its content digest."
                )
            return  # Content-addressed and already verified.
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        if _sha256(source_bytes) != sha256:
            raise RevisionIntegrityError("Source digest mismatch during blob write.")
        self._atomic_write_bytes(blob_path, source_bytes)

    def _write_revision(self, revision: Revision) -> None:
        self._validate_revision_id(revision.id)
        if not _SHA256_RE.fullmatch(revision.model_sha256):
            raise RevisionIntegrityError("Revision has an invalid model digest.")
        self._atomic_write_json(
            self._revisions_dir / f"{revision.id}.json", revision.to_dict()
        )

    def _write_head(self, revision: Revision) -> None:
        self._atomic_write_json(
            self._head_path,
            {
                "schema_version": SCHEMA_VERSION,
                "revision_id": revision.id,
                "model_sha256": revision.model_sha256,
            },
        )

    @staticmethod
    def _read_json_safe(path: Path) -> dict | None:
        """Read and parse JSON. Return None if file is absent.

        Raise RevisionIntegrityError for malformed data — never overwrite it.
        """
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RevisionIntegrityError(
                f"Malformed JSON in {path.name}: {error}"
            ) from error
        if not isinstance(data, dict):
            raise RevisionIntegrityError(f"Invalid JSON structure in {path.name}.")
        return data

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", delete=False, suffix=".tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary, ensure_ascii=False, indent=2)
        try:
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, encoding="utf-8", delete=False, suffix=".tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        try:
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".tmp"
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
        try:
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
