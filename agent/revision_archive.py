"""Round-trip archive helpers for ``RevisionStore``.

Both :func:`export_history` and :func:`import_history` were originally
methods on :class:`agent.revisions.RevisionStore` and were documented as
test-suite-only (audit_031). Extracting them into a sibling module keeps
the production class surface focused on commit/restore/prune while still
allowing the round-trip suite to exercise the JSON archive format.

The helpers depend on a handful of ``_underscore``-prefixed store
internals (paths, blob writers, revision writers); that coupling is the
deliberate cost of preserving the exact archive format the tests rely
on. Production callers should treat this module as internal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.io import atomic_write_json, utc_now_iso
from agent.revisions import (
    _REVISION_ID_RE,
    _SHA256_RE,
    SCHEMA_VERSION,
    BuildRecord,
    Revision,
    RevisionIntegrityError,
)


def export_history(store, target_dir: Path) -> Path:
    """Export the complete revision history as a portable JSON archive.

    The returned path points to ``cad-agent-history.json`` inside
    ``target_dir``. The archive payload embeds model source blobs and the
    canonical head pointer so :func:`import_history` can recreate the store.
    """
    with store._lock:
        target_dir = target_dir.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        revisions_data: list[dict] = []
        blobs: dict[str, str] = {}
        head_data = store._read_json_safe(store._head_path)
        for revision in store._all_revisions():
            rev_data = revision.to_dict()
            try:
                blobs[revision.model_sha256] = store.source(revision.id)
            except RevisionIntegrityError:
                rev_data["_source_missing"] = True
            build = store.build_for(revision.id)
            if build is not None:
                rev_data["build"] = build.to_dict()
            revisions_data.append(rev_data)
        archive = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": utc_now_iso(),
            "head": head_data,
            "revisions": revisions_data,
            "blobs": blobs,
        }
        archive_path = target_dir / "cad-agent-history.json"
        atomic_write_json(archive_path, archive)
        return archive_path


def _validate_blob_payload(blobs_data: dict) -> dict[str, bytes]:
    """Return a sha256 → bytes mapping after verifying every payload.

    Each key in the archive must be a 64-char hex SHA-256 and the
    corresponding value (utf-8 source) must hash to that digest. The
    previous implementation deferred this check to ``_write_blob_bytes``
    inside the per-revision write loop, which meant a corrupt blob on
    revision 20 of 50 left revisions 0..19 written to disk and the
    project in a zombie state — and the head pointer never advanced.
    Doing it up front means an archive either imports cleanly or not
    at all.
    """
    validated: dict[str, bytes] = {}
    for sha, content in blobs_data.items():
        if not (isinstance(sha, str) and _SHA256_RE.fullmatch(sha)):
            raise RevisionIntegrityError(
                f"Archive blob key is not a valid sha256: {sha!r}"
            )
        if not isinstance(content, str):
            raise RevisionIntegrityError(
                f"Archive blob {sha} payload is not a string."
            )
        encoded = content.encode("utf-8")
        actual = hashlib.sha256(encoded).hexdigest()
        if actual != sha:
            raise RevisionIntegrityError(
                f"Archive blob {sha} content does not match its declared digest "
                f"(got {actual})."
            )
        validated[sha] = encoded
    return validated


def import_history(store, archive_path: Path) -> int:
    """Import revisions from a previously exported archive.

    Existing history is preserved; imported revisions that would collide
    with existing IDs are skipped. Build records attached to imported
    revisions are restored. Returns the number of revisions imported.

    The import is transactional: every blob and revision manifest is
    validated in memory first, and only after the full validation
    passes do we touch disk. A single corrupt blob therefore leaves the
    store exactly as it was, instead of leaving a zombie partial import.
    """
    with store._lock:
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

        # Phase 1: validate every blob payload in memory. Any failure
        # aborts the import before a single byte is written.
        validated_blobs = _validate_blob_payload(blobs_data)

        # Phase 2: pre-parse every revision manifest + build record.
        # Build the list of pending writes; skip malformed entries
        # exactly as before, but never half-commit.
        pending: list[tuple[Revision, BuildRecord | None]] = []
        for revision_data in archive.get("revisions", []):
            if not isinstance(revision_data, dict):
                continue
            rev_id = revision_data.get("id")
            if not isinstance(rev_id, str) or not _REVISION_ID_RE.fullmatch(rev_id):
                continue
            # Skip if this revision already exists.
            if (store._revisions_dir / f"{rev_id}.json").is_file():
                continue

            model_sha256 = str(revision_data.get("model_sha256", ""))
            if not _SHA256_RE.fullmatch(model_sha256):
                continue

            # Skip revisions whose blob failed validation — they would
            # never be importable, but raise explicitly so an operator
            # sees the broken digest instead of silent loss.
            if model_sha256 not in validated_blobs and not (
                store._blobs_dir / f"{model_sha256}.py"
            ).is_file():
                raise RevisionIntegrityError(
                    f"Archive references revision {rev_id} with sha256 "
                    f"{model_sha256} but provides no matching blob."
                )

            try:
                revision = Revision.from_dict(revision_data)
            except (KeyError, TypeError):
                continue  # Skip malformed revision entries.

            build_record: BuildRecord | None = None
            build_data = revision_data.get("build")
            if isinstance(build_data, dict):
                try:
                    build_record = BuildRecord.from_dict(build_data)
                except (KeyError, TypeError):
                    build_record = None

            pending.append((revision, build_record))

        # Phase 3: write everything. ``_write_blob_bytes`` may still
        # raise ``RevisionIntegrityError`` if the on-disk blob disagrees
        # with its declared digest, but that is the canonical integrity
        # path — a different failure mode from the import-time
        # corruption we just guarded against.
        imported = 0
        for revision, build_record in pending:
            model_sha256 = revision.model_sha256
            if model_sha256 in validated_blobs and not (
                store._blobs_dir / f"{model_sha256}.py"
            ).is_file():
                store._write_blob_bytes(validated_blobs[model_sha256], model_sha256)
            store._write_revision(revision)
            if build_record is not None:
                store._append_build(build_record)
            imported += 1

        # If no head exists, adopt the archive head.
        if store._read_json_safe(store._head_path) is None:
            head_data = archive.get("head")
            if isinstance(head_data, dict):
                head_rev = head_data.get("revision_id")
                if (
                    isinstance(head_rev, str)
                    and (store._revisions_dir / f"{head_rev}.json").is_file()
                ):
                    store._write_head(store.get(head_rev))

        return imported
