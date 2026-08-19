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

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agent.revisions import (
    _REVISION_ID_RE,
    _SHA256_RE,
    SCHEMA_VERSION,
    BuildRecord,
    Revision,
    RevisionIntegrityError,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            "exported_at": _utc_now(),
            "head": head_data,
            "revisions": revisions_data,
            "blobs": blobs,
        }
        archive_path = target_dir / "cad-agent-history.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target_dir,
            encoding="utf-8",
            delete=False,
            suffix=".tmp",
        ) as temporary:
            tmp_path = Path(temporary.name)
            json.dump(archive, temporary, ensure_ascii=False, indent=2)
        try:
            tmp_path.replace(archive_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return archive_path


def import_history(store, archive_path: Path) -> int:
    """Import revisions from a previously exported archive.

    Existing history is preserved; imported revisions that would collide
    with existing IDs are skipped. Build records attached to imported
    revisions are restored. Returns the number of revisions imported.
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

        imported = 0
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

            # Write blob if provided.
            if model_sha256 in blobs_data and not (
                store._blobs_dir / f"{model_sha256}.py"
            ).is_file():
                blob_content = str(blobs_data[model_sha256])
                store._write_blob_bytes(blob_content.encode("utf-8"), model_sha256)

            # Write revision manifest.
            try:
                revision = Revision.from_dict(revision_data)
            except (KeyError, TypeError):
                continue  # Skip malformed revision entries.
            store._write_revision(revision)

            # Write build record if present.
            build_data = revision_data.get("build")
            if isinstance(build_data, dict):
                try:
                    store._append_build(BuildRecord.from_dict(build_data))
                except (KeyError, TypeError):
                    pass  # Best-effort; skip malformed build data.

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
