"""Unit tests for the RevisionStore source revision system."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.revisions import (
    RevisionIntegrityError,
    RevisionOrigin,
    RevisionStore,
)

MODEL_A = "from build123d import Box\nresult = Box(10, 20, 30)\n"
MODEL_B = "from build123d import Box\nresult = Box(40, 20, 30)\n"
MODEL_C = "from build123d import Box\nresult = Box(50, 60, 70)\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
#  First write, blob, manifest, head
# --------------------------------------------------------------------------- #

def test_first_model_write_creates_blob_manifest_and_head(tmp_path: Path):
    store = RevisionStore(tmp_path)
    revision = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit", operation="write"))

    assert revision.parent_id is None
    assert revision.model_sha256 == _sha(MODEL_A)
    assert revision.origin.kind == "agent_edit"

    # Blob exists and matches.
    blob = tmp_path / ".cad-agent" / "history" / "blobs" / f"{revision.model_sha256}.py"
    assert blob.is_file()
    assert blob.read_text(encoding="utf-8") == MODEL_A

    # Manifest exists.
    manifest = tmp_path / ".cad-agent" / "history" / "revisions" / f"{revision.id}.json"
    assert manifest.is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["id"] == revision.id
    assert data["model_sha256"] == revision.model_sha256

    # Head points to the revision.
    head = store.head()
    assert head is not None
    assert head.id == revision.id

    # model.py was written.
    assert (tmp_path / "model.py").read_text(encoding="utf-8") == MODEL_A


# --------------------------------------------------------------------------- #
#  Deduplication
# --------------------------------------------------------------------------- #

def test_identical_source_deduplicates_blob_and_creates_no_new_revision(tmp_path: Path):
    store = RevisionStore(tmp_path)
    first = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    second = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    assert first.id == second.id
    blobs = list((tmp_path / ".cad-agent" / "history" / "blobs").iterdir())
    assert len(blobs) == 1


# --------------------------------------------------------------------------- #
#  Sequential edits and parent IDs
# --------------------------------------------------------------------------- #

def test_sequential_edits_have_correct_parent_ids(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    r2 = store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))
    r3 = store.commit(MODEL_C, RevisionOrigin(kind="agent_edit"))

    assert r2.parent_id == r1.id
    assert r3.parent_id == r2.id
    assert store.head().id == r3.id


# --------------------------------------------------------------------------- #
#  Restore creates a new child revision
# --------------------------------------------------------------------------- #

def test_restore_creates_new_child_revision_with_restored_from(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))
    r3 = store.commit(MODEL_C, RevisionOrigin(kind="agent_edit"))

    restored = store.restore(r1.id)

    assert restored.parent_id == r3.id
    assert restored.restored_from == r1.id
    assert restored.model_sha256 == r1.model_sha256
    assert store.head().id == restored.id
    assert (tmp_path / "model.py").read_text(encoding="utf-8") == MODEL_A

    # Blob is shared (deduplicated).
    blobs = list((tmp_path / ".cad-agent" / "history" / "blobs").iterdir())
    assert len(blobs) == 3  # A, B, C — no new blob for restore


def test_restore_same_as_head_is_noop(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    restored = store.restore(r1.id)
    assert restored.id == r1.id


# --------------------------------------------------------------------------- #
#  Source blob verification
# --------------------------------------------------------------------------- #

def test_source_blob_hash_is_verified_before_restore(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    # Tamper with the blob.
    blob = tmp_path / ".cad-agent" / "history" / "blobs" / f"{r1.model_sha256}.py"
    blob.write_text("tampered content", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError, match="missing or corrupt"):
        store.restore(r1.id)


def test_commit_rejects_existing_corrupt_content_addressed_blob(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))
    blob = tmp_path / ".cad-agent" / "history" / "blobs" / f"{r1.model_sha256}.py"
    blob.write_text("tampered", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError, match="Existing source blob"):
        store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))


# --------------------------------------------------------------------------- #
#  Malformed data is preserved, not overwritten
# --------------------------------------------------------------------------- #

def test_malformed_head_raises_integrity_error(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    head_path = tmp_path / ".cad-agent" / "history" / "head.json"
    head_path.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError, match="Malformed JSON"):
        store.head()


def test_malformed_manifest_raises_integrity_error(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    manifest = tmp_path / ".cad-agent" / "history" / "revisions" / f"{r1.id}.json"
    manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError):
        store.get(r1.id)


def test_listing_revisions_reports_malformed_manifest(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    bad_manifest = tmp_path / ".cad-agent" / "history" / "revisions" / "not-a-revision.json"
    bad_manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError, match="filename is invalid"):
        store.list()


def test_malformed_build_record_raises_integrity_error(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    log_path = tmp_path / "builds.jsonl"
    log_path.write_text("garbage\n", encoding="utf-8")

    with pytest.raises(RevisionIntegrityError):
        store.build_for(r1.id)


# --------------------------------------------------------------------------- #
#  Crash reconciliation
# --------------------------------------------------------------------------- #

def test_interrupted_head_source_mismatch_creates_recovery_revision(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    # Simulate external edit to model.py (bypassing the store).
    (tmp_path / "model.py").write_text(MODEL_B, encoding="utf-8")

    recovery = store.reconcile()
    assert recovery is not None
    assert recovery.origin.kind == "recovery"
    assert recovery.parent_id == r1.id
    assert recovery.model_sha256 == _sha(MODEL_B)
    assert store.head().id == recovery.id


def test_consistent_state_reconcile_returns_none(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    assert store.reconcile() is None


def test_missing_model_with_head_raises_integrity_error(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    (tmp_path / "model.py").unlink()

    with pytest.raises(RevisionIntegrityError, match="model.py is missing"):
        store.reconcile()


# --------------------------------------------------------------------------- #
#  Lazy import
# --------------------------------------------------------------------------- #

def test_existing_source_is_imported_lazily(tmp_path: Path):
    (tmp_path / "model.py").write_text(MODEL_A, encoding="utf-8")

    store = RevisionStore(tmp_path)
    revision = store.reconcile()

    assert revision is not None
    assert revision.origin.kind == "import"
    assert revision.parent_id is None
    assert revision.model_sha256 == _sha(MODEL_A)
    assert store.head().id == revision.id


def test_no_model_and_no_history_reconcile_returns_none(tmp_path: Path):
    store = RevisionStore(tmp_path)
    assert store.reconcile() is None
    assert store.head() is None


# --------------------------------------------------------------------------- #
#  Pagination
# --------------------------------------------------------------------------- #

def test_pagination_is_stable_and_bounded(tmp_path: Path):
    store = RevisionStore(tmp_path)
    for i in range(10):
        store.commit(f"# model v{i}\nresult = {i}\n", RevisionOrigin(kind="agent_edit"))

    page1 = store.list(limit=5)
    page2 = store.list(limit=5, before=page1[-1].id)

    assert len(page1) == 5
    assert len(page2) == 5
    assert page1[0].id != page2[0].id
    # No overlap.
    page1_ids = {r.id for r in page1}
    page2_ids = {r.id for r in page2}
    assert page1_ids.isdisjoint(page2_ids)
    # Newest first.
    assert page1[0].created_at >= page1[-1].created_at


def test_list_max_limit_is_enforced(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    result = store.list(limit=10000)
    assert len(result) <= 200


# --------------------------------------------------------------------------- #
#  Build records
# --------------------------------------------------------------------------- #

def test_record_build_success_stores_metrics_and_preview(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    preview_path = tmp_path / "preview.stl"
    preview_path.write_bytes(b"stl data")
    metrics = {"solid_count": 1, "is_valid": True, "volume_mm3": 6000.0}

    store.record_build_success(r1.id, metrics, preview_path)

    build = store.build_for(r1.id)
    assert build is not None
    assert build.status == "succeeded"
    assert build.metrics == metrics
    assert build.preview_sha256 == _sha("stl data")


def test_record_build_failure_stores_bounded_error(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    long_error = "x" * 5000
    store.record_build_failure(r1.id, long_error)

    build = store.build_for(r1.id)
    assert build is not None
    assert build.status == "failed"
    assert build.metrics is None
    assert len(build.error) <= 2000


def test_build_cannot_attach_after_source_digest_changes(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))

    # Active model is now MODEL_B; r1's digest no longer matches.
    with pytest.raises(RevisionIntegrityError, match="does not match"):
        store.record_build_success(r1.id, {}, tmp_path / "preview.stl")


# --------------------------------------------------------------------------- #
#  Last-known-good
# --------------------------------------------------------------------------- #

def test_last_known_good_selects_newest_successful(tmp_path: Path):
    store = RevisionStore(tmp_path)
    # Commit and build each revision in sequence so each is active when built.
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.record_build_success(r1.id, {"solid_count": 1}, tmp_path / "preview.stl")

    r2 = store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))
    store.record_build_failure(r2.id, "broken")

    store.commit(MODEL_C, RevisionOrigin(kind="agent_edit"))
    # r3 is not built.

    lkg = store.last_known_good()
    assert lkg is not None
    # The newest successful build is r1 (r2 failed, r3 not built).
    assert lkg.model_sha256 == r1.model_sha256


def test_last_known_good_returns_none_when_no_successful_builds(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    assert store.last_known_good() is None


# --------------------------------------------------------------------------- #
#  Atomic write cleanup
# --------------------------------------------------------------------------- #

def test_atomic_write_temp_files_are_cleaned(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    # No .tmp files left behind.
    history = tmp_path / ".cad-agent" / "history"
    for subdir in ("blobs", "revisions", "builds"):
        d = history / subdir
        if d.is_dir():
            assert not any(p.suffix == ".tmp" for p in d.iterdir())
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
#  Import with existing metrics
# --------------------------------------------------------------------------- #

def test_import_creates_build_record_when_metrics_match(tmp_path: Path):
    (tmp_path / "model.py").write_text(MODEL_A, encoding="utf-8")
    metrics = {"solid_count": 1, "is_valid": True, "volume_mm3": 6000.0}
    (tmp_path / ".cad_metrics.json").write_text(
        json.dumps({"model_sha256": _sha(MODEL_A), "metrics": metrics}),
        encoding="utf-8",
    )
    (tmp_path / "preview.stl").write_bytes(b"stl data")

    store = RevisionStore(tmp_path)
    revision = store.reconcile()

    assert revision is not None
    build = store.build_for(revision.id)
    assert build is not None
    assert build.status == "succeeded"
    assert build.metrics == metrics


def test_import_does_not_create_build_when_metrics_mismatch(tmp_path: Path):
    (tmp_path / "model.py").write_text(MODEL_A, encoding="utf-8")
    (tmp_path / ".cad_metrics.json").write_text(
        json.dumps({"model_sha256": "wrong", "metrics": {}}),
        encoding="utf-8",
    )

    store = RevisionStore(tmp_path)
    revision = store.reconcile()

    assert revision is not None
    assert store.build_for(revision.id) is None


# --------------------------------------------------------------------------- #
#  Revision ID validation
# --------------------------------------------------------------------------- #

def test_invalid_revision_id_rejected(tmp_path: Path):
    store = RevisionStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid revision ID"):
        store.get("not-a-uuid")

    with pytest.raises(ValueError, match="Invalid revision ID"):
        store.restore("../../../etc/passwd")


# --------------------------------------------------------------------------- #
#  Source retrieval
# --------------------------------------------------------------------------- #

def test_source_returns_correct_content(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    r2 = store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))

    assert store.source(r1.id) == MODEL_A
    assert store.source(r2.id) == MODEL_B


def test_has_source_returns_false_for_missing_blob(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))

    blob = tmp_path / ".cad-agent" / "history" / "blobs" / f"{r1.model_sha256}.py"
    blob.unlink()

    assert not store.has_source(r1.id)


# --------------------------------------------------------------------------- #
#  Tool call ID in origin
# --------------------------------------------------------------------------- #

def test_tool_call_id_is_stored_in_origin(tmp_path: Path):
    store = RevisionStore(tmp_path)
    revision = store.commit(
        MODEL_A,
        RevisionOrigin(kind="agent_edit", operation="replace", tool_call_id="call_123"),
    )

    fetched = store.get(revision.id)
    assert fetched.origin.tool_call_id == "call_123"
    assert fetched.origin.operation == "replace"


# --------------------------------------------------------------------------- #
#  Retention / prune
# --------------------------------------------------------------------------- #

def test_prune_removes_old_revisions(tmp_path: Path):
    store = RevisionStore(tmp_path, retention_count=3)
    for i in range(6):
        store.commit(f"# model v{i}\nresult = {i}\n", RevisionOrigin(kind="agent_edit"))

    revisions = store.list(limit=200)
    assert len(revisions) <= 3


def test_prune_preserves_last_known_good(tmp_path: Path):
    store = RevisionStore(tmp_path, retention_count=2)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.record_build_success(r1.id, {"solid_count": 1}, tmp_path / "preview.stl")

    store.commit(MODEL_B, RevisionOrigin(kind="agent_edit"))
    store.commit(MODEL_C, RevisionOrigin(kind="agent_edit"))

    # r1 is the only LKG, so it must survive pruning even if old.
    revisions = store.list(limit=200)
    ids = {r.id for r in revisions}
    assert r1.id in ids


def test_prune_returns_zero_when_under_limit(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    assert store.prune(10) == 0


def test_prune_returns_zero_when_no_retention(tmp_path: Path):
    store = RevisionStore(tmp_path, retention_count=0)
    for i in range(10):
        store.commit(f"result = {i}\n", RevisionOrigin(kind="agent_edit"))
    assert len(store.list(limit=200)) == 10


def test_commit_honors_store_retention_count(tmp_path: Path):
    store = RevisionStore(tmp_path, retention_count=2)
    for i in range(5):
        store.commit(f"result = {i}\n", RevisionOrigin(kind="agent_edit"))

    assert len(store.list(limit=200)) <= 2


# --------------------------------------------------------------------------- #
#  Export / import
# --------------------------------------------------------------------------- #

def test_export_produces_archive_with_head_and_builds(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.record_build_success(r1.id, {"solid_count": 1}, tmp_path / "preview.stl")

    archive = store.export_history(tmp_path / "export")
    assert archive.is_file()
    data = json.loads(archive.read_text(encoding="utf-8"))
    assert len(data["revisions"]) == 1
    assert data["revisions"][0]["id"] == r1.id
    assert data["revisions"][0]["build"]["status"] == "succeeded"
    assert data["blobs"][r1.model_sha256] == MODEL_A


def test_import_restores_revisions_and_builds(tmp_path: Path):
    store = RevisionStore(tmp_path)
    r1 = store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    store.record_build_success(r1.id, {"solid_count": 1}, tmp_path / "preview.stl")

    archive = store.export_history(tmp_path / "export")

    # Import into a new project.
    target = tmp_path / "imported"
    target.mkdir()
    store2 = RevisionStore(target)
    count = store2.import_history(archive)
    assert count == 1
    assert store2.head() is not None
    assert store2.head().model_sha256 == r1.model_sha256
    assert store2.source(r1.id) == MODEL_A
    build = store2.build_for(r1.id)
    assert build is not None
    assert build.status == "succeeded"


def test_import_skips_existing_revisions(tmp_path: Path):
    store = RevisionStore(tmp_path)
    store.commit(MODEL_A, RevisionOrigin(kind="agent_edit"))
    archive = store.export_history(tmp_path / "export")

    # Import into the same store — should skip.
    count = store.import_history(archive)
    assert count == 0


def test_import_handles_malformed_archive(tmp_path: Path):
    store = RevisionStore(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(RevisionIntegrityError, match="Cannot read"):
        store.import_history(bad)


# --------------------------------------------------------------------------- #
#  Settings retention count
# --------------------------------------------------------------------------- #

def test_settings_revision_retention_count_defaults_to_zero():
    from agent.settings import Settings
    s = Settings(Path("/tmp"), "https://e.test", "m", 1, "h", 1)
    assert s.revision_retention_count == 0


def test_settings_revision_retention_count_can_be_set():
    from agent.settings import Settings
    s = Settings(Path("/tmp"), "https://e.test", "m", 1, "h", 1, revision_retention_count=50)
    assert s.revision_retention_count == 50
