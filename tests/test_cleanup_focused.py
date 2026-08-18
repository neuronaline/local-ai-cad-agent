"""Focused tests for the reduced workflow.

Covers the minimums introduced by docs/AGGRESSIVE_CLEANUP.md:
- builds.jsonl append-only build history
- 25 + last-successful revision retention
- basic geometry enforcement inside CadTool.build_and_verify
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.revisions import RevisionOrigin, RevisionStore
from agent.tools.cad_tool import CadTool

# --- Builds.jsonl --------------------------------------------------------- #


def test_builds_are_appended_to_builds_jsonl(tmp_path: Path):
    store = RevisionStore(tmp_path)
    revision = store.commit("result = 1\n", RevisionOrigin(kind="agent_edit"))
    (tmp_path / "preview.stl").write_bytes(b"solid")

    store.record_build_success(
        revision.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )
    store.record_build_failure(revision.id, "first failure")
    store.record_build_success(
        revision.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )

    log = tmp_path / "builds.jsonl"
    assert log.is_file()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    assert lines[0]["status"] == "succeeded"
    assert lines[1]["status"] == "failed"
    assert lines[2]["status"] == "succeeded"


def test_build_for_returns_latest_matching_entry(tmp_path: Path):
    store = RevisionStore(tmp_path)
    revision = store.commit("result = 1\n", RevisionOrigin(kind="agent_edit"))
    (tmp_path / "preview.stl").write_bytes(b"solid")

    store.record_build_success(
        revision.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )
    store.record_build_failure(revision.id, "transient")
    store.record_build_success(
        revision.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )

    latest = store.build_for(revision.id)
    assert latest is not None
    assert latest.status == "succeeded"
    # Earlier failed attempts must not become the latest.
    failures = [
        json.loads(line)
        for line in (tmp_path / "builds.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("status") == "failed"
    ]
    assert len(failures) == 1


def test_build_for_skips_entries_for_other_revisions(tmp_path: Path):
    store = RevisionStore(tmp_path)
    rev1 = store.commit("result = 1\n", RevisionOrigin(kind="agent_edit"))
    (tmp_path / "preview.stl").write_bytes(b"solid")

    store.record_build_success(
        rev1.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )
    # Snapshot the entry recorded for rev1 before a new revision supersedes
    # the on-disk model.py.
    rev1_latest = store.build_for(rev1.id)

    rev2_source = "result = 2\n"
    (tmp_path / "model.py").write_text(rev2_source, encoding="utf-8")
    rev2 = store.commit(rev2_source, RevisionOrigin(kind="agent_edit"))
    store.record_build_success(
        rev2.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
    )

    assert rev1_latest.revision_id == rev1.id
    assert store.build_for(rev1.id).revision_id == rev1.id
    assert store.build_for(rev2.id).revision_id == rev2.id


# --- Retention ------------------------------------------------------------ #


def test_commit_prunes_to_25_plus_last_successful(tmp_path: Path):
    store = RevisionStore(tmp_path)  # uses default retention
    # 30 revisions: 25 kept + LKG + 4 dropped.
    revisions = []
    for index in range(30):
        rev = store.commit(f"result = {index}\n", RevisionOrigin(kind="agent_edit"))
        # Mark the very first revision as the last-known-good.
        if index == 0:
            (tmp_path / "preview.stl").write_bytes(b"solid")
            store.record_build_success(
                rev.id, {"solid_count": 1, "is_valid": True}, tmp_path / "preview.stl"
            )
        revisions.append(rev)

    head = store.head()
    assert head.id == revisions[-1].id  # most recent is always head.

    # First revision (LKG) must survive even though it is older than 25 commits.
    assert store.has_source(revisions[0].id)

    # Revisions 1-4 are older than the kept window and newer than the LKG —
    # they should be pruned away. (5-29 are the 25 newest plus LKG.)
    for index in range(1, 5):
        assert not store.has_manifest(revisions[index].id), (
            f"revision {index} should be pruned"
        )

    # The 25 most-recent non-LKG revisions should remain.
    for index in range(5, 30):
        assert store.has_manifest(revisions[index].id), (
            f"revision {index} should be kept"
        )


# --- Basic geometry ------------------------------------------------------- #


def _write_minimal_metrics(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": 1,
        "model_sha256": hashlib.sha256(b"result = 1\n").hexdigest(),
        "metrics": {
            "solid_count": 1,
            "is_valid": True,
            "volume_mm3": 10.0,
            "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
    }
    payload["metrics"].update(overrides)
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_basic_geometry_rejects_missing_solids(tmp_path: Path):
    cad = CadTool(tmp_path)
    with pytest.raises(ValueError, match="solid"):
        cad._enforce_basic_geometry(
            {
                "solid_count": 0,
                "is_valid": True,
                "volume_mm3": 5.0,
                "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
            }
        )


def test_basic_geometry_rejects_invalid_brep(tmp_path: Path, monkeypatch):
    """The geometry check runs without invoking the sandbox subprocess."""
    cad = CadTool(tmp_path)
    with pytest.raises(ValueError, match="not valid"):
        cad._enforce_basic_geometry({"solid_count": 1, "is_valid": False})


def test_basic_geometry_rejects_zero_volume(tmp_path: Path):
    cad = CadTool(tmp_path)
    with pytest.raises(ValueError, match="volume"):
        cad._enforce_basic_geometry(
            {
                "solid_count": 1,
                "is_valid": True,
                "volume_mm3": 0.0,
                "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
            }
        )


def test_basic_geometry_rejects_negative_dimension(tmp_path: Path):
    cad = CadTool(tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        cad._enforce_basic_geometry(
            {
                "solid_count": 1,
                "is_valid": True,
                "volume_mm3": 5.0,
                "dimensions_mm": {"x": -1.0, "y": 1.0, "z": 1.0},
            }
        )


def test_basic_geometry_rejects_malformed_dimensions(tmp_path: Path):
    cad = CadTool(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        cad._enforce_basic_geometry(
            {
                "solid_count": 1,
                "is_valid": True,
                "volume_mm3": 5.0,
                "dimensions_mm": {"x": 1.0, "y": 1.0},
            }
        )
