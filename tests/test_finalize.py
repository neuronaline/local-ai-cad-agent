import json
from pathlib import Path
from typing import Any

import pytest

from agent.finalize import _parse_journey, finalize_project


class FailingExportCad:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def finalize(self, export_dir: str) -> dict[str, Any]:
        """Simulate combined finalize: produce metrics + render + export, then fail after writing partial exports."""
        (self.project_dir / "render.png").write_bytes(b"png")
        target = self.project_dir / export_dir
        target.mkdir(parents=True)
        (target / "model.step").write_bytes(b"partial")
        (target / "model.stl").write_bytes(b"partial")
        raise RuntimeError("simulated export failure")


def test_failed_export_preserves_previous_output(tmp_path: Path, monkeypatch):
    project = tmp_path / "demo"
    output = project / "output"
    output.mkdir(parents=True)
    (project / "model.py").write_text("# fake\n", encoding="utf-8")
    (output / "model.step").write_bytes(b"previous step")
    (output / "model.stl").write_bytes(b"previous stl")
    (output / "report.md").write_text("previous report", encoding="utf-8")
    monkeypatch.setattr("agent.finalize.CadTool", FailingExportCad)

    with pytest.raises(RuntimeError, match="simulated export failure"):
        finalize_project(project)

    assert (output / "model.step").read_bytes() == b"previous step"
    assert (output / "model.stl").read_bytes() == b"previous stl"
    assert (output / "report.md").read_text(encoding="utf-8") == "previous report"
    assert not list(project.glob(".finalize-*"))


def test_journey_counts_only_cad_run_iterations(tmp_path: Path):
    events = [
        {
            "type": "tool_status",
            "data": {
                "call_id": "inspect",
                "tool": "cad",
                "arguments": {"operation": "inspect"},
                "status": "completed",
            },
        },
        {
            "type": "tool_status",
            "data": {
                "call_id": "run-ok",
                "tool": "cad",
                "arguments": {"operation": "run"},
                "status": "completed",
            },
        },
        {
            "type": "tool_status",
            "data": {
                "call_id": "run-failed",
                "tool": "cad",
                "arguments": {"operation": "run"},
                "status": "error",
            },
        },
    ]
    (tmp_path / "conversation.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    journey = _parse_journey(tmp_path)

    assert journey["cad_runs"] == 1
    assert journey["cad_failures"] == 1
    assert journey["tools"] == "cad(3)"


def test_journey_counts_unified_cad_builds(tmp_path: Path):
    events = [
        {
            "type": "tool_status",
            "data": {
                "call_id": "build-ok",
                "tool": "cad_build_and_verify",
                "arguments": {},
                "status": "completed",
            },
        },
        {
            "type": "tool_status",
            "data": {
                "call_id": "build-failed",
                "tool": "cad_build_and_verify",
                "arguments": {},
                "status": "error",
            },
        },
    ]
    (tmp_path / "conversation.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    journey = _parse_journey(tmp_path)

    assert journey["cad_runs"] == 1
    assert journey["cad_failures"] == 1
    assert journey["tools"] == "cad_build_and_verify(2)"
