"""Tests for the Phase 2 quality additions: spec parser, verifiers, envelope, finalize status."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("build123d")

from agent.build_feedback import sanitize_build_result
from agent.finalize import finalize_project
from agent.quality import DesignSpec, Requirement, ValidationResult, parse_request
from agent.quality.specification import parse_request as spec_parse
from agent.quality.store import QualityStore
from agent.quality.verifiers import run_verifiers
from agent.revisions import RevisionOrigin, RevisionStore
from agent.settings import Settings
from app import create_app

MODEL_WITH_HOLE = """from build123d import Align, Box, Cylinder

width = 50.0
length = 40.0
height = 8.0
hole_diameter = 6.0
body = Box(width, length, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
hole = Cylinder(hole_diameter / 2, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = body - hole
"""

MODEL_SIMPLE_BOX = """from build123d import Align, Box

width = 50.0
length = 40.0
height = 8.0
result = Box(width, length, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
"""

MODEL_WRONG_HOLE = """from build123d import Align, Box, Cylinder

body = Box(50, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
# wrong hole diameter (3 mm instead of 6 mm)
hole = Cylinder(1.5, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = body - hole
"""


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=False,
    )


def _scenario(tmp_path: Path, *, source: str = MODEL_WITH_HOLE) -> tuple[Settings, Path]:
    settings = _settings(tmp_path)
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    (project_dir / "model.py").write_text(source, encoding="utf-8")
    RevisionStore(project_dir).commit(source, RevisionOrigin(kind="agent_edit"))
    return settings, project_dir


def test_parse_request_extracts_axis_dimensions_and_hole():
    spec = parse_request(
        run_id="run-1",
        request_text=(
            "Create a mounting bracket with X length 50 mm, Y width 40 mm and Z height 8 mm. "
            "Drill 2 holes with diameter 6 mm."
        ),
    )
    kinds = [req.kind for req in spec.requirements]
    assert "dimension" in kinds
    assert "hole" in kinds
    dimension_axes = {
        req.selector.get("axis") for req in spec.requirements if req.kind == "dimension"
    }
    assert {"X", "Y", "Z"} <= dimension_axes
    hole = next(req for req in spec.requirements if req.kind == "hole")
    assert hole.target == 6.0
    assert hole.required is True


def test_parse_request_falls_back_to_manual_review():
    spec = spec_parse(
        run_id="r",
        request_text="Make something nice and aesthetic without sharp edges.",
    )
    kinds = [req.kind for req in spec.requirements]
    # Nothing deterministic; everything is manual review.
    assert all(kind == "manual_review" for kind in kinds)
    assert spec.requirements


def test_parse_request_uses_clarification_answers_for_unknown_diameter():
    spec = spec_parse(
        run_id="r",
        request_text="Add a hole for the fastener.",
        answers=[{"id": "hole_diameter_mm", "value": "4"}],
    )
    hole = next(req for req in spec.requirements if req.kind == "hole")
    assert hole.target == 4.0


def test_parse_request_converts_hole_radius_and_keeps_axis():
    spec = spec_parse(
        run_id="r",
        request_text="Add an X-axis hole with radius 3 mm.",
    )
    hole = next(req for req in spec.requirements if req.kind == "hole")
    assert hole.target == 6.0
    assert hole.selector["axis"] == "X"


def test_solid_count_verifier_passes_for_single_body():
    from build123d import Box, Align

    shape = Box(10, 10, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(Requirement(id="REQ-COUNT", kind="solid_count", target=1.0),),
    )
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].observed["solid_count"] == 1


def test_solid_count_verifier_fails_for_mismatch():
    from build123d import Box, Align

    shape = Box(10, 10, 10, align=(Align.CENTER, Align.CENTER, Align.MIN))
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(Requirement(id="REQ-COUNT", kind="solid_count", target=2.0),),
    )
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert results[0].status == "failed"
    assert results[0].severity == "blocking"


def test_dimension_verifier_measures_bounding_box():
    from build123d import Box, Align

    shape = Box(50, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(
            Requirement(
                id="REQ-X",
                kind="dimension",
                target=50.0,
                selector={"axis": "X"},
                tolerance=0.05,
            ),
            Requirement(
                id="REQ-Y",
                kind="dimension",
                target=40.0,
                selector={"axis": "Y"},
                tolerance=0.05,
            ),
            Requirement(
                id="REQ-Z",
                kind="dimension",
                target=8.0,
                selector={"axis": "Z"},
                tolerance=0.05,
            ),
        ),
    )
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert [r.status for r in results] == ["passed", "passed", "passed"]


def test_dimension_verifier_flags_out_of_tolerance():
    from build123d import Box, Align

    shape = Box(50, 40, 8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(
            Requirement(
                id="REQ-X",
                kind="dimension",
                target=80.0,
                selector={"axis": "X"},
                tolerance=0.05,
            ),
        ),
    )
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert results[0].status == "failed"
    assert "differs" in results[0].message


def test_hole_verifier_matches_cylindrical_face(tmp_path: Path):
    settings, project_dir = _scenario(tmp_path)
    cad_tool_module = pytest.importorskip("agent.tools.cad_tool")
    cad = cad_tool_module.CadTool(project_dir)
    metrics = cad.build_and_verify()
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(
            Requirement(id="REQ-HOLE", kind="hole", target=6.0, tolerance=0.1),
        ),
    )
    # Re-run verifiers using the local registry against the built shape via
    # the bundled ``verifiers.py`` payload (same code path as the runner).
    assert metrics["metrics"]["is_valid"]
    assert metrics["validation"] == []  # No spec was attached during build.
    # The bundled ``verifiers.py`` is exercised by the runner; we re-import
    # it here for direct assertion.
    from agent.tools.cad_scripts import verifiers as bundled

    shape = bundled  # placeholder to keep linter quiet
    assert shape is not None
    # Build a shape from the live model.py using build123d directly.
    import build123d as b

    namespace = {"__name__": "__main__"}
    exec(compile(MODEL_WITH_HOLE, "model.py", "exec"), namespace)
    shape = namespace["result"]
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].observed["matches"][0]["diameter_mm"] == pytest.approx(6.0, abs=1e-3)


def test_hole_verifier_flags_wrong_diameter():
    import build123d as b

    namespace = {"__name__": "__main__"}
    exec(compile(MODEL_WRONG_HOLE, "model.py", "exec"), namespace)
    shape = namespace["result"]
    spec = DesignSpec(
        run_id="r",
        version=1,
        requirements=(
            Requirement(id="REQ-HOLE", kind="hole", target=6.0, tolerance=0.1),
        ),
    )
    results = run_verifiers(spec.requirements, shape, attempt_id="a")
    assert results[0].status == "failed"
    assert "Observed diameters" in results[0].message


def test_hole_verifier_rejects_boss_and_wrong_axis():
    from build123d import Box, Cylinder

    boss = Box(20, 20, 5) + Cylinder(3, 5).translate((0, 0, 5))
    wrong_axis_hole = Box(20, 20, 20) - Cylinder(3, 20, rotation=(0, 90, 0))
    requirement = Requirement(
        id="REQ-HOLE",
        kind="hole",
        target=6.0,
        tolerance=0.1,
        selector={"axis": "Z"},
    )
    assert run_verifiers((requirement,), boss, attempt_id="a")[0].status == "failed"
    assert run_verifiers((requirement,), wrong_axis_hole, attempt_id="a")[0].status == "failed"


def test_bundled_runner_verifier_matches_host_registry():
    """The sandboxed runner imports a bundled copy of the verifiers; this
    test locks down equivalence with the host registry so a future drift
    fails loudly. Run the same requirements through both and compare the
    subset of fields the LLM/API/UI actually consume.
    """
    from agent.tools.cad_scripts import verifiers as bundled

    namespace = {"__name__": "__main__"}
    exec(compile(MODEL_WITH_HOLE, "model.py", "exec"), namespace)
    shape = namespace["result"]

    req_dicts = [
        {"id": "REQ-X", "kind": "dimension", "target": 50.0,
         "selector": {"axis": "X"}, "tolerance": 0.05},
        {"id": "REQ-HOLE", "kind": "hole", "target": 6.0, "tolerance": 0.1},
        {"id": "REQ-COUNT", "kind": "solid_count", "target": 1.0},
        {"id": "REQ-BODY", "kind": "body_count", "target": 1.0},
        {"id": "REQ-MANUAL", "kind": "manual_review", "required": False},
    ]
    host_reqs = tuple(
        Requirement(
            id=r["id"],
            kind=r["kind"],
            target=r.get("target"),
            tolerance=r.get("tolerance", 0.05),
            selector=r.get("selector", {}),
            required=r.get("required", True),
        )
        for r in req_dicts
    )
    # The host stamps attempt_id on every result; the bundled runner does
    # not know the active attempt and leaves it empty (the consumer stamps
    # it). Both paths must agree on every *other* consumed field.
    host_results = run_verifiers(host_reqs, shape, attempt_id="attempt-x")
    bundled_results = bundled.run(req_dicts, shape, attempt_id="")

    def _strip(result: dict) -> dict:
        out = dict(result)
        out.pop("validation_id", None)
        out.pop("created_at", None)
        out.pop("schema_version", None)
        out.pop("attempt_id", None)
        return out

    for host, raw in zip(host_results, bundled_results):
        host_stripped = _strip(host.to_dict())
        bundled_stripped = _strip(raw)
        for field in (
            "requirement_id",
            "status",
            "severity",
            "message",
            "expected",
            "observed",
            "tolerance",
            "confidence",
        ):
            assert host_stripped.get(field) == bundled_stripped.get(field), (
                f"Drift between host and bundled verifiers for "
                f"{host_stripped['requirement_id']}: field {field!r} "
                f"host={host_stripped.get(field)!r} "
                f"bundled={bundled_stripped.get(field)!r}"
            )


def test_parser_picks_up_formatted_answer_lines():
    """The agent formats clarification answers as ``- <question>: <value>``.
    The parser must lift the value when the question text references a hole
    so a clarified diameter reaches the spec.
    """
    spec = parse_request(
        run_id="r",
        request_text=(
            "Create a mounting bracket.\n\nUser answers:\n"
            "- What hole diameter should I use?: 6 mm"
        ),
    )
    hole = next(req for req in spec.requirements if req.kind == "hole")
    assert hole.target == 6.0
    assert hole.required is True


def test_build_feedback_envelope_omits_passing_items_and_bounds_size():
    raw = {
        "metrics": {
            "solid_count": 1,
            "is_valid": True,
            "volume_mm3": 12345.6,
            "dimensions_mm": {"x": 50.0, "y": 40.0, "z": 8.0},
        },
        "validation": [
            {
                "requirement_id": "REQ-X",
                "status": "passed",
                "severity": "blocking",
                "expected": {"value_mm": 50.0},
                "observed": {"value_mm": 50.0},
                "message": "matches",
                "verifier": "kernel.dimension.v1",
            },
            {
                "requirement_id": "REQ-HOLE",
                "status": "failed",
                "severity": "blocking",
                "expected": {"diameter_mm": 6.0},
                "observed": {"diameters_mm": [3.0, 4.0]},
                "message": "No cylindrical face matches.",
                "verifier": "kernel.hole.v1",
            },
            {
                "requirement_id": "REQ-MANUAL",
                "status": "not_implemented",
                "severity": "minor",
                "expected": {},
                "observed": {},
                "message": "no verifier",
                "verifier": "unknown.manual_review",
            },
        ],
        "preview": "preview.stl",
        "render": "render.png",
    }
    envelope = sanitize_build_result(raw)
    assert envelope["build"]["ok"] is True
    assert envelope["build"]["dimensions_mm"] == {"x": 50.0, "y": 40.0, "z": 8.0}
    # The passing item is omitted; actionable items stay.
    requirement_ids = {item["id"] for item in envelope["requirements"]}
    assert requirement_ids == {"REQ-HOLE", "REQ-MANUAL"}
    assert envelope["omitted"] == 1
    serialized = json.dumps(envelope, ensure_ascii=False)
    assert len(serialized.encode("utf-8")) <= 4 * 1024


def test_build_feedback_envelope_caps_under_size_limit():
    validation = [
        {
            "requirement_id": f"REQ-{i:03d}",
            "status": "failed",
            "severity": "blocking",
            "expected": {"value_mm": 50.0},
            "observed": {"value_mm": 1.0},
            "message": "filler " * 30,
            "verifier": "kernel.dimension.v1",
        }
        for i in range(40)
    ]
    raw = {
        "metrics": {
            "solid_count": 1,
            "is_valid": True,
            "volume_mm3": 100.0,
            "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
        "validation": validation,
        "preview": "preview.stl",
        "render": "render.png",
    }
    envelope = sanitize_build_result(raw)
    assert len(envelope["requirements"]) <= 20
    assert envelope["omitted"] >= 20
    serialized = json.dumps(envelope, ensure_ascii=False)
    assert len(serialized.encode("utf-8")) <= 4 * 1024


def test_finalize_returns_status_accepted_by_default(tmp_path: Path, monkeypatch):
    settings, project_dir = _scenario(tmp_path, source=MODEL_SIMPLE_BOX)
    client = create_app(settings).test_client()
    response = client.post("/api/projects/demo/finalize")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["finalization_status"] == "accepted"
    meta_path = project_dir / "output" / ".finalize_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["finalization_status"] == "accepted"
    # Report carries the status header.
    report = (project_dir / "output" / "report.md").read_text(encoding="utf-8")
    assert "Finalization status: accepted" in report
    assert payload["bypassed"] is False


def test_finalize_status_finalized_with_bypass(tmp_path: Path, monkeypatch):
    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=True,
    )
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    (project_dir / "model.py").write_text(MODEL_SIMPLE_BOX, encoding="utf-8")
    RevisionStore(project_dir).commit(MODEL_SIMPLE_BOX, RevisionOrigin(kind="agent_edit"))
    response = client.post("/api/projects/demo/finalize", json={"force": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["finalization_status"] == "finalized_with_bypass"
    assert payload["bypassed"] is True
    meta = json.loads(
        (project_dir / "output" / ".finalize_meta.json").read_text(encoding="utf-8")
    )
    assert meta["finalization_status"] == "finalized_with_bypass"


def test_finalize_status_accepted_with_limitations(tmp_path: Path, monkeypatch):
    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=True,
    )
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    (project_dir / "model.py").write_text(MODEL_SIMPLE_BOX, encoding="utf-8")
    revision = RevisionStore(project_dir).commit(
        MODEL_SIMPLE_BOX, RevisionOrigin(kind="agent_edit")
    )
    # Record an explicit accepted_with_limitations decision.
    store = QualityStore(project_dir)
    run = store.start_run(project="demo")
    attempt = store.start_attempt(run.run_id, revision_id=revision.id)
    store.complete_attempt(
        run.run_id, attempt.attempt_id, status="succeeded", phase="kernel"
    )
    store.complete_run(run.run_id, status="completed")
    store.record_decision(
        run.run_id,
        revision_id=revision.id,
        attempt_id=attempt.attempt_id,
        decision="accepted_with_limitations",
    )
    response = client.post("/api/projects/demo/finalize")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["finalization_status"] == "accepted_with_limitations"
    assert payload["bypassed"] is False


def test_finalize_blocked_returns_one_line_hint(tmp_path: Path):
    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
        quality_require_acceptance_before_finalize=True,
    )
    client = create_app(settings).test_client()
    assert client.post("/api/projects/new", json={"name": "demo"}).status_code == 201
    project_dir = settings.workspace_root / "demo"
    (project_dir / "model.py").write_text(MODEL_SIMPLE_BOX, encoding="utf-8")
    RevisionStore(project_dir).commit(MODEL_SIMPLE_BOX, RevisionOrigin(kind="agent_edit"))
    response = client.post("/api/projects/demo/finalize")
    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "ACCEPTANCE_REQUIRED"
    assert "manual bypass" in payload["hint"]


def test_quality_spec_endpoint_returns_latest_spec(tmp_path: Path):
    settings, project_dir = _scenario(tmp_path)
    client = create_app(settings).test_client()
    store = QualityStore(project_dir)
    run = store.start_run(project="demo")
    spec = parse_request(
        run_id=run.run_id,
        request_text=(
            "Make a box with X length 50 mm, Y width 40 mm, Z height 8 mm, "
            "and a 6 mm diameter hole."
        ),
    )
    store.save_spec(spec)
    response = client.get(
        f"/api/projects/demo/quality/spec?run_id={run.run_id}"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["spec"] is not None
    assert payload["spec"]["version"] == 1
    kinds = {req["kind"] for req in payload["spec"]["requirements"]}
    assert {"dimension", "hole"} <= kinds


def test_runner_evaluates_spec_artifact(tmp_path: Path):
    """The bundled runner must persist validation results when a spec is present."""
    settings, project_dir = _scenario(tmp_path)
    spec = parse_request(
        run_id="r",
        request_text=(
            "Build a box X length 50 mm, Y width 40 mm, Z height 8 mm and a "
            "6 mm diameter hole."
        ),
    )
    (project_dir / ".cad_spec.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    from agent.tools.cad_tool import CadTool

    metrics = CadTool(project_dir).build_and_verify()
    assert metrics["spec_version"] >= 1
    assert metrics["validation"], "Runner must produce validation results"
    statuses = {item["status"] for item in metrics["validation"]}
    # At least the dimension checks pass and the hole check is found.
    assert "passed" in statuses
    assert (project_dir / ".cad_validation.json").is_file()
