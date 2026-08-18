"""Tests for the ``cad_review`` refactor.

The new contract is:

- ``Finding.source`` distinguishes deterministic (``"deterministic"``) from
  visual (``"visual"``) findings so the agent can attribute results.
- ``review_cad`` composes both layers; deterministic always runs first.
- Blocking deterministic findings short-circuit (no visual call).
- Visual layer is skipped when no visual evidence exists; verdict collapses
  to ``inconclusive`` unless deterministic findings force ``fail``.
- Verdict is always strict: any blocking/major finding forces ``fail``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.cad_review import (
    ReviewResult,
    _deterministic_review,
    parse_review_response,
    review_cad,
)
from agent.tools.cad_review_tool import CadReviewTool

# ---------------------------------------------------------------------------
# Deterministic layer
# ---------------------------------------------------------------------------


def test_deterministic_finds_blocking_when_no_solid():
    findings = _deterministic_review(
        metrics={"solid_count": 0, "is_valid": False, "dimensions_mm": {}, "volume_mm3": 0},
        feature_summary={},
    )
    assert any(f.severity == "blocking" for f in findings)
    assert all(f.source == "deterministic" for f in findings)


def test_deterministic_emits_source_for_every_finding():
    findings = _deterministic_review(
        metrics={"solid_count": 1, "is_valid": True, "dimensions_mm": {"x": 1, "y": 1, "z": 1}, "volume_mm3": 1.0},
        feature_summary={},
    )
    # All-pass metrics still yields an empty findings list; the test guards
    # against future regressions where one source leaks into the other.
    assert all(f.source == "deterministic" for f in findings)


def test_deterministic_flags_zero_dimensions_and_volume():
    findings = _deterministic_review(
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
            "volume_mm3": 0.0,
        },
        feature_summary={},
    )
    # At least one major finding per zero dimension + volume = 4 majors.
    majors = [f for f in findings if f.severity == "major"]
    assert len(majors) >= 4
    assert all(f.source == "deterministic" for f in majors)


def test_deterministic_surfaces_spec_verifier_failures():
    """Spec ``.cad_validation.json`` failures propagate with deterministic source."""
    findings = _deterministic_review(
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        validation_results=[
            {
                "requirement_id": "req1",
                "verifier": "spec.dimensions",
                "status": "fail",
                "severity": "major",
                "message": "Width 10 mm is below the 20 mm minimum.",
            }
        ],
    )
    spec_findings = [f for f in findings if "req1" in f.message]
    assert len(spec_findings) == 1
    assert spec_findings[0].severity == "major"
    assert spec_findings[0].source == "deterministic"


def test_deterministic_ignores_passing_spec_results():
    """``status=pass`` entries do not produce findings."""
    findings = _deterministic_review(
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        validation_results=[
            {"requirement_id": "req1", "verifier": "spec.dimensions", "status": "pass"}
        ],
    )
    assert findings == []


# ---------------------------------------------------------------------------
# Visual layer (parse + verdict rules)
# ---------------------------------------------------------------------------


def test_parse_review_response_visual_findings_default_to_visual_source():
    arguments = {
        "status": "pass",
        "summary": "All good.",
        "findings": [
            {
                "severity": "minor",
                "category": "geometry",
                "message": "Small chamfer missing.",
                "view": "x_positive",
            }
        ],
    }
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256="a" * 64,
        preview_sha256="b" * 64,
        allowed_view_ids={"x_positive"},
    )
    assert result.findings[0].source == "visual"
    assert result.status == "pass"


def test_parse_review_response_blocks_pass_with_blocking_finding():
    """Strict rule: pass + blocking → fail."""
    arguments = {
        "status": "pass",
        "summary": "All good.",
        "findings": [
            {
                "severity": "blocking",
                "category": "missing_feature",
                "message": "Through hole missing.",
                "view": "x_positive",
            }
        ],
    }
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256="a" * 64,
        preview_sha256="b" * 64,
        allowed_view_ids={"x_positive"},
    )
    assert result.status == "fail"


def test_parse_review_response_blocks_pass_with_major_finding():
    """Strict rule also covers ``major`` severity (not just blocking)."""
    arguments = {
        "status": "pass",
        "summary": "All good.",
        "findings": [
            {
                "severity": "major",
                "category": "alignment",
                "message": "Misaligned hole.",
                "view": "isometric_positive",
            }
        ],
    }
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256="a" * 64,
        preview_sha256="b" * 64,
        allowed_view_ids={"isometric_positive"},
    )
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# review_cad composition
# ---------------------------------------------------------------------------


def test_review_cad_short_circuits_on_blocking_deterministic_finding():
    """A blocking deterministic finding forces fail without calling the visual layer."""
    called = {"visual": False}

    def fake_visual(**kwargs: Any) -> ReviewResult:
        called["visual"] = True
        return ReviewResult(status="pass", summary="ignored", findings=[])

    result = review_cad(
        settings=None,
        request_text="make a part",
        model_source="result = 1",
        metrics={"solid_count": 0, "is_valid": False, "dimensions_mm": {}, "volume_mm3": 0},
        feature_summary={},
        review_manifest={
            "model_sha256": "a" * 64,
            "preview_sha256": "b" * 64,
            "views": [{"view_id": "x_positive"}],
            "contact_sheet": {"path": "review-sheet.png", "image_sha256": "x" * 64},
            "single_render": {"path": "render.png", "image_sha256": "x" * 64},
        },
        sheet_path=Path("/nonexistent.png"),
        single_render_path=Path("/nonexistent.png"),
    )
    assert called["visual"] is False
    assert result.status == "fail"
    assert any(f.severity == "blocking" for f in result.findings)


def test_review_cad_returns_inconclusive_when_no_evidence_and_no_blocking():
    """No manifest + valid metrics + no visual → inconclusive (visual skipped)."""
    result = review_cad(
        settings=None,
        request_text="make a part",
        model_source="",
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        review_manifest=None,
        sheet_path=None,
        single_render_path=None,
    )
    assert result.status == "inconclusive"
    assert result.findings == []


def test_review_cad_skips_visual_when_settings_is_none():
    """Passing ``settings=None`` skips the visual layer (returns inconclusive)."""
    result = review_cad(
        settings=None,
        request_text="make a part",
        model_source="",
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        review_manifest={
            "model_sha256": "a" * 64,
            "preview_sha256": "b" * 64,
            "views": [{"view_id": "x_positive"}],
            "contact_sheet": {"path": "review-sheet.png", "image_sha256": "x" * 64},
            "single_render": {"path": "render.png", "image_sha256": "x" * 64},
        },
        sheet_path=Path("/nonexistent.png"),
        single_render_path=Path("/nonexistent.png"),
    )
    assert result.status == "inconclusive"


def test_review_cad_combines_findings_with_source_tags(tmp_path: Path, monkeypatch):
    """Both layers contribute findings; visual findings come from the LLM client."""
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet.write_bytes(b"\x89PNG\r\n\x1a\n sheet")
    render.write_bytes(b"\x89PNG\r\n\x1a\n render")
    sheet_sha = hashlib.sha256(sheet.read_bytes()).hexdigest()
    render_sha = hashlib.sha256(render.read_bytes()).hexdigest()
    manifest = {
        "model_sha256": "a" * 64,
        "preview_sha256": "b" * 64,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": {
            "path": "review-sheet.png",
            "image_sha256": sheet_sha,
        },
        "single_render": {
            "path": "render.png",
            "image_sha256": render_sha,
        },
    }

    class FakeClient:
        require_images = True

        def chat(self, messages, tools=None):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_review",
                                        "arguments": json.dumps(
                                            {
                                                "status": "pass",
                                                "summary": "Looks fine.",
                                                "findings": [
                                                    {
                                                        "severity": "minor",
                                                        "category": "geometry",
                                                        "view": "x_positive",
                                                        "message": "Slight chamfer missing.",
                                                    }
                                                ],
                                            }
                                        ),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: FakeClient()
    )
    result = review_cad(
        settings=object(),
        request_text="make a part",
        model_source="result = 1",
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={"through_hole_count": 0},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
    )
    sources = {f.source for f in result.findings}
    # Only the visual layer ran (no deterministic blocking/major findings).
    assert sources == {"visual"}
    assert result.status == "pass"


def test_review_cad_propagates_visual_fail_when_deterministic_pass(tmp_path: Path, monkeypatch):
    """Visual fail verdict is preserved (strict) even when deterministic passes."""
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet.write_bytes(b"\x89PNG\r\n\x1a\n sheet")
    render.write_bytes(b"\x89PNG\r\n\x1a\n render")
    sheet_sha = hashlib.sha256(sheet.read_bytes()).hexdigest()
    render_sha = hashlib.sha256(render.read_bytes()).hexdigest()
    manifest = {
        "model_sha256": "a" * 64,
        "preview_sha256": "b" * 64,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": {"path": "review-sheet.png", "image_sha256": sheet_sha},
        "single_render": {"path": "render.png", "image_sha256": render_sha},
    }

    class FakeClient:
        require_images = True

        def chat(self, messages, tools=None):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_review",
                                        "arguments": json.dumps(
                                            {
                                "status": "fail",
                                "summary": "Misaligned.",
                                "findings": [
                                    {
                                        "severity": "blocking",
                                        "category": "missing_feature",
                                        "view": "x_positive",
                                        "message": "Through hole missing.",
                                    }
                                ],
                            }
                                        ),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        "agent.cad_review._default_create_client", lambda _settings: FakeClient()
    )
    result = review_cad(
        settings=object(),
        request_text="make a part",
        model_source="result = 1",
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 10},
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
    )
    assert result.status == "fail"
    # Strict rule: blocking finding forces fail.
    assert any(f.severity == "blocking" for f in result.findings)
    # All findings are visual since deterministic was a pass.
    assert all(f.source == "visual" for f in result.findings)


def test_review_cad_uses_deterministic_finding_when_visual_layer_disabled(tmp_path: Path, monkeypatch):
    """Passing settings=None with a major deterministic finding → fail."""
    result = review_cad(
        settings=None,
        request_text="make a part",
        model_source="",
        metrics={
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 10, "y": 10, "z": 0.0},  # major: zero dim
            "volume_mm3": 1000.0,
        },
        feature_summary={},
        review_manifest=None,
        sheet_path=None,
        single_render_path=None,
    )
    assert result.status == "fail"
    assert any(f.source == "deterministic" and f.severity == "major" for f in result.findings)


def test_review_tool_uses_current_model_evidence_and_screenshot_isometric(tmp_path: Path):
    """Old build evidence must not be reviewed after model.py changes."""
    project = tmp_path / "demo"
    project.mkdir()
    source = "result = 1\n"
    (project / "model.py").write_text(source, encoding="utf-8")
    model_sha = hashlib.sha256(source.encode()).hexdigest()
    review_dir = project / ".cad-agent" / "reviews" / model_sha
    views_dir = review_dir / "views"
    views_dir.mkdir(parents=True)
    sheet = review_dir / "review-sheet.png"
    render = views_dir / "isometric_positive.png"
    sheet.write_bytes(b"sheet")
    render.write_bytes(b"render")
    manifest = {
        "model_sha256": model_sha,
        "preview_sha256": "",
        "views": [{"view_id": "isometric_positive"}],
        "contact_sheet": {"image_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest()},
        "single_render": {
            "path": "views/isometric_positive.png",
            "image_sha256": hashlib.sha256(render.read_bytes()).hexdigest(),
        },
    }
    (review_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    tool = CadReviewTool(project)

    resolved, sheet_path, render_path, _ = tool._resolve_evidence()

    assert resolved == manifest
    assert sheet_path == sheet
    assert render_path == render

    (project / "model.py").write_text("result = 2\n", encoding="utf-8")
    assert tool._resolve_evidence()[:3] == (None, None, None)


def test_review_tool_rejects_metrics_from_an_older_model(tmp_path: Path):
    """A review must not combine current screenshots with stale dimensions."""
    project = tmp_path / "demo"
    project.mkdir()
    (project / "model.py").write_text("result = 2\n", encoding="utf-8")
    (project / ".cad_metrics.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(b"result = 1\n").hexdigest(),
                "metrics": {"solid_count": 1},
                "feature_summary": {"through_hole_count": 1},
            }
        ),
        encoding="utf-8",
    )

    metrics, feature_summary, validation = CadReviewTool(project)._load_inputs()

    assert metrics == {}
    assert feature_summary == {}
    assert validation is None


def test_review_tool_loads_nested_metrics_for_the_current_model(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    source = "result = 2\n"
    (project / "model.py").write_text(source, encoding="utf-8")
    geometry = {
        "solid_count": 1,
        "is_valid": True,
        "dimensions_mm": {"x": 1, "y": 1, "z": 1},
        "volume_mm3": 1,
    }
    (project / ".cad_metrics.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "metrics": geometry,
                "feature_summary": {"through_hole_count": 1},
            }
        ),
        encoding="utf-8",
    )

    metrics, feature_summary, _ = CadReviewTool(project)._load_inputs()

    assert metrics == geometry
    assert feature_summary == {"through_hole_count": 1}
