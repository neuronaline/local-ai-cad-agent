"""Unit tests for ``cad_screenshot`` (subset renderer + orchestrator).

The orchestrator's sandbox path requires bubblewrap + libseccomp and is
covered by the acceptance suite. These tests focus on the contract-level
behaviour the agent relies on: schema validation, view/quality
normalisation, cache lookup, and the deterministic subset-render
primitives in the sandbox-side module.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_screenshot_tool import CadScreenshotTool

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


def _schema_for(name: str) -> dict:
    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return schema
    raise AssertionError(f"{name!r} not found in TOOL_SCHEMAS")


def test_cad_screenshot_schema_has_no_additional_properties():
    """OpenAI strict mode requires ``additionalProperties: False`` on every tool."""
    schema = _schema_for("cad_screenshot")["function"]["parameters"]
    assert schema["additionalProperties"] is False
    # The plan only exposes optional parameters; no required keys keep
    # every arg optional so the LLM can drop the call back to defaults.
    assert schema.get("required", []) == []
    properties = schema["properties"]
    assert set(properties.keys()) == {
        "views",
        "contact_sheet",
        "quality",
        "timeout_seconds",
    }


def test_cad_screenshot_views_enum_covers_canonical_eight():
    """The schema enum matches the orchestrator's SUBSET_VIEWS + orchestrator order."""
    schema = _schema_for("cad_screenshot")
    enum = schema["function"]["parameters"]["properties"]["views"]["items"]["enum"]
    assert enum == list(CadScreenshotTool.SUBSET_VIEWS)


def test_cad_screenshot_quality_enum_matches_subsystem():
    schema = _schema_for("cad_screenshot")
    enum = schema["function"]["parameters"]["properties"]["quality"]["enum"]
    assert tuple(sorted(enum)) == CadScreenshotTool.QUALITY_TIERS


# ---------------------------------------------------------------------------
# View / quality normalisation
# ---------------------------------------------------------------------------


def test_normalize_views_default_returns_canonical_eight():
    """``views=None`` and empty list both expand to the canonical eight."""
    assert CadScreenshotTool._normalize_views(None) == CadScreenshotTool.SUBSET_VIEWS
    assert CadScreenshotTool._normalize_views([]) == CadScreenshotTool.SUBSET_VIEWS


def test_normalize_views_deduplicates_preserving_order():
    requested = ["isometric_positive", "x_positive", "isometric_positive"]
    assert CadScreenshotTool._normalize_views(requested) == (
        "isometric_positive",
        "x_positive",
    )


def test_normalize_views_rejects_unknown_ids():
    with pytest.raises(ValueError, match="Unknown view id"):
        CadScreenshotTool._normalize_views(["nope"])


def test_normalize_quality_accepts_known_tiers():
    for quality in ("low", "standard", "high"):
        assert CadScreenshotTool._normalize_quality(quality) == quality


def test_normalize_quality_defaults_to_standard():
    assert CadScreenshotTool._normalize_quality(None) == "standard"
    assert CadScreenshotTool._normalize_quality("") == "standard"


def test_normalize_quality_rejects_unknown_tiers():
    with pytest.raises(ValueError, match="Unknown quality tier"):
        CadScreenshotTool._normalize_quality("ultra")


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------


def _write_png(path: Path, payload: bytes = b"\x89PNG\r\n\x1a\n fake") -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _seed_review_dir(project_dir: Path, *, model_sha: str, views: list[str], quality: str = "standard") -> dict:
    review_dir = project_dir / ".cad-agent" / "reviews" / model_sha
    review_dir.mkdir(parents=True, exist_ok=True)
    views_dir = review_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for view_id in views:
        png_path = views_dir / f"{view_id}.png"
        sha = _write_png(png_path)
        entries.append(
            {
                "view_id": view_id,
                "label": view_id,
                "image_sha256": sha,
                "image_bytes": len(b"\x89PNG\r\n\x1a\n fake"),
                "width": 512,
                "height": 512,
            }
        )
    sheet_path = review_dir / "review-sheet.png"
    sheet_sha = _write_png(sheet_path)
    manifest = {
        "model_sha256": model_sha,
        "preview_sha256": "b" * 64,
        "views": entries,
        "quality": quality,
        "contact_sheet": {"image_sha256": sheet_sha},
    }
    (review_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def test_cache_lookup_matches_when_subset_already_built(tmp_path: Path):
    """A matching ``(model_sha, sorted(views), quality)`` tuple hits the cache."""
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    _seed_review_dir(project, model_sha=model_sha, views=["x_positive", "isometric_positive"])
    tool = CadScreenshotTool(project, publish=None)
    assert tool._cache_matches(
        tool._review_dir(model_sha), model_sha,
        ("x_positive", "isometric_positive"), "standard",
    ) is True


def test_cache_lookup_misses_when_view_missing(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    _seed_review_dir(project, model_sha=model_sha, views=["x_positive"])
    tool = CadScreenshotTool(project, publish=None)
    assert tool._cache_matches(
        tool._review_dir(model_sha), model_sha,
        ("x_positive", "y_positive"), "standard",
    ) is False


def test_cache_lookup_misses_on_quality_mismatch(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    _seed_review_dir(project, model_sha=model_sha, views=["x_positive"], quality="standard")
    tool = CadScreenshotTool(project, publish=None)
    assert tool._cache_matches(
        tool._review_dir(model_sha), model_sha,
        ("x_positive",), "high",
    ) is False


def test_cache_lookup_misses_when_contact_sheet_is_for_a_different_subset(tmp_path: Path):
    """A subset sheet must not be served as evidence for a wider request."""
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    _seed_review_dir(project, model_sha=model_sha, views=["x_positive"])
    tool = CadScreenshotTool(project, publish=None)
    assert tool._cache_matches(
        tool._review_dir(model_sha), model_sha,
        ("x_positive", "y_positive"), "standard",
    ) is False


def test_cache_lookup_misses_on_hash_drift(tmp_path: Path):
    """The cache refuses to serve a PNG whose SHA-256 no longer matches the manifest."""
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    _seed_review_dir(project, model_sha=model_sha, views=["x_positive"])
    review_dir = project / ".cad-agent" / "reviews" / model_sha
    # Overwrite the PNG with different bytes after seeding.
    (review_dir / "views" / "x_positive.png").write_bytes(b"different bytes")
    tool = CadScreenshotTool(project, publish=None)
    assert tool._cache_matches(
        tool._review_dir(model_sha), model_sha,
        ("x_positive",), "standard",
    ) is False


def test_cache_lookup_rejects_malformed_requested_views(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    model_sha = "a" * 64
    manifest = _seed_review_dir(project, model_sha=model_sha, views=["x_positive"])
    manifest["requested_views"] = [{}]
    (project / ".cad-agent" / "reviews" / model_sha / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert not CadScreenshotTool(project)._cache_matches(
        CadScreenshotTool(project)._review_dir(model_sha), model_sha,
        ("x_positive",), "standard",
    )


# ---------------------------------------------------------------------------
# Sandbox-side subset renderer primitives
# ---------------------------------------------------------------------------


def test_subset_views_matches_orchestrator_constant():
    """The sandbox-side SUBSET_VIEWS is the same tuple the orchestrator exposes."""
    sys.path.insert(0, str(Path("agent/tools/cad_scripts").resolve()))
    from screenshot import QUALITY_TOLERANCES, SUBSET_VIEWS  # type: ignore

    assert SUBSET_VIEWS == CadScreenshotTool.SUBSET_VIEWS
    assert set(QUALITY_TOLERANCES.keys()) == set(CadScreenshotTool.QUALITY_TIERS)


def test_subset_view_spec_resolves_known_ids():
    """Subset resolution returns matching ``ViewSpec`` entries for known ids."""
    sys.path.insert(0, str(Path("agent/tools/cad_scripts").resolve()))
    import renderer as _renderer  # type: ignore
    import screenshot as _screenshot  # type: ignore

    _screenshot.VIEWS = _renderer.VIEWS  # inject the canonical view specs
    spec = _screenshot._view_spec_by_id("isometric_positive")
    assert spec.view_id == "isometric_positive"


def test_subset_view_spec_rejects_unknown_ids():
    sys.path.insert(0, str(Path("agent/tools/cad_scripts").resolve()))
    import renderer as _renderer  # type: ignore
    import screenshot as _screenshot  # type: ignore

    _screenshot.VIEWS = _renderer.VIEWS
    with pytest.raises(ValueError, match="Unknown view id"):
        _screenshot._view_spec_by_id("not_a_real_view")


# ---------------------------------------------------------------------------
# Cache hit fast-path in execute() (no sandbox)
# ---------------------------------------------------------------------------


def test_execute_returns_cache_hit_without_subprocess(tmp_path: Path):
    """When the cache is primed the orchestrator returns immediately, no bwrap."""
    project = tmp_path / "demo"
    project.mkdir()
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    (project / "preview.stl").write_bytes(b"preview bytes")
    preview_sha = hashlib.sha256(b"preview bytes").hexdigest()
    model_sha = hashlib.sha256(b"result = 1\n").hexdigest()
    review_dir = project / ".cad-agent" / "reviews" / model_sha
    views = ("x_positive", "isometric_positive")
    view_shas = {}
    for view_id in views:
        png = review_dir / "views" / f"{view_id}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"\x89PNG\r\n\x1a\n fake")
        view_shas[view_id] = hashlib.sha256(png.read_bytes()).hexdigest()
    manifest = {
        "model_sha256": model_sha,
        "preview_sha256": preview_sha,
        "views": [
            {
                "view_id": vid,
                "label": vid,
                "image_sha256": view_shas[vid],
                "image_bytes": 12,
                "width": 512,
                "height": 512,
            }
            for vid in views
        ],
        "quality": "standard",
        "contact_sheet": {
            "image_sha256": _write_png(review_dir / "review-sheet.png"),
        },
    }
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    published = []
    tool = CadScreenshotTool(project, publish=lambda kind, data, **_: published.append((kind, data)))
    with patch(
        "agent.tools.cad_screenshot_tool.CadScreenshotTool._run_sandbox"
    ) as fake_run:
        result = tool.execute({"views": list(views), "quality": "standard"})
    assert fake_run.call_count == 0, "cache hit must not spawn the sandbox"
    assert result["cache_hit"] is True
    assert result["manifest"]["quality"] == "standard"
    view_ids = [entry["view_id"] for entry in result["manifest"]["views"]]
    assert view_ids == list(views)
    assert any(kind == "preview_updated" for kind, _ in published)
    assert any(kind == "screenshot_updated" for kind, _ in published)


def test_execute_rejects_unknown_view_before_sandbox(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    (project / "model.py").write_text("result = 1\n", encoding="utf-8")
    tool = CadScreenshotTool(project, publish=None)
    with pytest.raises(ValueError, match="Unknown view id"):
        tool.execute({"views": ["bogus_view"]})


def test_promote_replaces_an_incompatible_subset_and_cleans_staging(tmp_path: Path):
    """A new subset cannot leave old view entries behind in its manifest."""
    project = tmp_path / "demo"
    project.mkdir()
    tool = CadScreenshotTool(project)
    model_sha = "a" * 64
    review_dir = tool._review_dir(model_sha)
    _seed_review_dir(
        project, model_sha=model_sha, views=["x_positive", "y_positive"]
    )
    staging = tmp_path / "staging"
    (staging / "views").mkdir(parents=True)
    view_path = staging / "views" / "isometric_positive.png"
    view_sha = _write_png(view_path)
    sheet_sha = _write_png(staging / "review-sheet.png")

    tool._promote_to_cache(
        review_dir,
        {
            "_staging_dir": str(staging),
            "model_sha256": model_sha,
            "preview_sha256": "",
            "requested_views": ["isometric_positive"],
            "views": [
                {
                    "view_id": "isometric_positive",
                    "image_sha256": view_sha,
                    "image_bytes": view_path.stat().st_size,
                }
            ],
            "contact_sheet": {"image_sha256": sheet_sha},
            "single_render": {
                "path": "views/isometric_positive.png",
                "image_sha256": view_sha,
            },
            "quality": "high",
        },
    )

    manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["view_id"] for entry in manifest["views"]] == ["isometric_positive"]
    assert not staging.exists()
