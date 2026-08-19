"""Tests for the structured multimodal reviewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent.cad_review import (
    REVIEW_TOOL_SCHEMA,
    Finding,
    ReviewResult,
    _visual_review,
    parse_review_response,
    write_review_result,
)

MODEL_SHA = "a" * 64
PREVIEW_SHA = "b" * 64
ALLOWED_VIEWS = ("x_positive", "x_negative", "y_positive", "isometric_positive")
IMAGE_BYTES = b"\x89PNG\r\n\x1a\n fake-image"


def _write_image(path: Path) -> dict[str, object]:
    path.write_bytes(IMAGE_BYTES)
    return {
        "path": path.name,
        "width": 512,
        "height": 512,
        "image_sha256": hashlib.sha256(IMAGE_BYTES).hexdigest(),
    }


def _valid_arguments() -> dict[str, object]:
    return {
        "status": "pass",
        "summary": "All checks passed.",
        "findings": [
            {
                "severity": "minor",
                "category": "geometry",
                "view": "x_positive",
                "message": "Slight chamfer missing on the top edge.",
                "repair_hint": "Add a 0.5mm chamfer.",
            }
        ],
    }


def test_review_tool_schema_has_no_additional_properties():
    schema = REVIEW_TOOL_SCHEMA["function"]["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "summary", "findings"}


def test_parse_review_response_handles_pass_with_minor_finding():
    result = parse_review_response(
        json.dumps(_valid_arguments()),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert result.status == "pass"
    assert result.summary  # summary is forwarded as-is
    assert len(result.findings) == 1
    assert result.findings[0].severity == "minor"
    assert result.findings[0].view == "x_positive"


def test_parse_review_response_reclassifies_pass_with_blocking_finding():
    arguments = _valid_arguments()
    arguments["status"] = "pass"
    arguments["findings"] = [
        {
            "severity": "blocking",
            "category": "missing_feature",
            "view": "z_positive",
            "message": "Through hole is missing.",
            "repair_hint": "Subtract a cylinder.",
        }
    ]
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    # Incoherent: pass with a blocking finding is treated as fail.
    assert result.status == "fail"


def test_parse_review_response_drops_unknown_severity():
    arguments = _valid_arguments()
    arguments["findings"] = [
        {"severity": "catastrophic", "category": "geometry", "message": "Bad"},
        {
            "severity": "minor",
            "category": "dimensions",
            "message": "Too small.",
        },
    ]
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert result.status == "pass"
    assert len(result.findings) == 1
    assert result.findings[0].severity == "minor"


def test_parse_review_response_drops_unknown_view_id():
    arguments = _valid_arguments()
    arguments["findings"] = [
        {
            "severity": "major",
            "category": "alignment",
            "view": "does_not_exist",
            "message": "Off-axis hole.",
        }
    ]
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert result.findings[0].view is None


def test_parse_review_response_truncates_oversized_strings():
    arguments = _valid_arguments()
    arguments["summary"] = "x" * 500
    arguments["findings"] = [
        {
            "severity": "minor",
            "category": "geometry",
            "message": "m" * 1000,
            "repair_hint": "h" * 1000,
        }
    ]
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert len(result.summary) <= 160
    assert len(result.findings[0].message) <= 240
    assert len(result.findings[0].repair_hint) <= 240


def test_parse_review_response_rejects_malformed_payload():
    result = parse_review_response(
        "not json",
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert result.status == "inconclusive"


def test_parse_review_response_rejects_unknown_status():
    arguments = _valid_arguments()
    arguments["status"] = "approved"
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert result.status == "inconclusive"


def test_parse_review_response_caps_finding_count():
    arguments = _valid_arguments()
    arguments["findings"] = [
        {"severity": "minor", "category": "geometry", "message": f"m{i}"}
        for i in range(50)
    ]
    result = parse_review_response(
        json.dumps(arguments),
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
        allowed_view_ids=ALLOWED_VIEWS,
    )
    assert len(result.findings) <= 12


def test_write_review_result_round_trip(tmp_path: Path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    result = ReviewResult(
        status="pass",
        summary="All checks passed.",
        findings=[
            Finding(
                severity="minor",
                category="geometry",
                message="Slight chamfer missing.",
                view="x_positive",
            )
        ],
        model_sha256=MODEL_SHA,
        preview_sha256=PREVIEW_SHA,
    )
    path = write_review_result(review_dir, result)
    assert path == review_dir / "result.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "pass"
    assert payload["findings"][0]["category"] == "geometry"


class _FakeClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def chat(self, messages, tools=None):
        return self.response


def test_visual_review_returns_pass_for_valid_tool_call(tmp_path: Path):
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet_info = _write_image(sheet)
    render_info = _write_image(render)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [
            {"view_id": "x_positive"},
            {"view_id": "isometric_positive"},
        ],
        "contact_sheet": sheet_info,
        "single_render": render_info,
    }
    sent_image_parts: list[int] = []
    evidence_labels: list[list[str]] = []

    class _RecordingClient:
        def chat(self, messages, tools=None):
            image_parts = [
                part
                for message in messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
            sent_image_parts.append(len(image_parts))
            evidence_labels.append(
                [
                    part["text"]
                    for part in messages[-1]["content"]
                    if part.get("type") == "text"
                    and part.get("text", "").startswith("Visual evidence")
                ]
            )
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "function": {
                    "name": "submit_review",
                    "arguments": json.dumps(_valid_arguments()),
                }
            }]}}]}

    result = _visual_review(
        settings=object(),
        request_text="Build a 60x40 bracket.",
        model_source="result = 1",
        metrics={"solid_count": 1, "dimensions_mm": {"x": 60, "y": 40, "z": 8}},
        feature_summary={"through_hole_count": 1},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
        create_client=lambda _settings: _RecordingClient(),
    )
    assert result.status == "pass"
    # The reviewer receives independently-rendered isometric and multi-view
    # evidence; it must not silently retry without either image.
    assert sent_image_parts == [2]
    assert evidence_labels == [[
        "Visual evidence 1 of 2: single isometric render.",
        "Visual evidence 2 of 2: multi-view contact sheet.",
    ]]


def test_visual_review_returns_inconclusive_when_tool_call_missing(tmp_path: Path):
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet_info = _write_image(sheet)
    render_info = _write_image(render)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": sheet_info,
        "single_render": render_info,
    }
    client = _FakeClient({"choices": [{"message": {"role": "assistant", "content": "looks good"}}]})
    def create_client(_settings):
        return client
    result = _visual_review(
        settings=object(),
        request_text="Build",
        model_source="result = 1",
        metrics={},
        feature_summary={},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
        create_client=create_client,
    )
    assert result.status == "inconclusive"


def test_visual_review_returns_inconclusive_when_sheet_missing(tmp_path: Path):
    render = tmp_path / "render.png"
    render_info = _write_image(render)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": {"path": "review-sheet.png", "image_sha256": "a" * 64},
        "single_render": render_info,
    }
    client = _FakeClient({"choices": [{"message": {"role": "assistant", "content": ""}}]})
    result = _visual_review(
        settings=object(),
        request_text="Build",
        model_source="",
        metrics={},
        feature_summary={},
        review_manifest=manifest,
        sheet_path=tmp_path / "missing.png",
        single_render_path=render,
        create_client=lambda _settings: client,
    )
    assert result.status == "inconclusive"


def test_visual_review_requires_a_matching_single_render(tmp_path: Path):
    sheet = tmp_path / "sheet.png"
    sheet_info = _write_image(sheet)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": sheet_info,
        "single_render": {
            "path": "render.png",
            "image_sha256": "a" * 64,
        },
    }
    client = _FakeClient({})

    result = _visual_review(
        settings=object(),
        request_text="Build",
        model_source="",
        metrics={},
        feature_summary={},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=tmp_path / "missing-render.png",
        create_client=lambda _settings: client,
    )

    assert result.status == "inconclusive"


def test_visual_review_propagates_llm_errors_as_inconclusive(tmp_path: Path):
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet_info = _write_image(sheet)
    render_info = _write_image(render)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": sheet_info,
        "single_render": render_info,
    }

    class FailingClient:
        require_images = True

        def chat(self, messages, tools=None):
            raise RuntimeError("network down")

    def create_client(_settings):
        return FailingClient()
    result = _visual_review(
        settings=object(),
        request_text="Build",
        model_source="",
        metrics={},
        feature_summary={},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
        create_client=create_client,
    )
    assert result.status == "inconclusive"


def test_visual_review_uses_stop_event_when_provided(tmp_path: Path):
    sheet = tmp_path / "sheet.png"
    render = tmp_path / "render.png"
    sheet_info = _write_image(sheet)
    render_info = _write_image(render)
    manifest = {
        "model_sha256": MODEL_SHA,
        "preview_sha256": PREVIEW_SHA,
        "views": [{"view_id": "x_positive"}],
        "contact_sheet": sheet_info,
        "single_render": render_info,
    }
    stop_event = object()
    captured: dict[str, object] = {}

    class StopAwareClient:
        def chat(self, messages, tools=None):
            captured["stop_event"] = self.stop_event
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "function": {"name": "submit_review", "arguments": json.dumps(_valid_arguments())}
            }]}}]}

    def create_client(_settings):
        return StopAwareClient()
    _visual_review(
        settings=object(),
        request_text="Build",
        model_source="",
        metrics={},
        feature_summary={},
        review_manifest=manifest,
        sheet_path=sheet,
        single_render_path=render,
        stop_event=stop_event,
        create_client=create_client,
    )
    assert captured["stop_event"] is stop_event
