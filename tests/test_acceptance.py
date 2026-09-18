import json
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

if not shutil.which("openscad"):
    pytest.skip("openscad not installed", allow_module_level=True)

from agent.settings import Settings
from app import create_app

MODEL_CODE = """// Parametric mounting bracket
WIDTH = 60.0;
LENGTH = 40.0;
HEIGHT = 8.0;
HOLE_DIAMETER = 6.0;
$fn = 60;
EPS = 0.01;

difference() {
    cube([WIDTH, LENGTH, HEIGHT], center=true);
    cylinder(d=HOLE_DIAMETER, h=HEIGHT + EPS, center=true);
}
"""


def test_cylindrical_feature_extraction_handles_scad():
    """Cutout feature extraction extracts cylinder cutouts in difference blocks."""
    from agent.tools.cad_scripts.runner import _extract_scad_features

    scad = "difference() { cube([10, 10, 10]); cylinder(d=5, h=12); }"
    features = _extract_scad_features(scad, {"x": 10, "y": 10, "z": 10})
    assert len(features.get("cutouts", [])) == 1
    assert features["cutouts"][0]["radius"] == 2.5

    # Parametric variables and expressions
    features_parametric = _extract_scad_features(MODEL_CODE, {"x": 60, "y": 40, "z": 8})
    assert len(features_parametric.get("cutouts", [])) == 1
    assert features_parametric["cutouts"][0]["radius"] == 3.0
    assert features_parametric["through_hole_count"] == 1


class QuestionClient:
    def chat(self, _messages, _tools):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "question-1",
                                "function": {
                                    "name": "question",
                                    "arguments": (
                                        '{"question":"What hole diameter should I use?","input_type":"number"}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }


class BuildClient:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def chat(self, messages, _tools):
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "write-1",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": (
                                            json.dumps(
                                                {
                                                    "filename": "model.scad",
                                                    "content": MODEL_CODE,
                                                }
                                            )
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if self.calls == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "run-1",
                                    "function": {
                                        "name": "cad_build_and_verify",
                                        "arguments": json.dumps({"render": True}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The bracket is ready for finalization.",
                    }
                }
            ]
        }


def test_mvp_acceptance_flow(tmp_path: Path, monkeypatch):
    import agent.core
    from agent.revisions import RevisionStore

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    question_client = QuestionClient()
    build_client = BuildClient()

    # The agent asks one clarification first (using question_client),
    # then build_client handles every subsequent turn (file write, build, final answer).
    def factory(_settings):
        if not getattr(factory, "_seen", False):
            factory._seen = True
            return question_client
        return build_client

    factory._seen = False
    monkeypatch.setattr(agent.core, "create_llm_client", factory)
    app = create_app(settings)
    client = app.test_client()
    assert (
        client.post("/api/projects/new", json={"name": "mounting-bracket"}).status_code
        == 201
    )

    sketch = BytesIO()
    Image.new("RGB", (40, 30), "white").save(sketch, format="PNG")
    sketch.seek(0)
    response = client.post(
        "/api/chat",
        data={
            "project": "mounting-bracket",
            "message": "Create a mounting bracket from this sketch.",
            "attachments": (sketch, "bracket.png"),
        },
        headers={"Origin": "http://localhost:5000"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    runner = app.config["AGENT_RUNNER"]
    runner._thread.join(timeout=3)
    waiting = runner.waiting_question("mounting-bracket")
    assert isinstance(waiting, dict)
    assert len(waiting["questions"]) == 1
    assert waiting["questions"][0]["question"] == "What hole diameter should I use?"
    assert waiting["questions"][0]["input_type"] == "number"

    response = client.post(
        "/api/questions/answer", json={"project": "mounting-bracket", "answer": "6 mm"}
    )
    assert response.status_code == 202
    runner._thread.join(timeout=15)
    project = settings.workspace_root / "mounting-bracket"
    scad_file = project / "model.scad"
    stl_file = project / "preview.stl"
    png_file = project / "render.png"
    metrics_file = project / ".cad_metrics.json"

    assert scad_file.is_file()
    assert "difference()" in scad_file.read_text(encoding="utf-8")
    assert stl_file.is_file() and stl_file.stat().st_size > 84
    assert png_file.is_file() and png_file.stat().st_size > 0
    with Image.open(png_file) as img:
        assert img.size[0] > 0 and img.size[1] > 0

    assert metrics_file.is_file()
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics_data.get("metrics", {}).get("is_valid") is True
    assert metrics_data.get("metrics", {}).get("solid_count") == 1
    model_sha = metrics_data.get("model_sha256", "")
    assert len(model_sha) == 64 and all(c in "0123456789abcdef" for c in model_sha)

    rev_store = RevisionStore(project)
    head_rev = rev_store.head()
    assert head_rev is not None
    assert head_rev.model_sha256 == model_sha
    assert head_rev.origin.kind == "agent_edit"
    assert rev_store.source(head_rev.id) == scad_file.read_text(encoding="utf-8")

    state = client.get("/api/projects/mounting-bracket/state").get_json()
    assert state == {"status": "idle"}
    assert client.post(
        "/api/projects/mounting-bracket/preview/displayed",
        json={"preview_id": "any"},
    ).status_code == 404

    meta_resp = client.get("/api/projects/mounting-bracket/preview/meta")
    assert meta_resp.status_code == 200
    meta_json = meta_resp.get_json()
    assert meta_json["available"] is True
    assert meta_json["displayable"] is True
    assert meta_json["review_status"] == "not_required"
    assert meta_json["model_sha256"] == model_sha

    preview_resp = client.get("/api/projects/mounting-bracket/preview")
    assert preview_resp.status_code == 200
    assert len(preview_resp.data) == stl_file.stat().st_size

    render_resp = client.get("/api/projects/mounting-bracket/render")
    assert render_resp.status_code == 200
    assert render_resp.mimetype == "image/png"
    assert len(render_resp.data) == png_file.stat().st_size

    history = client.get("/api/projects/mounting-bracket/history").get_json()["events"]
    assert any(
        event.get("role") == "assistant"
        and isinstance(event.get("content"), str)
        and event.get("content")
        for event in history
    )
