import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("build123d")

from agent.settings import Settings
from app import create_app

MODEL_CODE = '''from build123d import Align, Box, Cylinder

# Dimensions (mm)
width = 60.0
length = 40.0
height = 8.0
hole_diameter = 6.0
body = Box(width, length, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
hole = Cylinder(hole_diameter / 2, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = body - hole
'''


class QuestionClient:
    def chat(self, _messages, _tools):
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
            "id": "question-1", "function": {"name": "question", "arguments": (
                '{"question":"What hole diameter should I use?","input_type":"number"}'
            )}
        }]}}]}


class BuildClient:
    def __init__(self):
        self.calls = 0
        self.messages = []

    def chat(self, messages, _tools):
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "write-1", "function": {"name": "file", "arguments": (
                    json.dumps({"operation": "write", "filename": "model.py", "content": MODEL_CODE})
                )}
            }]}}]}
        if self.calls == 2:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "run-1", "function": {"name": "cad", "arguments": '{"operation":"run"}'}
            }]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "The bracket is ready for finalization."}}]}


def test_mvp_acceptance_flow(tmp_path: Path, monkeypatch):
    import agent.core

    settings = Settings(tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000)
    build_client = BuildClient()
    clients = iter([QuestionClient(), build_client])
    monkeypatch.setattr(agent.core, "OpenRouterClient", lambda _settings: next(clients))
    app = create_app(settings)
    client = app.test_client()
    assert client.post("/api/projects/new", json={"name": "mounting-bracket"}).status_code == 201

    sketch = BytesIO()
    Image.new("RGB", (40, 30), "white").save(sketch, format="PNG")
    sketch.seek(0)
    response = client.post("/api/chat", data={
        "project": "mounting-bracket", "message": "Create a mounting bracket from this sketch.",
        "attachments": (sketch, "bracket.png"),
    })
    assert response.status_code == 202
    runner = app.config["AGENT_RUNNER"]
    runner._thread.join(timeout=3)
    assert runner.waiting_question("mounting-bracket") is not None

    response = client.post("/api/questions/answer", json={"project": "mounting-bracket", "answer": "6 mm"})
    assert response.status_code == 202
    runner._thread.join(timeout=15)
    project = settings.workspace_root / "mounting-bracket"
    assert (project / "model.py").is_file()
    assert (project / "preview.stl").is_file()
    assert (project / "render.png").is_file()
    critique = build_client.messages[2][-1]
    assert critique["role"] == "user"
    assert critique["content"][1]["type"] == "image_url"
    assert "Geometry metrics" in critique["content"][0]["text"]

    state = client.get("/api/projects/mounting-bracket/state").get_json()
    assert state["status"] == "rendering"
    response = client.post(
        "/api/projects/mounting-bracket/preview/displayed",
        json={"preview_id": state["preview_id"]},
    )
    assert response.status_code == 200
    history = client.get("/api/projects/mounting-bracket/history").get_json()["events"]
    assert any(
        event.get("role") == "assistant"
        and event.get("content") == "The bracket is ready for finalization."
        for event in history
    )

    response = client.post("/api/projects/mounting-bracket/finalize")
    assert response.status_code == 200
    assert response.get_json()["metrics"]["volume_mm3"] < 60 * 40 * 8
    for filename in ("model.step", "model.stl", "report.md"):
        assert (project / "output" / filename).is_file()
