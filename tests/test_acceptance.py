import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("build123d")

from agent.settings import Settings
from app import create_app

MODEL_CODE = """from build123d import Align, Box, Cylinder

# Dimensions (mm)
width = 60.0
length = 40.0
height = 8.0
hole_diameter = 6.0
body = Box(width, length, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
hole = Cylinder(hole_diameter / 2, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
result = body - hole
"""


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
                                        "name": "file_write",
                                        "arguments": (
                                            json.dumps(
                                                {
                                                    "filename": "model.py",
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
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        # Detect the structured reviewer call (it forces a single
        # ``submit_review`` tool) so the acceptance flow can complete without a
        # real multimodal model. Generator turns (no forced tools) return the
        # final assistant message instead.
        if _tools and any(
            isinstance(tool, dict)
            and tool.get("function", {}).get("name") == "submit_review"
            for tool in _tools
        ):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "review-pass",
                                    "function": {
                                        "name": "submit_review",
                                        "arguments": json.dumps(
                                            {
                                                "status": "pass",
                                                "summary": "Build matches the request.",
                                                "findings": [],
                                            }
                                        ),
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

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test", 1, "127.0.0.1", 5000
    )
    question_client = QuestionClient()
    build_client = BuildClient()
    # The reviewer is invoked after ``cad_build_and_verify``; the agent asks
    # one clarification first, then ``build_client`` handles every subsequent
    # turn (file write, build, structured review, final answer). The main loop
    # and ``_run_post_build_review`` both call ``create_llm_client``; the first
    # call returns the question client, every later call returns the build
    # client so the reviewer can deterministically emit a passing verdict.
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
    assert runner.waiting_question("mounting-bracket") is not None

    response = client.post(
        "/api/questions/answer", json={"project": "mounting-bracket", "answer": "6 mm"}
    )
    assert response.status_code == 202
    runner._thread.join(timeout=15)
    project = settings.workspace_root / "mounting-bracket"
    assert (project / "model.py").is_file()
    assert (project / "preview.stl").is_file()
    assert (project / "render.png").is_file()

    state = client.get("/api/projects/mounting-bracket/state").get_json()
    assert state["status"] == "rendering"
    response = client.post(
        "/api/projects/mounting-bracket/preview/displayed",
        json={"preview_id": state["preview_id"]},
    )
    assert response.status_code == 200
    history = client.get("/api/projects/mounting-bracket/history").get_json()["events"]
    # The final assistant turn reaches conversation.jsonl so the UI history
    # drawer can replay it; we only assert the canonical role/content pair
    # rather than exact LLM wording.
    assert any(
        event.get("role") == "assistant"
        and isinstance(event.get("content"), str)
        and event.get("content")
        for event in history
    )
