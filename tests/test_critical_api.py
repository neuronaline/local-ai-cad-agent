from pathlib import Path

from agent.settings import Settings
from app import create_app


def test_project_creation_and_cross_origin_chat_guard(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    created = client.post("/api/projects/new", json={"name": "Mounting Bracket"})

    assert created.status_code == 201
    assert created.get_json() == {"project": "mounting-bracket"}
    blocked = client.post(
        "/api/chat",
        json={"project": "mounting-bracket", "message": "make a bracket"},
        headers={"Origin": "https://attacker.test"},
    )
    assert blocked.status_code == 403
