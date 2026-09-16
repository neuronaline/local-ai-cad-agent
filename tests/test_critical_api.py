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


def test_synthetic_reminder_persisted_and_filtered_from_ui_history(tmp_path: Path) -> None:
    """Synthetic reminders are persisted in conversation store but filtered from UI history endpoint."""
    from agent.conversation import ConversationStore

    settings = Settings(
        tmp_path / "projects", "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    client = create_app(settings).test_client()

    client.post("/api/projects/new", json={"name": "Widget"})
    project_dir = tmp_path / "projects" / "widget"

    user_msg = {"role": "user", "content": "Initial user prompt"}
    synthetic_msg = {
        "role": "user",
        "content": "model.py exists but it has not been verified. Call cad_build_and_verify now.",
        "synthetic": True,
    }
    assistant_msg = {"role": "assistant", "content": "Done."}

    ConversationStore.append(project_dir, user_msg)
    ConversationStore.append(project_dir, synthetic_msg)
    ConversationStore.append(project_dir, assistant_msg)

    # ConversationStore loads both so prompt prefix is stable
    stored = ConversationStore.load(project_dir)
    assert len(stored) == 3
    assert stored[1]["synthetic"] is True

    # UI history endpoint returns only real user and assistant messages
    res = client.get("/api/projects/widget/history")
    assert res.status_code == 200
    events = res.get_json()["events"]
    assert len(events) == 2
    assert events[0]["content"] == "Initial user prompt"
    assert events[1]["content"] == "Done."

