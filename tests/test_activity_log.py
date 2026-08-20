"""Tests for the per-project activity log.

These tests pin the *behavioural* contract of :mod:`agent.activity_log`:
redaction rules, file layout, size-based trimming, and the no-failure
guarantee. They do not assert on the format of any individual SSE event —
those payloads are owned by their producers.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent.activity_log import (
    _DEFAULT_MAX_BYTES,
    ActivityLogger,
    is_enabled,
    redact,
)
from agent.settings import Settings

# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


def _settings(*, enabled: bool) -> Settings:
    return Settings(
        workspace_root=Path("/tmp/activity-log-tests"),
        openrouter_base_url="https://example.test",
        openrouter_model="test",
        openrouter_timeout_seconds=1,
        host="127.0.0.1",
        port=5000,
        agent_log_tool_activity=enabled,
    )


def test_is_enabled_reflects_settings_flag() -> None:
    """``is_enabled`` is the canonical on/off check for the agent loop."""
    assert is_enabled(_settings(enabled=True)) is True
    assert is_enabled(_settings(enabled=False)) is False


def test_default_activity_log_setting_is_false(tmp_path: Path) -> None:
    """The new ``agent.log_tool_activity`` flag defaults to False."""
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    from agent.settings import load_settings

    settings = load_settings(project_root=tmp_path)
    assert settings.agent_log_tool_activity is False


def test_activity_log_setting_round_trips(tmp_path: Path) -> None:
    """An explicit True override is read back as-is."""
    (tmp_path / "config.yaml").write_text(
        "agent:\n  log_tool_activity: true\n", encoding="utf-8"
    )
    from agent.settings import load_settings

    settings = load_settings(project_root=tmp_path)
    assert settings.agent_log_tool_activity is True


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redact_replaces_authorization_keys_at_any_depth() -> None:
    payload = {
        "headers": {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
            "authorization": "also-secret",
        },
        "meta": {"api_key": "sk-1234"},
        "list": [{"x-api-key": "y"}],
    }
    result = redact(payload)
    assert result["headers"]["Authorization"] == "[REDACTED]"
    assert result["headers"]["Content-Type"] == "application/json"
    assert result["headers"]["authorization"] == "[REDACTED]"
    assert result["meta"]["api_key"] == "[REDACTED]"
    assert result["list"][0]["x-api-key"] == "[REDACTED]"


def test_redact_replaces_inline_image_url_data_blobs() -> None:
    """OpenAI-style image_url parts lose their base64 payload but keep metadata."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see attached"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
    }
    result = redact(payload)
    image_url = result["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("[IMAGE_DATA_URL redacted,")
    # Size should be reported as the base64 portion length so debuggers can
    # still tell how big the dropped content was.
    assert "4 base64 chars" in image_url


def test_redact_passes_normal_payloads_through() -> None:
    payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    assert redact(payload) == payload


def test_redact_handles_non_dict_roots() -> None:
    assert redact("plain text") == "plain text"
    assert redact([1, 2, {"api_key": "x"}]) == [1, 2, {"api_key": "[REDACTED]"}]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_logger_writes_one_json_line_per_event(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path)
    logger.log("run_start", {"project": "demo"}, run_id="abc")
    logger.log("tool_call_start", {"tool": "read_file"}, run_id="abc")

    text = (tmp_path / ".cad-agent" / "activity.jsonl").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert entry["ts"]
        assert entry["run_id"] == "abc"
    assert json.loads(lines[0])["event"] == "run_start"
    assert json.loads(lines[1])["event"] == "tool_call_start"


def test_logger_creates_missing_directory(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path)
    logger.log("event", {"k": 1})
    assert (tmp_path / ".cad-agent").is_dir()
    assert (tmp_path / ".cad-agent" / "activity.jsonl").is_file()


def test_logger_isolates_run_ids(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path)
    logger.log("event", {"k": "a"}, run_id="run-1")
    logger.log("event", {"k": "b"}, run_id="run-2")

    lines = (tmp_path / ".cad-agent" / "activity.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    entries = [json.loads(line) for line in lines]
    assert entries[0]["run_id"] == "run-1"
    assert entries[1]["run_id"] == "run-2"
    assert entries[0]["data"]["k"] == "a"


def test_logger_tail_returns_recent_events(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path)
    for index in range(5):
        logger.log("event", {"i": index})

    tail = logger.tail(limit=3)
    assert len(tail) == 3
    assert [entry["data"]["i"] for entry in tail] == [2, 3, 4]


def test_logger_tail_handles_missing_file(tmp_path: Path) -> None:
    """``tail`` on a project with no log returns an empty list, not an error."""
    assert ActivityLogger(tmp_path).tail() == []


def test_logger_clear_removes_existing_file(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path)
    logger.log("event", {})
    assert logger.log_path.is_file()
    assert logger.clear() is True
    assert not logger.log_path.exists()


def test_logger_clear_returns_true_for_missing_file(tmp_path: Path) -> None:
    """``clear`` reports success even when the log file never existed."""
    # The ActivityLogger contract is idempotent; clearing twice is a no-op.
    logger = ActivityLogger(tmp_path)
    assert logger.clear() is True
    assert logger.clear() is True


def test_logger_trims_when_exceeding_max_bytes(tmp_path: Path) -> None:
    logger = ActivityLogger(tmp_path, max_bytes=512)
    for index in range(200):
        logger.log("event", {"i": index, "pad": "x" * 64})

    size = (tmp_path / ".cad-agent" / "activity.jsonl").stat().st_size
    assert size <= 1024  # coarse trim; exact size depends on padding.


def test_logger_swallows_os_errors(tmp_path: Path) -> None:
    """A logger pointed at an unwritable path never raises.

    We point the log path at a regular file so the subsequent open() in
    ``append`` mode fails. The contract is that activity logging must
    never break the agent loop.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    # Place the project inside ``blocker`` (which is a file, not a dir)
    # so ``log_path.parent.mkdir`` raises.
    logger = ActivityLogger(blocker / "nested")
    # Should be a silent no-op, not an exception:
    logger.log("event", {"x": 1})


def test_logger_default_max_bytes_constant_is_sane() -> None:
    """The default cap is at least 1 MiB so ordinary runs are never trimmed."""
    assert _DEFAULT_MAX_BYTES >= 1024 * 1024


# ---------------------------------------------------------------------------
# Integration: agent loop wire path
# ---------------------------------------------------------------------------


def test_get_logger_returns_independent_instances(tmp_path: Path) -> None:
    """Two projects get two log files; the same project reuses one."""
    from agent.activity_log import get_logger

    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    logger_a = get_logger(project_a)
    logger_b = get_logger(project_b)
    logger_a.log("event", {"who": "a"})
    logger_b.log("event", {"who": "b"})
    assert (project_a / ".cad-agent" / "activity.jsonl").is_file()
    assert (project_b / ".cad-agent" / "activity.jsonl").is_file()
    assert logger_a is not logger_b