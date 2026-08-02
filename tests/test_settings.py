from pathlib import Path

import pytest

from agent.settings import load_settings


def test_tool_call_limit_is_loaded_from_config(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        "agent:\n  tool_call_limit: 24\n",
        encoding="utf-8",
    )

    settings = load_settings(project_root=tmp_path, home=tmp_path / "home")

    assert settings.agent_tool_call_limit == 24


def test_tool_call_limit_must_be_positive(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        "agent:\n  tool_call_limit: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agent.tool_call_limit must be a positive integer"):
        load_settings(project_root=tmp_path, home=tmp_path / "home")
