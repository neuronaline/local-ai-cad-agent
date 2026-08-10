"""Tests for application configuration: settings, prompt, and dependencies."""
from pathlib import Path

import pytest

from agent.prompt import BUILD123D_PLAYBOOK, SYSTEM_PROMPT
from agent.settings import load_settings


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("agent:\n  tool_call_limit: 24\n", 24, id="loaded-from-config"),
        pytest.param("", 12, id="default-when-omitted"),
    ],
)
def test_tool_call_limit_loading(tmp_path, body, expected):
    _write_config(tmp_path, body)

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_tool_call_limit == expected


def test_tool_call_limit_must_be_positive(tmp_path):
    _write_config(tmp_path, "agent:\n  tool_call_limit: 0\n")

    with pytest.raises(ValueError, match="agent.tool_call_limit must be a positive integer"):
        load_settings(project_root=tmp_path)


def test_settings_uses_openai_model_when_provider_is_openai(tmp_path):
    _write_config(
        tmp_path,
        "llm:\n  provider: openai\nopenai:\n  model: gpt-4.1\nopenrouter:\n  model: openai/gpt-4o\n",
    )

    settings = load_settings(project_root=tmp_path)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4.1"
    # The other provider's model is still remembered for easy switching.
    assert settings.openrouter_model == "openai/gpt-4o"
    assert settings.openai_model == "gpt-4.1"


def test_settings_rejects_unknown_provider(tmp_path):
    _write_config(tmp_path, "llm:\n  provider: bogus\n")

    with pytest.raises(ValueError, match="llm.provider"):
        load_settings(project_root=tmp_path)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_build123d_playbook_is_injected_once_into_the_static_system_prompt():
    assert BUILD123D_PLAYBOOK.startswith("# build123d CAD CLI Playbook")
    assert "```markdown" not in BUILD123D_PLAYBOOK
    assert SYSTEM_PROMPT.count("<build123d_cli_playbook>") == 1
    assert BUILD123D_PLAYBOOK in SYSTEM_PROMPT
    assert "cad_build_and_verify performs the build" in SYSTEM_PROMPT
    assert "cad.run" not in SYSTEM_PROMPT


def test_build123d_playbook_has_versioned_curve_and_topology_guidance():
    assert "`build123d` **0.11.1**" in BUILD123D_PLAYBOOK
    assert "`Ellipse` creates a **complete filled 2D sketch**" in BUILD123D_PLAYBOOK
    assert "EllipticalCenterArc" in BUILD123D_PLAYBOOK
    assert "radius >= endpoint_distance / 2" in BUILD123D_PLAYBOOK
    assert "There is no top-level `max_fillet` function" in BUILD123D_PLAYBOOK
    assert "discard any cached edge/face indices" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Pinned dependency
# ---------------------------------------------------------------------------


def test_build123d_version_matches_the_versioned_playbook():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "build123d==0.11.1" in requirements
