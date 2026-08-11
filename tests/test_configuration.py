"""Tests for application configuration: settings, prompt, and dependencies."""

from pathlib import Path

import pytest

from agent.prompt import get_build123d_playbook, get_system_prompt
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

    with pytest.raises(
        ValueError, match="agent.tool_call_limit must be a positive integer"
    ):
        load_settings(project_root=tmp_path)


def test_debug_tool_error_log_setting(tmp_path):
    _write_config(tmp_path, "agent:\n  debug_log_tool_errors: true\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_debug_log_tool_errors is True


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
    playbook = get_build123d_playbook()
    system_prompt = get_system_prompt()
    assert playbook.startswith("# build123d CAD CLI Playbook")
    assert "```markdown" not in playbook
    assert system_prompt.count("<build123d_cli_playbook>") == 1
    assert playbook in system_prompt
    assert "cad_build_and_verify performs the build" in system_prompt
    assert "cad.run" not in system_prompt


def test_build123d_playbook_has_versioned_curve_and_topology_guidance():
    playbook = get_build123d_playbook()
    system_prompt = get_system_prompt()
    assert "`build123d` **0.11.1**" in playbook
    assert "`Ellipse` creates a **complete filled 2D sketch**" in playbook
    assert "EllipticalCenterArc" in playbook
    assert "radius >= endpoint_distance / 2" in playbook
    assert "There is no top-level `max_fillet` function" in playbook
    assert "Final exports are handled separately by the application" in playbook
    assert "Değişiklik Özeti" not in playbook
    assert "discard any cached edge/face indices" in system_prompt


def test_system_prompt_defines_a_verified_and_non_repetitive_workflow():
    system_prompt = get_system_prompt()

    assert "Do not claim success from source inspection alone" in system_prompt
    assert "never rebuild unchanged source" in system_prompt
    assert "Ask all blocking questions together" in system_prompt
    assert "State any important assumption in the final reply" in system_prompt


# ---------------------------------------------------------------------------
# Pinned dependency
# ---------------------------------------------------------------------------


def test_build123d_version_matches_the_versioned_playbook():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()

    assert "build123d==0.11.1" in requirements
