"""Tests for application configuration: settings loading and validation.

These tests pin the *behavioural* contract of ``load_settings`` — defaults,
overrides, validation, and provider routing — rather than mirroring the
contents of the prompt or requirements files.
"""
from pathlib import Path

import pytest
import yaml

from agent.settings import Settings, load_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> None:
    """Arrange: write a config.yaml body for the test."""
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Settings: defaults and overrides
# ---------------------------------------------------------------------------


def test_tool_call_limit_defaults_to_12_when_omitted(tmp_path: Path) -> None:
    """Arrange / Act: empty config -> Settings uses the documented default."""
    _write_config(tmp_path, "")

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_tool_call_limit == 12


def test_tool_call_limit_overrides_default_when_provided(tmp_path: Path) -> None:
    """Arrange / Act: explicit override is honoured, not silently coerced."""
    _write_config(tmp_path, "agent:\n  tool_call_limit: 24\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_tool_call_limit == 24


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            "agent:\n  tool_call_limit: 0\n",
            "must be a positive integer",
            id="zero",
        ),
        pytest.param(
            "agent:\n  tool_call_limit: -3\n",
            "must be a positive integer",
            id="negative",
        ),
        pytest.param(
            "agent:\n  tool_call_limit: 'many'\n",
            "must be a positive integer",
            id="non-numeric-string",
        ),
        pytest.param(
            "agent:\n  tool_call_limit: ''\n",
            "must be a positive integer",
            id="empty-string",
        ),
    ],
)
def test_tool_call_limit_rejects_invalid_inputs(
    tmp_path: Path, body: str, match: str
) -> None:
    """Boundary: zero, negative, garbage, and empty inputs must all be rejected."""
    _write_config(tmp_path, body)

    with pytest.raises(ValueError, match=match):
        load_settings(project_root=tmp_path)


def test_debug_tool_error_log_setting_round_trips(tmp_path: Path) -> None:
    """Arrange / Act / Assert: a boolean override is read back as-is."""
    _write_config(tmp_path, "agent:\n  debug_log_tool_errors: true\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_debug_log_tool_errors is True


def test_debug_tool_error_log_default_is_false(tmp_path: Path) -> None:
    """Default value (False) is preserved when the key is absent."""
    _write_config(tmp_path, "")

    settings = load_settings(project_root=tmp_path)

    assert settings.agent_debug_log_tool_errors is False


# ---------------------------------------------------------------------------
# Settings: LLM provider routing
# ---------------------------------------------------------------------------


def test_openai_provider_uses_openai_model_and_remembers_openrouter(
    tmp_path: Path,
) -> None:
    """When ``llm.provider`` is ``openai``, the OpenAI model is active and the
    OpenRouter model is preserved for easy switching back."""
    _write_config(
        tmp_path,
        "llm:\n  provider: openai\nopenai:\n  model: gpt-4.1\nopenrouter:\n  model: openai/gpt-4o\n",
    )

    settings = load_settings(project_root=tmp_path)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4.1"
    assert settings.openai_model == "gpt-4.1"
    assert settings.openrouter_model == "openai/gpt-4o"


def test_openrouter_provider_uses_openrouter_model_by_default(tmp_path: Path) -> None:
    """Default provider is openrouter and ``llm_model`` reflects that."""
    _write_config(tmp_path, "openrouter:\n  model: openai/gpt-4o\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.llm_provider == "openrouter"
    assert settings.llm_model == "openai/gpt-4o"


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    """An unknown provider must raise — silently falling back would mask config errors."""
    _write_config(tmp_path, "llm:\n  provider: bogus\n")

    with pytest.raises(ValueError, match="llm.provider"):
        load_settings(project_root=tmp_path)


def test_empty_or_whitespace_provider_defaults_to_openrouter(tmp_path: Path) -> None:
    """Empty/whitespace provider must not raise — they fall back to the documented default."""
    _write_config(tmp_path, "llm:\n  provider: '   '\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.llm_provider == "openrouter"


def test_settings_instance_retention_count_round_trip() -> None:
    """Constructed Settings preserve ``revision_retention_count`` overrides."""
    default_settings = Settings(Path("/tmp"), "https://e.test", "m", 1, "h", 1)
    assert default_settings.revision_retention_count == 0

    custom_settings = Settings(
        Path("/tmp"), "https://e.test", "m", 1, "h", 1, revision_retention_count=50
    )
    assert custom_settings.revision_retention_count == 50


# ---------------------------------------------------------------------------
# Settings: malformed inputs
# ---------------------------------------------------------------------------


def test_settings_rejects_completely_unparseable_yaml(tmp_path: Path) -> None:
    """A syntactically broken YAML file must raise rather than silently default."""
    _write_config(tmp_path, "this:\n - is\n  not: valid: yaml:\n")

    # The YAML parser raises ``yaml.YAMLError``; we accept any ``ValueError``
    # subtype or ``YAMLError`` since the settings layer may wrap either.
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_settings(project_root=tmp_path)


# ---------------------------------------------------------------------------
# Review gate: ``review_max_cycles`` is gone
# ---------------------------------------------------------------------------


def test_review_max_cycles_field_no_longer_exists(tmp_path: Path) -> None:
    """``review_max_cycles`` was removed when auto-review went opt-in.

    A leftover key in ``config.yaml`` must be silently ignored so legacy
    user files keep loading without error.
    """
    _write_config(tmp_path, "review:\n  max_cycles: 3\n")

    settings = load_settings(project_root=tmp_path)

    assert "review_max_cycles" not in settings.__dataclass_fields__
    # Settings still exposes the surviving review knobs.
    assert settings.review_enabled is True
    assert settings.review_render_workers == 4
    assert settings.review_required_views == 8


# ---------------------------------------------------------------------------
# Settings: preview viewer grid
# ---------------------------------------------------------------------------


def test_viewer_grid_extent_default_matches_legacy_grid(tmp_path: Path) -> None:
    """Without a ``viewer.grid`` block the preview grid stays 200×200 mm /
    20 divisions — the previous hard-coded numbers."""
    _write_config(tmp_path, "")

    settings = load_settings(project_root=tmp_path)

    assert settings.viewer_grid_width == 200.0
    assert settings.viewer_grid_depth == 200.0
    assert settings.viewer_grid_divisions == 20
    assert settings.viewer_grid_extent == (200.0, 200.0, 20)


def test_viewer_grid_extent_parses_3d_string(tmp_path: Path) -> None:
    """The user-facing ``WxDxD`` string format works end-to-end."""
    _write_config(tmp_path, "viewer:\n  grid:\n    extent: 256x256x256\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.viewer_grid_width == 256.0
    assert settings.viewer_grid_depth == 256.0
    assert settings.viewer_grid_divisions == 256
    assert settings.viewer_grid_extent == (256.0, 256.0, 256)


def test_viewer_grid_extent_accepts_mapping(tmp_path: Path) -> None:
    """Explicit ``width`` / ``depth`` / ``divisions`` keys are also valid."""
    _write_config(
        tmp_path,
        "viewer:\n  grid:\n    extent:\n      width: 320\n      depth: 240\n      divisions: 32\n",
    )

    settings = load_settings(project_root=tmp_path)

    assert settings.viewer_grid_width == 320.0
    assert settings.viewer_grid_depth == 240.0
    assert settings.viewer_grid_divisions == 32


def test_viewer_grid_extent_single_number_mirrors_depth(tmp_path: Path) -> None:
    """A bare number supplies both width and depth with the legacy 20
    divisions, so ``"500"`` means a 500×500 mm grid with 20 cells per side."""
    _write_config(tmp_path, "viewer:\n  grid:\n    extent: 500\n")

    settings = load_settings(project_root=tmp_path)

    assert settings.viewer_grid_width == 500.0
    assert settings.viewer_grid_depth == 500.0
    assert settings.viewer_grid_divisions == 20


def test_viewer_grid_extent_rejects_non_positive_values(tmp_path: Path) -> None:
    """A zero or negative side must be rejected — the grid would otherwise
    disappear or render inverted geometry."""
    _write_config(tmp_path, "viewer:\n  grid:\n    extent: 0x0x0\n")

    with pytest.raises(ValueError, match="positive"):
        load_settings(project_root=tmp_path)


def test_viewer_grid_extent_rejects_zero_divisions(tmp_path: Path) -> None:
    """Zero divisions would render a blank grid; reject it up front."""
    _write_config(tmp_path, "viewer:\n  grid:\n    extent: 256x256x0\n")

    with pytest.raises(ValueError, match="divisions"):
        load_settings(project_root=tmp_path)


def test_viewer_grid_extent_rejects_non_numeric_string(tmp_path: Path) -> None:
    """Garbage tokens must fail the loader rather than silently falling back."""
    _write_config(tmp_path, "viewer:\n  grid:\n    extent: 'wide'\n")

    with pytest.raises(ValueError, match="viewer.grid"):
        load_settings(project_root=tmp_path)
