"""Configuration loading for the local CAD agent."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REASONING_EFFORTS = {"minimal", "low", "medium", "high"}
LLM_PROVIDERS = {"openrouter", "openai"}


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    openrouter_base_url: str
    openrouter_model: str
    openrouter_timeout_seconds: int
    host: str
    port: int
    openrouter_app_title: str = "Local AI CAD Agent"
    openrouter_app_url: str = ""
    openrouter_session_prefix: str = "local-ai-cad-agent"
    openrouter_enable_anthropic_cache: bool = True
    openrouter_enable_gemini_cache: bool = True
    openrouter_reasoning_effort: str | None = None
    openrouter_provider: str | None = None
    openrouter_force_provider: bool = False
    # ── LLM provider selection ──
    llm_provider: str = "openrouter"
    # OpenAI Chat Completions adapter.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: int = 60
    openai_reasoning_effort: str | None = None
    # ── end provider selection ──
    show_info_messages: bool = True
    agent_tool_call_limit: int = 12
    revision_retention_count: int = 0  # 0 = unlimited, >0 = keep at most N revisions
    agent_debug_log_tool_errors: bool = False
    # ── Review rendering settings (used by cad_build_and_verify, cad_screenshot) ──
    # ``review_enabled`` is preserved for the existing ``cad_build_and_verify``
    # contract: when False the build skips the canonical eight-view rasteriser
    # and contact sheet. The structured verdict (cad_review) is opt-in by the
    # agent and never auto-toggles on this flag.
    review_enabled: bool = True
    review_render_workers: int = 4
    review_required_views: int = 8
    # ── 3D preview viewer (Three.js grid plane) ──
    # Square ground grid in mm; ``size`` covers both X and Z, ``divisions``
    # is the cell count per side. Defaults match the previous hard-coded grid.
    viewer_grid_size: float = 200.0
    viewer_grid_divisions: int = 20

    @property
    def llm_model(self) -> str:
        """Return the model name of the active provider."""
        if self.llm_provider == "openai":
            return self.openai_model
        return self.openrouter_model

    @property
    def viewer_grid_extent(self) -> tuple[float, int]:
        """Return ``(size, divisions)`` for the preview viewport grid."""
        return (self.viewer_grid_size, self.viewer_grid_divisions)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return data


def _strict_bool(value: Any, name: str) -> bool:
    """Reject stringly-typed boolean values so YAML quoted booleans fail fast."""
    if value is True or value is False:
        return value
    raise TypeError(f"{name} must be true or false, got {value!r} (quoted booleans like \"false\" are not supported).")


def _validate_port(value: Any, name: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer between 1 and 65535.") from error
    if port < 1 or port > 65535:
        raise ValueError(f"{name} must be between 1 and 65535, got {port}.")
    return port


def _validate_timeout_seconds(value: Any, name: str) -> int:
    return _positive_int(value, name)


def _parse_grid_extent(value: Any) -> tuple[float, int]:
    """Parse ``viewer.grid.size``/``viewer.grid.divisions`` into ``(size, divisions)``.

    Accepts either a mapping with explicit ``size`` / ``divisions`` keys or a
    bare numeric/string value for ``size`` (in mm). ``divisions`` always
    defaults to 20 to match the previous behaviour. The removed
    ``viewer.grid.extent`` mapping/string is accepted as a compatibility
    alias; its width remains the square grid size because that was also the
    dimension passed to Three.js previously.
    """
    size: float | None = None
    divisions: int | None = None
    if isinstance(value, dict):
        if "size" in value:
            size = float(value["size"])
        if "divisions" in value:
            divisions = int(value["divisions"])
        legacy_extent = value.get("extent")
        if size is None and legacy_extent is not None:
            if isinstance(legacy_extent, dict):
                legacy_size = legacy_extent.get("width", legacy_extent.get("depth"))
                if legacy_size is not None:
                    size = float(legacy_size)
                if divisions is None and "divisions" in legacy_extent:
                    divisions = int(legacy_extent["divisions"])
            elif isinstance(legacy_extent, str):
                parts = [part.strip() for part in legacy_extent.split("x")]
                if not 1 <= len(parts) <= 3 or any(not part for part in parts):
                    raise ValueError("viewer.grid.extent must contain 1 to 3 numbers.")
                size = float(parts[0])
                if divisions is None and len(parts) == 3:
                    divisions = int(float(parts[2]))
            elif isinstance(legacy_extent, (int, float)) and not isinstance(
                legacy_extent, bool
            ):
                size = float(legacy_extent)
            else:
                raise ValueError("viewer.grid.extent must be numeric or a mapping.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        size = float(value)
    elif isinstance(value, str):
        size = float(value.strip())
    elif value is None:
        pass
    else:
        raise ValueError("viewer.grid must be a number or a mapping with size/divisions.")
    if size is None:
        size = 200.0
    if divisions is None:
        divisions = 20
    if not math.isfinite(size) or size <= 0:
        raise ValueError("viewer.grid size must be a positive mm value.")
    if divisions < 1:
        raise ValueError("viewer.grid divisions must be a positive integer.")
    return size, divisions


def load_settings(project_root: Path | None = None) -> Settings:
    project_root = project_root or Path(__file__).resolve().parents[1]
    config = _read_yaml(project_root / "config.yaml")
    llm = config.get("llm", {})
    openrouter = config.get("openrouter", {})
    openai = config.get("openai", {})
    server = config.get("server", {})
    ui = config.get("ui", {})
    agent = config.get("agent", {})
    review = config.get("review", {})
    viewer = config.get("viewer", {})
    viewer_grid = viewer.get("grid", {}) if isinstance(viewer, dict) else {}

    # llm.provider selects which adapter AgentRunner should use. Settings
    # for the inactive provider are still loaded so users can switch without
    # losing their previous model choice.
    llm_provider_raw = _optional_string(llm.get("provider")) or "openrouter"
    if llm_provider_raw not in LLM_PROVIDERS:
        raise ValueError(
            f"llm.provider must be one of {sorted(LLM_PROVIDERS)}, got {llm_provider_raw!r}."
        )
    llm_provider = llm_provider_raw

    # Each provider keeps its own model; ``settings.llm_model`` returns the
    # active one so callers do not need to branch on the provider.
    grid_size, grid_divisions = _parse_grid_extent(viewer_grid)

    return Settings(
        workspace_root=Path(config.get("workspace_root", "~/CAD-Agent-Projects")).expanduser(),
        openrouter_base_url=str(openrouter.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_model=str(openrouter.get("model", "openai/gpt-4o-mini")),
        openrouter_timeout_seconds=_validate_timeout_seconds(openrouter.get("timeout_seconds", 60), "openrouter.timeout_seconds"),
        host=str(server.get("host", "127.0.0.1")),
        port=_validate_port(server.get("port", 5000), "server.port"),
        openrouter_app_title=str(openrouter.get("app_title", "Local AI CAD Agent")),
        openrouter_app_url=str(openrouter.get("app_url", "")).rstrip("/"),
        openrouter_session_prefix=str(openrouter.get("session_prefix", "local-ai-cad-agent")),
        openrouter_enable_anthropic_cache=_strict_bool(openrouter.get("enable_anthropic_cache", True), "openrouter.enable_anthropic_cache"),
        openrouter_enable_gemini_cache=_strict_bool(openrouter.get("enable_gemini_cache", True), "openrouter.enable_gemini_cache"),
        openrouter_reasoning_effort=_optional_effort(openrouter.get("reasoning_effort")),
        openrouter_provider=_optional_string(openrouter.get("provider")),
        openrouter_force_provider=_strict_bool(openrouter.get("force_provider", False), "openrouter.force_provider"),
        llm_provider=llm_provider,
        openai_base_url=str(openai.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
        openai_model=str(openai.get("model", "gpt-4o-mini")),
        openai_timeout_seconds=_validate_timeout_seconds(openai.get("timeout_seconds", 60), "openai.timeout_seconds"),
        openai_reasoning_effort=_optional_effort(openai.get("reasoning_effort")),
        show_info_messages=_strict_bool(ui.get("show_info_messages", True), "ui.show_info_messages"),
        agent_tool_call_limit=_positive_int(agent.get("tool_call_limit", 12), "agent.tool_call_limit"),
        revision_retention_count=_non_negative_int(agent.get("revision_retention_count", 0), "agent.revision_retention_count"),
        agent_debug_log_tool_errors=_strict_bool(
            agent.get("debug_log_tool_errors", False), "agent.debug_log_tool_errors"
        ),
        review_enabled=_strict_bool(review.get("enabled", True), "review.enabled"),
        review_render_workers=_positive_int(
            review.get("render_workers", 4), "review.render_workers"
        ),
        review_required_views=_positive_int(
            review.get("required_views", 8), "review.required_views"
        ),
        viewer_grid_size=grid_size,
        viewer_grid_divisions=grid_divisions,
    )


def _optional_string(value: Any) -> str | None:
    value = str(value).strip() if value is not None else ""
    return value or None


def _optional_effort(value: Any) -> str | None:
    effort = _optional_string(value)
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be minimal, low, medium, or high.")
    return effort


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if number < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer.") from error
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number
