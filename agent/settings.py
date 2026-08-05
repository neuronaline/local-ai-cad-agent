"""Configuration loading for the local CAD agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


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
    openrouter_reasoning_effort: str | None = None
    openrouter_provider: str | None = None
    openrouter_force_provider: bool = False
    show_info_messages: bool = True
    agent_tool_call_limit: int = 12
    revision_retention_count: int = 0  # 0 = unlimited, >0 = keep at most N revisions
    quality_enabled: bool = True  # passive run/attempt observability (plan Phase 1)
    quality_require_acceptance_before_finalize: bool = False  # plan Phase 2 gate (rollout stage 7)

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def load_settings(project_root: Path | None = None, home: Path | None = None) -> Settings:
    project_root = project_root or Path(__file__).resolve().parents[1]
    home = home or Path.home()
    config = _merge(_read_yaml(project_root / "config.yaml"), _read_yaml(home / ".cad-agent" / "config.yaml"))
    openrouter = config.get("openrouter", {})
    server = config.get("server", {})
    ui = config.get("ui", {})
    agent = config.get("agent", {})
    quality = config.get("quality", {})

    quality_enabled = _strict_bool(quality.get("enabled", True), "quality.enabled")
    quality_require_acceptance = _strict_bool(
        quality.get("require_acceptance_before_finalize", False),
        "quality.require_acceptance_before_finalize",
    )
    if not quality_enabled and quality_require_acceptance:
        raise ValueError(
            "quality.require_acceptance_before_finalize=true requires quality.enabled=true. "
            "Enable quality or disable the acceptance gate."
        )

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
        openrouter_reasoning_effort=_optional_effort(openrouter.get("reasoning_effort")),
        openrouter_provider=_optional_string(openrouter.get("provider")),
        openrouter_force_provider=_strict_bool(openrouter.get("force_provider", False), "openrouter.force_provider"),
        show_info_messages=_strict_bool(ui.get("show_info_messages", True), "ui.show_info_messages"),
        agent_tool_call_limit=_positive_int(agent.get("tool_call_limit", 12), "agent.tool_call_limit"),
        revision_retention_count=_non_negative_int(agent.get("revision_retention_count", 0), "agent.revision_retention_count"),
        quality_enabled=quality_enabled,
        quality_require_acceptance_before_finalize=quality_require_acceptance,
    )


def _optional_string(value: Any) -> str | None:
    value = str(value).strip() if value is not None else ""
    return value or None


def _optional_effort(value: Any) -> str | None:
    effort = _optional_string(value)
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError("openrouter.reasoning_effort must be minimal, low, medium, or high.")
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
