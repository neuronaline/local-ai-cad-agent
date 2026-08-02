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


def load_settings(project_root: Path | None = None, home: Path | None = None) -> Settings:
    project_root = project_root or Path(__file__).resolve().parents[1]
    home = home or Path.home()
    config = _merge(_read_yaml(project_root / "config.yaml"), _read_yaml(home / ".cad-agent" / "config.yaml"))
    openrouter = config.get("openrouter", {})
    server = config.get("server", {})
    ui = config.get("ui", {})
    agent = config.get("agent", {})
    return Settings(
        workspace_root=Path(config.get("workspace_root", "~/CAD-Agent-Projects")).expanduser(),
        openrouter_base_url=str(openrouter.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_model=str(openrouter.get("model", "openai/gpt-4o-mini")),
        openrouter_timeout_seconds=int(openrouter.get("timeout_seconds", 60)),
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 5000)),
        openrouter_app_title=str(openrouter.get("app_title", "Local AI CAD Agent")),
        openrouter_app_url=str(openrouter.get("app_url", "")).rstrip("/"),
        openrouter_session_prefix=str(openrouter.get("session_prefix", "local-ai-cad-agent")),
        openrouter_enable_anthropic_cache=bool(openrouter.get("enable_anthropic_cache", True)),
        openrouter_reasoning_effort=_optional_effort(openrouter.get("reasoning_effort")),
        openrouter_provider=_optional_string(openrouter.get("provider")),
        openrouter_force_provider=bool(openrouter.get("force_provider", False)),
        show_info_messages=bool(ui.get("show_info_messages", True)),
        agent_tool_call_limit=_positive_int(agent.get("tool_call_limit", 12), "agent.tool_call_limit"),
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
