"""Configuration loading for the local CAD agent."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_LOG = logging.getLogger(__name__)
_WARNED_KEYS: set[str] = set()

REASONING_EFFORTS = {"minimal", "low", "medium", "high"}
LLM_PROVIDERS = {"openrouter", "openai", "ollama", "gemini", "google"}


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
    # Ordered list of OpenRouter provider routing slugs (highest priority
    # first), e.g. ``("z-ai/fp8", "novita/fp8", "deepinfra/fp4")``. When
    # non-empty this takes precedence over the legacy ``openrouter.provider``
    # string. OpenRouter walks the list and falls back to its own internal
    # pool if every entry is unavailable. An empty tuple means "no explicit
    # routing requested"; OpenRouter then picks automatically.
    openrouter_provider_order: tuple[str, ...] = ()
    # ── LLM provider selection ──
    llm_provider: str = "openrouter"
    # Optional secondary provider. When set, a transient LLM failure (network,
    # rate limit, server error, …) on the primary client triggers a single
    # retry against this fallback before the agent surfaces an error. User
    # cancellations never fall back. Leave as the empty string to disable.
    llm_fallback_provider: str = ""
    # OpenAI Chat Completions adapter.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_timeout_seconds: int = 60
    openai_reasoning_effort: str | None = None
    # Ollama local Chat Completions adapter.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5-coder:14b-16k"
    ollama_timeout_seconds: int = 120
    ollama_reasoning_effort: str | None = None
    # Gemini / Google AI Studio Interactions API adapter.
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: int = 90
    gemini_reasoning_effort: str | None = None
    gemini_store: bool = True
    # ── end provider selection ──
    show_info_messages: bool = True
    agent_tool_call_limit: int = 12
    revision_retention_count: int = 0  # 0 = unlimited, >0 = keep at most N revisions
    agent_debug_log_tool_errors: bool = False
    # When True, every agent run writes a redacted, append-only JSONL
    # trace of the tool loop (LLM requests, SSE chunks, tool calls, sandbox
    # subprocess I/O) to ``<project>/.cad-agent/activity.jsonl``. Off by
    # default; only enable when debugging model behaviour or wire-level
    # provider errors because the log can grow quickly.
    agent_log_tool_activity: bool = False
    agent_log_mode: str = "off"  # "off", "debug", "dataset"
    # ── Review rendering settings (used by cad_build) ──
    # Multi-view rasterisation is the canonical eight-view + contact-sheet
    # output of ``cad_build``. The structured verdict (the
    # historical dedicated ``cad_review`` tool) was removed; nothing
    # auto-toggles on these flags.
    review_render_workers: int = 4
    review_required_views: int = 8
    # ── 3D preview viewer (Three.js grid plane) ──
    # Square ground grid in mm; ``size`` covers both X and Z, ``divisions``
    # is the cell count per side. Defaults match the previous hard-coded grid.
    viewer_grid_size: float = 200.0
    viewer_grid_divisions: int = 20
    llm_max_completion_tokens: int | None = 8192

    @property
    def llm_model(self) -> str:
        """Return the model name of the active provider."""
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider in ("gemini", "google"):
            return self.gemini_model
        return self.openrouter_model


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


def _reject_unknown(mapping: Any, allowed: set[str], namespace: str) -> None:
    if not isinstance(mapping, dict):
        raise TypeError(f"{namespace} must be a mapping.")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {namespace} setting(s): {', '.join(unknown)}")


# Settings that are still accepted for backwards compatibility but are no
# longer consulted by the agent. Each key is logged once per process so a
# user adding the legacy key to ``config.yaml`` notices the warning in the
# server log instead of assuming the setting does something.
_DEPRECATED_KEYS: dict[str, tuple[str, str]] = {
    "review.max_cycles": (
        "review",
        "the structured review tool was removed; remove max_cycles from config.yaml.",
    ),
}


def _warn_deprecated(namespace: str, key: str) -> None:
    full_key = f"{namespace}.{key}"
    if full_key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(full_key)
    note = _DEPRECATED_KEYS.get(full_key, ("", "no longer used; remove from config.yaml."))[1]
    _LOG.warning(
        "Ignored deprecated setting %s: %s",
        full_key,
        note,
    )


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
    llm = config.get("llm") or {}
    openrouter = config.get("openrouter") or {}
    openai = config.get("openai") or {}
    ollama = config.get("ollama") or {}
    gemini = config.get("gemini") or config.get("google") or {}
    server = config.get("server") or {}
    ui = config.get("ui") or {}
    agent = config.get("agent") or {}
    review = config.get("review") or {}
    viewer = config.get("viewer") or {}
    viewer_grid = viewer.get("grid") or {} if isinstance(viewer, dict) else {}
    _reject_unknown(
        ollama,
        {"base_url", "model", "timeout_seconds", "reasoning_effort", "think"},
        "ollama",
    )
    _reject_unknown(
        gemini,
        {"base_url", "model", "timeout_seconds", "reasoning_effort", "store"},
        "gemini",
    )
    _reject_unknown(
        agent,
        {
            "tool_call_limit",
            "revision_retention_count",
            "debug_log_tool_errors",
            "log_tool_activity",
            "log_mode",
        },
        "agent",
    )
    _reject_unknown(
        review,
        {"enabled", "render_workers", "required_views", "max_cycles"},
        "review",
    )
    # ``review.max_cycles`` is accepted for backward compatibility (legacy
    # configs keep loading) but ignored — the structured review verdict was
    # removed from the tool surface, so the auto-cycle loop no longer makes
    # sense. Emit a one-time warning so users notice the dead key in the
    # server log.
    if isinstance(review, dict) and "max_cycles" in review:
        _warn_deprecated("review", "max_cycles")

    # llm.provider selects which adapter AgentRunner should use. Settings
    # for the inactive provider are still loaded so users can switch without
    # losing their previous model choice.
    llm_provider_raw = _optional_string(llm.get("provider")) or "openrouter"
    if llm_provider_raw not in LLM_PROVIDERS:
        raise ValueError(
            f"llm.provider must be one of {sorted(LLM_PROVIDERS)}, got {llm_provider_raw!r}."
        )
    llm_provider = llm_provider_raw
    # ``llm.fallback_provider`` is optional. An empty string disables the
    # fallback path; any other value must still be a registered provider so a
    # misconfiguration fails at startup instead of mid-run.
    llm_fallback_provider_raw = _optional_string(llm.get("fallback_provider")) or ""
    if llm_fallback_provider_raw and llm_fallback_provider_raw not in LLM_PROVIDERS:
        raise ValueError(
            f"llm.fallback_provider must be one of {sorted(LLM_PROVIDERS)} "
            f"or empty, got {llm_fallback_provider_raw!r}."
        )
    if llm_fallback_provider_raw and llm_fallback_provider_raw == llm_provider:
        raise ValueError(
            "llm.fallback_provider must differ from llm.provider; "
            "leave it empty to disable fallback."
        )
    llm_fallback_provider = llm_fallback_provider_raw

    # OpenRouter provider routing: ``provider_order`` is the explicit,
    # priority-ordered list (highest first). It replaces the legacy
    # single-string ``provider`` setting so users can pin to multiple
    # upstreams in priority order instead of letting OpenRouter pick one
    # fallback internally. Empty list = no explicit routing; the legacy
    # ``provider`` string is consulted instead.
    openrouter_provider_order = _parse_provider_order(openrouter.get("provider_order"))
    if openrouter_provider_order and openrouter.get("provider"):
        _LOG.warning(
            "Both openrouter.provider_order and openrouter.provider are set; "
            "provider_order takes precedence."
        )
    # ``force_provider`` only pins to a single upstream; listing more than
    # one entry under ``provider_order`` is the new "let OpenRouter walk the
    # list" mechanism and is incompatible with pinning. Fail at startup so a
    # misconfigured ``config.yaml`` cannot silently degrade into
    # ``provider.only: [<first entry>]``.
    if (
        len(openrouter_provider_order) > 1
        and bool(openrouter.get("force_provider", False))
    ):
        raise ValueError(
            "openrouter.force_provider=true requires exactly one provider; "
            "remove force_provider (or shorten provider_order to one entry) "
            "to use an ordered fallback list."
        )

    # Each provider keeps its own model; ``settings.llm_model`` returns the
    # active one so callers do not need to branch on the provider.
    grid_size, grid_divisions = _parse_grid_extent(viewer_grid)
    agent_log_tool_activity, agent_log_mode = _validate_log_activity(
        agent.get("log_mode"),
        _strict_bool(agent.get("log_tool_activity", False), "agent.log_tool_activity"),
    )

    ollama_reasoning_effort = _optional_effort(ollama.get("reasoning_effort"))
    if ollama_reasoning_effort is None and "think" in ollama:
        think_val = ollama.get("think")
        if isinstance(think_val, bool):
            ollama_reasoning_effort = "high" if think_val else None
        elif think_val is not None:
            ollama_reasoning_effort = _optional_effort(think_val)

    return Settings(
        workspace_root=Path(config.get("workspace_root", "~/CAD-Agent-Projects")).expanduser(),
        openrouter_base_url=str(openrouter.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_model=str(openrouter.get("model") or ""),
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
        openrouter_provider_order=openrouter_provider_order,
        llm_provider=llm_provider,
        llm_fallback_provider=llm_fallback_provider,
        openai_base_url=str(openai.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
        openai_model=str(openai.get("model") or ""),
        openai_timeout_seconds=_validate_timeout_seconds(openai.get("timeout_seconds", 60), "openai.timeout_seconds"),
        openai_reasoning_effort=_optional_effort(openai.get("reasoning_effort")),
        ollama_base_url=str(ollama.get("base_url") or "http://localhost:11434/v1").rstrip("/"),
        ollama_model=str(ollama.get("model") or "qwen2.5-coder:14b-16k"),
        ollama_timeout_seconds=_validate_timeout_seconds(ollama.get("timeout_seconds") or 120, "ollama.timeout_seconds"),
        ollama_reasoning_effort=ollama_reasoning_effort,
        gemini_base_url=str(gemini.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/"),
        gemini_model=str(gemini.get("model") or "gemini-2.5-flash"),
        gemini_timeout_seconds=_validate_timeout_seconds(gemini.get("timeout_seconds") or 90, "gemini.timeout_seconds"),
        gemini_reasoning_effort=_optional_effort(gemini.get("reasoning_effort")),
        gemini_store=_strict_bool(gemini.get("store", True), "gemini.store"),
        show_info_messages=_strict_bool(ui.get("show_info_messages", True), "ui.show_info_messages"),
        agent_tool_call_limit=_positive_int(agent.get("tool_call_limit", 12), "agent.tool_call_limit"),
        revision_retention_count=_non_negative_int(agent.get("revision_retention_count", 0), "agent.revision_retention_count"),
        agent_debug_log_tool_errors=_strict_bool(
            agent.get("debug_log_tool_errors", False), "agent.debug_log_tool_errors"
        ),
        agent_log_tool_activity=agent_log_tool_activity,
        agent_log_mode=agent_log_mode,
        review_render_workers=_positive_int(
            review.get("render_workers", 4), "review.render_workers"
        ),
        review_required_views=_positive_int(
            review.get("required_views", 8), "review.required_views"
        ),
        viewer_grid_size=grid_size,
        viewer_grid_divisions=grid_divisions,
        llm_max_completion_tokens=_optional_positive_int(
            llm.get("max_completion_tokens", 8192), "llm.max_completion_tokens"
        ),
    )


def _optional_string(value: Any) -> str | None:
    value = str(value).strip() if value is not None else ""
    return value or None


def _parse_provider_order(value: Any) -> tuple[str, ...]:
    """Validate ``openrouter.provider_order`` from ``config.yaml``.

    Returns an ordered tuple of OpenRouter provider routing slugs (highest
    priority first). ``None``/missing/empty list → empty tuple, which makes
    the loader fall back to the legacy ``openrouter.provider`` string.
    Non-string or whitespace-only entries are rejected so a typo never
    silently drops a fallback from the list.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        # Treat a bare string as a single-element list for forgiveness; an
        # accidental ``provider_order: z-ai/fp8`` should not break startup.
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        raise TypeError(
            "openrouter.provider_order must be a list of provider slugs "
            "(e.g. ['z-ai/fp8', 'novita/fp8', 'deepinfra/fp4'])."
        )
    cleaned: list[str] = []
    for entry in candidates:
        if not isinstance(entry, str):
            raise TypeError(
                "openrouter.provider_order entries must be strings; "
                f"got {type(entry).__name__}."
            )
        slug = entry.strip()
        if not slug:
            raise ValueError(
                "openrouter.provider_order entries must be non-empty strings."
            )
        cleaned.append(slug)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(
            "openrouter.provider_order must not contain duplicate entries."
        )
    return tuple(cleaned)


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


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _validate_log_activity(
    log_mode_raw: Any, log_tool_activity: bool
) -> tuple[bool, str]:
    if log_mode_raw is None:
        mode = "debug" if log_tool_activity else "off"
        return log_tool_activity, mode
    mode = str(log_mode_raw).strip().lower()
    if mode not in {"off", "debug", "dataset"}:
        raise ValueError(
            f"agent.log_mode must be 'off', 'debug', or 'dataset', got {log_mode_raw!r}."
        )
    return mode != "off", mode
