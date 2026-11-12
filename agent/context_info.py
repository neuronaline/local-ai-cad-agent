"""Context window limit resolution and active context memory occupancy calculation.

Distinguishes active context window occupancy (tokens currently held in the
next/latest LLM prompt prefix) from cumulative billed token throughput across
multi-step tool loops.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote as _url_quote

import httpx

from agent.conversation import ConversationStore
from agent.llm_base import normalize_messages, relocate_tool_images
from agent.prompt import get_system_prompt
from agent.settings import Settings, canonical_provider
from agent.tool_schemas import TOOL_SCHEMAS

_LOG = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
# Maps cache key -> (limit_tokens_or_none, expires_monotonic)
_LIMIT_CACHE: dict[str, tuple[int | None, float]] = {}
_OPENROUTER_RETRY_AFTER: dict[str, float] = {}

_MAX_LIMIT_CACHE_ENTRIES = 1024
_POSITIVE_TTL_SECONDS = 3600.0
_NEGATIVE_TTL_SECONDS = 300.0
_NETWORK_TIMEOUT_SECONDS = 2.5

_CHARS_PER_TOKEN = 3.8
_IMAGE_PART_TOKENS = 765
_MESSAGE_OVERHEAD_TOKENS = 4

_TOOL_SCHEMA_TOKENS: int | None = None


def _put_limit_cache_locked(
    cache_key: str,
    value: int | None,
    expires: float,
    now: float,
) -> None:
    if cache_key not in _LIMIT_CACHE and len(_LIMIT_CACHE) >= _MAX_LIMIT_CACHE_ENTRIES:
        expired = [k for k, (_, exp) in _LIMIT_CACHE.items() if now >= exp]
        for k in expired:
            _LIMIT_CACHE.pop(k, None)
        while len(_LIMIT_CACHE) >= _MAX_LIMIT_CACHE_ENTRIES:
            _LIMIT_CACHE.pop(next(iter(_LIMIT_CACHE)))
    _LIMIT_CACHE[cache_key] = (value, expires)


def _get_cached_limit(cache_key: str) -> tuple[bool, int | None]:
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _LIMIT_CACHE.get(cache_key)
        if entry is None:
            return False, None
        val, expires = entry
        if now >= expires:
            _LIMIT_CACHE.pop(cache_key, None)
            return False, None
        return True, val


def _set_cached_limit(cache_key: str, value: int | None) -> None:
    now = time.monotonic()
    ttl = _POSITIVE_TTL_SECONDS if value else _NEGATIVE_TTL_SECONDS
    with _CACHE_LOCK:
        _put_limit_cache_locked(cache_key, value, now + ttl, now)


def _normalize_model_slug(model: str) -> str:
    slug = model.strip().lower()
    if "/" in slug:
        slug = slug.rsplit("/", 1)[-1]
    for sep in (":", "@"):
        if sep in slug:
            base, suffix = slug.split(sep, 1)
            # Keep Ollama parameter size tags only for prefix matching; base is cleaner
            slug = base
    return slug.strip()


def _builtin_context_limit(provider: str, model: str) -> int | None:
    """Return a conservative static context window limit when live metadata is unavailable."""
    raw = model.strip().lower()
    slug = _normalize_model_slug(raw)
    if not slug:
        return None

    # Gemini family (AI Studio, Vertex, OpenRouter)
    if slug.startswith("gemini-1.5-pro"):
        return 2_097_152
    if slug.startswith("gemini-"):
        return 1_048_576

    # Xiaomi MiMo family
    if slug.startswith("mimo-") or raw.startswith("xiaomi/mimo"):
        return 262_144

    # Anthropic Claude family
    if slug.startswith("claude-"):
        return 200_000

    # Z.AI / GLM family
    if slug.startswith("glm-5"):
        return 202_752
    if slug.startswith("glm-4"):
        return 131_072

    # DeepSeek family
    if slug.startswith(("deepseek-v3", "deepseek-r1", "deepseek-chat", "deepseek-reasoner")):
        return 131_072
    if slug.startswith("deepseek-"):
        return 131_072

    # OpenAI GPT & o-series
    if slug.startswith("gpt-4.1"):
        return 1_047_576
    if slug.startswith(("gpt-5", "gpt-6")):
        return 400_000
    if slug.startswith(("o1", "o3", "o4")):
        return 200_000
    if slug.startswith(("gpt-4o", "gpt-4-turbo", "chatgpt-4o")):
        return 128_000

    # xAI Grok, Moonshot Kimi, MiniMax
    if slug.startswith("grok-4"):
        return 256_000
    if slug.startswith("grok-"):
        return 131_072
    if slug.startswith(("kimi-", "moonshot-")):
        return 262_144
    if slug.startswith("minimax-"):
        return 204_800

    # Qwen, Llama, Mistral families
    if slug.startswith(("qwen3.5", "qwen3.8", "qwen3-coder")):
        return 262_144 if provider != "ollama" else 131_072
    if slug.startswith(("qwen3", "qwen2.5", "qwq")):
        return 131_072
    if slug.startswith(("llama-4", "llama4")):
        return 262_144 if provider != "ollama" else 131_072
    if slug.startswith(("llama-3", "llama3")):
        return 131_072
    if slug.startswith(("devstral", "codestral")):
        return 262_144
    if slug.startswith(("mistral-", "pixtral-")):
        return 131_072

    return None


def _lookup_openrouter_cached(or_root: str, model: str) -> tuple[bool, int | None]:
    hit, val = _get_cached_limit(f"openrouter:{or_root}:{model}")
    if hit:
        return True, val
    if ":" in model:
        base_model = model.split(":", 1)[0]
        hit, val = _get_cached_limit(f"openrouter:{or_root}:{base_model}")
        if hit:
            return True, val
    return False, None


def _fetch_openrouter_limits(base_url: str, target_model: str) -> int | None:
    root = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
    base_target = target_model.split(":", 1)[0] if ":" in target_model else target_model
    now = time.monotonic()
    with _CACHE_LOCK:
        if now < _OPENROUTER_RETRY_AFTER.get(root, 0.0):
            for candidate in (target_model, base_target):
                entry = _LIMIT_CACHE.get(f"openrouter:{root}:{candidate}")
                if entry and now < entry[1]:
                    return entry[0]
            return None
        _OPENROUTER_RETRY_AFTER[root] = now + _NEGATIVE_TTL_SECONDS

    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=_NETWORK_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = client.get(f"{root}/models", headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        _LOG.debug("OpenRouter model metadata lookup skipped: %s", exc)
        return None

    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None

    now_done = time.monotonic()
    expiry = now_done + _POSITIVE_TTL_SECONDS
    resolved: int | None = None
    fallback_resolved: int | None = None
    with _CACHE_LOCK:
        _OPENROUTER_RETRY_AFTER[root] = expiry
        for item in items:
            if not isinstance(item, dict):
                continue
            m_id = str(item.get("id") or "").strip()
            if not m_id:
                continue
            top_prov = item.get("top_provider")
            raw_ctx = (
                top_prov.get("context_length")
                if isinstance(top_prov, dict) and top_prov.get("context_length")
                else item.get("context_length")
            )
            try:
                ctx_len = int(raw_ctx) if raw_ctx is not None else 0
            except (TypeError, ValueError):
                ctx_len = 0
            if ctx_len > 0:
                _put_limit_cache_locked(f"openrouter:{root}:{m_id}", ctx_len, expiry, now_done)
                if m_id == target_model:
                    resolved = ctx_len
                elif m_id == base_target:
                    fallback_resolved = ctx_len
    return resolved or fallback_resolved


def _fetch_ollama_limit(base_url: str, model: str) -> int | None:
    root = (base_url or "http://localhost:11434/v1").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    cache_key = f"ollama:{root}:{model}"
    hit, cached = _get_cached_limit(cache_key)
    if hit:
        return cached

    try:
        with httpx.Client(timeout=1.8, follow_redirects=True) as client:
            resp = client.post(f"{root}/api/show", json={"name": model})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        _LOG.debug("Ollama /api/show lookup skipped for %s: %s", model, exc)
        _set_cached_limit(cache_key, None)
        return None

    if not isinstance(data, dict):
        _set_cached_limit(cache_key, None)
        return None

    # 1. Explicit Modelfile `PARAMETER num_ctx <N>` overrides architectural max
    params_str = data.get("parameters")
    if isinstance(params_str, str):
        match = re.search(r"(?m)^\s*num_ctx\s+(\d+)\b", params_str)
        if match:
            val = int(match.group(1))
            if val > 0:
                _set_cached_limit(cache_key, val)
                return val

    # 2. GGUF model_info `<arch>.context_length`
    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        for key, raw_val in model_info.items():
            if isinstance(key, str) and key.endswith(".context_length"):
                try:
                    val = int(raw_val)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    _set_cached_limit(cache_key, val)
                    return val

    _set_cached_limit(cache_key, None)
    return None


def _normalize_gemini_target(base_url: str, model: str) -> tuple[str, str]:
    root = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    for suffix in ("/v1beta", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    clean_model = model.strip().removeprefix("google/").removeprefix("models/").strip()
    return root, clean_model


def _fetch_gemini_limit(base_url: str, model: str, auth_mode: str) -> int | None:
    if auth_mode != "api_key":
        return None
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return None
    root, clean_model = _normalize_gemini_target(base_url, model)
    if not clean_model:
        return None
    cache_key = f"gemini:{root}:{clean_model}"
    hit, cached = _get_cached_limit(cache_key)
    if hit:
        return cached

    quoted_model = _url_quote(clean_model, safe="")
    try:
        with httpx.Client(timeout=_NETWORK_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = client.get(
                f"{root}/v1beta/models/{quoted_model}",
                headers={"x-goog-api-key": api_key, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        _LOG.debug("Gemini model metadata lookup skipped for %s: %s", clean_model, exc)
        _set_cached_limit(cache_key, None)
        return None

    raw_limit = data.get("inputTokenLimit") if isinstance(data, dict) else None
    try:
        limit = int(raw_limit) if raw_limit is not None else 0
    except (TypeError, ValueError):
        limit = 0
    resolved = limit if limit > 0 else None
    _set_cached_limit(cache_key, resolved)
    return resolved


def _split_model_key(settings: Settings, model_key: str | None) -> tuple[str, str, str]:
    effective_key = (model_key or "").strip() or settings.active_model_key
    if ":" in effective_key:
        prov_raw, model_name = effective_key.split(":", 1)
        provider = canonical_provider(prov_raw)
        model = model_name.strip()
    else:
        provider = canonical_provider(settings.llm_provider)
        model = effective_key or settings.llm_model
    return provider, model, f"{provider}:{model}"


def resolve_context_limit(
    settings: Settings,
    model_key: str | None = None,
    *,
    allow_network: bool = True,
) -> int | None:
    """Resolve the token context limit for ``model_key`` (or the active model).

    Order of precedence:
    1. Explicit ``context_length`` configured on the model or provider in ``config.yaml``
    2. Cached live provider metadata (OpenRouter ``/models``, Ollama ``/api/show``, Gemini ``models.get``)
    3. Built-in static catalog fallback by model family
    """
    provider, model, canonical_key = _split_model_key(settings, model_key)
    if not model:
        return None

    option = next(
        (opt for opt in settings.available_models if opt.key == canonical_key),
        None,
    )
    if option is not None and option.context_length and option.context_length > 0:
        return option.context_length

    # Check in-memory live metadata cache even when allow_network=False
    if provider == "openrouter":
        or_root = (settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        hit, cached = _lookup_openrouter_cached(or_root, model)
        if hit and cached:
            return cached
        if allow_network:
            live = _fetch_openrouter_limits(or_root, model)
            if live:
                return live
    elif provider == "ollama":
        ol_root = (settings.ollama_base_url or "http://localhost:11434/v1").rstrip("/")
        if ol_root.endswith("/v1"):
            ol_root = ol_root[:-3].rstrip("/")
        hit, cached = _get_cached_limit(f"ollama:{ol_root}:{model}")
        if hit and cached:
            return cached
        if allow_network:
            live = _fetch_ollama_limit(settings.ollama_base_url, model)
            if live:
                return live
    elif provider == "gemini":
        gm_root, clean_model = _normalize_gemini_target(settings.gemini_base_url, model)
        if clean_model:
            hit, cached = _get_cached_limit(f"gemini:{gm_root}:{clean_model}")
            if hit and cached:
                return cached
            if allow_network:
                live = _fetch_gemini_limit(settings.gemini_base_url, model, settings.gemini_auth_mode)
                if live:
                    return live

    return _builtin_context_limit(provider, model)


def resolve_all_model_limits(
    settings: Settings,
    *,
    allow_network: bool = False,
) -> dict[str, int | None]:
    """Return ``{model_key: context_length}`` for every configured model option."""
    limits: dict[str, int | None] = {}
    for opt in settings.available_models:
        limits[opt.key] = resolve_context_limit(
            settings,
            opt.key,
            allow_network=allow_network and opt.key == settings.active_model_key,
        )
    if settings.active_model_key not in limits:
        limits[settings.active_model_key] = resolve_context_limit(
            settings,
            settings.active_model_key,
            allow_network=allow_network,
        )
    return limits


def _estimate_single_message_tokens(message: dict[str, Any]) -> int:
    total_chars = 0
    image_tokens = 0
    content = message.get("content")
    if isinstance(content, str):
        total_chars += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    image_tokens += _IMAGE_PART_TOKENS
                elif "text" in part:
                    total_chars += len(str(part.get("text") or ""))
                else:
                    total_chars += len(json.dumps(part, ensure_ascii=False))
            elif isinstance(part, str):
                total_chars += len(part)

    tool_calls = message.get("tool_calls")
    if tool_calls:
        total_chars += len(json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":")))

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        total_chars += len(reasoning)

    text_tokens = math.ceil(total_chars / _CHARS_PER_TOKEN) if total_chars > 0 else 0
    return _MESSAGE_OVERHEAD_TOKENS + text_tokens + image_tokens


def _estimate_tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    global _TOOL_SCHEMA_TOKENS
    if not tools:
        return 0
    if tools is TOOL_SCHEMAS and _TOOL_SCHEMA_TOKENS is not None:
        return _TOOL_SCHEMA_TOKENS
    serialized = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    tokens = math.ceil(len(serialized) / _CHARS_PER_TOKEN)
    if tools is TOOL_SCHEMAS:
        _TOOL_SCHEMA_TOKENS = tokens
    return tokens


def estimate_messages_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Structurally estimate active prompt tokens after applying image-eviction rules."""
    prepared = relocate_tool_images(normalize_messages(messages))
    msg_tokens = sum(_estimate_single_message_tokens(m) for m in prepared)
    return msg_tokens + _estimate_tools_tokens(tools)


def _resolve_saved_project_model_key(project_dir: Path, settings: Settings) -> str:
    meta_path = project_dir / "project.json"
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                saved_key = str(data.get("model_key") or "").strip()
                if any(opt.key == saved_key for opt in settings.available_models):
                    return saved_key
        except (OSError, ValueError):
            pass
    return settings.active_model_key


def get_project_context_snapshot(
    project_dir: Path,
    settings: Settings,
    *,
    model_key: str | None = None,
    raw_events: list[dict[str, Any]] | None = None,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Compute the project's active context window usage, limit, and fill percentage."""
    effective_key = (
        model_key.strip()
        if isinstance(model_key, str) and model_key.strip()
        else _resolve_saved_project_model_key(project_dir, settings)
    )
    _, _, canonical_key = _split_model_key(settings, effective_key)

    events = (
        raw_events
        if raw_events is not None
        else ConversationStore.read_raw_events(project_dir)
    )

    used_tokens: int | None = None
    source = "estimated"
    for idx in range(len(events) - 1, -1, -1):
        ev = events[idx]
        if not isinstance(ev, dict) or ev.get("type") != "agent_usage":
            continue
        data = ev.get("data")
        if not isinstance(data, dict):
            continue
        raw_ctx = data.get("context_used_tokens")
        raw_prompt = data.get("prompt_tokens")
        raw_comp = data.get("completion_tokens")
        base_used = 0
        try:
            if raw_ctx is not None and int(raw_ctx) > 0:
                base_used = int(raw_ctx)
            elif raw_prompt is not None and int(raw_prompt) > 0:
                base_used = int(raw_prompt) + max(0, int(raw_comp or 0))
        except (TypeError, ValueError):
            base_used = 0
        if base_used <= 0:
            continue

        # `_publish_usage` is recorded immediately before the assistant turn
        # produced by that same LLM call; skip that single assistant message
        # so its completion tokens are not counted twice.
        skipped_turn_assistant = False
        trailing_delta = 0
        for tail_ev in events[idx + 1 :]:
            if not isinstance(tail_ev, dict):
                continue
            role = tail_ev.get("role")
            if not skipped_turn_assistant:
                skipped_turn_assistant = True
                if role == "assistant":
                    continue
            if role in {"user", "assistant", "tool"}:
                trailing_delta += _estimate_single_message_tokens(tail_ev)
        used_tokens = base_used + trailing_delta
        source = "provider"
        break

    if used_tokens is None:
        from agent.core import AgentRunner

        history = ConversationStore.load(project_dir)
        baseline_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": get_system_prompt(
                    bosl2_enabled=settings.openscad_bosl2_enabled
                ),
            },
            AgentRunner._project_state_message(
                project_dir,
                history,
                persist=False,
                user_printer=settings.user_printer,
                user_nozzle=settings.user_nozzle,
                user_filament=settings.user_filament,
                user_additional_info=settings.user_additional_info,
            ),
            *history,
        ]
        used_tokens = estimate_messages_tokens(baseline_messages, tools=TOOL_SCHEMAS)
        source = "estimated"

    limit_tokens = resolve_context_limit(
        settings,
        canonical_key,
        allow_network=allow_network,
    )
    model_limits = resolve_all_model_limits(settings, allow_network=False)
    if canonical_key:
        model_limits[canonical_key] = limit_tokens

    percent = (
        round((used_tokens / limit_tokens) * 100.0, 1)
        if limit_tokens and limit_tokens > 0
        else None
    )
    remaining_tokens = (
        max(0, limit_tokens - used_tokens)
        if limit_tokens and limit_tokens > 0
        else None
    )
    return {
        "used_tokens": used_tokens,
        "limit_tokens": limit_tokens,
        "remaining_tokens": remaining_tokens,
        "percent": percent,
        "source": source,
        "model_key": canonical_key,
        "model_limits": model_limits,
    }
