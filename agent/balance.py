"""Provider account balance and spent credit tracking service.

Queries provider APIs (such as OpenRouter's /credits and /auth/key or /key) to retrieve
account balance, total spent credits, and API key usage limits.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

from agent.settings import Settings, canonical_provider

_LOG = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 30.0
_balance_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def fetch_provider_balance(
    settings: Settings,
    provider: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Retrieve the current account and key balance for the specified or active LLM provider.

    Caches results per provider for 30 seconds unless ``force_refresh`` is True.
    """
    target_provider = canonical_provider(provider or settings.llm_provider)
    now = time.monotonic()

    if not force_refresh:
        with _cache_lock:
            cached = _balance_cache.get(target_provider)
            if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
                return dict(cached[1])

    result: dict[str, Any]

    if target_provider == "openrouter":
        result = _fetch_openrouter_balance(settings)
    elif target_provider == "ollama":
        result = {
            "provider": "ollama",
            "supported": True,
            "is_local": True,
            "currency": "USD",
            "balance": 0.0,
            "total_spent": 0.0,
            "key_spent": 0.0,
            "label": "Ollama (Local / Free)",
        }
    elif target_provider == "gemini":
        result = _fetch_gemini_balance(settings)
    elif target_provider == "vertex":
        result = _fetch_vertex_balance(settings)
    else:
        result = {
            "provider": target_provider,
            "supported": False,
            "currency": "USD",
            "reason": f"Provider '{target_provider}' does not expose an account balance API for standard keys.",
        }

    with _cache_lock:
        _balance_cache[target_provider] = (time.monotonic(), result)

    return dict(result)


def _fetch_openrouter_balance(settings: Settings) -> dict[str, Any]:
    """Query OpenRouter credit and key usage endpoints."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {
            "provider": "openrouter",
            "supported": True,
            "error": "OPENROUTER_API_KEY is not set in environment or .env.",
            "currency": "USD",
        }

    base = (settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    credits_data: dict[str, Any] = {}
    key_data: dict[str, Any] = {}
    error_msg: str | None = None

    # 1. Query /credits endpoint for total account balance
    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.get(f"{base}/credits", headers=headers)
            if resp.status_code == 200:
                data = resp.json().get("data")
                if isinstance(data, dict):
                    credits_data = data
            elif resp.status_code not in (401, 403):
                _LOG.warning("OpenRouter /credits returned status %d: %s", resp.status_code, resp.text[:200])
    except (httpx.HTTPError, OSError, ValueError) as exc:
        _LOG.debug("OpenRouter /credits request failed: %s", exc)

    # 2. Query /auth/key (or /key) endpoint for key-specific limits and usage
    for key_endpoint in (f"{base}/auth/key", f"{base}/key"):
        try:
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(key_endpoint, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data")
                    if isinstance(data, dict):
                        key_data = data
                        break
                elif resp.status_code in (401, 403):
                    error_msg = f"HTTP {resp.status_code}: Invalid or unauthorized API key."
                    break
                else:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except (httpx.HTTPError, OSError, ValueError) as exc:
            _LOG.debug("OpenRouter %s request failed: %s", key_endpoint, exc)
            if not error_msg:
                error_msg = str(exc)

    if not credits_data and not key_data:
        return {
            "provider": "openrouter",
            "supported": True,
            "error": error_msg or "Failed to connect to OpenRouter API.",
            "currency": "USD",
        }

    total_credits = credits_data.get("total_credits")
    total_usage = credits_data.get("total_usage")

    key_usage = key_data.get("usage")
    key_limit = key_data.get("limit")
    limit_remaining = key_data.get("limit_remaining")
    eff_budget = key_data.get("effective_budget")
    eff_remaining = eff_budget.get("remaining_usd") if isinstance(eff_budget, dict) else None
    usage_daily = key_data.get("usage_daily")
    usage_monthly = key_data.get("usage_monthly")
    is_free_tier = key_data.get("is_free_tier", False)
    key_label = key_data.get("label")

    # Remaining balance calculation (prioritize account credits, then key limit remaining)
    balance: float | None = None
    if total_credits is not None and total_usage is not None:
        try:
            balance = max(0.0, float(total_credits) - float(total_usage))
        except (ValueError, TypeError):
            pass
    elif limit_remaining is not None:
        try:
            balance = max(0.0, float(limit_remaining))
        except (ValueError, TypeError):
            pass
    elif eff_remaining is not None:
        try:
            balance = max(0.0, float(eff_remaining))
        except (ValueError, TypeError):
            pass
    elif key_limit is not None and key_usage is not None:
        try:
            balance = max(0.0, float(key_limit) - float(key_usage))
        except (ValueError, TypeError):
            pass

    return {
        "provider": "openrouter",
        "supported": True,
        "currency": "USD",
        "balance": round(balance, 4) if balance is not None else None,
        "total_credits": round(float(total_credits), 4) if total_credits is not None else None,
        "total_spent": round(float(total_usage), 4) if total_usage is not None else None,
        "key_spent": round(float(key_usage), 4) if key_usage is not None else None,
        "key_limit": round(float(key_limit), 4) if key_limit is not None else None,
        "usage_daily": round(float(usage_daily), 4) if usage_daily is not None else None,
        "usage_monthly": round(float(usage_monthly), 4) if usage_monthly is not None else None,
        "is_free_tier": bool(is_free_tier),
        "label": key_label or "OpenRouter Key",
        "error": error_msg,
    }


def _fetch_gemini_balance(settings: Settings) -> dict[str, Any]:
    """Verify Google AI Studio API key and connection status."""
    auth_mode = getattr(settings, "gemini_auth_mode", "api_key")
    if auth_mode == "gcloud":
        from agent.gcloud_auth import check_gcloud_status

        ok, detail = check_gcloud_status()
        return {
            "provider": "gemini",
            "supported": True,
            "status": "active" if ok else "error",
            "auth_mode": "gcloud",
            "label": "Google Cloud Vertex AI" if ok else "gcloud Error",
            "model": settings.gemini_model,
            "currency": "USD",
            "error": None if ok else (detail or "gcloud CLI not authenticated"),
        }

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {
            "provider": "gemini",
            "supported": True,
            "status": "unconfigured",
            "auth_mode": "api_key",
            "label": "Google AI Studio",
            "model": settings.gemini_model,
            "currency": "USD",
            "error": "GEMINI_API_KEY is not set in environment or .env.",
        }

    base = (settings.gemini_base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    probe_url = f"{base}/v1beta/models"
    try:
        with httpx.Client(timeout=4.0) as client:
            resp = client.get(probe_url, headers={"x-goog-api-key": api_key})
            if resp.status_code == 200:
                return {
                    "provider": "gemini",
                    "supported": True,
                    "status": "active",
                    "auth_mode": "api_key",
                    "label": "Google AI Studio",
                    "model": settings.gemini_model,
                    "currency": "USD",
                    "is_free_tier": True,
                }
            err_msg = f"HTTP {resp.status_code}: {resp.text[:150]}"
            _LOG.warning("Google AI Studio probe returned %s", err_msg)
            return {
                "provider": "gemini",
                "supported": True,
                "status": "error",
                "auth_mode": "api_key",
                "label": "Google AI Studio (Invalid Key)",
                "model": settings.gemini_model,
                "currency": "USD",
                "error": err_msg,
            }
    except Exception as exc:
        _LOG.debug("Google AI Studio probe connection failed: %s", exc)
        return {
            "provider": "gemini",
            "supported": True,
            "status": "error",
            "auth_mode": "api_key",
            "label": "Google AI Studio (Connection Error)",
            "model": settings.gemini_model,
            "currency": "USD",
            "error": f"Failed to reach Google AI Studio: {exc}",
        }


def _fetch_vertex_balance(settings: Settings) -> dict[str, Any]:
    """Verify Google Cloud Vertex AI credentials status."""
    from agent.gcloud_auth import check_gcloud_status

    ok, detail = check_gcloud_status()
    return {
        "provider": "vertex",
        "supported": True,
        "status": "active" if ok else "error",
        "auth_mode": "gcloud",
        "label": "Vertex AI (gcloud)" if ok else "gcloud Error",
        "model": settings.vertex_model,
        "currency": "USD",
        "error": None if ok else (detail or "gcloud CLI not authenticated"),
    }
