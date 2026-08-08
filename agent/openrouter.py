"""Small, testable OpenRouter chat-completions client.

The wire protocol is the OpenAI Chat Completions API, so this module reuses
the provider-neutral helpers in ``agent.llm_base`` and only carries the
OpenRouter-specific bits (request headers, session/cache hints, provider
routing payload).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import requests

from agent.llm_base import (
    parse_chat_stream,
    post_with_cancel,
    retry_delay,
    sanitize_assistant_message,  # noqa: F401 - backward-compatible public re-export.
    sanitize_messages,
    sleep_with_cancel,
    without_images,
)
from agent.settings import Settings


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stop_event = None
        self.session_id: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.stream_callback = None

    @staticmethod
    def normalize_messages(messages):
        from agent.llm_base import normalize_messages as _normalize

        return _normalize(messages)

    @classmethod
    def sanitize_messages(cls, messages):
        return sanitize_messages(messages)

    @staticmethod
    def _retry_delay(response, attempt):
        return retry_delay(response, attempt)

    def _endpoint(self) -> str:
        base = (self.settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        return f"{base}/chat/completions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.settings.openrouter_app_title,
        }
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        return headers

    def _apply_provider_payload(self, payload: dict[str, Any]) -> None:
        if self.session_id:
            payload["session_id"] = hashlib.sha256(self.session_id.encode()).hexdigest()[:64]
        if (
            self.settings.openrouter_enable_anthropic_cache
            and self.settings.openrouter_model.startswith("anthropic/")
        ):
            payload["cache_control"] = {"type": "ephemeral"}
        if self.settings.openrouter_reasoning_effort:
            payload["reasoning"] = {
                "effort": self.settings.openrouter_reasoning_effort,
                "exclude": False,
            }
        if self.settings.openrouter_provider:
            provider: dict[str, Any] = {"order": [self.settings.openrouter_provider]}
            if self.settings.openrouter_force_provider:
                provider.update(
                    {
                        "only": [self.settings.openrouter_provider],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                )
            payload["provider"] = provider

    def _apply_gemini_cache_breakpoint(self, payload: dict[str, Any]) -> None:
        """Mark the stable system prompt cacheable for Gemini on OpenRouter.

        Gemini prompt caching requires a ``cache_control`` marker on a content
        block. The system prompt is stable across agent turns, so it is the
        safest cache boundary; dynamic constraints remain later user messages.
        """
        if (
            not self.settings.openrouter_enable_gemini_cache
            or not self.settings.openrouter_model.startswith("google/gemini-")
        ):
            return
        for message in payload["messages"]:
            if message.get("role") != "system" or not isinstance(message.get("content"), str):
                continue
            message["content"] = [{
                "type": "text",
                "text": message["content"],
                "cache_control": {"type": "ephemeral"},
            }]
            return

    def _build_payload(self, messages, tools):
        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": self.sanitize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        self._apply_provider_payload(payload)
        self._apply_gemini_cache_breakpoint(payload)
        return payload

    def _post(self, payload, headers):
        return post_with_cancel(
            url=self._endpoint(),
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.openrouter_timeout_seconds,
            stop_event=self.stop_event,
        )

    def _stream_response(self, response):
        result = parse_chat_stream(
            response,
            provider_label="OpenRouter",
            stop_event=self.stop_event,
            stream_callback=self.stream_callback,
        )
        usage = result.get("usage") if isinstance(result, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        return result

    def _try_image_fallback(self, payload, response):
        messages_without_images, removed = without_images(payload["messages"])
        if removed:
            response.close()
            payload["messages"] = messages_without_images
        return removed

    def chat(self, messages, tools=None):
        self.last_usage = None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        payload = self._build_payload(messages, tools)
        headers = self._build_headers(api_key)

        image_fallback_used = False
        for attempt in range(3):
            response: requests.Response | None = None
            try:
                response = self._post(payload, headers)
            except Exception:
                if attempt == 2:
                    raise
            else:
                if response.status_code in {400, 404} and not image_fallback_used:
                    if self._try_image_fallback(payload, response):
                        image_fallback_used = True
                        continue
                    body_preview = ""
                    try:
                        body_preview = response.text[:500]
                    except Exception:  # noqa: BLE001,S110
                        pass
                    if body_preview:
                        raise RuntimeError(f"OpenRouter {response.status_code}: {body_preview}")
                if response.status_code not in {408, 429} and response.status_code < 500:
                    response.raise_for_status()
                    if hasattr(response, "iter_lines"):
                        return self._stream_response(response)
                    body = response.json()
                    self.last_usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
                    return body
                if attempt == 2:
                    response.raise_for_status()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            if attempt < 2:
                delay = self._retry_delay(response, attempt)
                sleep_with_cancel(delay, self.stop_event)
        raise RuntimeError("OpenRouter retry loop ended unexpectedly.")
