"""Thin OpenRouter adapter over the shared ChatCompletionsClient.

All retry logic, stream parsing, and image fallback live in the base class.
This module only carries the OpenRouter-specific bits: request headers,
session/cache hints, and provider routing payload.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from agent.llm_base import (
    ChatCompletionsClient,
    post_with_cancel,
    sanitize_assistant_message,  # noqa: F401 - backward-compatible public re-export.
    sanitize_messages,
)
from agent.settings import Settings


class OpenRouterClient(ChatCompletionsClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="OpenRouter")

    @classmethod
    def sanitize_messages(cls, messages):
        return sanitize_messages(messages)

    @staticmethod
    def _api_key() -> str:
        return os.getenv("OPENROUTER_API_KEY", "").strip()

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
            provider: dict[str, Any] = {}
            if self.settings.openrouter_force_provider:
                provider.update(
                    {
                        "only": [self.settings.openrouter_provider],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                )
            else:
                # Explicit provider order disables OpenRouter's sticky routing.
                provider["order"] = [self.settings.openrouter_provider]
            payload["provider"] = provider

    def _apply_gemini_cache_breakpoint(self, payload: dict[str, Any]) -> None:
        """Mark the stable system prompt cacheable for Gemini on OpenRouter."""
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
