"""Thin Ollama adapter over the shared ChatCompletionsClient.

Connects to a local or remote Ollama instance via its OpenAI-compatible
endpoint (by default ``http://localhost:11434/v1``).
"""
from __future__ import annotations

import os
from typing import Any

from agent.llm_base import ChatCompletionsClient, post_with_cancel
from agent.settings import Settings

DEFAULT_BASE_URL = "http://localhost:11434/v1"
API_KEY_ENV = "OLLAMA_API_KEY"


class OllamaClient(ChatCompletionsClient):
    requires_api_key = False

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="Ollama")

    @staticmethod
    def _api_key() -> str:
        return os.getenv(API_KEY_ENV, "").strip()

    def _endpoint(self) -> str:
        base = (self.settings.ollama_base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/chat/completions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_payload(self, messages, tools):
        wire_messages = self.sanitize_messages(messages)
        # Drop reasoning/thinking fields that might cause 400 errors on local models
        for message in wire_messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                message.pop("reasoning", None)
                message.pop("reasoning_details", None)
                message.pop("reasoning_content", None)
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": wire_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if self.settings.llm_max_completion_tokens:
            # Ollama supports max_tokens (max_completion_tokens is often ignored)
            payload["max_tokens"] = self.settings.llm_max_completion_tokens
        if self.settings.ollama_reasoning_effort:
            payload["reasoning_effort"] = self.settings.ollama_reasoning_effort
            payload["think"] = True
        return payload

    def _post(self, payload, headers):
        return post_with_cancel(
            url=self._endpoint(),
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.ollama_timeout_seconds,
            stop_event=self.stop_event,
        )
