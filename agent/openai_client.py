"""Thin OpenAI adapter over the shared ChatCompletionsClient.

All retry logic, stream parsing, and image fallback live in the base class.
This module only carries the OpenAI-specific bits: the endpoint, the env var
name, and the request headers/payload.
"""
from __future__ import annotations

import os
from typing import Any

from agent.llm_base import ChatCompletionsClient, post_with_cancel, sanitize_messages
from agent.prompt import get_prompt_cache_key
from agent.settings import Settings

DEFAULT_BASE_URL = "https://api.openai.com/v1"
API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIClient(ChatCompletionsClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="OpenAI")

    @classmethod
    def sanitize_messages(cls, messages):
        return sanitize_messages(messages)

    @staticmethod
    def _api_key() -> str:
        return os.getenv(API_KEY_ENV, "").strip()

    def _endpoint(self) -> str:
        base = (self.settings.openai_base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/chat/completions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages, tools):
        wire_messages = self.sanitize_messages(messages)
        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": wire_messages,
            "prompt_cache_key": get_prompt_cache_key(self.session_id),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.settings.openai_model.startswith("gpt-5.6"):
            self._mark_stable_system_prefix(wire_messages)
        if tools:
            payload["tools"] = tools
        if self.settings.llm_max_completion_tokens:
            payload["max_completion_tokens"] = self.settings.llm_max_completion_tokens
        if self.settings.openai_reasoning_effort:
            payload["reasoning_effort"] = self.settings.openai_reasoning_effort
        return payload

    @staticmethod
    def _mark_stable_system_prefix(messages: list[dict[str, Any]]) -> None:
        """Add the GPT-5.6 explicit breakpoint after static instructions."""
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            marker = {"mode": "explicit"}
            if isinstance(content, str):
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "prompt_cache_breakpoint": marker,
                    }
                ]
                return
            if isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["prompt_cache_breakpoint"] = marker
                        return

    def _post(self, payload, headers):
        return post_with_cancel(
            url=self._endpoint(),
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.openai_timeout_seconds,
            stop_event=self.stop_event,
        )
