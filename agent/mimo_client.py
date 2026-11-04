"""Xiaomi MiMo adapter over the shared ChatCompletionsClient.

Connects to Xiaomi's MiMo API Open Platform (by default https://api.xiaomimimo.com/v1)
supporting mimo-v2.6 models with deep thinking (reasoning_content), function calling,
and multimodal inputs.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from agent.llm_base import (
    ChatCompletionsClient,
    is_continuation_messages,
    post_with_cancel,
)
from agent.settings import Settings

DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
API_KEY_ENV = "MIMO_API_KEY"


class MiMoClient(ChatCompletionsClient):
    # Preserving reasoning blocks is critical for tool use continuity on Xiaomi MiMo:
    # the MiMo API contract strictly requires passing back historical reasoning_content
    # on assistant messages that executed tool calls when thinking mode is enabled.
    preserve_reasoning = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="Xiaomi MiMo")

    @staticmethod
    def _api_key() -> str:
        return os.getenv(API_KEY_ENV, "").strip()

    def _endpoint(self) -> str:
        base = (self.settings.mimo_base_url or DEFAULT_BASE_URL).rstrip("/")
        return f"{base}/chat/completions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "Content-Type": "application/json",
        }
        if self.session_id:
            session_hash = hashlib.sha256(self.session_id.encode()).hexdigest()[:64]
            headers["X-Session-ID"] = session_hash
            headers["x-session-id"] = session_hash
        return headers

    def _build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        wire_messages = self.sanitize_messages(messages)

        # Determine thinking mode:
        # continuation turns (token budget recovery or nudge) disable thinking to force tool generation;
        # explicit reasoning_effort == 'none' also disables thinking.
        is_continuation = is_continuation_messages(messages)
        effort = (self.settings.mimo_reasoning_effort or "").lower().strip()
        thinking_enabled = not (is_continuation or effort in ("none", "disabled", "off", "0", "false"))
        thinking_type = "enabled" if thinking_enabled else "disabled"

        # Adapt assistant messages for MiMo's API contract:
        # 1. Map reasoning -> reasoning_content.
        # 2. When thinking is enabled, any assistant message with tool_calls MUST retain reasoning_content.
        # 3. Clean any provider-specific leftovers (e.g. Gemini thought_signature/extra_content) from tool calls.
        for message in wire_messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                reasoning = message.pop("reasoning", None)
                message.pop("reasoning_details", None)
                if thinking_enabled:
                    if isinstance(reasoning, str) and reasoning.strip():
                        message["reasoning_content"] = reasoning
                    elif message.get("tool_calls") and "reasoning_content" not in message:
                        message["reasoning_content"] = ""
                else:
                    message.pop("reasoning_content", None)

                for tc in message.get("tool_calls", []):
                    if isinstance(tc, dict):
                        tc.pop("thought_signature", None)
                        tc.pop("extra_content", None)

        payload: dict[str, Any] = {
            "model": self.settings.mimo_model,
            "messages": wire_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "thinking": {"type": thinking_type},
        }

        if tools:
            payload["tools"] = tools
        if self.settings.llm_max_completion_tokens:
            payload["max_completion_tokens"] = self.settings.llm_max_completion_tokens

        return payload

    def _idle_timeout(self) -> float:
        return float(self.settings.mimo_timeout_seconds)

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        return post_with_cancel(
            url=self._endpoint(),
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.mimo_timeout_seconds,
            stop_event=self.stop_event,
        )
