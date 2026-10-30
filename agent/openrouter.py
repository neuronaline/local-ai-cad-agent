"""Thin OpenRouter adapter over the shared ChatCompletionsClient.

All retry logic, stream parsing, and image fallback live in the base class.
This module only carries the OpenRouter-specific bits: request headers,
session/cache hints, and provider routing payload.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from agent.llm_base import ChatCompletionsClient, is_continuation_messages, post_with_cancel
from agent.settings import Settings


class OpenRouterClient(ChatCompletionsClient):
    # Preserving reasoning blocks is critical for tool use continuity on OpenRouter,
    # specifically for models like Claude and OpenAI reasoning models where multi-turn
    # tool calling requires preserving the reasoning chain across tool responses.
    preserve_reasoning = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="OpenRouter")

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
            "X-Title": self.settings.openrouter_app_title,
            "X-OpenRouter-Title": self.settings.openrouter_app_title,
        }
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        if self.session_id:
            session_hash = hashlib.sha256(self.session_id.encode()).hexdigest()[:64]
            headers["X-Session-ID"] = session_hash
            headers["x-session-id"] = session_hash
        return headers

    def _is_continuation(self, messages: list[dict[str, Any]] | None) -> bool:
        return is_continuation_messages(messages)

    def _apply_provider_payload(
        self, payload: dict[str, Any], messages: list[dict[str, Any]] | None = None
    ) -> None:
        if self.session_id:
            payload["session_id"] = hashlib.sha256(self.session_id.encode()).hexdigest()[:64]
        # Reserved-for-future trace hook: the structured ``cad_review`` evaluator
        # that historically produced the only known role string was removed
        # from the tool surface, so ``self.agent_role`` is always ``None``
        # today. The branch is preserved so a future sub-agent can tag its
        # ``trace.span_name`` on OpenRouter without changing the cache
        # prefix; the default ``None`` skips the field entirely.
        if self.agent_role:
            trace = payload.setdefault("trace", {})
            if isinstance(trace, dict):
                trace.setdefault("span_name", self.agent_role)
        provider_order = self._provider_order()
        if (
            self.settings.openrouter_enable_anthropic_cache
            and self.settings.openrouter_model.startswith("anthropic/")
        ):
            # OpenRouter top-level cache_control enables automatic breakpoint advancing
            # across Anthropic, Google Vertex AI, Azure, and Amazon Bedrock endpoints.
            payload["cache_control"] = {"type": "ephemeral"}
            # Also explicitly mark the immutable system message and the last eligible
            # message to ensure breakpoint preservation across any proxy or sub-provider.
            self._mark_first_system_message(payload)
            self._mark_last_eligible_message(payload)

        is_continuation = self._is_continuation(messages)
        effort = (
            self.settings.openrouter_reasoning_effort.lower().strip()
            if self.settings.openrouter_reasoning_effort
            else None
        )
        max_tokens_cfg = self.settings.openrouter_reasoning_max_tokens

        # OpenRouter strictly requires choosing ONLY ONE of "effort" or "max_tokens".
        # Sending both simultaneously results in a 400 Bad Request error.
        if effort in ("none", "off", "0"):
            payload["reasoning"] = {"effort": "none"}
        elif effort:
            if is_continuation:
                payload["reasoning"] = {"effort": "low"}
            else:
                payload["reasoning"] = {"effort": effort}
        elif is_continuation:
            payload["reasoning"] = {"max_tokens": 512}
        elif max_tokens_cfg is not None and max_tokens_cfg > 0:
            payload["reasoning"] = {"max_tokens": max(256, int(max_tokens_cfg))}

        if provider_order:
            if self.settings.openrouter_force_provider:
                # Legacy single-provider pinning: stick to one upstream and
                # skip OpenRouter's own fallback fanout. ``provider_order``
                # is validated at startup to contain exactly one entry when
                # ``force_provider`` is on, so this branch always sees a
                # 1-element list.
                payload["provider"] = {
                    "only": list(provider_order),
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            else:
                # Ordered priority list (highest first). OpenRouter walks the
                # list and falls back to its internal pool if every named
                # upstream is unavailable. Disabling its sticky routing keeps
                # the call shape stable across the LLM client base class.
                payload["provider"] = {"order": list(provider_order)}

    def _provider_order(self) -> tuple[str, ...]:
        """Resolve the effective OpenRouter provider routing order.

        Prefers the explicit ``openrouter_provider_order`` tuple (highest
        priority first). Falls back to the legacy single-string
        ``openrouter_provider`` setting so existing configs keep working
        until users migrate to ``provider_order``.
        """
        order = self.settings.openrouter_provider_order
        if order:
            return order
        legacy = self.settings.openrouter_provider
        return (legacy,) if legacy else ()

    def _apply_gemini_cache_breakpoint(self, payload: dict[str, Any]) -> None:
        """Mark a single Gemini cache breakpoint on the system message.

        OpenRouter's Gemini path accepts multiple ``cache_control`` markers
        per request, but only the *last* one determines the cached prefix.
        Walking every message and rewriting every text block into
        ``list[dict]`` (just to splice a marker on each one) bloats the
        payload and triggers a transform on every chat-completions call.
        The stable system message is the prefix that we want cached — a
        single explicit marker on that slot is both cheaper and sufficient
        for prompt-cache stability.
        """
        if (
            not self.settings.openrouter_enable_gemini_cache
            or not self.settings.openrouter_model.startswith("google/gemini-")
        ):
            return
        self._mark_first_system_message(payload)

    @staticmethod
    def _mark_first_system_message(payload: dict[str, Any]) -> None:
        for message in payload.get("messages", []):
            if message.get("role") != "system":
                continue
            content = message.get("content")
            marker = {"type": "ephemeral"}
            if isinstance(content, str):
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": marker}
                ]
                return
            if isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["cache_control"] = marker
                        return

    @staticmethod
    def _mark_last_eligible_message(payload: dict[str, Any]) -> None:
        """Mark an explicit cache breakpoint on the last eligible turn message."""
        messages = payload.get("messages", [])
        for message in reversed(messages):
            role = message.get("role")
            if role not in ("user", "tool", "assistant"):
                continue
            content = message.get("content")
            marker = {"type": "ephemeral"}
            if isinstance(content, str) and content.strip():
                message["content"] = [
                    {"type": "text", "text": content, "cache_control": marker}
                ]
                return
            if isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") in ("text", "image_url"):
                        part["cache_control"] = marker
                        return

    def _build_payload(self, messages, tools):
        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": self.sanitize_messages(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if self.settings.llm_max_completion_tokens:
            # OpenRouter normalises the legacy ``max_tokens`` across every
            # upstream provider, whereas the newer ``max_completion_tokens``
            # (OpenAI-specific) is not advertised by providers such as
            # ``google-vertex/global``. When callers pin a provider with
            # ``force_provider`` + ``require_parameters``, OpenRouter rejects
            # any parameter that no endpoint handles, surfacing as a 404 with
            # the message "No endpoints found that can handle the requested
            # parameters". Using ``max_tokens`` keeps the budget applied while
            # staying compatible with every supported provider.
            payload["max_tokens"] = self.settings.llm_max_completion_tokens
        self._apply_provider_payload(payload, messages)
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
