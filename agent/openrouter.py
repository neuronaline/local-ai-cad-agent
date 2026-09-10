"""Thin OpenRouter adapter over the shared ChatCompletionsClient.

All retry logic, stream parsing, and image fallback live in the base class.
This module only carries the OpenRouter-specific bits: request headers,
session/cache hints, and provider routing payload.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from agent.llm_base import ChatCompletionsClient, post_with_cancel
from agent.settings import Settings


class OpenRouterClient(ChatCompletionsClient):
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
            "X-OpenRouter-Title": self.settings.openrouter_app_title,
        }
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        return headers

    def _apply_provider_payload(self, payload: dict[str, Any]) -> None:
        if self.session_id:
            payload["session_id"] = hashlib.sha256(self.session_id.encode()).hexdigest()[:64]
        # Tag subordinate evaluators through OpenRouter's documented tracing
        # surface so the request remains valid and observability can
        # distinguish reviewer spans from the parent agent loop. Only the
        # role strings produced by the agent — currently ``"reviewer"``
        # from :func:`agent.cad_review.review_cad` — are forwarded; the
        # default ``None`` skips the field entirely.
        if self.agent_role:
            trace = payload.setdefault("trace", {})
            if isinstance(trace, dict):
                trace.setdefault("span_name", self.agent_role)
        provider_order = self._provider_order()
        if (
            self.settings.openrouter_enable_anthropic_cache
            and self.settings.openrouter_model.startswith("anthropic/")
        ):
            # Treat the request as a direct Anthropic routing only when
            # every entry — or no entry at all — names Anthropic. A
            # multi-provider list with one Anthropic entry still routes via
            # a third-party upstream, so Bedrock/Vertex-style explicit
            # breakpoints are required.
            direct_anthropic = (
                not provider_order
                or all(slug == "anthropic" for slug in provider_order)
            )
            if direct_anthropic:
                # OpenRouter's automatic breakpoint advances without changing
                # historical message bytes, but it only routes to Anthropic's
                # own endpoint.
                payload["cache_control"] = {"type": "ephemeral"}
            else:
                # Bedrock and Vertex reject the top-level automatic control.
                # An explicit stable-system breakpoint works across all
                # Anthropic-compatible endpoints.
                self._mark_first_system_message(payload)
        if self.settings.openrouter_reasoning_effort:
            payload["reasoning"] = {
                "effort": self.settings.openrouter_reasoning_effort,
                "exclude": False,
            }
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
        """Mark text deterministically so old Gemini wire messages never change.

        OpenRouter uses the final explicit breakpoint for Gemini, but permits
        multiple markers. Marking only the latest text moves the marker on the
        next request and rewrites the previous prefix. Applying the same
        transform to every text block keeps the old prefix byte-stable while
        still making the final marker advance as the conversation grows.
        """
        if (
            not self.settings.openrouter_enable_gemini_cache
            or not self.settings.openrouter_model.startswith("google/gemini-")
        ):
            return
        for message in payload["messages"]:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    continue
                part["cache_control"] = {"type": "ephemeral"}

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

    def _build_payload(self, messages, tools):
        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": self.sanitize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
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
