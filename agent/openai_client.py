"""Small, testable OpenAI Chat Completions client.

OpenAI's ``/v1/chat/completions`` endpoint speaks the same wire protocol as
OpenRouter, so this adapter reuses the shared chat-completions helpers from
``agent.llm_base`` and only overrides the provider-specific bits: the API
endpoint, which env var supplies the key, and the request headers/payload.
"""
from __future__ import annotations

import os
from typing import Any

from agent.llm_base import (
    parse_chat_stream,
    post_with_cancel,
    retry_delay,
    sanitize_messages,
    sleep_with_cancel,
    without_images,
)
from agent.settings import Settings

DEFAULT_BASE_URL = "https://api.openai.com/v1"
API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stop_event = None  # set by AgentRunner before chat()
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
        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": self.sanitize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if self.settings.openai_reasoning_effort:
            payload["reasoning_effort"] = self.settings.openai_reasoning_effort
        return payload

    def _try_image_fallback(self, payload, response):
        messages_without_images, removed = without_images(payload["messages"])
        if removed:
            response.close()
            payload["messages"] = messages_without_images
        return removed

    def _stream_response(self, response):
        result = parse_chat_stream(
            response,
            provider_label="OpenAI",
            stop_event=self.stop_event,
            stream_callback=self.stream_callback,
        )
        usage = result.get("usage") if isinstance(result, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        return result

    def _post(self, payload, headers):
        return post_with_cancel(
            url=self._endpoint(),
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.openai_timeout_seconds,
            stop_event=self.stop_event,
        )

    def chat(self, messages, tools=None):
        self.last_usage = None
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(f"{API_KEY_ENV} is not configured.")
        payload = self._build_payload(messages, tools)
        headers = self._build_headers(api_key)

        image_fallback_used = False
        for attempt in range(3):
            response = None
            try:
                response = self._post(payload, headers)
            except Exception:
                if attempt == 2:
                    raise
            else:
                # Vision inputs are rare for plain OpenAI requests; honor the
                # same 400/404 image-fallback behavior used by OpenRouter so
                # models without vision still complete.
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
                        raise RuntimeError(f"OpenAI {response.status_code}: {body_preview}")
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
                delay = retry_delay(response, attempt)
                sleep_with_cancel(delay, self.stop_event)
        raise RuntimeError("OpenAI retry loop ended unexpectedly.")