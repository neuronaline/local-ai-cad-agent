"""Small, testable OpenRouter chat-completions client."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import requests

from agent.settings import Settings


def sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a model response while retaining standard content and tool fields."""
    sanitized = deepcopy(message)
    sanitized.pop("reasoning", None)
    sanitized.pop("reasoning_details", None)
    for key in list(sanitized):
        if key.startswith("_"):
            sanitized.pop(key, None)
    return sanitized


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stop_event: threading.Event | None = None
        self.session_id: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.stream_callback: Callable[[dict[str, Any]], None] | None = None

    @staticmethod
    def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return an API-safe copy with exactly one leading system message.

        OpenRouter's cache key benefits from one stable system prefix. Prior
        messages are immutable references — only assistant messages (which get
        sanitized downstream) are deep-copied where needed.
        """
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []
        for item in messages:
            if item.get("role") == "system" and isinstance(item.get("content"), str):
                system_parts.append(item["content"])
            else:
                non_system.append(item)
        if not system_parts:
            return non_system
        return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system

    @classmethod
    def sanitize_messages(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip only unsupported assistant metadata, retaining content and tools.

        System messages and prior turns are already immutable. Only deepcopy the
        last user message (which may hold image content) and assistant messages
        that need sanitization.
        """
        sanitized = cls.normalize_messages(messages)
        for index, message in enumerate(sanitized):
            if message.get("role") != "assistant":
                continue
            sanitized[index] = sanitize_assistant_message(message)
        return sanitized

    @staticmethod
    def _retry_delay(response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass
        return float(2**attempt)

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> requests.Response:
        """Run the blocking HTTP request behind a cancellable agent-loop boundary."""
        results: queue.Queue[requests.Response | BaseException] = queue.Queue(maxsize=1)

        def request_worker() -> None:
            try:
                results.put(
                    requests.post(
                        f"{self.settings.openrouter_base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=self.settings.openrouter_timeout_seconds,
                    )
                )
            except BaseException as error:  # noqa: BLE001 - Propagate worker failures unchanged.
                results.put(error)

        worker = threading.Thread(target=request_worker, daemon=True)
        if self.stop_event and self.stop_event.is_set():
            raise RuntimeError("OpenRouter request cancelled.")
        worker.start()
        while worker.is_alive():
            worker.join(timeout=0.1)
            if self.stop_event and self.stop_event.is_set():
                raise RuntimeError("OpenRouter request cancelled.")
        result = results.get()
        if isinstance(result, BaseException):
            raise result
        return result

    @staticmethod
    def _without_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        """Remove image parts while retaining their accompanying text."""
        stripped = deepcopy(messages)
        removed = False
        for message in stripped:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            text_parts = [
                part for part in content
                if isinstance(part, dict) and part.get("type") != "image_url"
            ]
            if len(text_parts) != len(content):
                removed = True
                message["content"] = text_parts or "An attached image was unavailable."
        return stripped, removed

    def _try_image_fallback(self, payload: dict[str, Any], response: requests.Response) -> bool:
        """Remove images from the payload and retry. Returns True if images were stripped."""
        messages_without_images, removed = self._without_images(payload["messages"])
        if removed:
            response.close()
            payload["messages"] = messages_without_images
        return removed

    def _stream_response(self, response: requests.Response) -> dict[str, Any]:
        """Consume OpenRouter SSE and rebuild a chat-completions response."""
        content = ""
        role = "assistant"
        tool_calls: dict[int, dict[str, Any]] = {}
        saw_done = False
        finish_reason: str | None = None

        for raw_line in response.iter_lines():
            if self.stop_event and self.stop_event.is_set():
                response.close()
                raise RuntimeError("OpenRouter request cancelled.")
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as error:
                raise RuntimeError("OpenRouter returned malformed streaming JSON.") from error
            stream_error = chunk.get("error")
            if stream_error:
                if isinstance(stream_error, dict):
                    detail = stream_error.get("message") or json.dumps(stream_error)
                else:
                    detail = str(stream_error)
                raise RuntimeError(f"OpenRouter stream failed: {detail}")
            if isinstance(chunk.get("usage"), dict):
                self.last_usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            if finish_reason == "error":
                raise RuntimeError("OpenRouter stream ended with an error.")
            delta = choice.get("delta") or {}
            role = delta.get("role") or role
            text = delta.get("content")
            if isinstance(text, str) and text:
                content += text
                if self.stream_callback:
                    self.stream_callback({"type": "content", "delta": text})
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if not isinstance(reasoning, str):
                details = delta.get("reasoning_details")
                reasoning = "".join(
                    detail.get("text", "")
                    for detail in details or []
                    if isinstance(detail, dict) and isinstance(detail.get("text"), str)
                )
            if reasoning and self.stream_callback:
                self.stream_callback({"type": "reasoning", "delta": reasoning})
            for call_delta in delta.get("tool_calls") or []:
                index = int(call_delta.get("index", 0))
                call = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call_delta.get("id"):
                    call["id"] = call_delta["id"]
                function = call_delta.get("function") or {}
                name_delta = function.get("name") or ""
                arguments_delta = function.get("arguments") or ""
                call["function"]["name"] += name_delta
                call["function"]["arguments"] += arguments_delta
                if self.stream_callback:
                    self.stream_callback(
                        {
                            "type": "tool_call",
                            "index": index,
                            "id": call["id"],
                            "name_delta": name_delta,
                            "arguments_delta": arguments_delta,
                        }
                    )

        if not saw_done:
            raise RuntimeError("OpenRouter stream ended before the completion marker.")
        if not content and not tool_calls:
            raise RuntimeError("OpenRouter returned an empty completion.")
        message: dict[str, Any] = {"role": role, "content": content or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return {"choices": [{"message": message}], "usage": self.last_usage}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        self.last_usage = None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        payload: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "messages": self.sanitize_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
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
                provider.update({"only": [self.settings.openrouter_provider], "allow_fallbacks": False, "require_parameters": True})
            payload["provider"] = provider
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.settings.openrouter_app_title,
        }
        if self.settings.openrouter_app_url:
            headers["HTTP-Referer"] = self.settings.openrouter_app_url
        image_fallback_used = False
        for attempt in range(3):
            response: requests.Response | None = None
            try:
                response = self._post(payload, headers)
            except requests.RequestException:
                if attempt == 2:
                    raise
            else:
                if response.status_code in {400, 404} and not image_fallback_used:
                    if self._try_image_fallback(payload, response):
                        image_fallback_used = True
                        continue
                    # Image fallback not applicable — include response body for debugging.
                    body_preview = ""
                    try:
                        body_preview = response.text[:500]
                    except Exception:  # noqa: BLE001,S110 — best-effort diagnostic logging.
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
                # Close every response that is not handed to the streamer,
                # including 408/429/5xx retry candidates, to avoid leaks.
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            if attempt < 2:
                delay = self._retry_delay(response, attempt)
                if self.stop_event and self.stop_event.wait(delay):
                    raise RuntimeError("OpenRouter request cancelled.")
                if not self.stop_event:
                    time.sleep(delay)
        raise RuntimeError("OpenRouter retry loop ended unexpectedly.")
