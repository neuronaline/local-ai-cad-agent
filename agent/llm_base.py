"""Provider-neutral helpers for OpenAI-compatible Chat Completions APIs.

Both OpenRouter and OpenAI implement the same wire protocol at
``POST /v1/chat/completions``. Anything that is not provider-specific lives
here so the two adapters (``agent/openrouter.py`` and ``agent/openai_client.py``)
stay thin and behave identically.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from copy import deepcopy
from typing import Any

import requests

from agent.settings import Settings


PROVIDER_LABELS = {"openrouter": "OpenRouter", "openai": "OpenAI"}


def sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a model response while retaining standard content and tool fields."""
    sanitized = deepcopy(message)
    sanitized.pop("reasoning", None)
    sanitized.pop("reasoning_details", None)
    for key in list(sanitized):
        if key.startswith("_"):
            sanitized.pop(key, None)
    return sanitized


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an API-safe copy with exactly one leading system message.

    A stable system prefix improves prompt-cache hit rates. Prior messages are
    immutable references — assistant messages get sanitized downstream.
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


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip only unsupported assistant metadata, retaining content and tools."""
    sanitized = normalize_messages(messages)
    for index, message in enumerate(sanitized):
        if message.get("role") != "assistant":
            continue
        sanitized[index] = sanitize_assistant_message(message)
    return sanitized


def without_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
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


def retry_delay(response: requests.Response | None, attempt: int) -> float:
    """Honor ``Retry-After`` when present, otherwise exponential backoff."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return float(2**attempt)


def post_with_cancel(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
    stop_event: threading.Event | None,
) -> requests.Response:
    """Run a blocking ``requests.post`` that can be cancelled by ``stop_event``.

    The HTTP call is dispatched to a daemon thread so the agent-loop stop
    signal can interrupt it within ``~100 ms`` instead of waiting for the
    underlying socket timeout.
    """
    results: queue.Queue[requests.Response | BaseException] = queue.Queue(maxsize=1)

    def request_worker() -> None:
        try:
            results.put(
                requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=timeout_seconds,
                )
            )
        except BaseException as error:  # noqa: BLE001 - Propagate worker failures unchanged.
            results.put(error)

    worker = threading.Thread(target=request_worker, daemon=True)
    if stop_event and stop_event.is_set():
        raise RuntimeError("LLM request cancelled.")
    worker.start()
    while worker.is_alive():
        worker.join(timeout=0.1)
        if stop_event and stop_event.is_set():
            raise RuntimeError("LLM request cancelled.")
    result = results.get()
    if isinstance(result, BaseException):
        raise result
    return result


def parse_chat_stream(
    response: requests.Response,
    *,
    provider_label: str,
    stop_event: threading.Event | None,
    stream_callback: Any | None,
) -> dict[str, Any]:
    """Consume an SSE Chat Completions stream and rebuild a non-streaming response."""
    content = ""
    role = "assistant"
    tool_calls: dict[int, dict[str, Any]] = {}
    saw_done = False
    finish_reason: str | None = None
    last_usage: dict[str, Any] | None = None

    for raw_line in response.iter_lines():
        if stop_event and stop_event.is_set():
            response.close()
            raise RuntimeError(f"{provider_label} request cancelled.")
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
            raise RuntimeError(
                f"{provider_label} returned malformed streaming JSON."
            ) from error
        stream_error = chunk.get("error")
        if stream_error:
            if isinstance(stream_error, dict):
                detail = stream_error.get("message") or json.dumps(stream_error)
            else:
                detail = str(stream_error)
            raise RuntimeError(f"{provider_label} stream failed: {detail}")
        if isinstance(chunk.get("usage"), dict):
            last_usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        if finish_reason == "error":
            raise RuntimeError(f"{provider_label} stream ended with an error.")
        delta = choice.get("delta") or {}
        role = delta.get("role") or role
        text = delta.get("content")
        if isinstance(text, str) and text:
            content += text
            if stream_callback:
                stream_callback({"type": "content", "delta": text})
        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
        if not isinstance(reasoning, str):
            details = delta.get("reasoning_details")
            reasoning = "".join(
                detail.get("text", "")
                for detail in details or []
                if isinstance(detail, dict) and isinstance(detail.get("text"), str)
            )
        if reasoning and stream_callback:
            stream_callback({"type": "reasoning", "delta": reasoning})
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
            if stream_callback:
                stream_callback(
                    {
                        "type": "tool_call",
                        "index": index,
                        "id": call["id"],
                        "name_delta": name_delta,
                        "arguments_delta": arguments_delta,
                    }
                )

    if not saw_done:
        raise RuntimeError(f"{provider_label} stream ended before the completion marker.")
    if not content and not tool_calls:
        raise RuntimeError(f"{provider_label} returned an empty completion.")
    message: dict[str, Any] = {"role": role, "content": content or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {"choices": [{"message": message}], "usage": last_usage}


def sleep_with_cancel(delay: float, stop_event: threading.Event | None) -> None:
    """Sleep for ``delay`` seconds unless ``stop_event`` fires first."""
    if delay <= 0:
        return
    if stop_event and stop_event.wait(delay):
        raise RuntimeError("LLM request cancelled.")
    if not stop_event:
        time.sleep(delay)


def provider_label(provider: str) -> str:
    """Human-readable provider name for user-facing error messages."""
    return PROVIDER_LABELS.get(provider, provider)


def api_key_env(provider: str) -> str:
    """Return the env var name that supplies the API key for ``provider``."""
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def create_llm_client(settings: Settings):
    """Return the chat-completions client selected by ``settings.llm_provider``.

    Importing the adapters lazily keeps this module importable even when one
    of the optional provider SDKs is not installed (none are required today,
    but this leaves room for future native-Responses adapters).
    """
    provider = settings.llm_provider
    if provider == "openai":
        from agent.openai_client import OpenAIClient

        return OpenAIClient(settings)
    if provider == "openrouter":
        from agent.openrouter import OpenRouterClient

        return OpenRouterClient(settings)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Shared Chat Completions client — shared retry loop and state
# ---------------------------------------------------------------------------


class ChatCompletionsClient:
    """Shared base for OpenAI-compatible Chat Completions clients.

    Provider-specific logic (endpoint, headers, payload extras) is pushed into
    the thin subclasses via ``_endpoint``, ``_build_headers``, and
    ``_build_payload``. The retry loop, stream parsing, and image fallback
    are fully shared here.
    """

    def __init__(self, settings: Settings, provider_label: str) -> None:
        self.settings = settings
        self.stop_event = None
        self.session_id: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.stream_callback = None
        self._provider_label = provider_label

    def _endpoint(self) -> str:  # pragma: no cover — overridden by subclass
        raise NotImplementedError

    def _build_headers(self, api_key: str) -> dict[str, str]:  # pragma: no cover
        raise NotImplementedError

    def _build_payload(self, messages, tools):  # pragma: no cover
        raise NotImplementedError

    def _post(self, payload, headers):  # pragma: no cover
        raise NotImplementedError

    def _api_key(self) -> str:  # pragma: no cover
        raise NotImplementedError

    def sanitize_messages(self, messages):
        return sanitize_messages(messages)

    def _stream_response(self, response):
        result = parse_chat_stream(
            response,
            provider_label=self._provider_label,
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
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(f"{api_key_env(self._provider_label.lower())} is not configured.")
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
                        raise RuntimeError(f"{self._provider_label} {response.status_code}: {body_preview}")
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
        raise RuntimeError(f"{self._provider_label} retry loop ended unexpectedly.")