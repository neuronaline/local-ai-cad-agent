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
TOOL_IMAGE_PROMPT = (
    "The attached image is the visual artifact returned by the latest tool "
    "call. Inspect it and continue the task."
)


class RequestCancelled(RuntimeError):
    """Raised when the local agent stops an in-flight LLM request."""


def sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a model response while retaining standard content and tool fields.

    Only the canonical Chat Completions fields are kept so the persisted
    record round-trips cleanly. Anything else the provider returns
    (``audio``, ``function_call``, ``refusal``, ``annotations``, ``logprobs``,
    ``name``, vendor-specific blobs, …) is dropped; otherwise the persisted
    record cannot be compared with the canonical ``{"role", "content"}``
    shape that ``AgentRunner._complete`` writes when it needs to re-append
    the final assistant turn, and a duplicate assistant entry would be
    emitted on the next history load.
    """
    sanitized = deepcopy(message)
    sanitized.pop("reasoning", None)
    sanitized.pop("reasoning_details", None)
    for key in list(sanitized):
        if key.startswith("_"):
            sanitized.pop(key, None)
    # Drop any non-canonical field. Anything not in this whitelist is
    # provider-specific metadata that the canonical assistant record
    # (and the LLM context) does not carry.
    allowed = {"role", "content", "tool_calls"}
    for key in list(sanitized):
        if key not in allowed:
            sanitized.pop(key, None)
    return sanitized


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an API-safe copy while preserving a stable system prefix.

    System messages are kept as separate, leading messages. Combining a static
    prompt with dynamic workspace state changes the cacheable prefix on every
    state change, defeating provider prompt caches.
    """
    system_messages: list[dict[str, Any]] = []
    non_system: list[dict[str, Any]] = []
    for original in messages:
        # Provider adapters add cache hints to message content. Always detach
        # the wire payload from the agent's canonical in-memory transcript so
        # those hints cannot accumulate across tool-loop iterations.
        item = deepcopy(original)
        if item.get("role") == "system" and isinstance(item.get("content"), str):
            system_messages.append(item)
        else:
            non_system.append(item)
    return system_messages + non_system


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip only unsupported assistant metadata, retaining content and tools."""
    sanitized = normalize_messages(messages)
    for index, message in enumerate(sanitized):
        if message.get("role") != "assistant":
            continue
        sanitized[index] = sanitize_assistant_message(message)
    _compact_completed_history(sanitized)
    return relocate_tool_images(sanitized)


def _compact_completed_history(messages: list[dict[str, Any]]) -> None:
    """Shrink old file bodies and edit arguments in the API-only copy.

    The latest tool result remains verbatim because it drives the model's next
    action. Older read bodies and completed edit arguments are recoverable from
    the current ``model.py`` and otherwise dominate long repair-loop prompts.
    """
    tool_result_ids = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
        and isinstance(message.get("tool_call_id"), str)
    }
    completed_assistants = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and any(
            isinstance(call, dict) and call.get("id") in tool_result_ids
            for call in message.get("tool_calls") or []
        )
    ]
    keep_assistant = completed_assistants[-1] if completed_assistants else -1
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and index != len(messages) - 1:
            message["content"] = _compact_tool_content(message.get("content"))
        if message.get("role") != "assistant" or index == keep_assistant:
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("id") not in tool_result_ids:
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if name == "edit_file":
                function["arguments"] = json.dumps(
                    {
                        "filename": "model.py",
                        "old_string": "[omitted from completed call]",
                        "new_string": "[omitted from completed call]",
                    }
                )
            elif name == "write_file":
                function["arguments"] = json.dumps(
                    {
                        "filename": "model.py",
                        "content": "[omitted from completed call]",
                    }
                )
            elif name == "insert_file":
                function["arguments"] = json.dumps(
                    {
                        "filename": "model.py",
                        "anchor": "[omitted from completed call]",
                        "content": "[omitted from completed call]",
                        "position": "before",
                    }
                )


def _compact_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        return _compact_tool_text(content)
    if not isinstance(content, list):
        return content
    compacted = deepcopy(content)
    for part in compacted:
        if (
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ):
            part["text"] = _compact_tool_text(part["text"])
    return compacted


def _compact_tool_text(text: str) -> str:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return text
    tool = payload.get("tool")
    if tool == "cad_build_and_verify":
        from agent.tool_results import compact_for_context

        return compact_for_context(tool, text)
    if tool != "read_file":
        return text
    data = payload.get("data")
    if not isinstance(data, str):
        return text
    try:
        read_data = json.loads(data)
    except json.JSONDecodeError:
        return text
    if not isinstance(read_data, dict) or "content" not in read_data:
        return text
    payload["data"] = json.dumps(
        {
            key: read_data[key]
            for key in (
                "exists",
                "sha256",
                "total_lines",
                "offset",
                "returned_lines",
                "next_offset",
            )
            if key in read_data
        },
        ensure_ascii=False,
    )
    return json.dumps(payload, ensure_ascii=False)


def relocate_tool_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move ``image_url`` parts out of ``tool`` messages into user messages.

    The OpenAI Chat Completions spec (and the providers that implement it,
    including OpenAI itself and Google Vertex's Gemini) only permits
    ``image_url`` content parts inside messages with ``role: user``. Tool-role
    messages carrying inline images are rejected:

    * OpenAI: ``Image URLs are only allowed for messages with role 'user',
      but this message with role 'tool' contains an image URL.``
    * Google Vertex (Gemini): ``Requests ending with a model turn are not
      supported.`` when the trailing tool message contains image parts.

    ``cad_build_and_verify`` attaches its rendered PNG to the tool message so
    the agent can inspect the build in-band. Only a render on the final
    message is current enough to relocate. Older images are removed rather
    than repeatedly re-sent as stale, expensive base64 payloads.
    """
    latest_index = len(messages) - 1
    relocated: list[dict[str, Any]] = []
    latest_images: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if (
            role != "tool"
            or not isinstance(content, list)
            or not any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
        ):
            relocated.append(message)
            continue
        text_parts = [
            part
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "image_url")
        ]
        image_parts = [
            part
            for part in content
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        tool_copy = deepcopy(message)
        tool_copy["content"] = text_parts or "An attached image was unavailable."
        relocated.append(tool_copy)
        if index == latest_index:
            latest_images = [deepcopy(part) for part in image_parts]
    if latest_images:
        relocated.append(
            {
                "role": "user",
                "content": (
                    [
                        {
                            "type": "text",
                            "text": TOOL_IMAGE_PROMPT,
                        }
                    ]
                    + latest_images
                ),
            }
        )
    return relocated


def without_images(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Remove image parts while retaining their accompanying text."""
    stripped: list[dict[str, Any]] = []
    removed = False
    for original in messages:
        message = deepcopy(original)
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue
        text_parts = [
            part for part in content
            if isinstance(part, dict) and part.get("type") != "image_url"
        ]
        if len(text_parts) != len(content):
            removed = True
            if (
                message.get("role") == "user"
                and len(text_parts) == 1
                and text_parts[0].get("type") == "text"
                and text_parts[0].get("text") == TOOL_IMAGE_PROMPT
            ):
                # This user turn exists only to carry a tool render. If the
                # provider rejects images, keep the preceding tool result as
                # the final turn instead of leaving a false instruction to
                # inspect an attachment that no longer exists.
                continue
            message["content"] = text_parts or "An attached image was unavailable."
        stripped.append(message)
    return stripped, removed


def _response_text(response: requests.Response) -> str:
    """Return an error body from real and test response objects."""
    try:
        value = response.text
        if callable(value):
            value = value()
        return value if isinstance(value, str) else ""
    except Exception:  # noqa: BLE001 - Error reporting must not mask the HTTP error.
        return ""


def _is_image_rejection(response: requests.Response) -> bool:
    """Whether a client error specifically rejects visual input."""
    if response.status_code not in {400, 404}:
        return False
    body = _response_text(response).lower()
    return any(
        marker in body
        for marker in (
            "image_url",
            "image input",
            "image inputs",
            "image content",
            "vision",
            "multimodal",
        )
    )


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
        raise RequestCancelled("LLM request cancelled.")
    worker.start()
    while worker.is_alive():
        worker.join(timeout=0.1)
        if stop_event and stop_event.is_set():
            raise RequestCancelled("LLM request cancelled.")
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
    reasoning_text = ""

    for raw_line in response.iter_lines():
        if stop_event and stop_event.is_set():
            response.close()
            raise RequestCancelled(f"{provider_label} request cancelled.")
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
        if reasoning:
            reasoning_text += reasoning
            if stream_callback:
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
    if finish_reason == "length":
        raise RuntimeError(
            f"{provider_label} completion was truncated (finish_reason='length'); "
            "no tool calls were executed."
        )
    # Some reasoning-first models (Anthropic extended thinking, OpenAI o-series,
    # Gemini thinking) emit reasoning deltas with no text content and finish
    # cleanly. Surface the reasoning as the assistant's substantive response so
    # the caller can still complete its turn instead of crashing the loop on
    # an opaque "empty completion" error.
    if not content and not tool_calls:
        if reasoning_text.strip() and finish_reason in (None, "stop"):
            content = reasoning_text.strip()
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"{provider_label} returned an empty completion "
                f"(finish_reason={reason!r})."
            )
    message: dict[str, Any] = {"role": role, "content": content or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {"choices": [{"message": message}], "usage": last_usage}


def sleep_with_cancel(delay: float, stop_event: threading.Event | None) -> None:
    """Sleep for ``delay`` seconds unless ``stop_event`` fires first."""
    if delay <= 0:
        return
    if stop_event and stop_event.wait(delay):
        raise RequestCancelled("LLM request cancelled.")
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
        # Subordinate evaluator hook. Currently the only supported role string is
        # ``"reviewer"`` (set by :func:`agent.cad_review.review_cad`); the base
        # agent loop leaves this ``None``. The field is preserved on the base
        # client so the OpenRouter adapter can tag the ``trace.span_name`` of
        # subordinate requests without changing the cache prefix. OpenAI's
        # adapter does not consume it today.
        self.agent_role: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_image_fallback_used = False
        self.stream_callback = None
        self.require_images = False
        self._provider_label = provider_label
        # Optional activity-log hook. The agent runner wires this when
        # ``agent.log_tool_activity`` is enabled; ``None`` keeps the wire
        # path inert for tests and review sub-sessions that do not log.
        self._activity_logger = None
        self._run_id: str | None = None

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
        self.last_image_fallback_used = False
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(f"{api_key_env(self._provider_label.lower())} is not configured.")
        payload = self._build_payload(messages, tools)
        headers = self._build_headers(api_key)
        # Capture the request body once; ``payload`` is mutated by the
        # image-fallback path so we deep-copy the original shape to keep
        # the log independent of subsequent retry mutations.
        log_payload = self._activity_logger is not None
        request_snapshot = (
            {
                "url": self._endpoint(),
                "model": payload.get("model"),
                "headers": dict(headers),
                "payload": deepcopy(payload),
            }
            if log_payload
            else None
        )

        image_fallback_used = False
        for attempt in range(3):
            response: requests.Response | None = None
            try:
                response = self._post(payload, headers)
            except RequestCancelled:
                if log_payload:
                    self._activity_logger.log(
                        "llm_cancelled",
                        {"attempt": attempt, "model": payload.get("model")},
                        run_id=getattr(self, "_run_id", None),
                    )
                raise
            except Exception:
                if attempt == 2:
                    raise
            else:
                image_rejection = _is_image_rejection(response)
                if image_rejection and not image_fallback_used:
                    # Review calls require images: a refusal that traces back
                    # to ``image_url`` parts is treated as an inconclusive
                    # review and surfaces to the caller instead of retrying
                    # the request without its visual evidence.
                    if self.require_images:
                        response.close()
                        raise RuntimeError(
                            "Review model rejected image inputs; cannot "
                            "complete review without visual evidence."
                        )
                    if self._try_image_fallback(payload, response):
                        image_fallback_used = True
                        self.last_image_fallback_used = True
                        if log_payload:
                            self._activity_logger.log(
                                "llm_visual_fallback",
                                {"attempt": attempt, "model": payload.get("model")},
                                run_id=getattr(self, "_run_id", None),
                            )
                        continue
                    body_preview = _response_text(response)[:500]
                    if body_preview:
                        raise RuntimeError(f"{self._provider_label} {response.status_code}: {body_preview}")
                if response.status_code not in {408, 429} and response.status_code < 500:
                    # Surface the upstream body so the caller sees the real
                    # reason (e.g. OpenRouter's "No endpoints found that can
                    # handle the requested parameters" 404) instead of just
                    # ``requests``' generic ``HTTPError``.
                    body_preview = _response_text(response)[:500]
                    if 400 <= response.status_code < 500 and body_preview:
                        raise RuntimeError(
                            f"{self._provider_label} {response.status_code}: {body_preview}"
                        )
                    response.raise_for_status()
                    if log_payload:
                        self._activity_logger.log(
                            "llm_request",
                            {
                                "attempt": attempt,
                                "url": request_snapshot["url"],
                                "model": request_snapshot["model"],
                                "headers": request_snapshot["headers"],
                                "payload": request_snapshot["payload"],
                                "status": response.status_code,
                            },
                            run_id=getattr(self, "_run_id", None),
                        )
                    if hasattr(response, "iter_lines"):
                        return self._stream_response(response)
                    body = response.json()
                    self.last_usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
                    choices = body.get("choices") if isinstance(body, dict) else None
                    if choices and choices[0].get("finish_reason") == "length":
                        raise RuntimeError(
                            f"{self._provider_label} completion was truncated "
                            "(finish_reason='length'); no tool calls were executed."
                        )
                    return body
                if log_payload:
                    self._activity_logger.log(
                        "llm_retry",
                        {
                            "attempt": attempt,
                            "status": response.status_code,
                            "model": payload.get("model"),
                        },
                        run_id=getattr(self, "_run_id", None),
                    )
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
