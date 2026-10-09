"""Provider-neutral helpers for OpenAI-compatible Chat Completions APIs.

Both OpenRouter and OpenAI implement the same wire protocol at
``POST /v1/chat/completions``. Anything that is not provider-specific lives
here so the two adapters (``agent/openrouter.py`` and ``agent/openai_client.py``)
stay thin and behave identically.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import time
from copy import deepcopy
from typing import Any

import httpx

from agent.activity_log import summarize_llm_messages
from agent.prompt import TOOL_IMAGE_PROMPT
from agent.settings import Settings

PROVIDER_LABELS = {
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "ollama": "Ollama",
    "gemini": "Google AI Studio",
    "google": "Google AI Studio",
}

# Module-level HTTP/2 client for streaming Chat Completions without HTTP/1.1
# chunk-size reassembly buffering. HTTP/2 binary DATA frames multiplex over
# a persistent connection, eliminating the standard library chunked transfer
# accumulation defect (http.client._read_chunked with amt=None).
_HTTP_CLIENT: httpx.Client = httpx.Client(
    http2=True,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    timeout=None,
)
atexit.register(_HTTP_CLIENT.close)


class RequestCancelled(RuntimeError):
    """Raised when the local agent stops an in-flight LLM request."""


class StreamResponseError(RuntimeError):
    """An error event received after a successful streaming HTTP response."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def _stream_error_detail(error: Any) -> str:
    """Return a useful, bounded provider error from an SSE error payload."""
    if isinstance(error, dict):
        detail = error.get("message") or error.get("detail") or json.dumps(error)
    else:
        detail = str(error)
    return detail[:500]


def extract_text_tool_calls(
    content: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract embedded tool call markup (e.g. <function=...>, <tool_call>...) from content.

    Returns (cleaned_content, extracted_tool_calls).
    Handles XML-style function tags (<function=name><parameter=key>value</parameter></function>
    or unclosed variants), JSON-style <tool_call> blocks, and [TOOL_CALLS] blocks.
    Strips raw tool-calling tags so internal function syntax never leaks into the user-facing chat.
    """
    if not content:
        return "", []

    extracted_calls: list[dict[str, Any]] = []
    cleaned = content

    # 1. XML style: <function=name> ... </function> (or unclosed / parameter-delimited)
    if "<function=" in cleaned:
        func_matches = list(
            re.finditer(
                r"<function=([a-zA-Z0-9_-]+)>(.*?)(?:</function>|(?=<function=)|\Z)",
                cleaned,
                re.DOTALL,
            )
        )
        for idx, fm in enumerate(func_matches):
            tool_name = fm.group(1).strip()
            body = fm.group(2)
            args: dict[str, Any] = {}
            param_matches = list(
                re.finditer(
                    r"<parameter=([a-zA-Z0-9_-]+)>(.*?)(?:</parameter>|<parameter=\1>|(?=<parameter=)|\Z)",
                    body,
                    re.DOTALL,
                )
            )
            for pm in param_matches:
                k = pm.group(1).strip()
                v = pm.group(2).strip()
                try:
                    if (
                        (v.startswith("{") and v.endswith("}"))
                        or (v.startswith("[") and v.endswith("]"))
                        or v in ("true", "false", "null")
                    ):
                        args[k] = json.loads(v)
                    else:
                        args[k] = v
                except Exception:
                    args[k] = v
            extracted_calls.append(
                {
                    "id": f"call_text_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict)
                        else "{}",
                    },
                }
            )
        cleaned = re.sub(
            r"<function=([a-zA-Z0-9_-]+)>.*?(?:</function>|(?=<function=)|\Z)",
            "",
            cleaned,
            flags=re.DOTALL,
        ).strip()

    # 2. Tool call tags: <tool_call> ... </tool_call>
    if "<tool_call>" in cleaned or "<tool_call " in cleaned:
        tc_matches = list(
            re.finditer(
                r"<tool_call[^>]*>(.*?)(?:</tool_call>|\Z)", cleaned, re.DOTALL
            )
        )
        for tcm in tc_matches:
            raw = tcm.group(1).strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and "name" in data:
                    call_args = data.get("arguments", {})
                    extracted_calls.append(
                        {
                            "id": f"call_tc_{len(extracted_calls)}",
                            "type": "function",
                            "function": {
                                "name": data["name"],
                                "arguments": json.dumps(
                                    call_args, ensure_ascii=False
                                )
                                if isinstance(call_args, dict)
                                else str(call_args),
                            },
                        }
                    )
            except Exception:
                pass
        cleaned = re.sub(
            r"<tool_call[^>]*>.*?(?:</tool_call>|\Z)", "", cleaned, flags=re.DOTALL
        ).strip()

    # 3. Mistral / Command R style: [TOOL_CALLS] ... [/TOOL_CALLS]
    if "[TOOL_CALLS]" in cleaned:
        tc_matches = list(
            re.finditer(
                r"\[TOOL_CALLS\](.*?)(?:\[/TOOL_CALLS\]|\Z)", cleaned, re.DOTALL
            )
        )
        for tcm in tc_matches:
            raw = tcm.group(1).strip()
            try:
                data = json.loads(raw)
                calls_list = data if isinstance(data, list) else [data]
                for item in calls_list:
                    if isinstance(item, dict) and "name" in item:
                        call_args = item.get("arguments", {})
                        extracted_calls.append(
                            {
                                "id": f"call_tc_{len(extracted_calls)}",
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(
                                        call_args, ensure_ascii=False
                                    )
                                    if isinstance(call_args, dict)
                                    else str(call_args),
                                },
                            }
                        )
            except Exception:
                pass
        cleaned = re.sub(
            r"\[TOOL_CALLS\].*?(?:\[/TOOL_CALLS\]|\Z)", "", cleaned, flags=re.DOTALL
        ).strip()

    # Clean up any leftover stray closing tags
    cleaned = re.sub(r"</?(?:function|parameter|tool_call)[^>]*>", "", cleaned).strip()

    return cleaned, extracted_calls


def sanitize_assistant_message(
    message: dict[str, Any],
    *,
    preserve_reasoning: bool = False,
    for_storage: bool = False,
) -> dict[str, Any]:
    """Copy a model response while retaining supported continuation fields.

    When ``preserve_reasoning`` or ``for_storage`` is True, reasoning and
    reasoning_details are retained. Storage records preserve thoughts for
    historical sessions and UI display, while outgoing wire requests can
    safely omit previous reasoning to prevent provider schema rejection.
    """
    sanitized = deepcopy(message)
    keep_reasoning = preserve_reasoning or for_storage
    # Normalize reasoning_content alias to reasoning if reasoning is not already present
    reasoning_content = sanitized.pop("reasoning_content", None)
    if (
        keep_reasoning
        and isinstance(reasoning_content, str)
        and reasoning_content.strip()
        and not sanitized.get("reasoning")
    ):
        sanitized["reasoning"] = reasoning_content

    if not keep_reasoning:
        sanitized.pop("reasoning", None)
        sanitized.pop("reasoning_details", None)
    else:
        # Strip empty reasoning fields so they do not trigger 400s on strict providers
        reasoning = sanitized.get("reasoning")
        if isinstance(reasoning, str) and not reasoning.strip():
            sanitized.pop("reasoning", None)
        details = sanitized.get("reasoning_details")
        if isinstance(details, list) and not details:
            sanitized.pop("reasoning_details", None)
        # OpenRouter specification: echo back either reasoning_details (preferred)
        # or reasoning (plaintext fallback), never both in the same assistant turn.
        if not for_storage and sanitized.get("reasoning_details"):
            sanitized.pop("reasoning", None)
            sanitized.pop("reasoning_content", None)

    # Strip any leaked tool-call tags from content in assistant messages
    if isinstance(sanitized.get("content"), str):
        cleaned_content, extra_calls = extract_text_tool_calls(sanitized["content"])
        sanitized["content"] = cleaned_content or None
        if not sanitized.get("tool_calls") and extra_calls:
            sanitized["tool_calls"] = extra_calls

    for key in list(sanitized):
        if key.startswith("_"):
            sanitized.pop(key, None)
    # Drop any non-canonical field. Anything not in this whitelist is
    # provider-specific metadata that the canonical assistant record
    # (and the LLM context) does not carry.
    allowed = {"role", "content", "tool_calls"}
    if keep_reasoning:
        allowed.update({"reasoning", "reasoning_details"})
    for key in list(sanitized):
        if key not in allowed:
            sanitized.pop(key, None)
    return sanitized


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an API-safe copy while preserving a stable system prefix.

    System messages are kept as separate, leading messages. Combining a static
    prompt with dynamic workspace state changes the cacheable prefix on every
    state change, defeating provider prompt caches.
    Adjacent non-system messages with the same role (e.g. consecutive 'user'
    messages) are merged to satisfy strict provider role-alternation contracts.
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

    merged_non_system: list[dict[str, Any]] = []
    for msg in non_system:
        if (
            merged_non_system
            and merged_non_system[-1].get("role") == "user"
            and msg.get("role") == "user"
        ):
            prev = merged_non_system[-1]
            prev_content = prev.get("content")
            curr_content = msg.get("content")
            if isinstance(prev_content, str) and isinstance(curr_content, str):
                prev["content"] = f"{prev_content}\n\n{curr_content}"
            elif isinstance(prev_content, list) and isinstance(curr_content, str):
                prev["content"] = [*prev_content, {"type": "text", "text": curr_content}]
            elif isinstance(prev_content, str) and isinstance(curr_content, list):
                prev["content"] = [{"type": "text", "text": prev_content}, *curr_content]
            elif isinstance(prev_content, list) and isinstance(curr_content, list):
                prev["content"] = [*prev_content, *curr_content]
        else:
            merged_non_system.append(msg)

    return system_messages + merged_non_system


def sanitize_messages(
    messages: list[dict[str, Any]], *, preserve_reasoning: bool = False
) -> list[dict[str, Any]]:
    """Strip only unsupported assistant metadata, retaining content and tools.

    When ``preserve_reasoning`` is set, ``reasoning`` / ``reasoning_details`` are
    retained across all assistant turns so the prompt prefix remains strictly
    append-only. Mutating historical turns by dropping reasoning from earlier
    assistant messages breaks provider prompt caches (causing cache misses)
    and causes input token counts to fluctuate erratically across iterations.
    """
    sanitized = normalize_messages(messages)
    for index, message in enumerate(sanitized):
        if message.get("role") != "assistant":
            continue
        sanitized[index] = sanitize_assistant_message(
            message,
            preserve_reasoning=preserve_reasoning,
        )
    return relocate_tool_images(sanitized)


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

    Tool-role messages produced by tools like ``get_view_images`` may attach
    rendered PNGs to the message so the agent can inspect geometry in-band. Keep each
    completed tool batch's visual evidence in the subsequent conversation: dropping it
    after one turn changes the prompt prefix and causes avoidable cache misses (and
    misleading input-token drops). Images are emitted only after the complete
    tool-result batch so multi-tool assistant responses retain the required
    contiguous tool-result protocol.
    """
    relocated: list[dict[str, Any]] = []
    batch_images: list[dict[str, Any]] = []

    def flush_batch_images() -> None:
        if not batch_images:
            return
        relocated.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TOOL_IMAGE_PROMPT},
                    *batch_images,
                ],
            }
        )
        batch_images.clear()

    for message in messages:
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
            if role != "tool":
                flush_batch_images()
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
        if len(text_parts) == 1 and isinstance(text_parts[0], dict):
            text = text_parts[0].get("text")
            tool_copy["content"] = (
                text if isinstance(text, str) else text_parts
            )
        else:
            tool_copy["content"] = text_parts or "An attached image was unavailable."
        relocated.append(tool_copy)
        batch_images.extend(deepcopy(part) for part in image_parts)
    flush_batch_images()
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


def _response_text(response: Any) -> str:
    """Return an error body from real and test response objects."""
    try:
        if hasattr(response, "read") and callable(response.read):
            try:
                response.read()
            except Exception:
                pass
        value = response.text
        if callable(value):
            value = value()
        return value if isinstance(value, str) else ""
    except Exception:  # Error reporting must not mask the HTTP error.
        return ""


def _is_image_rejection(response: Any) -> bool:
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
            "does not support image",
            "does not support images",
            "cannot process image",
        )
    )


def retry_delay(response: Any | None, attempt: int) -> float:
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
) -> Any:
    """Run a blocking ``POST`` that can be cancelled by ``stop_event``.

    Dispatches the HTTP call to a daemon worker thread and streams via
    the module-level :data:`_HTTP_CLIENT` (using HTTP/2 multiplexing) to
    eliminate HTTP/1.1 chunk-size reassembly buffering and token coalescing.
    """
    results: queue.Queue[Any | BaseException] = queue.Queue(maxsize=1)
    response_holder: dict[str, Any] = {}

    def request_worker() -> None:
        try:
            req_headers = dict(headers)
            req_headers.setdefault("Accept", "text/event-stream")
            req = _HTTP_CLIENT.build_request(
                "POST",
                url,
                headers=req_headers,
                json=payload,
                timeout=httpx.Timeout(timeout_seconds, connect=15.0),
            )
            response = _HTTP_CLIENT.send(req, stream=True)
        except BaseException as error:  # Propagate worker failures unchanged.
            results.put(error)
            return
        response_holder["response"] = response
        results.put(response)

    worker = threading.Thread(target=request_worker, daemon=True)
    if stop_event and stop_event.is_set():
        raise RequestCancelled("LLM request cancelled.")
    worker.start()
    try:
        while worker.is_alive():
            worker.join(timeout=0.1)
            if stop_event and stop_event.is_set():
                raise RequestCancelled("LLM request cancelled.")
        result = results.get(timeout=5)
        if isinstance(result, BaseException):
            raise result
        return result
    except RequestCancelled:
        worker.join(timeout=0.5)
        _force_close_response(response_holder.get("response"))
        try:
            queued = results.get_nowait()
        except queue.Empty:
            queued = None
        if queued is not None and not isinstance(queued, BaseException):
            _force_close_response(queued)
        raise


def _force_close_response(response: Any | None) -> None:
    """Force-close a streaming ``Response``, releasing the underlying stream/socket."""
    if response is None:
        return
    try:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def strip_encrypted_reasoning(text: str | None) -> str:
    """Strip leaked base64 encrypted reasoning / signature blobs from human-readable text."""
    if not text:
        return ""
    s = text.strip()
    if len(s) >= 50 and bool(re.fullmatch(r"[A-Za-z0-9+/=]+", s)):
        return ""
    # Strip any trailing base64 signature/encrypted ciphertext block (>= 40 chars, possibly newline/space separated)
    return re.sub(r"(?:\r?\n|\s)+(?:[A-Za-z0-9+/=]{40,}(?:\r?\n|\s)*)+$", "", text)


def extract_think_tags(
    content: str | None,
    existing_reasoning: str | None = None,
) -> tuple[str, str | None]:
    """Extract <think>...</think> blocks from content.

    Handles complete think blocks, multiple blocks, and truncated/unclosed think
    tags without leaving raw tags in the substantive message content.
    """
    if existing_reasoning:
        existing_reasoning = strip_encrypted_reasoning(existing_reasoning) or None
    if not content:
        return "", existing_reasoning

    if "<think>" not in content and "</think>" not in content:
        return content, existing_reasoning or None

    cleaned_content = content
    extracted_parts: list[str] = []

    # 1. Extract any completed <think>...</think> blocks
    think_blocks = re.findall(r"<think>(.*?)</think>", cleaned_content, flags=re.DOTALL)
    if think_blocks:
        extracted_parts.extend(b.strip() for b in think_blocks if b.strip())
        cleaned_content = re.sub(r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL).strip()

    # 2. Extract any remaining unclosed <think> tag
    if "<think>" in cleaned_content:
        start_idx = cleaned_content.find("<think>")
        unclosed = cleaned_content[start_idx + 7 :].strip()
        if unclosed:
            extracted_parts.append(unclosed)
        cleaned_content = cleaned_content[:start_idx].strip()

    # 3. Strip any stray closing tags
    if "</think>" in cleaned_content:
        cleaned_content = cleaned_content.replace("</think>", "").strip()

    extracted = "\n\n".join(extracted_parts) if extracted_parts else None
    if existing_reasoning and extracted:
        combined = f"{existing_reasoning}\n\n{extracted}".strip()
        return cleaned_content, combined or None
    return cleaned_content, (extracted or existing_reasoning or None)


def parse_chat_stream(
    response: Any,
    *,
    provider_label: str,
    stop_event: threading.Event | None = None,
    stream_callback: Any | None = None,
) -> dict[str, Any]:
    """Consume an SSE Chat Completions stream and rebuild a non-streaming response."""
    content = ""
    role = "assistant"
    tool_calls: dict[int, dict[str, Any]] = {}
    saw_done = False
    finish_reason: str | None = None
    last_usage: dict[str, Any] | None = None
    reasoning_text = ""
    reasoning_details: list[dict[str, Any]] = []
    reasoning_details_map: dict[Any, dict[str, Any]] = {}
    active_detail_key: Any = None
    in_think_tag = False
    in_tool_tag = False
    tag_buffer = ""
    tool_tag_buffer = ""

    try:
        for raw_line in response.iter_lines():
            if stop_event and stop_event.is_set():
                _force_close_response(response)
                raise RequestCancelled(f"{provider_label} request cancelled.")
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line:
                continue
            line = line.strip()
            if not line.startswith("data:"):
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
                raise StreamResponseError(
                    f"{provider_label} stream failed: {_stream_error_detail(stream_error)}",
                    retryable=not (content or tool_calls or reasoning_text or reasoning_details),
                )
            if isinstance(chunk.get("usage"), dict):
                last_usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            if finish_reason == "error":
                delta = choice.get("delta") or {}
                stream_error = choice.get("error") or delta.get("error")
                detail = (
                    _stream_error_detail(stream_error)
                    if stream_error
                    else "upstream completion returned finish_reason='error'"
                )
                raise StreamResponseError(
                    f"{provider_label} stream failed: {detail}",
                    retryable=not (content or tool_calls or reasoning_text or reasoning_details),
                )
            delta = choice.get("delta") or {}
            role = delta.get("role") or role
            text = delta.get("content")
            if isinstance(text, str) and text:
                if tag_buffer:
                    text = tag_buffer + text
                    tag_buffer = ""
                while text:
                    if not in_think_tag and not in_tool_tag:
                        think_pos = text.find("<think>")
                        tool_tags = (
                            "<function=",
                            "<tool_call>",
                            "<tool_call ",
                            "[TOOL_CALLS]",
                            "<function_call>",
                        )
                        found_tool_pos = -1
                        for tt in tool_tags:
                            p = text.find(tt)
                            if p != -1 and (found_tool_pos == -1 or p < found_tool_pos):
                                found_tool_pos = p

                        if think_pos != -1 and (found_tool_pos == -1 or think_pos < found_tool_pos):
                            before, _, text = text.partition("<think>")
                            if before:
                                content += before
                                if stream_callback:
                                    stream_callback({"type": "content", "delta": before})
                            in_think_tag = True
                            continue
                        elif found_tool_pos != -1:
                            before = text[:found_tool_pos]
                            text = text[found_tool_pos:]
                            if before:
                                content += before
                                if stream_callback:
                                    stream_callback({"type": "content", "delta": before})
                            in_tool_tag = True
                            tool_tag_buffer += text
                            text = ""
                            break

                        matched_prefix = False
                        for candidate in (
                            "<think>",
                            "<function=",
                            "<tool_call>",
                            "[TOOL_CALLS]",
                        ):
                            for k in range(min(len(text), len(candidate) - 1), 0, -1):
                                if text.endswith(candidate[:k]):
                                    tag_buffer = text[-k:]
                                    emit_text = text[:-k]
                                    if emit_text:
                                        content += emit_text
                                        if stream_callback:
                                            stream_callback({"type": "content", "delta": emit_text})
                                    matched_prefix = True
                                    break
                            if matched_prefix:
                                break
                        if not matched_prefix:
                            content += text
                            if stream_callback:
                                stream_callback({"type": "content", "delta": text})
                        break
                    elif in_think_tag:
                        if "</think>" in text:
                            think_text, _, text = text.partition("</think>")
                            if think_text:
                                reasoning_text += think_text
                                if stream_callback:
                                    stream_callback({"type": "reasoning", "delta": think_text})
                            in_think_tag = False
                            continue
                        matched_prefix = False
                        for k in range(min(len(text), 7), 0, -1):
                            if text.endswith("</think>"[:k]):
                                tag_buffer = text[-k:]
                                emit_text = text[:-k]
                                if emit_text:
                                    reasoning_text += emit_text
                                    if stream_callback:
                                        stream_callback({"type": "reasoning", "delta": emit_text})
                                matched_prefix = True
                                break
                        if not matched_prefix:
                            reasoning_text += text
                            if stream_callback:
                                stream_callback({"type": "reasoning", "delta": text})
                        break
                    else:  # in_tool_tag
                        tool_tag_buffer += text
                        closing_tags = (
                            "</function>",
                            "</tool_call>",
                            "[/TOOL_CALLS]",
                            "</function_call>",
                        )
                        closed = False
                        for ct in closing_tags:
                            if ct in text:
                                _, _, rem = text.partition(ct)
                                in_tool_tag = False
                                text = rem
                                closed = True
                                break
                        if closed:
                            continue
                        break
            reasoning = (
                delta.get("reasoning")
                or delta.get("reasoning_content")
                or delta.get("thinking")
            )
            details = delta.get("reasoning_details")
            details_text = ""
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, dict):
                        # Only extract human-readable thought text. Never treat
                        # encrypted ciphertext ('data', type='reasoning.encrypted')
                        # or verification signatures ('signature') as display text.
                        if detail.get("type") != "reasoning.encrypted":
                            t = detail.get("text") or detail.get("summary")
                            if isinstance(t, str) and t:
                                details_text += t
                        idx = detail.get("index")
                        detail_id = detail.get("id")
                        if idx is not None:
                            key = idx
                        elif detail_id is not None:
                            key = detail_id
                        elif reasoning_details and not (detail.get("type") and detail.get("type") != reasoning_details[-1].get("type")):
                            key = active_detail_key
                        else:
                            key = len(reasoning_details)
                            active_detail_key = key
                        if key in reasoning_details_map:
                            target = reasoning_details_map[key]
                            for field in ("text", "summary", "data"):
                                if field in detail and isinstance(detail[field], str):
                                    target[field] = target.get(field, "") + detail[field]
                            for field in ("signature", "id", "type", "format"):
                                if field in detail and detail[field] is not None:
                                    target[field] = detail[field]
                        else:
                            target = deepcopy(detail)
                            reasoning_details_map[key] = target
                            reasoning_details.append(target)
            if isinstance(reasoning, str) and reasoning:
                reasoning_delta_text = reasoning
            elif details_text:
                reasoning_delta_text = details_text
            else:
                reasoning_delta_text = None
            if reasoning_delta_text:
                clean_delta = strip_encrypted_reasoning(reasoning_delta_text)
                if clean_delta:
                    reasoning_text += clean_delta
                    if stream_callback:
                        stream_callback({"type": "reasoning", "delta": clean_delta})
            for call_delta in delta.get("tool_calls") or []:
                index = int(call_delta.get("index", 0))
                call = tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call_delta.get("id"):
                    call["id"] = call_delta["id"]
                elif not call["id"]:
                    call["id"] = f"call_{index}"
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
                            "name": call["function"]["name"],
                            "name_delta": name_delta,
                            "arguments_delta": arguments_delta,
                        }
                    )
        if tag_buffer:
            if in_think_tag:
                reasoning_text += tag_buffer
                if stream_callback:
                    stream_callback({"type": "reasoning", "delta": tag_buffer})
            elif in_tool_tag:
                tool_tag_buffer += tag_buffer
            else:
                if len(tag_buffer) >= 2 and any(
                    tag_buffer.startswith(tt[: len(tag_buffer)])
                    for tt in ("<function=", "<tool_call>", "[TOOL_CALLS]")
                ):
                    tool_tag_buffer += tag_buffer
                else:
                    content += tag_buffer
                    if stream_callback:
                        stream_callback({"type": "content", "delta": tag_buffer})
            tag_buffer = ""
    except Exception:
        if stop_event and stop_event.is_set():
            _force_close_response(response)
            raise RequestCancelled(f"{provider_label} request cancelled.")
        raise

    if not saw_done:
        # The connection closed before the provider emitted ``[DONE]``.
        # Treat the error as retryable when no partial content was streamed
        # so a transient network drop does not fail the agent run mid-tool
        # loop. The user did not stop, ``stop_event`` is still clear, so
        # ``RequestCancelled`` (raised on the stop path above) cannot have
        # fired. When partial state was already streamed (text, reasoning,
        # or partial tool_calls) retrying would duplicate tool execution,
        # so we surface the truncation as a non-retryable error instead.
        had_partial = bool(content or tool_calls or reasoning_text or reasoning_details or tool_tag_buffer)
        message = (
            f"{provider_label} stream ended before the completion marker"
            + ("; partial response preserved." if had_partial else ".")
        )
        raise StreamResponseError(message, retryable=not had_partial)
    if finish_reason == "length":
        raise RuntimeError(
            f"{provider_label} completion was truncated (finish_reason='length'); "
            "no tool calls were executed. Consider increasing 'llm.max_completion_tokens' "
            "in config.yaml or reducing 'reasoning_effort'."
        )
    # Fallback: extract <think>...</think> from content when reasoning was not in separate delta
    content, extracted_reasoning = extract_think_tags(content, reasoning_text)
    if extracted_reasoning:
        reasoning_text = extracted_reasoning

    # Process any tool call tags leaked into content or buffered in tool_tag_buffer
    cleaned_content, text_calls_from_content = extract_text_tool_calls(content)
    cleaned_buffer, text_calls_from_buffer = extract_text_tool_calls(tool_tag_buffer)
    all_extracted_calls = text_calls_from_buffer + text_calls_from_content
    if cleaned_buffer and not text_calls_from_buffer:
        cleaned_content = (cleaned_content + " " + cleaned_buffer).strip() if cleaned_content else cleaned_buffer
    content = cleaned_content

    # If the provider did not supply tool_calls via API, recover them from text
    if not tool_calls and all_extracted_calls:
        for idx, call in enumerate(all_extracted_calls):
            tool_calls[idx] = call

    # Some reasoning-first models (Anthropic extended thinking, OpenAI o-series,
    # Gemini thinking) emit reasoning deltas with no text content and finish
    # cleanly. Set a safe placeholder message instead of leaking raw internal
    # monologue into the user chat bubble.
    if not content and not tool_calls:
        if reasoning_text.strip() and finish_reason in (None, "stop"):
            content = (
                "Model completed its reasoning process but did not produce an action or final response."
            )
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"{provider_label} returned an empty completion "
                f"(finish_reason={reason!r})."
            )
    message: dict[str, Any] = {"role": role, "content": content or None}
    if tool_calls:
        for idx, call in tool_calls.items():
            if not call.get("id"):
                call["id"] = f"call_{idx}"
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    clean_reasoning = strip_encrypted_reasoning(reasoning_text)
    if clean_reasoning:
        message["reasoning"] = clean_reasoning
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
    key = provider.strip().lower()
    if key in ("openai",):
        return "OPENAI_API_KEY"
    if key in ("openrouter",):
        return "OPENROUTER_API_KEY"
    if key in ("ollama",):
        return "OLLAMA_API_KEY"
    if key in ("gemini", "google", "google ai studio"):
        if os.getenv("GEMINI_API_KEY", "").strip():
            return "GEMINI_API_KEY"
        if os.getenv("GOOGLE_API_KEY", "").strip():
            return "GOOGLE_API_KEY"
        return "GEMINI_API_KEY"
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class BaseLLMClient:
    """Shared interface and execution engine for all LLM provider adapters."""

    preserve_reasoning: bool = False
    requires_api_key: bool = True
    _DEFAULT_RETRY_ATTEMPTS: int = 3

    def __init__(self, settings: Settings, provider_label: str) -> None:
        self.settings = settings
        self.stop_event: threading.Event | None = None
        self.session_id: str | None = None
        self.agent_role: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_image_fallback_used: bool = False
        self.stream_callback: Any | None = None
        self.require_images: bool = False
        self._provider_label: str = provider_label
        self.activity_logger: Any | None = None
        self.run_id: str | None = None
        self._active_response: Any | None = None
        self._response_lock = threading.Lock()

    def sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sanitize_messages(messages, preserve_reasoning=self.preserve_reasoning)

    def abort(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        with self._response_lock:
            resp = self._active_response
        if resp is not None:
            _force_close_response(resp)

    def _api_key(self) -> str:
        raise NotImplementedError

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _build_headers(self, api_key: str) -> dict[str, str]:
        raise NotImplementedError

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        raise NotImplementedError

    def _stream_response(self, response: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _parse_non_stream_response(self, response: Any) -> dict[str, Any]:
        raise NotImplementedError

    def _try_image_fallback(self, payload: dict[str, Any], response: Any) -> bool:
        raise NotImplementedError

    def _summarize_payload_for_logging(
        self, payload: dict[str, Any], is_dataset_mode: bool
    ) -> dict[str, Any] | None:
        return None

    def _log_llm_response(
        self,
        attempt: int,
        model: Any,
        call_start: float,
        body: dict[str, Any] | None,
    ) -> None:
        if self.activity_logger is None or not isinstance(body, dict):
            return
        total_duration_ms = round((time.perf_counter() - call_start) * 1000, 2)
        choices = body.get("choices") if isinstance(body.get("choices"), list) else None
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        choice_msg = (
            first_choice.get("message", {})
            if isinstance(first_choice.get("message"), dict)
            else {}
        )
        self.activity_logger.log(
            "llm_response",
            {
                "attempt": attempt,
                "model": model,
                "duration_ms": total_duration_ms,
                "usage": self.last_usage,
                "has_reasoning": bool(
                    choice_msg.get("reasoning")
                    or choice_msg.get("reasoning_details")
                ),
                "tool_call_count": len(choice_msg.get("tool_calls") or []),
            },
            run_id=self.run_id,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        attempts = (
            max_attempts
            if max_attempts is not None
            else self._DEFAULT_RETRY_ATTEMPTS
        )
        self.last_usage = None
        self.last_image_fallback_used = False
        api_key = self._api_key()
        if self.requires_api_key and not api_key:
            raise RuntimeError(f"{api_key_env(self._provider_label)} is not configured.")
        payload = self._build_payload(messages, tools)
        headers = self._build_headers(api_key)
        log_payload = self.activity_logger is not None

        image_fallback_used = False
        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1
            response: Any | None = None
            attempt_payload = None
            if log_payload:
                is_dataset_mode = getattr(self.activity_logger, "log_mode", "") == "dataset"
                attempt_payload = self._summarize_payload_for_logging(payload, is_dataset_mode)
            call_start = time.perf_counter()
            try:
                response = self._post(payload, headers)
            except RequestCancelled:
                if log_payload:
                    self.activity_logger.log(
                        "llm_cancelled",
                        {"attempt": attempt, "model": payload.get("model")},
                        run_id=self.run_id,
                    )
                raise
            except Exception:
                if is_last_attempt:
                    raise
            else:
                image_rejection = _is_image_rejection(response)
                if image_rejection and not image_fallback_used:
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
                            self.activity_logger.log(
                                "llm_visual_fallback",
                                {"attempt": attempt, "model": payload.get("model")},
                                run_id=self.run_id,
                            )
                        continue
                    body_preview = _response_text(response)[:500]
                    if body_preview:
                        raise RuntimeError(f"{self._provider_label} {response.status_code}: {body_preview}")
                if response.status_code not in {408, 429} and response.status_code < 500:
                    if 400 <= response.status_code < 500:
                        body_preview = _response_text(response)[:500]
                        if body_preview:
                            raise RuntimeError(
                                f"{self._provider_label} {response.status_code}: {body_preview}"
                            )
                    response.raise_for_status()
                    duration_ms = round((time.perf_counter() - call_start) * 1000, 2)
                    if log_payload:
                        self.activity_logger.log(
                            "llm_request",
                            {
                                "attempt": attempt,
                                "url": self._endpoint(),
                                "model": payload.get("model"),
                                "headers": dict(headers),
                                "payload": attempt_payload,
                                "status": response.status_code,
                                "duration_ms": duration_ms,
                            },
                            run_id=self.run_id,
                        )
                    if hasattr(response, "iter_lines"):
                        try:
                            stream_res = self._stream_response(response)
                            if log_payload:
                                self._log_llm_response(
                                    attempt, payload.get("model"), call_start, stream_res
                                )
                            return stream_res
                        except StreamResponseError as error:
                            if not error.retryable or is_last_attempt:
                                raise
                            if log_payload:
                                self.activity_logger.log(
                                    "llm_stream_retry",
                                    {
                                        "attempt": attempt,
                                        "model": payload.get("model"),
                                        "error": str(error),
                                    },
                                    run_id=self.run_id,
                                )
                    else:
                        parsed_body = self._parse_non_stream_response(response)
                        if log_payload:
                            self._log_llm_response(
                                attempt, payload.get("model"), call_start, parsed_body
                            )
                        return parsed_body
                if log_payload:
                    self.activity_logger.log(
                        "llm_retry",
                        {
                            "attempt": attempt,
                            "status": response.status_code,
                            "model": payload.get("model"),
                        },
                        run_id=self.run_id,
                    )
                if is_last_attempt:
                    response.raise_for_status()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            if not is_last_attempt:
                delay = retry_delay(response, attempt)
                sleep_with_cancel(delay, self.stop_event)
        raise RuntimeError(f"{self._provider_label} retry loop ended unexpectedly.")


def create_llm_client(settings: Settings) -> BaseLLMClient:
    """Return the chat client selected by ``settings.llm_provider``.

    When ``settings.llm_fallback_provider`` is set, the primary client is
    wrapped in :class:`FallbackChatClient` so a transient primary failure
    triggers a single retry against the fallback before the agent surfaces
    an error. User cancellations never fall back.

    Importing the adapters lazily keeps this module importable even when one
    of the optional provider SDKs is not installed (none are required today,
    but this leaves room for future native-Responses adapters).
    """
    primary = _build_provider_client(settings.llm_provider, settings)
    fallback_label = settings.llm_fallback_provider
    if not fallback_label:
        return primary
    fallback = _build_provider_client(fallback_label, settings)
    return FallbackChatClient(primary, fallback, fallback_label)


def _build_provider_client(provider: str, settings: Settings) -> BaseLLMClient:
    """Construct the adapter for a single provider label.

    Extracted from :func:`create_llm_client` so the fallback wrapper can
    reuse the exact same factory without re-implementing lazy imports.
    """
    if provider == "openai":
        from agent.openai_client import OpenAIClient

        return OpenAIClient(settings)
    if provider == "openrouter":
        from agent.openrouter import OpenRouterClient

        return OpenRouterClient(settings)
    if provider == "ollama":
        from agent.ollama_client import OllamaClient

        return OllamaClient(settings)
    if provider in ("gemini", "google"):
        from agent.gemini_client import GoogleInteractionsClient

        return GoogleInteractionsClient(settings)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class FallbackChatClient(BaseLLMClient):
    """Try a primary ``BaseLLMClient`` and fall back once on failure.

    The wrapper mirrors the public surface of :class:`BaseLLMClient`
    so :class:`agent.core.AgentRunner` can treat it as a drop-in replacement.
    Any exception from the primary (after its own retry loop is exhausted)
    triggers a single attempt against the fallback — except user
    cancellations, which always propagate so the stop signal is observed
    cleanly. The fallback path is intentionally conservative: one shot, no
    nested retries, no automatic re-fallback on a fallback failure.
    """

    # Per-call attributes the agent runner writes on the wrapper; we forward
    # them to both inner clients at the top of every ``chat()`` call.
    _MIRRORED_ATTRS = (
        "stop_event",
        "session_id",
        "agent_role",
        "stream_callback",
        "require_images",
        "activity_logger",
        "run_id",
    )
    # Per-call attributes the runner reads back after a successful chat; we
    # copy them from whichever inner client answered so callers see the
    # provider that actually responded.
    _CAPTURED_ATTRS = (
        "last_usage",
        "last_image_fallback_used",
        "preserve_reasoning",
    )

    def __init__(
        self,
        primary: BaseLLMClient,
        fallback: BaseLLMClient,
        fallback_label: str,
    ) -> None:
        super().__init__(primary.settings, provider_label=primary._provider_label)
        self._primary = primary
        self._fallback = fallback
        self._fallback_label = fallback_label
        # Mirror the inner clients' defaults so a freshly constructed
        # wrapper is a safe no-op pass-through (no AttributeError for
        # callers that read it before any ``chat()`` call).
        for name in self._MIRRORED_ATTRS:
            setattr(self, name, getattr(primary, name, None))
        # Seed captured-attribute defaults so reads return the same
        # values the legacy class-level ``preserve_reasoning = False``
        # annotation did. ``_capture_result`` overwrites them after
        # each successful chat; leaving these here keeps the
        # pre-chat read contract intact without copying the primary's
        # values (which would silently override stubbed state).
        self.last_usage = None
        self.last_image_fallback_used = False
        self.preserve_reasoning = False

    def _sync_state(self) -> None:
        """Mirror per-call state onto both inner clients."""
        for name in self._MIRRORED_ATTRS:
            value = getattr(self, name)
            setattr(self._primary, name, value)
            setattr(self._fallback, name, value)

    def abort(self) -> None:
        stop = getattr(self, "stop_event", None)
        if stop is not None:
            stop.set()
        for client in (self._primary, self._fallback):
            try:
                client.abort()
            except AttributeError:
                pass
            except Exception:
                pass

    def _capture_result(self, client: BaseLLMClient) -> None:
        """Copy per-call metrics from the successful inner client.

        ``preserve_reasoning`` differs between providers — copying it
        from the provider that actually answered avoids stripping
        reasoning from a successful fallback response on a
        ``True``-using provider.
        """
        for name in self._CAPTURED_ATTRS:
            setattr(self, name, getattr(client, name))

    def _log_fallback(self, primary_error: BaseException) -> None:
        """Best-effort record of the fallback hop."""
        logger = getattr(self, "activity_logger", None)
        if logger is None:
            return
        logger.log(
            "llm_fallback",
            {
                "from_provider": self._primary.settings.llm_provider,
                "from_model": self._primary.settings.llm_model,
                "to_provider": self._fallback_label,
                "to_model": self._fallback.settings.llm_model,
                "primary_error_type": type(primary_error).__name__,
                "primary_error": str(primary_error),
            },
            run_id=self.run_id,
        )

    def _split_attempts(
        self, max_attempts: int | None
    ) -> tuple[int, int]:
        """Distribute ``max_attempts`` between primary and fallback.

        Without a caller cap, each inner client uses its own default
        retry budget. With a cap, primary gets up to two attempts so a
        single transient failure still recovers, and fallback takes the
        remainder. A one-attempt cap skips the fallback hop entirely.
        """
        if max_attempts is None:
            return None, None
        if max_attempts < 1:
            return 1, 0
        primary_attempts = min(2, max_attempts)
        fallback_attempts = max(0, max_attempts - primary_attempts)
        return primary_attempts, fallback_attempts

    def chat(
        self,
        messages,
        tools=None,
        *,
        max_attempts: int | None = None,
    ):
        self._sync_state()
        primary_attempts, fallback_attempts = self._split_attempts(max_attempts)
        try:
            result = self._primary.chat(
                messages, tools, max_attempts=primary_attempts
            )
        except RequestCancelled:
            raise
        except Exception as primary_error:
            # Stop events raised mid-flight must still win, even if the
            # primary surfaced a different exception first.
            stop = getattr(self, "stop_event", None)
            if stop is not None and stop.is_set():
                raise
            self._log_fallback(primary_error)
            if fallback_attempts == 0:
                raise
            try:
                result = self._fallback.chat(
                    messages, tools, max_attempts=fallback_attempts
                )
            except RequestCancelled:
                raise
            except Exception as fallback_error:
                raise fallback_error from primary_error
            self._capture_result(self._fallback)
            return result
        self._capture_result(self._primary)
        return result


# ---------------------------------------------------------------------------
# Shared Chat Completions client — shared retry loop and state
# ---------------------------------------------------------------------------


class ChatCompletionsClient(BaseLLMClient):
    """Shared base for OpenAI-compatible Chat Completions clients.

    Provider-specific logic (endpoint, headers, payload extras) is pushed into
    the thin subclasses via ``_endpoint``, ``_build_headers``, and
    ``_build_payload``. The retry loop, stream parsing, and image fallback
    are fully shared here.
    """

    preserve_reasoning: bool = False
    requires_api_key: bool = True

    def _stream_response(self, response: Any) -> dict[str, Any]:
        with self._response_lock:
            self._active_response = response
        try:
            result = parse_chat_stream(
                response,
                provider_label=self._provider_label,
                stop_event=self.stop_event,
                stream_callback=self.stream_callback,
            )
            usage = result.get("usage") if isinstance(result, dict) else None
            self.last_usage = usage if isinstance(usage, dict) else None
            return result
        finally:
            with self._response_lock:
                self._active_response = None
            if self.stop_event is not None and self.stop_event.is_set():
                _force_close_response(response)
            elif response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    def _try_image_fallback(self, payload: dict[str, Any], response: Any) -> bool:
        messages_without_images, removed = without_images(payload["messages"])
        if removed:
            response.close()
            payload["messages"] = messages_without_images
        return removed

    def _summarize_payload_for_logging(
        self, payload: dict[str, Any], is_dataset_mode: bool
    ) -> dict[str, Any] | None:
        attempt_payload = deepcopy(payload)
        if (
            isinstance(attempt_payload, dict)
            and isinstance(attempt_payload.get("messages"), list)
        ):
            attempt_payload["messages"] = summarize_llm_messages(
                attempt_payload["messages"],
                lossless=is_dataset_mode,
            )
        return attempt_payload

    def _parse_non_stream_response(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "read") and callable(response.read):
            try:
                response.read()
            except Exception:
                pass
        body = response.json()
        self.last_usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        choices = body.get("choices") if isinstance(body, dict) else None
        if choices and choices[0].get("finish_reason") == "length":
            raise RuntimeError(
                f"{self._provider_label} completion was truncated "
                "(finish_reason='length'); no tool calls were executed. "
                "Consider increasing 'llm.max_completion_tokens' "
                "in config.yaml or reducing 'reasoning_effort'."
            )
        if choices and isinstance(choices[0].get("message"), dict):
            msg = choices[0]["message"]
            r_content = msg.get("reasoning_content") or msg.get("thinking")
            if r_content and not msg.get("reasoning"):
                msg["reasoning"] = r_content
            clean_text, extracted_r = extract_think_tags(
                msg.get("content"), msg.get("reasoning")
            )
            if extracted_r and not msg.get("reasoning"):
                msg["reasoning"] = extracted_r
            msg["content"] = clean_text or None
        return body
