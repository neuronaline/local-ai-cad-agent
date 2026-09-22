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

from agent.activity_log import summarize_llm_messages
from agent.settings import Settings

PROVIDER_LABELS = {"openrouter": "OpenRouter", "openai": "OpenAI"}
TOOL_IMAGE_PROMPT = (
    "The attached image is the visual artifact returned by the latest tool "
    "call. Inspect it and continue the task."
)


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


def sanitize_assistant_message(
    message: dict[str, Any], *, preserve_reasoning: bool = False
) -> dict[str, Any]:
    """Copy a model response while retaining supported continuation fields.

    When ``preserve_reasoning`` is True, reasoning and reasoning_details are
    retained across all assistant turns (both tool-calling turns and conversational
    responses) so the model's reasoning memory is never dropped and prompt prefixes
    remain strictly append-only. Non-whitelisted fields (audio, logprobs, etc.)
    are dropped.
    """
    sanitized = deepcopy(message)
    # Normalize reasoning_content alias to reasoning if reasoning is not already present
    reasoning_content = sanitized.pop("reasoning_content", None)
    if (
        preserve_reasoning
        and isinstance(reasoning_content, str)
        and reasoning_content.strip()
        and not sanitized.get("reasoning")
    ):
        sanitized["reasoning"] = reasoning_content

    if not preserve_reasoning:
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

    for key in list(sanitized):
        if key.startswith("_"):
            sanitized.pop(key, None)
    # Drop any non-canonical field. Anything not in this whitelist is
    # provider-specific metadata that the canonical assistant record
    # (and the LLM context) does not carry.
    allowed = {"role", "content", "tool_calls"}
    if preserve_reasoning:
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

    ``cad_build_and_verify`` attaches its rendered PNG to the tool message so
    the agent can inspect the build in-band. Keep each completed tool batch's
    visual evidence in the subsequent conversation: dropping it after one
    turn changes the prompt prefix and causes avoidable cache misses (and
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


def _response_text(response: requests.Response) -> str:
    """Return an error body from real and test response objects."""
    try:
        value = response.text
        if callable(value):
            value = value()
        return value if isinstance(value, str) else ""
    except Exception:  # Error reporting must not mask the HTTP error.
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

    The worker publishes its :class:`requests.Response` to ``response_holder``
    *before* queueing it so the main thread can force-close the socket on
    the cancel path. Without that hook a rapid stop/restart cycle leaks a
    TCP socket (and the keepalive timer backing it) every time the user
    cancels before the worker has finished draining ``requests.post`` —
    the daemon thread keeps the underlying connection alive until process
    exit.
    """
    results: queue.Queue[requests.Response | BaseException] = queue.Queue(maxsize=1)
    # Mutable slot for the in-flight Response. ``dict``-based rather than
    # a list so ``setdefault``/``get`` give us atomic single-key access
    # without an extra lock.
    response_holder: dict[str, requests.Response] = {}

    def request_worker() -> None:
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout_seconds,
            )
        except BaseException as error:  # Propagate worker failures unchanged.
            results.put(error)
            return
        # Publish the Response *before* queueing it so a concurrent
        # cancel can close the socket even if the worker has not yet
        # finished its ``put`` call.
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
        # ``worker.join`` returned without timing out — the worker has
        # already put something on the queue, but ``get`` is bounded so
        # a stray stop between ``join`` and ``get`` cannot wedge us.
        result = results.get(timeout=5)
        if isinstance(result, BaseException):
            raise result
        return result
    except RequestCancelled:
        # Cancel path: force-close any Response we can see so the
        # underlying socket (and its keepalive timer) is released
        # immediately. The daemon worker may still be alive while its
        # ``requests.post`` is mid-handshake; wait briefly for it to
        # publish, then drain the queue and close whatever is there
        #.
        worker.join(timeout=2.0)
        _force_close_response(response_holder.get("response"))
        try:
            queued = results.get_nowait()
        except queue.Empty:
            queued = None
        if isinstance(queued, requests.Response):
            _force_close_response(queued)
        raise


def _force_close_response(response: requests.Response | None) -> None:
    """Force-close a streaming ``Response``, releasing the underlying socket.

    ``stream=True`` keeps the connection attached to the :class:`Response`
    until :meth:`Response.close` runs; without this helper the connection
    pool keeps the socket alive until the read times out, leaking FDs on
    rapid cancel/restart cycles.
    """
    if response is None:
        return
    try:
        response.close()
    except Exception:  # best-effort cleanup; the daemon worker will GC anyway
        pass


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
    reasoning_details: list[dict[str, Any]] = []

    try:
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
                content += text
                if stream_callback:
                    stream_callback({"type": "content", "delta": text})
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            details = delta.get("reasoning_details")
            if isinstance(details, list):
                reasoning_details.extend(
                    deepcopy(detail) for detail in details if isinstance(detail, dict)
                )
                # ``reasoning_details`` is the authoritative source for structured
                # thinking content. Do not also accumulate ``.text`` into
                # ``reasoning_text`` — that path is reserved for providers that emit
                # only a plain ``reasoning`` string per delta and would otherwise
                # double-count the same text into both the structured list and the
                # fallback string.
                reasoning_delta_text: str | None = None
            elif isinstance(reasoning, str) and reasoning:
                reasoning_delta_text = reasoning
            else:
                reasoning_delta_text = None
            if reasoning_delta_text:
                reasoning_text += reasoning_delta_text
                if stream_callback:
                    stream_callback({"type": "reasoning", "delta": reasoning_delta_text})
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
    except (requests.exceptions.RequestException, OSError):
        if stop_event and stop_event.is_set():
            try:
                response.close()
            except Exception:
                pass
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
        had_partial = bool(content or tool_calls or reasoning_text or reasoning_details)
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
        if reasoning_details:
            message["reasoning_details"] = reasoning_details
        elif reasoning_text:
            message["reasoning"] = reasoning_text
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


def _build_provider_client(provider: str, settings: Settings):
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
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class FallbackChatClient:
    """Try a primary ``ChatCompletionsClient`` and fall back once on failure.

    The wrapper mirrors the public surface of :class:`ChatCompletionsClient`
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
        primary: ChatCompletionsClient,
        fallback: ChatCompletionsClient,
        fallback_label: str,
    ) -> None:
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

    def _capture_result(self, client: ChatCompletionsClient) -> None:
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


class ChatCompletionsClient:
    """Shared base for OpenAI-compatible Chat Completions clients.

    Provider-specific logic (endpoint, headers, payload extras) is pushed into
    the thin subclasses via ``_endpoint``, ``_build_headers``, and
    ``_build_payload``. The retry loop, stream parsing, and image fallback
    are fully shared here.
    """

    # Whether this provider needs the latest assistant turn's reasoning
    # payloads (``reasoning`` / ``reasoning_details``) on the wire so it can
    # continue an interrupted tool-call response. OpenRouter-compatible
    # providers (Anthropic extended thinking, xai reasoning, Gemini thinking)
    # require this; OpenAI rejects unknown reasoning fields and never sets
    # the flag. Both the wire-payload sanitizer and the agent runner's
    # per-message sanitizer read this attribute, so there is a single source
    # of truth for "should we keep reasoning?" rather than two parallel
    # implementations that can drift.
    preserve_reasoning: bool = False

    def __init__(self, settings: Settings, provider_label: str) -> None:
        self.settings = settings
        self.stop_event = None
        self.session_id: str | None = None
        # Subordinate evaluator hook. The structured ``cad_review`` tool (the
        # historical setter of ``"reviewer"``) was removed from the tool
        # surface; the field is preserved here so a future sub-agent can
        # tag its ``trace.span_name`` on the OpenRouter adapter without
        # changing the cache prefix on the base client. Today nothing
        # outside this module sets the attribute, so the value is always
        # ``None`` in production. The OpenAI adapter does not consume it.
        self.agent_role: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_image_fallback_used = False
        self.stream_callback = None
        self.require_images = False
        self._provider_label = provider_label
        # Optional activity-log hook. The agent runner wires this when
        # ``agent.log_tool_activity`` is enabled; ``None`` keeps the wire
        # path inert for tests and review sub-sessions that do not log.
        # Public on purpose: :class:`agent.core.AgentRunner` writes these
        # from outside the client, so they are part of the documented
        # surface and consumed by ``chat()`` directly.
        self.activity_logger = None
        self.run_id: str | None = None
        self._active_response: requests.Response | None = None
        self._response_lock = threading.Lock()

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
        return sanitize_messages(messages, preserve_reasoning=self.preserve_reasoning)

    def _stream_response(self, response):
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

    def abort(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        with self._response_lock:
            resp = self._active_response
        if resp is not None:
            try:
                resp.close()
                if hasattr(resp, "raw") and resp.raw is not None:
                    resp.raw.close()
            except Exception:
                pass

    def _try_image_fallback(self, payload, response):
        messages_without_images, removed = without_images(payload["messages"])
        if removed:
            response.close()
            payload["messages"] = messages_without_images
        return removed

    # How many attempts the inner retry loop makes when ``chat()`` is
    # called directly. ``FallbackChatClient`` passes a per-call override
    # so a primary failure does not burn the wrapper's documented
    # "one shot" budget on the fallback hop.
    _DEFAULT_RETRY_ATTEMPTS = 3

    def chat(self, messages, tools=None, *, max_attempts: int | None = None):
        attempts = (
            max_attempts
            if max_attempts is not None
            else self._DEFAULT_RETRY_ATTEMPTS
        )
        self.last_usage = None
        self.last_image_fallback_used = False
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(f"{api_key_env(self._provider_label.lower())} is not configured.")
        payload = self._build_payload(messages, tools)
        headers = self._build_headers(api_key)
        log_payload = self.activity_logger is not None

        image_fallback_used = False
        for attempt in range(attempts):
            is_last_attempt = attempt == attempts - 1
            response: requests.Response | None = None
            # When activity logging is enabled, replace the raw
            # ``messages`` array with a structural summary so a single
            # multi-image upload cannot exhaust the 5 MiB rolling cap
            # and the activity log never echoes prompt content in
            # plaintext. ``redact()`` already scrubs ``authorization`` /
            # provider-key headers.
            attempt_payload = deepcopy(payload) if log_payload else None
            if (
                log_payload
                and isinstance(attempt_payload, dict)
                and isinstance(attempt_payload.get("messages"), list)
            ):
                attempt_payload["messages"] = summarize_llm_messages(
                    attempt_payload["messages"]
                )
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
                        self.activity_logger.log(
                            "llm_request",
                            {
                                "attempt": attempt,
                                "url": self._endpoint(),
                                "model": payload.get("model"),
                                "headers": dict(headers),
                                "payload": attempt_payload,
                                "status": response.status_code,
                            },
                            run_id=self.run_id,
                        )
                    if hasattr(response, "iter_lines"):
                        try:
                            return self._stream_response(response)
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
                        return body
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
