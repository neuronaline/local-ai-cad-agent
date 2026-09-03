"""Tests for the OpenAI-compatible chat completions adapters (OpenRouter and OpenAI)."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import pytest
import requests

from agent.llm_base import create_llm_client
from agent.openai_client import OpenAIClient
from agent.openrouter import OpenRouterClient
from agent.prompt import get_prompt_cache_key
from agent.settings import LLM_PROVIDERS, Settings, load_settings
from agent.tool_schemas import TOOL_SCHEMAS

# ---------------------------------------------------------------------------
# Shared fake responses
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` for non-streaming responses."""

    status_code: int = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": []}

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    def text(self) -> str:  # pragma: no cover - default no-op
        return ""


class _StreamingResponse(_FakeResponse):
    """SSE stream that exercises reasoning, content, and tool-call deltas."""

    def __init__(self, first_text: str = "Building ", rest_text: str = "now.") -> None:
        self._lines = [
            b'data: {"choices":[{"delta":{"role":"assistant","reasoning":"Checking geometry. "}}]}',
            f'data: {{"choices":[{{"delta":{{"role":"assistant","content":"{first_text}"}}}}]}}'.encode(),
            (
                b'data: {"choices":[{"delta":{"content":"'
                + rest_text.encode()
                + b'","tool_calls":[{"index":0,"id":"call-1","function":{"name":"cad","arguments":"{\\"operation\\":"}}]}}]}'
            ),
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"run\\"}"}}]}}]}',
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}',
            b"data: [DONE]",
        ]

    def iter_lines(self):
        return iter(self._lines)


class _ReasoningDetailsToolResponse(_FakeResponse):
    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"Inspecting ","id":"test-rd-1","format":"xai-responses-v1","index":0}]}}]}',
                b'data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"the model.","id":"test-rd-2","format":"xai-responses-v1","index":1}],"tool_calls":[{"index":0,"id":"call-1","function":{"name":"write_file","arguments":"{}"}}]}}]}',
                b'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
                b"data: [DONE]",
            ]
        )


class _EmptyResponse(_FakeResponse):
    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _MidStreamErrorResponse(_FakeResponse):
    def __init__(self, message: str = "provider failed") -> None:
        self._lines = [
            f'data: {{"error":{{"message":"{message}"}},"choices":[{{"finish_reason":"error","delta":{{}}}}]}}'.encode(),
            b"data: [DONE]",
        ]

    def iter_lines(self):
        return iter(self._lines)


class _TruncatedStreamingResponse(_FakeResponse):
    def iter_lines(self):
        return iter([b'data: {"choices":[{"delta":{"content":"partial"}}]}'])


class _ReasoningOnlyStopResponse(_FakeResponse):
    """Reasoning-first model with finish_reason='stop' and no text content."""

    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning":"Thinking through geometry."}}]}',
                b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}',
                b'data: [DONE]',
            ]
        )


class _ReasoningOnlyLengthResponse(_FakeResponse):
    """Reasoning-first model that ran out of tokens before producing content."""

    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning":"Trying to plan."}}]}',
                b'data: {"choices":[{"finish_reason":"length","delta":{}}]}',
                b'data: [DONE]',
            ]
        )


class _EmptyLengthResponse(_FakeResponse):
    """No content, no reasoning, finish_reason='length'."""

    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"finish_reason":"length","delta":{}}]}',
                b'data: [DONE]',
            ]
        )


class _ToolCallLengthResponse(_FakeResponse):
    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"cad_build_and_verify","arguments":"{}"}}]}}]}',
                b'data: {"choices":[{"finish_reason":"length","delta":{}}]}',
                b'data: [DONE]',
            ]
        )


class _ClientErrorResponse(_FakeResponse):
    status_code: int = 400

    def raise_for_status(self) -> None:
        raise requests.HTTPError("bad request")


class _ImageRejectedResponse(_FakeResponse):
    status_code: int = 404
    text: str = '{"error":{"message":"Model does not support image inputs."}}'

    def close(self) -> None:
        return None

    def raise_for_status(self) -> None:
        raise requests.HTTPError("bad request")


class _RateLimitedResponse(_FakeResponse):
    status_code: int = 429
    headers: ClassVar[dict[str, str]] = {"Retry-After": "0"}

    def raise_for_status(self) -> None:
        raise requests.HTTPError("rate limited")


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path, provider: str, **overrides) -> Settings:
    base = {
        "workspace_root": tmp_path,
        "openrouter_base_url": "https://example.test",
        "openrouter_model": "openai/gpt-4o-mini",
        "openrouter_timeout_seconds": 1,
        "host": "127.0.0.1",
        "port": 5000,
        "openai_base_url": "https://example.test",
        "openai_model": "gpt-4o-mini",
        "openai_timeout_seconds": 1,
    }
    if provider == "openai":
        base.update(llm_provider="openai", **overrides)
    else:
        base.update(llm_provider="openrouter", **overrides)
    return Settings(**base)


_CLIENT_KINDS = [
    pytest.param("openai", id="openai"),
    pytest.param("openrouter", id="openrouter"),
]


def _client_for(kind: str, settings: Settings):
    return OpenAIClient(settings) if kind == "openai" else OpenRouterClient(settings)


def _patch_post_for(monkeypatch, kind: str, response_factory):
    """Patch ``requests.post`` on the shared base module that both clients use."""
    monkeypatch.setattr("agent.llm_base.requests.post", response_factory)


def _capture_post(monkeypatch, monkeypatch_target: str, sink: dict[str, Any], response_factory):
    """Capture ``url`` and ``kwargs`` of the next ``requests.post`` call into ``sink``."""
    monkeypatch.setattr(
        monkeypatch_target,
        lambda url, *_a, **kwargs: sink.update(url=url, **kwargs) or response_factory(),
    )


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_create_llm_client_routes_to_selected_provider(tmp_path, kind):
    settings = _settings(tmp_path, kind)
    client = create_llm_client(settings)
    assert isinstance(client, _client_for(kind, settings).__class__)


# ---------------------------------------------------------------------------
# Behavior shared across both providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_streams_content_and_tool_calls(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    _patch_post_for(monkeypatch, kind, lambda *_a, **_kw: _StreamingResponse())
    client = _client_for(kind, _settings(tmp_path, kind))
    events: list[dict[str, Any]] = []
    client.stream_callback = events.append

    response = client.chat([{"role": "user", "content": "build"}])

    message = response["choices"][0]["message"]
    assert message["content"] == "Building now."
    assert message["tool_calls"][0]["function"] == {"name": "cad", "arguments": '{"operation":"run"}'}
    assert [e["type"] for e in events] == ["reasoning", "content", "content", "tool_call", "tool_call"]
    assert events[0]["delta"] == "Checking geometry. "
    assert client.last_usage["completion_tokens"] == 5


@pytest.mark.parametrize(
    ("kind", "response", "message"),
    [
        pytest.param(
            "openai",
            _MidStreamErrorResponse("openai failure"),
            "openai failure",
            id="openai-midstream-error",
        ),
        pytest.param(
            "openrouter",
            _MidStreamErrorResponse("provider failed"),
            "provider failed",
            id="openrouter-midstream-error",
        ),
        pytest.param(
            "openrouter",
            _TruncatedStreamingResponse(),
            "completion marker",
            id="openrouter-truncated",
        ),
    ],
)
def test_client_rejects_failed_or_truncated_stream(
    monkeypatch, tmp_path, kind, response, message
):
    env_var = "OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY"
    monkeypatch.setenv(env_var, "test")
    target = "agent.llm_base.requests.post"
    monkeypatch.setattr(target, lambda *_a, **_kw: response)

    with pytest.raises(RuntimeError, match=message):
        _client_for(kind, _settings(tmp_path, kind)).chat(
            [{"role": "user", "content": "build"}]
        )


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_surfaces_reasoning_when_completion_has_no_text(
    monkeypatch, tmp_path, kind
):
    """Reasoning-first models (Anthropic extended thinking, OpenAI o-series,
    Gemini thinking) can finish cleanly without emitting text. The reasoning
    text is the only assistant payload; surface it as the message content
    instead of crashing with a generic empty-completion error."""
    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    target = "agent.llm_base.requests.post"
    monkeypatch.setattr(target, lambda *_a, **_kw: _ReasoningOnlyStopResponse())

    response = _client_for(kind, _settings(tmp_path, kind)).chat(
        [{"role": "user", "content": "build"}]
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "Thinking through geometry."
    assert "tool_calls" not in message


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_reports_finish_reason_when_completion_is_truly_empty(
    monkeypatch, tmp_path, kind
):
    """A response with finish_reason='length' and no content/tool calls is a
    token-budget exhaustion. The error must call out 'length' so the operator
    can raise max_completion_tokens; reasoning is *not* substituted."""
    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    target = "agent.llm_base.requests.post"
    monkeypatch.setattr(target, lambda *_a, **_kw: _EmptyLengthResponse())

    with pytest.raises(RuntimeError, match=r"finish_reason='length'"):
        _client_for(kind, _settings(tmp_path, kind)).chat(
            [{"role": "user", "content": "build"}]
        )


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_does_not_substitute_reasoning_when_truncated(
    monkeypatch, tmp_path, kind
):
    """Reasoning streamed but finish_reason='length' (output truncated before
    any visible content) is still an error — substituting reasoning for the
    missing answer would mask the real problem."""
    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    target = "agent.llm_base.requests.post"
    monkeypatch.setattr(target, lambda *_a, **_kw: _ReasoningOnlyLengthResponse())

    with pytest.raises(RuntimeError, match=r"finish_reason='length'"):
        _client_for(kind, _settings(tmp_path, kind)).chat(
            [{"role": "user", "content": "build"}]
        )


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_rejects_tool_call_from_truncated_completion(
    monkeypatch, tmp_path, kind
):
    monkeypatch.setenv(
        "OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test"
    )
    monkeypatch.setattr(
        "agent.llm_base.requests.post", lambda *_a, **_kw: _ToolCallLengthResponse()
    )

    with pytest.raises(RuntimeError, match="no tool calls were executed"):
        _client_for(kind, _settings(tmp_path, kind)).chat(
            [{"role": "user", "content": "build"}]
        )

@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_falls_back_when_provider_rejects_vision(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    payloads: list[dict[str, Any]] = []

    def post(*_a, **kwargs):
        payloads.append(kwargs["json"])
        return _ImageRejectedResponse() if len(payloads) == 1 else _StreamingResponse()

    _patch_post_for(monkeypatch, kind, post)
    settings = _settings(tmp_path, kind, openrouter_model="text-only-model", openai_model="text-only-model")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review this render."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }]

    response = _client_for(kind, settings).chat(messages)

    assert response["choices"][0]["message"]["content"] == "Building now."
    assert len(payloads) == 2
    assert payloads[1]["messages"][0]["content"] == [{"type": "text", "text": "Review this render."}]


# ---------------------------------------------------------------------------
# OpenAI-specific behavior
# ---------------------------------------------------------------------------


def test_openai_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(_settings(tmp_path, "openai")).chat([{"role": "user", "content": "hi"}])


def test_openai_builds_payload_with_reasoning_effort(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _EmptyResponse)
    settings = _settings(tmp_path, "openai", openai_model="o4-mini", openai_reasoning_effort="high")

    OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])

    payload = captured["json"]
    assert payload["model"] == "o4-mini"
    assert payload["reasoning_effort"] == "high"
    assert payload["prompt_cache_key"] == get_prompt_cache_key()
    # OpenAI does not consume OpenRouter-specific keys.
    for key in ("provider", "session_id", "cache_control"):
        assert key not in payload
    assert captured["headers"]["Authorization"] == "Bearer test"
    assert "HTTP-Referer" not in captured["headers"]
    assert "X-OpenRouter-Title" not in captured["headers"]


def test_openai_cache_key_is_stable_and_dynamic_state_follows_system_prompt(tmp_path):
    client = OpenAIClient(_settings(tmp_path, "openai"))
    payload = client._build_payload(
        [
            {"role": "system", "content": "Stable instructions."},
            {"role": "user", "content": "<project_state>dynamic</project_state>"},
            {"role": "user", "content": "Build a bracket."},
        ],
        None,
    )

    assert payload["prompt_cache_key"] == get_prompt_cache_key()
    assert payload["messages"] == [
        {"role": "system", "content": "Stable instructions."},
        {"role": "user", "content": "<project_state>dynamic</project_state>"},
        {"role": "user", "content": "Build a bracket."},
    ]


def test_openai_endpoint_uses_configured_base_url(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _EmptyResponse)
    settings = _settings(tmp_path, "openai", openai_base_url="https://proxy.example.test/v1")

    OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])

    assert captured["url"] == "https://proxy.example.test/v1/chat/completions"


# ---------------------------------------------------------------------------
# Tool-message image relocation (OpenAI spec compliance)
# ---------------------------------------------------------------------------


def test_relocate_tool_images_moves_image_url_parts_to_user_message():
    """Multimodal tool messages are rejected by both OpenAI and Gemini because
    the Chat Completions spec only permits ``image_url`` in user messages.
    ``relocate_tool_images`` must move the image parts into a trailing user
    turn so the model still sees the visuals without violating the spec."""
    from agent.llm_base import relocate_tool_images

    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAA"},
    }
    messages = [
        {"role": "user", "content": "build a bracket"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "cad_build_and_verify", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "build ok"},
                image,
            ],
        },
    ]

    relocated = relocate_tool_images(messages)

    assert len(relocated) == 4
    # The tool message keeps its text but loses the image parts.
    tool = relocated[2]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "c1"
    assert tool["content"] == "build ok"
    # The trailing user turn carries the relocated image.
    follow_up = relocated[3]
    assert follow_up["role"] == "user"
    assert isinstance(follow_up["content"], list)
    assert follow_up["content"][0]["type"] == "text"
    # Image parts must be deep-copied, not shared with the original list.
    assert follow_up["content"][1] == image
    assert follow_up["content"][1] is not image
    # The original tool message content is untouched in the caller's buffer.
    assert messages[2]["content"][1] is image


def test_relocate_tool_images_passes_text_only_tool_messages_through():
    from agent.llm_base import relocate_tool_images

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "c1", "content": "no images here"},
    ]
    assert relocate_tool_images(messages) is not messages
    assert relocate_tool_images(messages) == messages


def test_relocate_tool_images_handles_image_only_tool_messages():
    from agent.llm_base import relocate_tool_images

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    messages = [{"role": "tool", "tool_call_id": "c1", "content": [image]}]

    relocated = relocate_tool_images(messages)
    assert relocated[0]["role"] == "tool"
    # With no text parts left, fall back to a deterministic placeholder so
    # the tool message is never empty.
    assert relocated[0]["content"] == "An attached image was unavailable."
    assert relocated[1]["role"] == "user"
    assert relocated[1]["content"][1] == image


def test_relocate_tool_images_only_sends_the_latest_render():
    from agent.llm_base import relocate_tool_images

    old_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,OLD"},
    }
    new_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,NEW"},
    }
    messages = [
        {
            "role": "tool",
            "tool_call_id": "old",
            "content": [{"type": "text", "text": "old build"}, old_image],
        },
        {"role": "assistant", "content": "adjusting"},
        {
            "role": "tool",
            "tool_call_id": "new",
            "content": [{"type": "text", "text": "new build"}, new_image],
        },
    ]

    relocated = relocate_tool_images(messages)

    image_messages = [
        message
        for message in relocated
        if message.get("role") == "user" and isinstance(message.get("content"), list)
    ]
    assert len(image_messages) == 1
    assert image_messages[0]["content"][1] == new_image
    assert relocated[0]["content"] == "old build"


def test_image_fallback_removes_synthetic_render_instruction():
    from agent.llm_base import TOOL_IMAGE_PROMPT, without_images

    messages = [
        {"role": "tool", "tool_call_id": "c1", "content": "build ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TOOL_IMAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
            ],
        },
    ]

    stripped, removed = without_images(messages)

    assert removed is True
    assert stripped == [messages[0]]


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_client_does_not_hide_unrelated_400_with_image_fallback(
    monkeypatch, tmp_path, kind
):
    monkeypatch.setenv(
        "OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test"
    )
    calls = []

    def post(*_args, **kwargs):
        calls.append(kwargs["json"])
        return _ClientErrorResponse()

    _patch_post_for(monkeypatch, kind, post)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        }
    ]

    with pytest.raises(requests.HTTPError):
        _client_for(kind, _settings(tmp_path, kind)).chat(messages)

    assert len(calls) == 1


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_tool_render_fallback_does_not_leave_a_false_user_instruction(
    monkeypatch, tmp_path, kind
):
    monkeypatch.setenv(
        "OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test"
    )
    payloads = []

    def post(*_args, **kwargs):
        payloads.append(deepcopy(kwargs["json"]))
        return _ImageRejectedResponse() if len(payloads) == 1 else _StreamingResponse()

    _patch_post_for(monkeypatch, kind, post)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "cad_build_and_verify",
                        "arguments": '{"render":true}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "build ok"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
    ]

    _client_for(kind, _settings(tmp_path, kind)).chat(messages)

    assert payloads[0]["messages"][-1]["role"] == "user"
    assert payloads[1]["messages"][-1]["role"] == "tool"
    assert all(message.get("role") != "user" for message in payloads[1]["messages"])


def test_sanitize_messages_removes_image_url_from_tool_messages(tmp_path, monkeypatch):
    """End-to-end: ``OpenAIClient.chat`` must not POST a payload where any
    ``tool`` message still contains ``image_url`` parts."""
    from agent.llm_base import sanitize_messages

    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _EmptyResponse)
    settings = _settings(tmp_path, "openai")
    client = OpenAIClient(settings)

    messages = sanitize_messages(
        [
            {"role": "user", "content": "build a bracket"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "cad_build_and_verify", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {"type": "text", "text": "ok"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                ],
            },
        ]
    )
    client.chat(messages)

    sent = captured["json"]["messages"]
    for message in sent:
        if message["role"] != "tool":
            continue
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            assert not (
                isinstance(part, dict) and part.get("type") == "image_url"
            ), f"tool message still carries image_url: {message!r}"
    # The trailing image parts must end up in a user-role message.
    assert sent[-1]["role"] == "user"
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in sent[-1]["content"]
    )


def test_sanitize_messages_preserves_latest_mutation_and_read_results():
    from agent.llm_base import sanitize_messages

    old_read = json.dumps(
        {
            "ok": True,
            "tool": "read_file",
            "data": json.dumps(
                {
                    "exists": True,
                    "content": "x" * 10_000,
                    "sha256": "a" * 64,
                    "total_lines": 200,
                }
            ),
        }
    )
    latest_read = old_read.replace("x" * 10_000, "latest source")
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "edit-old",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {
                                "filename": "model.py",
                                "old_string": "a" * 5_000,
                                "new_string": "b" * 5_000,
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "edit-old", "content": "edited"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-old",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"filename":"model.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read-old", "content": old_read},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "read-latest",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"filename":"model.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read-latest", "content": latest_read},
    ]

    sanitized = sanitize_messages(messages)

    old_args = sanitized[0]["tool_calls"][0]["function"]["arguments"]
    assert "a" * 1_000 in old_args
    assert "x" * 100 in sanitized[3]["content"]
    assert "latest source" in sanitized[5]["content"]
    assert "x" * 1_000 in old_read
    assert "a" * 1_000 in messages[0]["tool_calls"][0]["function"]["arguments"]


def test_sanitize_messages_compacts_superseded_mutations_truthfully():
    from agent.llm_base import sanitize_messages

    messages = []
    for index, content in enumerate(("a" * 5_000, "b" * 5_000), 1):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"write-{index}",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"filename": "model.py", "content": content}),
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": f"write-{index}", "content": "written"},
            ]
        )

    sanitized = sanitize_messages(messages)
    first_args = sanitized[0]["tool_calls"][0]["function"]["arguments"]
    latest_args = sanitized[2]["tool_calls"][0]["function"]["arguments"]
    assert len(first_args) < 300
    assert "Historical successful write" in first_args
    assert "omitted" not in first_args
    assert "b" * 1_000 in latest_args


def test_sanitize_messages_preserves_inflight_latest_mutation():
    """An in-flight mutation (no tool response yet) must keep its full args.

    ``_compact_completed_history`` previously computed ``latest_mutation_id``
    over completed mutations only; an in-flight edit would be excluded from
    the preserved set and overwritten with the historical-success placeholder
    before the tool response arrived, shifting the wire payload mid-turn.
    """
    from agent.llm_base import sanitize_messages

    earlier_args = json.dumps({"filename": "model.py", "content": "earlier"})
    latest_args = json.dumps({"filename": "model.py", "content": "latest"})
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "edit-old",
                "type": "function",
                "function": {"name": "edit_file", "arguments": earlier_args},
            }],
        },
        {"role": "tool", "tool_call_id": "edit-old", "content": "edited"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "edit-latest",
                "type": "function",
                "function": {"name": "edit_file", "arguments": latest_args},
            }],
        },
    ]

    sanitized = sanitize_messages(messages)

    old_args = sanitized[0]["tool_calls"][0]["function"]["arguments"]
    new_args = sanitized[2]["tool_calls"][0]["function"]["arguments"]
    assert "earlier" not in old_args
    assert "superseded" in old_args
    assert latest_args in new_args


# ---------------------------------------------------------------------------
# OpenRouter-specific behavior
# ---------------------------------------------------------------------------


def test_openrouter_retries_transient_failure(monkeypatch, tmp_path):
    calls: list[int] = []

    def post(*_a, **_kw):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary")
        return _FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.llm_base.requests.post", post)
    monkeypatch.setattr("agent.llm_base.sleep_with_cancel", lambda _delay, _event: None)

    response = OpenRouterClient(_settings(tmp_path, "openrouter")).chat([{"role": "user", "content": "hi"}])

    assert response == {"choices": []}
    assert len(calls) == 2


def test_openrouter_does_not_retry_non_transient_client_error(monkeypatch, tmp_path):
    calls: list[int] = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda *_a, **_kw: calls.append(1) or _ClientErrorResponse(),
    )

    with pytest.raises(requests.HTTPError):
        OpenRouterClient(_settings(tmp_path, "openrouter")).chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 1


def test_openrouter_builds_cache_safe_sticky_payload_and_keeps_tool_content(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _FakeResponse)
    settings = _settings(
        tmp_path,
        "openrouter",
        openrouter_model="anthropic/claude-sonnet-4",
        openrouter_app_title="CAD Test",
        openrouter_app_url="https://cad.example",
    )
    client = OpenRouterClient(settings)
    client.session_id = "project:demo"
    client.chat([
        {"role": "system", "content": "static"},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "cad", "arguments": "{}"}}],
            "reasoning_details": [{
                "type": "reasoning.text",
                "text": "internal",
                "id": "reasoning-1",
                "format": "xai-responses-v1",
                "index": 0,
            }],
            "_provider_debug": True,
        },
    ])

    assistant = captured["json"]["messages"][1]
    assert assistant["content"] == "I will inspect it."
    assert "tool_calls" in assistant
    assert assistant["reasoning_details"][0]["text"] == "internal"
    assert "reasoning" not in assistant
    assert "_provider_debug" not in assistant
    assert captured["json"]["session_id"] == hashlib.sha256(b"project:demo").hexdigest()[:64]
    assert captured["json"]["cache_control"] == {"type": "ephemeral"}
    assert captured["headers"]["X-OpenRouter-Title"] == "CAD Test"
    assert captured["headers"]["HTTP-Referer"] == "https://cad.example"


def test_openrouter_stream_preserves_reasoning_details_for_tool_continuation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda *_args, **_kwargs: _ReasoningDetailsToolResponse(),
    )
    client = OpenRouterClient(_settings(tmp_path, "openrouter"))

    response = client.chat([{"role": "user", "content": "build"}])
    assistant = response["choices"][0]["message"]
    assert [item["text"] for item in assistant["reasoning_details"]] == [
        "Inspecting ",
        "the model.",
    ]

    payload = client._build_payload(
        [
            {"role": "user", "content": "build"},
            assistant,
            {"role": "tool", "tool_call_id": "call-1", "content": "written"},
        ],
        TOOL_SCHEMAS[:1],
    )
    assert payload["messages"][1]["reasoning_details"] == assistant["reasoning_details"]
    # ``reasoning_text`` must NOT be a duplicate of the structured list;
    # ``parse_chat_stream`` should keep one source of truth.
    assert "reasoning" not in payload["messages"][1]


def test_openrouter_wire_payload_drops_reasoning_on_historical_assistant_turns(tmp_path):
    """Historical assistant turns must not carry ``reasoning_details`` on the wire.

    The provider only needs the latest assistant turn's reasoning chain to
    continue an interrupted tool-call response. Re-sending every previous
    turn's reasoning would grow the wire payload linearly with the iteration
    count. ``sanitize_messages(..., preserve_reasoning=True)`` therefore
    preserves reasoning only on the last assistant message.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        messages = [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "build a bracket"},
            {
                "role": "assistant",
                "content": "First plan.",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{}"},
                }],
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "text": "first",
                    "id": "rd-1",
                    "format": "xai-responses-v1",
                    "index": 0,
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {
                "role": "assistant",
                "content": "Second plan.",
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": "{}"},
                }],
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "text": "second",
                    "id": "rd-2",
                    "format": "xai-responses-v1",
                    "index": 0,
                }],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "ok"},
            {
                "role": "assistant",
                "content": "Third plan.",
                "tool_calls": [{
                    "id": "call-3",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": "{}"},
                }],
                "reasoning_details": [{
                    "type": "reasoning.text",
                    "text": "third",
                    "id": "rd-3",
                    "format": "xai-responses-v1",
                    "index": 0,
                }],
            },
        ]
        client = OpenRouterClient(_settings(tmp_path, "openrouter"))
        wire = client._build_payload(messages, None)["messages"]
        assistant_indices = [
            index for index, message in enumerate(wire)
            if message.get("role") == "assistant"
        ]
        last_index = assistant_indices[-1]
        for index in assistant_indices:
            if index == last_index:
                assert wire[index].get("reasoning_details"), (
                    "Latest assistant turn must keep reasoning_details for "
                    "provider tool-call continuation."
                )
            else:
                assert "reasoning_details" not in wire[index], (
                    f"Historical assistant turn at index {index} must not carry "
                    "reasoning_details on the wire."
                )
                assert "reasoning" not in wire[index]
    finally:
        monkeypatch.undo()


def test_openrouter_preserve_reasoning_attribute_replaces_override(tmp_path):
    """The shared ``preserve_reasoning`` class attribute replaces the override.

    The override used to live on ``OpenRouterClient.sanitize_messages``; that
    hook has been removed in favour of a single class attribute that the base
    ``sanitize_messages`` reads. Both the wire-payload sanitizer and the agent
    runner's per-message sanitizer now read ``client.preserve_reasoning``.
    """
    assert OpenRouterClient.preserve_reasoning is True
    # Inherited default keeps OpenAI from leaking reasoning fields.
    from agent.llm_base import ChatCompletionsClient
    assert ChatCompletionsClient.preserve_reasoning is False
    assert OpenAIClient.preserve_reasoning is False

    # Sanity: the override hook is gone from the public surface.
    assert "sanitize_messages" not in vars(OpenRouterClient)
    assert "sanitize_messages" not in vars(OpenAIClient)


def test_parse_chat_stream_reasoning_only_detail_deltas_emit_no_duplicate_text(monkeypatch):
    """``reasoning_details`` deltas with no parallel ``reasoning`` string.

    Verifies the rebuilt message carries the structured details and does not
    also stuff ``.text`` into a plain ``reasoning`` field — those are two
    representations of the same content, and double-counting would let the
    ``empty content with no tool_calls`` fallback copy the duplicated text into
    ``content``.
    """
    from agent.llm_base import parse_chat_stream

    response = _ReasoningDetailsToolResponse()
    result = parse_chat_stream(
        response,
        provider_label="OpenRouter",
        stop_event=None,
        stream_callback=None,
    )
    message = result["choices"][0]["message"]
    assert "reasoning_details" in message
    assert "reasoning" not in message


def test_parse_chat_stream_plain_reasoning_string_still_falls_back(monkeypatch):
    """When the provider emits only ``reasoning`` strings, still accumulate."""
    from agent.llm_base import parse_chat_stream

    response = _StreamingResponse()
    result = parse_chat_stream(
        response,
        provider_label="OpenRouter",
        stop_event=None,
        stream_callback=None,
    )
    # ``_StreamingResponse`` carries ``reasoning`` first, then content + tool;
    # the rebuilt message has tool_calls so ``reasoning`` is attached.
    message = result["choices"][0]["message"]
    assert message["reasoning"] == "Checking geometry. "


def test_openai_drops_openrouter_reasoning_details(tmp_path):
    client = OpenAIClient(_settings(tmp_path, "openai"))
    payload = client._build_payload(
        [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "write_file", "arguments": "{}"},
            }],
            "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
        }],
        None,
    )
    assert "reasoning_details" not in payload["messages"][0]


def test_openai_strips_plain_reasoning_string_defensively(tmp_path):
    """Defensive strip on the OpenAI client catches every assistant message.

    ``sanitize_messages`` already drops these on the default path, but the
    defensive strip in :meth:`OpenAIClient._build_payload` is the only layer
    that protects against a future caller flipping ``preserve_reasoning=True``
    or bypassing the sanitizer. Confirm both keys are absent on the wire.
    """
    from agent.llm_base import sanitize_messages

    sanitized = sanitize_messages([{
        "role": "assistant",
        "content": "decided",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "noop", "arguments": "{}"},
        }],
        "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
        "reasoning": "private string",
    }])
    settings = _settings(tmp_path, "openai")
    # Force the message past ``sanitize_messages`` with reasoning preserved so
    # the defensive strip has something to drop.
    sanitized[0]["reasoning"] = "private string"
    payload = OpenAIClient(settings)._build_payload(sanitized, None)
    assistant = payload["messages"][0]
    assert "reasoning" not in assistant
    assert "reasoning_details" not in assistant


def test_openrouter_forced_provider_keeps_sticky_routing_eligible(tmp_path):
    settings = _settings(
        tmp_path,
        "openrouter",
        openrouter_provider="google-vertex/global",
        openrouter_force_provider=True,
    )
    payload = OpenRouterClient(settings)._build_payload([], None)

    assert payload["provider"] == {
        "only": ["google-vertex/global"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_openrouter_agent_role_trace_propagates_to_payload(tmp_path):
    """Subordinate evaluators identify their trace span.

    The ``agent_role`` attribute on the client is forwarded through
    OpenRouter's documented ``trace.span_name`` field. The base prompt and
    ``session_id`` are unchanged, so the cache prefix key stays stable.
    """
    client = OpenRouterClient(_settings(tmp_path, "openrouter"))
    client.agent_role = "reviewer"
    client.session_id = "test-session-abc"
    payload = client._build_payload([], None)
    assert payload.get("trace") == {"span_name": "reviewer"}
    assert payload.get("session_id") == hashlib.sha256(
        b"test-session-abc"
    ).hexdigest()[:64]


def test_openrouter_agent_role_default_is_absent(tmp_path):
    """The default agent loop must not identify a subordinate trace span."""
    payload = OpenRouterClient(_settings(tmp_path, "openrouter"))._build_payload(
        [], None
    )
    assert "trace" not in payload

def test_openrouter_marks_gemini_messages_without_rewriting_old_prefix(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _FakeResponse)
    settings = _settings(tmp_path, "openrouter", openrouter_model="google/gemini-2.5-flash")

    messages = [
        {"role": "system", "content": "Stable CAD instructions."},
        {"role": "user", "content": "<project_state>dynamic</project_state>"},
        {"role": "user", "content": "Build a bracket."},
    ]
    OpenRouterClient(settings).chat(messages)

    system = captured["json"]["messages"][0]
    assert system == {
        "role": "system",
        "content": [{
            "type": "text",
            "text": "Stable CAD instructions.",
            "cache_control": {"type": "ephemeral"},
        }],
    }
    assert captured["json"]["messages"][1] == {
        "role": "user",
        "content": [{
            "type": "text",
            "text": "<project_state>dynamic</project_state>",
            "cache_control": {"type": "ephemeral"},
        }],
    }
    assert captured["json"]["messages"][2] == {
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Build a bracket.",
            "cache_control": {"type": "ephemeral"},
        }],
    }
    assert messages[-1]["content"] == "Build a bracket."

    client = OpenRouterClient(settings)
    first_wire = client._build_payload(messages, None)["messages"]
    extended = messages + [
        {"role": "assistant", "content": "I will inspect it."},
        {"role": "user", "content": "Continue."},
    ]
    second_wire = client._build_payload(extended, None)["messages"]
    assert second_wire[:len(first_wire)] == first_wire


def test_openrouter_caches_through_tool_results_for_gemini(tmp_path):
    settings = _settings(tmp_path, "openrouter", openrouter_model="google/gemini-2.5-flash")
    payload = OpenRouterClient(settings)._build_payload(
        [
            {"role": "system", "content": "Stable CAD instructions."},
            {"role": "user", "content": "Build a bracket."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "write_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Wrote model.py."},
        ],
        TOOL_SCHEMAS[:1],
    )

    assert payload["messages"][1]["content"] == [{
        "type": "text",
        "text": "Build a bracket.",
        "cache_control": {"type": "ephemeral"},
    }]
    assert payload["messages"][3]["content"] == [{
        "type": "text",
        "text": "Wrote model.py.",
        "cache_control": {"type": "ephemeral"},
    }]


def test_openrouter_honors_retry_after(monkeypatch, tmp_path):
    calls: list[int] = []
    delays: list[float] = []

    def post(*_a, **_kw):
        calls.append(1)
        return _RateLimitedResponse() if len(calls) == 1 else _FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.llm_base.requests.post", post)
    monkeypatch.setattr("agent.llm_base.sleep_with_cancel", lambda delay, _event: delays.append(delay))

    response = OpenRouterClient(_settings(tmp_path, "openrouter")).chat([{"role": "user", "content": "hi"}])

    assert response == {"choices": []}
    assert len(calls) == 2
    assert delays == [0.0]


def test_openrouter_sends_reasoning_and_forced_provider_preferences(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _FakeResponse)
    settings = _settings(
        tmp_path,
        "openrouter",
        openrouter_model="openai/o4-mini",
        openrouter_reasoning_effort="high",
        openrouter_provider="openai",
        openrouter_force_provider=True,
    )

    OpenRouterClient(settings).chat([{"role": "user", "content": "hi"}])

    payload = captured["json"]
    assert payload["model"] == "openai/o4-mini"
    assert payload["reasoning"] == {"effort": "high", "exclude": False}
    assert payload["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_supported_providers_routing_is_consistent_with_settings(tmp_path: Path):
    """Every provider in ``LLM_PROVIDERS`` must be accepted by ``load_settings``
    and surface through ``Settings.llm_provider`` so the agent loop can route
    to the right adapter.
    """
    for provider in LLM_PROVIDERS:
        # Arrange / Act: load_settings accepts the provider string.
        (tmp_path / "config.yaml").write_text(
            f"llm:\n  provider: {provider}\n", encoding="utf-8"
        )
        settings = load_settings(project_root=tmp_path)

        # Assert: the active provider is what the config asked for.
        assert settings.llm_provider == provider
        # Assert: llm_model resolves to a non-empty string (the property must
        # always be populated regardless of provider).
        assert settings.llm_model
        assert isinstance(settings.llm_model, str)


# ---------------------------------------------------------------------------
# Activity log integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", _CLIENT_KINDS)
def test_llm_request_log_records_successful_payload_after_image_fallback(
    monkeypatch, tmp_path, kind
):
    """Cache diagnostics record the successful retry, not a rejected body."""
    from agent.activity_log import ActivityLogger

    monkeypatch.setenv("OPENAI_API_KEY" if kind == "openai" else "OPENROUTER_API_KEY", "test")
    payloads: list[dict[str, Any]] = []

    def post(*_a, **kwargs):
        payloads.append(kwargs["json"])
        return _ImageRejectedResponse() if len(payloads) == 1 else _StreamingResponse()

    _patch_post_for(monkeypatch, kind, post)
    settings = _settings(
        tmp_path, kind,
        openrouter_model="text-only-model",
        openai_model="text-only-model",
    )
    logger = ActivityLogger(tmp_path)
    client = _client_for(kind, settings)
    client._activity_logger = logger
    client._run_id = "run-snapshot"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review this render."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }]

    client.chat(messages)

    request_entries = [
        entry for entry in logger.tail(limit=10) if entry["event"] == "llm_request"
    ]
    assert len(request_entries) == 1
    logged_payload = request_entries[0]["data"]["payload"]
    logged_content = logged_payload["messages"][0]["content"]
    assert isinstance(logged_content, list)
    assert {"type": "text", "text": "Review this render."} in logged_content
    image_parts = [part for part in logged_content if part.get("type") == "image_url"]
    assert not image_parts
    assert payloads[1]["messages"][0]["content"] == [
        {"type": "text", "text": "Review this render."}
    ]


def test_wire_history_is_append_only_after_completed_tools(tmp_path):
    from agent.llm_base import sanitize_messages

    read_result = json.dumps({
        "ok": True,
        "tool": "read_file",
        "data": json.dumps({
            "exists": True,
            "content": "result = Box(10, 20, 30)\n",
            "sha256": "a" * 64,
        }),
    })
    first = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "read-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"filename":"model.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "read-1", "content": read_result},
    ]
    second = first + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "edit-1",
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "arguments": json.dumps({
                        "filename": "model.py",
                        "old_string": "Box(10, 20, 30)",
                        "new_string": "Box(20, 20, 30)",
                    }),
                },
            }],
        },
        {"role": "tool", "tool_call_id": "edit-1", "content": "edited"},
    ]

    first_wire = sanitize_messages(first)
    second_wire = sanitize_messages(second)

    assert second_wire[:len(first_wire)] == first_wire
    assert "result = Box" in first_wire[-1]["content"]


def test_openai_56_marks_stable_system_and_namespaces_cache_key(tmp_path):
    client = OpenAIClient(_settings(tmp_path, "openai", openai_model="gpt-5.6-terra"))
    client.session_id = "openai:demo"

    payload = client._build_payload([
        {"role": "system", "content": "Stable CAD instructions."},
        {"role": "user", "content": "Build a bracket."},
    ], None)

    assert payload["prompt_cache_key"] == get_prompt_cache_key("openai:demo")
    assert payload["messages"][0]["content"] == [{
        "type": "text",
        "text": "Stable CAD instructions.",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }]


def test_anthropic_vertex_uses_portable_explicit_cache_marker(tmp_path):
    settings = _settings(
        tmp_path,
        "openrouter",
        openrouter_model="anthropic/claude-sonnet-4",
        openrouter_provider="google-vertex/global",
        openrouter_force_provider=True,
    )
    payload = OpenRouterClient(settings)._build_payload([
        {"role": "system", "content": "Stable CAD instructions."},
        {"role": "user", "content": "Build a bracket."},
    ], None)

    assert "cache_control" not in payload
    assert payload["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
