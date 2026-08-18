"""Tests for the OpenAI-compatible chat completions adapters (OpenRouter and OpenAI)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar

import pytest
import requests

from agent.llm_base import create_llm_client
from agent.openai_client import OpenAIClient
from agent.openrouter import OpenRouterClient
from agent.prompt import get_prompt_cache_key
from agent.settings import LLM_PROVIDERS, Settings, load_settings


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


class _ClientErrorResponse(_FakeResponse):
    status_code: int = 400

    def raise_for_status(self) -> None:
        raise requests.HTTPError("bad request")


class _ImageRejectedResponse(_FakeResponse):
    status_code: int = 404

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
    base = dict(
        workspace_root=tmp_path,
        openrouter_base_url="https://example.test",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_timeout_seconds=1,
        host="127.0.0.1",
        port=5000,
        openai_base_url="https://example.test",
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=1,
    )
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
            "reasoning": "internal",
            "reasoning_details": {"x": 1},
            "_provider_debug": True,
        },
    ])

    assistant = captured["json"]["messages"][1]
    assert assistant["content"] == "I will inspect it."
    assert "tool_calls" in assistant
    assert "reasoning" not in assistant and "reasoning_details" not in assistant
    assert "_provider_debug" not in assistant
    assert captured["json"]["session_id"] == hashlib.sha256(b"project:demo").hexdigest()[:64]
    assert captured["json"]["cache_control"] == {"type": "ephemeral"}
    assert captured["headers"]["X-OpenRouter-Title"] == "CAD Test"
    assert captured["headers"]["HTTP-Referer"] == "https://cad.example"


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


def test_openrouter_marks_stable_system_prompt_cacheable_for_gemini(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _capture_post(monkeypatch, "agent.llm_base.requests.post", captured, _FakeResponse)
    settings = _settings(tmp_path, "openrouter", openrouter_model="google/gemini-2.5-flash")

    OpenRouterClient(settings).chat([
        {"role": "system", "content": "Stable CAD instructions."},
        {"role": "user", "content": "<project_state>dynamic</project_state>"},
        {"role": "user", "content": "Build a bracket."},
    ])

    system = captured["json"]["messages"][0]
    assert system["content"] == [{
        "type": "text",
        "text": "Stable CAD instructions.",
        "cache_control": {"type": "ephemeral"},
    }]
    assert captured["json"]["messages"][1] == {
        "role": "user",
        "content": "<project_state>dynamic</project_state>",
    }


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
