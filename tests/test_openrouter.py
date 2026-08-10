import hashlib
from typing import ClassVar

import pytest
import requests

from agent.openrouter import OpenRouterClient
from agent.settings import Settings


class Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": []}


class ErrorResponse(Response):
    status_code = 400

    def raise_for_status(self):
        raise requests.HTTPError("bad request")


class ImageRejectedResponse(ErrorResponse):
    status_code = 404

    def close(self):
        return None


class RateLimitedResponse(Response):
    status_code = 429
    headers: ClassVar[dict[str, str]] = {"Retry-After": "0"}

    def raise_for_status(self):
        raise requests.HTTPError("rate limited")


class StreamingResponse(Response):
    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning":"Checking geometry. "}}]}',
                b'data: {"choices":[{"delta":{"role":"assistant","content":"Building "}}]}',
                b'data: {"choices":[{"delta":{"content":"now.","tool_calls":[{"index":0,"id":"call-1","function":{"name":"cad","arguments":"{\\"operation\\":"}}]}}]}',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"run\\"}"}}]}}]}',
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5}}',
                b"data: [DONE]",
            ]
        )


class MidStreamErrorResponse(Response):
    def iter_lines(self):
        return iter([
            b'data: {"error":{"message":"provider failed"},"choices":[{"finish_reason":"error","delta":{}}]}',
            b"data: [DONE]",
        ])


class TruncatedStreamingResponse(Response):
    def iter_lines(self):
        return iter([b'data: {"choices":[{"delta":{"content":"partial"}}]}'])


def test_openrouter_streams_content_and_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.openrouter.requests.post",
        lambda *_args, **_kwargs: StreamingResponse(),
    )
    settings = Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000)
    client = OpenRouterClient(settings)
    events = []
    client.stream_callback = events.append

    response = client.chat([{"role": "user", "content": "build"}])

    message = response["choices"][0]["message"]
    assert message["content"] == "Building now."
    assert message["tool_calls"][0]["function"] == {
        "name": "cad",
        "arguments": '{"operation":"run"}',
    }
    assert [event["type"] for event in events] == [
        "reasoning",
        "content",
        "content",
        "tool_call",
        "tool_call",
    ]
    assert events[0]["delta"] == "Checking geometry. "
    assert client.last_usage["completion_tokens"] == 5


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (MidStreamErrorResponse(), "provider failed"),
        (TruncatedStreamingResponse(), "completion marker"),
    ],
)
def test_openrouter_rejects_failed_or_truncated_stream(monkeypatch, tmp_path, response, message):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.openrouter.requests.post", lambda *_args, **_kwargs: response)
    settings = Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000)

    with pytest.raises(RuntimeError, match=message):
        OpenRouterClient(settings).chat([{"role": "user", "content": "build"}])


def test_openrouter_retries_without_images_when_provider_rejects_vision(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    payloads = []

    def post(*_args, **kwargs):
        payloads.append(kwargs["json"])
        return ImageRejectedResponse() if len(payloads) == 1 else StreamingResponse()

    monkeypatch.setattr("agent.openrouter.requests.post", post)
    settings = Settings(tmp_path, "https://example.test", "text-only-model", 1, "127.0.0.1", 5000)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review this render."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }]

    response = OpenRouterClient(settings).chat(messages)

    assert response["choices"][0]["message"]["content"] == "Building now."
    assert len(payloads) == 2
    assert payloads[1]["messages"][0]["content"] == [
        {"type": "text", "text": "Review this render."},
    ]


def test_openrouter_retries_transient_failure(monkeypatch, tmp_path):
    calls = []

    def post(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary")
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.openrouter.requests.post", post)
    monkeypatch.setattr("agent.llm_base.sleep_with_cancel", lambda _delay, _event: None)
    settings = Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000)
    assert OpenRouterClient(settings).chat([{"role": "user", "content": "hi"}]) == {"choices": []}
    assert len(calls) == 2


def test_openrouter_does_not_retry_non_transient_client_error(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.openrouter.requests.post",
        lambda *_args, **_kwargs: calls.append(1) or ErrorResponse(),
    )
    settings = Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000)

    with pytest.raises(requests.HTTPError):
        OpenRouterClient(settings).chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 1


def test_openrouter_builds_cache_safe_sticky_payload_and_keeps_tool_content(monkeypatch, tmp_path):
    captured = {}

    def post(*_args, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.openrouter.requests.post", post)
    settings = Settings(
        tmp_path,
        "https://example.test",
        "anthropic/claude-sonnet-4",
        1,
        "127.0.0.1",
        5000,
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

    payload = captured["json"]
    assistant = payload["messages"][1]
    assert assistant["content"] == "I will inspect it."
    assert "tool_calls" in assistant
    assert "reasoning" not in assistant and "reasoning_details" not in assistant
    assert "_provider_debug" not in assistant
    assert payload["session_id"] == hashlib.sha256(b"project:demo").hexdigest()[:64]
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert captured["headers"]["X-OpenRouter-Title"] == "CAD Test"
    assert captured["headers"]["HTTP-Referer"] == "https://cad.example"


def test_openrouter_marks_stable_system_prompt_cacheable_for_gemini(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.openrouter.requests.post",
        lambda *_args, **kwargs: captured.update(kwargs) or Response(),
    )
    settings = Settings(
        tmp_path,
        "https://example.test",
        "google/gemini-2.5-flash",
        1,
        "127.0.0.1",
        5000,
    )

    OpenRouterClient(settings).chat([
        {"role": "system", "content": "Stable CAD instructions."},
        {"role": "user", "content": "Build a bracket."},
    ])

    system = captured["json"]["messages"][0]
    assert system["content"] == [{
        "type": "text",
        "text": "Stable CAD instructions.",
        "cache_control": {"type": "ephemeral"},
    }]


def test_openrouter_honors_retry_after(monkeypatch, tmp_path):
    calls = []
    delays = []

    def post(*_args, **_kwargs):
        calls.append(1)
        return RateLimitedResponse() if len(calls) == 1 else Response()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr("agent.openrouter.requests.post", post)
    # The retry path uses ``sleep_with_cancel``; patch it so the assertion can
    # observe the requested delay without actually sleeping.
    monkeypatch.setattr("agent.llm_base.sleep_with_cancel", lambda delay, _event: delays.append(delay))
    settings = Settings(tmp_path, "https://example.test", "test", 1, "127.0.0.1", 5000)

    assert OpenRouterClient(settings).chat([{"role": "user", "content": "hi"}]) == {"choices": []}
    assert len(calls) == 2
    assert delays == [0.0]


def test_openrouter_sends_reasoning_and_forced_provider_preferences(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(
        "agent.openrouter.requests.post",
        lambda *_args, **kwargs: captured.update(kwargs) or Response(),
    )
    settings = Settings(
        tmp_path,
        "https://example.test",
        "openai/o4-mini",
        1,
        "127.0.0.1",
        5000,
        openrouter_reasoning_effort="high",
        openrouter_provider="openai",
        openrouter_force_provider=True,
    )

    OpenRouterClient(settings).chat([{"role": "user", "content": "hi"}])

    assert captured["json"]["model"] == "openai/o4-mini"
    assert captured["json"]["reasoning"] == {"effort": "high", "exclude": False}
    assert captured["json"]["provider"] == {
        "order": ["openai"],
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
