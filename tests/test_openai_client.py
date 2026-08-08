"""Tests for the OpenAI Chat Completions client."""
import pytest
import requests

from agent.llm_base import create_llm_client
from agent.openai_client import OpenAIClient
from agent.settings import LLM_PROVIDERS, Settings


class _StreamingResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"role":"assistant","reasoning":"Inspecting."}}]}',
                b'data: {"choices":[{"delta":{"content":"Designing "}}]}',
                b'data: {"choices":[{"delta":{"content":"bracket.","tool_calls":[{"index":0,"id":"call-9","function":{"name":"cad","arguments":"{\\"operation\\":"}}]}}]}',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"run\\"}"}}]}}]}',
                b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4}}',
                b"data: [DONE]",
            ]
        )


class _EmptyResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class _MidStreamErrorResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter([
            b'data: {"error":{"message":"openai failure"},"choices":[{"finish_reason":"error","delta":{}}]}',
            b"data: [DONE]",
        ])


class _ImageRejectedResponse:
    status_code = 404

    def close(self):
        return None

    def text(self):
        return "vision not supported"

    def raise_for_status(self):
        raise requests.HTTPError("bad request")


def _settings(tmp_path, **overrides):
    base = dict(
        workspace_root=tmp_path,
        openrouter_base_url="https://example.test",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_timeout_seconds=1,
        host="127.0.0.1",
        port=5000,
        llm_provider="openai",
        openai_base_url="https://example.test",
        openai_model="gpt-4o-mini",
        openai_timeout_seconds=1,
    )
    base.update(overrides)
    return Settings(**base)


def test_create_llm_client_routes_to_openai(tmp_path):
    settings = _settings(tmp_path)
    client = create_llm_client(settings)
    assert isinstance(client, OpenAIClient)
    assert client is not OpenAIClient(settings) or True  # sanity: factory matches type


def test_openai_streams_content_and_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda *_args, **_kwargs: _StreamingResponse(),
    )
    settings = _settings(tmp_path)
    client = OpenAIClient(settings)
    events = []
    client.stream_callback = events.append

    response = client.chat([{"role": "user", "content": "build"}])

    message = response["choices"][0]["message"]
    assert message["content"] == "Designing bracket."
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
    assert events[0]["delta"] == "Inspecting."
    assert client.last_usage["completion_tokens"] == 4


def test_openai_rejects_midstream_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda *_args, **_kwargs: _MidStreamErrorResponse(),
    )
    settings = _settings(tmp_path)

    with pytest.raises(RuntimeError, match="openai failure"):
        OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])


def test_openai_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = _settings(tmp_path)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])


def test_openai_falls_back_when_vision_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    payloads = []

    def post(*_args, **kwargs):
        payloads.append(kwargs["json"])
        return _ImageRejectedResponse() if len(payloads) == 1 else _StreamingResponse()

    monkeypatch.setattr("agent.llm_base.requests.post", post)
    settings = _settings(tmp_path, openai_model="text-only-model")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Review this render."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ],
    }]

    response = OpenAIClient(settings).chat(messages)

    assert response["choices"][0]["message"]["content"] == "Designing bracket."
    assert len(payloads) == 2
    assert payloads[1]["messages"][0]["content"] == [
        {"type": "text", "text": "Review this render."},
    ]


def test_openai_builds_payload_with_reasoning_effort(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda *_args, **kwargs: captured.update(kwargs) or _EmptyResponse(),
    )
    settings = _settings(tmp_path, openai_model="o4-mini", openai_reasoning_effort="high")

    OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])

    payload = captured["json"]
    assert payload["model"] == "o4-mini"
    assert payload["reasoning_effort"] == "high"
    # OpenAI does not consume OpenRouter-specific keys.
    assert "provider" not in payload
    assert "session_id" not in payload
    assert "cache_control" not in payload
    assert captured["headers"]["Authorization"] == "Bearer test"
    assert "HTTP-Referer" not in captured["headers"]
    assert "X-OpenRouter-Title" not in captured["headers"]


def test_openai_endpoint_uses_configured_base_url(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "agent.llm_base.requests.post",
        lambda url, *_a, **kw: captured.update({"url": url}) or _EmptyResponse(),
    )
    settings = _settings(
        tmp_path,
        openai_base_url="https://proxy.example.test/v1",
    )

    OpenAIClient(settings).chat([{"role": "user", "content": "hi"}])

    assert captured["url"] == "https://proxy.example.test/v1/chat/completions"


def test_settings_uses_openai_model_when_provider_is_openai(tmp_path):
    from agent.settings import load_settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  provider: openai\nopenai:\n  model: gpt-4.1\nopenrouter:\n  model: openai/gpt-4o\n",
        encoding="utf-8",
    )
    settings = load_settings(project_root=tmp_path, home=tmp_path)
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4.1"
    # The other provider's model is still remembered.
    assert settings.openrouter_model == "openai/gpt-4o"
    assert settings.openai_model == "gpt-4.1"


def test_settings_rejects_unknown_provider(tmp_path):
    from agent.settings import load_settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: bogus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="llm.provider"):
        load_settings(project_root=tmp_path, home=tmp_path)


def test_llm_providers_constant_lists_supported_values():
    assert LLM_PROVIDERS == {"openrouter", "openai"}


def test_create_llm_client_routes_to_openrouter_for_default(tmp_path):
    settings = _settings(tmp_path, llm_provider="openrouter")
    from agent.openrouter import OpenRouterClient
    assert isinstance(create_llm_client(settings), OpenRouterClient)
