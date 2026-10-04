"""Tests for Ollama client adapter, settings, and runtime resilience."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from agent.core import AgentRunner
from agent.llm_base import (
    _is_image_rejection,
    api_key_env,
    create_llm_client,
    parse_chat_stream,
    provider_label,
)
from agent.ollama_client import DEFAULT_BASE_URL, OllamaClient
from agent.settings import Settings, load_settings
from routes import _api_key_configured, _run_preflight


def _make_settings(
    workspace: Path,
    *,
    provider: str = "ollama",
    base_url: str = DEFAULT_BASE_URL,
    model: str = "qwen2.5-coder:14b-16k",
    timeout: int = 120,
) -> Settings:
    return Settings(
        workspace_root=workspace,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openrouter/test",
        openrouter_timeout_seconds=60,
        host="127.0.0.1",
        port=5000,
        llm_provider=provider,
        ollama_base_url=base_url,
        ollama_model=model,
        ollama_timeout_seconds=timeout,
        llm_max_completion_tokens=4096,
    )


def test_ollama_client_endpoint_and_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama client constructs correct endpoint and omits Authorization header when key is unset."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = _make_settings(tmp_path, base_url="http://192.168.1.50:11434/v1")
    client = OllamaClient(settings)

    assert client.requires_api_key is False
    assert client._endpoint() == "http://192.168.1.50:11434/v1/chat/completions"
    assert client._api_key() == ""

    headers = client._build_headers(client._api_key())
    assert headers == {"Content-Type": "application/json"}
    assert "Authorization" not in headers

    # Custom API key in environment
    monkeypatch.setenv("OLLAMA_API_KEY", "custom-secret-token")
    assert client._api_key() == "custom-secret-token"
    custom_headers = client._build_headers(client._api_key())
    assert custom_headers["Authorization"] == "Bearer custom-secret-token"


def test_ollama_client_chat_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OllamaClient.chat executes successfully without requiring any API key."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = _make_settings(tmp_path)
    client = OllamaClient(settings)

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = iter([
        b'data: {"choices":[{"delta":{"role":"assistant","content":"echo"}}]}',
        b"data: [DONE]",
    ])

    with patch.object(client, "_post", return_value=mock_resp):
        response = client.chat([{"role": "user", "content": "hello"}])
        assert response["choices"][0]["message"]["content"] == "echo"


def test_ollama_client_payload_construction(tmp_path: Path) -> None:
    """Payload uses max_tokens, drops reasoning fields, and includes tools."""
    settings = _make_settings(tmp_path, model="qwen2.5-coder:7b-16k")
    client = OllamaClient(settings)

    messages = [
        {"role": "system", "content": "You are a CAD designer."},
        {
            "role": "assistant",
            "content": "Designing model...",
            "reasoning": "Thinking step...",
            "reasoning_details": [{"type": "text", "text": "details"}],
        },
        {"role": "user", "content": "Build a cube."},
    ]
    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {}}}]

    payload = client._build_payload(messages, tools)

    assert payload["model"] == "qwen2.5-coder:7b-16k"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload
    assert "prompt_cache_key" not in payload
    assert "reasoning_effort" not in payload
    assert payload["tools"] == tools

    # Reasoning fields must be stripped from assistant messages
    assistant_wire = payload["messages"][1]
    assert "reasoning" not in assistant_wire
    assert "reasoning_details" not in assistant_wire
    assert assistant_wire["content"] == "Designing model..."


def test_ollama_stream_handles_tool_calls_without_id() -> None:
    """Local model streaming tool calls without explicit IDs receives synthetic ID."""
    sse_lines = [
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"{\\"content\\": \\"cube(10);\\"}"}}]}}]}',
        b'data: {"choices":[{"finish_reason":"tool_calls"}]}',
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":20}}',
        b"data: [DONE]",
    ]

    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.iter_lines.return_value = iter(sse_lines)

    result = parse_chat_stream(
        mock_resp,
        provider_label="Ollama",
        stop_event=None,
        stream_callback=None,
    )

    choice = result["choices"][0]["message"]
    assert len(choice["tool_calls"]) == 1
    call = choice["tool_calls"][0]
    assert call["id"] == "call_0"
    assert call["function"]["name"] == "write_file"
    assert json.loads(call["function"]["arguments"]) == {"content": "cube(10);"}


def test_ollama_image_rejection_detection() -> None:
    """_is_image_rejection catches Ollama's 'model does not support images' error."""
    resp_400 = MagicMock(spec=requests.Response)
    resp_400.status_code = 400
    resp_400.text = "error: model qwen2.5-coder does not support images"
    assert _is_image_rejection(resp_400) is True

    resp_500 = MagicMock(spec=requests.Response)
    resp_500.status_code = 500
    resp_500.text = "internal server error"
    assert _is_image_rejection(resp_500) is False


def test_settings_load_ollama_config(tmp_path: Path) -> None:
    """Settings loader reads ollama namespace and validates correctly."""
    config_data = {
        "workspace_root": str(tmp_path / "workspace"),
        "llm": {
            "provider": "ollama",
            "fallback_provider": "openrouter",
            "max_completion_tokens": 16384,
        },
        "ollama": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5-coder:32b",
            "timeout_seconds": 180,
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemini-2.5-flash",
        },
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    settings = load_settings(tmp_path)
    assert settings.llm_provider == "ollama"
    assert settings.llm_fallback_provider == "openrouter"
    assert settings.llm_model == "qwen2.5-coder:32b"
    assert settings.ollama_base_url == "http://127.0.0.1:11434/v1"
    assert settings.ollama_model == "qwen2.5-coder:32b"
    assert settings.ollama_timeout_seconds == 180
    assert settings.llm_max_completion_tokens == 16384

    # Unknown key under ollama must raise ValueError
    config_data["ollama"]["invalid_field"] = "bad"
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown ollama setting"):
        load_settings(tmp_path)


def test_ollama_user_error_mapping() -> None:
    """AgentRunner._user_error_message maps Ollama connection and model errors."""
    conn_msg = AgentRunner._user_error_message(
        "HTTPConnectionPool: Max retries exceeded with url: Failed to establish a new connection: [Errno 111] Connection refused",
        "ConnectionError",
        provider="ollama",
    )
    assert "Could not connect to Ollama server" in conn_msg
    assert "ollama serve" in conn_msg

    model_msg = AgentRunner._user_error_message(
        "model 'qwen2.5-coder:14b-16k' not found, try pulling it first",
        "RuntimeError",
        provider="ollama",
    )
    assert "Ollama model not found" in model_msg
    assert "ollama pull" in model_msg

    timeout_msg = AgentRunner._user_error_message(
        "Read timed out after 120 seconds",
        "Timeout",
        provider="ollama",
    )
    assert "Connection to Ollama server timed out" in timeout_msg

    auth_msg = AgentRunner._user_error_message(
        "401 Unauthorized",
        "HTTPError",
        provider="ollama",
    )
    assert "Authentication failed for Ollama" in auth_msg
    assert "OLLAMA_API_KEY" in auth_msg


def test_ollama_preflight_and_api_key_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When provider is ollama, preflight and api_key checks pass without env var."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = _make_settings(tmp_path, provider="ollama")

    assert _api_key_configured(settings) is True
    assert api_key_env("ollama") == "OLLAMA_API_KEY"
    assert provider_label("ollama") == "Ollama"

    preflight = _run_preflight(settings)
    assert preflight["api_key"] is True
    assert preflight["provider"] == "ollama"
    assert preflight["model_configured"] is True

    # Factory correctly creates OllamaClient
    client = create_llm_client(settings)
    assert isinstance(client, OllamaClient)


def test_ollama_live_tcp_socket_interaction(tmp_path: Path) -> None:
    """OllamaClient communicates over a real TCP socket with an OpenAI-compatible daemon."""
    import http.server
    import threading

    class OllamaHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            assert body["model"] == "qwen2.5-coder:14b-16k"
            assert body["max_tokens"] == 4096

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            events = [
                'data: {"choices":[{"delta":{"role":"assistant","content":"Generating "}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"CAD model."}}]}\n\n',
                'data: [DONE]\n\n',
            ]
            for ev in events:
                self.wfile.write(ev.encode("utf-8"))
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), OllamaHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        settings = _make_settings(tmp_path, base_url=f"http://127.0.0.1:{port}/v1")
        client = OllamaClient(settings)
        collected_deltas = []
        client.stream_callback = (
            lambda ev: collected_deltas.append(ev["delta"]) if ev.get("type") == "content" else None
        )
        response = client.chat([{"role": "user", "content": "Make a cylinder"}])
        assert response["choices"][0]["message"]["content"] == "Generating CAD model."
        assert "".join(collected_deltas) == "Generating CAD model."
    finally:
        server.server_close()
