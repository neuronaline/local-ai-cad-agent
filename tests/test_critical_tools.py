from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent import dispatcher as dispatcher_module
from agent.dispatcher import process_tool_call
from agent.llm_base import (
    FallbackChatClient,
    RequestCancelled,
    StreamResponseError,
    parse_chat_stream,
)
from agent.revisions import RevisionStore, compute_model_sha256
from agent.tools.file_tool import FileTool


def test_model_write_creates_a_revision_and_rejects_unsafe_code(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)

    tool.write_file("model.py", "from build123d import Box\nresult = Box(10, 20, 30)\n")

    assert RevisionStore(tmp_path).head() is not None
    with pytest.raises(ValueError, match="Unsafe import blocked"):
        tool.write_file("model.py", "import subprocess\n")
    assert "result = Box" in (tmp_path / "model.py").read_text(encoding="utf-8")


def _seed_metrics(project_dir: Path) -> str:
    """Write a ``.cad_metrics.json`` whose ``model_sha256`` matches model.py.

    Returns the recorded sha so tests can reuse it for assertions.
    """
    model_sha = compute_model_sha256(project_dir)
    assert model_sha is not None
    metrics = {
        "model_sha256": model_sha,
        "preview_sha256": "",
        "metrics": {
            "solid_count": 1,
            "is_valid": True,
            "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
            "volume_mm3": 1.0,
        },
    }
    (project_dir / ".cad_metrics.json").write_text(
        __import__("json").dumps(metrics), encoding="utf-8"
    )
    return model_sha


def _make_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"{name}-call",
        "function": {"name": name, "arguments": __import__("json").dumps(arguments)},
    }


def test_dispatcher_allows_visual_tool_after_metrics_only_build(tmp_path: Path) -> None:
    """``cad_screenshot`` must NOT be blocked after a cheap ``render=false`` build.

    Regression: the dispatcher previously keyed the screenshot/review gate on
    ``cad_fix_required``, which was only cleared on a *rendered* build. That
    forced the agent to do a wasteful second rendered build whenever it
    iterated with ``render=false`` first. The gate now consults
    ``model_is_built(project_dir)`` directly, so any successful build — even
    a metrics-only one — unblocks visual tools.
    """
    FileTool(tmp_path).write_file(
        "model.py",
        "from build123d import Box\nresult = Box(10, 20, 30)\n",
    )
    _seed_metrics(tmp_path)
    # The loop passes ``cad_fix_required=True`` after a metrics-only build
    # because the loop's final-rendered-verification gate stays armed; the
    # dispatcher must ignore that flag for the screenshot gate.
    dispatched: list[tuple[str, dict]] = []

    def fake_dispatch(tools, project, name, args, call_id):
        dispatched.append((name, args))
        return {"summary": "skipped"}, False

    original = dispatcher_module.dispatch
    dispatcher_module.dispatch = fake_dispatch
    try:
        _preview_id, cad_error, cad_fix_required, waiting = process_tool_call(
            tools=MagicMock(),
            project="probe",
            project_dir=tmp_path,
            call=_make_call("cad_screenshot", {}),
            cad_fix_required=True,
            prev_preview_id=None,
            cad_error=None,
            messages=[],
            publish=lambda *_args, **_kwargs: None,
            register_preview=lambda *_args, **_kwargs: "preview-id",
            append_message=lambda *_args, **_kwargs: None,
        )
    finally:
        dispatcher_module.dispatch = original

    assert dispatched == [("cad_screenshot", {})], (
        "Dispatcher gate rejected cad_screenshot after a metrics-only build; "
        "the gate must consult model_is_built(project_dir), not cad_fix_required."
    )
    assert cad_fix_required is True, (
        "Loop-side cad_fix_required must keep its rendered-verification "
        "semantics; only the dispatcher gate was relaxed."
    )
    assert cad_error is None
    assert waiting is False


def test_dispatcher_still_blocks_visual_tool_without_a_build(tmp_path: Path) -> None:
    """No ``.cad_metrics.json`` for the current model must keep the gate armed."""
    FileTool(tmp_path).write_file(
        "model.py",
        "from build123d import Box\nresult = Box(10, 20, 30)\n",
    )
    # Note: no _seed_metrics() call — the project has a model.py but no
    # matching .cad_metrics.json, simulating an unverified edit.
    appended: list[dict] = []

    def capture_append(_project_dir: Path, payload: dict) -> None:
        appended.append(payload)

    preview_id, cad_error, cad_fix_required, waiting = process_tool_call(
        tools=MagicMock(),
        project="probe",
        project_dir=tmp_path,
        call=_make_call("cad_screenshot", {}),
        cad_fix_required=False,
        prev_preview_id=None,
        cad_error=None,
        messages=[],
        publish=lambda *_args, **_kwargs: None,
        register_preview=lambda *_args, **_kwargs: "preview-id",
        append_message=capture_append,
    )
    # ``process_tool_call`` converts tool exceptions into a failure envelope
    # rather than re-raising. For non-cad_build tools the failure lands in the
    # appended tool message; ``cad_error`` stays None because the loop's own
    # terminal-error machinery is gated on ``is_cad_build``.
    assert len(appended) == 1
    assert "requires a successful build" in appended[0]["content"]
    assert cad_fix_required is False  # loop gate stays as the loop set it
    assert cad_error is None
    assert waiting is False
    assert preview_id is None


class _FakeStreamResponse:
    """Minimal stand-in for ``requests.Response.iter_lines`` semantics."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def iter_lines(self):
        yield from self._lines

    def close(self) -> None:
        pass


def test_parse_chat_stream_marks_truncated_drop_retryable_without_partial_state() -> None:
    """A connection drop before ``[DONE]`` with no streamed payload must retry.

    Regression: when OpenRouter's stream ended mid-flight the parser used to
    raise a plain ``RuntimeError`` that bypassed the LLM client's retry loop,
    which then crashed the agent run mid-tool-loop even though the user did
    not press stop. Now it surfaces as ``StreamResponseError(retryable=True)``
    so the existing retry path kicks in transparently.
    """
    response = _FakeStreamResponse(
        [
            b"data: "
            + b'{"choices":[{"delta":{"role":"assistant"},"index":0}]}',
            # Connection drops here: no ``[DONE]``, no content yet.
        ]
    )
    with pytest.raises(StreamResponseError) as excinfo:
        parse_chat_stream(
            response,
            provider_label="openrouter",
            stop_event=None,
            stream_callback=None,
        )
    assert excinfo.value.retryable is True
    assert "stream ended before the completion marker" in str(excinfo.value)


def test_parse_chat_stream_marks_truncated_drop_non_retryable_with_partial_state() -> None:
    """Partial streamed content must NOT trigger a retry that duplicates tool calls."""
    response = _FakeStreamResponse(
        [
            b"data: "
            + b'{"choices":[{"delta":{"role":"assistant","content":"partial "},"index":0}]}',
            # Connection drops mid-message; the parser already accumulated
            # ``content="partial "``. Retrying would discard this evidence and
            # potentially replay a partially-built tool call.
        ]
    )
    with pytest.raises(StreamResponseError) as excinfo:
        parse_chat_stream(
            response,
            provider_label="openrouter",
            stop_event=None,
            stream_callback=None,
        )
    assert excinfo.value.retryable is False
    assert "partial response preserved" in str(excinfo.value)


class _StubChatClient:
    """Minimal stand-in for ``ChatCompletionsClient``.

    The fallback wrapper only inspects the public attributes the agent runner
    mutates between iterations; recording every assignment keeps the test
    honest about which surface area the wrapper depends on.
    """

    # Mirrors ``ChatCompletionsClient.preserve_reasoning``; tests that need
    # a specific value override it after construction.
    preserve_reasoning: bool = False

    def __init__(
        self,
        chat_result=None,
        chat_error=None,
        *,
        usage=None,
        image_fallback=False,
    ) -> None:
        self.chat_calls: list[tuple] = []
        self._chat_result = chat_result
        self._chat_error = chat_error
        self.last_usage = usage
        self.last_image_fallback_used = image_fallback
        # Mirrors the base-class defaults the agent runner sets.
        self.stop_event = None
        self.session_id = None
        self.agent_role = None
        self.stream_callback = None
        self.require_images = False
        self.activity_logger = None
        self.run_id = None

    def chat(self, messages, tools=None):
        self.chat_calls.append((messages, tools))
        if self._chat_error is not None:
            raise self._chat_error
        return self._chat_result


def test_fallback_wrapper_succeeds_via_primary() -> None:
    """A healthy primary must not trigger the fallback path."""
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    result = wrapper.chat([{"role": "user", "content": "hi"}], tools=[])

    assert result == {"choices": [{"message": {}}]}
    assert len(primary.chat_calls) == 1
    assert fallback.chat_calls == []
    assert wrapper.last_usage is None


def test_fallback_wrapper_recovers_via_fallback_on_primary_error() -> None:
    """A non-cancellation primary failure must trigger exactly one fallback attempt."""
    primary = _StubChatClient(
        chat_error=RuntimeError("openrouter: rate limit reached"),
    )
    fallback = _StubChatClient(
        chat_result={"choices": [{"message": {"content": "fallback ok"}}]},
        usage={"prompt_tokens": 12, "completion_tokens": 4},
    )
    wrapper = FallbackChatClient(primary, fallback, "openai")

    result = wrapper.chat([{"role": "user", "content": "hi"}])

    assert result == {"choices": [{"message": {"content": "fallback ok"}}]}
    assert len(primary.chat_calls) == 1
    assert len(fallback.chat_calls) == 1
    assert wrapper.last_usage == {"prompt_tokens": 12, "completion_tokens": 4}
    # The fallback client's metrics, not the primary's, must surface on the
    # wrapper so ``AgentRunner._publish_usage`` reports what actually answered.
    assert fallback.last_image_fallback_used is False


def test_fallback_wrapper_propagates_cancellation_without_falling_back() -> None:
    """User cancellation must short-circuit before the fallback runs."""
    primary = _StubChatClient(chat_error=RequestCancelled("stop"))
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    with pytest.raises(RequestCancelled):
        wrapper.chat([{"role": "user", "content": "hi"}])
    assert fallback.chat_calls == []


def test_fallback_wrapper_propagates_late_cancellation_from_primary_error() -> None:
    """A primary error raised while ``stop_event`` is set must not fall back."""
    primary = _StubChatClient(chat_error=RuntimeError("network down"))
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")
    stop_event = MagicMock()
    stop_event.is_set.return_value = True
    wrapper.stop_event = stop_event

    with pytest.raises(RuntimeError, match="network down"):
        wrapper.chat([{"role": "user", "content": "hi"}])
    assert fallback.chat_calls == []


def test_fallback_wrapper_surfaces_fallback_error_with_primary_cause() -> None:
    """When both providers fail, the fallback error wins and the primary is the cause."""
    primary = _StubChatClient(
        chat_error=StreamResponseError("primary stream failed", retryable=False),
    )
    fallback = _StubChatClient(
        chat_error=RuntimeError("openai: invalid api key"),
    )
    wrapper = FallbackChatClient(primary, fallback, "openai")

    with pytest.raises(RuntimeError, match="invalid api key") as excinfo:
        wrapper.chat([{"role": "user", "content": "hi"}])
    assert isinstance(excinfo.value.__cause__, StreamResponseError)
    assert len(primary.chat_calls) == 1
    assert len(fallback.chat_calls) == 1


def test_fallback_wrapper_mirrors_state_onto_inner_clients() -> None:
    """Per-call attributes set on the wrapper must reach both inner clients.

    ``AgentRunner`` mutates ``stream_callback`` each iteration; if the wrapper
    forgot to forward it the inner clients would stream straight to the
    previous callback, breaking the live SSE feed.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")
    wrapper.session_id = "session-x"
    callback = lambda *_args, **_kwargs: None

    wrapper.chat([])  # initial value: stream_callback=None
    wrapper.stream_callback = callback
    wrapper.chat([{"role": "user", "content": "hi"}])

    assert primary.session_id == "session-x"
    assert fallback.session_id == "session-x"
    assert primary.stream_callback is callback
    assert fallback.stream_callback is callback


def test_fallback_wrapper_propagates_preserve_reasoning_from_successful_client() -> None:
    """The wrapper must surface the inner client's ``preserve_reasoning``.

    ``AgentRunner`` reads ``client.preserve_reasoning`` after ``chat()``
    returns to decide whether to strip reasoning from the response.
    ``OpenRouterClient`` requires ``True`` on tool-call turns so the
    provider can continue the interrupted response; if the wrapper hides
    that flag the runner silently drops reasoning from a successful
    OpenRouter answer and breaks the multi-turn tool-call flow.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    primary.preserve_reasoning = True
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback.preserve_reasoning = False
    wrapper = FallbackChatClient(primary, fallback, "openai")

    # Class default is ``False``; this is the regression guard.
    assert wrapper.preserve_reasoning is False

    wrapper.chat([{"role": "user", "content": "hi"}])
    assert wrapper.preserve_reasoning is True


def test_fallback_wrapper_falls_back_to_fallback_preserve_reasoning() -> None:
    """When the primary errors out, the fallback's flag wins."""
    primary = _StubChatClient(
        chat_error=RuntimeError("primary offline"),
    )
    primary.preserve_reasoning = True
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback.preserve_reasoning = False
    wrapper = FallbackChatClient(primary, fallback, "openai")

    wrapper.chat([{"role": "user", "content": "hi"}])
    assert wrapper.preserve_reasoning is False


def test_file_tool_write_reports_line_count_not_character_count(tmp_path: Path) -> None:
    """The write confirmation surfaces line count first; character count stays secondary."""
    tool = FileTool(tmp_path)

    message = tool.write_file(
        "model.py",
        "from build123d import Box\nresult = Box(10, 20, 30)\n",
    )

    assert "lines" in message
    assert "chars" in message
    # Two newline-terminated lines (1 import + 1 result) — splitlines() drops
    # the trailing empty entry so a 2-line file always reports "2 lines".
    assert "2 lines" in message
    assert "characters" not in message
