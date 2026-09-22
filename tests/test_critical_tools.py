import hashlib
import io
import json
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent import tool_results
from agent.conversation import ConversationStore
from agent.core import AgentRunner
from agent.llm_base import (
    ChatCompletionsClient,
    FallbackChatClient,
    RequestCancelled,
    StreamResponseError,
    normalize_messages,
    parse_chat_stream,
    sanitize_assistant_message,
    sanitize_messages,
)
from agent.revisions import RevisionStore, compute_model_sha256
from agent.settings import Settings
from agent.tools.file_tool import FileTool


def test_model_write_creates_a_revision_and_rejects_unsafe_code(tmp_path: Path) -> None:
    tool = FileTool(tmp_path)

    tool.write_file("model.scad", "WIDTH = 10;\ncube([WIDTH, 20, 30]);\n")

    head = RevisionStore(tmp_path).head()
    assert head is not None
    assert len(head.id) == 36 and head.id.count("-") == 4
    assert head.model_sha256 == hashlib.sha256(b"WIDTH = 10;\ncube([WIDTH, 20, 30]);\n").hexdigest()
    assert head.parent_id is None
    assert head.origin.kind == "agent_edit"
    with pytest.raises(ValueError, match="Python keyword 'import' found"):
        tool.write_file("model.scad", "import subprocess\n")
    with pytest.raises(ValueError, match="Unclosed string literal"):
        tool.write_file("model.scad", 'WIDTH = 10;\nstr = "unclosed;\n')
    with pytest.raises(ValueError, match="Mismatched delimiter"):
        tool.write_file("model.scad", "WIDTH = 10;\ncube([WIDTH, 20, 30);\n")
    with pytest.raises(ValueError, match="Unclosed delimiter"):
        tool.write_file("model.scad", "WIDTH = 10;\ncube([WIDTH, 20, 30;\n")
    assert "cube([WIDTH, 20, 30])" in (tmp_path / "model.scad").read_text(encoding="utf-8")


def _seed_metrics(project_dir: Path) -> str:
    """Write a ``.cad_metrics.json`` whose ``model_sha256`` matches model.scad.

    Returns the recorded sha so tests can reuse it for assertions.
    """
    model_sha = compute_model_sha256(project_dir)
    assert model_sha is not None and len(model_sha) == 64 and all(c in "0123456789abcdef" for c in model_sha)
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


def test_question_validator_number_with_units() -> None:
    """QuestionValidator accepts whitespace-tolerant numbers and common length/angle units."""
    from agent.tools.question_validator import QuestionValidator

    q = {"id": "length", "question": "Enter length", "input_type": "number"}
    # Valid inputs without space, with space, decimals, and various units
    assert QuestionValidator.validate(q, "10")
    assert QuestionValidator.validate(q, "10mm")
    assert QuestionValidator.validate(q, "10 mm")
    assert QuestionValidator.validate(q, "10.5mm")
    assert QuestionValidator.validate(q, "10.5 cm")
    assert QuestionValidator.validate(q, "45 deg")
    assert QuestionValidator.validate(q, "45°")
    assert QuestionValidator.validate(q, "2 in")
    assert QuestionValidator.validate(q, "2inches")
    assert QuestionValidator.validate(q, "0.5")

    # Invalid inputs
    assert not QuestionValidator.validate(q, "")
    assert not QuestionValidator.validate(q, "abc")
    assert not QuestionValidator.validate(q, "10 xyz")
    assert not QuestionValidator.validate(q, "10 mm extra")


def test_parse_tool_arguments_deduplicates_keys() -> None:
    """Regression: streaming LLMs occasionally emit duplicate JSON keys.

    Python's stdlib ``json.loads`` silently keeps the LAST value when keys
    repeat, which in production corrupted ``model.scad`` — the LLM intended
    the OpenSCAD body but the dispatcher only saw a 6-char literal
    (``"FEMALE"``) or empty string (``""``). Strictly rejecting duplicate
    keys caused LLMs to enter an unrecoverable error loop. The parser now
    safely deduplicates keys by picking the non-empty, substantial payload.
    """
    from agent.dispatcher import _parse_tool_arguments

    # Baseline: clean JSON still parses.
    assert _parse_tool_arguments('{"content": "WIDTH = 10;"}') == {
        "content": "WIDTH = 10;"
    }
    assert _parse_tool_arguments("") == {}
    assert _parse_tool_arguments("   \n\t  ") == {}

    # Empty / non-object payloads surface a clear error.
    with pytest.raises(ValueError, match="Tool arguments must be a JSON object"):
        _parse_tool_arguments("[1, 2, 3]")
    with pytest.raises(ValueError, match="Expecting property name"):
        _parse_tool_arguments("{not json")

    # The failure mode from docs/temp_files/raw-api.json: a ``write_file``
    # call where the LLM emitted ``content`` twice and the second value
    # (a 6-char literal) was previously winning. It must keep the real code.
    assert _parse_tool_arguments(
        '{"content": "WIDTH = 65;\\nEPS = 0.01;\\n", "content": "FEMALE"}'
    ) == {"content": "WIDTH = 65;\nEPS = 0.01;\n"}

    # The trailing empty string failure mode from raw-api.json and raw-api-2.json.
    assert _parse_tool_arguments(
        '{"content": "WIDTH = 65;\\n", "content": "", "content": ""}'
    ) == {"content": "WIDTH = 65;\n"}

    # Reverse order: empty first, real content second.
    assert _parse_tool_arguments(
        '{"content": "", "content": "WIDTH = 65;\\n"}'
    ) == {"content": "WIDTH = 65;\n"}

    # Equal-length or other duplicate keys preserve the first emitted value.
    assert _parse_tool_arguments(
        '{"old_string": "a", "new_string": "b", "old_string": "c"}'
    ) == {"old_string": "a", "new_string": "b"}
    assert _parse_tool_arguments(
        '{"old_string": "a", "new_string": "b", "old_string": "ccc"}'
    ) == {"old_string": "a", "new_string": "b"}

    # Numeric 0 is a valid domain value and must not be treated as empty.
    assert _parse_tool_arguments('{"offset": 0, "offset": ""}') == {"offset": 0}
    assert _parse_tool_arguments('{"offset": "", "offset": 0}') == {"offset": 0}

    # Structural trailing commas in JSON are stripped, while trailing commas
    # inside string literals (e.g. OpenSCAD code) are preserved verbatim.
    assert _parse_tool_arguments(
        '{"content": "points = [[0, 0], [10, 0], ];", }'
    ) == {"content": "points = [[0, 0], [10, 0], ];"}

    # Nested duplicate keys inside an array of edits must also resolve cleanly.
    assert _parse_tool_arguments(
        '{"edits": [{"filename": "model.scad", "filename": "model.scad"}]}'
    ) == {"edits": [{"filename": "model.scad"}]}


def test_process_tool_call_handles_duplicate_key_payload(tmp_path: Path) -> None:
    """End-to-end: a ``write_file`` call with trailing duplicate keys preserves
    the intended OpenSCAD code and successfully writes it to ``model.scad``.

    Drives ``process_tool_call`` directly with the exact JSON shape that
    broke the user's session in docs/temp_files/raw-api.json, asserting
    that ``model.scad`` is written with the real code (not "FEMALE").
    """
    from agent.dispatcher import process_tool_call

    captured: list[dict] = []
    appended: list[dict] = []
    messages: list[dict] = []
    mock_file_tool = MagicMock()
    mock_file_tool.with_call_id.return_value = mock_file_tool
    mock_file_tool.write_file.return_value = "Wrote model.scad"
    mock_tools = MagicMock()
    mock_tools.file = mock_file_tool

    preview_id, cad_error, fix_required, waiting = process_tool_call(
        tools=mock_tools,
        project="demo",
        project_dir=tmp_path,
        call={
            "id": "call_dup",
            "function": {
                "name": "write_file",
                "arguments": '{"content": "WIDTH = 65;\\n", "content": "FEMALE"}',
            },
        },
        cad_fix_required=False,
        prev_preview_id=None,
        cad_error=None,
        messages=messages,
        publish=lambda event, payload: captured.append({"event": event, **payload}),
        register_preview=lambda *_: "preview-id",
        append_message=lambda _dir, msg: appended.append(msg),
    )
    # The tool call should succeed with the deduplicated intended content
    assert fix_required is True
    assert waiting is False
    mock_file_tool.write_file.assert_called_once_with("model.scad", "WIDTH = 65;\n")
    status_events = [event for event in captured if event["event"] == "tool_status"]
    assert status_events[-1]["status"] == "completed"
    assert appended and appended[-1]["role"] == "tool"
    assert appended[-1]["tool_call_id"] == "call_dup"


def test_edit_file_dispatch_strips_code_fences() -> None:
    """Regression: LLMs wrapping new_string in markdown code blocks must be stripped."""
    from agent.dispatcher import _dispatch_edit_file

    mock_file_tool = MagicMock()
    mock_file_tool.with_call_id.return_value = mock_file_tool

    # Single edit with code fence
    _dispatch_edit_file(
        mock_file_tool,
        {
            "old_string": "WIDTH = 50;",
            "new_string": "```scad\nWIDTH = 80;\n```",
        },
        "call_1",
    )
    mock_file_tool.edit_file.assert_called_once_with(
        "model.scad", "WIDTH = 50;", "WIDTH = 80;"
    )

    # Batch edits with code fence
    _dispatch_edit_file(
        mock_file_tool,
        {
            "edits": [
                {
                    "old_string": "HEIGHT = 20;",
                    "new_string": "```\nHEIGHT = 40;\n```",
                }
            ]
        },
        "call_2",
    )
    mock_file_tool.edit_file.assert_called_with(
        "model.scad", "HEIGHT = 20;", "HEIGHT = 40;"
    )

    # Single-line code fence
    _dispatch_edit_file(
        mock_file_tool,
        {
            "old_string": "DEPTH = 10;",
            "new_string": "```scad DEPTH = 15; ```",
        },
        "call_3",
    )
    mock_file_tool.edit_file.assert_called_with(
        "model.scad", "DEPTH = 10;", "DEPTH = 15;"
    )


def test_question_tool_robustness() -> None:
    """QuestionTool must tolerate missing/numeric IDs, numeric options, and single dict format."""
    from agent.tools.question_tool import QuestionTool, normalize_questions

    # Single dict questions argument
    normalized = normalize_questions({
        "questions": {"question": "What length?"}
    })
    assert isinstance(normalized, list)
    assert len(normalized) == 1

    # Missing ID, numeric options, duplicate IDs, option degradation
    questions = [
        {"question": "Enter width", "id": 1, "input_type": "number"},
        {"question": "Select size", "options": [10, 20, 30]},
        {"question": "Second width", "id": "1", "options": ["only_one"], "input_type": "select"},
    ]
    published: list[tuple[str, dict]] = []
    tool = QuestionTool(publish=lambda ev, pl: published.append((ev, pl)))
    res, waiting, out_questions = tool.execute({"questions": questions}, project="demo")
    assert waiting is True
    assert len(out_questions) == 3
    assert out_questions[0]["id"] == "1"
    assert "options" not in out_questions[0]
    assert out_questions[1]["id"] == "q2"
    assert out_questions[1]["input_type"] == "select"
    assert out_questions[1]["options"] == ["10", "20", "30"]
    # Duplicate ID made unique
    assert out_questions[2]["id"] == "1_3"
    # Degradation from < 2 options to text clears options so UI does not render invalid select
    assert out_questions[2]["input_type"] == "text"
    assert out_questions[2]["options"] == []

    # Batch with > 3 questions is smoothly capped to 3 without crashing
    four_questions = [{"question": f"Question {i}"} for i in range(4)]
    _, _, capped = tool.execute({"questions": four_questions}, project="demo")
    assert len(capped) == 3


def test_format_answer_handles_zero_and_multiselect() -> None:
    """_format_answer must not drop numeric 0 answers and must format lists cleanly."""
    from agent.core import AgentRunner

    schema = {
        "questions": [
            {"id": "offset", "question": "Wall offset (mm)"},
            {"id": "features", "question": "Selected features"},
        ]
    }
    # User entered 0 for numeric offset and two choices for features
    answer_json = json.dumps({"offset": 0, "features": ["Bevel", "Holes"]})
    formatted = AgentRunner._format_answer(schema, answer_json)

    assert "- Wall offset (mm): 0" in formatted
    assert "- Selected features: Bevel, Holes" in formatted


def test_cancel_remaining_tool_calls_produces_standard_failure_envelope(tmp_path: Path) -> None:
    """cancel_remaining_tool_calls wraps cancelled tools in standard tool_failure envelopes."""
    from agent.dispatcher import cancel_remaining_tool_calls

    appended: list[dict] = []
    tool_calls = [
        {"id": "call_1", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        {"id": "call_2", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    messages: list[dict] = []
    cancel_remaining_tool_calls(
        tmp_path,
        tool_calls,
        processed_call_ids={"call_1"},
        messages=messages,
        append_message=lambda _dir, msg: appended.append(msg),
    )
    assert len(messages) == 1
    assert len(appended) == 1
    msg = messages[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_2"
    parsed = json.loads(msg["content"])
    assert parsed["ok"] is False
    assert parsed["tool"] == "read_file"
    assert parsed["error"]["message"] == "Tool call cancelled (question or stop)."


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

    def chat(self, messages, tools=None, *, max_attempts=None):
        # Record the budget the wrapper forwarded so the new retry-budget
        # tests can assert how the wrapper distributed its attempt cap.
        self.chat_calls.append((messages, tools, max_attempts))
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

    def callback(*_args, **_kwargs):
        return None

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


def test_fallback_wrapper_mirrors_all_wrapped_attrs_via_sync() -> None:
    """Every per-call attr set on the wrapper reaches both inner clients
    once ``chat()`` triggers ``_sync_state()``.

    The agent runner writes ``stream_callback`` each iteration and
    ``stop_event`` / ``session_id`` / ``run_id`` once per run; if any of
    them failed to reach the inner clients the runner would silently
    stream to the previous callback or miss a stop signal.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    stop = MagicMock()
    logger = MagicMock()
    callback = lambda *_args, **_kwargs: None
    wrapper.stop_event = stop
    wrapper.session_id = "s-1"
    wrapper.agent_role = "planner"
    wrapper.stream_callback = callback
    wrapper.require_images = True
    wrapper.activity_logger = logger
    wrapper.run_id = "run-42"

    wrapper.chat([{"role": "user", "content": "hi"}])

    for client in (primary, fallback):
        assert client.stop_event is stop
        assert client.session_id == "s-1"
        assert client.agent_role == "planner"
        assert client.stream_callback is callback
        assert client.require_images is True
        assert client.activity_logger is logger
        assert client.run_id == "run-42"


def test_fallback_wrapper_init_seeds_mirrored_attrs_from_primary() -> None:
    """Pre-construction primary attrs are observable on the wrapper.

    A read against the wrapper before any ``chat()`` returns the
    primary's value (the value it had at wrap time). The inner clients'
    defaults flow through to the wrapper so external code that
    introspects the wrapper before any chat does not see ``AttributeError``.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    primary.session_id = "session-y"
    primary.run_id = "run-7"
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    assert wrapper.session_id == "session-y"
    assert wrapper.run_id == "run-7"


def test_fallback_wrapper_init_seeds_safe_captured_defaults() -> None:
    """``preserve_reasoning`` and friends default to safe no-op values.

    Without a chat the wrapper must report ``preserve_reasoning = False``
    (so the agent runner drops reasoning from a stubbed response) and a
    ``None`` / empty usage payload. Wiring captured attrs to the primary
    here would silently override stubbed test state, so ``__init__``
    intentionally seeds the safe defaults and ``_capture_result``
    overwrites them only after a real ``chat()`` runs.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    assert wrapper.preserve_reasoning is False
    assert wrapper.last_usage is None
    assert wrapper.last_image_fallback_used is False


def test_fallback_wrapper_passes_none_budget_when_no_caller_cap() -> None:
    """Without a caller cap, primary/fallback choose their own retry budget.

    The wrapper never invents a cap when the caller did not supply one;
    each inner client then runs its own default retry loop. This is the
    historical contract the production code path relies on.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    wrapper.chat([{"role": "user", "content": "hi"}])

    assert primary.chat_calls[-1][2] is None
    assert fallback.chat_calls == []  # primary succeeded, fallback not called


def test_fallback_wrapper_caps_primary_attempts_in_caller_budget() -> None:
    """A caller-supplied budget must split, not blow past, the cap.

    With ``max_attempts=3`` the wrapper must give primary 2 attempts
    (so a single transient 5xx still gets one retry) and the fallback 1.
    Previously the wrapper passed no budget at all and let each inner
    client run its full 3-attempt loop, so a "3-attempt budget" was
    really 6 attempts.
    """
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    wrapper.chat([{"role": "user", "content": "hi"}], max_attempts=3)

    assert primary.chat_calls[-1][2] == 2
    assert fallback.chat_calls == []


def test_fallback_wrapper_forwards_split_budget_to_fallback_on_primary_error() -> None:
    """When primary fails, the fallback gets the remaining budget.

    A ``max_attempts=4`` budget must reach the fallback as 2 attempts
    (not silently collapse to 0 and skip the fallback entirely).
    """
    primary = _StubChatClient(chat_error=RuntimeError("primary down"))
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    wrapper.chat([{"role": "user", "content": "hi"}], max_attempts=4)

    assert primary.chat_calls[-1][2] == 2
    assert fallback.chat_calls[-1][2] == 2
    assert primary.chat_calls[-1][2] + fallback.chat_calls[-1][2] == 4


def test_fallback_wrapper_skips_fallback_when_budget_exhausted_by_primary() -> None:
    """A 1-attempt cap must not silently extend into a fallback hop.

    The wrapper refuses to invent retries the caller didn't authorise:
    with ``max_attempts=1`` the primary gets the single shot and the
    primary's error propagates without invoking the fallback. This is
    the boundary case for the retry-budget cap.
    """
    primary = _StubChatClient(chat_error=RuntimeError("primary down"))
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")

    with pytest.raises(RuntimeError, match="primary down"):
        wrapper.chat([{"role": "user", "content": "hi"}], max_attempts=1)

    assert primary.chat_calls[-1][2] == 1
    assert fallback.chat_calls == []


def test_fallback_wrapper_tracks_active_client_across_chats() -> None:
    """``_capture_result`` updates the wrapper's active client pointer.

    ``last_usage`` / ``last_image_fallback_used`` / ``preserve_reasoning``
    reads after each chat reflect the provider that actually answered.
    Reading them again before a subsequent chat must still surface the
    previous successful provider's values — useful for diagnostics when
    a later iteration crashes before reaching ``_capture_result``.
    """
    primary = _StubChatClient(
        chat_result={"choices": [{"message": {}}]},
        usage={"prompt_tokens": 5, "completion_tokens": 6},
    )
    fallback = _StubChatClient(
        chat_result={"choices": [{"message": {}}]},
        usage={"prompt_tokens": 7, "completion_tokens": 8},
    )
    wrapper = FallbackChatClient(primary, fallback, "openai")

    wrapper.chat([{"role": "user", "content": "first"}])
    assert wrapper.last_usage == {"prompt_tokens": 5, "completion_tokens": 6}

    primary._chat_error = RuntimeError("primary offline")
    wrapper.chat([{"role": "user", "content": "second"}])
    assert wrapper.last_usage == {"prompt_tokens": 7, "completion_tokens": 8}


def test_fallback_wrapper_abort_sets_stop_and_calls_inner_abort() -> None:
    """``abort()`` propagates the stop signal to both inner clients."""
    primary = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    fallback = _StubChatClient(chat_result={"choices": [{"message": {}}]})
    wrapper = FallbackChatClient(primary, fallback, "openai")
    stop = MagicMock()
    wrapper.stop_event = stop
    primary.abort = MagicMock()
    fallback.abort = MagicMock()

    wrapper.abort()

    stop.set.assert_called_once()
    primary.abort.assert_called_once()
    fallback.abort.assert_called_once()


def test_sanitize_messages_preserves_reasoning_across_all_assistant_turns() -> None:
    """Historical assistant messages must keep reasoning across all turns for prompt cache stability."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
            "reasoning_details": [{"type": "reasoning.text", "text": "Turn 1 thinking"}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "edit_file", "arguments": "{}"}}],
            "reasoning_details": [{"type": "reasoning.text", "text": "Turn 2 thinking"}],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    ]

    sanitized = sanitize_messages(messages, preserve_reasoning=True)

    # Both older (index 2) and latest (index 4) assistant messages must retain their reasoning
    assert sanitized[2].get("reasoning_details") == [{"type": "reasoning.text", "text": "Turn 1 thinking"}]
    assert sanitized[4].get("reasoning_details") == [{"type": "reasoning.text", "text": "Turn 2 thinking"}]


def test_sanitize_messages_strips_reasoning_when_preserve_reasoning_is_false() -> None:
    """When preserve_reasoning is False (e.g. OpenAI), all reasoning fields are stripped."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
            "reasoning_details": [{"type": "reasoning.text", "text": "Turn 1 thinking"}],
            "reasoning": "Turn 1 raw thinking",
        },
    ]

    sanitized = sanitize_messages(messages, preserve_reasoning=False)
    assert "reasoning_details" not in sanitized[2]
    assert "reasoning" not in sanitized[2]


def test_sanitize_messages_multi_turn_prefix_stability() -> None:
    """Turn N messages must be a strict prefix of Turn N+1 messages to guarantee prompt cache hits."""
    turn_1 = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user request"},
        {
            "role": "assistant",
            "content": "Step 1",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
            "reasoning_details": [{"type": "reasoning.text", "text": "Thinking step 1"}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    sanitized_t1 = sanitize_messages(turn_1, preserve_reasoning=True)

    turn_2 = list(turn_1) + [
        {
            "role": "assistant",
            "content": "Step 2",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "cad_build_and_verify", "arguments": "{}"}}],
            "reasoning_details": [{"type": "reasoning.text", "text": "Thinking step 2"}],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    ]
    sanitized_t2 = sanitize_messages(turn_2, preserve_reasoning=True)

    # The prefix of turn_2 (up to len(sanitized_t1)) must be 100% identical to sanitized_t1
    assert sanitized_t2[:len(sanitized_t1)] == sanitized_t1


def test_conversation_store_load_preserves_images(tmp_path: Path) -> None:
    """ConversationStore.load preserves multimodal image_url parts so prompt prefixes don't diverge."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    msg_with_img = {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": [
            {"type": "text", "text": "build succeeded"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ],
    }
    ConversationStore.append(project_dir, msg_with_img)

    # Multimodal content is preserved verbatim so the prompt prefix stays
    # byte-stable across turns. Prompt caches rely on the same bytes the
    # provider hashed on the previous request; rewriting to placeholders
    # would invalidate every cached prefix.
    loaded = ConversationStore.load(project_dir)
    assert len(loaded) == 1
    assert loaded[0]["content"][1]["type"] == "image_url"
    assert loaded[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="


def test_agent_runner_multi_turn_prefix_and_cache_stability(tmp_path: Path) -> None:
    """Multi-turn chats must keep the system prompt stable and report truthful state.

    The cacheable contract is:
    * The system prompt (index 0) is byte-stable across every turn of a
      session — that is what provider-side prompt caching keys against.
    * The ``<project_state>`` user message (index 1) reflects the current
      disk state. It is allowed to change between turns when the file's
      existence flips; flipping it on the first turn after ``write_file``
      is correct (and was previously hidden by a buggy
      ``.agent_initial_state.json`` cache that lied to the model).
    * Once the file exists, subsequent turns without further file-system
      changes must keep the system prompt AND the ``<project_state>``
      prefix byte-stable so the provider cache can be reused.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "test-project"
    project_dir.mkdir()

    settings = Settings(
        workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # --- Turn 1 ---
    # Model does not exist initially
    context_t1 = runner._context(project_dir, "Create a cylinder", [])
    assert "model.scad does not exist" in context_t1[1]["content"]

    # Assistant executes write_file, creating model.scad
    (project_dir / "model.scad").write_text("cylinder(r=10, h=20);", encoding="utf-8")
    assistant_t1_1 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_write", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
        ],
        "reasoning_details": [{"type": "reasoning.text", "text": "Creating model.scad"}],
    }
    tool_write_result = {"role": "tool", "tool_call_id": "call_write", "content": "File written"}
    assistant_t1_2 = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_build", "type": "function", "function": {"name": "cad_build_and_verify", "arguments": "{}"}}
        ],
        "reasoning_details": [{"type": "reasoning.text", "text": "Building CAD"}],
    }
    tool_build_result = {
        "role": "tool",
        "tool_call_id": "call_build",
        "content": [
            {"type": "text", "text": "CAD build ok"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}},
        ],
    }
    assistant_t1_final = {
        "role": "assistant",
        "content": "Cylinder has been created and verified.",
    }

    # Append all messages to store as would happen during run
    for msg in [assistant_t1_1, tool_write_result, assistant_t1_2, tool_build_result, assistant_t1_final]:
        ConversationStore.append(project_dir, msg)

    # --- Turn 2 ---
    # The file now exists, so the state message must flip from
    # "does not exist" to "exists" so the model knows it can edit / verify.
    context_t2 = runner._context(project_dir, "Now drill a 5mm hole through the center", [])
    assert "model.scad exists" in context_t2[1]["content"]

    # The system prompt (index 0) must be byte-identical between turns —
    # that is what provider prompt caches key against.
    assert context_t2[0] == context_t1[0]
    # The ``<project_state>`` prefix legitimately flipped; subsequent turns
    # with stable state must keep it byte-stable from now on.
    assert "model.scad exists" in context_t2[1]["content"]

    # --- Turn 3 ---
    # With the file still present and no further disk changes, the cached
    # portion of the prompt (system + project_state + history up to the
    # last final assistant message) must be byte-identical across turns.
    # Each turn appends a new trailing user message, so we compare the
    # common prefix by slicing the trailing per-turn messages off both.
    context_t3 = runner._context(project_dir, "Add a chamfer to the edge", [])
    assert context_t3[0] == context_t2[0]
    assert context_t3[1] == context_t2[1]
    # context_t2 = [system, project_state, *history_t1, user_t2]
    # context_t3 = [system, project_state, *history_t1, user_t2, user_t3]
    # The shared cached prefix is everything before the new user message
    # each turn added. Both slices below produce that exact prefix.
    assert context_t2[:-1] == context_t3[:-2]

    # Sanity check: the tool image survives in the history slice at the
    # same logical position so the follow-up turn's tool evidence stays
    # accessible without a redundant rebuild.
    tool_with_image = context_t3[6]
    assert tool_with_image["role"] == "tool"
    assert tool_with_image["content"][1]["type"] == "image_url"
    assert tool_with_image["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def test_clear_history_removes_state_files(tmp_path: Path) -> None:
    """``clear_history`` removes the per-project state files alongside the log."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "test-project"
    project_dir.mkdir()

    settings = Settings(
        workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # Populate state files including the legacy ``.agent_initial_state.json``
    # left over from a previous release; ``clear_history`` cleans it up so
    # a project upgraded in place resets to a clean state.
    (project_dir / ".agent_initial_state.json").write_text("{}", encoding="utf-8")
    (project_dir / ".agent_state.json").write_text("{}", encoding="utf-8")
    (project_dir / "conversation.jsonl").write_text("{}", encoding="utf-8")

    # Clear history
    runner.clear_history(project_dir)
    assert not (project_dir / ".agent_initial_state.json").is_file()
    assert not (project_dir / ".agent_state.json").is_file()
    assert not (project_dir / "conversation.jsonl").is_file()


def test_sanitize_assistant_message_preserves_reasoning_for_non_tool_turn() -> None:
    """Assistant messages without tool calls still preserve reasoning when preserve_reasoning=True."""
    msg = {
        "role": "assistant",
        "content": "This is a direct response without any tool calls.",
        "reasoning": "I thought deeply about how to answer.",
        "reasoning_details": [{"type": "reasoning.text", "text": "Deep thinking text"}],
    }
    sanitized = sanitize_messages([msg], preserve_reasoning=True)
    assert len(sanitized) == 1
    assert sanitized[0].get("reasoning") == "I thought deeply about how to answer."
    assert sanitized[0].get("reasoning_details") == [{"type": "reasoning.text", "text": "Deep thinking text"}]


def test_conversation_store_unlimited_history_no_truncation(tmp_path: Path) -> None:
    """ConversationStore does not truncate history, keeping all messages permanently in memory."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # Append 150 messages (more than the legacy 100 limit)
    for i in range(150):
        ConversationStore.append(
            project_dir,
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"},
        )

    loaded = ConversationStore.load(project_dir)
    assert len(loaded) == 150
    assert loaded[0]["content"] == "Message 0"
    assert loaded[-1]["content"] == "Message 149"


def test_conversation_store_cache_deepcopy_protects_mutation(tmp_path: Path) -> None:
    """Mutating loaded history dicts does not corrupt the internal ConversationStore cache."""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    ConversationStore.append(project_dir, {"role": "user", "content": "Original Content"})

    first_load = ConversationStore.load(project_dir)
    assert first_load[0]["content"] == "Original Content"

    # Mutate the loaded dict in-place
    first_load[0]["content"] = "MUTATED"

    # Next load from cache must still have original content
    second_load = ConversationStore.load(project_dir)
    assert second_load[0]["content"] == "Original Content"


def test_compact_for_context_preserves_full_tool_payload() -> None:
    """compact_for_context returns tool results intact without lossy compression."""
    from agent.tool_results import compact_for_context

    raw_result = json.dumps({
        "ok": True,
        "tool": "cad_build_and_verify",
        "data": {
            "metrics": {
                "feature_summary": {
                    "cylinder_table": [{"radius": 5.0, "height": 10.0}],
                    "through_hole_count": 1,
                }
            },
            "preview": "preview.stl",
            "render": "render.png",
            "review_manifest": {"views": ["top", "front"]},
        }
    })

    result = compact_for_context("cad_build_and_verify", raw_result)
    parsed = json.loads(result)
    # None of the fields were stripped
    assert "cylinder_table" in parsed["data"]["metrics"]["feature_summary"]
    assert parsed["data"]["preview"] == "preview.stl"
    assert parsed["data"]["render"] == "render.png"
    assert "review_manifest" in parsed["data"]


def test_complete_does_not_duplicate_assistant_turn_with_reasoning(tmp_path: Path) -> None:
    """_complete does not re-append a duplicate assistant turn when one already exists with reasoning."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "test-project"
    project_dir.mkdir()

    settings = Settings(
        workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # Simulate assistant message with reasoning already in conversation log
    msg = {
        "role": "assistant",
        "content": "Finished part.",
        "reasoning_details": [{"type": "reasoning.text", "text": "Reasoning"}],
    }
    ConversationStore.append(project_dir, msg)

    # Call _complete
    runner._complete("test-project", "Finished part.")

    # Must still only be 1 message, not duplicated
    loaded = ConversationStore.load(project_dir)
    assert len(loaded) == 1
    assert loaded[0]["content"] == "Finished part."




def test_file_tool_write_reports_line_count_not_character_count(tmp_path: Path) -> None:
    """The write confirmation surfaces line count first; character count stays secondary."""
    tool = FileTool(tmp_path)

    message = tool.write_file(
        "model.scad",
        "WIDTH = 10;\ncube([WIDTH, 20, 30]);\n",
    )

    assert "lines" in message
    assert "chars" in message
    # Two newline-terminated lines — splitlines() drops
    # the trailing empty entry so a 2-line file always reports "2 lines".
    assert "2 lines" in message
    assert "characters" not in message


# ---------------------------------------------------------------------------
# Audited manufacturing defects from docs/issues-2.md
#
# Each test is intentionally narrow and self-contained so a regression points
# straight at the offending module without dragging in unrelated fixtures.
# ---------------------------------------------------------------------------


def test_revisions_trim_builds_log_runs_in_linear_time(tmp_path: Path) -> None:
    """``_trim_builds_log`` must not pop from the front of the line list.

    The old implementation did ``while ... lines.pop(0)`` and recomputed
    the byte total on every iteration, producing O(N^2) work and pinning
    the worker thread once the log filled up. A direct call should leave
    the file under the cap after a single linear pass.
    """
    import json

    from agent.revisions import _BUILDS_LOG_NAME, _BUILDS_MAX_BYTES

    log_path = tmp_path / _BUILDS_LOG_NAME
    # Build a log well past the 2 MB cap so the trim path triggers.
    target_bytes = _BUILDS_MAX_BYTES + 200 * 1024  # ~200 KB over the cap.
    with log_path.open("w", encoding="utf-8") as handle:
        written = 0
        index = 0
        while written < target_bytes:
            payload = json.dumps({"i": index, "pad": "x" * 1024})
            handle.write(payload + "\n")
            written += len(payload) + 1
            index += 1

    store = RevisionStore(tmp_path)
    store._trim_builds_log(log_path)

    # The trimmed log must be at or under the cap (the loop drops just
    # enough to fit). Crucially, it must NOT be empty — the trimmer
    # preserves the tail.
    assert log_path.stat().st_size <= _BUILDS_MAX_BYTES + 1024
    assert log_path.stat().st_size > 0


def test_append_message_persists_assistant_turn_to_history(tmp_path: Path) -> None:
    """``_append_message`` writes assistant turns to ``conversation.jsonl``.

    The agent loop now intentionally skips persistence for ``invalid_final``
    turns so the UI never shows a ghost message that was rejected by the
    visual verification gate. The low-level helper still commits every
    record it is given; the gate is enforced one layer up in
    ``AgentRunner._run``.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "test-project"
    project_dir.mkdir()

    settings = Settings(
        workspace, "https://example.test", "test-model", 5, "127.0.0.1", 5000
    )
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # Initial model.scad does not exist
    runner._context(project_dir, "Make part", [])
    assistant_msg = {
        "role": "assistant",
        "content": "I finished the part.",
    }
    runner._append_message(project_dir, assistant_msg)
    history = ConversationStore.load(project_dir)
    assert any(m.get("role") == "assistant" and m.get("content") == "I finished the part." for m in history)


def test_cad_failure_threshold_cancels_remaining_tool_calls(tmp_path: Path) -> None:
    """When CAD build failure threshold stops the run, remaining tool calls are cancelled."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "test-project"
    project_dir.mkdir()

    # Pre-populate model.scad with broken code
    (project_dir / "model.scad").write_text("broken syntax !!", encoding="utf-8")

    tool_calls = [
        {"id": "call_cad", "function": {"name": "cad_build_and_verify", "arguments": "{}"}},
        {"id": "call_read", "function": {"name": "read_file", "arguments": "{}"}},
    ]
    messages: list[dict] = []
    appended: list[dict] = []

    from agent.dispatcher import cancel_remaining_tool_calls
    # If call_cad failed and loop stops, call_read must be cancelled
    cancel_remaining_tool_calls(
        project_dir,
        tool_calls,
        processed_call_ids={"call_cad"},
        messages=messages,
        append_message=lambda _dir, msg: appended.append(msg),
    )
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "call_read"
    res = json.loads(messages[0]["content"])
    assert res["ok"] is False
    assert res["tool"] == "read_file"


def test_activity_log_trims_on_cadence_not_every_call(tmp_path: Path) -> None:
    """A burst of small appends must not trigger a disk rewrite per call.

    The previous ``_maybe_trim`` did a full read+rewrite of the log on
    every ``log()`` invocation. The cadence-gated version only rewrites
    once every ``_TRIM_EVERY`` writes (or sooner when a single event
    blows past ``2 * max_bytes``).
    """
    from agent.activity_log import _TRIM_EVERY, ActivityLogger

    logger = ActivityLogger(tmp_path, max_bytes=4 * 1024)  # tiny cap to force a trim

    rewrite_calls: list[None] = []
    real_atomic_write_text = ActivityLogger.log.__globals__["atomic_write_text"]

    def spy_atomic_write_text(path, content):  # type: ignore[no-untyped-def]
        rewrite_calls.append(None)
        return real_atomic_write_text(path, content)

    # Monkeypatch the bound module-level import used inside
    # ``ActivityLogger._maybe_trim``. ``logger.log`` itself goes through
    # the raw ``open`` path, so we only count trims.
    import agent.activity_log as activity_log_module
    original_trim = activity_log_module.atomic_write_text
    activity_log_module.atomic_write_text = spy_atomic_write_text
    try:
        # Many small writes: stay below the force-trim threshold so the
        # cadence is the only thing that should fire a rewrite.
        for _ in range(_TRIM_EVERY * 2):
            logger.log("tick")
    finally:
        activity_log_module.atomic_write_text = original_trim

    # We expect a tiny number of trims (one per cadence window), not
    # one per append. Allow a generous upper bound so a future tweak to
    # the cadence constant does not break the test, but flag a regression
    # back to per-call rewrites.
    assert len(rewrite_calls) < _TRIM_EVERY / 5, (
        f"Activity log was rewritten {len(rewrite_calls)} times for "
        f"{_TRIM_EVERY * 2} appends; the cadence is broken."
    )


def test_images_store_rejects_oversized_dimensions_before_decode(tmp_path: Path) -> None:
    """An image with declared pixels above ``MAX_IMAGE_PIXELS`` is rejected up front.

    ``image.draft()`` is a no-op for PNG/WebP so the old code could
    fully decode a 99 MP PNG (≈400 MB RAM) before down-sizing. The new
    guard combines a tight ``MAX_IMAGE_PIXELS`` with a header check
    before ``load()``.
    """
    from PIL import Image as PILImage
    from werkzeug.datastructures import FileStorage

    from agent.images import MAX_IMAGE_PIXELS, store_images

    assert MAX_IMAGE_PIXELS <= 10_000_000, (
        "Pixel cap regressed above the safe 10 MP threshold — "
        "decompression-bomb risk reintroduced."
    )

    # Fabricate a PNG whose declared dimensions exceed the cap. The
    # header is valid enough for ``Image.open`` to read width/height
    # without doing the full pixel decode.
    over = MAX_IMAGE_PIXELS + 1
    width = max(1, int(over ** 0.5))
    height = over // width + 1
    raw = PILImage.new("RGB", (1, 1)).resize((width, height)).tobytes()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)
        PILImage.frombytes("RGB", (width, height), raw).save(buf, format="PNG")
    payload = buf.getvalue()

    upload = FileStorage(stream=io.BytesIO(payload), filename="bomb.png", content_type="image/png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PILImage.DecompressionBombWarning)
        with pytest.raises(ValueError, match="dimensions exceed"):
            store_images([upload], tmp_path)


def test_revision_archive_import_aborts_atomically_on_corrupt_blob(tmp_path: Path) -> None:
    """A single mismatched blob sha must leave the store untouched.

    The previous implementation wrote blobs and revision manifests in
    the same loop, so a corrupt blob on revision 20 of 50 left the
    first 19 revisions on disk and the project in a zombie state. The
    new helper validates every blob in memory before touching disk.
    """
    import hashlib

    from agent.revision_archive import import_history
    from agent.revisions import RevisionOrigin, RevisionStore

    store = RevisionStore(tmp_path)
    head = store.commit(source="result = Box(1, 1, 1)\n", origin=RevisionOrigin(kind="agent_edit"))

    # Forge an archive: valid head + a known-good first revision, then a
    # second revision whose declared sha256 does not match its payload.
    good_source = "result = Box(2, 2, 2)\n"
    good_sha = hashlib.sha256(good_source.encode("utf-8")).hexdigest()
    archive = {
        "schema_version": 1,
        "exported_at": "2024-01-01T00:00:00+00:00",
        "head": {"revision_id": head.id, "model_sha256": head.model_sha256},
        "revisions": [
            {
                "id": head.id,
                "parent_id": None,
                "model_sha256": head.model_sha256,
                "created_at": head.created_at,
                "origin": head.origin.to_dict(),
                "restored_from": None,
            },
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "parent_id": None,
                "model_sha256": good_sha,
                "created_at": "2024-01-01T00:00:00+00:00",
                "origin": {"kind": "import"},
                "restored_from": None,
            },
        ],
        "blobs": {
            head.model_sha256: "result = Box(1, 1, 1)\n",
            good_sha: "this text does NOT hash to good_sha",
        },
    }
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(__import__("json").dumps(archive), encoding="utf-8")

    with pytest.raises(Exception) as exc:
        import_history(store, archive_path)

    # The store must not have partially imported the good revision; the
    # failure must surface before any disk write.
    assert "digest" in str(exc.value).lower() or "match" in str(exc.value).lower()
    assert (tmp_path / ".cad-agent" / "history" / "blobs" / f"{good_sha}.py").is_file() is False


def test_prompt_cache_does_not_stat_playbook_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_playbook`` must not hit the filesystem on the hot path.

    The previous implementation called ``playbook_path.stat()`` on every
    ``get_playbook()`` invocation, which fires once per LLM request.
    The new behaviour caches the playbook at import time and only
    re-reads when ``hot_reload=True`` is explicitly set.
    """

    class _TrackingPath:
        """Path-like stand-in whose ``stat`` and ``read_text`` count calls."""

        def __init__(self, real: Path) -> None:
            self._real = real
            self.stat_calls = 0
            self.read_calls = 0

        def stat(self) -> object:
            self.stat_calls += 1
            return self._real.stat()

        def read_text(self, encoding: str = "utf-8") -> str:
            self.read_calls += 1
            return self._real.read_text(encoding=encoding)

    from agent import prompt

    tracker = _TrackingPath(prompt._PLAYBOOK_PATH)
    monkeypatch.setattr(prompt, "_PLAYBOOK_PATH", tracker)

    cache = prompt.PromptCache(hot_reload=False)
    for _ in range(50):
        cache.get_playbook()

    assert tracker.stat_calls == 0, (
        f"get_playbook() hit the filesystem {tracker.stat_calls} times — "
        "the per-call stat regression has returned."
    )
    assert tracker.read_calls == 0


def test_file_tool_public_api_no_sha_kwargs(tmp_path: Path) -> None:
    """File tool methods must not accept ``expected_sha256`` / ``known_sha256``.

    These parameters were removed because no caller sent them and no schema
    exposed them — silently ignoring unknown kwargs would have hidden a
    future optimistic-concurrency regression (no real SHA was ever checked).
    """
    tool = FileTool(tmp_path)
    initial_code = "WIDTH = 10;\ncube([WIDTH, 20, 30]);\n"
    tool.write_file("model.scad", initial_code)

    # Each method must reject unknown kwargs explicitly via ``TypeError``
    # so a future caller wiring up optimistic-concurrency either lands on
    # a real implementation or fails loudly at the dispatch boundary.
    with pytest.raises(TypeError):
        tool.write_file("model.scad", initial_code, expected_sha256="x" * 64)
    with pytest.raises(TypeError):
        tool.read_file("model.scad", known_sha256="x" * 64)
    with pytest.raises(TypeError):
        tool.edit_file("model.scad", "WIDTH = 10", "WIDTH = 25", expected_sha256="x" * 64)
    with pytest.raises(TypeError):
        tool.edit_file_atomic(
            "model.scad",
            [{"old_string": "WIDTH = 10", "new_string": "WIDTH = 30"}],
            expected_sha256="x" * 64,
        )


def test_tool_schemas_do_not_expose_sha_parameters() -> None:
    """Tool schemas presented to the LLM must not include sha parameter properties."""
    from agent.tool_schemas import TOOL_SCHEMAS

    for tool in TOOL_SCHEMAS:
        func = tool.get("function", {})
        name = func.get("name", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        assert "expected_sha256" not in props, f"{name} should not expose expected_sha256"
        assert "known_sha256" not in props, f"{name} should not expose known_sha256"
        desc = func.get("description", "").lower()
        assert "expected_sha256" not in desc, f"{name} description should not mention expected_sha256"
        assert "known_sha256" not in desc, f"{name} description should not mention known_sha256"


def test_tool_schemas_streamlined_to_five_core_tools() -> None:
    """TOOL_SCHEMAS must contain exactly the 6 streamlined tools."""
    from agent.tool_schemas import TOOL_SCHEMAS

    names = [tool["function"]["name"] for tool in TOOL_SCHEMAS]
    assert names == [
        "read_file",
        "write_file",
        "edit_file",
        "cad_build_and_verify",
        "get_view_images",
        "question",
    ]


def test_edit_file_dispatcher_supports_root_level_strings_and_dict(tmp_path: Path) -> None:
    """Dispatcher must accept root-level old_string/new_string and single edit dict."""
    from agent.core import ProjectTools
    from agent.dispatcher import dispatch

    tools = ProjectTools(tmp_path, lambda *args: None)
    tools.file.write_file("model.scad", "WIDTH = 10;\nHEIGHT = 20;\n")

    # Root-level old_string and new_string
    _result, waiting = dispatch(tools, "test", "edit_file", {"old_string": "WIDTH = 10;", "new_string": "WIDTH = 15;"})
    assert not waiting
    assert "WIDTH = 15;" in (tmp_path / "model.scad").read_text(encoding="utf-8")

    # Dict in edits
    _result, waiting = dispatch(tools, "test", "edit_file", {"edits": {"old_string": "HEIGHT = 20;", "new_string": "HEIGHT = 25;"}})
    assert not waiting
    assert "HEIGHT = 25;" in (tmp_path / "model.scad").read_text(encoding="utf-8")


def test_write_file_does_not_emit_preflight_warning_for_valid_code(tmp_path: Path) -> None:
    """Writing valid OpenSCAD code must not emit PRE-FLIGHT WARNING."""
    tool = FileTool(tmp_path)
    code = (
        "WIDTH = 10;\n"
        "cube([WIDTH, 10, 10]);\n"
    )
    result = tool.write_file("model.scad", code)
    assert "PRE-FLIGHT WARNING" not in result
    assert "Wrote model.scad" in result


def test_read_file_defaults_to_full_file(tmp_path: Path) -> None:
    """Calling read_file without parameters returns the full file content."""
    import json
    tool = FileTool(tmp_path)
    code = "line1\nline2\nline3\n"
    tool.write_file("model.scad", code)

    res = tool.read_file("model.scad")
    assert isinstance(res, dict), "read_file must return a dict, not a JSON string."
    assert res["content"] == code
    assert res["total_lines"] == 3

    # ``tool_success`` is responsible for the JSON envelope; verify the
    # round-trip works end-to-end without producing a double-encoded blob.
    # The ``data`` field must hold a *dict* (not a stringified JSON blob):
    # a double-encoded ``data`` would surface as ``envelope["data"]`` being
    # a ``str`` and ``json.loads(envelope["data"])`` would still parse, but
    # the model receives needlessly escaped newlines / quotes. Asserting
    # ``isinstance(..., dict)`` catches regressions cheaply.
    envelope = json.loads(tool_results.success("read_file", res))
    assert isinstance(envelope["data"], dict), (
        "tool_success must surface read_file's dict payload as-is, "
        "without an extra JSON encoding layer."
    )
    inner = envelope["data"]
    assert inner["exists"] is True
    assert inner["content"] == code


def test_edit_file_dispatcher_edge_cases(tmp_path: Path) -> None:
    """Dispatcher must handle None new_string, empty edits list with old_string, and reject malformed edits."""
    import pytest

    from agent.core import ProjectTools
    from agent.dispatcher import dispatch

    tools = ProjectTools(tmp_path, lambda *args: None)
    tools.file.write_file("model.scad", "WIDTH = 10;\nHEIGHT = 20;\nDEPTH = 30;\n")

    # 1. new_string is None or omitted (treated as deletion)
    dispatch(tools, "test", "edit_file", {"old_string": "DEPTH = 30;\n", "new_string": None})
    assert "DEPTH = 30;" not in (tmp_path / "model.scad").read_text(encoding="utf-8")

    # 2. edits is [] but old_string is provided at top level
    dispatch(tools, "test", "edit_file", {"old_string": "HEIGHT = 20;", "new_string": "HEIGHT = 22;", "edits": []})
    assert "HEIGHT = 22;" in (tmp_path / "model.scad").read_text(encoding="utf-8")

    # 3. edits contains non-dict element
    with pytest.raises(ValueError, match="must be a list of objects"):
        dispatch(tools, "test", "edit_file", {"edits": ["invalid"]})

    # 4. empty edits with no old_string
    with pytest.raises(ValueError, match="edit_file requires 'edits'"):
        dispatch(tools, "test", "edit_file", {"edits": []})

    # 5. atomic edit with new_string=None deletes
    tools.file.edit_file_atomic("model.scad", [{"old_string": "HEIGHT = 22;\n", "new_string": None}])
    assert "HEIGHT = 22;" not in (tmp_path / "model.scad").read_text(encoding="utf-8")


def test_edit_file_whitespace_tolerance(tmp_path: Path) -> None:
    """edit_file and edit_file_atomic must tolerate indentation and whitespace discrepancies."""
    from agent.tools.file_tool import FileTool

    tool = FileTool(tmp_path)
    initial_code = (
        "hex_pts = [\n"
        "    [30, 30],\n"
        "    [90, 90]  // vertex\n"
        "];\n\n"
        "union() {\n"
        "    translate([0, 0, 0]) {\n"
        "        cube([10, 10, 10]);\n"
        "    }\n"
        "}\n"
    )
    tool.write_file("model.scad", initial_code)

    # 1. Single-line indentation mismatch (LLM passes 8 spaces, file has 4 spaces)
    tool.edit_file(
        "model.scad",
        "        [90, 90]  // vertex",
        "        [60, 60]  // peaked vertex",
    )
    content = (tmp_path / "model.scad").read_text(encoding="utf-8")
    assert "    [60, 60]  // peaked vertex" in content

    # 2. Multi-line nested block with indent mismatch (re-aligns to file indent)
    tool.edit_file(
        "model.scad",
        "        translate([0, 0, 0]) {\n            cube([10, 10, 10]);",
        "        translate([0, 0, 5]) {\n            cylinder(r=5, h=10);",
    )
    content = (tmp_path / "model.scad").read_text(encoding="utf-8")
    assert "    translate([0, 0, 5]) {\n        cylinder(r=5, h=10);" in content

    # 3. edit_file_atomic handles multiple edits with whitespace discrepancies
    tool.edit_file_atomic(
        "model.scad",
        [
            {
                "old_string": "    [60, 60]  // peaked vertex   ",  # trailing spaces
                "new_string": "    [0, 0]",
            },
            {
                "old_string": "        cylinder(r=5, h=10);",
                "new_string": "        cylinder(r=6, h=12);",
            },
        ],
    )
    content = (tmp_path / "model.scad").read_text(encoding="utf-8")
    assert "    [0, 0]" in content
    assert "        cylinder(r=6, h=12);" in content


def test_failure_signature_threshold_allows_escalation() -> None:
    """AgentRunner must allow 2 retries (total 3 attempts on same error) before stopping."""
    from agent.core import (
        _BUILD_FAILURE_PER_SIGNATURE_MAX,
        AgentRunner,
    )

    err_msg = 'CAD execution failed:\n  File "model.scad", line 111\nERROR: Parser error'
    sig = AgentRunner._failure_signature(err_msg)
    assert len(sig) == 16

    # Verify line numbers are normalized in signature
    err_msg_diff_line = 'CAD execution failed:\n  File "model.scad", line 125\nERROR: Parser error'
    sig_diff_line = AgentRunner._failure_signature(err_msg_diff_line)
    assert sig == sig_diff_line

    signatures: dict[str, int] = {}
    total_count = 0
    # Attempt 1: Initial failure — must not terminate
    total_count += 1
    signatures[sig] = signatures.get(sig, 0) + 1
    assert not AgentRunner._build_failure_exhausted(
        total_count, signatures, sig
    ), "Attempt 1 must not terminate"

    # Attempt 2: First repair attempt fails with same error — must NOT terminate
    total_count += 1
    signatures[sig] = signatures.get(sig, 0) + 1
    assert not AgentRunner._build_failure_exhausted(
        total_count, signatures, sig
    ), "Attempt 2 must not terminate, allowing 3-step escalation"

    # Attempt 3: Second repair attempt fails with same error — now terminates
    total_count += 1
    signatures[sig] = signatures.get(sig, 0) + 1
    assert AgentRunner._build_failure_exhausted(
        total_count, signatures, sig
    ), "Attempt 3 must terminate repeated failures"
    # Sanity-check the constant — locks the 3-attempt cadence to the
    # numeric budget so a future tuning cannot silently change behaviour.
    assert _BUILD_FAILURE_PER_SIGNATURE_MAX == 3


def test_build_failure_exhausted_total_cap_separate_from_signature() -> None:
    """The total-count cap trips independently of any single signature."""
    from agent.core import (
        _BUILD_FAILURE_PER_SIGNATURE_MAX,
        _BUILD_FAILURE_TOTAL_MAX,
        AgentRunner,
    )

    # Use several distinct signatures; each stays below the per-sig cap
    # but the rolling total crosses the budget.
    distinct = [f"sig_{i}" for i in range(_BUILD_FAILURE_TOTAL_MAX)]
    signatures: dict[str, int] = {sig: 1 for sig in distinct}
    assert AgentRunner._build_failure_exhausted(
        _BUILD_FAILURE_TOTAL_MAX, signatures, distinct[-1]
    )
    # Sanity: the per-sig budget never fires here because each signature
    # only saw one occurrence.
    assert _BUILD_FAILURE_PER_SIGNATURE_MAX > 1
    assert not AgentRunner._build_failure_exhausted(1, {distinct[0]: 1}, distinct[0])


def test_cad_tool_failure_detail_preserves_multiline_errors() -> None:
    """CadTool._failure_detail must preserve error details on lines following RuntimeError."""
    from agent.tools.cad_tool import CadTool

    traceback_text = (
        'Traceback (most recent call last):\n'
        '  File "runner.py", line 440, in <module>\n'
        '    main()\n'
        '  File "runner.py", line 338, in _run_model\n'
        '    raise RuntimeError(f"OpenSCAD execution failed:\\n{summary}")\n'
        'RuntimeError: OpenSCAD execution failed:\n'
        'ERROR: Parser error: syntax error in file model.scad, line 10\n'
        "Can't parse file 'model.scad'!\n"
    )
    detail = CadTool._failure_detail(traceback_text)
    assert "ERROR: Parser error: syntax error in file model.scad, line 10" in detail
    assert "Can't parse file 'model.scad'!" in detail


def test_sanitize_assistant_message_handles_reasoning_content() -> None:
    """sanitize_assistant_message must strip or normalize reasoning_content and drop empty reasoning."""
    # When preserve_reasoning is False, reasoning_content is dropped
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test"}}],
        "reasoning_content": "thinking...",
    }
    sanitized = sanitize_assistant_message(msg, preserve_reasoning=False)
    assert "reasoning_content" not in sanitized
    assert "reasoning" not in sanitized

    # When preserve_reasoning is True, non-empty reasoning_content is normalized to reasoning
    sanitized_preserved = sanitize_assistant_message(msg, preserve_reasoning=True)
    assert "reasoning_content" not in sanitized_preserved
    assert sanitized_preserved.get("reasoning") == "thinking..."

    # Empty reasoning and empty reasoning_content are always stripped
    empty_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [],
        "reasoning": "   ",
        "reasoning_content": "",
    }
    sanitized_empty = sanitize_assistant_message(empty_msg, preserve_reasoning=True)
    assert "reasoning_content" not in sanitized_empty
    assert "reasoning" not in sanitized_empty


def test_normalize_messages_merges_consecutive_user_messages() -> None:
    """normalize_messages must merge adjacent user messages to satisfy strict role alternation."""
    messages = [
        {"role": "system", "content": "You are a CAD assistant."},
        {"role": "user", "content": "<project_state>model.scad does not exist</project_state>"},
        {"role": "user", "content": "Create a mounting plate."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Follow-up question."},
    ]
    normalized = normalize_messages(messages)
    assert len(normalized) == 4
    assert normalized[0]["role"] == "system"
    assert normalized[1]["role"] == "user"
    assert "<project_state>model.scad does not exist</project_state>\n\nCreate a mounting plate." == normalized[1]["content"]
    assert normalized[2]["role"] == "assistant"
    assert normalized[3]["role"] == "user"


def test_project_state_reflects_current_disk_state(tmp_path: Path) -> None:
    """The ``<project_state>`` message always reflects the current disk state.

    Earlier versions cached the initial existence state in
    ``.agent_initial_state.json`` to keep the prompt prefix byte-stable
    across turns, but that lied to the model after ``write_file`` created
    ``model.scad`` (it kept telling the model the file did not exist). The
    state must now follow the file system on every turn so the model can
    plan its next ``edit_file`` / ``cad_build_and_verify`` call correctly.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_dir = workspace / "fresh-proj"
    project_dir.mkdir()

    settings = Settings(workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    runner = AgentRunner(settings, publish=lambda *args, **kwargs: None)

    # Initial turn: model does not exist
    c1 = runner._context(project_dir, "Create plate", [])
    assert "model.scad does not exist" in c1[1]["content"]

    # Model is created on disk by a write_file tool
    (project_dir / "model.scad").write_text("cube([10, 10, 10]);", encoding="utf-8")
    ConversationStore.append(project_dir, {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "write_file"}}]
    })

    # Second turn: state MUST now read "exists" so the model can edit/verify.
    c2 = runner._context(project_dir, "Drill hole", [])
    assert "model.scad exists" in c2[1]["content"]
    assert "does not exist" not in c2[1]["content"]


# ---------------------------------------------------------------------------
# Activity-log payload sanitization
# ---------------------------------------------------------------------------


def test_summarize_llm_messages_replaces_image_payloads_with_structural_metadata() -> None:
    """Inline image data URLs must not survive sanitization.

    ``llm_request`` activity-log events must never carry base64 image
    data URLs or raw user text. ``summarize_llm_messages`` rewrites
    each message to ``{role, content_bytes, image_count}`` plus an
    optional truncated preview, so the contract can be verified purely
    from the helper's output without standing up a Chat Completions
    server.
    """
    from agent.activity_log import summarize_llm_messages

    messages = [
        {"role": "system", "content": "You are a CAD assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this sketch and build it."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + "A" * 500_000
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + "B" * 250_000
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll write the model.scad now."}
            ],
        },
        {"role": "user", "content": "Continue."},
    ]

    summary = summarize_llm_messages(messages)

    # The structure must match the helper's documented contract exactly.
    assert summary[0] == {
        "role": "system",
        "content_bytes": len(b"You are a CAD assistant."),
        "image_count": 0,
        "text_preview": "You are a CAD assistant.",
    }
    assert summary[1]["role"] == "user"
    assert summary[1]["image_count"] == 2
    # The 500 KiB of base64 is excluded from the byte count; only the
    # textual payload contributes so the operator can still tell how much
    # *text* the prompt carried.
    assert summary[1]["content_bytes"] == len(
        b"Look at this sketch and build it."
    )
    # No image data URL may survive — the redaction must strip both ``A``s
    # and ``B``s entirely. The preview is short and safe.
    assert summary[1]["text_preview"] == "Look at this sketch and build it."
    assert "A" * 100 not in json.dumps(summary)
    assert "B" * 100 not in json.dumps(summary)
    assert summary[2]["image_count"] == 0
    assert summary[2]["content_bytes"] == len(
        b"I'll write the model.scad now."
    )
    assert summary[3] == {
        "role": "user",
        "content_bytes": len(b"Continue."),
        "image_count": 0,
        "text_preview": "Continue.",
    }


def test_summarize_llm_messages_truncates_long_text_preview() -> None:
    """Text previews are hard-capped at 200 characters with an ellipsis."""
    from agent.activity_log import (
        _LLM_MESSAGE_TEXT_PREVIEW_CHARS,
        summarize_llm_messages,
    )

    long_text = "X" * (_LLM_MESSAGE_TEXT_PREVIEW_CHARS * 3)
    summary = summarize_llm_messages([{"role": "user", "content": long_text}])

    assert summary[0]["content_bytes"] == len(long_text.encode("utf-8"))
    preview = summary[0]["text_preview"]
    # 200 'X's followed by '...'
    assert preview.endswith("...")
    assert preview.count(".") == 3
    assert preview.startswith("X" * _LLM_MESSAGE_TEXT_PREVIEW_CHARS)


def test_summarize_llm_messages_suppresses_preview_for_credential_payloads() -> None:
    """Messages that look like they contain a credential are preview-stripped.

    ``_LLM_PREVIEW_SECRET_PATTERNS`` is conservative on purpose: a missing
    preview is a debugging nuisance; a leaked API key on disk is a security
    incident. The structural summary still records the byte count so the
    operator can see *that* a credential-shaped payload was sent.
    """
    from agent.activity_log import summarize_llm_messages

    cases = [
        "My api_key=sk-abcdefghijklmnopqrstuvwxyz012345",
        "Authorization: Bearer sk-or-vwxyz0123456789abcdefghij",
        "password=hunter2-trustme-please-rotate-me-now",
        "secret=shhh-this-is-very-private-data-1234567",
        "api-key: sk-AAAaaa111bbb222ccc333ddd444eee555",
    ]
    for text in cases:
        summary = summarize_llm_messages([{"role": "user", "content": text}])
        assert "text_preview" not in summary[0], (
            f"Credential-shaped payload leaked preview: {text!r}"
        )
        # The byte count is still reported so the operator can see the prompt
        # was non-trivial.
        assert summary[0]["content_bytes"] == len(text.encode("utf-8"))
        assert summary[0]["role"] == "user"
        assert summary[0]["image_count"] == 0


def test_summarize_llm_messages_image_only_payload_has_no_preview() -> None:
    """An image-only user turn has no text preview to redact."""
    from agent.activity_log import summarize_llm_messages

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBOR"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo="
                    },
                },
            ],
        }
    ]
    summary = summarize_llm_messages(messages)
    assert summary[0]["role"] == "user"
    assert summary[0]["image_count"] == 2
    assert summary[0]["content_bytes"] == 0
    assert "text_preview" not in summary[0]


def test_summarize_llm_messages_handles_non_dict_entries() -> None:
    """Malformed entries are reported as ``role=None`` rather than crashing."""
    from agent.activity_log import summarize_llm_messages

    summary = summarize_llm_messages(["not a dict", 42, None])
    assert summary == [
        {"role": None, "content_bytes": 0, "image_count": 0},
        {"role": None, "content_bytes": 0, "image_count": 0},
        {"role": None, "content_bytes": 0, "image_count": 0},
    ]


class _RecordingActivityLogger:
    """Stand-in for ``ActivityLogger`` that records every event payload.

    The contract ``chat()`` honours is that ``llm_request`` payloads
    no longer carry raw messages. Mirroring the production log surface
    as a lightweight recorder keeps the test honest about which fields
    the sanitization step touches (the whole payload) and which it
    leaves alone (headers, model, status, attempt).
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, event: str, payload, *, run_id=None):  # type: ignore[no-untyped-def]
        self.events.append({"event": event, "payload": payload, "run_id": run_id})


class _RecordingChatClient(ChatCompletionsClient):
    """Concrete subclass of ``ChatCompletionsClient`` for sanitization tests.

    The base class declares ``_endpoint``/``_build_headers``/``_build_payload``/
    ``_post``/``_api_key`` abstract (``pragma: no cover``). Filling them in
    with no-op stubs lets us drive ``chat()`` end-to-end and observe the
    payload the activity logger receives without standing up an HTTP server.
    """

    def __init__(self, settings, *responses):  # type: ignore[no-untyped-def]
        super().__init__(settings, provider_label="openai")
        # The base class owns ``activity_logger`` and ``run_id``; we set
        # them so ``chat()`` takes the logging branch.
        self._stub_responses = list(responses)
        self._stub_index = 0

    def _endpoint(self) -> str:
        return "https://example.test/v1/chat/completions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "content-type": "application/json",
        }

    def _build_payload(self, messages, tools):  # type: ignore[no-untyped-def]
        return {"model": "gpt-test", "messages": deepcopy(messages), "tools": tools}

    def _post(self, payload, headers):  # type: ignore[no-untyped-def]
        # Return a stub streaming response that parses to a valid
        # chat-completion body. The exact shape is irrelevant — only the
        # ``status < 500`` path matters because that is where ``llm_request``
        # is logged.
        class _StubResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def __getattr__(self, name):
                # ``chat()`` checks ``hasattr(response, "iter_lines")`` to
                # pick between streaming and JSON paths; point it at the
                # JSON branch so we do not need a full SSE parser.
                if name == "iter_lines":
                    raise AttributeError(name)
                raise AttributeError(name)

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _StubResponse()

    def _api_key(self) -> str:
        return "test-api-key"


def test_chat_completions_client_logs_sanitized_llm_request(tmp_path: Path) -> None:
    """End-to-end: the activity log records a structural summary, not raw messages.

    A single ``llm_request`` event in ``activity.jsonl`` must not carry the
    base64 image data URL of an attached image, nor the raw user prompt.
    The summary fields exposed by ``summarize_llm_messages`` are the only
    allowed shape.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(workspace, "https://example.test", "test-model", 1, "127.0.0.1", 5000)
    # ``agent_log_tool_activity`` gates activity logging in production.
    # ``is_enabled`` only inspects this flag, so flipping it on is enough
    # to drive ``chat()`` down the logging branch.
    object.__setattr__(settings, "agent_log_tool_activity", True)

    recorder = _RecordingActivityLogger()
    client = _RecordingChatClient(settings)
    client.activity_logger = recorder
    client.run_id = "test-run"

    messages = [
        {"role": "system", "content": "You are a CAD assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Build a flange like the attached image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + "Z" * 200_000
                    },
                },
            ],
        },
    ]
    client.chat(messages, tools=[])

    llm_events = [event for event in recorder.events if event["event"] == "llm_request"]
    assert len(llm_events) == 1, (
        f"Expected exactly one llm_request event, got {[e['event'] for e in recorder.events]}"
    )
    logged_payload = llm_events[0]["payload"]["payload"]
    assert isinstance(logged_payload, dict)
    logged_messages = logged_payload["messages"]

    # The structural summary has two entries (system + user) and *no*
    # ``image_url`` parts survived.
    serialized = json.dumps(logged_messages)
    assert "data:image" not in serialized
    assert "Z" * 100 not in serialized
    # The summary helper allows a 200-char preview of textual prompts so a
    # short prompt legitimately surfaces in ``text_preview`` — but the
    # raw ``content`` field must never appear, and no image part may
    # survive. Verify the structure replaced the original payload
    # wholesale.
    assert all(
        isinstance(entry, dict)
        and set(entry.keys()).issubset(
            {"role", "content_bytes", "image_count", "text_preview"}
        )
        and "content" not in entry
        for entry in logged_messages
    ), "Structural summary must replace (not augment) the raw content field"
    assert logged_messages[0]["role"] == "system"
    assert logged_messages[0]["content_bytes"] == len(
        b"You are a CAD assistant."
    )
    assert logged_messages[0]["image_count"] == 0
    assert logged_messages[1]["role"] == "user"
    assert logged_messages[1]["image_count"] == 1
    # 200 KiB of base64 must NOT contribute to content_bytes (only text
    # counts) so the rolling 5 MiB cap cannot be exhausted by an image.
    assert logged_messages[1]["content_bytes"] == len(
        b"Build a flange like the attached image."
    )

    # Headers are redacted by ``_REDACTED_KEYS`` deeper in the logging
    # pipeline (``activity_log.redact`` walks the entire payload dict).
    # The recording logger used here captures the *pre-redaction* dict, so
    # we apply redaction manually to verify the headers are wired up
    # correctly end-to-end.
    from agent.activity_log import redact

    logged_headers = redact(llm_events[0]["payload"]["headers"])
    serialized_headers = json.dumps(logged_headers)
    assert "test-api-key" not in serialized_headers


# ---------------------------------------------------------------------------
# Phase 3 regression tests.
# ---------------------------------------------------------------------------


def test_eval_expr_raises_on_unknown_identifier() -> None:
    """The silent 0.0 fallback is gone.

    Unknown identifiers must raise
    :class:`ExpressionEvaluationError` rather than being silently
    substituted with ``0.0`` and shipped as a "valid" zero-diameter
    geometry.
    """
    from agent.tools.cad_scripts.runner import ExpressionEvaluationError, _eval_expr

    with pytest.raises(ExpressionEvaluationError) as exc_info:
        _eval_expr("UNKNOWN_PARAM", {})
    assert "UNKNOWN_PARAM" in str(exc_info.value)


def test_eval_expr_returns_float_for_known_expression() -> None:
    """Happy path still resolves params and arithmetic."""
    from agent.tools.cad_scripts.runner import _eval_expr

    assert _eval_expr("5.0", {}) == 5.0
    assert _eval_expr("HOLE_DIAMETER", {"HOLE_DIAMETER": 6.0}) == 6.0
    assert _eval_expr("HEIGHT + EPS", {"HEIGHT": 8.0, "EPS": 0.01}) == 8.01
    # Division must also be evaluated, not rejected as a non-numeric
    # symbol — OpenSCAD expressions frequently use ratios (e.g.
    # ``HEIGHT/2``) and the previous regex regression silently broke
    # every model that scaled a parameter by a divisor.
    assert _eval_expr("10/4", {}) == 2.5
    assert _eval_expr("HEIGHT/2", {"HEIGHT": 10.0}) == 5.0
    assert _eval_expr(
        "(WIDTH + 2 * RIM) / 2", {"WIDTH": 12.0, "RIM": 1.5}
    ) == 7.5


def test_extract_scad_features_ignores_non_numeric_args() -> None:
    """``center=true`` / ``$fn`` keywords must not raise.

    OpenSCAD cylinder calls routinely mix numeric dimensions with
    positioning flags (``center=true``) and fragment counts
    (``$fn=60``). Those keywords are not numeric expressions; feeding
    them to ``_eval_expr`` would otherwise raise
    ``ExpressionEvaluationError`` for the right reason on the wrong
    operand.
    """
    from agent.tools.cad_scripts.runner import _extract_scad_features

    scad = (
        "difference() {\n"
        "    cube([10, 10, 10]);\n"
        "    cylinder(d=5, h=12, center=true, $fn=60);\n"
        "}\n"
    )
    features = _extract_scad_features(scad, {"x": 10, "y": 10, "z": 10})
    assert len(features["cutouts"]) == 1
    assert features["cutouts"][0]["radius"] == 2.5


def test_through_hole_tolerance_ratio_is_documented_constant() -> None:
    """``0.9`` must live as a named constant with rationale."""
    from agent.tools.cad_scripts.runner import (
        _THROUGH_HOLE_TOLERANCE_RATIO,
        _extract_scad_features,
    )

    assert _THROUGH_HOLE_TOLERANCE_RATIO == 0.9
    # Empirical contract: cylinder spanning >= 90% of the shortest
    # bounding-box dimension is classified as through.
    scad_through = (
        "difference() {\n"
        "    cube([10, 10, 10]);\n"
        "    cylinder(d=5, h=10);\n"  # h = 100% of z = through
        "}\n"
    )
    scad_blind = (
        "difference() {\n"
        "    cube([10, 10, 10]);\n"
        "    cylinder(d=5, h=5);\n"  # h = 50% of z = blind
        "}\n"
    )
    through_features = _extract_scad_features(scad_through, {"x": 10, "y": 10, "z": 10})
    blind_features = _extract_scad_features(scad_blind, {"x": 10, "y": 10, "z": 10})
    assert through_features["through_hole_count"] == 1
    assert blind_features["blind_hole_count"] == 1


def test_render_mode_enum_properties() -> None:
    """The enum is the validated source of truth for what artifacts are staged."""
    from agent.tools.cad_tool import RenderMode

    assert RenderMode.NONE.produces_render is False
    assert RenderMode.NONE.produces_manifest is False
    assert RenderMode.FULL_REVIEW.produces_render is True
    assert RenderMode.FULL_REVIEW.produces_manifest is True


def test_runner_settings_are_derived_from_validated_mode() -> None:
    """Runner flags match :class:`RenderMode`, not booleans.

    ``render_views`` and ``write_isometric`` only fire for
    ``FULL_REVIEW`` so the ``/render`` endpoint's SHA gate always finds
    ``render.png`` together with the multi-view manifest.
    """
    from agent.tools.cad_tool import CadTool, RenderMode

    tool = CadTool.__new__(CadTool)
    tool._review_render_workers = 4
    tool._review_required_views = 8
    settings = tool._runner_settings(RenderMode.FULL_REVIEW)
    assert settings["render_views"] is True
    assert settings["write_isometric"] is True

    settings = tool._runner_settings(RenderMode.NONE)
    assert settings["render_views"] is False
    assert settings["write_isometric"] is False


# ---------------------------------------------------------------------------
# ``get_view_images`` tool
# ---------------------------------------------------------------------------


from agent.tools.image_tool import (  # noqa: E402  (intentional late import)
    CANONICAL_VIEW_IDS,
    ImageTool,
)


def _make_tiny_png(path: Path, *, color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Write a real 512x512 RGB PNG to ``path`` and return its bytes."""
    import io

    from PIL import Image

    image = Image.new("RGB", (512, 512), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


def _seed_review_dir(
    project_dir: Path,
    *,
    view_ids: tuple[str, ...] = CANONICAL_VIEW_IDS,
) -> tuple[str, dict[str, str]]:
    """Seed a synthetic review dir keyed by the current ``model.scad``.

    Returns ``(review_sha256, view_sha256s)`` where ``view_sha256s`` maps
    each view id to the freshly written PNG's sha256. ``model.scad`` is
    written so :func:`compute_model_sha256` produces a stable digest.
    """
    from agent.revisions import MODEL_FILENAME, compute_model_sha256

    (project_dir / MODEL_FILENAME).write_text("cube([1, 2, 3]);\n", encoding="utf-8")
    digest = compute_model_sha256(project_dir)
    assert digest is not None
    review_root = project_dir / ".cad-agent" / "reviews" / digest
    views_dir = review_root / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_entries: list[dict] = []
    view_sha: dict[str, str] = {}
    for index, view_id in enumerate(view_ids):
        path = views_dir / f"{view_id}.png"
        colour = (
            (10 + 20 * index) % 255,
            (20 + 30 * index) % 255,
            (30 + 40 * index) % 255,
        )
        data = _make_tiny_png(path, color=colour)
        sha = hashlib.sha256(data).hexdigest()
        view_sha[view_id] = sha
        view_entries.append(
            {
                "view_id": view_id,
                "image_sha256": sha,
                "path": f"views/{view_id}.png",
            }
        )
    (review_root / "manifest.json").write_text(
        json.dumps({"views": view_entries}),
        encoding="utf-8",
    )
    return digest, view_sha


def test_get_view_images_returns_full_views_and_paths(tmp_path: Path) -> None:
    """Happy path: a full-view request returns the manifest hash and dims."""
    digest, view_sha = _seed_review_dir(tmp_path)
    tool = ImageTool(tmp_path)
    result = tool.get_view_images(
        [
            {"view": "z_positive"},
            {"view": "isometric_positive"},
        ]
    )
    assert result["review_sha256"] == digest
    images = result["images"]
    assert [item["view"] for item in images] == ["z_positive", "isometric_positive"]
    for item in images:
        assert item["cropped"] is False
        assert item["width"] == 512
        assert item["height"] == 512
        assert item["sha256"] == view_sha[item["view"]]
        assert item["path"] == f"views/{item['view']}.png"


def test_get_view_images_creates_and_reuses_crop_cache(tmp_path: Path) -> None:
    """Cropped requests persist under crops/ with stable filenames."""
    _, _ = _seed_review_dir(tmp_path)
    tool = ImageTool(tmp_path)
    crop = {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.5}
    first = tool.get_view_images([{"view": "x_positive", "crop": crop}])
    entry = first["images"][0]
    assert entry["cropped"] is True
    crop_path = tmp_path / ".cad-agent" / "reviews" / first["review_sha256"] / entry["path"]
    assert crop_path.is_file()
    mtime = crop_path.stat().st_mtime_ns
    assert entry["width"] == 256
    assert entry["height"] == 256
    # Second call must reuse the cached file (no rewrite).
    second = tool.get_view_images([{"view": "x_positive", "crop": crop}])
    assert second["images"][0]["path"] == entry["path"]
    assert crop_path.stat().st_mtime_ns == mtime


def test_get_view_images_rejects_unknown_view(tmp_path: Path) -> None:
    """``ValueError`` when the LLM sends a view id not in the enum."""
    _seed_review_dir(tmp_path)
    tool = ImageTool(tmp_path)
    with pytest.raises(ValueError, match="Unknown view id"):
        tool.get_view_images([{"view": "bogus_view"}])


def test_get_view_images_rejects_degenerate_crop(tmp_path: Path) -> None:
    """Crops smaller than 8 px on either side must be rejected."""
    _seed_review_dir(tmp_path)
    tool = ImageTool(tmp_path)
    with pytest.raises(ValueError, match="degenerate"):
        tool.get_view_images(
            [
                {
                    "view": "y_positive",
                    "crop": {"x": 0.0, "y": 0.0, "width": 0.001, "height": 0.5},
                }
            ]
        )


def test_get_view_images_deduplicates_identical_requests(tmp_path: Path) -> None:
    """Two identical (view, crop) tuples in one call collapse to one entry."""
    _seed_review_dir(tmp_path)
    tool = ImageTool(tmp_path)
    crop = {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5}
    result = tool.get_view_images(
        [
            {"view": "z_positive", "crop": crop},
            {"view": "z_positive", "crop": crop},
            {"view": "z_positive"},
        ]
    )
    assert len(result["images"]) == 2


def test_get_view_images_requires_existing_review(tmp_path: Path) -> None:
    """Without a successful build the tool raises a clear ValueError."""
    tool = ImageTool(tmp_path)
    with pytest.raises(ValueError, match="cad_build_and_verify"):
        tool.get_view_images([{"view": "z_positive"}])


def test_get_view_images_detects_stale_review_after_model_edit(tmp_path: Path) -> None:
    """Editing ``model.scad`` after a build must surface a stale-review error."""
    from agent.revisions import MODEL_FILENAME

    _seed_review_dir(tmp_path)
    # The seeded review is keyed to the original hash. Mutate model.scad
    # so its current sha no longer matches the review directory name.
    (tmp_path / MODEL_FILENAME).write_text("cube([4, 5, 6]);\n", encoding="utf-8")
    tool = ImageTool(tmp_path)
    with pytest.raises(ValueError, match="cad_build_and_verify"):
        tool.get_view_images([{"view": "z_positive"}])


def test_get_view_images_rejects_tampered_view_png(tmp_path: Path) -> None:
    """Tampered bytes -> the manifest hash check must skip the view.

    A single tampered view must not produce a result if it's the only
    request, and must be dropped from the returned list when other
    views are requested alongside.
    """
    from agent.revisions import MODEL_FILENAME, compute_model_sha256

    digest, _view_sha = _seed_review_dir(tmp_path)
    review_root = tmp_path / ".cad-agent" / "reviews" / digest
    tampered_view = review_root / "views" / "z_positive.png"
    tampered_view.write_bytes(b"not-a-png-at-all")
    assert compute_model_sha256(tmp_path) is not None
    tool = ImageTool(tmp_path)
    # Only the tampered view was requested -> overall failure.
    with pytest.raises(ValueError, match="None of the requested views"):
        tool.get_view_images([{"view": "z_positive"}])
    # Mix in a clean view: tampered entry is dropped, valid one survives.
    result = tool.get_view_images(
        [{"view": "z_positive"}, {"view": "x_positive"}]
    )
    assert [item["view"] for item in result["images"]] == ["x_positive"]
    # Sanity: the model on disk still matches the review dir (no edit happened).
    assert (tmp_path / MODEL_FILENAME).read_text(encoding="utf-8")


def test_dispatcher_routes_get_view_images_to_image_tool(tmp_path: Path) -> None:
    """``dispatch`` returns ``(result, False)`` for ``get_view_images``."""
    _seed_review_dir(tmp_path)
    from agent.core import ProjectTools
    from agent.dispatcher import dispatch

    tools = ProjectTools(tmp_path, lambda *args: None)
    _result, waiting = dispatch(
        tools,
        "test",
        "get_view_images",
        {"images": [{"view": "z_positive"}]},
    )
    assert waiting is False
    # The raw result is the dict that ``tool_success`` will wrap.
    assert isinstance(_result, dict)
    assert _result["review_sha256"]
    assert _result["images"][0]["view"] == "z_positive"


def test_get_view_images_failure_envelope_via_dispatcher(tmp_path: Path) -> None:
    """Errors raised by ImageTool must surface as a structured ``ok:false`` envelope."""
    from agent.core import ProjectTools
    from agent.dispatcher import dispatch
    from agent.tool_results import failure as tool_failure

    tools = ProjectTools(tmp_path, lambda *args: None)
    with pytest.raises(ValueError, match="cad_build_and_verify"):
        dispatch(
            tools,
            "test",
            "get_view_images",
            {"images": [{"view": "z_positive"}]},
        )
    # ``tool_failure`` produces the same envelope shape ``dispatch`` would
    # emit on the failure branch of ``process_tool_call``.
    envelope = json.loads(tool_failure("get_view_images", ValueError("boom")))
    assert envelope["ok"] is False
    assert envelope["tool"] == "get_view_images"
    assert envelope["error"]["code"] == "VALIDATION_ERROR"


def test_build_view_image_multimodal_content_appends_image_parts(tmp_path: Path) -> None:
    """The dispatcher helper attaches one ``image_url`` per returned path."""
    from agent.tool_results import build_view_image_multimodal_content

    digest, _view_sha = _seed_review_dir(tmp_path)
    success_json = json.dumps(
        {
            "ok": True,
            "tool": "get_view_images",
            "data": {
                "review_sha256": digest,
                "images": [
                    {
                        "view": "z_positive",
                        "cropped": False,
                        "path": "views/z_positive.png",
                        "sha256": _view_sha["z_positive"],
                        "width": 512,
                        "height": 512,
                    },
                ],
            },
        }
    )
    multimodal = build_view_image_multimodal_content(success_json, tmp_path)
    assert multimodal is not None
    parts = multimodal["content"]
    assert parts[0]["type"] == "text"
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert len(multimodal["image_paths"]) == 1


def test_build_view_image_multimodal_content_returns_none_on_failure_envelope(
    tmp_path: Path,
) -> None:
    """Failure envelopes must never produce a multimodal payload."""
    from agent.tool_results import build_view_image_multimodal_content

    failure_json = json.dumps(
        {
            "ok": False,
            "tool": "get_view_images",
            "error": {"code": "VALIDATION_ERROR", "message": "boom"},
        }
    )
    assert build_view_image_multimodal_content(failure_json, tmp_path) is None


def test_get_view_images_dispatcher_preserves_history_prefix_stability(tmp_path: Path) -> None:
    """Appending a get_view_images result must not mutate earlier messages.

    Core guarantee from PLAN_get_view_images.md §4: images are only ever
    appended at the tail of ``conversation.jsonl``. Pre-existing entries
    remain byte-identical after the tool call, and the new images appear
    only after ``relocate_tool_images`` rewrites them into a trailing
    ``role: user`` message.
    """
    from agent.conversation import ConversationStore
    from agent.core import ProjectTools
    from agent.dispatcher import dispatch, process_tool_call
    from agent.llm_base import relocate_tool_images

    _seed_review_dir(tmp_path)
    tools = ProjectTools(tmp_path, lambda *args: None)
    # The system prompt is injected into ``messages`` at LLM-call time
    # (it is *not* persisted in ``conversation.jsonl`` — see
    # ``ConversationStore._KEPT_ROLES``). Mirror the real agent loop by
    # seeding the system message in-memory only.
    seed_system = {"role": "system", "content": "system-prompt"}
    seed_user = {"role": "user", "content": "Look at my plate"}
    seed_assistant = {
        "role": "assistant",
        "content": "I'll request a view.",
        "tool_calls": [
            {
                "id": "call_view",
                "type": "function",
                "function": {"name": "get_view_images", "arguments": "{}"},
            }
        ],
    }
    # Persist only the user/assistant turns (system is regenerated each call).
    ConversationStore.append(tmp_path, seed_user)
    ConversationStore.append(tmp_path, seed_assistant)

    pre_history = ConversationStore.load(tmp_path)
    # In-memory snapshot reflects what the agent loop actually carries:
    # system (in-memory only) + persisted history.
    pre_snapshot = [deepcopy(seed_system), *pre_history]

    messages: list[dict] = [deepcopy(item) for item in pre_snapshot]
    dispatch_args = {"images": [{"view": "z_positive"}, {"view": "x_positive"}]}
    captured: list[dict] = []
    append_calls: list[dict] = []
    process_tool_call(
        tools=tools,
        project="test",
        project_dir=tmp_path,
        call={
            "id": "call_view",
            "function": {
                "name": "get_view_images",
                "arguments": json.dumps(dispatch_args),
            },
        },
        cad_fix_required=False,
        prev_preview_id=None,
        cad_error=None,
        messages=messages,
        publish=lambda event, payload: captured.append({"event": event, **payload}),
        register_preview=lambda *_: "preview-id",
        append_message=lambda _dir, msg: append_calls.append(deepcopy(msg)),
    )

    # Pre-existing messages must still be byte-identical to the snapshot
    # (the dispatcher mutated ``messages`` in-place, but only appended at
    # the tail).
    for index, pre in enumerate(pre_snapshot):
        assert messages[index] == pre, (
            f"Pre-existing message at index {index} was mutated: "
            f"role={pre.get('role')}"
        )
    # The new tail entry is a tool message carrying the JSON envelope.
    new_tail = messages[-1]
    assert new_tail["role"] == "tool"
    assert new_tail["tool_call_id"] == "call_view"
    content = new_tail["content"]
    assert isinstance(content, list)
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(image_parts) == 2

    # ``relocate_tool_images`` (the wire-payload stage) must move those
    # images into a trailing user message without touching any earlier
    # message (including the system prompt and the assistant turn).
    relocated = relocate_tool_images(messages)
    assert relocated[0] == seed_system
    assert relocated[1] == seed_user
    # The assistant message survives unchanged (it's not a tool message).
    assert relocated[2] == seed_assistant
    # Tool message keeps its text but loses the image parts.
    assert relocated[3]["role"] == "tool"
    assert isinstance(relocated[3]["content"], str)
    # Trailing user message carries the relocated images.
    tail_user = relocated[-1]
    assert tail_user["role"] == "user"
    tail_images = [p for p in tail_user["content"] if p.get("type") == "image_url"]
    assert len(tail_images) == 2

    # The persisted ``conversation.jsonl`` must include the tool message
    # with its image_url parts — the on-disk record is the permanent
    # memory, the relocated version is just the wire payload.
    # The dispatcher captures append calls in ``append_calls``; replay the
    # captured tool message through ``ConversationStore.append`` so the
    # on-disk record mirrors what the real agent loop would persist.
    persisted_tool_msg = next(msg for msg in append_calls if msg.get("role") == "tool")
    ConversationStore.append(tmp_path, persisted_tool_msg)
    post_history = ConversationStore.load(tmp_path)
    # Pre-existing persisted entries are untouched (system message is not
    # persisted at all — only user/assistant/tool survive the load filter).
    assert post_history[:2] == pre_history
    persisted_tool = post_history[2]
    assert persisted_tool["role"] == "tool"
    persisted_images = [
        p for p in persisted_tool["content"] if p.get("type") == "image_url"
    ]
    assert len(persisted_images) == 2


def test_get_view_images_falls_back_to_text_when_provider_rejects_images(
    tmp_path: Path,
) -> None:
    """``without_images`` strips the trailing image parts but keeps stored history.

    Even when the provider rejects the inline image, the tool's success
    envelope is already in ``conversation.jsonl`` and will be re-sent
    verbatim on the next request.
    """
    from agent.conversation import ConversationStore
    from agent.core import ProjectTools
    from agent.dispatcher import dispatch, process_tool_call
    from agent.llm_base import TOOL_IMAGE_PROMPT, without_images

    _seed_review_dir(tmp_path)
    tools = ProjectTools(tmp_path, lambda *args: None)
    captured: list[dict] = []
    append_calls: list[dict] = []
    messages: list[dict] = []
    process_tool_call(
        tools=tools,
        project="test",
        project_dir=tmp_path,
        call={
            "id": "call_view",
            "function": {
                "name": "get_view_images",
                "arguments": json.dumps(
                    {"images": [{"view": "isometric_positive"}]}
                ),
            },
        },
        cad_fix_required=False,
        prev_preview_id=None,
        cad_error=None,
        messages=messages,
        publish=lambda event, payload: captured.append({"event": event, **payload}),
        register_preview=lambda *_: "preview-id",
        append_message=lambda _dir, msg: append_calls.append(deepcopy(msg)),
    )

    # Strip the trailing image parts as the provider's image rejection
    # would; the tool text must survive so the model still sees evidence
    # the call succeeded. ``without_images`` keeps the list-shaped content
    # with just the text part intact (the str conversion happens later in
    # ``relocate_tool_images``).
    stripped, removed = without_images(messages)
    assert removed is True
    tool_msg = next(msg for msg in stripped if msg.get("role") == "tool")
    text_parts = [
        part for part in tool_msg["content"]
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    assert len(text_parts) == 1
    payload = json.loads(text_parts[0]["text"])
    assert payload["ok"] is True
    # The injected TOOL_IMAGE_PROMPT user message should have been dropped
    # along with the rejected images so the model isn't told to look at
    # an attachment that isn't there.
    assert not any(
        msg.get("role") == "user"
        and isinstance(msg.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("text") == TOOL_IMAGE_PROMPT
            for part in msg["content"]
        )
        for msg in stripped
    )

    # The stored history still carries the images — only the wire payload
    # was stripped.
    persisted_tool_msg = next(msg for msg in append_calls if msg.get("role") == "tool")
    ConversationStore.append(tmp_path, persisted_tool_msg)
    history = ConversationStore.load(tmp_path)
    tool_history = next(msg for msg in history if msg.get("role") == "tool")
    history_images = [
        p for p in tool_history["content"] if p.get("type") == "image_url"
    ]
    assert len(history_images) == 1

