"""Focused tests for the History/AI-Activity fixes in
docs/HISTORY_ACTIVITY_IMPLEMENTATION_PLAN.md.

Each test exercises the *observable* behaviour of the change it pins rather
than the literal source code:

- Backend: a Flask test client drives the real `/api/projects/<name>/history`
  endpoint with seeded conversation entries and asserts the response shape.
- Frontend: ``static/js/app.js`` is loaded into a Node ``vm`` context with a
  small stubbed DOM so individual functions (``normalizeHistoryContent``,
  ``activityLabel``, ``markActivityRecovered``, ``loadHistory``, the SSE
  streaming-card lifecycle) can be called and their behaviour asserted.

Behavioural failures surface as assertion errors with concrete values, not as
"this token does not appear in the source".
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent.settings import Settings
from app import _redact_history_event, create_app

JS_SOURCE = Path(__file__).resolve().parents[1].joinpath("static/js/app.js")


# ---------------------------------------------------------------------------
# Node-backed runtime helpers
# ---------------------------------------------------------------------------


_NODE = shutil.which("node")
assert _NODE is not None, "node binary required to exercise frontend helpers"


def _run_js(code: str) -> str:
    """Run a Node script (plain CJS, not ESM) and return stdout. Stderr is
    included in the AssertionError on non-zero exit. Writing to a temp file
    avoids quoting pitfalls when the embedded JS contains backtick template
    literals (which would collide with ``--input-type=module`` parsing)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(code)
        script_path = handle.name
    try:
        result = subprocess.run(
            [_NODE, script_path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)
    if result.returncode != 0:
        raise AssertionError(
            f"node exited {result.returncode}\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
        )
    return result.stdout


def _extract_js(name: str, *, kind: str = "function") -> str:
    """Return the source of a top-level ``function name(...) { ... }`` or
    ``const name = { ... }`` literal extracted from ``static/js/app.js``.
    ``kind`` is ``"function"`` for `function name(…)` or ``"object"`` for
    `const name = { … }`. Brace matching is depth-tracked so nested blocks
    and regex literals do not throw it off."""
    if kind == "function":
        pattern = rf"(?:async\s+)?function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{"
    elif kind == "object":
        pattern = rf"(?:const|let|var)\s+{re.escape(name)}\s*=\s*\{{"
    else:
        raise ValueError(kind)
    text = JS_SOURCE.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if not match:
        raise AssertionError(f"{kind} {name!r} not found in static/js/app.js")
    start = match.start()
    i = match.end() - 1
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    raise AssertionError(f"unterminated {kind} {name!r}")


def _eval_helper(bindings: str, body: str) -> str:
    """Concatenate ``bindings`` (stubbed DOM/test setup) and ``body`` (the
    assertions driving the helpers under test), then ``eval`` the combined
    script as plain CJS so backtick template literals in either string are
    passed through unchanged."""
    code = bindings + "\n" + body + "\n"
    return _run_js(code)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_client(tmp_path: Path):
    settings = Settings(
        tmp_path / "projects",
        "https://example.test",
        "test-model",
        1,
        "127.0.0.1",
        5000,
    )
    return settings, create_app(settings).test_client()


def _seed_history(project_dir: Path, *entries: dict) -> None:
    log_path = project_dir / "conversation.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for entry in entries:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Issue 4 (backend): History responses redact inline image data URLs.
# ---------------------------------------------------------------------------


def test_history_response_redacts_inline_image_data_urls(tmp_path: Path):
    """Inline base64 image payloads never reach the wire through /history."""
    settings, client = _create_client(tmp_path)
    client.post("/api/projects/new", json={"name": "demo"})
    project_dir = settings.workspace_root / "demo"

    base64 = "A" * 384
    _seed_history(
        project_dir,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Use this sketch"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64}},
            ],
        },
    )

    events = client.get("/api/projects/demo/history").get_json()["events"]
    assert len(events) == 1
    content = events[0]["content"]
    # Text part still arrives.
    text_values = [part["text"] for part in content if part.get("type") == "text"]
    assert "Use this sketch" in text_values
    # image_url parts are gone and a placeholder takes their place.
    assert not any(part.get("type") == "image_url" for part in content)
    assert any("Reference image" in t for t in text_values)
    # Base64 payload and the data URL prefix never appear in the response.
    body = client.get("/api/projects/demo/history").data.decode()
    assert "data:image/png;base64," not in body
    assert base64 not in body


def test_history_response_redacts_only_user_role_list_content(tmp_path: Path):
    """The redactor must be scoped to user-role list content; assistant
    content passes through so the LLM-facing format is unchanged."""
    settings, client = _create_client(tmp_path)
    client.post("/api/projects/new", json={"name": "demo"})
    project_dir = settings.workspace_root / "demo"

    _seed_history(
        project_dir,
        {"role": "user", "content": "plain text"},
        # Assistant content can contain anything; it is not redacted.
        {"role": "assistant", "content": "Here is a base64 ref: data:image/png;base64,XYZ"},
        # String user content is not redacted either (only list content).
        {"role": "user", "content": "another plain string"},
    )

    events = client.get("/api/projects/demo/history").get_json()["events"]
    user_events = [e for e in events if e.get("role") == "user"]
    assistant_events = [e for e in events if e.get("role") == "assistant"]
    assert all(e["content"] == "plain text" or e["content"] == "another plain string"
               for e in user_events)
    assert all("data:image/png;base64,XYZ" in e["content"] for e in assistant_events)


def test_redact_helper_is_a_pure_function():
    """``_redact_history_event`` must be importable and round-trip the same
    payload through a no-op when the input is not user/list content."""
    untouched = _redact_history_event({"role": "assistant", "content": [{"type": "text", "text": "hi"}]})
    assert untouched == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    plain = _redact_history_event({"role": "user", "content": "hi"})
    assert plain == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# Issue 4 (frontend): normalizeHistoryContent handles list content.
# ---------------------------------------------------------------------------


def test_normalize_history_content_handles_list_and_string_content():
    """Structured list content is normalised; plain strings pass through."""
    bindings = (
        _extract_js("normalizeHistoryContent")
    )
    body = (
        "process.stdout.write(JSON.stringify({\n"
        "  plain: normalizeHistoryContent('hello'),\n"
        "  arr: normalizeHistoryContent([\n"
        "    {type: 'text', text: 'Use this'},\n"
        "    {type: 'image_url', image_url: {url: 'data:image/png;base64,XYZ'}},\n"
        "  ]),\n"
        "  multi: normalizeHistoryContent([\n"
        "    {type: 'image_url'},\n"
        "    {type: 'image_url'},\n"
        "    {type: 'text', text: 'after'},\n"
        "  ]),\n"
        "  bad: normalizeHistoryContent(null),\n"
        "  empty: normalizeHistoryContent(''),\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["plain"] == "hello"
    # Order: text part first, image placeholder after.
    assert out["arr"] == "Use this\n[Reference image 1]"
    assert out["multi"] == "[Reference image 1]\n[Reference image 2]\nafter"
    # Defensive defaults — null and empty string return empty text.
    assert out["bad"] == ""
    assert out["empty"] == ""


def test_normalize_history_content_is_defensive_against_malformed_parts():
    """Edge case: the helper must never throw on weird shapes. Garbage parts
    are skipped, unicode text round-trips, and a long stream of images gets
    monotonically numbered placeholders."""
    bindings = _extract_js("normalizeHistoryContent")
    body = (
        "var out = {\n"
        # 1. Empty array → empty string.
        "  empty_arr: normalizeHistoryContent([]),\n"
        # 2. Only malformed parts (nulls, primitives, missing type) → empty.
        "  only_garbage: normalizeHistoryContent([\n"
        "    null, undefined, 42, 'string-part', {no_type: 1},\n"
        "    {type: 'unknown'},\n"
        "  ]),\n"
        # 3. Unicode and special characters in text pass through verbatim.
        "  unicode: normalizeHistoryContent([\n"
        "    {type: 'text', text: 'ünïcödé ✓ <script>alert(1)</script>'},\n"
        "  ]),\n"
        # 4. Mixed valid text + multiple images preserves order and numbering.
        "  long: normalizeHistoryContent([\n"
        "    {type: 'text', text: 'first'},\n"
        + ", ".join(["{type: 'image_url'}"] * 5) + ",\n"
        "    {type: 'text', text: 'last'},\n"
        "  ]),\n"
        "};\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    out = json.loads(_eval_helper(bindings, body))

    assert out["empty_arr"] == ""
    assert out["only_garbage"] == ""
    assert "ünïcödé" in out["unicode"]
    assert "✓" in out["unicode"]
    # Sanitization is a separate concern; we only assert the helper does not
    # mangle content here.
    assert "<script>" in out["unicode"]
    # The 5-image sequence must be numbered 1..5 in order.
    expected_long = "first\n[Reference image 1]\n[Reference image 2]\n[Reference image 3]\n[Reference image 4]\n[Reference image 5]\nlast"
    assert out["long"] == expected_long


def test_redact_history_event_handles_unicode_image_data():
    """Edge case: the backend redactor must process unicode/oversized image
    payloads without crashing and replace them with the standard placeholder.

    The redaction is structural (it looks at ``type`` not at the payload
    content), so this test pins two real risks: (1) large unicode payloads
    must not blow the stack or exceed any implicit size limits, and (2) the
    image counter must still increment to 1 in a single-image event.
    """
    # Arrange: a multi-kilobyte base64 payload with embedded unicode
    # (simulating file names, EXIF strings, etc.).
    base64 = ("ünïcödé" * 500).encode("utf-8").hex()
    long_payload = "data:image/png;base64," + base64

    event = {
        "role": "user",
        "content": [
            {"type": "text", "text": "see attached sketch"},
            {"type": "image_url", "image_url": {"url": long_payload}},
        ],
    }

    # Act: redactor processes the unicode-heavy payload.
    redacted = _redact_history_event(event)

    # Assert: the text part survives and the image is replaced by the
    # numbered placeholder. The unicode substring must not leak anywhere.
    parts = redacted["content"]
    assert len(parts) == 2
    assert parts[0] == {"type": "text", "text": "see attached sketch"}
    assert parts[1] == {"type": "text", "text": "[Reference image 1]"}
    serialised = " ".join(str(p) for p in parts)
    assert "base64" not in serialised
    assert "ünïcödé" not in serialised


# ---------------------------------------------------------------------------
# Issue 6: Activity labels map tool schema names to user-facing wording and
# fall back to a sanitized human string for unknown tools.
# ---------------------------------------------------------------------------


def test_activity_label_strips_underscores_for_known_tools():
    """Every known tool/status label must render as a human phrase without
    underscores — the visible-in-UI contract that prevents tool names from
    leaking into the activity drawer.
    """
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    # Drive the helper over every key in activityLabels (the source of truth).
    body = (
        "var out = [];\n"
        "for (const name of Object.keys(activityLabels)) {\n"
        "  out.push([name, activityLabel(name)]);\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    rendered = json.loads(_eval_helper(bindings, body))

    assert rendered, "activityLabels must declare at least one entry"

    for name, label in rendered:
        assert isinstance(label, str) and label, (
            f"{name!r} produced a non-string/empty label"
        )
        assert "_" not in label, (
            f"{name!r} label {label!r} still contains underscores"
        )


def test_activity_label_groups_share_the_same_wording():
    """The activity panel renders tool families as a single phrase so users
    do not see three different rows when the agent edits the file three ways.
    Pinned: write_file/edit_file all share 'Updating model'.
    """
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    body = (
        "process.stdout.write(JSON.stringify({\n"
        "  write_file: activityLabel('write_file'),\n"
        "  edit_file: activityLabel('edit_file'),\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))

    assert out["write_file"] == out["edit_file"]


@pytest.mark.parametrize(
    ("js_input", "expected"),
    [
        # Snake-case unknown name: the fallback ``String(value || 'Activity')``
        # replaces ``_`` and ``-`` with spaces.
        pytest.param("'future_tool_name'", "future tool name", id="snake-case-fallback"),
        # Empty string is falsy in JS, so the ``||`` branch returns 'Activity'.
        pytest.param("''", "Activity", id="empty-string"),
        # null is falsy too; same 'Activity' fallback.
        pytest.param("null", "Activity", id="js-null"),
        # undefined is falsy too; same 'Activity' fallback.
        pytest.param("undefined", "Activity", id="js-undefined"),
    ],
)
def test_activity_label_fallback_sanitizes_unknown_inputs(js_input, expected):
    """Unknown / empty / null tool names must fall back to a sanitized label
    rather than leak underscores or render the raw tool name in the UI."""
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    body = (
        f"process.stdout.write(JSON.stringify({{"
        f"label: activityLabel({js_input})"
        f"}}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))

    assert out["label"] == expected


def test_activity_detail_hides_model_facing_json_envelopes():
    bindings = _extract_js("activityDetail")
    body = """
const success = JSON.stringify({ok: true, tool: 'write_file', data: 'Wrote model.py.'});
const failure = JSON.stringify({ok: false, tool: 'cad_build_and_verify', error: {message: 'Bad edge.'}});
const build = JSON.stringify({ok: true, tool: 'cad_build_and_verify', data: {metrics: {solid_count: 1}}});
process.stdout.write(JSON.stringify({
  success: activityDetail('write_file', 'completed', success),
  failure: activityDetail('cad_build_and_verify', 'error', failure),
  build: activityDetail('cad_build_and_verify', 'completed', build),
}));
"""
    out = json.loads(_eval_helper(bindings, body))
    assert out == {
        "success": "Wrote model.py.",
        "failure": "Bad edge.",
        "build": "Model built and review artifacts created.",
    }


def test_activity_detail_prefers_cad_build_summary_when_present():
    """``cad_build_and_verify`` now returns a compact ``summary`` string. The
    activity drawer should surface that one-liner instead of the legacy fixed
    phrase so render-less iterations read as "metrics-only" rather than
    "Model built and review artifacts created." which would mislead the user.
    """
    bindings = _extract_js("activityDetail")
    body = """
const full = JSON.stringify({ok: true, tool: 'cad_build_and_verify', data: {
  summary: 'Solid 1 (valid); bbox 10.0x20.0x30.0 mm; 6.0 cm3; 6 features; with render.',
  metrics: {solid_count: 1},
  preview: 'preview.stl',
}});
const minimal = JSON.stringify({ok: true, tool: 'cad_build_and_verify', data: {
  summary: 'Solid 2 (INVALID); bbox 0.0x0.0x0.0 mm; 0.0 cm3; 0 features; metrics-only.',
  metrics: {solid_count: 2},
  preview: 'preview.stl',
  render: null,
}});
process.stdout.write(JSON.stringify({
  full: activityDetail('cad_build_and_verify', 'completed', full),
  minimal: activityDetail('cad_build_and_verify', 'completed', minimal),
}));
"""
    out = json.loads(_eval_helper(bindings, body))
    assert out["full"].startswith("Solid 1 (valid)")
    assert "with render" in out["full"]
    assert out["minimal"].startswith("Solid 2 (INVALID)")
    assert "metrics-only" in out["minimal"]
    # The legacy phrase must NOT leak through when a summary is provided.
    assert "Model built and review artifacts created." not in out["full"]
    assert "Model built and review artifacts created." not in out["minimal"]


# ---------------------------------------------------------------------------
# cad_screenshot / cad_review split: phase labels must be discoverable so the
# UI activity drawer reflects the new tool_status events and the terminal
# ``screenshot_updated`` / ``review_updated`` events wired into the SSE
# handlers map. Pinned by docs/PLAN_cad_screenshot_review_split.md §6 Stage 7.
# ---------------------------------------------------------------------------


def test_activity_labels_cover_split_screenshot_review_keys():
    """The split tools publish ``cad_screenshot`` / ``cad_review`` and the
    intermediate ``rendering_subset`` / ``screenshot_auto`` statuses that the
    activity panel must render as a non-empty human phrase — without raw
    underscores or schema names leaking through."""
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    body = "process.stdout.write(JSON.stringify({});\n".join([
        f"  {key}: activityLabel({key!r}),\n"
        for key in (
            "cad_screenshot",
            "cad_review",
            "rendering_subset",
            "screenshot_auto",
            "reviewing",
            "pass",
            "fail",
            "inconclusive",
            "limit_reached",
        )
    ])
    body = (
        "process.stdout.write(JSON.stringify({"
        + body.replace("process.stdout.write(JSON.stringify({});\n", "")
        + "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))

    for key, label in out.items():
        assert isinstance(label, str) and label, (
            f"{key!r} label must be a non-empty string, got {label!r}"
        )
        assert "_" not in label, (
            f"{key!r} label {label!r} leaks an underscore into the UI"
        )

    # Review verdicts must read like user-facing words, not raw schema values.
    assert out["pass"] == "Pass"
    assert out["fail"] == "Fail"
    assert out["inconclusive"] == "Inconclusive"


def test_sse_handlers_wire_screenshot_and_review_updated():
    """The SSE ``handlers`` map inside ``connectStream`` must register both
    ``screenshot_updated`` and ``review_updated`` so the activity pill updates
    when the split tools publish their terminal events. The handlers live
    inline inside ``connectStream`` rather than as a top-level binding, so we
    extract the function and assert the map keys."""
    text = JS_SOURCE.read_text(encoding="utf-8")
    func_match = re.search(
        r"function connectStream\(\) \{(.*?)\n\}\n",
        text,
        re.DOTALL,
    )
    assert func_match, "connectStream not found in static/js/app.js"
    func_body = func_match.group(1)
    handlers_match = re.search(
        r"const handlers = \{(.*?)\n  \};",
        func_body,
        re.DOTALL,
    )
    assert handlers_match, "handlers map literal not found in connectStream"

    body = handlers_match.group(1)
    # The map must register both new terminal events so the UI can react.
    assert "screenshot_updated:" in body, (
        "screenshot_updated handler missing from SSE handlers map"
    )
    assert "review_updated:" in body, (
        "review_updated handler missing from SSE handlers map"
    )


def test_running_filter_includes_new_split_statuses():
    """``updateActivitySummary`` and ``markActivityRecovered`` decide which
    activity rows are still 'in flight' by checking against a hard-coded
    list. The split tools' intermediate statuses (``rendering_subset``,
    ``screenshot_auto``) must be present so the activity pill keeps showing
    "Reviewing build · Running" until the terminal event arrives."""
    for function_name in ("updateActivitySummary", "markActivityRecovered"):
        func_body = _extract_js(function_name)
        assert "'rendering_subset'" in func_body, (
            f"{function_name} must include 'rendering_subset' in its running "
            "filter so cad_screenshot in-flight rows stay visible"
        )
        assert "'screenshot_auto'" in func_body, (
            f"{function_name} must include 'screenshot_auto' in its running "
            "filter so cad_review auto-screenshot rows stay visible"
        )


# ---------------------------------------------------------------------------
# Issue 5: markActivityRecovered must leave error rows visible.
# ---------------------------------------------------------------------------


def test_mark_activity_recovered_leaves_error_rows_visible():
    """A successful final turn flips running rows to completed but never
    hides error rows."""
    bindings = (
        "var CSS = {escape: (s) => String(s)};\n"
        "var activityItems = new Map();\n"
        "var activityList = {\n"
        "  rows: new Map(),\n"
        "  querySelector(sel) {\n"
        "    var m = sel.match(/data-call-id=\"([^\"]+)\"/);\n"
        "    return m ? this.rows.get(m[1]) : null;\n"
        "  },\n"
        "};\n"
        "var updateActivitySummary = function() {};\n"
        "function makeRow(callId, status) {\n"
        "  var row = {\n"
        "    dataset: {callId: callId, status: status},\n"
        "    hidden: false,\n"
        "    querySelector(sel) {\n"
        "      if (sel === '.activity-state') {\n"
        "        return { set textContent(v) { this._t = v; }, get textContent() { return this._t; } };\n"
        "      }\n"
        "      return null;\n"
        "    },\n"
        "  };\n"
        "  activityList.rows.set(callId, row);\n"
        "  return row;\n"
        "}\n"
        + _extract_js("markActivityRecovered")
    )
    body = (
        # Running row should transition to completed.
        "activityItems.set('ok-1', {callId: 'ok-1', tool: 'cad_build_and_verify', status: 'running'});\n"
        "makeRow('ok-1', 'running');\n"
        # Error row must stay error and stay visible.
        "activityItems.set('err-1', {callId: 'err-1', tool: 'write_file', status: 'error'});\n"
        "makeRow('err-1', 'error');\n"
        # Already-failed row at error status must also remain untouched.
        "activityItems.set('err-2', {callId: 'err-2', tool: 'write_file', status: 'error'});\n"
        "makeRow('err-2', 'error');\n"
        "markActivityRecovered();\n"
        "process.stdout.write(JSON.stringify({\n"
        "  ok_status: activityItems.get('ok-1').status,\n"
        "  ok_row_status: activityList.rows.get('ok-1').dataset.status,\n"
        "  err1_status: activityItems.get('err-1').status,\n"
        "  err1_hidden: activityList.rows.get('err-1').hidden,\n"
        "  err2_status: activityItems.get('err-2').status,\n"
        "  err2_hidden: activityList.rows.get('err-2').hidden,\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["ok_status"] == "completed"
    assert out["ok_row_status"] == "completed"
    assert out["err1_status"] == "error"
    assert out["err1_hidden"] is False
    assert out["err2_status"] == "error"
    assert out["err2_hidden"] is False


# ---------------------------------------------------------------------------
# Issue 3: agent_content_delta appends to the streaming card; agent_message
# reuses the same card without duplicating.
# ---------------------------------------------------------------------------


def _streaming_card_bindings():
    return (
        # Stub the DOM bits the streaming functions touch.
        "var crypto = { randomUUID: () => 'fixed-uuid' };\n"
        # document.createElement returns an object that records what was
        # appended. The streaming helpers set innerHTML and read .querySelector
        # only inside sanitizeHTML/marked, which we never call from these tests.
        "var document = {\n"
        "  createElement(tag) {\n"
        "    return {\n"
        "      tagName: tag,\n"
        "      className: '',\n"
        "      dataset: {},\n"
        "      innerHTML: '',\n"
        "      querySelector: function() { return null; },\n"
        "      _text: '',\n"
        "      remove: function() { this.removed = true; },\n"
        "    };\n"
        "  }\n"
        "};\n"
        "var feed = { items: [], appendChild(el) { this.items.push(el); return el; } };\n"
        "var renderAgentContent = function(card, text) { card._rendered = text; };\n"
        "var addMessage = function(text, type) {\n"
        "  var item = {text: text, type: type, dataset: {}};\n"
        "  feed.appendChild(item);\n"
        "  return item;\n"
        "};\n"
        "var markActivityRecovered = function() {};\n"
        "var setThinking = function() {};\n"
        # Inlined module-level state the streaming helpers read from
        # closure; using var so it lands on the script's global object.
        "var streamingMessages = new Map();\n"
        "var pendingFinalCard = null;\n"
        + _extract_js("startStreamingCard")
        + "\n"
        + _extract_js("appendStreamingDelta")
        + "\n"
        + _extract_js("finalizeStreamingCard")
        + "\n"
        + _extract_js("discardEmptyStreamingCard")
    )


def test_streaming_delta_accumulates_into_one_card():
    """Sequential content deltas share one card and accumulate in order."""
    bindings = _streaming_card_bindings()
    body = (
        "appendStreamingDelta('m-1', 'Hello, ');\n"
        "appendStreamingDelta('m-1', 'world!');\n"
        "var card = startStreamingCard('m-1');\n"
        "process.stdout.write(JSON.stringify({\n"
        "  count: feed.items.length,\n"
        "  rendered: card._rendered,\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["count"] == 1
    assert out["rendered"] == "Hello, world!"


def test_finalize_keeps_pending_card_for_agent_message():
    """agent_stream_end finalizes the card text without clearing the
    pending reference, so the subsequent agent_message can reuse it."""
    bindings = _streaming_card_bindings()
    body = (
        "appendStreamingDelta('m-1', 'streamed ');\n"
        "var card = startStreamingCard('m-1');\n"
        "finalizeStreamingCard('m-1', 'streamed final');\n"
        "process.stdout.write(JSON.stringify({\n"
        "  rendered: card._rendered,\n"
        "  raw: card.dataset.raw,\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["rendered"] == "streamed final"
    assert out["raw"] == "streamed final"


def test_discard_empty_streaming_card_removes_tool_only_card():
    """Tool-only turns (empty assistant content) must remove the empty card
    so the chat feed does not grow with empty placeholders."""
    bindings = _streaming_card_bindings()
    body = (
        "var card = startStreamingCard('m-2');\n"
        # discardEmptyStreamingCard should remove the empty card.
        "discardEmptyStreamingCard('m-2');\n"
        "process.stdout.write(JSON.stringify({\n"
        "  removed: !!card.removed,\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["removed"] is True


# ---------------------------------------------------------------------------
# Issue 1: loadHistory drawer mode never mutates the main feed.
# ---------------------------------------------------------------------------


def test_load_history_drawer_mode_does_not_mutate_main_feed():
    """When called with target='drawer', loadHistory populates the drawer
    container and leaves the main feed and activity list untouched."""
    bindings = (
        # DOM stubs.
        "var feed = { items: [], replaceChildren() { this.items.length = 0; } };\n"
        "var historyContent = { items: [], replaceChildren() { this.items.length = 0; } };\n"
        "var questionArea = { replaceChildren() {} };\n"
        "var activityPanel = { hidden: false };\n"
        "var activityList = { replaceChildren() {} };\n"
        "var activityItems = new Map();\n"
        "var toolMessages = new Map();\n"
        "var showInfoMessages = true;\n"
        "var currentProject = 'demo';\n"
        "var api = async function() { return { events: [\n"
        "  {role: 'user', content: 'hi'},\n"
        "  {role: 'assistant', content: 'hello there'},\n"
        "]}; };\n"
        # addMessage records which target it was called with.
        "function addMessage(text, type, opts) {\n"
        "  var t = (opts && opts.target) || feed;\n"
        "  var item = { text: text, type: type, target: t };\n"
        "  t.items.push(item);\n"
        "  return item;\n"
        "}\n"
        "function normalizeHistoryContent(c) { return typeof c === 'string' ? c : ''; }\n"
        "function addInfoMessage() {}\n"
        "function clearActivity() { activityItems.clear(); toolMessages.clear(); }\n"
        "var chatEmptyTpl = null;\n"
        + _extract_js("loadHistory")
    )
    body = (
        # Drive loadHistory in drawer mode.
        "loadHistory('demo', {target: 'drawer'}).then(() => {\n"
        "  process.stdout.write(JSON.stringify({\n"
        "    drawer_items: historyContent.items.map(i => ({text: i.text, type: i.type})),\n"
        "    feed_items: feed.items,\n"
        "  }));\n"
        "});\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    drawer_items = out["drawer_items"]
    feed_items = out["feed_items"]
    # The drawer received both events.
    assert [i["text"] for i in drawer_items] == ["hi", "hello there"]
    assert [i["type"] for i in drawer_items] == ["user", "agent"]
    # The main feed and activity state were not touched.
    assert feed_items == []


def test_load_history_main_mode_clears_feed_and_renders_events():
    """When called without target='drawer', loadHistory clears the main feed
    and renders user/assistant/error events into it."""
    bindings = (
        "var feed = { items: [], replaceChildren() { this.items.length = 0; } };\n"
        "var historyContent = { items: [], replaceChildren() {} };\n"
        "var questionArea = { replaceChildren() {} };\n"
        "var activityPanel = { hidden: false };\n"
        "var activityList = { replaceChildren() {} };\n"
        "var activityItems = new Map();\n"
        "var toolMessages = new Map();\n"
        "var showInfoMessages = true;\n"
        "var api = async function() { return { events: [\n"
        "  {role: 'user', content: 'first message'},\n"
        "  {type: 'agent_error', data: {message: 'boom'}},\n"
        "]}; };\n"
        "function addMessage(text, type, opts) {\n"
        "  var t = (opts && opts.target) || feed;\n"
        "  var item = { text: text, type: type };\n"
        "  t.items.push(item);\n"
        "  return item;\n"
        "}\n"
        "function normalizeHistoryContent(c) { return typeof c === 'string' ? c : ''; }\n"
        "function addInfoMessage() {}\n"
        "function clearActivity() { activityItems.clear(); toolMessages.clear(); }\n"
        "var chatEmptyTpl = null;\n"
        + _extract_js("loadHistory")
    )
    body = (
        "loadHistory('demo').then(() => {\n"
        "  process.stdout.write(JSON.stringify(feed.items.map(i => ({text: i.text, type: i.type}))));\n"
        "});\n"
    )
    items = json.loads(_eval_helper(bindings, body))
    assert items == [
        {"text": "first message", "type": "user"},
        {"text": "boom", "type": "error"},
    ]


# ---------------------------------------------------------------------------
# Issue 2: init() loads the main chat history before opening the SSE stream.
# ---------------------------------------------------------------------------


def test_init_loads_main_history_before_connecting_stream():
    """The init IIFE in ``static/js/app.js`` must call
    loadHistory(currentProject) before connectStream() so the persisted
    conversation is visible immediately on project open."""
    text = JS_SOURCE.read_text(encoding="utf-8")
    init_match = re.search(
        r"\(async function init\(\) \{(.*?)\}\)\(\);",
        text,
        re.DOTALL,
    )
    assert init_match, "init IIFE not found in static/js/app.js"
    # Re-host the init body inside a Node script with stubbed collaborators
    # so we can observe the real call order from the source.
    bindings = (
        "var calls = [];\n"
        "function loadCurrentState() { calls.push('state'); return Promise.resolve(); }\n"
        "function loadHistory(project) { calls.push('history:' + project); return Promise.resolve(); }\n"
        "function syncCurrentPreview() { calls.push('preview'); return Promise.resolve(); }\n"
        "function connectStream() { calls.push('stream'); return undefined; }\n"
        "var currentProject = 'demo';\n"
        "var initBody = `"
        + init_match.group(0).replace("\\", "\\\\").replace("`", "\\`")
        + "`;\n"
    )
    body = (
        "eval(initBody).then(() => {\n"
        "  process.stdout.write(JSON.stringify(calls));\n"
        "});\n"
    )
    observed = json.loads(_eval_helper(bindings, body))
    assert observed == ["state", "history:demo", "preview", "stream"]


# ---------------------------------------------------------------------------
# Issue 5 (frontend): conversation_reset SSE event wipes the chat UI.
# ---------------------------------------------------------------------------


def test_apply_conversation_reset_clears_feed_drawer_and_activity():
    """``applyConversationReset`` must mirror an empty backend state: main
    feed and history drawer are wiped, the question area is reset, and the
    thinking indicator is released so the next user message starts on a
    clean canvas. Activity state must also be cleared."""
    bindings = (
        # DOM stubs that record what the helper touches.
        "var feed = { items: ['prev'], replaceChildren() { this.items.length = 0; }, appendChild(n) { this.items.push(n); } };\n"
        "var historyContent = { items: ['prev'], replaceChildren() { this.items.length = 0; } };\n"
        "var questionArea = { cleared: false, replaceChildren() { this.cleared = true; } };\n"
        "var activityItems = new Map([['x', {}]]);\n"
        "var toolMessages = new Map([['y', {}]]);\n"
        "function clearActivity() { activityItems.clear(); toolMessages.clear(); }\n"
        "var thinkingCalls = [];\n"
        "function setThinking(active) { thinkingCalls.push(active); }\n"
        "var currentProject = 'demo';\n"
        # No chat-empty template in this stub: helper falls back to a synthesized empty-state node.
        "var document = { querySelector() { return null; }, createElement(tag) { return { tag: tag, className: '', textContent: '' }; } };\n"
        + _extract_js("applyConversationReset")
    )
    body = (
        "applyConversationReset();\n"
        "process.stdout.write(JSON.stringify({\n"
        "  feed_items: feed.items,\n"
        "  history_items: historyContent.items,\n"
        "  question_cleared: questionArea.cleared,\n"
        "  activity_items: Array.from(activityItems.keys()),\n"
        "  tool_messages: Array.from(toolMessages.keys()),\n"
        "  thinking_calls: thinkingCalls,\n"
        "}));"
    )
    out = json.loads(_eval_helper(bindings, body))
    # The prior card is dropped and the empty-state placeholder is re-inserted.
    assert len(out["feed_items"]) == 1
    assert out["feed_items"][0]["tag"] == "div"
    assert out["feed_items"][0]["className"] == "empty-state"
    assert out["feed_items"][0]["textContent"] == (
        'Project "demo" selected. Describe a part to begin.'
    )
    assert out["history_items"] == []
    assert out["question_cleared"] is True
    assert out["activity_items"] == []
    assert out["tool_messages"] == []
    # Thinking indicator is released (False) so a stale spinner cannot leak past reset.
    assert out["thinking_calls"] == [False]


def test_apply_conversation_reset_reclones_chat_empty_template():
    """When the #chat-empty template is present in the DOM, the helper must
    clone it back into the freshly cleared feed (matching the empty-state
    recovery logic in ``loadHistory``)."""
    bindings = (
        "var feed = { items: ['prev'], replaceChildren() { this.items.length = 0; }, appendChild(n) { this.items.push(n); } };\n"
        "var historyContent = { replaceChildren() {} };\n"
        "var questionArea = { replaceChildren() {} };\n"
        "var activityItems = new Map();\n"
        "var toolMessages = new Map();\n"
        "function clearActivity() {}\n"
        "function setThinking() {}\n"
        "var currentProject = 'demo';\n"
        "var cloneCalls = 0;\n"
        "var document = { querySelector() { return { content: { cloneNode: function(_d){ cloneCalls++; return '<cloned-empty/>'; } } }; } };\n"
        + _extract_js("applyConversationReset")
    )
    body = (
        "applyConversationReset();\n"
        "process.stdout.write(JSON.stringify({\n"
        "  cloned: cloneCalls,\n"
        "  feed_size: feed.items.length,\n"
        "}));"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["cloned"] == 1
    assert out["feed_size"] == 1


def test_conversation_reset_sse_listener_is_wired_in_app_js():
    """The SSE ``conversation_reset`` listener must exist in ``app.js`` and
    delegate to ``applyConversationReset`` for the matching project only.
    Source-level check: it pins the contract that the backend event reaches
    the right helper without parsing every line of the listener body."""
    text = JS_SOURCE.read_text(encoding="utf-8")
    assert "eventSource.addEventListener('conversation_reset'" in text, (
        "conversation_reset listener missing from static/js/app.js"
    )
    assert "applyConversationReset" in text, (
        "applyConversationReset helper missing from static/js/app.js"
    )
    # The listener body must guard on the project name so events for other
    # tabs/projects cannot wipe the wrong chat feed.
    listener_match = re.search(
        r"eventSource\.addEventListener\('conversation_reset', event => \{(.*?)\}\);",
        text,
        re.DOTALL,
    )
    assert listener_match, "conversation_reset listener body not found"
    body = listener_match.group(1)
    assert "currentProject" in body, (
        "conversation_reset listener must compare data.project against currentProject"
    )
    assert "applyConversationReset()" in body, (
        "conversation_reset listener must call applyConversationReset()"
    )


def test_reset_button_handler_posts_to_reset_endpoint():
    """The reset button handler must POST to ``/api/projects/<name>/reset``
    and ask the user to confirm before destroying the conversation."""
    text = JS_SOURCE.read_text(encoding="utf-8")
    assert "id=\"reset-context\"" in Path(
        Path(__file__).resolve().parents[1] / "templates/index.html"
    ).read_text(encoding="utf-8"), (
        "Reset context button missing from templates/index.html"
    )
    handler_match = re.search(
        r"resetButton\.addEventListener\('click', async \(\) => \{(.*?)\}\);",
        text,
        re.DOTALL,
    )
    assert handler_match, "resetButton click handler missing from static/js/app.js"
    body = handler_match.group(1)
    assert "/reset" in body, "Reset handler must call the /reset endpoint"
    assert "window.confirm" in body, (
        "Reset handler must ask the user to confirm before destroying memory"
    )


# ---------------------------------------------------------------------------
# Issue 2 (template): empty state is re-clonable after history loads.
# ---------------------------------------------------------------------------


def test_empty_state_template_can_be_recloned_into_feed():
    """The chat-empty template element is present in the rendered index HTML
    and is structurally separate from #chat-feed so loadHistory can clear
    the feed without destroying the example-prompt UI."""
    tmp = Path(tempfile.mkdtemp(prefix="cad-empty-state-"))
    try:
        _settings, client = _create_client(tmp)
        client.post("/api/projects/new", json={"name": "demo"})
        body = client.get("/project/demo").data.decode()

        # Contract: the template lives outside #chat-feed and is addressable by ID.
        template_match = re.search(
            r'<template id="chat-empty">(.*?)</template>',
            body,
            re.DOTALL,
        )
        assert template_match is not None, "chat-empty template missing from /project/demo"
        template_html = template_match.group(1)
        # The template content must include the prompt buttons (at least one).
        assert template_html.count('class="example-prompt"') >= 1
        # The feed must NOT contain the empty-state markup directly — if it did,
        # loadHistory's feed.replaceChildren() would erase the example prompts.
        feed_match = re.search(r'<div id="chat-feed"[^>]*>(.*?)</div>', body, re.DOTALL)
        assert feed_match is not None
        assert "example-prompt" not in feed_match.group(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Issue 7: stream_reset triggers a re-sync through existing endpoints.
# ---------------------------------------------------------------------------


def test_stream_reset_handler_runs_all_three_syncs():
    """The stream_reset listener must call loadHistory, loadCurrentState,
    and syncCurrentPreview so the UI converges to persisted state."""
    bindings = (
        "var calls = [];\n"
        "var loadHistory = function(p) { calls.push('history:' + p); return Promise.resolve(); };\n"
        "var loadCurrentState = function() { calls.push('state'); return Promise.resolve(); };\n"
        "var syncCurrentPreview = function() { calls.push('preview'); return Promise.resolve(); };\n"
        "var currentProject = 'demo';\n"
        + _extract_js("syncAfterStreamReset")
    )
    body = (
        "syncAfterStreamReset().then(() => {\n"
        "  process.stdout.write(JSON.stringify(calls));\n"
        "});\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    # Each sync must run once, in the documented order.
    assert out == ["history:demo", "state", "preview"]
