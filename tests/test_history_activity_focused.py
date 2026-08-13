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


# ---------------------------------------------------------------------------
# Issue 6: Activity labels map tool schema names to user-facing wording and
# fall back to a sanitized human string for unknown tools.
# ---------------------------------------------------------------------------


def test_activity_label_uses_intended_wording_for_every_tool():
    """Every currently-exposed tool schema name produces a label that
    contains no underscore (visible-in-UI contract)."""
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    body = (
        "var labels = {};\n"
        "for (const name of ['cad_build_and_verify', 'file_write', 'file_replace',\n"
        "                    'file_regex_replace', 'file_read', 'terminal_run',\n"
        "                    'terminal_check', 'experience_search', 'experience_add',\n"
        "                    'experience_update', 'question', 'agent', 'usage',\n"
        "                    'preparing', 'running', 'rendering', 'reviewing',\n"
        "                    'started', 'completed', 'error', 'stopped']) {\n"
        "  labels[name] = activityLabel(name);\n"
        "}\n"
        "process.stdout.write(JSON.stringify(labels));\n"
    )
    labels = json.loads(_eval_helper(bindings, body))

    # Every label must be a non-empty human-readable string.
    assert all(isinstance(v, str) and v and "_" not in v for v in labels.values())

    # Spot-check the most-visible labels: anything the user actually sees for
    # the heavy hitters must read as an English phrase, not a tool name.
    assert labels["cad_build_and_verify"] == "Building model"
    assert labels["file_write"] == labels["file_replace"] == labels["file_regex_replace"]
    assert labels["file_write"] == "Updating model"
    assert labels["terminal_run"] == labels["terminal_check"] == "Checking model"
    assert labels["experience_search"] == "Checking past solutions"
    assert labels["question"] == "Requesting input"


def test_activity_label_fallback_sanitizes_unknown_tools():
    """Unknown tool names must not leak underscores into the UI."""
    bindings = (
        _extract_js("activityLabels", kind="object")
        + "\n"
        + _extract_js("activityLabel")
    )
    body = (
        "process.stdout.write(JSON.stringify({\n"
        "  unknown: activityLabel('future_tool_name'),\n"
        "  empty: activityLabel(''),\n"
        "  undef: activityLabel(undefined),\n"
        "}));\n"
    )
    out = json.loads(_eval_helper(bindings, body))
    assert out["unknown"] == "future tool name"
    assert out["empty"] == "Activity"
    assert out["undef"] == "Activity"


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
        "activityItems.set('err-1', {callId: 'err-1', tool: 'file_write', status: 'error'});\n"
        "makeRow('err-1', 'error');\n"
        # Already-failed row at error status must also remain untouched.
        "activityItems.set('err-2', {callId: 'err-2', tool: 'terminal_run', status: 'error'});\n"
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
