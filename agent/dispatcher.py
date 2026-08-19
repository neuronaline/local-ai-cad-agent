"""Tool name → call + result envelope extracted from ``agent.core.AgentRunner``.

The dispatcher is a single function (well, two — dispatch and process) that
takes a tool name + arguments and runs it against the per-project
``ProjectTools`` bundle. Pulling these out of ``AgentRunner`` lets the agent
loop focus on its lifecycle responsibilities (thread management, prompt
construction, terminal events) instead of re-stating which tool does what.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

from agent.tool_results import build_cad_build_multimodal_content, compact_for_context
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success
from agent.tools.question_tool import normalize_questions


def is_model_mutation(name: str, arguments: dict) -> bool:
    """True for tool calls that change ``model.py`` and so invalidate the
    current preview/review state."""
    return (
        name in {"write_file", "edit_file"}
        and arguments.get("filename") == "model.py"
    )


def is_cad_build(name: str, arguments: dict) -> bool:
    """True for the canonical CAD build tool call."""
    return name == "cad_build_and_verify"


def dispatch(
    tools,
    project: str,
    name: str,
    args: dict,
    call_id: str = "",
) -> tuple[object, bool]:
    """Run a single tool call by name. Returns ``(result, waiting)`` where
    ``waiting`` is True only for the question tool (the LLM is parked
    until the user replies).

    Tool instances expose both a per-call ``with_call_id`` method (used to
    bind a publish call_id) and a generic ``execute(args)`` entry. The
    dispatcher selects the right one based on name; tools without the
    explicit prefix (e.g. ``terminal_run``, ``experience_search``) fall
    through to ``getattr(tools, name).execute(args)``.
    """
    if name == "cad_build_and_verify":
        # ``render`` defaults to False, matching both the schema and
        # ``CadTool.build_and_verify``. Final verification opts in explicitly.
        return tools.cad.with_call_id(call_id).build_and_verify(
            args.get("render", False)
        ), False
    if name == "cad_screenshot":
        tool = (
            tools.screenshot.with_call_id(call_id)
            if call_id
            else tools.screenshot
        )
        return tool.execute(args), False
    if name == "cad_review":
        tool = (
            tools.review.with_call_id(call_id)
            if call_id
            else tools.review
        )
        return tool.execute(args), False
    if name == "read_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        return (
            tool.read_file(
                args["filename"],
                args.get("offset", 1),
                args.get("limit"),
                args.get("known_sha256"),
            ),
            False,
        )
    if name == "write_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        return (
            tool.write_file(
                args["filename"],
                args.get("content", ""),
                args.get("expected_sha256"),
            ),
            False,
        )
    if name == "edit_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        return (
            tool.edit_file(
                args["filename"],
                args["old_string"],
                args.get("new_string", ""),
                args.get("expected_sha256"),
            ),
            False,
        )
    if name == "regex_replace":
        # Reserved for parity with the file_tool surface; not exposed in
        # TOOL_SCHEMAS yet but the dispatcher passes through so future
        # schema additions work without touching agent.core.
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        execute = getattr(tool, "regex_replace", None)
        if not callable(execute):
            return {"error": f"Tool {name!r} is not registered."}, False
        return execute(args), False
    if name == "question":
        # execute() validates, normalizes, and publishes the questions;
        # the normalized list is also needed for the persisted state.
        result, _waiting = tools.question.execute(args, project=project)
        questions = normalize_questions(args)
        title = args.get("title", "")
        question_state = {
            "title": title.strip() if isinstance(title, str) else "",
            "questions": questions,
        }
        state_path = tools.project_dir / ".agent_state.json"
        temporary_state = tools.project_dir / ".agent_state.json.tmp"
        temporary_state.write_text(
            json.dumps(
                {"status": "WAITING_FOR_USER", "waiting_question": question_state}
            ),
            encoding="utf-8",
        )
        temporary_state.replace(state_path)
        return result, True
    tool = getattr(tools, name)
    return tool.execute(args), False


def normalize_tool_calls(raw_calls: object) -> list[dict]:
    """Ensure persisted tool calls remain valid protocol messages."""
    if not isinstance(raw_calls, list):
        return []
    normalized: list[dict] = []
    for raw_call in raw_calls:
        call = raw_call if isinstance(raw_call, dict) else {}
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        arguments = function.get("arguments")
        normalized.append(
            {
                "id": str(call.get("id") or f"invalid-{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": name
                    if isinstance(name, str) and name
                    else "unknown_tool",
                    "arguments": arguments if isinstance(arguments, str) else "{}",
                },
            }
        )
    return normalized


def process_tool_call(
    tools,
    project: str,
    project_dir: Path,
    call: dict,
    cad_fix_required: bool,
    prev_preview_id: str | None,
    cad_error: str | None,
    messages: list[dict],
    *,
    publish: Callable[[str, dict], None],
    register_preview: Callable[[str, Path], str],
    append_message: Callable[[Path, dict], None],
    debug_log: Callable[[Path, str, str, Exception, str], None] | None = None,
) -> tuple[str | None, str | None, bool, bool]:
    """Execute a single tool call, update state, return ``(preview, error,
    fix_required, waiting)``.

    ``publish`` and ``register_preview`` are injected so this function does
    not depend on the ``AgentRunner`` instance. ``append_message`` is the
    usual :func:`ConversationStore.append` (or its thin wrapper on the
    runner). ``debug_log`` is optional; the runner passes a logger that
    records tool errors to ``debug-errors.jsonl`` when configured.
    """
    call = call if isinstance(call, dict) else {}
    call_id = str(call.get("id") or f"invalid-{uuid.uuid4().hex}")
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else ""
    if not isinstance(name, str) or not name:
        name = "unknown_tool"
    preview_id = prev_preview_id
    waiting = False
    arguments: dict = {}
    try:
        argument_text = (
            function.get("arguments") if isinstance(function, dict) else ""
        )
        arguments = json.loads(argument_text or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")
        tool_event = {
            "project": project,
            "call_id": call_id,
            "tool": name,
            "arguments": arguments,
        }
        publish("tool_status", {**tool_event, "status": "running"})
        if is_cad_build(name, arguments):
            preview_id = None
        raw_result, waiting = dispatch(tools, project, name, arguments, call_id)
        result = tool_success(name, raw_result)
        if is_model_mutation(name, arguments):
            preview_id = None
            cad_error = None
            cad_fix_required = True
            publish("revision_updated", {"project": project})
        if is_cad_build(name, arguments):
            preview_id = register_preview(project, project_dir)
            publish(
                "preview_updated",
                {"project": project, "preview_id": preview_id},
            )
        publish(
            "tool_status",
            {**tool_event, "status": "completed", "result": result},
        )
    except Exception as error:  # noqa: BLE001 - Tool errors are useful LLM context.
        result, waiting = tool_failure(name, error), False
        if debug_log is not None:
            debug_log(project_dir, call_id, name, error, result)
        if is_cad_build(name, arguments):
            cad_error = str(error)
            cad_fix_required = True
            preview_id = None
        publish(
            "tool_status",
            {
                "project": project,
                "call_id": call_id,
                "tool": name,
                "arguments": arguments,
                "status": "error",
                "result": result,
            },
        )
    context_result = compact_for_context(name, result)
    context_content: str | list = context_result
    image_paths: list[Path] = []
    if name == "cad_build_and_verify":
        # A successful build with render=true can attach the rendered PNG/STL
        # directly to the tool message so the agent evaluates it in-band
        # instead of calling the subordinate visual reviewer. This avoids the
        # isolated sub-session that previously re-derived the design rationale
        # from scratch. Drive the multimodal-decision off the *raw* result
        # because ``compact_for_context`` strips ``render`` to shrink the
        # prompt — the render flag is still the cheapest signal that
        # inline-image evidence is available.
        multimodal = build_cad_build_multimodal_content(result, project_dir)
        if multimodal is not None:
            context_content = multimodal["content"]
            image_paths = list(multimodal.get("image_paths") or [])
    if name == "read_file":
        _prune_prior_read_file(messages, arguments, context_result)
    tool_message = {"role": "tool", "tool_call_id": call_id, "content": context_content}
    messages.append(tool_message)
    append_message(project_dir, tool_message)
    _remember_inline_tool_images(project_dir, call_id, image_paths)
    return preview_id, cad_error, cad_fix_required, waiting


def _remember_inline_tool_images(
    project_dir: Path, call_id: str, image_paths: list[Path]
) -> None:
    """Record host-relative image paths for a tool message.

    The conversation log stores the multimodal content verbatim so the next
    turn can re-inject the same evidence, but the persisted paths let the
    history redaction step replace stale inline images with ``[Inline render
    from <view>]`` placeholders on subsequent loads.
    """
    if not image_paths or not call_id:
        return
    try:
        rel = [
            str(path.relative_to(project_dir))
            for path in image_paths
            if path.is_file()
        ]
    except ValueError:
        return
    if not rel:
        return
    index_path = project_dir / ".agent_tool_images.json"
    try:
        index: dict[str, list[str]] = {}
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    index = payload
            except (OSError, json.JSONDecodeError):
                index = {}
        index[call_id] = rel
        index_path.write_text(
            json.dumps(index, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        return


def _prune_prior_read_file(
    messages: list[dict], arguments: dict, new_result: str
) -> None:
    """Drop the file body from earlier ``read_file`` tool results.

    The first ``read_file`` call in the current run still ships the file body
    so the agent can build ``edit_file`` ``old_string`` arguments from it.
    Once the agent has seen the body once, every subsequent read returns just
    the SHA — keeping the duplicate file body in every older turn inflates the
    prompt by ~4-5 KB per turn with no information gain.

    The new tool result itself is left untouched so the agent always has the
    latest body available; only older read_file entries are compacted.
    """
    filename = arguments.get("filename")
    if not isinstance(filename, str) or not filename:
        return
    try:
        new_payload = json.loads(new_result)
    except (TypeError, json.JSONDecodeError):
        return
    if not isinstance(new_payload, dict) or new_payload.get("ok") is not True:
        return
    new_data = new_payload.get("data")
    new_sha: str | None = None
    if isinstance(new_data, str):
        try:
            parsed = json.loads(new_data)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            sha = parsed.get("sha256")
            if isinstance(sha, str):
                new_sha = sha
    if not new_sha:
        return
    for entry in messages:
        if entry.get("role") != "tool":
            continue
        prior_result = entry.get("content")
        if not isinstance(prior_result, str):
            continue
        try:
            prior_payload = json.loads(prior_result)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(prior_payload, dict)
            or prior_payload.get("tool") != "read_file"
            or prior_payload.get("ok") is not True
        ):
            continue
        prior_data = prior_payload.get("data")
        if not isinstance(prior_data, str):
            continue
        try:
            prior_parsed = json.loads(prior_data)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(prior_parsed, dict)
            or prior_parsed.get("exists") is not True
        ):
            continue
        if prior_parsed.get("content") is None:
            continue
        compacted = {
            "exists": True,
            "unchanged": True,
            "sha256": prior_parsed.get("sha256") or new_sha,
            "total_lines": prior_parsed.get("total_lines"),
        }
        prior_payload["data"] = json.dumps(compacted, ensure_ascii=False)
        entry["content"] = json.dumps(prior_payload, ensure_ascii=False)


def cancel_remaining_tool_calls(
    project_dir: Path,
    tool_calls: list[dict],
    processed_call_ids: set[str],
    messages: list[dict],
    *,
    append_message: Callable[[Path, dict], None],
) -> None:
    """Persist cancelled tool-result entries for unprocessed calls."""
    cancelled = json.dumps({"error": "Tool call cancelled (question or stop)."})
    for tc in tool_calls:
        cid = tc.get("id", "")
        if cid and cid not in processed_call_ids:
            entry = {"role": "tool", "tool_call_id": cid, "content": cancelled}
            messages.append(entry)
            append_message(project_dir, entry)
