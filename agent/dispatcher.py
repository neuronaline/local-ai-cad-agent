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

from agent.activity_log import ActivityLogger
from agent.io import atomic_write_json
from agent.revisions import MODEL_FILENAME
from agent.tool_results import (
    build_cad_build_multimodal_content,
    compact_for_context,
)
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success


def is_model_mutation(name: str) -> bool:
    """True for tool calls that mutate ``model.scad`` and invalidate
    preview/review state."""
    return name in {"write_file", "edit_file"}


def is_cad_build(name: str) -> bool:
    """True for the canonical CAD build tool call."""
    return name == "cad_build_and_verify"


def _build_render(arguments: dict) -> bool:
    """Resolve the optional render switch used by CAD build calls."""
    render = arguments.get("render", True)
    if not isinstance(render, bool):
        raise ValueError("cad_build_and_verify render must be a boolean.")
    return render


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

    Only the five model-facing tools published by
    :mod:`agent.tool_schemas` (``cad_build_and_verify``, ``read_file``,
    ``write_file``, ``edit_file``, and ``question``) are recognised. Tool
    instances expose a per-call ``with_call_id`` method that propagates
    the call id into activity-log / debug-log entries; this dispatcher
    uses an explicit ``if/elif`` table so unknown names surface as a
    clear ``ValueError`` instead of a bare ``AttributeError`` from a
    stray ``getattr`` lookup on the ``ProjectTools`` bundle.
    """
    if name == "cad_build_and_verify":
        cad = tools.cad.with_call_id(call_id)
        return cad.build_and_verify(render=_build_render(args)), False
    if name == "read_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        return (
            tool.read_file(
                MODEL_FILENAME,
                args.get("offset") or 1,
                args.get("limit"),
            ),
            False,
        )
    if name == "write_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        return (
            tool.write_file(
                MODEL_FILENAME,
                args.get("content", ""),
            ),
            False,
        )
    if name == "edit_file":
        return _dispatch_edit_file(tools.file, args, call_id)
    if name == "question":
        return _dispatch_question(tools.question, tools.project_dir, project, args)
    raise ValueError(f"Unknown or unsupported tool: {name!r}")


def _dispatch_edit_file(file_tool, args: dict, call_id: str) -> tuple[object, bool]:
    """Resolve ``edit_file`` arguments into a single ``edit_file`` /
    ``edit_file_atomic`` call.

    ``edit_file`` accepts either ``{old_string, new_string}`` for a
    single replacement, or ``edits`` (a list of such pairs) for an
    atomic batch. ``edit_file_atomic`` performs the batch safely so the
    revision captures the final post-state under the file tool's lock.
    """
    tool = (
        file_tool.with_call_id(call_id) if call_id else file_tool
    )
    edits = args.get("edits")
    if not edits and "old_string" in args:
        new_str = args.get("new_string")
        edits = [
            {
                "old_string": args["old_string"],
                "new_string": "" if new_str is None else new_str,
            }
        ]
    elif isinstance(edits, dict):
        edits = [edits]
    if not edits:
        raise ValueError(
            "edit_file requires 'edits' (list of {old_string, new_string}) or 'old_string' and 'new_string'."
        )
    if not isinstance(edits, list):
        raise ValueError("edit_file 'edits' must be a list of objects.")
    if len(edits) == 1:
        entry = edits[0]
        if not isinstance(entry, dict):
            raise ValueError("edit_file 'edits' must be a list of objects.")
        old_str = entry.get("old_string")
        new_str = entry.get("new_string")
        return (
            tool.edit_file(
                MODEL_FILENAME,
                "" if old_str is None else old_str,
                "" if new_str is None else new_str,
            ),
            False,
        )
    return (
        tool.edit_file_atomic(MODEL_FILENAME, edits),
        False,
    )


def _dispatch_question(
    question_tool, project_dir: Path, project: str, args: dict
) -> tuple[object, bool]:
    """Persist the ``WAITING_FOR_USER`` state and return the question tool
    envelope.

    ``QuestionTool.execute`` already validates and normalises the
    questions; this dispatcher only writes the persistent state marker
    so the next user request can resume the conversation cleanly.
    """
    # execute() returns the normalized list as its third element; reuse
    # it for the persisted state instead of re-running normalize_questions
    # here. Validation also already ran inside execute(), so ask() can
    # publish straight away.
    result, _waiting, questions = question_tool.execute(args, project=project)
    title = args.get("title", "")
    question_state = {
        "title": title.strip() if isinstance(title, str) else "",
        "questions": questions,
    }
    state_path = project_dir / ".agent_state.json"
    atomic_write_json(
        state_path,
        {"status": "WAITING_FOR_USER", "waiting_question": question_state},
    )
    return result, True


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
    activity_logger: ActivityLogger | None = None,
    run_id: str | None = None,
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
    build_succeeded = False
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
        if activity_logger is not None:
            activity_logger.log(
                "tool_call_start",
                {
                    "project": project,
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                },
                run_id=run_id,
            )
        if is_cad_build(name):
            preview_id = None
        raw_result, waiting = dispatch(tools, project, name, arguments, call_id)
        result = tool_success(name, raw_result)
        if is_model_mutation(name):
            preview_id = None
            cad_error = None
            cad_fix_required = True
            publish("revision_updated", {"project": project})
        if is_cad_build(name):
            preview_id = register_preview(project, project_dir)
            cad_error = None
            build_succeeded = True
            publish(
                "preview_updated",
                {"project": project, "preview_id": preview_id},
            )
        publish(
            "tool_status",
            {**tool_event, "status": "completed", "result": result},
        )
        if activity_logger is not None:
            activity_logger.log(
                "tool_call_result",
                {
                    "project": project,
                    "call_id": call_id,
                    "tool": name,
                    "result": result,
                },
                run_id=run_id,
            )
    except Exception as error:  # noqa: BLE001 - Tool errors are useful LLM context.
        result, waiting = tool_failure(name, error), False
        if debug_log is not None:
            debug_log(project_dir, call_id, name, error, result)
        if is_cad_build(name):
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
        if activity_logger is not None:
            activity_logger.log(
                "tool_call_result",
                {
                    "project": project,
                    "call_id": call_id,
                    "tool": name,
                    "result": result,
                    "error": True,
                },
                run_id=run_id,
            )
    context_result = compact_for_context(name, result)
    context_content: str | list = context_result
    image_paths: list[Path] = []
    if name == "cad_build_and_verify":
        multimodal = build_cad_build_multimodal_content(
            result, project_dir, context_result=context_result
        )
        if multimodal is not None:
            context_content = multimodal["content"]
            image_paths = list(multimodal.get("image_paths") or [])
        if build_succeeded and _build_render(arguments) and image_paths:
            cad_fix_required = False
    tool_message = {"role": "tool", "tool_call_id": call_id, "content": context_content}
    messages.append(tool_message)
    append_message(project_dir, tool_message)
    return preview_id, cad_error, cad_fix_required, waiting


def cancel_remaining_tool_calls(
    project_dir: Path,
    tool_calls: list[dict],
    processed_call_ids: set[str],
    messages: list[dict],
    *,
    append_message: Callable[[Path, dict], None],
) -> None:
    """Persist cancelled tool-result entries for unprocessed calls."""
    for tc in tool_calls:
        cid = tc.get("id", "")
        if cid and cid not in processed_call_ids:
            tool_name = tc.get("function", {}).get("name", "unknown_tool")
            cancelled = tool_failure(
                tool_name, RuntimeError("Tool call cancelled (question or stop).")
            )
            entry = {"role": "tool", "tool_call_id": cid, "content": cancelled}
            messages.append(entry)
            append_message(project_dir, entry)
