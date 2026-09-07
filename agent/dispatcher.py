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
from agent.tool_results import (
    build_cad_build_multimodal_content,
    build_cad_screenshot_multimodal_content,
    compact_for_context,
)
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success
from agent.tools.question_tool import normalize_questions


def is_model_mutation(name: str, arguments: dict) -> bool:
    """True for tool calls that change ``model.py`` and so invalidate the
    current preview/review state."""
    return name in {"write_file", "edit_file", "insert_file"}


def is_cad_build(name: str, arguments: dict) -> bool:
    """True for the canonical CAD build tool call."""
    return name == "cad_build_and_verify"


def _require_final_parameter_checks() -> None:
    """Reject a ``mode="final"`` build that omits ``parameter_checks``.

    The JSON schema describes ``parameter_checks`` as required when
    ``mode=final`` so every user-stated dimension gets compared against
    named ``model.py`` parameters before the renderer finalises the
    design. The dispatcher validates the schema's intent at the boundary
    rather than letting the runner treat an absent-checks final build
    as verified. The schema has not been tightened to ``"required":
    ["mode", "parameter_checks"]`` to preserve optional ``mode=check``
    usage; the rule below makes the contract explicit at runtime.
    """
    raise ValueError(
        "cad_build_and_verify with mode='final' requires a non-empty "
        "'parameter_checks' list. Provide one bound per user-stated "
        "dimension/angle/clearance/count so the runner can verify each "
        "named model.py parameter."
    )


def _build_mode(arguments: dict) -> str:
    """Resolve the build mode, validating the enum early.

    Accepts the new ``mode`` enum (``check`` | ``final``) and, for backwards
    compatibility with transcripts that still use ``render=True`` / ``False``,
    maps a literal boolean to the corresponding mode. Anything else raises.
    """
    raw = arguments.get("mode")
    if raw is None:
        if arguments.get("render") is True:
            return "final"
        return "check"
    if isinstance(raw, bool):
        return "final" if raw else "check"
    if raw in {"check", "final"}:
        return raw
    raise ValueError(
        f"cad_build_and_verify mode must be 'check' or 'final', got {raw!r}."
    )


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

    The dispatcher knows only the seven tool names published by
    :mod:`agent.tool_schemas`: ``cad_build_and_verify``, ``cad_screenshot``,
    ``cad_review``, ``read_file``, ``write_file``, ``edit_file``,
    ``insert_file``, ``regex_replace`` (parity), and ``question``. Tool
    instances expose a per-call ``with_call_id`` method plus a generic
    ``execute(args)`` entry; the dispatcher selects the right one based on
    name.
    """
    if name == "cad_build_and_verify":
        # ``mode`` defaults to ``check`` so a missing argument still maps to
        # the cheap path. The runner accepts the explicit ``final`` mode for
        # final verification (renders + inline image + parameter_checks).
        mode = _build_mode(args)
        checks = args.get("parameter_checks") or []
        legacy_render = "mode" not in args and args.get("render") is True
        # Always enforce the bound presence: ``parameter_checks`` is only
        # useful when it carries an equals / minimum / maximum / tolerance,
        # and a name without a bound is silently treated as a no-op pass by
        # the runner. Catching it here keeps the LLM from producing useless
        # checks on the final render and from accidentally suppressing
        # critical dimension verification.
        for index, check in enumerate(checks):
            if not isinstance(check, dict):
                raise ValueError(f"parameter_checks[{index}] must be an object.")
            name_present = isinstance(check.get("name"), str) and check["name"]
            # ``tolerance`` only narrows a comparison target; the sandbox
            # runner reads it as ``abs_tol`` for ``equals`` and as a slack
            # margin for ``minimum``/``maximum``, but it never produces a
            # comparison of its own. A check carrying only ``tolerance``
            # would therefore emit a passing result without measuring
            # anything, so reject it at the boundary.
            has_target = any(
                key in check for key in ("equals", "minimum", "maximum")
            )
            if not name_present:
                raise ValueError(
                    f"parameter_checks[{index}] requires a non-empty 'name'."
                )
            if not has_target:
                raise ValueError(
                    f"parameter_checks[{index}] requires at least one of "
                    "equals, minimum, maximum (tolerance alone is not a "
                    "valid bound)."
                )
        cad = tools.cad.with_call_id(call_id)
        if checks:
            return cad.build_and_verify(mode, checks), False
        # ``render=true`` was the pre-mode public shape. Keep it compatible
        # with existing transcripts and clients; new explicit final calls
        # must carry the parameter checks promised by the schema.
        if mode == "final" and not legacy_render:
            _require_final_parameter_checks()
        return cad.build_and_verify(mode), False
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
                "model.py",
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
                "model.py",
                args.get("content", ""),
                args.get("expected_sha256"),
            ),
            False,
        )
    if name == "edit_file":
        tool = (
            tools.file.with_call_id(call_id) if call_id else tools.file
        )
        edits = args.get("edits")
        if edits is None:
            raise ValueError(
                "edit_file requires 'edits' (list of {old_string, new_string})."
            )
        if not isinstance(edits, list):
            raise ValueError("edit_file 'edits' must be a list of objects.")
        # Apply all edits in one lock so the revision captures the final
        # post-state atomically; the file tool exposes an
        # ``edit_file_atomic`` helper that performs the batch safely.
        if len(edits) == 1:
            entry = edits[0]
            return (
                tool.edit_file(
                    "model.py",
                    entry.get("old_string", ""),
                    entry.get("new_string", ""),
                    args.get("expected_sha256"),
                ),
                False,
            )
        return (
            tool.edit_file_atomic("model.py", edits, args.get("expected_sha256")),
            False,
        )
    if name == "insert_file":
        tool = tools.file.with_call_id(call_id) if call_id else tools.file
        return (
            tool.insert_file(
                "model.py",
                anchor=args["anchor"],
                content=args["content"],
                position=args["position"],
                expected_sha256=args.get("expected_sha256"),
            ),
            False,
        )
    if name == "regex_replace":
        # ``regex_replace`` is implemented on ``FileTool`` for parity with
        # the read/write/edit/insert surface. It is intentionally not
        # surfaced in :mod:`agent.tool_schemas` so it cannot be invoked
        # by the LLM. The dispatcher still routes the call for any
        # internal caller that holds a ``FileTool`` reference.
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
        atomic_write_json(
            state_path,
            {"status": "WAITING_FOR_USER", "waiting_question": question_state},
        )
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
        if is_cad_build(name, arguments):
            preview_id = None
        if cad_fix_required and name in {"cad_screenshot", "cad_review"}:
            raise ValueError(
                f"{name} requires a successful build of the current revision."
            )
        raw_result, waiting = dispatch(tools, project, name, arguments, call_id)
        result = tool_success(name, raw_result)
        if is_model_mutation(name, arguments):
            preview_id = None
            cad_error = None
            cad_fix_required = True
            publish("revision_updated", {"project": project})
        if is_cad_build(name, arguments):
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
        # A successful build with mode=final can attach the rendered PNG
        # directly to the tool message so the agent evaluates it in-band
        # instead of calling the subordinate visual reviewer. This avoids the
        # isolated sub-session that previously re-derived the design rationale
        # from scratch. Drive the multimodal-decision off the *raw* result
        # because ``compact_for_context`` strips ``render`` to shrink the
        # prompt — the mode flag is still the cheapest signal that inline-
        # image evidence is available.
        multimodal = build_cad_build_multimodal_content(
            result, project_dir, context_result=context_result
        )
        if multimodal is not None:
            context_content = multimodal["content"]
            image_paths = list(multimodal.get("image_paths") or [])
        # ``mode`` is the canonical signal that triggers the inline-image
        # final-build evidence path. The renderer is the actual source of truth
        # and ``compact_for_context`` strips the raw mode flag. Normalize here
        # so legacy ``render=true`` callers clear the same final-verification
        # gate as the new ``mode="final"`` schema.
        build_is_final = build_succeeded and _build_mode(arguments) == "final"
        if build_succeeded and build_is_final and image_paths:
            cad_fix_required = False
    elif name == "cad_screenshot":
        # Attach the requested views + contact sheet inline so the reviewer
        # can inspect the rendered output without a separate read step. Skip
        # the branch when no inline image is present (e.g. cache miss that
        # raced the tool result path); the next turn can request a fresh
        # screenshot.
        multimodal = build_cad_screenshot_multimodal_content(
            result, project_dir, context_result=context_result
        )
        if multimodal is not None:
            context_content = multimodal["content"]
            image_paths = list(multimodal.get("image_paths") or [])
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
