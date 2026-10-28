"""Google AI Studio Interactions API adapter (POST /v1beta/interactions).

Implements the stateful, step-based Interactions API described at:
https://aistudio.google.com/docs/interactions-overview

Converts the agent runner's message and tool structures into Google's typed
steps (user_input, model_output, thought, function_call, function_result)
and translates SSE streams (step.start, step.delta, interaction.completed)
into the standard agent event stream and chat completion dictionary.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from typing import Any

from agent.activity_log import summarize_llm_messages
from agent.llm_base import (
    BaseLLMClient,
    CompletionTruncatedError,
    RequestCancelled,
    THOUGHT_SIGNATURE_SENTINEL,
    StreamResponseError,
    _force_close_response,
    attach_thought_signature,
    _stream_error_detail,
    extract_text_tool_calls,
    extract_think_tags,
    get_thought_signature,
    is_continuation_messages,
    iter_lines_with_cancel,
    normalize_messages,
    post_with_cancel,
    relocate_tool_images,
    strip_encrypted_reasoning,
)
from agent.settings import Settings

_LOG = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
PRIMARY_API_KEY_ENV = "GEMINI_API_KEY"
FALLBACK_API_KEY_ENV = "GOOGLE_API_KEY"

_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "$defs", "$id"}
)


def _parse_data_uri(uri: str) -> tuple[str, str] | None:
    """Extract (mime_type, base64_data) from a data: URI."""
    if not uri.startswith("data:"):
        return None
    match = re.match(r"^data:([^;]+);base64,(.+)$", uri, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Recursively strip unsupported JSON Schema keys for Google Gemini API."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, val in schema.items():
        if key in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(val, dict):
            cleaned[key] = _sanitize_schema_for_gemini(val)
        elif isinstance(val, list):
            cleaned[key] = [
                _sanitize_schema_for_gemini(item) if isinstance(item, dict) else item
                for item in val
            ]
        else:
            cleaned[key] = val
    return cleaned


def convert_tools_to_declarations(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert OpenAI-compatible tool specifications to Google function declarations."""
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function") or {}
            decl: dict[str, Any] = {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
            }
            if "parameters" in fn:
                decl["parameters"] = _sanitize_schema_for_gemini(fn["parameters"])
            declarations.append(decl)
        elif "function_declarations" in tool or "functionDeclarations" in tool:
            raw_decls = tool.get("function_declarations") or tool.get("functionDeclarations") or []
            for decl in raw_decls:
                cleaned_decl = dict(decl)
                if "parameters" in cleaned_decl:
                    cleaned_decl["parameters"] = _sanitize_schema_for_gemini(
                        cleaned_decl["parameters"]
                    )
                declarations.append(cleaned_decl)
    return [{"function_declarations": declarations}] if declarations else None


def convert_messages_to_interactions(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system instruction and convert conversation history into typed steps.

    Returns:
        (system_instruction, steps)
    """
    messages = normalize_messages(relocate_tool_images(messages))
    system_instruction: str | None = None
    steps: list[dict[str, Any]] = []
    call_id_to_name: dict[str, str] = {}

    def _append_user_step(parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if steps and steps[-1].get("type") == "user_input" and isinstance(steps[-1].get("content"), list):
            steps[-1]["content"].extend(parts)
        else:
            steps.append({"type": "user_input", "content": list(parts)})

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            text = ""
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                text = content.strip()
            elif isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = " ".join(parts).strip()
            if text:
                if system_instruction:
                    system_instruction += "\n\n" + text
                else:
                    system_instruction = text
            continue

        if role == "user":
            content = msg.get("content")
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content.strip():
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        t = item.get("text", "")
                        if t:
                            parts.append({"type": "text", "text": t})
                    elif item_type == "image_url":
                        img_dict = item.get("image_url") or {}
                        url = img_dict.get("url", "")
                        parsed = _parse_data_uri(url)
                        if parsed:
                            mime, b64_data = parsed
                            parts.append(
                                {
                                    "type": "image",
                                    "data": b64_data,
                                    "mime_type": mime,
                                }
                            )
                        elif url:
                            parts.append({"type": "image", "image_url": url})
            if parts:
                _append_user_step(parts)

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            # The turn's real signature (thought step + unsigned calls); stored
            # calls without one carry the sentinel, so skip those.
            turn_sig = get_thought_signature(msg) or next(
                (
                    sig
                    for sig in map(get_thought_signature, tool_calls)
                    if sig and sig != THOUGHT_SIGNATURE_SENTINEL
                ),
                THOUGHT_SIGNATURE_SENTINEL,
            )

            # 1. Intermediate thoughts
            reasoning = (
                msg.get("reasoning")
                or msg.get("reasoning_content")
                or msg.get("thinking")
            )
            if isinstance(reasoning, str) and reasoning.strip():
                clean_r = strip_encrypted_reasoning(reasoning)
                if clean_r:
                    steps.append(
                        {
                            "type": "thought",
                            "signature": turn_sig,
                            "thought_signature": turn_sig,
                            "summary": [{"type": "text", "text": clean_r}],
                            "content": [{"type": "text", "text": clean_r}],
                        }
                    )

            # 2. Text content — placed before function_call steps so that
            # function_call immediately precedes function_result in the steps array.
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                steps.append(
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": content}],
                    }
                )

            # 3. Function calls — directly preceding subsequent function_results
            for tc in tool_calls:
                fn = tc.get("function") or {}
                fn_name = fn.get("name") or ""
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                if fn_name:
                    call_id_to_name[call_id] = fn_name
                raw_args = fn.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                sig = get_thought_signature(tc) or turn_sig
                step_item = {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": fn_name,
                    "arguments": args,
                    "args": args,
                    "thought_signature": sig,
                    "signature": sig,
                }
                steps.append(step_item)

        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            fn_name = msg.get("name") or call_id_to_name.get(call_id, "unknown_tool")
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
                ]
                content = "\n".join(text_parts)

            if isinstance(content, str):
                try:
                    res_val = json.loads(content)
                except (json.JSONDecodeError, TypeError, ValueError):
                    res_val = {"output": content}
            elif isinstance(content, dict):
                res_val = content
            else:
                res_val = {"output": str(content)}

            steps.append(
                {
                    "type": "function_result",
                    "call_id": call_id,
                    "id": call_id,
                    "name": fn_name,
                    "result": res_val if isinstance(res_val, dict) else {"output": res_val},
                }
            )

    if steps and steps[-1].get("type") in ("model_output", "thought", "function_call"):
        steps.append({
            "type": "user_input",
            "content": [{"type": "text", "text": "Continue."}],
        })
    return system_instruction, steps


def _extract_text(obj: Any) -> str:
    """Extract plain text recursively from text, content, summary, or part structures."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            return obj["text"]
        if "content" in obj:
            return _extract_text(obj["content"])
        if "summary" in obj:
            return _extract_text(obj["summary"])
        if "thought" in obj and isinstance(obj["thought"], (str, dict, list)):
            return _extract_text(obj["thought"])
        if "output" in obj:
            return _extract_text(obj["output"])
    elif isinstance(obj, list):
        return "".join(_extract_text(item) for item in obj)
    return ""


def parse_interactions_stream(
    response: Any,
    *,
    provider_label: str = "Google AI Studio",
    stop_event: threading.Event | None = None,
    stream_callback: Any | None = None,
    idle_timeout: float = 180.0,
) -> dict[str, Any]:
    """Parse Server-Sent Events from the Interactions API."""
    content = ""
    reasoning_text = ""
    tool_calls: dict[str, dict[str, Any]] = {}
    step_meta: dict[str, dict[str, Any]] = {}
    active_step: dict[str, Any] = {}
    last_thought_signature: str | None = None
    last_usage: dict[str, Any] | None = None
    interaction_id: str | None = None
    finish_reason: str | None = None
    finish_time: float | None = None
    last_activity_time = time.monotonic()
    current_event_type: str | None = None

    try:
        for raw_line in iter_lines_with_cancel(response, stop_event=stop_event):
            if stop_event and stop_event.is_set():
                _force_close_response(response)
                raise RequestCancelled(f"{provider_label} request cancelled.")
            now = time.monotonic()
            if raw_line is None:
                if finish_reason is not None and finish_time is not None:
                    if last_usage is not None or (now - finish_time) >= 1.0:
                        break
                elif idle_timeout > 0 and (now - last_activity_time) >= idle_timeout:
                    _force_close_response(response)
                    raise StreamResponseError(
                        f"{provider_label} stream timed out after {idle_timeout:.0f}s of inactivity.",
                        retryable=not (content or tool_calls),
                    )
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line:
                current_event_type = None
                continue
            line = line.strip()
            if line.startswith("event:"):
                current_event_type = line[6:].strip()
                last_activity_time = now
                continue
            if not line.startswith("data:"):
                if finish_reason is not None and finish_time is not None:
                    if last_usage is not None or (now - finish_time) >= 1.0:
                        break
                elif idle_timeout > 0 and (now - last_activity_time) >= idle_timeout:
                    _force_close_response(response)
                    raise StreamResponseError(
                        f"{provider_label} stream timed out after {idle_timeout:.0f}s of inactivity.",
                        retryable=not (content or tool_calls),
                    )
                continue
            last_activity_time = now
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{provider_label} returned malformed streaming JSON.") from error

            if not isinstance(event, dict):
                continue

            if "error" in event:
                raise StreamResponseError(
                    f"{provider_label} stream failed: {_stream_error_detail(event['error'])}",
                    retryable=not (content or tool_calls),
                )

            interaction_obj = (
                event.get("interaction")
                if isinstance(event.get("interaction"), dict)
                else {}
            )

            event_type = (
                current_event_type
                or event.get("event_type")
                or event.get("type")
            )
            event_type_str = str(event_type or "").strip().lower()
            is_step_event = bool(
                event_type_str.startswith(("step.", "content.", "delta", "interaction.step."))
                or event_type_str in ("step", "delta", "thought", "function_call", "model_output")
                or ("step" in event and "interaction" not in event)
            )

            ev_id = (None if is_step_event else event.get("id")) or interaction_obj.get("id")
            if ev_id:
                interaction_id = str(ev_id)

            ev_finish = (
                (None if is_step_event else event.get("finish_reason"))
                or interaction_obj.get("finish_reason")
            )
            ev_status = (
                (None if is_step_event else event.get("status"))
                or interaction_obj.get("status")
            )

            if ev_finish:
                if finish_reason is None:
                    finish_time = time.monotonic()
                finish_reason = str(ev_finish)
            elif ev_status:
                status_val = str(ev_status)
                if status_val.lower() in ("completed", "complete", "done", "stop", "end_turn", "cancelled", "failed"):
                    if finish_reason is None:
                        finish_time = time.monotonic()
                    finish_reason = status_val
            elif event_type_str in ("interaction.completed", "interaction.complete", "response.completed"):
                if finish_reason is None:
                    finish_time = time.monotonic()
                    finish_reason = "completed"

            usage_meta = (
                event.get("usage")
                or event.get("usage_metadata")
                or interaction_obj.get("usage")
                or interaction_obj.get("usage_metadata")
            )
            if isinstance(usage_meta, dict):
                p_tokens = (
                    usage_meta.get("prompt_tokens")
                    or usage_meta.get("prompt_token_count")
                    or 0
                )
                c_tokens = (
                    usage_meta.get("completion_tokens")
                    or usage_meta.get("candidates_token_count")
                    or 0
                )
                r_tokens = (
                    usage_meta.get("reasoning_tokens")
                    or usage_meta.get("thoughts_token_count")
                    or 0
                )
                cached_tokens = (
                    usage_meta.get("cached_content_token_count")
                    or usage_meta.get("cached_tokens")
                    or (usage_meta.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                )
                last_usage = {
                    "prompt_tokens": int(p_tokens),
                    "completion_tokens": int(c_tokens),
                    "reasoning_tokens": int(r_tokens),
                    "completion_tokens_details": {
                        "reasoning_tokens": int(r_tokens),
                    },
                    "prompt_tokens_details": {
                        "cached_tokens": int(cached_tokens),
                    },
                }

            # Handle step.start / step.stop events carrying a single 'step' object
            step_obj = event.get("step") if isinstance(event.get("step"), dict) else None
            s_idx = event.get("index") if event.get("index") is not None else event.get("step_index")
            if step_obj:
                sid = (
                    step_obj.get("id")
                    or step_obj.get("step_id")
                    or step_obj.get("call_id")
                    or event.get("step_id")
                    or event.get("id")
                )
                raw_stype = step_obj.get("type") or step_obj.get("step_type") or ""
                if raw_stype in ("model_generation", "model_output", "text"):
                    stype = "model_output"
                elif raw_stype in ("thought", "thought_summary"):
                    stype = "thought"
                elif raw_stype in ("function_call", "tool_call"):
                    stype = "function_call"
                else:
                    stype = raw_stype

                sig = (
                    step_obj.get("signature")
                    or step_obj.get("thought_signature")
                    or event.get("signature")
                    or event.get("thought_signature")
                )
                if sig:
                    last_thought_signature = str(sig)

                step_info: dict[str, Any] = {
                    "id": str(sid) if sid else None,
                    "index": s_idx,
                    "type": stype,
                    "name": step_obj.get("name") or "",
                    "signature": str(sig) if sig else None,
                }

                if stype == "function_call":
                    call_id = (
                        step_obj.get("call_id")
                        or step_obj.get("id")
                        or sid
                        or f"call_{uuid.uuid4().hex[:8]}"
                    )
                    step_info["call_id"] = str(call_id)
                    fn_name = step_obj.get("name") or ""
                    raw_args = step_obj.get("arguments") or step_obj.get("args")
                    arg_str = ""
                    if isinstance(raw_args, dict) and raw_args:
                        arg_str = json.dumps(raw_args)
                    elif isinstance(raw_args, str) and raw_args:
                        arg_str = raw_args
                    entry = tool_calls.setdefault(
                        call_id,
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": ""},
                            "index": len(tool_calls),
                        },
                    )
                    if fn_name and not entry["function"]["name"]:
                        entry["function"]["name"] = fn_name
                    if arg_str and not entry["function"]["arguments"]:
                        entry["function"]["arguments"] = arg_str
                    if sig and not entry.get("thought_signature"):
                        entry["thought_signature"] = str(sig)
                elif stype == "thought" and event_type in ("step.stop", "interaction.step.done"):
                    if not reasoning_text:
                        th_text = _extract_text(step_obj.get("summary")) or _extract_text(step_obj.get("content"))
                        if th_text:
                            reasoning_text += th_text
                elif stype == "model_output" and event_type in ("step.stop", "interaction.step.done"):
                    if not content:
                        out_text = _extract_text(step_obj.get("content")) or _extract_text(step_obj.get("output"))
                        if out_text:
                            content += out_text

                active_step = step_info
                if sid:
                    step_meta[str(sid)] = step_info
                if s_idx is not None:
                    step_meta[str(s_idx)] = step_info
                if step_info.get("call_id"):
                    step_meta[str(step_info["call_id"])] = step_info

            delta = event.get("delta")
            delta_sid = event.get("step_id") or (delta.get("step_id") if isinstance(delta, dict) else None)
            delta_idx = event.get("index") if event.get("index") is not None else event.get("step_index")
            if delta_sid and str(delta_sid) in step_meta:
                matched_step = step_meta[str(delta_sid)]
            elif delta_idx is not None and str(delta_idx) in step_meta:
                matched_step = step_meta[str(delta_idx)]
            else:
                matched_step = active_step

            active_step_type = matched_step.get("type")

            # Handle step.delta events
            if (event_type in ("step.delta", "content.delta") or delta) and isinstance(delta, dict):
                delta_type = delta.get("type") or ""
                d_sig = (
                    delta.get("signature")
                    or delta.get("thought_signature")
                    or event.get("signature")
                    or event.get("thought_signature")
                )
                if d_sig:
                    last_thought_signature = str(d_sig)

                # 1. Thought deltas (explicit thought delta or delta inside a thought step)
                is_thought = bool(
                    delta_type in ("thought", "thought_summary")
                    or active_step_type == "thought"
                    or "thought" in delta
                    or "summary" in delta
                    or "thought_summary" in delta
                )

                if is_thought:
                    raw_th = delta.get("thought")
                    th = (
                        _extract_text(delta.get("content"))
                        or _extract_text(raw_th if not isinstance(raw_th, bool) else None)
                        or _extract_text(delta.get("summary"))
                        or _extract_text(delta.get("text"))
                    )
                    if th:
                        reasoning_text += th
                        if stream_callback:
                            stream_callback({"type": "reasoning", "delta": th})

                # 2. Function call / arguments_delta events
                elif (
                    delta_type in ("function_call", "arguments_delta", "arguments", "args")
                    or "function_call" in delta
                    or "arguments_delta" in delta
                    or "args_delta" in delta
                    or "partial_arguments" in delta
                    or (active_step_type == "function_call" and ("arguments" in delta or "partial_arguments" in delta))
                ):
                    fc_data = (
                        delta.get("function_call")
                        if isinstance(delta.get("function_call"), dict)
                        else delta
                    )
                    call_id = (
                        fc_data.get("call_id")
                        or fc_data.get("id")
                        or matched_step.get("call_id")
                        or delta_sid
                        or matched_step.get("id")
                    )
                    if not call_id:
                        if tool_calls and not fc_data.get("name"):
                            call_id = list(tool_calls.keys())[-1]
                        else:
                            call_id = f"call_{uuid.uuid4().hex[:8]}"
                    call_id = str(call_id)
                    fn_name = fc_data.get("name") or matched_step.get("name") or ""
                    args_delta = (
                        fc_data.get("partial_arguments")
                        or fc_data.get("arguments_delta")
                        or fc_data.get("args_delta")
                        or fc_data.get("arguments")
                        or fc_data.get("args")
                        or ""
                    )
                    if isinstance(args_delta, dict):
                        args_delta = json.dumps(args_delta)

                    entry = tool_calls.setdefault(
                        call_id,
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": ""},
                            "index": len(tool_calls),
                        },
                    )
                    if fn_name and not entry["function"]["name"]:
                        entry["function"]["name"] = fn_name
                    if args_delta:
                        entry["function"]["arguments"] += args_delta

                    sig = (
                        fc_data.get("thought_signature")
                        or fc_data.get("signature")
                        or d_sig
                    )
                    if sig and not entry.get("thought_signature"):
                        entry["thought_signature"] = str(sig)

                    if stream_callback:
                        stream_callback(
                            {
                                "type": "tool_call",
                                "index": entry["index"],
                                "id": call_id,
                                "name": entry["function"]["name"],
                                "name_delta": fn_name,
                                "arguments_delta": args_delta,
                            }
                        )

                # 3. Text deltas
                elif delta_type in ("text", "model_output", "model_generation") or "text" in delta or active_step_type == "model_output":
                    t = (
                        _extract_text(delta.get("text"))
                        or _extract_text(delta.get("content"))
                        or _extract_text(delta.get("output"))
                    )
                    if t:
                        content += t
                        if stream_callback:
                            stream_callback({"type": "content", "delta": t})

            # Handle direct step completions or full step items in the event / interaction
            steps = event.get("steps") or interaction_obj.get("steps") or []
            if isinstance(steps, list):
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    stype = s.get("type") or s.get("step_type")
                    s_sig = s.get("signature") or s.get("thought_signature")
                    if s_sig:
                        last_thought_signature = str(s_sig)
                    if stype in ("model_output", "model_generation") and not content:
                        out_t = _extract_text(s.get("content")) or _extract_text(s.get("output")) or _extract_text(s.get("text"))
                        if out_t:
                            content += out_t
                    elif stype == "thought" and not reasoning_text:
                        th_t = _extract_text(s.get("summary")) or _extract_text(s.get("content"))
                        if th_t:
                            reasoning_text += th_t
                    elif stype == "function_call":
                        cid = str(
                            s.get("call_id")
                            or s.get("id")
                            or s.get("step_id")
                            or f"call_{uuid.uuid4().hex[:8]}"
                        )
                        fname = s.get("name") or ""
                        fargs = s.get("args") or s.get("arguments") or {}
                        arg_str = json.dumps(fargs) if isinstance(fargs, dict) else str(fargs)
                        tc_entry = {
                            "id": cid,
                            "type": "function",
                            "function": {"name": fname, "arguments": arg_str},
                            "index": tool_calls.get(cid, {}).get("index", len(tool_calls)),
                        }
                        sig = s_sig or last_thought_signature
                        if sig:
                            tc_entry["thought_signature"] = str(sig)
                        tool_calls[cid] = tc_entry
            if (
                event_type_str in ("interaction.completed", "interaction.complete", "response.completed")
                and finish_reason is not None
                and last_usage is not None
            ):
                break

    except Exception as err:
        _force_close_response(response)
        if isinstance(err, RequestCancelled):
            raise
        if stop_event and stop_event.is_set():
            raise RequestCancelled(f"{provider_label} request cancelled.") from err
        had_partial = bool(content or tool_calls or reasoning_text)
        if finish_reason is not None and had_partial:
            pass
        else:
            raise StreamResponseError(
                f"{provider_label} streaming error: {err}",
                retryable=not (content or tool_calls),
            ) from err

    if stop_event and stop_event.is_set():
        _force_close_response(response)
        raise RequestCancelled(f"{provider_label} request cancelled.")

    # Fallback tag cleanup
    content, extracted_reasoning = extract_think_tags(content, reasoning_text)
    if extracted_reasoning:
        reasoning_text = extracted_reasoning

    cleaned_content, text_calls = extract_text_tool_calls(content)
    content = cleaned_content
    if not tool_calls and text_calls:
        for idx, call in enumerate(text_calls):
            cid = call.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            call["id"] = cid
            tool_calls[cid] = call

    finish_norm = str(finish_reason or "").strip().lower()
    if finish_norm in ("length", "max_tokens"):
        raise CompletionTruncatedError(
            f"{provider_label} completion was truncated (finish_reason={finish_reason!r}); "
            "token limit was reached before completion. Consider increasing 'llm.max_completion_tokens' "
            "in config.yaml or reducing 'reasoning_effort'.",
            partial_content=content,
            reasoning_text=reasoning_text,
            partial_tool_calls=list(tool_calls.values()) if tool_calls else [],
            provider_label=provider_label,
        )

    if not content and not tool_calls:
        if reasoning_text.strip() and (finish_reason is None or finish_norm in ("stop", "completed", "end_turn")):
            content = ""
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"{provider_label} returned an empty completion (finish_reason={reason!r})."
            )

    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        for tc in tool_calls.values():
            attach_thought_signature(tc, tc.get("thought_signature") or last_thought_signature)
        message["tool_calls"] = list(tool_calls.values())
    if reasoning_text.strip():
        clean_r = strip_encrypted_reasoning(reasoning_text)
        if clean_r:
            message["reasoning"] = clean_r

    res: dict[str, Any] = {
        "choices": [{"message": message}],
        "usage": last_usage,
    }
    if interaction_id:
        res["id"] = interaction_id
    return res


def parse_interactions_response(body: dict[str, Any]) -> dict[str, Any]:
    """Parse non-streaming JSON response from the Interactions API."""
    content = ""
    reasoning_text = ""
    tool_calls: list[dict[str, Any]] = []
    last_thought_signature: str | None = None

    steps = body.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type") or step.get("step_type")
        step_sig = step.get("signature") or step.get("thought_signature")
        if step_sig:
            last_thought_signature = str(step_sig)
        if step_type in ("model_output", "model_generation"):
            out_t = _extract_text(step.get("content")) or _extract_text(step.get("output")) or _extract_text(step.get("text"))
            if out_t:
                content += out_t
        elif step_type in ("thought", "thought_summary"):
            th_t = _extract_text(step.get("summary")) or _extract_text(step.get("content")) or _extract_text(step.get("thought"))
            if th_t:
                reasoning_text += th_t
        elif step_type == "function_call":
            call_id = step.get("call_id") or step.get("id") or step.get("step_id") or f"call_{uuid.uuid4().hex[:8]}"
            fn_name = step.get("name") or ""
            args = step.get("args") or step.get("arguments") or {}
            arg_str = json.dumps(args) if isinstance(args, dict) else str(args)
            tc_dict: dict[str, Any] = {
                "id": call_id,
                "type": "function",
                "function": {"name": fn_name, "arguments": arg_str},
            }
            if step_sig or last_thought_signature:
                tc_dict["thought_signature"] = step_sig or last_thought_signature
            tool_calls.append(tc_dict)

    usage_meta = body.get("usage") or body.get("usage_metadata")
    usage: dict[str, Any] | None = None
    if isinstance(usage_meta, dict):
        p_tokens = (
            usage_meta.get("prompt_tokens")
            or usage_meta.get("prompt_token_count")
            or 0
        )
        c_tokens = (
            usage_meta.get("completion_tokens")
            or usage_meta.get("candidates_token_count")
            or 0
        )
        r_tokens = (
            usage_meta.get("reasoning_tokens")
            or usage_meta.get("thoughts_token_count")
            or 0
        )
        cached_tokens = (
            usage_meta.get("cached_content_token_count")
            or usage_meta.get("cached_tokens")
            or (usage_meta.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        usage = {
            "prompt_tokens": int(p_tokens),
            "completion_tokens": int(c_tokens),
            "reasoning_tokens": int(r_tokens),
            "completion_tokens_details": {
                "reasoning_tokens": int(r_tokens),
            },
            "prompt_tokens_details": {
                "cached_tokens": int(cached_tokens),
            },
        }

    content, extracted_reasoning = extract_think_tags(content, reasoning_text)
    if extracted_reasoning:
        reasoning_text = extracted_reasoning

    cleaned_content, text_calls = extract_text_tool_calls(content)
    content = cleaned_content
    if not tool_calls and text_calls:
        for tc in text_calls:
            if not tc.get("id"):
                tc["id"] = f"call_{uuid.uuid4().hex[:8]}"
        tool_calls = text_calls

    finish_reason = body.get("finish_reason") or body.get("status")
    finish_norm = str(finish_reason or "").strip().lower()
    if finish_norm in ("length", "max_tokens"):
        raise CompletionTruncatedError(
            f"Google AI Studio completion was truncated (finish_reason={finish_reason!r}); "
            "token limit was reached before completion. Consider increasing 'llm.max_completion_tokens' "
            "in config.yaml or reducing 'reasoning_effort'.",
            partial_content=content,
            reasoning_text=reasoning_text,
            partial_tool_calls=tool_calls,
            provider_label="Google AI Studio",
        )

    if not content and not tool_calls:
        if reasoning_text.strip() and (finish_reason is None or finish_norm in ("stop", "completed", "end_turn")):
            content = ""
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"Google AI Studio returned an empty completion (finish_reason={reason!r})."
            )

    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        for tc in tool_calls:
            attach_thought_signature(tc, tc.get("thought_signature") or last_thought_signature)
        message["tool_calls"] = tool_calls
    if reasoning_text.strip():
        clean_r = strip_encrypted_reasoning(reasoning_text)
        if clean_r:
            message["reasoning"] = clean_r

    result: dict[str, Any] = {
        "choices": [{"message": message}],
        "usage": usage,
    }
    if "id" in body:
        result["id"] = body["id"]
    return result


class GoogleInteractionsClient(BaseLLMClient):
    """Google AI Studio Interactions API adapter."""

    preserve_reasoning: bool = True
    requires_api_key: bool = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="Google AI Studio")
        self._last_interaction_id: str | None = None

    @staticmethod
    def _api_key() -> str:
        key = os.getenv(PRIMARY_API_KEY_ENV, "").strip()
        if not key:
            key = os.getenv(FALLBACK_API_KEY_ENV, "").strip()
        return key

    def _endpoint(self) -> str:
        base = (self.settings.gemini_base_url or DEFAULT_BASE_URL).rstrip("/")
        if base.endswith(("/v1beta", "/v1")):
            return f"{base}/interactions"
        return f"{base}/v1beta/interactions"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        wire_messages = self.sanitize_messages(messages)
        system_instruction, steps = convert_messages_to_interactions(wire_messages)
        declarations = convert_tools_to_declarations(tools)

        # The Interactions API expects bare model IDs (no OpenRouter / REST prefixes).
        raw_model = (self.settings.gemini_model or "").strip().removeprefix("google/").removeprefix("models/")

        payload: dict[str, Any] = {
            "model": raw_model,
            "input": steps,
            "stream": True,
            "thinking_summaries": "auto",
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if declarations:
            payload["tools"] = declarations
        if not self.settings.gemini_store:
            payload["store"] = False
        else:
            payload["store"] = True

        gen_config: dict[str, Any] = {"thinking_summaries": "auto"}
        if self.settings.llm_max_completion_tokens:
            # Google GenAI API enforces maxOutputTokens < 65536.
            gen_config["max_output_tokens"] = min(self.settings.llm_max_completion_tokens, 65535)

        model_name = raw_model.lower()
        is_gemini_2 = "2.5" in model_name or "gemini-2" in model_name
        effort = (self.settings.gemini_reasoning_effort or "").lower().strip()
        eff = effort if effort else ("minimal" if "flash" in model_name else "low")
        if is_continuation_messages(messages) and eff != "none":
            eff = "minimal" if "flash" in model_name else "low"

        thinking_config: dict[str, Any] = {
            "include_thoughts": eff != "none",
            "thinking_summaries": "auto",
        }
        if eff != "none":
            if not is_gemini_2:
                level_map = {
                    "minimal": "MINIMAL" if "flash" in model_name else "LOW",
                    "low": "LOW",
                    "medium": "MEDIUM",
                    "high": "HIGH",
                    "xhigh": "HIGH",
                    "max": "HIGH",
                }
                level = level_map.get(eff, "LOW")
                if "gemini-4" in model_name and level == "MINIMAL":
                    level = "LOW"
                thinking_config["thinking_level"] = level
            else:
                budget_map = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 4096,
                    "high": 6144,
                    "xhigh": 8192,
                    "max": 8192,
                }
                thinking_config["thinking_budget"] = budget_map.get(eff, 2048)
        elif eff == "none":
            thinking_config.pop("thinking_summaries", None)
            gen_config.pop("thinking_summaries", None)
            if not is_gemini_2:
                thinking_config["thinking_level"] = (
                    "MINIMAL"
                    if ("flash" in model_name and "gemini-4" not in model_name)
                    else "LOW"
                )
            elif "pro" not in model_name:
                thinking_config["thinking_budget"] = 0

        gen_config["thinking_config"] = thinking_config
        if gen_config:
            payload["generation_config"] = gen_config

        return payload

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        url = self._endpoint()
        separator = "&" if "?" in url else "?"
        url_with_param = f"{url}{separator}alt=sse"
        return post_with_cancel(
            url=url_with_param,
            payload=payload,
            headers=headers,
            timeout_seconds=self.settings.gemini_timeout_seconds,
            stop_event=self.stop_event,
        )

    def _stream_response(self, response: Any) -> dict[str, Any]:
        with self._response_lock:
            self._active_response = response
        try:
            result = parse_interactions_stream(
                response,
                provider_label=self._provider_label,
                stop_event=self.stop_event,
                stream_callback=self.stream_callback,
            )
            usage = result.get("usage") if isinstance(result, dict) else None
            self.last_usage = usage if isinstance(usage, dict) else None
            if isinstance(result, dict) and "id" in result:
                self._last_interaction_id = result["id"]
            return result
        finally:
            with self._response_lock:
                self._active_response = None
            if self.stop_event is not None and self.stop_event.is_set():
                _force_close_response(response)
            elif response is not None:
                with contextlib.suppress(Exception):
                    response.close()

    def _parse_non_stream_response(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "read") and callable(response.read):
            with contextlib.suppress(Exception):
                response.read()
        body = response.json()
        parsed_body = parse_interactions_response(body)
        self.last_usage = parsed_body.get("usage")
        if "id" in parsed_body:
            self._last_interaction_id = parsed_body["id"]
        return parsed_body

    def _try_image_fallback(self, payload: dict[str, Any], response: Any) -> bool:
        """Strip image parts from payload['input'] when visual input is rejected."""
        steps = payload.get("input")
        if not isinstance(steps, list):
            return False

        removed = False
        new_steps: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("type") == "image":
                removed = True
                continue
            if step.get("type") == "user_input" and isinstance(step.get("content"), list):
                filtered_content = [
                    p
                    for p in step["content"]
                    if isinstance(p, dict) and p.get("type") != "image"
                ]
                if len(filtered_content) != len(step["content"]):
                    removed = True
                if filtered_content:
                    step_copy = deepcopy(step)
                    step_copy["content"] = filtered_content
                    new_steps.append(step_copy)
            else:
                new_steps.append(step)

        if removed:
            response.close()
            payload["input"] = new_steps
        return removed

    def _summarize_payload_for_logging(
        self, payload: dict[str, Any], is_dataset_mode: bool
    ) -> dict[str, Any] | None:
        attempt_payload = deepcopy(payload)
        if isinstance(attempt_payload, dict) and isinstance(
            attempt_payload.get("input"), list
        ):
            attempt_payload["input"] = summarize_llm_messages(
                attempt_payload["input"],
                lossless=is_dataset_mode,
            )
        return attempt_payload


# Backward-compatible alias
GeminiClient = GoogleInteractionsClient
