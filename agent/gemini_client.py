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
    RequestCancelled,
    StreamResponseError,
    _force_close_response,
    _stream_error_detail,
    extract_text_tool_calls,
    extract_think_tags,
    iter_lines_with_cancel,
    normalize_messages,
    post_with_cancel,
    strip_encrypted_reasoning,
)
from agent.prompt import TOOL_IMAGE_PROMPT
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
        elif "function_declarations" in tool:
            for decl in tool["function_declarations"]:
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
    messages = normalize_messages(messages)
    system_instruction: str | None = None
    steps: list[dict[str, Any]] = []
    call_id_to_name: dict[str, str] = {}
    pending_tool_images: list[dict[str, Any]] = []

    def _flush_pending_images() -> None:
        nonlocal pending_tool_images
        if pending_tool_images:
            steps.append(
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": TOOL_IMAGE_PROMPT}] + pending_tool_images,
                }
            )
            pending_tool_images = []

    for msg in messages:
        role = msg.get("role")
        if role != "tool":
            _flush_pending_images()

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
                                    "inline_data": {
                                        "mime_type": mime,
                                        "data": b64_data,
                                    },
                                }
                            )
                        elif url:
                            parts.append({"type": "image", "image_url": url})
            if parts:
                steps.append({"type": "user_input", "content": parts})

        elif role == "assistant":
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
            tool_calls = msg.get("tool_calls") or []
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
                step_item = {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": fn_name,
                    "arguments": args,
                    "args": args,
                }
                sig = tc.get("thought_signature") or "skip_thought_signature_validator"
                if sig:
                    step_item["thought_signature"] = sig
                steps.append(step_item)

        elif role == "tool":
            call_id = msg.get("tool_call_id") or ""
            fn_name = msg.get("name") or call_id_to_name.get(call_id, "unknown_tool")
            content = msg.get("content")
            image_steps: list[dict[str, Any]] = []
            if isinstance(content, list):
                text_parts: list[str] = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "text":
                        t = p.get("text", "")
                        if t:
                            text_parts.append(t)
                    elif ptype == "image_url":
                        img_dict = p.get("image_url") or {}
                        url = img_dict.get("url", "")
                        parsed = _parse_data_uri(url)
                        if parsed:
                            mime, b64_data = parsed
                            image_steps.append({
                                "type": "image",
                                "inline_data": {
                                    "mime_type": mime,
                                    "data": b64_data,
                                },
                            })
                        elif url:
                            image_steps.append({"type": "image", "image_url": url})
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
            if image_steps:
                pending_tool_images.extend(image_steps)

    _flush_pending_images()
    if steps and steps[-1].get("type") in ("model_output", "thought", "function_call"):
        steps.append({
            "type": "user_input",
            "content": [{"type": "text", "text": "Continue."}],
        })
    return system_instruction, steps


def parse_interactions_stream(
    response: Any,
    *,
    provider_label: str = "Google AI Studio",
    stop_event: threading.Event | None = None,
    stream_callback: Any | None = None,
) -> dict[str, Any]:
    """Parse Server-Sent Events from the Interactions API."""
    content = ""
    reasoning_text = ""
    tool_calls: dict[str, dict[str, Any]] = {}
    last_usage: dict[str, Any] | None = None
    interaction_id: str | None = None
    finish_reason: str | None = None
    finish_time: float | None = None

    try:
        for raw_line in iter_lines_with_cancel(response, stop_event=stop_event):
            if stop_event and stop_event.is_set():
                _force_close_response(response)
                raise RequestCancelled(f"{provider_label} request cancelled.")
            if raw_line is None:
                if finish_reason is not None and finish_time is not None:
                    if time.monotonic() - finish_time >= 1.5:
                        break
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line:
                continue
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("[DONE]", ""):
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{provider_label} returned malformed streaming JSON.") from error

            if not isinstance(event, dict):
                continue

            if "error" in event:
                raise StreamResponseError(
                    f"{provider_label} stream failed: {_stream_error_detail(event['error'])}",
                    retryable=not (content or tool_calls or reasoning_text),
                )

            if "id" in event:
                interaction_id = event["id"]

            if event.get("finish_reason"):
                if finish_reason is None:
                    finish_time = time.monotonic()
                finish_reason = str(event["finish_reason"])
            elif event.get("status"):
                status_val = str(event["status"])
                if finish_reason is None and status_val == "completed":
                    finish_time = time.monotonic()
                finish_reason = status_val

            usage_meta = event.get("usage") or event.get("usage_metadata")
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
                    "prompt_tokens_details": {
                        "cached_tokens": int(cached_tokens),
                    },
                }

            event_type = event.get("event_type") or event.get("type")
            delta = event.get("delta")

            # Handle step.delta events
            if (event_type == "step.delta" or delta) and isinstance(delta, dict):
                delta_type = delta.get("type")
                # 1. Text deltas
                if delta_type == "text" or "text" in delta:
                    t = delta.get("text", "")
                    if isinstance(t, str) and t:
                        content += t
                        if stream_callback:
                            stream_callback({"type": "content", "delta": t})

                # 2. Thought deltas
                elif delta_type == "thought" or "thought" in delta:
                    th = delta.get("thought", "")
                    if isinstance(th, str) and th:
                        reasoning_text += th
                        if stream_callback:
                            stream_callback({"type": "reasoning", "delta": th})

                # 3. Function call deltas
                elif delta_type == "function_call" or "function_call" in delta:
                    fc_data = (
                        delta.get("function_call")
                        if isinstance(delta.get("function_call"), dict)
                        else delta
                    )
                    call_id = (
                        fc_data.get("call_id")
                        or fc_data.get("id")
                        or event.get("step_id")
                    )
                    if not call_id:
                        if tool_calls and not fc_data.get("name"):
                            call_id = list(tool_calls.keys())[-1]
                        else:
                            call_id = f"call_{uuid.uuid4().hex[:8]}"
                    fn_name = fc_data.get("name") or ""
                    args_delta = (
                        fc_data.get("arguments_delta")
                        or fc_data.get("args_delta")
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

                    sig = fc_data.get("thought_signature") or delta.get("thought_signature")
                    if sig and not entry.get("thought_signature"):
                        entry["thought_signature"] = sig

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

            # Handle direct step completions or full step items in the event
            steps = event.get("steps") or []
            if isinstance(steps, list):
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    stype = s.get("type")
                    if stype == "model_output" and not content:
                        parts = s.get("content") or []
                        for p in parts:
                            if isinstance(p, dict) and p.get("type") == "text":
                                content += p.get("text", "")
                    elif stype == "thought" and not reasoning_text:
                        parts = s.get("content") or []
                        for p in parts:
                            if isinstance(p, dict) and p.get("type") == "text":
                                reasoning_text += p.get("text", "")
                    elif stype == "function_call":
                        cid = s.get("call_id") or s.get("id") or s.get("step_id") or f"call_{uuid.uuid4().hex[:8]}"
                        fname = s.get("name") or ""
                        fargs = s.get("args") or s.get("arguments") or {}
                        arg_str = json.dumps(fargs) if isinstance(fargs, dict) else str(fargs)
                        tc_entry = {
                            "id": cid,
                            "type": "function",
                            "function": {"name": fname, "arguments": arg_str},
                            "index": tool_calls.get(cid, {}).get("index", len(tool_calls)),
                        }
                        sig = s.get("thought_signature")
                        if sig:
                            tc_entry["thought_signature"] = sig
                        tool_calls[cid] = tc_entry
            if finish_reason is not None and last_usage is not None:
                break

    except Exception as err:
        _force_close_response(response)
        if isinstance(err, RequestCancelled):
            raise
        if stop_event and stop_event.is_set():
            raise RequestCancelled(f"{provider_label} request cancelled.") from err
        had_partial = bool(content or tool_calls or reasoning_text)
        raise StreamResponseError(
            f"{provider_label} streaming error: {err}",
            retryable=not had_partial,
        ) from err

    if stop_event and stop_event.is_set():
        _force_close_response(response)
        raise RequestCancelled(f"{provider_label} request cancelled.")

    if finish_reason in ("length", "MAX_TOKENS"):
        raise RuntimeError(
            f"{provider_label} completion was truncated (finish_reason={finish_reason!r}); "
            "no tool calls were executed. Consider increasing 'llm.max_completion_tokens' "
            "in config.yaml or reducing 'reasoning_effort'."
        )

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

    if not content and not tool_calls:
        if reasoning_text.strip() and finish_reason in (None, "stop", "completed"):
            content = (
                "Model completed its reasoning process but did not produce an action or final response."
            )
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"{provider_label} returned an empty completion (finish_reason={reason!r})."
            )

    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        for tc in tool_calls.values():
            if not tc.get("thought_signature"):
                tc["thought_signature"] = "skip_thought_signature_validator"
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

    steps = body.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_type = step.get("type")
        if step_type == "model_output":
            parts = step.get("content") or []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    content += part.get("text", "")
        elif step_type == "thought":
            parts = step.get("content") or []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    reasoning_text += part.get("text", "")
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
            sig = step.get("thought_signature") or "skip_thought_signature_validator"
            if sig:
                tc_dict["thought_signature"] = sig
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
            if not tc.get("thought_signature"):
                tc["thought_signature"] = "skip_thought_signature_validator"
        tool_calls = text_calls

    finish_reason = body.get("finish_reason") or body.get("status")
    if finish_reason in ("length", "MAX_TOKENS"):
        raise RuntimeError(
            f"Google AI Studio completion was truncated (finish_reason={finish_reason!r}); "
            "no tool calls were executed. Consider increasing 'llm.max_completion_tokens' "
            "in config.yaml or reducing 'reasoning_effort'."
        )

    if not content and not tool_calls:
        if reasoning_text.strip() and finish_reason in (None, "stop", "completed"):
            content = (
                "Model completed its reasoning process but did not produce an action or final response."
            )
        else:
            reason = finish_reason or "unknown"
            raise RuntimeError(
                f"Google AI Studio returned an empty completion (finish_reason={reason!r})."
            )

    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        for tc in tool_calls:
            if not tc.get("thought_signature"):
                tc["thought_signature"] = "skip_thought_signature_validator"
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

        payload: dict[str, Any] = {
            "model": self.settings.gemini_model,
            "input": steps,
            "stream": True,
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if declarations:
            payload["tools"] = declarations
        if not self.settings.gemini_store:
            payload["store"] = False
        else:
            payload["store"] = True

        gen_config: dict[str, Any] = {}
        if self.settings.llm_max_completion_tokens:
            gen_config["max_output_tokens"] = self.settings.llm_max_completion_tokens

        model_name = (self.settings.gemini_model or "").lower()
        is_gemini_3 = "3." in model_name or "gemini-3" in model_name
        effort = (self.settings.gemini_reasoning_effort or "").lower().strip()

        thinking_config: dict[str, Any] = {"include_thoughts": True}
        if effort:
            if is_gemini_3:
                level_map = {
                    "low": "LOW",
                    "medium": "MEDIUM",
                    "high": "HIGH",
                    "minimal": "MINIMAL",
                }
                thinking_config["thinking_level"] = level_map.get(effort, effort.upper())
            else:
                budget_map = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 4096,
                    "high": 8192,
                }
                thinking_config["thinking_budget"] = budget_map.get(effort, 8192)
        elif is_gemini_3:
            thinking_config["thinking_level"] = "HIGH"
        else:
            thinking_config["thinking_budget"] = 8192

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
