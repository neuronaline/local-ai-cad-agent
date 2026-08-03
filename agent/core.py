"""Background OpenRouter tool-calling loop for one local CAD task at a time."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agent.constraints import ConstraintStore, ModelConstraintValidator
from agent.images import as_openrouter_image
from agent.openrouter import OpenRouterClient, sanitize_assistant_message
from agent.prompt import get_system_prompt
from agent.revisions import RevisionStore
from agent.settings import Settings
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_tool import CadTool
from agent.tools.experience_tool import ExperienceTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import QuestionTool, normalize_questions
from agent.tools.question_validator import QuestionValidator
from agent.tools.terminal_tool import TerminalTool


class ProjectTools:
    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None],
        settings: Settings | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.revisions = RevisionStore(
            project_dir,
            retention_count=settings.revision_retention_count if settings else 0,
        )
        self.constraints = ConstraintStore(project_dir)
        # Reconcile on project load: import existing model or recover from crash.
        self.revisions.reconcile()
        self.file = FileTool(project_dir, self.revisions, constraints=self.constraints)
        self.terminal = TerminalTool(project_dir)
        self.cad = CadTool(project_dir, publish, self.revisions)
        self.question = QuestionTool(publish)
        # Experience tool lives at workspace scope, so we need the workspace root.
        # project_dir is <workspace>/<project>; parent yields the workspace.
        self.experience = ExperienceTool(project_dir.parent, project_dir.name)

    def stop(self) -> None:
        self.terminal.stop()
        self.cad.stop()


class AgentRunner:
    def __init__(
        self, settings: Settings, publish: Callable[[str, dict], None]
    ) -> None:
        self.settings = settings
        self.publish = publish
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_tools: ProjectTools | None = None
        self._active_project: str | None = None
        self._lock = threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}
        self._preview_attempts: dict[tuple[str, str], dict[str, str]] = {}
        self._pending_completions: dict[str, dict[str, str]] = {}

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def active_project(self) -> str | None:
        with self._lock:
            return self._active_project

    def is_awaiting_preview(self, project: str | None = None) -> bool:
        with self._lock:
            return bool(
                self._pending_completions
                if project is None
                else project in self._pending_completions
            )

    def pending_preview_id(self, project: str) -> str | None:
        with self._lock:
            pending = self._pending_completions.get(project)
            return pending["preview_id"] if pending else None

    def confirm_preview(self, project: str, preview_id: str) -> bool:
        completion = None
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt or not self._preview_matches(project, attempt):
                return False
            attempt["status"] = "displayed"
            pending = self._pending_completions.get(project)
            if pending and pending["preview_id"] == preview_id:
                completion = self._pending_completions.pop(project)
        if completion:
            self._complete(project, completion["message"])
        return True

    def fail_preview(self, project: str, preview_id: str, message: str) -> bool:
        error_message = (
            f"Drawing could not be displayed: {message or 'the preview is invalid.'}"
        )
        should_publish = False
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt:
                return False
            attempt.update({"status": "error", "error": error_message})
            pending = self._pending_completions.get(project)
            if pending and pending["preview_id"] == preview_id:
                self._pending_completions.pop(project)
                should_publish = True
        if should_publish:
            self.publish("agent_error", {"project": project, "message": error_message})
        return True

    def waiting_question(self, project: str) -> dict[str, object] | None:
        with self._lock:
            question = self._waiting_questions.get(project)
        if question:
            return question.copy()
        state_path = self.settings.workspace_root / project / ".agent_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        question = state.get("waiting_question") if isinstance(state, dict) else None
        if not isinstance(question, dict):
            return None
        with self._lock:
            self._waiting_questions[project] = question
        return question.copy()

    def start(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
    ) -> bool:
        if self.waiting_question(project):
            return False
        with self._lock:
            return self._start_locked(project, message, image_paths or [])

    def _start_locked(
        self, project: str, message: str, image_paths: list[Path]
    ) -> bool:
        if (self._thread and self._thread.is_alive()) or self._pending_completions:
            return False
        self._stop_event.clear()
        self._active_project = project
        self._thread = threading.Thread(
            target=self._run, args=(project, message, image_paths), daemon=True
        )
        self._thread.start()
        return True

    def answer(self, project: str, answer: str) -> bool:
        question = self.waiting_question(project)
        if not question or not self._validate_answer(question, answer):
            return False
        with self._lock:
            previous_thread = self._thread
        if previous_thread and previous_thread.is_alive():
            previous_thread.join(timeout=2)
        project_dir = self.settings.workspace_root / project
        formatted = self._format_answer(question, answer)
        with self._lock:
            if (self._thread and self._thread.is_alive()) or self._pending_completions:
                return False
            self._log(project_dir, "user", formatted)
            if not self._start_locked(project, formatted, []):
                return False
            self._waiting_questions.pop(project, None)
            (project_dir / ".agent_state.json").unlink(missing_ok=True)
            return True

    def stop(self, project: str | None = None) -> None:
        with self._lock:
            target_project = project or self._active_project
            stop_active_task = (
                target_project is None or target_project == self._active_project
            )
            if target_project:
                self._waiting_questions.pop(target_project, None)
                self._pending_completions.pop(target_project, None)
                stale_attempts = [
                    key for key in self._preview_attempts if key[0] == target_project
                ]
                for key in stale_attempts:
                    self._preview_attempts.pop(key, None)
            if stop_active_task:
                self._stop_event.set()
            if stop_active_task and self._active_tools:
                self._active_tools.stop()
        if target_project:
            (
                self.settings.workspace_root / target_project / ".agent_state.json"
            ).unlink(missing_ok=True)

    def _run(
        self,
        project: str,
        message: str,
        image_paths: list[Path] | None = None,
    ) -> None:
        self.publish(
            "agent_status",
            {
                "project": project,
                "status": "started",
                "message": "Planning CAD task...",
            },
        )
        project_dir = self.settings.workspace_root / project
        tools = ProjectTools(project_dir, self.publish, self.settings)
        with self._lock:
            self._active_tools = tools
        try:
            messages = self._context(project_dir, message, image_paths or [])
            preview_id: str | None = None
            cad_error: str | None = None
            cad_fix_required = not self._model_is_built(project_dir)
            any_tool_used = False
            nudged_cad = False
            # Keep one-argument construction compatible with test doubles and older integrations.
            client = OpenRouterClient(self.settings)
            client.stop_event = self._stop_event
            client.session_id = f"{self.settings.openrouter_session_prefix}:{project}"
            for _ in range(self.settings.agent_tool_call_limit):
                if self._stop_event.is_set():
                    self.publish(
                        "agent_status",
                        {
                            "project": project,
                            "status": "stopped",
                            "message": "Task stopped.",
                        },
                    )
                    return
                message_id = uuid.uuid4().hex
                self.publish(
                    "agent_stream_start", {"project": project, "message_id": message_id}
                )

                def publish_stream(
                    event: dict,
                    current_message_id: str = message_id,
                ) -> None:
                    event_type = event.pop("type")
                    self.publish(
                        f"agent_{event_type}_delta",
                        {"project": project, "message_id": current_message_id, **event},
                    )

                client.stream_callback = publish_stream
                response = client.chat(messages, TOOL_SCHEMAS)
                self._publish_usage(project, getattr(client, "last_usage", None))
                assistant_message = sanitize_assistant_message(
                    response["choices"][0]["message"]
                )
                tool_calls = self._normalize_tool_calls(
                    assistant_message.get("tool_calls")
                )
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                else:
                    assistant_message.pop("tool_calls", None)
                self.publish(
                    "agent_stream_end",
                    {
                        "project": project,
                        "message_id": message_id,
                        "message": assistant_message.get("content") or "",
                    },
                )
                messages.append(assistant_message)
                self._append_api_message(project_dir, assistant_message)
                if not tool_calls:
                    content = assistant_message.get("content") or "Task completed."
                    if not preview_id:
                        if cad_error:
                            self.publish(
                                "agent_error",
                                {
                                    "project": project,
                                    "message": f"Drawing was not created: {cad_error}",
                                },
                            )
                        elif any_tool_used:
                            if not nudged_cad and (project_dir / "model.py").is_file():
                                nudged_cad = True
                                reminder = {
                                    "role": "user",
                                    "content": (
                                        "model.py exists but it has not been verified. "
                                        "Call cad_build_and_verify now."
                                    ),
                                }
                                messages.append(reminder)
                                self._append_api_message(project_dir, reminder)
                                continue
                            self.publish(
                                "agent_error",
                                {
                                    "project": project,
                                    "message": "Drawing was not created: the task did not produce a new CAD preview.",
                                },
                            )
                        else:
                            # No tools were used at all — just a conversation; complete silently.
                            self._complete(project, content)
                        return
                    self._await_preview(project, preview_id, content)
                    return
                any_tool_used = True
                needs_visual_review = False
                for call in tool_calls:
                    if self._stop_event.is_set():
                        return
                    (
                        preview_id,
                        cad_error,
                        cad_fix_required,
                        needs_critique,
                        waiting,
                    ) = self._process_tool_call(
                        tools,
                        project,
                        project_dir,
                        call,
                        cad_fix_required,
                        preview_id,
                        cad_error,
                        messages,
                    )
                    if needs_critique:
                        needs_visual_review = True
                    if waiting:
                        try:
                            q_args = json.loads(call["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            q_args = {}
                        questions_list = q_args.get("questions", [])
                        if isinstance(questions_list, list) and questions_list:
                            question_text = "; ".join(
                                q.get("question", "")
                                for q in questions_list
                                if isinstance(q, dict)
                            )
                        else:
                            question_text = q_args.get(
                                "question", "Clarification requested"
                            )
                        self._log(
                            project_dir, "assistant", f"Question: {question_text}"
                        )
                        self.publish(
                            "agent_status",
                            {
                                "project": project,
                                "status": "waiting_for_user",
                                "message": "Waiting for user input.",
                            },
                        )
                        return
                if needs_visual_review and not cad_fix_required:
                    critique = self._verification_context(project, project_dir)
                    messages.append(critique)
                    self._append_api_message(project_dir, critique)
            self.publish(
                "agent_error",
                {
                    "project": project,
                    "message": (
                        f"Tool-call limit ({self.settings.agent_tool_call_limit}) reached; "
                        "increase agent.tool_call_limit or continue with a narrower request."
                    ),
                },
            )
        except Exception as error:  # noqa: BLE001 - Surface all agent failures to the local UI.
            import traceback

            detail = str(error)
            err_type = type(error).__name__
            # Log the full traceback server-side.
            traceback.print_exc()
            if self._stop_event.is_set():
                self.publish(
                    "agent_status",
                    {
                        "project": project,
                        "status": "stopped",
                        "message": "Task stopped.",
                    },
                )
            else:
                self.publish(
                    "agent_error",
                    {
                        "project": project,
                        "message": self._user_error_message(detail, err_type),
                    },
                )
        finally:
            with self._lock:
                self._active_tools = None
                if self._active_project == project:
                    self._active_project = None

    def _context(
        self, project_dir: Path, message: str, image_paths: list[Path]
    ) -> list[dict]:
        history = self._load_api_history(project_dir)
        self._append_constraint_context(project_dir, history)
        user_message: dict[str, object] = {"role": "user", "content": message}
        if image_paths:
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": message}]
                + [as_openrouter_image(path) for path in image_paths],
            }
        if not history or history[-1] != user_message:
            history.append(user_message)
            self._append_api_message(project_dir, user_message)
        return [{"role": "system", "content": get_system_prompt()}] + history

    @classmethod
    def _append_constraint_context(cls, project_dir: Path, history: list[dict]) -> None:
        """Record constraint changes without mutating the cacheable system prefix."""
        tag = "<runtime_active_constraints>"
        summary = ModelConstraintValidator(ConstraintStore(project_dir)).constraint_summary()
        content = f"{tag}\n{summary or 'None.'}\n</runtime_active_constraints>"
        previous = next(
            (
                item.get("content")
                for item in reversed(history)
                if item.get("role") == "user"
                and isinstance(item.get("content"), str)
                and item["content"].startswith(tag)
            ),
            None,
        )
        if previous == content or (previous is None and not summary):
            return
        context = {"role": "user", "content": content}
        history.append(context)
        cls._append_api_message(project_dir, context)

    @staticmethod
    def _api_history_path(project_dir: Path) -> Path:
        return project_dir / "api_messages.jsonl"

    @classmethod
    def _load_api_history(cls, project_dir: Path) -> list[dict]:
        """Load the append-only, protocol-level history for cache-stable turns."""
        history: list[dict] = []
        history_path = cls._api_history_path(project_dir)
        if history_path.exists():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("role") in {
                    "user",
                    "assistant",
                    "tool",
                }:
                    history.append(item)
            return history

        # Projects created by earlier releases retain their readable transcript.
        # Convert it only once into the protocol history; never rewrite it.
        log_path = project_dir / "conversation.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("role") in {"user", "assistant"} and isinstance(
                    item.get("content"), str
                ):
                    history.append({"role": item["role"], "content": item["content"]})
        for item in history:
            cls._append_api_message(project_dir, item)
        return history

    @classmethod
    def _append_api_message(cls, project_dir: Path, message: dict) -> None:
        with cls._api_history_path(project_dir).open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    def _publish_usage(self, project: str, usage: dict | None) -> None:
        if not usage:
            return
        details = usage.get("prompt_tokens_details")
        cached_tokens = (
            details.get("cached_tokens", 0) if isinstance(details, dict) else 0
        )
        self.publish(
            "agent_usage",
            {
                "project": project,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached_tokens,
                "cache_write_tokens": details.get("cache_write_tokens", 0)
                if isinstance(details, dict)
                else 0,
            },
        )

    def _execute(
        self,
        tools: ProjectTools,
        project: str,
        name: str,
        args: dict,
        call_id: str = "",
    ) -> tuple[object, bool]:
        required_arguments = {
            "file_read": ("filename",),
            "file_write": ("filename", "content"),
            "file_replace": ("filename", "old", "new"),
            "file_regex_replace": ("filename", "pattern", "replacement"),
            "terminal_run": ("arguments",),
            "terminal_check": ("arguments",),
            "experience_search": ("query",),
            "experience_add": ("problem", "solution"),
            "experience_update": ("id",),
        }
        missing = [key for key in required_arguments.get(name, ()) if key not in args]
        if missing:
            raise ValueError(f"Missing required argument(s): {', '.join(missing)}.")

        if name == "cad_build_and_verify":
            return tools.cad.build_and_verify(), False
        if name.startswith("file_"):
            tool = tools.file.with_call_id(call_id) if call_id else tools.file
            operation = name.removeprefix("file_")
            if operation == "read":
                return tool.read(args["filename"]), False
            if operation == "write":
                return tool.write(args["filename"], args["content"]), False
            if operation == "replace":
                return tool.replace(args["filename"], args["old"], args["new"]), False
            return (
                tool.regex_replace(
                    args["filename"],
                    args["pattern"],
                    args["replacement"],
                    args.get("count", 1),
                ),
                False,
            )
        if name == "terminal_run":
            return tools.terminal.run(
                args["arguments"], args.get("timeout_seconds", 30)
            ), False
        if name == "terminal_check":
            return tools.terminal.check(
                args["arguments"], args.get("timeout_seconds", 15)
            ), False
        if name == "experience_search":
            return tools.experience.search(args["query"]), False
        if name == "experience_add":
            return tools.experience.add(
                args["problem"], args["solution"], args.get("tags")
            ), False
        if name == "experience_update":
            return tools.experience.update(
                args["id"],
                args.get("problem"),
                args.get("solution"),
                args.get("tags"),
            ), False

        # Legacy names remain dispatchable for append-only protocol histories and
        # older integrations, but are intentionally absent from TOOL_SCHEMAS.
        if name == "screenshot":
            view = args.get("view", "isometric")
            proximity = args.get("proximity", 1.0)
            return json.dumps(tools.cad.screenshot(view, proximity)), False
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
            with self._lock:
                self._waiting_questions[project] = question_state
            return result, True
        if name == "file" and call_id:
            # Bind the tool-call ID so revision manifests can correlate with
            # the protocol log without FileTool reading conversation logs.
            file_tool = tools.file.with_call_id(call_id)
            return file_tool.execute(args)
        tool = getattr(tools, name)
        return tool.execute(args)

    @staticmethod
    def _normalize_tool_calls(raw_calls: object) -> list[dict]:
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

    @staticmethod
    def _is_model_mutation(name: str, arguments: dict) -> bool:
        if name in {"file_write", "file_replace", "file_regex_replace"}:
            return arguments.get("filename") == "model.py"
        return (
            name == "file"
            and arguments.get("filename") == "model.py"
            and arguments.get("operation") in {"write", "replace", "regex_replace"}
        )

    @staticmethod
    def _is_cad_build(name: str, arguments: dict) -> bool:
        return name == "cad_build_and_verify" or (
            name == "cad" and arguments.get("operation") == "run"
        )

    def _process_tool_call(
        self,
        tools: ProjectTools,
        project: str,
        project_dir: Path,
        call: dict,
        cad_fix_required: bool,
        prev_preview_id: str | None,
        cad_error: str | None,
        messages: list[dict],
    ) -> tuple[str | None, str | None, bool, bool, bool]:
        """Execute a single tool call, update state, and feed visual results back.

        Returns:
            (preview_id, cad_error, cad_fix_required, needs_visual_review, waiting)
        """
        call = call if isinstance(call, dict) else {}
        call_id = str(call.get("id") or f"invalid-{uuid.uuid4().hex}")
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else ""
        if not isinstance(name, str) or not name:
            name = "unknown_tool"
        tool_succeeded = False
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
            self.publish("tool_status", {**tool_event, "status": "running"})
            if (
                name == "cad"
                and arguments.get("operation") == "inspect"
                and cad_fix_required
            ):
                raise RuntimeError(
                    "cad.inspect was skipped because model.py has not built successfully. "
                    "Run cad.run after fixing model.py."
                )
            if self._is_cad_build(name, arguments):
                preview_id = None
            raw_result, waiting = self._execute(
                tools, project, name, arguments, call_id
            )
            result = tool_success(name, raw_result)
            tool_succeeded = True
            if self._is_model_mutation(name, arguments):
                preview_id = None
                cad_error = None
                cad_fix_required = True
                self.publish("revision_updated", {"project": project})
            if self._is_cad_build(name, arguments):
                preview_id = self._register_preview(project, project_dir)
                cad_error = None
                cad_fix_required = False
                self.publish(
                    "preview_updated",
                    {"project": project, "preview_id": preview_id},
                )
            self.publish(
                "tool_status",
                {**tool_event, "status": "completed", "result": result},
            )
        except Exception as error:  # noqa: BLE001 - Tool errors are useful LLM context.
            result, waiting = tool_failure(name, error), False
            if self._is_cad_build(name, arguments):
                cad_error = str(error)
                cad_fix_required = True
                preview_id = None
            self.publish(
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
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        self._append_api_message(project_dir, messages[-1])
        self._log(
            project_dir,
            "tool",
            {
                "call_id": call_id,
                "name": name,
                "operation": arguments.get("operation"),
                "result": result,
            },
        )
        # Post-processing hooks: visual feed and verification trigger.
        needs_visual_review = self._post_process_tool_result(
            project_dir,
            name,
            arguments,
            tool_succeeded,
            waiting,
            cad_fix_required,
            messages,
        )
        return preview_id, cad_error, cad_fix_required, needs_visual_review, waiting

    @staticmethod
    def _post_process_tool_result(
        project_dir: Path,
        name: str,
        arguments: dict,
        tool_succeeded: bool,
        waiting: bool,
        cad_fix_required: bool,
        messages: list[dict],
    ) -> bool:
        """Run post-tool-call hooks: screenshot visual feed and self-critique trigger.

        Returns True if a self-critique should follow this tool call.
        """
        # Feed screenshot result back as a user message with visual content.
        if tool_succeeded and name == "screenshot":
            ss_path = project_dir / "screenshot.png"
            if ss_path.is_file():
                ss_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Screenshot of the 3D preview from the {arguments.get('view', 'current')} view.",
                        },
                        as_openrouter_image(ss_path),
                    ],
                }
                messages.append(ss_msg)
                AgentRunner._append_api_message(project_dir, ss_msg)
        # The unified build already produced metrics and a render. Append its
        # visual verification context only after every parallel tool result.
        return (
            tool_succeeded
            and AgentRunner._is_cad_build(name, arguments)
            and not waiting
            and not cad_fix_required
        )

    def deliver_screenshot(
        self,
        project: str,
        request_id: str,
        image_base64: str = "",
        error: str = "",
    ) -> bool:
        with self._lock:
            if self._active_project != project or not self._active_tools:
                return False
            try:
                return self._active_tools.cad.receive_screenshot(
                    request_id, image_base64, error
                )
            except (TypeError, ValueError):
                return False

    @staticmethod
    def _user_error_message(detail: str, err_type: str) -> str:
        """Map technical error messages to user-friendly messages."""
        lower = detail.lower()

        if "401" in detail or "unauthorized" in lower or "invalid api key" in lower:
            return "Invalid OpenRouter API key. Check your key at https://openrouter.ai/keys."
        if "429" in detail or "rate limit" in lower:
            return "OpenRouter rate limit reached. Wait a moment and try again."
        if "model" in lower and (
            "not found" in lower or "invalid" in lower or "not available" in lower
        ):
            return f"The configured model is not available: {detail}"
        if (
            "bubblewrap" in lower
            or "bwrap" in lower
            or "sandbox" in lower
            or "seccomp" in lower
        ):
            return "CAD sandbox failed. Ensure bubblewrap and libseccomp2 are installed (sudo apt install bubblewrap libseccomp2)."
        if "openrouter" in lower and (
            "timeout" in lower or "timed out" in lower or "connection" in lower
        ):
            return "Connection to OpenRouter timed out. Check your internet connection."
        if "timeout" in lower or "timed out" in lower:
            return "CAD code execution timed out. Try simplifying the design or increasing the timeout."
        if "export" in lower and ("step" in lower or "stl" in lower):
            return f"Export failed: {detail}"
        if "permission" in lower or "access denied" in lower or "not writable" in lower:
            return f"Workspace permission error: {detail}"
        if "cancelled" in lower or "stop" in lower:
            return "Task was cancelled."

        # Fallback: return the original detail (it will be logged fully server-side).
        return detail

    def _register_preview(self, project: str, project_dir: Path) -> str:
        preview_path = project_dir / "preview.stl"
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError("CAD execution did not save a usable preview.")
        preview_id = uuid.uuid4().hex
        digest = hashlib.sha256(preview_path.read_bytes()).hexdigest()
        with self._lock:
            self._preview_attempts[(project, preview_id)] = {
                "digest": digest,
                "model_digest": self._model_digest(project_dir) or "",
                "status": "loading",
            }
        return preview_id

    def _preview_matches(self, project: str, attempt: dict[str, str]) -> bool:
        preview_path = self.settings.workspace_root / project / "preview.stl"
        project_dir = preview_path.parent
        try:
            return (
                preview_path.is_file()
                and preview_path.stat().st_size > 0
                and hashlib.sha256(preview_path.read_bytes()).hexdigest()
                == attempt["digest"]
                and (self._model_digest(project_dir) or "") == attempt["model_digest"]
            )
        except (KeyError, OSError):
            return False

    @staticmethod
    def _model_digest(project_dir: Path) -> str | None:
        model_path = project_dir / "model.py"
        try:
            return (
                hashlib.sha256(model_path.read_bytes()).hexdigest()
                if model_path.is_file()
                else None
            )
        except OSError:
            return None

    @classmethod
    def _model_is_built(cls, project_dir: Path) -> bool:
        model_digest = cls._model_digest(project_dir)
        if model_digest is None:
            return True
        try:
            cached = json.loads(
                (project_dir / ".cad_metrics.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return isinstance(cached, dict) and cached.get("model_sha256") == model_digest

    def _await_preview(self, project: str, preview_id: str, message: str) -> None:
        completion = None
        error_message = None
        with self._lock:
            attempt = self._preview_attempts.get((project, preview_id))
            if not attempt or not self._preview_matches(project, attempt):
                error_message = "Drawing was not saved correctly; the preview file is missing or changed."
            elif attempt["status"] == "displayed":
                completion = message
            elif attempt["status"] == "error":
                error_message = attempt["error"]
            else:
                self._pending_completions[project] = {
                    "preview_id": preview_id,
                    "message": message,
                }
        if completion:
            self._complete(project, completion)
        elif error_message:
            self.publish("agent_error", {"project": project, "message": error_message})
        else:
            self.publish(
                "agent_status",
                {
                    "project": project,
                    "status": "rendering",
                    "message": "Waiting for the drawing to become visible...",
                },
            )

    def _complete(self, project: str, message: str) -> None:
        project_dir = self.settings.workspace_root / project
        self._log(project_dir, "assistant", message)
        self.publish("agent_message", {"project": project, "message": message})

    @staticmethod
    def _validate_answer(question: dict[str, object], answer: str) -> bool:
        return QuestionValidator.validate(question, answer)

    @staticmethod
    def _format_answer(question: dict[str, object], answer: str) -> str:
        """Format the user's answer as readable context for the LLM."""
        questions = question.get("questions")
        if isinstance(questions, list) and questions:
            try:
                answers = json.loads(answer)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(answers, dict):
                    lines = ["User answers:"]
                    for q in questions:
                        qid = q.get("id", "")
                        qtext = q.get("question", qid)
                        value = answers.get(qid, "")
                        if value:
                            lines.append(f"- {qtext}: {value}")
                    return "\n".join(lines)
        return answer

    @staticmethod
    def _log(project_dir: Path, role: str, content: object) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "content": content,
        }
        with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _verification_context(self, project: str, project_dir: Path) -> dict:
        self.publish(
            "agent_status",
            {
                "project": project,
                "status": "reviewing",
                "message": "Reviewing CAD preview...",
            },
        )
        render_path = project_dir / "render.png"
        text = (
            "Review the image and verified geometry from cad_build_and_verify against "
            "the user's request. Fix model.py if anything is missing, misaligned, or implausible."
        )
        if render_path.is_file():
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    as_openrouter_image(render_path),
                ],
            }
        return {"role": "user", "content": text}
