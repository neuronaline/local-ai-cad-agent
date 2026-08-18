"""Background tool-calling loop for one local CAD task at a time.

The agent runs one user request at a time per workspace.  A worker thread
streams the model's chat-completions response, dispatches tool calls, and
re-prompts the model with the tool results until it returns a final message
or the configured tool-call budget is exhausted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agent.images import as_chat_image
from agent.llm_base import (
    RequestCancelled,
    create_llm_client,
    provider_label,
    sanitize_assistant_message,
)
from agent.prompt import get_system_prompt
from agent.revisions import RevisionStore
from agent.settings import Settings
from agent.tool_results import failure as tool_failure
from agent.tool_results import success as tool_success
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_review_tool import CadReviewTool
from agent.tools.cad_screenshot_tool import CadScreenshotTool
from agent.tools.cad_tool import CadTool
from agent.tools.experience_tool import ExperienceTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import QuestionTool, normalize_questions
from agent.tools.question_validator import QuestionValidator
from agent.tools.terminal_tool import TerminalTool

_LOG = logging.getLogger(__name__)


class ProjectTools:
    """Per-request bundle of tool instances for one project."""

    def __init__(
        self,
        project_dir: Path,
        publish: Callable[[str, dict], None],
        settings: Settings | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.revisions = RevisionStore(
            project_dir,
            retention_count=settings.revision_retention_count if settings else 0,
        )
        # Reconcile on project load: import existing model or recover from crash.
        self.revisions.reconcile()
        self.file = FileTool(project_dir, self.revisions)
        self.terminal = TerminalTool(project_dir)
        self.cad = CadTool(
            project_dir,
            publish,
            self.revisions,
            review_render_workers=(settings.review_render_workers if settings else 4),
            review_required_views=(settings.review_required_views if settings else 8),
            review_enabled=(settings.review_enabled if settings else True),
        )
        # Screenshot/review orchestrators share the build-time review cache.
        # The agent calls them explicitly; ``cad_build_and_verify`` no longer
        # auto-triggers review.
        self.screenshot = CadScreenshotTool(project_dir, publish=publish)
        self.review = CadReviewTool(
            project_dir,
            publish=publish,
            settings=settings,
            stop_event=stop_event,
        )
        self.question = QuestionTool(publish)
        # Experience tool lives at workspace scope, so we need the workspace root.
        # project_dir is <workspace>/<project>; parent yields the workspace.
        self.experience = ExperienceTool(project_dir.parent, project_dir.name)

    def stop(self) -> None:
        self.terminal.stop()
        self.cad.stop()
        # Best-effort: screenshot/review may have never been used in this
        # project, so guard against missing attributes on cold start.
        for tool in (getattr(self, "screenshot", None), getattr(self, "review", None)):
            if tool is None:
                continue
            try:
                tool.stop()
            except Exception:  # noqa: BLE001 - stop is best-effort.
                _LOG.debug("Tool stop failed", exc_info=True)


# Lock shared with the SSE EventBus so conversation.jsonl writes from the agent
# serialise with the events the bus emits.
_history_lock_slot: list[threading.Lock] = [threading.Lock()]


def _shared_history_lock() -> threading.Lock:
    """Return the active history lock (configurable by the Flask app)."""
    return _history_lock_slot[0]


class AgentRunner:
    """One-thread-per-task chat-completions driver with tool dispatch."""

    # Per-project cache of the truncated conversation history. Bounded so
    # project create/delete churn does not accumulate unbounded memory; entries
    # for directories that no longer exist are evicted on access.
    _HISTORY_CACHE_MAX = 32
    _history_cache: dict[str, list[dict]] = {}

    def __init__(
        self,
        settings: Settings,
        publish: Callable[..., None],
        *,
        history_lock: threading.Lock | None = None,
    ) -> None:
        self.settings = settings
        self.publish = publish
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_tools: ProjectTools | None = None
        self._active_project: str | None = None
        self._lock = threading.Lock()
        self._history_lock: threading.Lock = history_lock or threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}
        self._preview_attempts: dict[tuple[str, str], dict[str, str]] = {}
        self._pending_completions: dict[str, dict[str, str]] = {}
        # Set by the active _run() inside its finally clause so callers can
        # observe completion reliably without polling thread.is_alive().
        self._run_complete: threading.Event = threading.Event()
        self._run_complete.set()

    # ------------------------------------------------------------------ state

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            complete = self._run_complete.is_set()
        # A run is considered active if either we know the thread is alive or
        # the run-completion flag has not yet been raised by its finally. The
        # completion flag is the authoritative signal; we still fall back to
        # is_alive() for tests/synthetic runners that inject a thread object.
        if thread is not None and thread.is_alive():
            return True
        return not complete

    def has_active_state_for(self, project: str) -> bool:
        """Return whether the project has an agent run that must not be deleted."""
        if self.is_running() and self.active_project() == project:
            return True
        if self.is_awaiting_preview(project):
            return True
        # A persisted waiting question is recoverable UI state, not an active
        # worker.  It must not prevent deleting the project.
        return False

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

    # ------------------------------------------------------------------ lifecycle

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
        if (
            (self._thread is not None and not self._run_complete.is_set())
            or self._pending_completions
        ):
            return False
        self._stop_event.clear()
        self._run_complete.clear()
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
        # Wait for the previous run to fully finish (including its finally
        # block) so we never observe a half-completed run while still accepting
        # a follow-up message. _run_complete is set inside _run's finally and
        # replaces a fragile is_alive() / fixed-timeout join. A defensive
        # timeout guards against a runaway run that never reaches finally.
        with self._lock:
            previous_event = self._run_complete
        previous_event.wait(timeout=10.0)
        project_dir = self.settings.workspace_root / project
        formatted = self._format_answer(question, answer)
        with self._lock:
            if (
                (self._thread is not None and not self._run_complete.is_set())
                or self._pending_completions
            ):
                return False
            # Persist the user answer BEFORE starting the new thread so the
            # thread's _context() reads it from the canonical log instead of
            # racing against this append and re-writing a duplicate entry.
            self._append_message(project_dir, {"role": "user", "content": formatted})
            if not self._start_locked(project, formatted, []):
                return False
            self._waiting_questions.pop(project, None)
            (project_dir / ".agent_state.json").unlink(missing_ok=True)
            return True
    def stop(self, project: str | None = None) -> list[str]:
        """Stop agent work and clear pending state for one or all projects."""
        affected: list[str] = []
        with self._lock:
            target_project = project or self._active_project
            stop_active_task = (
                target_project is None or target_project == self._active_project
            )
            if project is None:
                affected = list(
                    dict.fromkeys(
                        list(self._waiting_questions)
                        + list(self._pending_completions)
                        + [key[0] for key in self._preview_attempts]
                    )
                )
                self._waiting_questions.clear()
                self._pending_completions.clear()
                self._preview_attempts.clear()
            else:
                self._waiting_questions.pop(project, None)
                self._pending_completions.pop(project, None)
                for key in [
                    key for key in self._preview_attempts if key[0] == project
                ]:
                    self._preview_attempts.pop(key, None)
                affected = [project]
            if stop_active_task:
                self._stop_event.set()
            if stop_active_task and self._active_tools:
                self._active_tools.stop()
        for cleared in affected:
            (
                self.settings.workspace_root / cleared / ".agent_state.json"
            ).unlink(missing_ok=True)
        return affected

    # ------------------------------------------------------------------ main loop

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
        tools = ProjectTools(
            project_dir, self.publish, self.settings, self._stop_event
        )
        with self._lock:
            self._active_tools = tools
        try:
            messages = self._context(project_dir, message, image_paths or [])
            preview_id: str | None = None
            cad_error: str | None = None
            cad_fix_required = not self._model_is_built(project_dir)
            any_tool_used = False
            nudged_cad = False
            client = create_llm_client(self.settings)
            client.stop_event = self._stop_event
            client.session_id = f"{self.settings.llm_provider}:{project}"
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
                # Persist every assistant turn (intermediate tool-call turns and
                # final user-facing turns) as the single canonical history entry.
                self._append_message(project_dir, assistant_message)
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
                            self._publish_terminal_failure(project)
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
                                self._append_message(project_dir, reminder)
                                continue
                            self.publish(
                                "agent_error",
                                {
                                    "project": project,
                                    "message": "Drawing was not created: the task did not produce a new CAD preview.",
                                },
                            )
                            self._publish_terminal_failure(project)
                        else:
                            # No tools were used at all — just a conversation; complete silently.
                            self._complete(project, content)
                        return
                    self._await_preview(project, preview_id, content)
                    return
                any_tool_used = True
                processed_call_ids: set[str] = set()
                waiting = False
                for call in tool_calls:
                    call_id = call.get("id", "")
                    if self._stop_event.is_set():
                        self._cancel_remaining_tool_calls(
                            project_dir, tool_calls, processed_call_ids, messages
                        )
                        return
                    preview_id, cad_error, cad_fix_required, waiting = self._process_tool_call(
                        tools,
                        project,
                        project_dir,
                        call,
                        cad_fix_required,
                        preview_id,
                        cad_error,
                        messages,
                    )
                    processed_call_ids.add(call_id)
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
                        self._cancel_remaining_tool_calls(
                            project_dir, tool_calls, processed_call_ids, messages
                        )
                        return
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
            self._publish_terminal_failure(project)
        except RequestCancelled:
            # Stopping is an expected control-flow path, not an OpenRouter error.
            self.publish(
                "agent_status",
                {
                    "project": project,
                    "status": "stopped",
                    "message": "Task stopped.",
                },
            )
        except Exception as error:  # noqa: BLE001 - Surface all agent failures to the local UI.
            import traceback

            detail = str(error)
            err_type = type(error).__name__
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
                        "message": self._user_error_message(detail, err_type, provider=self.settings.llm_provider),
                    },
                )
                self._publish_terminal_failure(project)
        finally:
            with self._lock:
                self._active_tools = None
                if self._active_project == project:
                    self._active_project = None
                # Mark this run finished and drop the thread reference so a
                # subsequent start/answer() never observes a stale _thread.
                self._thread = None
                self._run_complete.set()

    # ------------------------------------------------------------------ context

    def _context(
        self, project_dir: Path, message: str, image_paths: list[Path]
    ) -> list[dict]:
        history = self._load_history(project_dir)
        user_message: dict[str, object] = {"role": "user", "content": message}
        if image_paths:
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": message}]
                + [as_chat_image(path) for path in image_paths],
            }
        if not history or history[-1] != user_message:
            history.append(user_message)
            self._append_message(project_dir, user_message)
        issue_index = ExperienceTool(
            project_dir.parent, project_dir.name
        ).context_index()
        return [
            {"role": "system", "content": get_system_prompt()},
            self._past_issues_message(issue_index),
            *history,
        ]

    @staticmethod
    def _past_issues_message(issue_index: dict[str, object]) -> dict[str, str]:
        """Format one run-scoped, compact past-issues snapshot."""
        if not issue_index.get("available"):
            content = "<past_issues>\n(unavailable)\n</past_issues>"
        else:
            issues = issue_index.get("issues")
            lines = ["<past_issues>"]
            if isinstance(issues, list) and issues:
                for issue in issues:
                    if not isinstance(issue, dict):
                        continue
                    title = issue.get("title")
                    record_id = issue.get("id")
                    if isinstance(title, str) and isinstance(record_id, str):
                        lines.append(f"- {title} — id: {record_id}")
            if len(lines) == 1:
                lines.append("(empty)")
            lines.append("</past_issues>")
            content = "\n".join(lines)
        return {"role": "user", "content": content}

    @staticmethod
    def _strip_image_parts(item: dict) -> dict:
        """Replace inline image parts with placeholders when loading history."""
        content = item.get("content")
        if item.get("role") != "user" or not isinstance(content, list):
            return item
        parts: list[object] = []
        replaced = False
        image_index = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                replaced = True
                image_index += 1
                name = part.get("filename") or f"reference-{image_index}"
                parts.append({"type": "text", "text": f"[image: {name}]"})
            else:
                parts.append(part)
        if not replaced:
            return item
        cleaned = dict(item)
        cleaned["content"] = parts
        return cleaned

    @classmethod
    def _truncate_history(cls, history: list[dict]) -> list[dict]:
        if len(history) <= cls.MAX_HISTORY:
            return history
        return history[-cls.MAX_HISTORY:]

    @classmethod
    def _load_history(cls, project_dir: Path) -> list[dict]:
        """Load the conversation history from a single canonical jsonl file."""
        cache_key = str(project_dir)
        cached = cls._history_cache.get(cache_key)
        if cached is not None:
            # Evict stale entries whose project directory has been removed.
            if not project_dir.exists():
                cls._history_cache.pop(cache_key, None)
            else:
                return list(cached)
        history: list[dict] = []
        log_path = project_dir / "conversation.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
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
        history = [cls._strip_image_parts(item) for item in history]
        history = cls._truncate_history(history)
        # Bound the cache size with FIFO eviction to avoid unbounded growth
        # from long-lived processes with many project create/delete cycles.
        while len(cls._history_cache) >= cls._HISTORY_CACHE_MAX:
            oldest = next(iter(cls._history_cache))
            if oldest == cache_key:
                break
            cls._history_cache.pop(oldest, None)
        cls._history_cache[cache_key] = list(history)
        return history

    @classmethod
    def _append_message(cls, project_dir: Path, message: dict) -> None:
        """Append a message to the conversation log (the single canonical sink)."""
        cls._history_cache.pop(str(project_dir), None)
        with _shared_history_lock():
            with (project_dir / "conversation.jsonl").open("a", encoding="utf-8") as log:
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

    # ------------------------------------------------------------------ dispatch

    MAX_HISTORY = 100

    def _execute(
        self,
        tools: ProjectTools,
        project: str,
        name: str,
        args: dict,
        call_id: str = "",
    ) -> tuple[object, bool]:
        if name == "cad_build_and_verify":
            # ``render`` defaults to True to preserve legacy behavior; the
            # schema documents the iteration vs. final-verification split.
            return tools.cad.with_call_id(call_id).build_and_verify(
                args.get("render", True)
            ), False
        if name == "cad_screenshot":
            tool = tools.screenshot.with_call_id(call_id) if call_id else tools.screenshot
            return tool.execute(args), False
        if name == "cad_review":
            tool = tools.review.with_call_id(call_id) if call_id else tools.review
            return tool.execute(args), False
        if name.startswith("file_"):
            tool = tools.file.with_call_id(call_id) if call_id else tools.file
            operation = name.removeprefix("file_")
            if operation == "read":
                return tool.read(
                    args["filename"],
                    args.get("offset", 1),
                    args.get("limit"),
                    args.get("known_sha256"),
                ), False
            if operation == "write":
                return tool.write(
                    args["filename"],
                    args["content"],
                    args.get("expected_sha256"),
                ), False
            if operation == "replace":
                return tool.replace(
                    args["filename"],
                    args["old"],
                    args["new"],
                    args.get("expected_sha256"),
                    args.get("expected_matches"),
                ), False
            return (
                tool.regex_replace(
                    args["filename"],
                    args["pattern"],
                    args["replacement"],
                    args.get("count", 1),
                    args.get("expected_sha256"),
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
        if name == "terminal_bash":
            return tools.terminal.bash(
                args["command"], args.get("timeout_seconds", 15)
            ), False
        if name == "experience_search":
            return tools.experience.search(args["query"]), False
        if name == "experience_get":
            return tools.experience.get(args["id"]), False
        if name == "experience_add":
            return tools.experience.add(
                args["title"],
                args["problem"], args["solution"], args.get("tags")
            ), False
        if name == "experience_update":
            return tools.experience.update(
                args["id"],
                args.get("title"),
                args.get("problem"),
                args.get("solution"),
                args.get("tags"),
            ), False
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
        return (
            name in {"file_write", "file_replace", "file_regex_replace"}
            and arguments.get("filename") == "model.py"
        )

    @staticmethod
    def _is_cad_build(name: str, arguments: dict) -> bool:
        return name == "cad_build_and_verify"

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
    ) -> tuple[str | None, str | None, bool, bool]:
        """Execute a single tool call, update state, return new (preview, error, fix_required, waiting)."""
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
            self.publish("tool_status", {**tool_event, "status": "running"})
            if self._is_cad_build(name, arguments):
                preview_id = None
            raw_result, waiting = self._execute(
                tools, project, name, arguments, call_id
            )
            result = tool_success(name, raw_result)
            if self._is_model_mutation(name, arguments):
                preview_id = None
                cad_error = None
                cad_fix_required = True
                self.publish("revision_updated", {"project": project})
            if self._is_cad_build(name, arguments):
                preview_id = self._register_preview(project, project_dir)
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
            self._debug_tool_error(project_dir, call_id, name, error, result)
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
        self._append_message(project_dir, messages[-1])
        return preview_id, cad_error, cad_fix_required, waiting

    @classmethod
    def _cancel_remaining_tool_calls(
        cls,
        project_dir: Path,
        tool_calls: list[dict],
        processed_call_ids: set[str],
        messages: list[dict],
    ) -> None:
        """Write cancelled tool-result entries for unprocessed calls."""
        cancelled = json.dumps({"error": "Tool call cancelled (question or stop)."})
        for tc in tool_calls:
            cid = tc.get("id", "")
            if cid and cid not in processed_call_ids:
                entry = {"role": "tool", "tool_call_id": cid, "content": cancelled}
                messages.append(entry)
                cls._append_message(project_dir, entry)

    @staticmethod
    def _user_error_message(detail: str, err_type: str, *, provider: str = "openrouter") -> str:
        """Map technical error messages to user-friendly messages."""
        lower = detail.lower()
        provider_name = provider_label(provider)
        key_url = (
            "https://platform.openai.com/api-keys"
            if provider == "openai"
            else "https://openrouter.ai/keys"
        )

        if "401" in detail or "unauthorized" in lower or "invalid api key" in lower:
            return f"Invalid {provider_name} API key. Check your key at {key_url}."
        if "429" in detail or "rate limit" in lower:
            return f"{provider_name} rate limit reached. Wait a moment and try again."
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
        if ("openrouter" in lower or "openai" in lower) and (
            "timeout" in lower or "timed out" in lower or "connection" in lower
        ):
            return f"Connection to {provider_name} timed out. Check your internet connection."
        if "timeout" in lower or "timed out" in lower:
            return "CAD code execution timed out. Try simplifying the design or increasing the timeout."
        if "export" in lower and ("step" in lower or "stl" in lower):
            return f"Export failed: {detail}"
        if "permission" in lower or "access denied" in lower or "not writable" in lower:
            return f"Workspace permission error: {detail}"
        if "cancelled" in lower or "stop" in lower:
            return "Task was cancelled."

        return detail

    # ------------------------------------------------------------------ preview tracking

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
            self._publish_terminal_failure(project)
        else:
            self.publish(
                "agent_status",
                {
                    "project": project,
                    "status": "rendering",
                    "message": "Waiting for the drawing to become visible...",
                },
            )

    def _publish_terminal_failure(self, project: str) -> None:
        # Mirror _complete's success-side agent_status so the UI clears the
        # thinking indicator on every error path. agent_status:stopped is
        # already published for user-initiated stops. Marked transient so
        # the conversation log stays clean (the agent_error event that just
        # preceded this is the canonical terminal record).
        self.publish(
            "agent_status",
            {"project": project, "status": "failed", "message": "Task failed."},
            transient=True,
        )

    def _complete(self, project: str, message: str) -> None:
        """Persist the final assistant turn and publish it to subscribers.

        The agent loop persists the assistant turn via ``_append_message``;
        this method ensures the final user-facing response is recorded
        even when the loop reaches ``_complete`` without persisting the
        assistant message (e.g. the ``_await_preview`` path). The agent
        loop and ``_complete`` together guarantee a single canonical entry.
        """
        project_dir = self.settings.workspace_root / project
        history = self._load_history(project_dir)
        if not history or history[-1] != {"role": "assistant", "content": message}:
            self._append_message(
                project_dir, {"role": "assistant", "content": message}
            )
        self.publish("agent_message", {"project": project, "message": message})
        # Terminal status so the UI clears the thinking indicator on success.
        # The agent loop publishes agent_error on failure paths and
        # agent_status:stopped on user-initiated stops, so this complements
        # both without overriding them. Marked transient so the agent_message
        # above remains the canonical terminal entry in the conversation log.
        self.publish(
            "agent_status",
            {
                "project": project,
                "status": "completed",
                "message": "Task completed.",
            },
            transient=True,
        )

    def _debug_tool_error(
        self,
        project_dir: Path,
        call_id: str,
        tool: str,
        error: Exception,
        result: str,
    ) -> None:
        """Append recoverable tool failures to a project-local debug log."""
        if not self.settings.agent_debug_log_tool_errors:
            return
        try:
            payload = json.loads(result)
            error_detail = payload.get("error", {}) if isinstance(payload, dict) else {}
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "call_id": call_id,
                "tool": tool,
                "error_type": type(error).__name__,
                "message": str(error),
                "classification": error_detail,
            }
            with _shared_history_lock():
                with (project_dir / "debug-errors.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            # Debug logging must not affect the agent's recovery path.
            return

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
