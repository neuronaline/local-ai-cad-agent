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
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from agent.activity_log import ActivityLogger, get_logger
from agent.activity_log import is_enabled as activity_logging_enabled
from agent.conversation import (
    ConversationStore,
    shared_history_lock,
)
from agent.conversation import (
    _history_lock_slot as _conversation_history_lock_slot,
)
from agent.dispatcher import (
    cancel_remaining_tool_calls,
    dispatch,
    normalize_tool_calls,
    process_tool_call,
)
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
from agent.tool_schemas import TOOL_SCHEMAS
from agent.tools.cad_review_tool import CadReviewTool
from agent.tools.cad_screenshot_tool import CadScreenshotTool
from agent.tools.cad_tool import CadTool
from agent.tools.file_tool import FileTool
from agent.tools.question_tool import QuestionTool
from agent.tools.question_validator import QuestionValidator

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

    def stop(self) -> None:
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


# Re-export the conversation store's lock slot under the legacy local name.
# Flask replaces this slot with EventBus._history_lock at startup; keeping the
# *same list* ensures ConversationStore appends and EventBus status writes
# serialize against one another.
_history_lock_slot = _conversation_history_lock_slot


def _shared_history_lock() -> threading.Lock:
    """Return the active history lock (configurable by the Flask app)."""
    return shared_history_lock()


class AgentRunner:
    """One-thread-per-task chat-completions driver with tool dispatch."""

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
        # Per-run activity-log handles. ``_run()`` assigns these when the
        # operator enabled ``agent.log_tool_activity`` and clears them in
        # its ``finally`` block; initialising to ``None`` keeps ``getattr``
        # out of the hot path for tests and disabled runs.
        self._active_activity_logger: ActivityLogger | None = None
        self._active_run_id: str | None = None
        self._lock = threading.Lock()
        self._history_lock: threading.Lock = history_lock or threading.Lock()
        self._waiting_questions: dict[str, dict[str, object]] = {}
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
        # A persisted waiting question is recoverable UI state, not an active
        # worker.  It must not prevent deleting the project.
        return self.is_running() and self.active_project() == project

    def active_project(self) -> str | None:
        with self._lock:
            return self._active_project

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
        if self._thread is not None and not self._run_complete.is_set():
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
            if self._thread is not None and not self._run_complete.is_set():
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
                affected = list(dict.fromkeys(list(self._waiting_questions)))
                self._waiting_questions.clear()
            else:
                self._waiting_questions.pop(project, None)
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
            nudged_final_verification = False
            build_failure_count = 0
            build_failure_signatures: dict[str, int] = {}
            client = create_llm_client(self.settings)
            client.stop_event = self._stop_event
            session_prefix = (
                self.settings.openrouter_session_prefix
                if self.settings.llm_provider == "openrouter"
                else self.settings.llm_provider
            )
            client.session_id = f"{session_prefix}:{project}"
            # Wire the activity logger when the operator enabled
            # ``agent.log_tool_activity``. The logger is opt-in so default
            # runs keep the same on-disk footprint; tests and reviewers
            # leave it ``None`` and the wire path becomes a no-op.
            activity_logger: ActivityLogger | None = None
            run_id = uuid.uuid4().hex
            if activity_logging_enabled(self.settings):
                activity_logger = get_logger(project_dir)
                client._activity_logger = activity_logger
                client._run_id = run_id
                # Stash on the runner so the per-call dispatcher wrapper
                # can read the logger without re-deriving it. Both
                # attributes are reset in ``finally`` to keep a stale
                # logger from leaking across runs.
                self._active_activity_logger = activity_logger
                self._active_run_id = run_id
                activity_logger.log(
                    "run_start",
                    {
                        "project": project,
                        "model": self.settings.llm_model,
                        "provider": self.settings.llm_provider,
                    },
                    run_id=run_id,
                )
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

                def publish_stream(
                    event: dict,
                    current_message_id: str = message_id,
                ) -> None:
                    event_type = event.pop("type")
                    self.publish(
                        f"agent_{event_type}_delta",
                        {"project": project, "message_id": current_message_id, **event},
                    )
                    if activity_logger is not None:
                        # Record the raw delta so the activity log mirrors
                        # the SSE stream the UI consumes. Payloads are tiny
                        # text fragments; the redact step is a no-op here
                        # but kept consistent with the rest of the logger.
                        activity_logger.log(
                            f"agent_{event_type}_delta",
                            {"message_id": current_message_id, **event},
                            run_id=run_id,
                        )

                client.stream_callback = publish_stream
                awaiting_tool_render = self._last_message_has_tool_image(messages)
                response = client.chat(messages, TOOL_SCHEMAS)
                if awaiting_tool_render and getattr(
                    client, "last_image_fallback_used", False
                ):
                    # The provider rejected the trailing inline render. The
                    # agent did nothing wrong, but visual verification is no
                    # longer possible from this turn. Take ownership of
                    # ``cad_error`` for the rest of this iteration: the build
                    # did not produce verifiable evidence, so the
                    # final-verification gate must treat the model as still
                    # unverified. The same ``cad_error`` string also feeds
                    # the build-failure tracker in the tool-dispatch loop,
                    # so keep the message stable across iterations.
                    cad_fix_required = True
                    preview_id = None
                    cad_error = (
                        "The model provider rejected the required final render; "
                        "visual verification could not be completed."
                    )
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
                invalid_final = (
                    not tool_calls
                    and any_tool_used
                    and (not preview_id or cad_fix_required)
                )
                self.publish(
                    "agent_stream_end",
                    {
                        "project": project,
                        "message_id": message_id,
                        "message": ""
                        if invalid_final
                        else assistant_message.get("content") or "",
                    },
                )
                if not invalid_final:
                    messages.append(assistant_message)
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
                    if cad_fix_required:
                        if not nudged_final_verification:
                            nudged_final_verification = True
                            reminder = {
                                "role": "user",
                                "content": (
                                    "The current model revision has not passed final visual "
                                    "verification. Call cad_build_and_verify with render=true "
                                    "and parameter_checks for explicit dimensions before finishing."
                                ),
                            }
                            messages.append(reminder)
                            self._append_message(project_dir, reminder)
                            continue
                        self.publish(
                            "agent_error",
                            {
                                "project": project,
                                "message": "Task stopped: final visual verification is still missing.",
                            },
                        )
                        self._publish_terminal_failure(project)
                        return
                    # CAD succeeded and passed the final-verification gate.
                    self._complete(project, content)
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
                    if call.get("function", {}).get("name") == "cad_build_and_verify":
                        if cad_error:
                            build_failure_count += 1
                            signature = self._failure_signature(cad_error)
                            build_failure_signatures[signature] = (
                                build_failure_signatures.get(signature, 0) + 1
                            )
                            if (
                                build_failure_count >= 6
                                or build_failure_signatures[signature] >= 2
                            ):
                                self.publish(
                                    "agent_error",
                                    {
                                        "project": project,
                                        "message": (
                                            "Task stopped after repeated CAD build failures. "
                                            "Review the latest error or continue with a narrower repair."
                                        ),
                                    },
                                )
                                self._publish_terminal_failure(project)
                                return
                        else:
                            build_failure_count = 0
                            build_failure_signatures.clear()
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
            if "message_id" in locals():
                self.publish(
                    "agent_stream_end",
                    {"project": project, "message_id": message_id, "message": ""},
                )
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
            self._active_activity_logger = None
            self._active_run_id = None
            if activity_logger is not None:
                activity_logger.log(
                    "run_end",
                    {"project": project, "cancelled": self._stop_event.is_set()},
                    run_id=run_id,
                )

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
        return [
            {"role": "system", "content": get_system_prompt()},
            self._project_state_message(project_dir),
            *history,
        ]

    @staticmethod
    def _project_state_message(project_dir: Path) -> dict[str, str]:
        """Provide dynamic workspace state after the cacheable system prefix."""
        if (project_dir / "model.py").is_file():
            content = (
                "<project_state>\n"
                "model.py exists. Read it before making a targeted edit.\n"
                "</project_state>"
            )
        else:
            content = (
                "<project_state>\n"
                "model.py does not exist. Create it directly with write_file; do not "
                "call read_file, edit_file, cad_build_and_verify, "
                "cad_screenshot, or cad_review first.\n"
                "</project_state>"
            )
        # Gemini normalizes all system messages into an immutable instruction.
        # Keeping mutable workspace state in a later user message lets its
        # explicit system-message cache breakpoint remain reusable.
        return {"role": "user", "content": content}

    @staticmethod
    def _last_message_has_tool_image(messages: list[dict]) -> bool:
        if not messages or messages[-1].get("role") != "tool":
            return False
        content = messages[-1].get("content")
        return isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        )

    @staticmethod
    def _failure_signature(message: str) -> str:
        """Collapse volatile line numbers so equivalent failures match."""
        normalized = re.sub(r"\bline \d+\b", "line #", message.lower())
        normalized = re.sub(r"\b0x[0-9a-f]+\b", "0x#", normalized)
        return normalized[-1000:]

    @classmethod
    def _load_history(cls, project_dir: Path) -> list[dict]:
        """Load the canonical conversation.jsonl for ``project_dir``."""
        return ConversationStore.load(project_dir)

    @classmethod
    def _append_message(cls, project_dir: Path, message: dict) -> None:
        """Append a message to the canonical conversation log."""
        ConversationStore.append(project_dir, message)

    @classmethod
    def clear_history(cls, project_dir: Path) -> bool:
        """Reset the agent's memory for ``project_dir``.

        Truncates the canonical ``conversation.jsonl`` log and evicts the
        in-memory history cache entry. Returns ``True`` when a log file was
        removed, ``False`` when the project had no recorded conversation.
        The model, preview, renders, and revision blobs are left untouched.
        """
        return ConversationStore.clear(project_dir)

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

    def _execute(
        self,
        tools: ProjectTools,
        project: str,
        name: str,
        args: dict,
        call_id: str = "",
    ) -> tuple[object, bool]:
        """Backward-compatible thin wrapper around :func:`dispatcher.dispatch`."""
        return dispatch(tools, project, name, args, call_id)

    @staticmethod
    def _normalize_tool_calls(raw_calls: object) -> list[dict]:
        """Backwards-compatible alias for :func:`dispatcher.normalize_tool_calls`."""
        return normalize_tool_calls(raw_calls)

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
        """Backwards-compatible thin wrapper around :func:`dispatcher.process_tool_call`."""
        return process_tool_call(
            tools,
            project,
            project_dir,
            call,
            cad_fix_required,
            prev_preview_id,
            cad_error,
            messages,
            publish=self.publish,
            register_preview=self._register_preview,
            append_message=self._append_message,
            debug_log=self._debug_tool_error,
            activity_logger=self._active_activity_logger,
            run_id=self._active_run_id,
        )

    @classmethod
    def _cancel_remaining_tool_calls(
        cls,
        project_dir: Path,
        tool_calls: list[dict],
        processed_call_ids: set[str],
        messages: list[dict],
    ) -> None:
        """Backwards-compatible thin wrapper around the dispatcher's equivalent."""
        cancel_remaining_tool_calls(
            project_dir,
            tool_calls,
            processed_call_ids,
            messages,
            append_message=cls._append_message,
        )

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
        """Return a fresh ``preview_id`` correlation token.

        The token is published alongside ``preview_updated`` so the UI can
        decide whether to fetch the new STL. The agent turn no longer parks
        its completion on a browser ACK; the host's render-failure path
        surfaces any consumer error through ``agent_error`` independently.
        """
        preview_path = project_dir / "preview.stl"
        if not preview_path.is_file() or preview_path.stat().st_size == 0:
            raise RuntimeError("CAD execution did not save a usable preview.")
        return uuid.uuid4().hex

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
        assistant message (the legacy preview-Await path used to do this).
        The agent loop and ``_complete`` together guarantee a single
        canonical entry.
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
            with (
                _shared_history_lock(),
                (project_dir / "debug-errors.jsonl").open(
                    "a", encoding="utf-8"
                ) as log,
            ):
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
        # Route through ``ConversationStore.append`` so the canonical write
        # path holds ``shared_history_lock`` and invalidates the cache. The
        # previous direct ``open(..., "a")`` shortcut interleaved bytes with
        # concurrent canonical writes and left the in-memory cache stale,
        # which then dropped the question-log line on the next history load.
        ConversationStore.append(project_dir, item)
